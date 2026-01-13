# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)
from aimet_torch.v2.nn.transformers.models.qwen2.modeling_qwen2 import (
    QuantizedQwen2RMSNorm,
)
from aimet_torch.v2.nn.transformers.models.qwen3.modeling_qwen3 import (
    QuantizedQwen3RMSNorm,
)
from aimet_torch.v2.nn.transformers.models.phi3.modeling_phi3 import (
    QuantizedPhi3RMSNorm,
)
from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch.utils import place_model

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.helpers.metrics import PPL, MMLU, Interactive

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096


def load_manifest(path: Path):
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Missing manifest {manifest_path}")
        sys.exit(2)
    with open(manifest_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        help="Huggingface model id of desired LLM variant",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to the exported quantized model path.",
        required=True,
    )
    parser.add_argument(
        "--eval-ppl",
        help="Evaluate perplexity of the quantized model on WikiText",
        action="store_true",
    )
    parser.add_argument(
        "--eval-mmlu",
        help="Evaluate MMLU of the quantized model (warning: can be slow!)",
        action="store_true",
    )
    parser.add_argument(
        "--eval-interactive",
        help="Evaluate qualitatively on prompts",
        action="store_true",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        raise ValueError(f"--checkpoint must be a directory, got {checkpoint_path}.")

    # Collect weights .pt file
    matches = [
        f
        for f in checkpoint_path.iterdir()
        if f.is_file() and f.suffix == ".pt" and not f.name.endswith("_quantizer.pt")
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No weights .pt file found in {checkpoint_path}.")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple weights .pt files found in {checkpoint_path}.")
    weights_ckpt_path = matches[0]

    # Collect quantizer .pt file
    matches = [
        f
        for f in checkpoint_path.iterdir()
        if f.is_file() and f.name.endswith("_quantizer.pt")
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No quantizer .pt file found in {checkpoint_path}.")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple quantizer .pt files found in {checkpoint_path}."
        )
    quantizers_ckpt_path = matches[0]

    # Load the manifest file
    m = load_manifest(checkpoint_path)
    if m["model_id"] != args.model_id:
        raise ValueError(
            f"Mismatch between manifest model-id ({m['model_id']}) and provided model-id ({args.model_id})."
        )

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=True
    )

    # Load model weights/buffers into original HF model (quantizers states will be ignored.)
    state_dict = torch.load(
        weights_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    hf_model.load_state_dict(state_dict, strict=False)
    print(f"Model weights are loaded successfully from {weights_ckpt_path}.")

    # Need to wrap model in this in order to enable JIT trace
    traceable_model = ONNXExportableModuleWithCache(hf_model)

    # Create dummy inputs used to initialize QuantizationSimModel
    dummy_input_ids = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int)
    dummy_attention_mask = torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int)
    assembled_dummy_inputs = Generator.prepare_inputs(
        model=traceable_model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        past_key_values=[],
        context_length=CONTEXT_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
    )

    quantsim = QuantizationSimModel(
        model=traceable_model,
        quant_scheme=QuantScheme.post_training_tf,
        dummy_input=assembled_dummy_inputs,
        default_output_bw=16,
        default_param_bw=4,
        in_place=True,
        config_file=LLM.get_quantsim_config(),
    )

    # Apply mixed precision to model
    quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
    for _, module in quantsim.model.named_modules():
        if isinstance(
            module,
            (
                QuantizedLlamaRMSNorm,
                QuantizedQwen2RMSNorm,
                QuantizedQwen3RMSNorm,
                QuantizedPhi3RMSNorm,
            ),
        ):
            module.param_quantizers["weight"].bitwidth = 16

    # Load only quantizer states into quantsim.
    state_dict = torch.load(
        quantizers_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "lpbq" in str(m["recipe"]).lower():
        arg = lambda module: (
            isinstance(module, (QuantizedConv2d, QuantizedLinear))
            and module.param_quantizers["weight"]
            and module.param_quantizers["weight"].bitwidth == 4
        )
        set_grouped_blockwise_quantization_for_weights(
            quantsim,
            arg,
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=64,
            block_grouping=-1,
        )
        print(f"Setting LPBQ quantizers.")

    quantsim.model.load_state_dict(state_dict, strict=False)
    print(f"Quantizers states are loaded successfully from {quantizers_ckpt_path}.")

    # Create a generator object to accurately simulate inference with static graph constraints while maintaining the
    # same interface. Use the generator object to do all forward passes through the model, including calibration, eval
    generator = Generator(quantsim.model, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)

    # Use CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        if args.eval_ppl:
            ppl_score = PPL.evaluate(generator, tokenizer, CONTEXT_LENGTH)
            print(f"PPL: {ppl_score}")

        if args.eval_mmlu:
            mmlu_score = MMLU.evaluate(generator, tokenizer, CONTEXT_LENGTH)
            print(f"MMLU: {mmlu_score}")

        if args.eval_interactive:
            Interactive.evaluate(generator, tokenizer, CONTEXT_LENGTH)
