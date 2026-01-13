# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Example flattened script showcasing quantization recipes for LLMs"""

import os
import json
import argparse
import itertools
import torch
from tqdm import tqdm
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
from aimet_torch.utils import place_model

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.helpers.datasets import Wikitext
from GenAITests.torch.helpers.quant_recipes import _prefill_inputs

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096
CALIB_NUM_BATCHES = 20
SEQMSE_NUM_BATCHES = 20
ADASCALE_NUM_BATCHES = 128
ADASCALE_NUM_ITERATIONS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        help="Huggingface model id of desired variant.",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--recipe",
        help="Quantization recipe",
        choices=["pcq_spinquant_adascale", "lpbq_seqmse"],
        required=True,
    )
    parser.add_argument(
        "--export-path",
        help="Path to export quantized model.",
        required=True,
    )
    parser.add_argument(
        "--calib-num-batches",
        type=int,
        default=CALIB_NUM_BATCHES,
    )
    parser.add_argument(
        "--seqmse-num-batches",
        type=int,
        default=SEQMSE_NUM_BATCHES,
    )
    parser.add_argument(
        "--adascale-num-batches",
        type=int,
        default=ADASCALE_NUM_BATCHES,
    )
    parser.add_argument(
        "--adascale-num-iterations",
        type=int,
        default=ADASCALE_NUM_ITERATIONS,
    )

    return parser.parse_args()


def apply_spinquant_if_needed(hf_model: torch.nn.Module, recipe: str):
    """Apply SpinQuant when needed"""
    if recipe != "pcq_spinquant_adascale":
        return

    from aimet_torch.experimental.spinquant import apply_spinquant

    # Embedding layer and lm_head need to be separated for SpinQuant to be applied
    old_weight = hf_model.lm_head.weight
    new_weight = torch.nn.Parameter(
        old_weight.data.clone().detach().to(old_weight.device),
        requires_grad=True,
    )
    hf_model.lm_head.weight = new_weight

    # Apply SpinQuant to make model activations easier to quantize
    apply_spinquant(hf_model)
    print(f"SpinQuant applied successfully.")


def apply_recipe_lpbq_seqmse(quantsim, prefilled_inputs):
    """Apply quantization recipe: LPBQ + SequentialMSE"""
    from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
    from aimet_torch.v2.quantsim.config_utils import (
        set_grouped_blockwise_quantization_for_weights,
    )
    from aimet_torch.seq_mse import apply_seq_mse

    # Set weight parameter quantizers of modules to LPBQ.
    arg = lambda module: (
        isinstance(module, (QuantizedConv2d, QuantizedLinear))
        and module.param_quantizers["weight"]
        and module.param_quantizers["weight"].bitwidth == 4
    )
    set_grouped_blockwise_quantization_for_weights(
        sim=quantsim,
        arg=arg,
        bitwidth=4,
        symmetric=True,
        decompressed_bw=8,
        block_size=64,
        block_grouping=-1,
    )

    # Apply SequentialMSE.
    apply_seq_mse(
        sim=quantsim,
        data_loader=prefilled_inputs,
        num_candidates=20,
    )
    print(f"SequentialMSE applied successfully.")


def apply_recipe_pcq_spinquant_adascale(
    quantsim, prefilled_inputs, adascale_num_iterations: int
):
    """Apply quantization recipe: PCQ + SpinQuant + AdaScale"""
    from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale

    apply_adascale(
        qsim=quantsim,
        data_loader=prefilled_inputs,
        num_iterations=adascale_num_iterations,
    )
    print(f"AdaScale applied successfully.")


def write_manifest(path: str, meta: dict):
    manifest_path = os.path.join(path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=4)
    return manifest_path


if __name__ == "__main__":
    args = parse_args()

    # Fetch specified model and tokenizer from huggingface
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=True
    )

    # Apply SpinQuant to model to change weights in-place
    apply_spinquant_if_needed(hf_model, args.recipe)

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

    # Create QuantizationSimModel with 4-bit integer weights and 16-bit integer activations
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

    # Create a generator object to accurately simulate inference with static graph constraints while maintaining the
    # same interface. Use the generator object to do all forward passes through the model, including calibration, eval
    generator = Generator(quantsim.model, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)

    # Load WikiText dataset from Huggingface
    train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")

    # Use CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        num_batches = (
            args.adascale_num_batches
            if args.recipe == "pcq_spinquant_adascale"
            else args.seqmse_num_batches
        )
        prefilled_inputs = _prefill_inputs(
            generator, train_dataset, num_batches, torch.device("cpu")
        )

        if args.recipe == "lpbq_seqmse":
            apply_recipe_lpbq_seqmse(quantsim, prefilled_inputs)
        elif args.recipe == "pcq_spinquant_adascale":
            apply_recipe_pcq_spinquant_adascale(
                quantsim, prefilled_inputs, args.adascale_num_iterations
            )
        else:
            raise NotImplementedError(f"Unknown recipe: {args.recipe}")

        def calibration_callback(model: torch.nn.Module):
            sliced_dataloader = itertools.islice(train_dataset, args.calib_num_batches)
            for batch in tqdm(
                sliced_dataloader, total=args.calib_num_batches, desc="Calibrating"
            ):
                # Use generator for forward passes
                generator(input_ids=batch["input_ids"].to(device=model.device))

        # Compute activation encodings
        quantsim.compute_encodings(calibration_callback)

    os.makedirs(args.export_path, exist_ok=True)
    # Save model weights/buffers and quantizer states
    torch.save(
        quantsim.model.model.state_dict(),
        os.path.join(args.export_path, f"model_cls{CONTEXT_LENGTH}.pt"),
    )
    # Save only quantizer states
    torch.save(
        quantsim.quantizer_state_dict(),
        os.path.join(args.export_path, f"model_cls{CONTEXT_LENGTH}_quantizer.pt"),
    )
    print(
        f"PyTorch model weights and quantizer states are saved in {args.export_path} successfully."
    )
    quantsim.model.config.save_pretrained(args.export_path)
    tokenizer.save_pretrained(args.export_path)

    quantsim.onnx.export(
        f=os.path.join(args.export_path, f"model_cl{CONTEXT_LENGTH}.onnx"),
        args=assembled_dummy_inputs,
        input_names=Generator.get_input_names(
            quantsim.model.model.config.num_hidden_layers
        ),
        output_names=Generator.get_output_names(
            quantsim.model.model.config.num_hidden_layers
        ),
        dynamo=False,
        opset_version=17,
        export_int32_bias=False,
    )
    print(
        f"ONNX model and JSON encodings are saved in {args.export_path} successfully."
    )

    manifest = {
        "model_id": args.model_id,
        "sequence_length": SEQUENCE_LENGTH,
        "context_length": CONTEXT_LENGTH,
        "recipe": args.recipe,
        "calib_num_batches": args.calib_num_batches,
        "seqmse_num_batches": args.seqmse_num_batches
        if args.recipe == "lpbq_seqmse"
        else None,
        "adascale_num_iterations": args.adascale_num_iterations
        if args.recipe == "pcq_spinquant_adascale"
        else None,
        "adascale_num_batches": args.adascale_num_batches
        if args.recipe == "pcq_spinquant_adascale"
        else None,
    }
    write_manifest(args.export_path, manifest)
