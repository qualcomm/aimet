# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import sys
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.helpers.metrics import PPL, MMLU, Interactive

import onnx
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    load_encodings_to_sim,
    set_grouped_blockwise_quantization_for_weights,
)
from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_to_8b,
)

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096
KV_BITS = 8


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

    # Collect .onnx file
    matches = [
        f for f in checkpoint_path.iterdir() if f.is_file() and f.suffix == ".onnx"
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No ONNX model found in {checkpoint_path}.")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple ONNX models found in {checkpoint_path}.")
    onnx_path = matches[0]

    # Collect JSON .encodings file
    matches = [
        f for f in checkpoint_path.iterdir() if f.is_file() and f.suffix == ".encodings"
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No JSON Encodings file found in {checkpoint_path}.")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple JSON Encodings files found in {checkpoint_path}."
        )
    encodings_path = matches[0]

    # Load the manifest file
    m = load_manifest(checkpoint_path)
    if m["model_id"] != args.model_id:
        raise ValueError(
            f"Mismatch between manifest model-id ({m['model_id']}) and provided model-id ({args.model_id})."
        )

    # Fetch the tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(args.model_id)

    onnx_model = onnx.load(onnx_path)
    print(f"ONNX model loaded successfully from {onnx_path}.")

    quantsim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme="min_max",
        default_activation_bw=16,
        default_param_bw=4,
        config_file=LLM.get_quantsim_config(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Setting kv_cache and some other layers to 8-bit
    _set_tensors_to_output_n_bit_symmmetric(quantsim, KV_BITS)
    # Setting the LM head weights to 8-bit.
    _set_lm_head_to_8b(quantsim)
    # Tie kv_cache
    _tie_quantizers_for_kv_cache(quantsim)

    if "lpbq" in str(m["recipe"]).lower():
        set_grouped_blockwise_quantization_for_weights(
            quantsim,
            op_types=("MatMul", "Conv", "Gemm"),
            bitwidth=4,
            decompressed_bw=8,
            block_size=64,
            nodes_to_exclude=["/model/lm_head/MatMul"],
        )
        print(f"Setting LPBQ quantizers.")

    load_encodings_to_sim(
        quantsim,
        encodings_path,
        strict=False,  # Quantizer settings will be updated to align with the encodings to load.
        allow_overwrite=False,  # Loaded encodings will be frozen.
        disable_missing_quantizers=False,  # Quantizers which do not have encodings will not be disabled.
    )
    print(f"Encodings are loaded successfully from {encodings_path}.")

    quantsim_with_torch_interface = TorchONNXInterface(quantsim, config)
    generator = Generator(
        quantsim_with_torch_interface,
        tokenizer,
        SEQUENCE_LENGTH,
        CONTEXT_LENGTH,
    )

    if args.eval_ppl:
        ppl_score = PPL.evaluate(generator, tokenizer, CONTEXT_LENGTH)
        print(f"PPL: {ppl_score}")

    if args.eval_mmlu:
        mmlu_score = MMLU.evaluate(generator, tokenizer, CONTEXT_LENGTH)
        print(f"MMLU: {mmlu_score}")

    if args.eval_interactive:
        Interactive.evaluate(generator, tokenizer, CONTEXT_LENGTH)
