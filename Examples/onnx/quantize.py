# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Example flattened script showcasing quantization recipes for LLMs"""

import os
import json
import onnx
import argparse
import torch
from tqdm import tqdm
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM

from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.onnx.models.utils.torch_onnx_interface import (
    TorchONNXInterface,
)
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_precision,
)
from GenAITests.shared.helpers.precision_config import WeightPrecision
from aimet_onnx.common.defs import int8
from GenAITests.onnx.helpers.quant_recipes import (
    _prefill_inputs,
)
from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.helpers.datasets import Wikitext

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096
CALIB_NUM_BATCHES = 20
SEQMSE_NUM_BATCHES = 20
ADASCALE_NUM_BATCHES = 128
ADASCALE_NUM_ITERATIONS = 2048
KV_BITS = 8


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
        "--kv-bits",
        type=int,
        default=KV_BITS,
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
    from aimet_onnx.quantsim import set_grouped_blockwise_quantization_for_weights
    from aimet_onnx import apply_seq_mse

    # Set weight parameter quantizers of modules to LPBQ.
    set_grouped_blockwise_quantization_for_weights(
        sim=quantsim,
        op_types=("Gemm", "MatMul", "Conv"),
        bitwidth=4,
        decompressed_bw=8,
        block_size=64,
        nodes_to_exclude=["/model/lm_head/MatMul"],
    )
    apply_seq_mse(
        quantsim,
        prefilled_inputs,
        num_candidates=20,
        nodes_to_exclude=["/model/lm_head/MatMul"],
    )
    print(f"SequentialMSE applied successfully.")


def apply_recipe_pcq_spinquant_adascale(
    quantsim, prefilled_inputs, adascale_num_iterations: int
):
    from aimet_onnx.experimental.adascale.adascale_optimizer import (
        AdaScale,
        adascale_model_config_dict,
    )

    AdaScale.apply_adascale(
        quantsim,
        prefilled_inputs,
        adascale_model_config=adascale_model_config_dict[generator.config.model_type],
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
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
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

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.onnx.export(
            traceable_model,
            assembled_dummy_inputs,
            os.path.join(tmpdir, "model.onnx"),
            input_names=LLM.get_backbone_input_names(hf_model.config.num_hidden_layers),
            output_names=LLM.get_backbone_output_names(
                hf_model.config.num_hidden_layers
            ),
            opset_version=17,
            dynamo=False,
        )
        onnx_model = onnx.load(os.path.join(tmpdir, "model.onnx"))

    quantsim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme="min_max",
        default_activation_bw=16,
        default_param_bw=4,
        config_file=LLM.get_quantsim_config(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # Setting kv_cache and some other layers to 8-bit
    _set_tensors_to_output_n_bit_symmmetric(quantsim, args.kv_bits)
    # Setting the LM head weights to 8-bit.
    _set_lm_head_precision(quantsim, WeightPrecision(qtype=int8, granularity="PCQ"))
    # Tie kv_cache
    _tie_quantizers_for_kv_cache(quantsim)

    # Create a generator object to accurately simulate inference with static graph constraints while maintaining the
    # same interface. Use the generator object to do all forward passes through the model, including calibration, eval
    quantsim_with_torch_interface = TorchONNXInterface(quantsim, hf_model.config)
    generator = Generator(
        quantsim_with_torch_interface, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH
    )

    # Load WikiText dataset from Huggingface
    train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")
    num_batches = (
        args.adascale_num_batches
        if args.recipe == "pcq_spinquant_adascale"
        else args.seqmse_num_batches
    )
    prefilled_inputs = _prefill_inputs(quantsim, generator, train_dataset, num_batches)

    if args.recipe == "lpbq_seqmse":
        apply_recipe_lpbq_seqmse(quantsim, prefilled_inputs)
    elif args.recipe == "pcq_spinquant_adascale":
        apply_recipe_pcq_spinquant_adascale(
            quantsim, prefilled_inputs, args.adascale_num_iterations
        )
    else:
        raise NotImplementedError(f"Unknown recipe: {args.recipe}")

    calib_inputs = _prefill_inputs(
        quantsim, generator, train_dataset, args.calib_num_batches
    )

    def _forward(session, _):
        for batch in tqdm(calib_inputs, total=len(calib_inputs), desc="Calibrating"):
            session.run(None, batch)

    quantsim.compute_encodings(_forward, tuple())

    os.makedirs(args.export_path, exist_ok=True)
    hf_model.config.save_pretrained(args.export_path)
    tokenizer.save_pretrained(args.export_path)
    quantsim.export(
        path=args.export_path,
        filename_prefix=f"model_cl{CONTEXT_LENGTH}",
        export_model=True,
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
