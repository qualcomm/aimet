# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

# [model-setup]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096

model_id = "meta-llama/Llama-3.2-1B-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

# Wrap model to satisfy static graph constraints for JIT trace
traceable_model = ONNXExportableModuleWithCache(hf_model)
# End of [model-setup]

# [create-sim]
import os
import tempfile
import onnx
from aimet_onnx.quantsim import QuantizationSimModel
from GenAILab.qai_hub_lm.models.base import LLM
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.generator import Generator
from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import TorchONNXInterface
from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_precision,
)
from GenAILab.bench.precision import WeightPrecision
from aimet_onnx.common.defs import int8

assembled_dummy_inputs = tuple(
    Generator.prepare_inputs(
        model=traceable_model,
        input_ids=torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int),
        attention_mask=torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int),
        past_key_values=[],
        context_length=CONTEXT_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
    ).values()
)

layer_cache_descs = build_layer_cache_descriptors(hf_model.config)
with tempfile.TemporaryDirectory() as tmpdir:
    torch.onnx.export(
        traceable_model,
        assembled_dummy_inputs,
        os.path.join(tmpdir, "model.onnx"),
        input_names=LLM.get_backbone_input_names(layer_cache_descs),
        output_names=LLM.get_backbone_output_names(layer_cache_descs),
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(os.path.join(tmpdir, "model.onnx"))

quantsim = QuantizationSimModel(
    model=onnx_model,
    quant_scheme="min_max",
    default_activation_bw=16,
    default_param_bw=8,
    config_file="htp_v73",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
_set_tensors_to_output_n_bit_symmmetric(quantsim, 8)
_set_lm_head_precision(quantsim, WeightPrecision(qtype=int8, granularity="PCQ"))
_tie_quantizers_for_kv_cache(quantsim)

quantsim_with_torch_interface = TorchONNXInterface(quantsim, hf_model.config)
generator = Generator(quantsim_with_torch_interface, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)
# End of [create-sim]

# [spinquant-apply]
from aimet_onnx.experimental.spinquant import apply_spinquant

# apply_spinquant modifies quantsim in-place. Must be called BEFORE compute_encodings.
apply_spinquant(quantsim)
# End of [spinquant-apply]

# [compute-encodings]
from tqdm import tqdm
from GenAILab.bench.datasets import Wikitext
from GenAILab.bench.onnx.quant_recipes import _prefill_inputs

train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")
calib_inputs = _prefill_inputs(quantsim, generator, train_dataset, 20)


def _forward(session, _):
    for batch in tqdm(calib_inputs, total=len(calib_inputs), desc="Calibrating"):
        session.run(None, batch)


quantsim.compute_encodings(_forward, tuple())
# End of [compute-encodings]
