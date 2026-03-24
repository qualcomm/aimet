# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

# [model-setup]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache

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
from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_precision,
)
from GenAITests.shared.helpers.precision_config import WeightPrecision
from aimet_onnx.common.defs import int8

assembled_dummy_inputs = Generator.prepare_inputs(
    model=traceable_model,
    input_ids=torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int),
    attention_mask=torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int),
    past_key_values=[],
    context_length=CONTEXT_LENGTH,
    sequence_length=SEQUENCE_LENGTH,
)

# Export to ONNX using LLM.get_backbone_input_names to produce the input naming
# convention required by AdaScale (input_ids, attention_mask, position_ids,
# past_key_0_in, past_value_0_in, ...)
with tempfile.TemporaryDirectory() as tmpdir:
    torch.onnx.export(
        traceable_model,
        assembled_dummy_inputs,
        os.path.join(tmpdir, "model.onnx"),
        input_names=LLM.get_backbone_input_names(hf_model.config.num_hidden_layers),
        output_names=LLM.get_backbone_output_names(hf_model.config.num_hidden_layers),
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(os.path.join(tmpdir, "model.onnx"))

quantsim = QuantizationSimModel(
    model=onnx_model,
    quant_scheme="min_max",
    default_activation_bw=16,
    default_param_bw=4,
    config_file="htp_v73",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
# Setting kv_cache and some other layers to 8-bit
_set_tensors_to_output_n_bit_symmmetric(quantsim, kv_bits=8)
_set_lm_head_precision(quantsim, WeightPrecision(qtype=int8, granularity="PCQ"))
_tie_quantizers_for_kv_cache(quantsim)

quantsim_with_torch_interface = TorchONNXInterface(quantsim, hf_model.config)
generator = Generator(quantsim_with_torch_interface, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)
# End of [create-sim]

# [adascale-apply]
from aimet_onnx.experimental.adascale.adascale_optimizer import (
    AdaScale,
    adascale_model_config_dict,
)
from GenAITests.shared.helpers.datasets import Wikitext
from GenAITests.onnx.helpers.quant_recipes import _prefill_inputs

ADASCALE_NUM_BATCHES = 128   # reduce for larger models to control runtime
ADASCALE_NUM_ITERATIONS = 2048  # reduce for larger models; see quantization recipes

train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")
prefilled_inputs = _prefill_inputs(
    quantsim, generator, train_dataset, num_batches=ADASCALE_NUM_BATCHES
)

AdaScale.apply_adascale(
    quantsim,
    prefilled_inputs,
    adascale_model_config=adascale_model_config_dict[generator.config.model_type],
    num_iterations=ADASCALE_NUM_ITERATIONS,
)
# End of [adascale-apply]

# [compute-encodings]
from tqdm import tqdm

calib_inputs = _prefill_inputs(quantsim, generator, train_dataset, num_batches=20)


def _forward(session, _):
    for batch in tqdm(calib_inputs, total=len(calib_inputs), desc="Calibrating"):
        session.run(None, batch)


quantsim.compute_encodings(_forward, tuple())
# End of [compute-encodings]
