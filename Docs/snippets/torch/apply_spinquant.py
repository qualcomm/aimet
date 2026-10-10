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
from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from GenAILab.qai_hub_lm.models.generator import Generator

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

quantsim = QuantizationSimModel(
    model=traceable_model,
    quant_scheme=QuantScheme.post_training_tf,
    dummy_input=assembled_dummy_inputs,
    default_output_bw=16,
    default_param_bw=8,
    in_place=True,
    config_file="htp_v73",
)

generator = Generator(quantsim.model, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)
# End of [create-sim]

# [spinquant-apply]
from aimet_torch.spinquant import apply_spinquant

# apply_spinquant modifies the model in-place. Must be called BEFORE compute_encodings.
apply_spinquant(model=quantsim.model)
# End of [spinquant-apply]

# [compute-encodings]
import itertools
from tqdm import tqdm
from GenAILab.bench.datasets import Wikitext

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")


def calibration_callback(model):
    for batch in tqdm(
        itertools.islice(train_dataset, 20), total=20, desc="Calibrating"
    ):
        generator(input_ids=batch["input_ids"].to(device=device))


quantsim.compute_encodings(calibration_callback)
# End of [compute-encodings]
