# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=missing-docstring

# [setup]
from functools import partial
from itertools import chain

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

# Load model and LoRa adapter from Huggingface
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)

# This ensures that use_cache and return_dict are always set to false
# These settings are selected so that the model is JIT-traceable
model.forward = partial(model.forward, use_cache=False, return_dict=False)

# Load train and test splits of dataset
def tokenize(examples):
    seq_length = 512
    examples = tokenizer(examples["text"])
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length
    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='train').map(tokenize, batched=True, remove_columns=['text'])
test_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='test').map(tokenize, batched=True, remove_columns=['text'])

train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=default_data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=default_data_collator)

# [create_quantsim]
import torch
from transformers.models import opt

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.peft import replace_lora_layers_with_quantizable_layers

# Generate dummy data used to instantiate QuantizationSimModel
tokenized_dummy_text = tokenizer("Here is some sample text used to create dummy input ids")
dummy_input_ids = torch.Tensor(tokenized_dummy_text['input_ids']).to(dtype=torch.int32).unsqueeze(0)
dummy_attention_mask = torch.Tensor(tokenized_dummy_text['attention_mask']).to(dtype=torch.int32).unsqueeze(0)

# Modify LoRa layers so they are quantizable
replace_lora_layers_with_quantizable_layers(model)

# Register quantized version of OPTLearnedPositionalEmbedding
@QuantizationMixin.implements(opt.modeling_opt.OPTLearnedPositionalEmbedding)
class QuantizedOPTLearnedPositionalEmbedding(QuantizationMixin, opt.modeling_opt.OPTLearnedPositionalEmbedding):
    """ Dummy placeholder - we don't want to quantize OPTLearnedPositionalEmbedding """
    forward = opt.modeling_opt.OPTLearnedPositionalEmbedding.forward

# Create QuantizationSimModel
quantsim = QuantizationSimModel(model=model,
                                dummy_input=(dummy_input_ids, dummy_attention_mask),
                                default_output_bw=16,
                                default_param_bw=4,
                                in_place=True)

# [calibration_callback]
from tqdm import tqdm

# Callback function to pass calibration data through the model
def generate_calibration_callback(dataloader, max_iterations: int, device: torch.device):
    def forward_pass(model: torch.nn.Module):
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(dataloader, total=max_iterations)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                if batch_id >= max_iterations:
                    break

    return forward_pass

# [lora_training_callback]
# Function to perform one epoch of training
def train_one_epoch(model, dataloader, device=torch.device("cuda")):
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_id, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute the loss and its gradients
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss = loss_fn(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
        loss.backward()

        # # Adjust learning weights
        optimizer.step()

# [qwa_lora]
from aimet_torch.utils import place_model
from aimet_torch.peft import LoraLayer
import aimet_torch.v2.quantization as Q

from aimet_torch.v2.utils import remove_all_quantizers

lora_layers = [module for module in quantsim.model.modules() if isinstance(module, LoraLayer)]

with place_model(model, torch.device("cuda")):
    # Temporarily remove all LoRa layer quantizers, leaving only base model quantizers
    with remove_all_quantizers(lora_layers):
        # Only compute encodings for base model weights, activations
        calibration_callback = generate_calibration_callback(dataloader=train_dataloader, max_iterations=20, device=torch.device("cuda"))
        quantsim.compute_encodings(calibration_callback)

        # prevent quantization encoding getting overwritten by sim.compute_encodings()
        for module_name, module in model.named_modules():
            if isinstance(module, Q.base.QuantizerBase):
                module.allow_overwrite(False)

        # Configure model so that only LoRa layers are trainable
        model.requires_grad_(False)
        for module_name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)

        # Perform LoRa QAT with base model weight, activation encodings frozen
        train_one_epoch(quantsim.model, train_dataloader, torch.device("cuda"))

    # Compute all other encodings
    calibration_callback = generate_calibration_callback(dataloader=train_dataloader, max_iterations=20, device=torch.device("cuda"))
    quantsim.compute_encodings(calibration_callback)
