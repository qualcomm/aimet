# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
import torch
from torch.utils.data import Dataset, DataLoader
from aimet_torch.experimental.omniquant import apply_omniquant
from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, default_data_collator
from transformers.models.llama import modeling_llama
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from itertools import chain


# [setup]
# Load the model
# General setup that can be changed as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_config = AutoConfig.from_pretrained(model_id)
model_config.return_dict=False
model_config.use_cache = False

model = modeling_llama.LlamaForCausalLM.from_pretrained(model_id, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

# End of [setup]

# [prepare-dataloader]
def tokenize(examples):
    seq_length = 2048
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
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)

# Custom class to use limited samples from dataloader
dataloader_wrapper_len = 40
class LimitedBatchDataLoader(DataLoader):
    def __init__(self, data_loader):
        self.data_loader = data_loader
 
    def __len__(self):
        return dataloader_wrapper_len
 
    def __iter__(self):
        return iter(self.data_loader)

# End of [prepare-dataloader]

# [create-sim]
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

seq_length = 2048
input_ids = torch.randint(0, model_config.vocab_size, (1, seq_length), device=device)
attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)
dummy_input = (input_ids, attention_mask)
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           default_param_bw=4,
                           default_output_bw=16,
                           in_place=True)
# End of [create-sim]

# [apply-omniquant]
# Find and freeze optimal encodings candidate for weight parameters of supported layers

apply_omniquant(quant_sim=sim,
               dataloader=train_dataloader,
               forward_fn=lambda model, input: model.forward(**input),
               num_iterations=800)

# End of [apply-omniquant]

# [compute_encodings]
def calibration_wrapper(model, dataloader, max_iterations: int):
    for batch_id, batch in enumerate(dataloader):
        if batch_id < max_iterations:
            batch = tuple((d.to(device) for d in batch.values()))
            model.to(device)(*batch)
        else:
            break

# Compute the Quantization Encodings
# compute encodings for all activations and parameters of uninitialized layer(s)/operations(s)
sim.compute_encodings(calibration_wrapper, dataloader = train_dataloader, max_iterations=40)
# End of [compute_encodings]

# [evaluation]
# Determine simulated quantized accuracy
...
# End of [evaluation]

# [export]
# Export the model for on-target inference
path = './'
filename = 'dummy_model'
sim.export(path=path, filename_prefix="quantized_" + filename, dummy_input=dummy_input.cpu())
# End of [export]