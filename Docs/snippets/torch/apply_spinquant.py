# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.utils.data import DataLoader
from aimet_torch.experimental.spinquant import spinquant_optimizer, apply_spinquant
from transformers import AutoTokenizer, AutoConfig
from transformers import default_data_collator
from transformers.models.llama import modeling_llama
from torch.utils.data import DataLoader
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
model_config.tie_word_embeddings = False

model = modeling_llama.LlamaForCausalLM.from_pretrained(model_id, config=model_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

# End of [setup]

# [register-rmsnorm-r1-points]

# Example model type to register
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed_tokens = ...
        self.rmsnorm = ...
        self.q_proj = ...
        self.k_proj = ...
        self.v_proj = ...
        self.o_proj = ...
        self.gate_proj = ...
        self.up_proj = ...
        self.down_proj = ...
        self.lm_head = ...
        ...
    
    def forward(self, input):
        ...

from aimet_torch.experimental.spinquant import spinquant_optimizer

# Define a function to identify rmsnorm fusion pairs
def rmsnorm_fusion_pairs(model):
    return [(model.rmsnorm, [model.q_proj, model.k_proj, model.v_proj])]

# Define a function to identify R1 placement
def r1_placement(model):
    return [(model.embed_tokens, False),
            (model.q_proj, True),
            (model.k_proj, True),
            (model.v_proj, True),
            (model.o_proj, False),
            (model.gate_proj, True),
            (model.up_proj, True),
            (model.down_proj, False),
            (model.lm_head, True)
            ]

# Register MyModel type with rmsnorm fusion and R1 placement functions
spinquant_optimizer.SUPPORTED_MODULE_DICT[MyModel] = {spinquant_optimizer.RMSNORM_LINEAR_PAIRS: rmsnorm_fusion_pairs,
                                                      spinquant_optimizer.R1_FUSION_PAIRS: r1_placement}

# End of [register-rmsnorm-r1-points]

# [apply-spinquant]

# The model is updated in place
apply_spinquant(model=model)

# End of [apply-spinquant]

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
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)

# Custom class to use limited samples from dataloader
class LimitedBatchDataLoader:
    """Internal helper class to reduce number of accessible batches in Dataloader"""

    def __init__(self, dataloader, num_batches):
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.iterator = iter(self.dataloader)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            return next(self.iterator)
        raise StopIteration

    def __len__(self):
        return min(len(self.dataloader), self.num_batches)

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

# [compute_encodings]
def calibration_wrapper(model, dataloader):
    for batch in dataloader:
        batch = tuple((d.to(device) for d in batch.values()))
        model(*batch)

# Compute the Quantization Encodings
# compute encodings for all activations and parameters of uninitialized layer(s)/operations(s)
sim.compute_encodings(calibration_wrapper, LimitedBatchDataLoader(train_dataloader, num_batches=40))

# End of [compute_encodings]

# [evaluation]
# Determine simulated quantized accuracy
...
# End of [evaluation]
