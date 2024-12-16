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
[setup]
import torch
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import evaluator

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
num_batches = 32
data = load_dataset('imagenet-1k', streaming=True, split="train")
data_loader = DataLoader(data, batch_size=num_batches, num_workers = 4)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

def forward_pass(model: torch.nn.Module):
    with torch.no_grad():
        for images, _ in data_loader:
            model(images)

path = './'
filename = 'mobilenet'

[step_1]
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.adaround.adaround_weight import Adaround, AdaroundParameters

params = AdaroundParameters(data_loader=data_loader, num_batches=num_batches)

# Returns model with AdaRound-ed weights and their corresponding encodings
adarounded_model = Adaround.apply_adaround(model, dummy_input, params, path=path, filename_prefix=filename)
[step_2]
sim = QuantizationSimModel(adarounded_model, dummy_input)

# AdaRound optimizes the rounding of weight quantizers only. These values are preserved through load_encodings()
sim.load_encodings(encodings=path + filename, allow_overwrite=False)

# The activation quantizers remain uninitialized and derived through compute_encodings()
sim.compute_encodings(forward_pass)
[step_3]
evaluator = evaluator("image-classification")
accuracy = evaluator.compute(model_or_pipeline=model, data=data, metric="accuracy")
[step_4]
sim.export(path=path, filename_prefix="quantized_" + filename, dummy_input=dummy_input.cpu())