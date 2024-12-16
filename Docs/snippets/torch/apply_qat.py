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
# pylint: disable=all

# setup
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torchvision.models.mobilenet_v2(pretrained=True).eval().to(device)

batch_size = 64
PATH_TO_IMAGENET = ...
data = torchvision.datasets.ImageNet(PATH_TO_IMAGENET, split="train")
data_loader = DataLoader(data, batch_size=batch_size)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
fold_all_batch_norms(model, dummy_input.shape)

# Callback function to pass calibration data through the model
def forward_pass(model: torch.nn.Module, batches):
    with torch.no_grad():
        for batch, (images, _) in enumerate(data_loader):
            images = images.to(device)
            model(images)
            if batch >= batches:
                break

# Basic ImageNet evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            correct += (logits.argmax(1) == labels).type(torch.float).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return accuracy

# step_1
from aimet_torch.v2.quantsim import QuantizationSimModel, QuantScheme
sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.training_range_learning_with_tf_init)

calibration_batches = 10
sim.compute_encodings(forward_pass, calibration_batches)

accuracy = evaluate(sim.model, data_loader)
print(f"PTQ model accuracy: {accuracy}")
# step_2
# Training loop can be replaced with any custom training loop
def train(model, data_loader, optimizer, loss_fn):
    model.train()
    for data, labels in tqdm(data_loader):
        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5)

epochs = 2
for epoch in range(epochs):
    train(sim.model, data_loader, optimizer, loss_fn)
# step_3
accuracy = evaluate(sim.model, data_loader)
print(f"Model accuracy after QAT: {accuracy}")
# step_4
sim.export(path="./", filename_prefix="quantized_mobilenetv2", dummy_input=dummy_input.cpu())