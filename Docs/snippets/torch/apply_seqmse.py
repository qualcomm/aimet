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
import torch
import torchvision
from torchvision import transforms

# Load the model
# General setup that can be changed as needed
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval().to(device)
# End of load the model

# Prepare the dataloader
DATASET_ROOT = ... # Set your path to imagenet dataset root directory
BATCH_SIZE = 32
NUM_CALIBRATION_SAMPLES = 128

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_data = torchvision.datasets.ImageNet(
    DATASET_ROOT,
    split="val",
    transform=preprocess    
)

dataloader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
# End of dataloader

# Create Quantization Simulation Model
from aimet_torch.quantsim import QuantizationSimModel

dummy_input = torch.randn(1, 3, 224, 224).to(device)
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           default_param_bw=4,
                           default_output_bw=8)
# End of QuantizationSimModel

# Apply Seq MSE
import itertools
from aimet_torch.seq_mse import  apply_seq_mse, SeqMseParams

# Get unlabeled data
num_batches = NUM_CALIBRATION_SAMPLES // BATCH_SIZE
unlabeled_data = [data[0] for data in itertools.islice(dataloader, num_batches)]

# Configure SeqMSE parameters
params = SeqMseParams(num_batches=num_batches,
                      num_candidates=20)

# Find and freeze optimal encodings candidate for parameters of supported layer(s)/operations(s).
apply_seq_mse(sim=sim, data_loader=unlabeled_data, num_candidates=20)
# End of Seq MSE

# Calibration callback
@torch.no_grad()
def forward_pass(model: torch.nn.Module):
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        model(images.to(device))

# compute encodings for all activations and parameters of uninitialized layers.
sim.compute_encodings(forward_pass)
# End of compute_encodings

# Evaluation
# Determine simulated quantized accuracy
from tqdm import tqdm

correct_predictions = 0
total_samples = 0
for inputs, labels in tqdm(dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = sim.model(inputs)
    _, pred_labels = torch.max(outputs, 1)
    correct_predictions += torch.sum(pred_labels == labels)
    total_samples += labels.shape[0]

accuracy = correct_predictions / total_samples
print(f"Quantized accuracy: {accuracy}")
# End of evaluation

# Export
# Export the model for on-target inference.
sim.export(path=".", filename_prefix="quantized_mobilenet_v2", dummy_input=dummy_input.cpu())
# End of export
