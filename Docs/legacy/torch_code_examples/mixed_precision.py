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
""" Code example for mixed precision """

# Step 0. Import statements
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantizationDataType, CallbackFunc
from aimet_torch.v1.mixed_precision import choose_mixed_precision
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
# End step 0

# Step 1
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
# End step 1

# Step 2
default_bitwidth = 16

# ((activation bitwidth, activation data type), (param bitwidth, param data type))
candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
             ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
             ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]
# Allowed accuracy drop in absolute value
allowed_accuracy_drop = 0.5 # Implies 50% drop

eval_callback_for_phase_1 = CallbackFunc(evaluate, func_callback_args=data_loader)
eval_callback_for_phase_2 = CallbackFunc(evaluate, func_callback_args=data_loader)

calibration_batches = 10
forward_pass_call_back = CallbackFunc(forward_pass, func_callback_args=calibration_batches)

# Create quant sim
sim = QuantizationSimModel(model,
                           default_param_bw=default_bitwidth,
                           default_output_bw=default_bitwidth,
                           dummy_input=dummy_input)
sim.compute_encodings(forward_pass, forward_pass_callback_args=calibration_batches)

# Enable phase-3 (optional)
GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

# Call the mixed precision algo with clean start = True i.e. new accuracy list and pareto list will be generated
# If set to False then pareto front list and accuracy list will be loaded from the provided directory path
# A allowed_accuracy_drop can be specified to export the final model with reference to the pareto list
pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback_for_phase_1,
                                           eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                           clean_start=True, forward_pass_callback=forward_pass_call_back)
print(pareto_front_list)

# Set clean_start to False to start from an existing cache
# Set allowed_accuracy_drop to 0.9 to export the 90% drop point in pareto list
allowed_accuracy_drop = 0.9
pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback_for_phase_1,
                                           eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                           clean_start=False, forward_pass_callback=forward_pass_call_back)
print(pareto_front_list)
sim.export("./data", str(allowed_accuracy_drop), dummy_input)
# End step 2
