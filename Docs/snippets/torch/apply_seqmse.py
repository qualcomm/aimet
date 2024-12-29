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
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import evaluator

# Load the model
# General setup that can be changed as needed
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval().to(device)
# End of load the model

# Prepare the dataloader
num_batches = 32
data = load_dataset('imagenet-1k', streaming=True, split="train")
data_loader = DataLoader(data, batch_size=num_batches, num_workers=4)
# End of dataloader

# Create Quantization Simulation Model
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel

dummy_input = torch.randn(1, 3, 224, 224).to(device)
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           default_param_bw=4,
                           default_output_bw=8,
                           config_file=get_path_for_per_channel_config())
# End of QuantizationSimModel

# Apply Seq MSE
# Find and freeze optimal encodings candidate for parameters of supported layer(s)/operations(s).
from aimet_torch.seq_mse import  apply_seq_mse, SeqMseParams
params = SeqMseParams(num_batches=num_batches,
                      num_candidates=20,
                      inp_symmetry='symqt',
                      loss_fn='mse')

apply_seq_mse(model=model, sim=sim, data_loader=data_loader, params=params)
# End of Seq MSE

# Calibration callback
def forward_pass(model: torch.nn.Module):
    with torch.no_grad():
        for images, _ in data_loader:
            model(images)
# End of calibration callback

# Compute the Quantization Encodings
# compute encodings for all activations and parameters of uninitialized layer(s)/operations(s).
sim.compute_encodings(forward_pass)
# End of compute_encodings

# Evaluation
# Determine simulated quantized accuracy
evaluator = evaluator("image-classification")
accuracy = evaluator.compute(model_or_pipeline=model, data=data, metric="accuracy")
# End of evaluation

# Export
# Export the model for on-target inference.
# Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
# activations and parameters in JSON format at provided path.
path = './'
filename = 'mobilenet'
sim.export(path=path, filename_prefix="quantized_" + filename, dummy_input=dummy_input.cpu())
# End of export
