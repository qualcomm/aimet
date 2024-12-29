# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

# pylint: skip-file

""" QuantSim and QAT code example to be used for documentation generation. """

# PyTorch imports
import torch
import torch.cuda
from tqdm import tqdm
# End of PyTorch imports

# Dataloaders
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_calibration_and_eval_data_loaders(path: str):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(path, transform=transform)

    batch_size = 64
    calibration_data_size = batch_size * 16
    eval_data_size = len(dataset) - calibration_data_size

    calibration_dataset, eval_dataset = random_split(
        dataset, [calibration_data_size, eval_data_size]
    )

    calibration_data_loader = DataLoader(calibration_dataset, batch_size=batch_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size)
    return calibration_data_loader, eval_data_loader

PATH_TO_IMAGENET = '<your_imagenet_validation_data_path>'
calibration_data_loader, eval_data_loader = get_calibration_and_eval_data_loaders(PATH_TO_IMAGENET)
# End of dataloaders

# Calibration callback
from typing import Any, Optional

def pass_calibration_data(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    """
    The User of the QuantizationSimModel API is expected to write this callback based on their dataset.
    """
    data_loader = forward_pass_args

    # batch_size (64) * num_batches (16) should be 1024
    num_batches = 16

    model.eval()
    with torch.no_grad():
        for batch, (input_data, _) in enumerate(data_loader):
            inputs_batch = input_data.to("cuda")  # labels are ignored
            model(inputs_batch)
            if batch >= num_batches:
                break
# End of calibration callback

# Load the model
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).cuda()
# End of load the model

# Create Quantization Simulation Model
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape).cuda()
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           default_param_bw=8,
                           default_output_bw=16,
                           config_file=get_path_for_per_channel_config())
# End of QuantizationSimModel

# Compute the Quantization Encodings
sim.compute_encodings(pass_calibration_data, forward_pass_callback_args=calibration_data_loader)
# End of compute_encodings

# Evaluation
# Determine simulated quantized accuracy
sim.model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(eval_data_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = sim.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total:.4f}')
# End of evaluation

# Export
# Export the model for on-target inference.
# Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
# activations and parameters in JSON format at provided path.
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2', dummy_input=dummy_input.cpu())
# End of export

# Finetune the model
# User action required
# The following line of code illustrates that the model is getting fine-tuned.
# Replace the following lines to fit your pipeline.
sim.model.train()
num_epochs = 1
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5)

for _ in range(num_epochs):
    for inputs, labels in tqdm(calibration_data_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        logits = sim.model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
# activations and parameters in JSON format
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2', dummy_input=dummy_input.cpu())
# End of export
# End of example
