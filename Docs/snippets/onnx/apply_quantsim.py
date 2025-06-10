# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# imports start
import os
import numpy as np
import onnx
import torch
from tqdm import tqdm
# imports end

# Load the model
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path as you wish, we are using temporary directory for now
file_path = os.path.join('/tmp', f'mobilenet_v2.onnx')
torch.onnx.export(pt_model,
                  (dummy_input,),
                  file_path,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'},
                  },
                  )

# Load exported ONNX model
model = onnx.load_model(file_path)
# End of loading the model

# Prepare model with onnx-simplifier
import onnxsim
model, _ = onnxsim.simplify(model)
# End of prepare model

# Set up dataloader
import torchvision
from torchvision import transforms

DATASET_ROOT = ... # Set your path to imagenet dataset root directory
BATCH_SIZE = 32
NUM_CALIBRATION_SAMPLES = 1024
NUM_EVAL_SAMPLES = 50000

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_data = torchvision.datasets.ImageNet(DATASET_ROOT,
                                              split="val",
                                              transform=preprocess)

dataloader = torch.utils.data.DataLoader(imagenet_data,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
# End of setting up dataloader

# Create QuantSim object
from aimet_common.defs import QuantScheme
import aimet_onnx
from aimet_onnx import QuantizationSimModel

# Optionally use ["CUDAExecutionProvider", "CPUExecutionProvider"] for GPU enablement
providers = ["CPUExecutionProvider"]
sim = QuantizationSimModel(model,
                           param_type=aimet_onnx.int8,
                           activation_type=aimet_onnx.int16,
                           quant_scheme=QuantScheme.min_max,
                           config_file="default",
                           providers=providers)
# End of creating QuantSim object

# Calibration callback
input_name = model.graph.input[0].name
def onnx_data_generator(num_batches):
    """
    Example conversion from torch dataloader to onnx model inputs
    """
    for i, (data, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        yield {input_name: data.numpy()}
# End of calibration callback

# Compute quantization encodings
sim.compute_encodings(onnx_data_generator(NUM_CALIBRATION_SAMPLES // BATCH_SIZE))
# End of computing quantization encodings

# Evaluate quantized accuracy
correct_predictions = 0
total_samples = 0
for i, (inputs, labels) in enumerate(tqdm(dataloader)):
    pred_probs, *_ = sim.session.run(None, {input_name: inputs.numpy()})
    pred_labels = np.argmax(pred_probs, axis=1)
    correct_predictions += np.sum(pred_labels == labels.numpy())
    total_samples += labels.shape[0]

accuracy = correct_predictions / total_samples
print(f'Quantized accuracy (W8A16): {accuracy:.4f}')
# End of quantized accuracy

# Export the model
# Export the model for on-target inference. Saves ONNX model without quantization nodes
# and encodings file with all tensor encodings in JSON format at provided path.
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2')
# End of exporting the model
