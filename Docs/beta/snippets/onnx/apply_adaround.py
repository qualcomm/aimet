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
# Set up model
import math
import os

import numpy as np
import onnx
import onnxsim
import torch
from aimet_common.defs import QuantScheme
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_onnx.defs import DataLoader
from aimet_onnx.quantsim import QuantizationSimModel
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path as you wish, we are using temporary directory for now
file_path = os.path.join('/tmp', f'mobilenet_v2.onnx')
torch.onnx.export(
    pt_model,
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
try:
    model, _ = onnxsim.simplify(model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')
# End of model

# Set up dataloader
dataset = load_dataset(
    'ILSVRC/imagenet-1k',
    split='validation',
)


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        iterations: int,
        unlabeled: bool = True,
    ):
        super().__init__(data, batch_size, iterations)
        self._current_iteration = 0
        self._unlabeled = unlabeled

    def __iter__(self):
        self._current_iteration = 0
        return self

    def __next__(self):
        if self._current_iteration < self.iterations:
            start = self._current_iteration * self.batch_size
            end = start + self.batch_size
            self._current_iteration += 1

            batch_data = self._data[start:end]
            if self._unlabeled:
                return np.stack(batch_data['image'])
            else:
                return np.stack(batch_data['image']), np.stack(batch_data['label'])
        else:
            raise StopIteration


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def transforms(examples):
    examples['image'] = [
        preprocess(image.convert('RGB')) for image in examples['image']
    ]
    return examples


dataset.set_transform(transforms)

BATCH_SIZE = 32
NUM_SAMPLES = 256
unlabeled_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_SAMPLES / BATCH_SIZE)
)
# End of dataloader


# Step 1
def pass_calibration_data(session, _):
    input_name = session.get_inputs()[0].name
    for inputs in unlabeled_data_loader:
        session.run(None, {input_name: inputs})


PARAM_BITWIDTH = 4
ACTIVATION_BITWIDTH = 8
params = AdaroundParameters(
    data_loader=unlabeled_data_loader,
    num_batches=math.ceil(NUM_SAMPLES / BATCH_SIZE),
    default_num_iterations=5,
    forward_fn=pass_calibration_data,
    forward_pass_callback_args=None,
)
ada_rounded_model = Adaround.apply_adaround(
    model,
    params,
    path='/tmp',
    filename_prefix='mobilenet_v2',
    default_param_bw=PARAM_BITWIDTH,
)
# End of step 1

# Step 2
sim = QuantizationSimModel(
    ada_rounded_model,
    quant_scheme=QuantScheme.post_training_tf,
    default_param_bw=PARAM_BITWIDTH,
    default_activation_bw=ACTIVATION_BITWIDTH,
)

# AdaRound optimizes the rounding of weight quantizers only. These values are preserved through set_and_freeze_param_encodings()
sim.set_and_freeze_param_encodings(encoding_path='/tmp/mobilenet_v2.encodings')

# The activation quantizers remain uninitialized and derived through compute_encodings()
sim.compute_encodings(pass_calibration_data, None)
# End of step 2

# Step 3
eval_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_SAMPLES / BATCH_SIZE), unlabeled=False
)
correct_predictions = 0
total_samples = 0
for inputs, labels in eval_data_loader:
    input_name = sim.session.get_inputs()[0].name
    pred_probs, *_ = sim.session.run(None, {input_name: inputs})
    pred_labels = np.argmax(pred_probs, axis=1)
    correct_predictions += np.sum(pred_labels == labels)
    total_samples += labels.shape[0]

accuracy = correct_predictions / total_samples
# End of step 3

# Step 4
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2')
# End of step 4
