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
# Step 1
import math
import os
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from aimet_common.defs import QuantizationDataType
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_onnx.adaround.adaround_weight import AdaroundParameters
from aimet_onnx.auto_quant_v2 import AutoQuantWithAutoMixedPrecision
from aimet_onnx.defs import DataLoader
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
print(pt_model)

# Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
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
# End of step 1

# Step 2
try:
    model, _ = onnxsim.simplify(model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')


dataset = load_dataset(
    'ILSVRC/imagenet-1k',
    split='validation',
).shuffle()


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
NUM_CALIBRATION_SAMPLES = 2048
NUM_EVAL_SAMPLES = 50000
unlabeled_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_CALIBRATION_SAMPLES / BATCH_SIZE)
)
eval_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_EVAL_SAMPLES / BATCH_SIZE), unlabeled=False
)
# End of step 2


# Step 3
def eval_callback(
    session: ort.InferenceSession, num_of_samples: Optional[int] = None
) -> float:
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in tqdm(eval_data_loader):
        input_name = session.get_inputs()[0].name
        pred_probs, *_ = session.run(None, {input_name: inputs})
        pred_labels = np.argmax(pred_probs, axis=1)
        correct_predictions += np.sum(pred_labels == labels)
        total_samples += labels.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy
# End of step 3

# Step 4. Create AutoQuant object
dummy_input = {'input': np.random.randn(*input_shape).astype(np.float32)}
auto_quant = AutoQuantWithAutoMixedPrecision(
    model,
    dummy_input,
    unlabeled_data_loader,
    eval_callback,
    param_bw=4,
    output_bw=8,
    config_file=get_path_for_per_channel_config(),
)
# End of step 4

# Step 5. Set AdaRound params
adaround_params = AdaroundParameters(
    unlabeled_data_loader, num_batches=len(unlabeled_data_loader)
)
auto_quant.set_adaround_params(adaround_params)
# End of step 5

# Step 6. Set AMP params
W4A8 = (
    (8, QuantizationDataType.int),  # A: int8
    (4, QuantizationDataType.int),  # W: int4
)
W8A8 = (
    (8, QuantizationDataType.int),  # A: int8
    (8, QuantizationDataType.int),  # W: int8
)
auto_quant.set_mixed_precision_params(candidates=[W4A8, W8A8])
# End of step 6

# Step 7. Run AutoQuant
sim, initial_accuracy = auto_quant.run_inference()
model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(
    allowed_accuracy_drop=0.01
)

print(f'- Quantized Accuracy (before optimization): {initial_accuracy:.4f}')
print(f'- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}')
# End of step 7
