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

import torch
import math
import numpy as np

from torchvision.models import mobilenet_v2
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)

# End of step 1

# Step 2
num_batches = 32
labeled_data = load_dataset('imagenet-1k', streaming=True, split="train")
labeled_data_loader = DataLoader(labeled_data, batch_size=num_batches, num_workers = 4)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

dataset = load_dataset(
    'ILSVRC/imagenet-1k',
    split='validation',
)
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
        preprocess(image.convert('RGB')).numpy() for image in examples['image']
    ]
    examples['image'] = [np.expand_dims(image, axis=0) for image in examples['image']]
    return examples


dataset.set_transform(transforms)

BATCH_SIZE = 1
EVAL_DATASET_SIZE = 64
CALIBRATION_DATASET_SIZE = 32

class CustomDataLoader(DataLoader):
    def __init__(self, data: np.ndarray, batch_size: int, iterations: int):
        super().__init__(data, batch_size, iterations)
        self._batch_index = 0

    def __iter__(self):
        self._batch_index = 0
        return self

    def __next__(self):
        if self._batch_index < self.iterations:
            str_idx = self._batch_index
            end_idx = self._batch_index + self.batch_size
            self._batch_index += 1
            return self._data[str_idx:end_idx]
        else:
            raise StopIteration


unlabelled_data_loader = CustomDataLoader(
    dataset['image'], BATCH_SIZE, math.ceil(CALIBRATION_DATASET_SIZE / BATCH_SIZE)
)
unlabeled_data = load_dataset('imagenet-1k', streaming=True, split="unlabeled")
unlabeled_data_loader = DataLoader(unlabeled_data, batch_size=num_batches, num_workers = 4)

# End of step 2

# Step 3
from typing import Optional

def eval_callback(model: torch.nn.Module, num_of_samples: Optional[int] = None) -> float:
    data_loader = CustomDataLoader(
        dataset, BATCH_SIZE, math.ceil(EVAL_DATASET_SIZE / BATCH_SIZE)
    )
    if num_of_samples:
        iterations = math.ceil(num_of_samples / data_loader.batch_size)
    else:
        iterations = len(data_loader)
    batch_cntr = 1
    acc_top1 = 0
    for data in data_loader:
        input_data = data['image'][0]
        target = data['label']
        pred_probs = model(input_data.cuda())
        pred_labels = np.argmax(pred_probs, axis=1)
        acc_top1 += np.sum(pred_labels == target)

        batch_cntr += 1
        if batch_cntr > iterations:
            break
    acc_top1 /= len(data_loader)
    return acc_top1

# End of step 3

# Step 4. Create AutoQuant object
from aimet_torch.auto_quant import AutoQuantWithAutoMixedPrecision

auto_quant = AutoQuantWithAutoMixedPrecision(
    model, dummy_input, unlabelled_data_loader, eval_callback
)
# End of step 4

# Step 5. (Optional) Set AdaRound params
from aimet_onnx.adaround.adaround_weight import AdaroundParameters

ADAROUND_DATASET_SIZE = 128
adaround_data_loader = DataLoader(
    data=dataset['image'],
    batch_size=BATCH_SIZE,
    iterations=math.ceil(ADAROUND_DATASET_SIZE / BATCH_SIZE),
)
adaround_params = AdaroundParameters(
    adaround_data_loader, num_batches=len(adaround_data_loader)
)
auto_quant.set_adaround_params(adaround_params)
# End of step 5

# Step 6. (Optional) Set AMP params
from aimet_common.defs import QuantizationDataType

W8A8 = (
    (8, QuantizationDataType.int),  # A: int8
    (8, QuantizationDataType.int),  # W: int8
)
W8A16 = (
    (16, QuantizationDataType.int),  # A: int16
    (8, QuantizationDataType.int),  # W: int8
)
auto_quant.set_mixed_precision_params(candidates=[W8A16, W8A8])
# End of step 6

# Step 7. Run AutoQuant
sim, initial_accuracy = auto_quant.run_inference()
model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(allowed_accuracy_drop=0.01)

print(f"- Quantized Accuracy (before optimization): {initial_accuracy:.4f}")
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}")
# End of step 7