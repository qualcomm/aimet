# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Code example for AutoQuant """

import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, datasets, transforms

from aimet_torch.v1.adaround.adaround_weight import AdaroundParameters
from aimet_torch.v1.auto_quant import AutoQuant

# Step 1. Define constants and helper functions
EVAL_DATASET_SIZE = 5000
CALIBRATION_DATASET_SIZE = 2000
BATCH_SIZE = 100

_subset_samplers = {}

def _create_sampled_data_loader(dataset, num_samples):
    if num_samples not in _subset_samplers:
        indices = random.sample(range(len(dataset)), num_samples)
        _subset_samplers[num_samples] = SubsetRandomSampler(indices=indices)
    return DataLoader(dataset,
                      sampler=_subset_samplers[num_samples],
                      batch_size=BATCH_SIZE)

# Step 2. Prepare model and dataset
fp32_model = models.resnet18(pretrained=True).eval()

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

transform = transforms.Compose((
    transforms.ToTensor(),
))
# NOTE: In the actual use cases, a real dataset should provide by the users.
eval_dataset = datasets.FakeData(size=EVAL_DATASET_SIZE,
                                 image_size=input_shape[1:],
                                 num_classes=1000,
                                 transform=transform)

# Step 3. Prepare unlabeled dataset
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        images, _ = self._dataset[index]
        return images

unlabeled_dataset = UnlabeledDatasetWrapper(eval_dataset)
unlabeled_data_loader = _create_sampled_data_loader(unlabeled_dataset, CALIBRATION_DATASET_SIZE)

# Step 4. Prepare eval callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    if num_samples is None:
        num_samples = len(eval_dataset)

    eval_data_loader = _create_sampled_data_loader(eval_dataset, num_samples)

    num_correct_predictions = 0
    for images, labels in eval_data_loader:
        predictions = torch.argmax(model(images.cuda()), dim=1)
        num_correct_predictions += torch.sum(predictions.cpu() == labels)

    return int(num_correct_predictions) / num_samples

# Step 5. Create AutoQuant object
auto_quant = AutoQuant(allowed_accuracy_drop=0.01,
                       unlabeled_dataset_iterable=unlabeled_data_loader,
                       eval_callback=eval_callback)

# Step 6. (Optional) Set adaround params
ADAROUND_DATASET_SIZE = 2000
adaround_data_loader = _create_sampled_data_loader(unlabeled_dataset, ADAROUND_DATASET_SIZE)
adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader))
auto_quant.set_adaround_params(adaround_params)

# Step 7. Run AutoQuant
model, accuracy, encoding_path =\
    auto_quant.apply(fp32_model.cuda(),
                     dummy_input_on_cpu=dummy_input.cpu(),
                     dummy_input_on_gpu=dummy_input.cuda())

print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}")
