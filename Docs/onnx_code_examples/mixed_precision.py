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
import math
import os
import numpy as np
import onnx
import onnxsim
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm
from aimet_onnx.defs import DataLoader
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_common.defs import QuantizationDataType, CallbackFunc
from aimet_onnx.mixed_precision import choose_mixed_precision
# End step 0

# Step 1
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
# End of loading the model

# Prepare model with onnx-simplifier
try:
    model, _ = onnxsim.simplify(model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')
# End of prepare model

# Set up dataloader
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
NUM_CALIBRATION_SAMPLES = 1024
NUM_EVAL_SAMPLES = 50000
unlabeled_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_CALIBRATION_SAMPLES / BATCH_SIZE)
)
eval_data_loader = CustomDataLoader(
    dataset, BATCH_SIZE, math.ceil(NUM_EVAL_SAMPLES / BATCH_SIZE), unlabeled=False
)
# End of setting up dataloader

def forward_pass(session, _):
    input_name = session.get_inputs()[0].name
    for inputs in tqdm(unlabeled_data_loader):
        session.run(None, {input_name: inputs})

def evaluate(session, _):
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in tqdm(eval_data_loader):
        input_name = sim.session.get_inputs()[0].name
        pred_probs, *_ = sim.session.run(None, {input_name: inputs})
        pred_labels = np.argmax(pred_probs, axis=1)
        correct_predictions += np.sum(pred_labels == labels)
        total_samples += labels.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy
# End step 1

# Step 2
# Define parameters to pass to mixed precision algo
default_bitwidth = 16

# ((activation bitwidth, activation data type), (param bitwidth, param data type))
candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
             ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
             ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]
# Allowed accuracy drop in absolute value
allowed_accuracy_drop = 0.5 # Implies 50% drop

eval_callback_for_phase_1 = CallbackFunc(evaluate, func_callback_args=None)
eval_callback_for_phase_2 = CallbackFunc(evaluate, func_callback_args=None)

forward_pass_callback = CallbackFunc(forward_pass, func_callback_args=None)

# Create quant sim
sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_activation_bw=default_bitwidth)
sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=None)

# Call the mixed precision algo with clean start = True i.e. new accuracy list and pareto list will be generated
# If set to False then pareto front list and accuracy list will be loaded from the provided directory path
# A allowed_accuracy_drop can be specified to export the final model with reference to the pareto list
pareto_front_list = choose_mixed_precision(sim, candidates, eval_callback_for_phase_1,
                                           eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                           clean_start=True, forward_pass_callback=forward_pass_callback)
print(pareto_front_list)

# Set clean_start to False to start from an existing cache
# Set allowed_accuracy_drop to 0.9 to export the 90% drop point in pareto list
allowed_accuracy_drop = 0.9
pareto_front_list = choose_mixed_precision(sim, candidates, eval_callback_for_phase_1,
                                           eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                           clean_start=False, forward_pass_callback=forward_pass_callback)
print(pareto_front_list)
sim.export("./data", str(allowed_accuracy_drop))
# End step 2

