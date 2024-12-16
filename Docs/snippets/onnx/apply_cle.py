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
import os

import onnx
import onnxsim
import torch
from aimet_onnx.cross_layer_equalization import equalize_model
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

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
)
# Load exported ONNX model
model = onnx.load_model(file_path)

# Simplifying the model
try:
    model, _ = onnxsim.simplify(model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')

initializers = {init.name: init for init in model.graph.initializer}
prev_conv_weight = onnx.numpy_helper.to_array(
    initializers[model.graph.node[4].input[1]]
)
next_conv_weight = onnx.numpy_helper.to_array(
    initializers[model.graph.node[5].input[1]]
)

print("*** Before cross-layer equalization ***")

print("\nmodel.graph.node[4]:")
print(model.graph.node[4].name)

print("\nmodel.graph.node[5]:")
print(model.graph.node[5].name)

print("\nPrev Conv weight")
print(prev_conv_weight)

print("\nNext Conv weight")
print(next_conv_weight)
[step_1]
# Cross-layer equalization is working as in-place manner
equalize_model(model=model)

prev_conv_weight = onnx.numpy_helper.to_array(
    initializers[model.graph.node[4].input[1]]
)
next_conv_weight = onnx.numpy_helper.to_array(
    initializers[model.graph.node[5].input[1]]
)

print("*** After cross-layer equalization ***")

print("\nPrev Conv weight")
print(prev_conv_weight)

print("\nNext Conv weight")
print(next_conv_weight)
