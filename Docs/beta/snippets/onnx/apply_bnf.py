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
import os
import tempfile

import onnx
import onnxsim
import torch
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
print(pt_model)
# MobileNetV2(
#   (features): Sequential(
#     (0): Conv2dNormActivation(
#       (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#     )
#     (1): InvertedResidual(
#       (conv): Sequential(
#         (0): Conv2dNormActivation(
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
#           (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU6(inplace=True)
#         )
#         (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     ...
#     (18): Conv2dNormActivation(
#       (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#     )
#   )
#   (classifier): Sequential(
#     (0): Dropout(p=0.2, inplace=False)
#     (1): Linear(in_features=1280, out_features=1000, bias=True)
#   )
# )
# Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path as you wish, we are using temporary directory for now
temp_dir = tempfile.TemporaryDirectory()
file_path = os.path.join(temp_dir.name, f'mobilenet_v2.onnx')
torch.onnx.export(
    pt_model,
    (dummy_input,),
    file_path,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
)
# Load exported ONNX model
model = onnx.load_model(file_path)
print(model.graph.node[0])
# input: "input"
# input: "features.0.0.weight"
# output: "/features/features.0/features.0.0/Conv_output_0"
# name: "/features/features.0/features.0.0/Conv"
# op_type: "Conv"
# attribute {
#   name: "dilations"
#   ...
print(model.graph.node[1])
# input: "/features/features.0/features.0.0/Conv_output_0"
# input: "features.0.1.weight"
# input: "features.0.1.bias"
# input: "features.0.1.running_mean"
# input: "features.0.1.running_var"
# output: "/features/features.0/features.0.1/BatchNormalization_output_0"
# name: "/features/features.0/features.0.1/BatchNormalization"
# op_type: "BatchNormalization"
# attribute {
#   name: "epsilon"
#   ...
# End of step 1

# Step 2
# Unlike AIMET, which supports both forward/backward folding, ONNX simplifier only performs backward folding.
# Therefore, we disable the corresponding optimization in `skipped_optimizers` and proceed with the example
try:
    model, _ = onnxsim.simplify(model, skipped_optimizers=['fuse_bn_into_conv'])
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')

print(model.graph.node[0])
# input: "input"
# input: "features.0.0.weight"
# output: "/features/features.0/features.0.0/Conv_output_0"
# name: "/features/features.0/features.0.0/Conv"
# op_type: "Conv"
# attribute {
#   name: "dilations"
#   ...
initializers = {init.name: init for init in model.graph.initializer}
conv_weight_name = model.graph.node[0].input[1]
conv_weight = initializers[conv_weight_name]
conv_weight_array = onnx.numpy_helper.to_array(conv_weight)
print(conv_weight_array)
# [[[[-6.31080866e-02 -1.87656835e-01 -1.51876003e-01]
#    [-4.93787616e-01 -6.42477691e-01 -5.89348674e-01]
#    [-6.80053532e-01 -9.74478185e-01 -7.63172388e-01]]
#
#   [[-1.63499508e-02 -1.84824076e-02  6.27826452e-02]
#    [ 3.54360677e-02  5.89796454e-02  1.06927991e-01]
#    [ 1.69947863e-01  1.46988630e-01  1.85209349e-01]]
#
#   [[ 1.13947354e-01  1.63159192e-01  1.04832031e-01]
#    [ 4.08243328e-01  5.74886918e-01  4.72697765e-01]
#    [ 5.75474739e-01  7.15027273e-01  5.37017465e-01]]]
#   ...
#  [[[-6.72712876e-03  2.03205068e-02 -4.88756672e-02]
#    [ 4.24612552e-01  1.23068285e+00  2.31590062e-01]
#    [-4.40555036e-01 -1.26949096e+00 -1.79759294e-01]]
#
#   [[ 1.68908127e-02 -3.78823304e-03  7.27318786e-03]
#    [ 7.86119163e-01  1.94402826e+00  4.12017554e-01]
#    [-8.11430693e-01 -2.00229359e+00 -3.28911036e-01]]
#
#   [[ 1.24257803e-02 -4.73242160e-03 -1.81884710e-02]
#    [ 2.32141271e-01  7.22583652e-01  1.21250950e-01]
#    [-2.59643137e-01 -7.18673885e-01 -9.19778645e-02]]]]
print(model.graph.node[1])
# input: "/features/features.0/features.0.0/Conv_output_0"
# input: "features.0.1.weight"
# input: "features.0.1.bias"
# input: "features.0.1.running_mean"
# input: "features.0.1.running_var"
# output: "/features/features.0/features.0.1/BatchNormalization_output_0"
# name: "/features/features.0/features.0.1/BatchNormalization"
# op_type: "BatchNormalization"
# attribute {
#   name: "epsilon"
#   ...
# End of step 2

# Step 3
_ = fold_all_batch_norms_to_weight(model=model)
# End of step 3

# Step 4
print(model.graph.node[0])
# input: "input"
# input: "features.0.0.weight"
# input: "/features/features.0/features.0.0/Conv.bias"
# output: "/features/features.0/features.0.0/Conv_output_0"
# name: "/features/features.0/features.0.0/Conv"
# op_type: "Conv"
# attribute {
#   name: "dilations"
#   ...
conv_weight = initializers[conv_weight_name]
conv_weight_array = onnx.numpy_helper.to_array(conv_weight)
print(conv_weight_array)
# [[[[-2.00183112e-02 -5.95260113e-02 -4.81760912e-02]
#    [-1.56632766e-01 -2.03798249e-01 -1.86945379e-01]
#    [-2.15717569e-01 -3.09111059e-01 -2.42083430e-01]]
#
#   [[-5.18631469e-03 -5.86274406e-03  1.99150778e-02]
#    [ 1.12405596e-02  1.87087413e-02  3.39182802e-02]
#    [ 5.39086089e-02  4.66257855e-02  5.87496534e-02]]
#
#   [[ 3.61448675e-02  5.17551936e-02  3.32534276e-02]
#    [ 1.29497528e-01  1.82358012e-01  1.49942920e-01]
#    [ 1.82544470e-01  2.26811469e-01  1.70345560e-01]]]
#   ...
#  [[[-6.55435352e-03  1.97986085e-02 -4.76203784e-02]
#    [ 4.13707078e-01  1.19907475e+00  2.25642055e-01]
#    [-4.29240108e-01 -1.23688614e+00 -1.75142467e-01]]
#
#   [[ 1.64570007e-02 -3.69093847e-03  7.08638830e-03]
#    [ 7.65928984e-01  1.89409912e+00  4.01435554e-01]
#    [-7.90590465e-01 -1.95086801e+00 -3.20463508e-01]]
#
#   [[ 1.21066449e-02 -4.61087702e-03 -1.77213307e-02]
#    [ 2.26179108e-01  7.04025269e-01  1.18136823e-01]
#    [-2.52974629e-01 -7.00215936e-01 -8.96155685e-02]]]]
print(model.graph.node[1])
# input: "/features/features.0/features.0.0/Conv_output_0"
# input: "/features/features.0/features.0.2/Constant_output_0"
# input: "/features/features.0/features.0.2/Constant_1_output_0"
# output: "/features/features.0/features.0.2/Clip_output_0"
# name: "/features/features.0/features.0.2/Clip"
# op_type: "Clip"
temp_dir.cleanup()
# End of step 4
