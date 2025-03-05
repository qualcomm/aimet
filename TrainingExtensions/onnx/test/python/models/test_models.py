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
"""Dummy models for testing"""
import torch
import torch.nn as nn
from onnx import load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from torch.nn import functional as F
import tempfile

class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(80, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(self.conv2_drop(x))
        x = x.view(-1, 80)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithOneSplit(nn.Module):
    def __init__(self):
        super(ModelWithOneSplit, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        z = self.conv3(x)
        return z + y


def model_with_split():
    x = torch.randn(1, 1, 10, 10, requires_grad=True)
    model = ModelWithOneSplit()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_with_one_split.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_with_one_split.onnx'))
    return model


def model_small_mnist():
    x = torch.randn(1, 1, 10, 10, requires_grad=True)
    model = SmallMnist()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_simple_mnist.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_simple_mnist.onnx'))
    return model


class SingleResidualWithAvgPool(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidualWithAvgPool, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AvgPool2d(5)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def single_residual_model():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model = SingleResidualWithAvgPool()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_single_residual.onnx",  # where to save the model (can be a file or file-like object)
                      training=torch.onnx.TrainingMode.EVAL, # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_single_residual.onnx'))
    return model


class ConcatModel(nn.Module):
    """ A model with concat op.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ConcatModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2)
        self.fc = nn.Linear(504, num_classes)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[1])
        c3 = self.conv3(inputs[2])
        cat_inputs = [c1, c2, c3]
        x = torch.cat(cat_inputs, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def concat_model():
    x = (torch.rand(1, 3, 8, 8, requires_grad=True), torch.rand(1, 3, 8, 8, requires_grad=True),
         torch.rand(1, 3, 8, 8, requires_grad=True))
    model = ConcatModel()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./concat_model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./concat_model.onnx'))
    return model

class MLP(nn.Module):
    """Implementation of a Multi-layer perceptron (also called FFN) with ReLu normalization on the hidden layers"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            decoder_small_weights_init_last_layer: bool = False,
            decoder_small_weights_init_last_layer_std: float = 0.001,
    ) -> None:
        """Initializes the linear layers

        :param input_dim: Input layer dimension
        :type input_dim: int
        :param hidden_dim: Hidden layer(s) dimension
        :type hidden_dim: int
        :param output_dim: Output layer dimension
        :type output_dim: int
        :param num_layers: Number of layers (including input/output layers)
        :type num_layers: int
        :param decoder_small_weights_init_last_layer: whether to use small weights on decoder last layer
        :type decoder_small_weights_init_last_layer: bool
        :param decoder_small_weights_init_last_layer_std: the std to set decoder last layer (only effective
            when setting decoder_small_weights_init_last_layer to true)
        :type decoder_small_weights_init_last_layer_std: float
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(inp, out) for inp, out in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters(decoder_small_weights_init_last_layer, decoder_small_weights_init_last_layer_std)

    def reset_parameters(self, decoder_small_weights_init_last_layer: bool,
                         decoder_small_weights_init_last_layer_std: float) -> None:
        """Initialize MLP weights with kaiming normalization"""
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        if decoder_small_weights_init_last_layer:
            nn.init.normal_(self.layers[-1].weight, std=decoder_small_weights_init_last_layer_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward pass

        :param x: Input features or embedding data
        :type x: torch.Tensor

        :return: Embedding tensor
        :rtype: torch.Tensor
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x.view(-1)

def linear_layer_model():
    model = MLP(input_dim=258, hidden_dim=512, output_dim=90, num_layers=2)

    x = torch.randn(512, 258, requires_grad=True)
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./linear_layer_model.onnx",  # where to save the model (can be a file or file-like object)
                      training=torch.onnx.TrainingMode.EVAL,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])

    model = ONNXModel(load_model('./linear_layer_model.onnx'))
    return model


def layernorm_model(elementwise_affine=True, bias=True, include_add_ops = False):
    class LayerNormModel(torch.nn.Module):
        def __init__(self, include_add_ops = False):
            super(LayerNormModel, self).__init__()
            self.include_add_ops = include_add_ops
            self.layer_norm = torch.nn.LayerNorm(32, elementwise_affine=elementwise_affine, bias=bias)

        def forward(self, x: torch.Tensor):
            if self.include_add_ops:
                x = x + x
            x = self.layer_norm(x)
            if self.include_add_ops:
                x = x + x
            return x

    torch.manual_seed(10)
    model = LayerNormModel(include_add_ops=include_add_ops).eval()
    with tempfile.NamedTemporaryFile(prefix="layernorm_", suffix=".onnx") as onnx_model_path:
        x = torch.randn((1, 3, 32, 32))
        torch.onnx.export(model, x, onnx_model_path.name, input_names=['input'], output_names=['output'], opset_version=16)
        model = load_model(onnx_model_path.name)

    return model


def conv_relu_model():
    class ConvReluModel(torch.nn.Module):
        def __init__(self):
            super(ConvReluModel, self).__init__()
            self._conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)
            self._relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self._relu(self._conv_0(x))

    torch.manual_seed(10)
    model = ConvReluModel().eval()
    x = torch.randn((1, 3, 8, 8))

    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./conv_relu.onnx", # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.EVAL,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'},
                      })

    model = load_model('./conv_relu.onnx')
    return model


class SingleLinearLayerModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SingleLinearLayerModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


def single_linear_layer_model():
    model = SingleLinearLayerModel(100,100)
    x = torch.randn(1, 100, 100, requires_grad=True)
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./single_linear_layer_model.onnx",  # where to save the model (can be a file or file-like object)
                      training=torch.onnx.TrainingMode.EVAL,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])

    model = ONNXModel(load_model('./single_linear_layer_model.onnx'))
    return model

class SingleConvLayerModel(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(SingleConvLayerModel, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        return x


def single_conv_layer_model():
    model = SingleConvLayerModel(5,10)
    x = torch.randn(1, 5, 5, 5, requires_grad=True)
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./single_conv_layer_model.onnx",  # where to save the model (can be a file or file-like object)
                      training=torch.onnx.TrainingMode.EVAL,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])

    model = ONNXModel(load_model('./single_conv_layer_model.onnx'))
    return model

class ConvTransposeConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvTransposeConvModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=2, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        self.relu1 = torch.nn.ReLU()

    def forward(self, inp):
        outs = self.conv1(inp)
        outs = outs.permute(0, 2, 3, 1)
        outs1 = self.conv2(outs)
        outs2 = self.relu1(outs1)
        return outs2
