# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Models for use in unit testing """

import torch
from torch import nn

import aimet_torch._base.nn.modules.custom as aimet_modules
from aimet_torch.v2.nn import QuantizationMixin


class SimpleConditional(torch.nn.Module):
    """
    Model using conditional paths
    Expected input shape = (1, 3)
    """
    def __init__(self):
        super(SimpleConditional, self).__init__()
        self.prelu1 = torch.nn.PReLU(init=.3)
        self.prelu2 = torch.nn.PReLU(init=.4)
        self.linear1 = torch.nn.Linear(3, 2)
        self.linear2 = torch.nn.Linear(3, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, _input, condition):
        if condition:
            x = self.linear1(_input)
            x = x.view(x.size(0), -1)
            x = self.prelu1(x)
            return x
        x = self.linear2(_input)
        x = self.prelu2(x)
        x = self.softmax(x)
        return x

class ModelWithTwoInputs(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputs, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        x = x1 + x2
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class FakeMultiOutputOp(torch.autograd.Function):
    """
    This function helps create a custom onnx op to simulate a 5 output tensor onnx node
    Note: the forward pass has some tensor computation to prevent torch onnx export from removing onnx node.
    """

    @staticmethod
    def symbolic(g, inp):
        """
        Magic method that helps with exporting a custom ONNX node
        """
        return g.op('aimet_torch::FakeMultiOutputOp', inp, outputs=5)

    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        return x * 2, x * 4, x * 8, x * 16, x * 32

    @staticmethod
    def backward(ctx, _grad):  # pylint: disable=arguments-differ
        raise NotImplementedError()


class ModuleWith5Output(torch.nn.Module):
    def forward(self, x):
        return FakeMultiOutputOp.apply(x)


class ModelWith5Output(torch.nn.Module):
    def __init__(self):
        super(ModelWith5Output, self).__init__()
        self.cust = ModuleWith5Output()

    def forward(self, x):
        return self.cust(x)


class SoftMaxAvgPoolModel(torch.nn.Module):
    def __init__(self):
        super(SoftMaxAvgPoolModel, self).__init__()
        self.sfmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(3)

    def forward(self, inp):
        x = self.sfmax(inp)
        return self.avgpool(x)


class SingleResidual(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidual, self).__init__()
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
        self.ada = nn.AdaptiveAvgPool2d(5)
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


class SingleResidualWithModuleAdd(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidualWithModuleAdd, self).__init__()
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
        self.add = aimet_modules.Add()
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AdaptiveAvgPool2d(5)
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
        x = self.add(x, residual)
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiInput(nn.Module):
    """ A model with multiple inputs.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=3):
        super(MultiInput, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[1])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithBertCustomLayerNormGelu(torch.nn.Module):
    """ Model with PyTorch LayerNorm and gelu """

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        # default attribute -
        # eps = 1e-05 and elementwise_affine = True
        # parameters : weight and bias
        self.customln1 = torch.nn.LayerNorm(4)
        self.gelu1 = torch.nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.customln1(x)
        x = self.gelu1(x)
        return x


class QuantSimTinyModel(nn.Module):
    """ Use this model for quantsim_config unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(QuantSimTinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.fc = nn.Linear(36, 12)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@QuantizationMixin.implements(ModuleWith5Output)
class QuantizationModuleWith5Output(QuantizationMixin, ModuleWith5Output):
    def __quant_init__(self):
        super().__quant_init__()
        self.output_quantizers = torch.nn.ModuleList([None, None, None, None, None])

    def quantized_forward(self, input):
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)
        outputs = super().forward(input)
        return tuple(
            quantizer(out) if quantizer else out
            for out, quantizer in zip(outputs, self.output_quantizers)
        )


class BasicConv2d(nn.Module):
    """ A Simple Super Node Model used as building block in Hierarchical Model  """

    def __init__(self, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(64, 64, bias=False, **kwargs)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.bn = nn.BatchNorm2d(64, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        x = self.conv(inputs[0])
        x = self.dropout(x)
        x = self.bn(x)
        return self.relu(x)
