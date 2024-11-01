# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Any
import os
import tempfile
from pathlib import Path

import onnx
import torch.nn.functional as F
from torch import nn as nn
from torchvision.ops import roi_align
import numpy as np
import torch
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.batchnorm import _BatchNorm
from onnx import helper, numpy_helper, OperatorSetIdProto, TensorProto, load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_common import libquant_info
from .mobilenet import MockMobileNetV1, MockMobileNetV11


class Add(torch.nn.Module):
    """ Add module for a functional add"""
    # pylint:disable=arguments-differ
    def forward(self, x: Any, y: Any) -> Any:
        """
        Forward-pass routine for add op
        """
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.add(x, y)
        else:
            out = x + y
        return out

class Multiply(torch.nn.Module):
    """ Multiply module for a functional multiply"""
    # pylint:disable=arguments-differ
    def forward(self, x: Any, y: Any) -> Any:
        """
        Forward-pass routine for multiply op
        """
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.mul(x, y)
        else:
            out = x * y
        return out


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


class DictInputModel(nn.Module):
    """ Model with dictionary as input. """
    def __init__(self, num_classes=3):
        super(DictInputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0]['inp_1'])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[0]['inp_2'])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class ModuleListModel(nn.Module):
    """ A model with modules defined using ModuleLists.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ModuleListModel, self).__init__()
        self.mod_list = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),       # use 4th
            nn.ReLU(inplace=True),      # use 3rd
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),                # use 5th
            nn.ReLU(),      # dummy unused op
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)        # use 1st
        ])
        self.seq_list = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),                # use 6th
            nn.ReLU(),      # dummy unused op
            nn.BatchNorm2d(16),     # use 2nd
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.mod_list[4](inputs[0])
        x = self.seq_list[2](x)
        x = self.mod_list[1](x)
        x = self.mod_list[0](x)
        x = self.mod_list[2](x)
        x = self.seq_list[0](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinyModel(nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
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


class ModelWithDropouts(nn.Module):
    """ Use this model for unit testing purposes. """

    def __init__(self):
        super(ModelWithDropouts, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=.4)
        self.dropout2 = nn.Dropout2d(p=.6)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithReusedNodes(nn.Module):
    """ Model that reuses a relu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithReusedNodes, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithFunctionalOps(nn.Module):
    """ Model that uses functional modules instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithFunctionalOps, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, torch.randn(10, 2592))
        return x


class SequentialModel(nn.Module):
    """ A model with modules defined using nn.Sequential.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(SequentialModel, self).__init__()
        self.seq_list = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class MultiConv2dModel(nn.Module):
    """ Sequential Model contains sequences of BasicConv2d Model  """

    def __init__(self):
        super(MultiConv2dModel, self).__init__()
        self.seq_list = nn.Sequential(
            BasicConv2d(kernel_size=3),
            BasicConv2d(kernel_size=1),
            BasicConv2d(kernel_size=3)
        )

    def forward(self, *inputs):
        return self.seq_list(inputs[0])


class NestedModel(nn.Module):
    """ Aggregation Model contains two instance of Tiny Model """

    def __init__(self):
        super(NestedModel, self).__init__()
        self.tm1 = TinyModel()
        self.tm2 = TinyModel()

    def forward(self, *inputs):
        c1 = self.tm1(inputs[0])
        c2 = self.tm2(inputs[1])
        cat_inputs = [c1, c2]
        x = torch.cat(cat_inputs, 1)
        return x


class HierarchicalModel(nn.Module):
    """ Aggregation Model contains multi-level of PyTorch Module
        Expected 5 inputs with shapes  in the following order:
            (1, 64, 32, 32)
            (1,  3, 32, 32)
            (1, 64, 32, 32)
            (1,  3, 32, 32)
            (1,  3,  8,  8) """

    def __init__(self):
        super(HierarchicalModel, self).__init__()
        self.conv1 = BasicConv2d(kernel_size=3)
        self.conv2 = BasicConv2d(kernel_size=3)
        self.multi_conv = MultiConv2dModel()
        self.nm1 = NestedModel()
        self.nm2 = NestedModel()
        self.sq = SequentialModel()

    def forward(self, *inputs):
        x = self.conv1((inputs[0]))
        x = x.narrow(1, 0, 3)
        c1 = self.nm1(x, inputs[1])
        x = self.conv2(inputs[2])
        x = self.multi_conv(x)
        x = x.narrow(1, 0, 3)
        c2 = self.nm2(x, inputs[3])
        c3 = self.sq(inputs[4])
        cat_inputs = [c1, c2, c3]
        x = torch.cat(cat_inputs, 1)
        return x


class PassThroughOpLastLayerModel(nn.Module):
    """ Model with PassThroughOp as last layer. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(PassThroughOpLastLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.passthrough = torch.nn.Identity()

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.passthrough(x)
        return x


class TransposedConvModel(torch.nn.Module):
    """
    Model with transposed conv2D
    """
    def __init__(self):
        super(TransposedConvModel, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose2d(10, 20, 3)
        self.bn2 = torch.nn.BatchNorm2d(20)

    # pylint: disable=arguments-differ
    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return x

class DepthwiseTransposedConvModel(TransposedConvModel):

    def __init__(self):
        super(DepthwiseTransposedConvModel, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
        self.conv2 = torch.nn.ConvTranspose2d(10, 20, 3, groups=10)


class TransposedConvModelWithoutBN(torch.nn.Module):
    """
    Model with transposed conv2D
    """
    def __init__(self):
        super(TransposedConvModelWithoutBN, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3)

    # pylint: disable=arguments-differ
    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class TupleOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output
    """
    def __init__(self):
        super(TupleOutputModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[0])
        c3 = self.conv3(inputs[0])
        return c1, c2, c3


class MultiOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output
    """
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.layer = TupleOutputModel()
        self.conv1 = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 4, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x, y, z = self.layer(inputs[0])
        x1 = self.conv1(x)
        x2 = self.conv2(y)
        x3 = self.conv3(z)
        return torch.cat([x1, x2, x3], 1)


class ConfigurableTupleOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output with configurable channels
    """
    def __init__(self, channels=(2, 4, 6)):
        super(ConfigurableTupleOutputModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[1])
        c3 = self.conv3(inputs[2])
        return c1, c2, c3


class SingleLayerRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(SingleLayerRNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.rnn(x, hx)


class SingleLayerBidirectionalLstmModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(SingleLayerBidirectionalLstmModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=1, bidirectional=True)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.lstm(x, hx)


class TwoLayerBidirectionalLSTMModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionalLSTMModel, self).__init__()
        self.recurrent = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionaRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionaRNNModel, self).__init__()
        self.recurrent = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionalGRUModel(nn.Module):
    """
    Model using torch.nn.GRU module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionalGRUModel, self).__init__()
        self.recurrent = torch.nn.GRU(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class MultiLayerRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module with multiple layers
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(MultiLayerRNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=2)

    # pylint: disable=arguments-differ
    def forward(self, x, hx=None):
        return self.rnn(x, hx)


class RNNCellModel(nn.Module):
    """
    Model using torch.nn.RNNCell module
    Expected input shape = (SEQ_LENGTH=10, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(RNNCellModel, self).__init__()
        self.rnn_cell = torch.nn.RNNCell(input_size=3, hidden_size=5)

    # pylint: disable=arguments-differ
    def forward(self, x, hx0=None):
        output = []
        for i in range(x.shape[0]):
            hx0 = self.rnn_cell(x[i], hx0)
            output.append(hx0)
        return tuple(output), hx0


class LSTMModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=1)

    # pylint: disable=arguments-differ
    def forward(self, x, hx_cx=None):
        return self.rnn(x, hx_cx)


class NestedSequentialModel(nn.Module):
    """
    Model using nested Sequential modules
    Expected input shape = (1, 3, 8, 8)
    """
    def __init__(self, num_classes=3):
        super(NestedSequentialModel, self).__init__()
        self.inner_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16)
        )
        self.seq_list = nn.Sequential(
            self.inner_seq,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithFunctionalReLU(nn.Module):
    """ Model that uses functional ReLU instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


class ModelWithDuplicateReLU(nn.Module):
    """ Model that uses single ReLU instances multiple times in the forward. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
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


class ModelWithTransposeConv(nn.Module):

    def __init__(self):
        super(ModelWithTransposeConv, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1280, 50)
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


class ConstantElementwiseInputModel(torch.nn.Module):
    def __init__(self):
        super(ConstantElementwiseInputModel, self).__init__()
        self.add = Add()
        self.mul = Multiply()

    def forward(self, inp):
        x = self.add(inp, torch.tensor(2.0))
        x = self.mul(torch.tensor(3.0), x)
        return x


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


class LinearAndLSTMModel(torch.nn.Module):
    def __init__(self):
        super(LinearAndLSTMModel, self).__init__()

        self.linear = torch.nn.Linear(10, 4)
        self.prelu = torch.nn.PReLU(init=.3)
        self.recurrent = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=2)

    def forward(self, x, h_and_c=None):
        x = self.linear(x)
        x = self.prelu(x)
        x = torch.unsqueeze(x, 1)
        return self.recurrent(x, h_and_c)



class RoiAlignPyTorch(torch.nn.Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoiAlignPyTorch, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_align(input=features,
                         boxes = rois,
                         output_size = [self.aligned_height, self.aligned_width],
                         spatial_scale = self.spatial_scale,
                         sampling_ratio = 0)

class RoiModel(torch.nn.Module):

    def __init__(self, height, width, scale):
        super(RoiModel, self).__init__()
        self.roi = RoiAlignPyTorch(height, width, scale)

    def forward(self, *inputs):
        return self.roi(*inputs)


# pylint: disable=no-member
def build_dummy_model():
    """BUild dummy ONNX model for testing"""
    op = OperatorSetIdProto()
    op.version = 13
    input_info = helper.make_tensor_value_info(name='input', elem_type=TensorProto.FLOAT,
                                               shape=[1, 3, 32, 32])

    output_info = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT,
                                                shape=[1, 10])
    conv_node = helper.make_node('Conv',
                                 ['input', 'conv_w', 'conv_b'],
                                 ['3'],
                                 'conv',
                                 kernel_shape=[3, 3],
                                 pads=[1, 1, 1, 1],)
    relu_node = helper.make_node('Relu',
                                 ['3'],
                                 ['4'],
                                 'relu')
    maxpool_node = helper.make_node('MaxPool',
                                    ['4'],
                                    ['5'],
                                    'maxpool',
                                    kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1],
                                    strides=[2, 2],)

    flatten_node = helper.make_node('Flatten',
                                    ['5'],
                                    ['6'],
                                    'flatten')
    fc_node = helper.make_node('Gemm',
                               ['6', 'fc_w', 'fc_b'],
                               ['output'],
                               'fc')

    conv_w_init = numpy_helper.from_array(np.random.rand(1, 3, 3, 3).astype(np.float32), 'conv_w')
    conv_b_init = numpy_helper.from_array(np.random.rand(1).astype(np.float32), 'conv_b')
    fc_w_init = numpy_helper.from_array(np.random.rand(256, 10).astype(np.float32), 'fc_w')
    fc_b_init = numpy_helper.from_array(np.random.rand(10).astype(np.float32), 'fc_b')

    onnx_graph = helper.make_graph([conv_node, relu_node, maxpool_node, flatten_node, fc_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [conv_w_init, conv_b_init, fc_w_init, fc_b_init])

    model = helper.make_model(onnx_graph, opset_imports=[op])

    return model


def build_lstm_gru_dummy_model():
    op = OperatorSetIdProto()
    op.version = 13

    input_info = helper.make_tensor_value_info(name='input', elem_type=TensorProto.FLOAT,
                                               shape=[1, 8, 64])
    output_info = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT,
                                                shape=[1, 1, 8, 16])

    lstm_node = helper.make_node('LSTM',
                                 ['input', 'lstm_w', 'lstm_r_w'],
                                 ['2'],
                                 'lstm',
                                 hidden_size=16)
    squeeze_node = helper.make_node('Squeeze',
                                    ['2', 'axis'],
                                    ['3'],
                                    'squeeze')
    gru_node = helper.make_node('GRU',
                                ['3', 'gru_w', 'gru_r_w'],
                                ['output'],
                                'gru',
                                hidden_size=16)

    lstm_w_init = numpy_helper.from_array(np.random.rand(1, 64, 64).astype(np.float32), 'lstm_w')
    lstm_r_w_init = numpy_helper.from_array(np.random.rand(1, 64, 16).astype(np.float32), 'lstm_r_w')
    squeeze_axis_init = numpy_helper.from_array(np.array([1]).astype(np.int64), 'axis')
    gru_w_init = numpy_helper.from_array(np.random.rand(1, 48, 16).astype(np.float32), 'gru_w')
    gru_r_w_init = numpy_helper.from_array(np.random.rand(1, 48, 16).astype(np.float32), 'gru_r_w')

    onnx_graph = helper.make_graph([lstm_node, squeeze_node, gru_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [lstm_w_init, lstm_r_w_init, squeeze_axis_init, gru_w_init, gru_r_w_init])

    model = helper.make_model(onnx_graph, opset_imports=[op])

    return model


def single_residual_model(training=torch.onnx.TrainingMode.EVAL):
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model = SingleResidualWithAvgPool()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_single_residual.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          training=training,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model_onnx = ONNXModel(load_model(save_path))
    return model_onnx

def multi_input_model():
    x = (torch.rand(32, 1, 28, 28, requires_grad=True), torch.rand(32, 1, 28, 28, requires_grad=True))
    model = ModelWithTwoInputs()
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_multi_input.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model

def multi_output_model():
    model = MultipleOutputModel()
    sample_input = np.random.rand(128, 3, 32, 32).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_filename = Path(tmp_dir, "dummy_model_multiple_outputs.onnx")
        input_names = ["input"]
        output_names = ["output_mul", "output_add"]
        torch.onnx.export(model, torch.as_tensor(sample_input), str(onnx_filename), verbose=True, input_names=input_names,
                        output_names=output_names)

        model = ONNXModel(load_model(onnx_filename))
    return model

def transposed_conv_model():
    with tempfile.TemporaryDirectory() as save_dir:
        x = torch.randn(10, 10, 4, 4, requires_grad=True)
        model = TransposedConvModel()
        save_path = os.path.join(save_dir, "model_transposed_conv.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def depthwise_transposed_conv_model():
    with tempfile.TemporaryDirectory() as save_dir:
        x = torch.randn(10, 10, 4, 4, requires_grad=True)
        model = DepthwiseTransposedConvModel()
        save_path = os.path.join(save_dir, "model_transposed_conv.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def transposed_conv_model_without_bn():
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = TransposedConvModelWithoutBN()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_transposed_conv_without_bn.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def linear_split_into_matmul_add():
    with tempfile.TemporaryDirectory() as save_dir:
        # 3D input will split the linear layer in MatMul + Add in ONNX graph.
        dummy_input = torch.randn(1, 2, 4)

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear(x)
                return x

        model = LinearModel().eval()
        save_path = os.path.join(save_dir, "matmul_add.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def depthwise_conv_model():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MockMobileNetV1()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_mock_mobilenet.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model

def depthwise_conv_model_with_relu6():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MockMobileNetV11()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_mock_mobilenet.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model

def concat_model():
    x = (torch.rand(1, 3, 8, 8, requires_grad=True), torch.rand(1, 3, 8, 8, requires_grad=True),
         torch.rand(1, 3, 8, 8, requires_grad=True))
    model = ConcatModel()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "concat_model.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model

def hierarchical_model():
    conv_shape = (1, 64, 32, 32)
    inp_shape = (1, 3, 32, 32)
    seq_shape = (1, 3, 8, 8)

    x = (torch.rand(1, 64, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32, requires_grad=True),
         torch.rand(1, 64, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32, requires_grad=True),
         torch.rand(1, 3, 8, 8, requires_grad=True))
    model = HierarchicalModel()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "hierarchical_model.onnx")
        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model

class BNBeforeConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNBeforeConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=bias, padding=padding, stride=stride,
                                     dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        return x


class BNAfterConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=bias, padding=padding, stride=stride,
                                     dilation=dilation, groups=groups)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        return x


class BNAfterLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNAfterLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        return x


class BNBeforeLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNBeforeLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.fc2 = torch.nn.Linear(20, 20, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)

        return x


class BNBeforeFlattenLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNBeforeFlattenLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, padding=1, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.flatten = torch.nn.Flatten()
        self.fc2 = torch.nn.Linear(20 * 12 * 12, 20, bias=bias)
        self.bn2 = torch.nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class BNAfterConvTranspose(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, output_padding=0):
        super(BNAfterConvTranspose, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class BNAfterConvTranspose1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, output_padding=0):
        super(BNAfterConvTranspose1d, self).__init__()
        self.conv1 = torch.nn.ConvTranspose1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class BNBeforeConvTranspose(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, output_padding=0):
        super(BNBeforeConvTranspose, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 10, 3, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        return x


class BNAfterConv1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterConv1d, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation, groups=groups,
                                     bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return x

class BNBeforeConv1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):

        super(BNBeforeConv1d, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1d(x)

        return x

class BNAfterDynamicMatMul(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterDynamicMatMul, self).__init__()
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups,
                                      bias=bias)
        self.fc1 = torch.nn.Linear(24*10, 20)
        self.flatten = torch.nn.Flatten()
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups,
                                      bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        x1 = self.conv1d(x)
        x2 = self.fc1(self.flatten(x)).unsqueeze(1)
        x = torch.matmul(x2, x1)
        x = self.bn1(x)
        return x

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(20, 15, 3)
        self.relu2 = torch.nn.ReLU()

        self.bn2 = torch.nn.BatchNorm2d(15)
        self.conv3 = torch.nn.Conv2d(15, 20, 3)

        self.conv4 = torch.nn.Conv2d(20, 20, 3)
        self.bn3 = torch.nn.BatchNorm2d(20)
        self.bn4 = torch.nn.BatchNorm2d(20)

        self.fc1 = torch.nn.Linear(5120, 10)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.bn2(x)
        x = self.conv3(x)

        # No fold if there is a split between conv and BN
        x = self.conv4(x)
        bn1_out = self.bn3(x)
        bn2_out = self.bn4(x)

        x = bn1_out + bn2_out

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class InstanceNormModel(torch.nn.Module):
    def __init__(self):
        super(InstanceNormModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.in1 = torch.nn.InstanceNorm2d(20)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)

        return x


class MyModelFoldFoward(torch.nn.Module):
    def __init__(self):
        super(MyModelFoldFoward, self).__init__()

        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu1 = torch.nn.ReLU6()

        self.conv2 = torch.nn.Conv2d(20, 15, 3)
        self.bn2 = torch.nn.BatchNorm2d(15)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(15, 20, 3)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.conv3(x)

        return x

class MultipleOutputModel(SingleResidual):
    """
    Model
    """
    def __init__(self):
        super().__init__()
        # change padding size to 0, onnxruntime only support input size is the factor of output size for pooling
        self.conv4 = torch.nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=0, bias=True)
        self.fc2 = torch.nn.Linear(10, 3)
        # remove bn layer for currently not supporting non-4 dim param tensors
        del self.bn1
        del self.bn2

    def forward(self, inputs):
        x = self.conv1(inputs)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        y = self.fc2(x)

        return x, y


def _convert_to_onnx_no_fold(model: torch.nn.Module, dummy_input, filename='temp_model.onnx'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, filename)
        torch.onnx.export(model.eval(),
                          dummy_input,
                          save_path,
                          training=torch.onnx.TrainingMode.PRESERVE,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=False,
                          input_names=['input'],
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def _convert_to_onnx(model: torch.nn.Module, dummy_input, filename='temp_model.onnx'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, filename)
        torch.onnx.export(model.eval(),
                          dummy_input,
                          save_path,
                          training=torch.onnx.TrainingMode.EVAL,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'])
        model = ONNXModel(load_model(save_path))
    return model


def my_model_with_bns():
    torch.manual_seed(10)
    model = MyModelFoldFoward().eval()
    initialize_bn_params(model)

    input_shape = (2, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_single_residual.onnx",
                      # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.TRAINING,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model_onnx = ONNXModel(load_model('./model_single_residual.onnx'))
    return model_onnx


def initialize_bn_params(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
                module.bias.copy_(torch.randn_like(module.bias))
                module.running_mean.copy_(torch.randn_like(module.bias))
                module.running_var.add_(torch.randn_like(module.bias).abs())

def elementwise_op_model():
    torch.manual_seed(10)
    model = ConstantElementwiseInputModel().eval()

    input_shape = (1, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_elementwise.onnx",
                      # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.EVAL,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model_onnx = ONNXModel(load_model('./model_elementwise.onnx'))
    return model_onnx

class MultiInputWithConstant(torch.nn.Module):
    """ A model with multiple inputs.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=3):
        super(MultiInputWithConstant, self).__init__()
        self.add0 = Add()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.add1 = Add()
        self.add2 = Add()

    def forward(self, *inputs):
        x1 = self.add0(inputs[0], torch.tensor(0.02))
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[1])
        x = self.add1(x1, x2)
        x = self.add2(x, torch.tensor(2.0))
        return x

def multi_input_with_constant_model():
    torch.manual_seed(10)
    model = MultiInputWithConstant().eval()

    x = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_with_constant.onnx",
                      # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.EVAL,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model_onnx = ONNXModel(load_model('./model_with_constant.onnx'))
    return model_onnx


# pylint: disable=no-member
def build_dummy_model_with_dynamic_input():
    """ Build dummy ONNX model for testing. The batch-size dimension of the input is dynamic. """
    op = OperatorSetIdProto()
    op.version = 13
    input_info = helper.make_tensor_value_info(name='input', elem_type=TensorProto.FLOAT,
                                               shape=['batch_size', 3, 32, 32])

    output_info = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT,
                                                shape=['batch_size', 10])
    conv_node = helper.make_node('Conv',
                                 ['input', 'conv_w', 'conv_b'],
                                 ['conv/output.3'],
                                 'conv',
                                 kernel_shape=[3, 3],
                                 pads=[1, 1, 1, 1],)
    relu_node = helper.make_node('Relu',
                                 ['conv/output.3'],
                                 ['relu/output.4'],
                                 'relu')
    maxpool_node = helper.make_node('MaxPool',
                                    ['relu/output.4'],
                                    ['maxpool/output.5'],
                                    'maxpool',
                                    kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1],
                                    strides=[2, 2],)

    flatten_node = helper.make_node('Flatten',
                                    ['maxpool/output.5'],
                                    ['flatten/output.6'],
                                    'flatten')
    fc_node = helper.make_node('Gemm',
                               ['flatten/output.6', 'fc_w', 'fc_b'],
                               ['output'],
                               'fc')

    conv_w_init = numpy_helper.from_array(np.random.rand(1, 3, 3, 3).astype(np.float32), 'conv_w')
    conv_b_init = numpy_helper.from_array(np.random.rand(1).astype(np.float32), 'conv_b')
    fc_w_init = numpy_helper.from_array(np.random.rand(256, 10).astype(np.float32), 'fc_w')
    fc_b_init = numpy_helper.from_array(np.random.rand(10).astype(np.float32), 'fc_b')

    onnx_graph = helper.make_graph([conv_node, relu_node, maxpool_node, flatten_node, fc_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [conv_w_init, conv_b_init, fc_w_init, fc_b_init])

    model = helper.make_model(onnx_graph, opset_imports=[op])

    return model


def simple_relu_model():
    class ReLUModel(torch.nn.Module):
        def __init__(self):
            super(ReLUModel, self).__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(x)
            return x

    torch.manual_seed(10)
    model = ReLUModel().eval()

    input_shape = (1, 3, 32, 32)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./simple_relu.onnx",
                      # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.TRAINING,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model_onnx = ONNXModel(load_model('./simple_relu.onnx'))
    return model_onnx

def instance_norm_model():
    model = InstanceNormModel().eval()
    for module in model.modules():
        if isinstance(module, _InstanceNorm) and not module.affine:
            with torch.no_grad():
                module.weight = torch.nn.Parameter(torch.randn(20))
                module.bias = torch.nn.Parameter(torch.randn(20))

    input_shape = (2, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_instance_norm.onnx",
                      # where to save the model (can be a file or file-like object),
                      training=torch.onnx.TrainingMode.TRAINING,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_instance_norm.onnx'))
    return model

def custom_add_model():
    class CustomAddModel(nn.Module):
        """Simple model using custom addition op"""

        def __init__(self):
            super(CustomAddModel, self).__init__()

            custom_ops_path = os.path.dirname(libquant_info.__file__)
            custom_ops_path = os.path.join(custom_ops_path, "customops")
            torch_library = os.path.join(custom_ops_path, "libtorch_custom_add.so")

            torch.ops.load_library(torch_library)

            def my_add(g, x, y):
                return g.op("my_ops::custom_add", x, y)

            torch.onnx.register_custom_op_symbolic("my_ops::custom_add", my_add, 9)

            self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x_conv = self.conv(x)
            x_add1 = torch.ops.my_ops.custom_add(x_conv, x_conv)
            return x_add1

    model = CustomAddModel()
    torch.onnx.export(model,
                      torch.randn(1, 3, 64, 64),
                      './simple_custom_model.onnx',
                      verbose=True,
                      input_names=["input"],
                      output_names=["output_add"],
                      custom_opsets={"my_ops": 2})
    model_onnx = ONNXModel(load_model('./simple_custom_model.onnx'))
    return model_onnx


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


def const_param_model():
    """ ONNX model having constant tensors as op parameters """

    model = helper.make_model(
        graph=helper.make_graph(
            name='ConstantParamModel',
            inputs=[helper.make_tensor_value_info('latent', TensorProto.FLOAT, shape=[1, 4, 64, 64])],
            outputs=[helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0', TensorProto.FLOAT, shape=[1, 32, 40960])],
            initializer=[
                numpy_helper.from_array(np.random.randn(320, 4, 3, 3).astype('float32'), name='conv_in.weight'),
                numpy_helper.from_array(np.random.randn(320).astype('float32'), name='conv_in.bias'),
            ],
            value_info=[
                helper.make_tensor_value_info('/conv_in/Conv_output_0', TensorProto.FLOAT, shape=[1, 320, 64, 64]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Constant_output_0', TensorProto.INT64, shape=[3]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Reshape_output_0', TensorProto.FLOAT, shape=[1, 32, 40960]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Constant_1_output_0', TensorProto.FLOAT, shape=[32]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Constant_2_output_0', TensorProto.FLOAT, shape=[32])
            ],
            nodes=[
                helper.make_node(
                    'Conv',
                    inputs=['latent', 'conv_in.weight', 'conv_in.bias'],
                    outputs=['/conv_in/Conv_output_0'],
                    name='/conv_in/Conv',
                    doc_string='',
                    dilations=[1, 1],
                    group=1,
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['/down_blocks.0/resnets.0/norm1/Constant_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Constant',
                    doc_string='',
                    value=numpy_helper.from_array(np.array([0, 32, -1], dtype='int64'), name=''),
                ),
                helper.make_node(
                    'Reshape',
                    inputs=['/conv_in/Conv_output_0', '/down_blocks.0/resnets.0/norm1/Constant_output_0'],
                    outputs=['/down_blocks.0/resnets.0/norm1/Reshape_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Reshape',
                    doc_string='',
                    allowzero=0,
                ),
                helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['/down_blocks.0/resnets.0/norm1/Constant_1_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Constant_1',
                    doc_string='',
                    value=numpy_helper.from_array(np.random.randn(32).astype('float32'), name=''),
                ),
                helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['/down_blocks.0/resnets.0/norm1/Constant_2_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Constant_2',
                    doc_string='',
                    value=numpy_helper.from_array(np.random.randn(32).astype('float32'), name=''),
                ),
                helper.make_node(
                    'InstanceNormalization',
                    inputs=['/down_blocks.0/resnets.0/norm1/Reshape_output_0', '/down_blocks.0/resnets.0/norm1/Constant_1_output_0', '/down_blocks.0/resnets.0/norm1/Constant_2_output_0'],
                    outputs=['/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/InstanceNormalization',
                    doc_string='',
                    epsilon=9.999999747378752e-06,
                ),
            ],
        ),
    )

    return model

def weight_matmul_model(in_features=10, out_features=20):
    seq_len = 10
    matmul_layer = helper.make_node("MatMul", inputs=["input", "weight"], name="matmul", outputs=["output"])
    weight = numpy_helper.from_array(np.empty((in_features, out_features), dtype=np.float32), name="weight")
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, seq_len, in_features])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, seq_len, out_features])
    graph = helper.make_graph([matmul_layer], "matmul_graph", initializer=[weight], inputs=[input_tensor], outputs=[output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model

def weight_gemm_model(in_features, out_features, transposed_weight=False):
    matmul_layer = helper.make_node("Gemm", inputs=["input", "weight"], name="matmul", outputs=["output"],
                                    transB=transposed_weight)
    weight_shape = (in_features, out_features) if not transposed_weight else (out_features, in_features)
    weight = numpy_helper.from_array(np.empty(weight_shape, dtype=np.float32), name="weight")
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, in_features])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, out_features])
    graph = helper.make_graph([matmul_layer], "matmul_graph", initializer=[weight], inputs=[input_tensor], outputs=[output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model

def conv_model(weight_shape, input_shape, output_shape, transpose=False, **kwargs):
    layer_type = "Conv" if not transpose else "ConvTranspose"
    conv_layer = helper.make_node(layer_type, inputs=["input", "weight"], name="conv", outputs=["output"],
                                  **kwargs)
    weight = numpy_helper.from_array(np.empty(weight_shape, dtype=np.float32), name="weight")
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    graph = helper.make_graph([conv_layer], "conv_graph", initializer=[weight], inputs=[input_tensor], outputs=[output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model

def pointwise_conv1d(input_shape):
    return conv_model(weight_shape=(input_shape[1], input_shape[1], 1),
                      input_shape=input_shape,
                      output_shape=input_shape,
                      auto_pad="SAME_UPPER")

def pointwise_conv3d(input_shape):
    return conv_model(weight_shape=(input_shape[1], input_shape[1], 1, 1, 1),
                      input_shape=input_shape,
                      output_shape=input_shape,
                      auto_pad="SAME_UPPER")

def pointwise_convtranspose1d(input_shape):
    return conv_model(weight_shape=(input_shape[1], input_shape[1], 1),
                      input_shape=input_shape,
                      output_shape=input_shape,
                      transpose=True,
                      auto_pad="SAME_UPPER")

def pointwise_convtranspose3d(input_shape):
    return conv_model(weight_shape=(input_shape[1], input_shape[1], 1, 1, 1),
                      input_shape=input_shape,
                      output_shape=input_shape,
                      transpose=True,
                      auto_pad="SAME_UPPER")


def dynamic_matmul_model(batch_size):
    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            # Add some nonsense operations that will get folded in onnx simplifier
            y = self.linear(x)
            return torch.nn.functional.linear(x, y)

    model = Model()

    with tempfile.TemporaryDirectory() as tempdir:
        fname = os.path.join(tempdir, "model.onnx")
        torch.onnx.export(model, torch.randn(batch_size, 10), fname, input_names=["input"], output_names=["output"],
                          do_constant_folding=False)
        onnx_model = onnx.load(fname)

    return onnx_model


def simplifiable_model(batch_size=1):

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.empty(10, 10))
            self.weight2 = torch.nn.Parameter(torch.empty(20, 10))

        def forward(self, x):
            # Add some nonsense operations that will get folded in onnx simplifier
            weight3 = torch.nn.functional.linear(self.weight2, self.weight1)
            return torch.nn.functional.linear(x, weight3)

    model = Model()

    with tempfile.TemporaryDirectory() as tempdir:
        fname = os.path.join(tempdir, "model.onnx")
        torch.onnx.export(model, torch.randn(batch_size, 10), fname, input_names=["input"], output_names=["output"],
                          do_constant_folding=False)
        onnx_model = onnx.load(fname)

    return onnx_model


def layernorm_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='LayerNormModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 4, 64, 64])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 32, 40960])],
            initializer=[
                numpy_helper.from_array(np.random.randn(320, 4, 3, 3).astype('float32'), name='conv_in.weight'),
                numpy_helper.from_array(np.random.randn(320).astype('float32'), name='conv_in.bias'),
                numpy_helper.from_array(np.random.randn(32).astype('float32'), name='layernorm.scale'),
                numpy_helper.from_array(np.random.randn(32).astype('float32'), name='layernorm.bias'),
            ],
            value_info=[
                helper.make_tensor_value_info('/conv_in/Conv_output_0', TensorProto.FLOAT, shape=[1, 320, 64, 64]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Constant_output_0', TensorProto.INT64, shape=[3]),
                helper.make_tensor_value_info('/down_blocks.0/resnets.0/norm1/Reshape_output_0', TensorProto.FLOAT, shape=[1, 32, 40960])
            ],
            nodes=[
                helper.make_node(
                    'Conv',
                    inputs=['model_input', 'conv_in.weight', 'conv_in.bias'],
                    outputs=['/conv_in/Conv_output_0'],
                    name='/conv_in/Conv',
                    doc_string='',
                    dilations=[1, 1],
                    group=1,
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['/down_blocks.0/resnets.0/norm1/Constant_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Constant',
                    doc_string='',
                    value=numpy_helper.from_array(np.array([0, 32, -1], dtype='int64'), name=''),
                ),
                helper.make_node(
                    'Reshape',
                    inputs=['/conv_in/Conv_output_0', '/down_blocks.0/resnets.0/norm1/Constant_output_0'],
                    outputs=['/down_blocks.0/resnets.0/norm1/Reshape_output_0'],
                    name='/down_blocks.0/resnets.0/norm1/Reshape',
                    doc_string='',
                    allowzero=0,
                ),
                helper.make_node(
                    'LayerNormalization',
                    inputs=['/down_blocks.0/resnets.0/norm1/Reshape_output_0', 'layernorm.scale', 'layernorm.bias'],
                    outputs=['model_output'],
                    name='/down_blocks.0/resnets.0/norm1/LayerNormalization',
                    doc_string='',
                    epsilon=9.999999747378752e-06,
                ),
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def model_with_exceptional_ops():
    opset = OperatorSetIdProto()
    opset.version = 18
    model = helper.make_model(
        graph=helper.make_graph(
            name='MatMul_GroupNorm_Model',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 12, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 12, 6, 6])],
            initializer=[
                numpy_helper.from_array(np.random.randn(12, 12, 3, 3).astype('float32'), name='conv_0.weight'),
                numpy_helper.from_array(np.random.randn(12).astype('float32'), name='conv_0.bias'),
                numpy_helper.from_array(np.random.randn(1, 12, 6, 6).astype('float32'), name='matmul_0.weight'),
                numpy_helper.from_array(np.random.randn(2).astype('float32'), name='groupnorm_0.scale'),
                numpy_helper.from_array(np.random.randn(2).astype('float32'), name='groupnorm_0.bias'),
            ],
            value_info=[
                helper.make_tensor_value_info('/conv_0/output_0', TensorProto.FLOAT, shape=[1, 12, 6, 6]),
                helper.make_tensor_value_info('/matmul_0/output_0', TensorProto.FLOAT, shape=[1, 12, 6, 6]),
                helper.make_tensor_value_info('/groupnorm_0/output_0', TensorProto.FLOAT, shape=[1, 12, 6, 6])
            ],
            nodes=[
                helper.make_node(
                    'Conv',
                    inputs=['model_input', 'conv_0.weight', 'conv_0.bias'],
                    outputs=['/conv_0/output_0'],
                    name='conv_0',
                    kernel_shape=[3, 3],
                ),
                helper.make_node(
                    'MatMul',
                    inputs=['/conv_0/output_0', 'matmul_0.weight'],
                    outputs=['/matmul_0/output_0'],
                    name='matmul_0',
                ),
                helper.make_node(
                    'GroupNormalization',
                    inputs=['/matmul_0/output_0', 'groupnorm_0.scale', 'groupnorm_0.bias'],
                    outputs=['/groupnorm_0/output_0'],
                    name='groupnorm_0',
                    num_groups=2,
                ),
                helper.make_node(
                    'Relu',
                    inputs=['/groupnorm_0/output_0'],
                    outputs=['model_output'],
                    name='relu_0',
                )
            ],
        ),
        opset_imports=[opset],
    )
    onnx.checker.check_model(model)
    return model


def resize_op_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='ResizeModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 3, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 1, 12, 12])],
            initializer=[
                numpy_helper.from_array(np.random.randn(1, 3, 3, 3).astype('float32'), name='conv_weight')
            ],
            value_info=[
                helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, shape=[1, 1, 6, 6]),
                helper.make_tensor_value_info('const_scale', TensorProto.FLOAT, shape=[4])
            ],
            nodes=[
                helper.make_node(
                    'Conv',
                    inputs=['model_input', 'conv_weight'],
                    outputs=['conv_output'],
                    kernel_shape=[3, 3],
                    name='conv'
                ),
                helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['const_scale'],
                    value=numpy_helper.from_array(np.array([1, 1, 2, 2]).astype('float32'), name='')
                ),
                helper.make_node(
                    'Resize',
                    inputs=['conv_output', '', 'const_scale'],
                    outputs=['model_output'],
                    name='resize'
                )
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def add_matmul_model():
    # Define the fixed shape input tensor
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [3, 3])
    input_tensor_2 = helper.make_tensor_value_info('input_2', onnx.TensorProto.FLOAT, [3, 3])

    # Define the constant 1D vector to be added to the input
    constant_vector = np.random.randn(3).astype(np.float32)
    constant_tensor = helper.make_tensor('constant_vector', onnx.TensorProto.FLOAT, [3], constant_vector)

    # Define the weights for both matmuls
    weight1 = helper.make_tensor('weight1', onnx.TensorProto.FLOAT, [3, 3], np.random.randn(3, 3).astype(np.float32))

    # Define the Add node
    add_node = helper.make_node(
        'Add',  # The operation is an addition
        ['input', 'constant_vector'],  # The inputs: input and constant vector
        ['added_output'],  # The output name
        name='AddNode'
    )

    # Define the MatMul nodes
    matmul1 = helper.make_node(
        'MatMul',  # The operation is a matrix multiplication
        ['added_output', 'weight1'],  # The inputs: added output and weight1
        ['output1'],  # The output name
        name='MatMul1'
    )

    matmul2 = helper.make_node(
        'MatMul',  # The operation is a matrix multiplication
        ['input_2', 'added_output'],  # The inputs: added output and weight2
        ['output2'],  # The output name
        name='MatMul2'
    )

    # Define the output tensors
    output_tensor1 = helper.make_tensor_value_info('output1', onnx.TensorProto.FLOAT, [3, 3])
    output_tensor2 = helper.make_tensor_value_info('output2', onnx.TensorProto.FLOAT, [3, 3])

    # Create the graph
    graph_def = helper.make_graph(
        [add_node, matmul1, matmul2],  # The list of nodes in the graph
        'add_matmul_graph',  # A name for the graph
        [input_tensor, input_tensor_2],  # The list of inputs
        [output_tensor1, output_tensor2],  # The list of outputs
        [constant_tensor, weight1]  # Initializers (weights and constants)
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='add_matmul')
    return model_def


def model_with_initializers_as_activations():
    model = helper.make_model(
        graph=helper.make_graph(
            name='IntializersAsActivationsModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 3, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 3, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.random.randn(1,).astype('float32'), name='mul_input2'),
                numpy_helper.from_array(np.random.randn(1,).astype('float32'), name='add_input2')
            ],
            value_info=[
                helper.make_tensor_value_info('mul_output', TensorProto.FLOAT, shape=[1, 3, 8, 8]),
            ],
            nodes=[
                helper.make_node(
                    'Mul',
                    inputs=['model_input', 'mul_input2'],
                    outputs=['mul_output'],
                    name='mul_0'
                ),
                helper.make_node(
                    'Add',
                    inputs=['mul_output', 'add_input2'],
                    outputs=['model_output'],
                    name='add_0'
                )
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def gather_op_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='GatherModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.INT64, shape=[1, 4])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 4, 3])],
            initializer=[
                numpy_helper.from_array(np.random.randn(10, 3).astype('float32'), name='gather_weight')
            ],
            nodes=[
                helper.make_node(
                    'Gather',
                    inputs=['gather_weight', 'model_input'],
                    outputs=['model_output'],
                    name='gather'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def gather_op_with_int_data_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='GatherModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.INT64, shape=[1, 4])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.INT64, shape=[1, 4])],
            initializer=[
                numpy_helper.from_array(np.array([1, 3, 224, 224]), name='gather_weight')
            ],
            nodes=[
                helper.make_node(
                    'Gather',
                    inputs=['gather_weight', 'model_input'],
                    outputs=['model_output'],
                    name='gather'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def matmul_add_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='MatMulAddModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 1, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 1, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.full((1, 1, 8, 8), 25.0).astype('float32'), name='matmul_1.weight'),
                numpy_helper.from_array(np.full((1, 1, 8, 8), 100.0).astype('float32'), name='add_1.weight'),
                numpy_helper.from_array(np.full((1, 1, 8, 8), -50.0).astype('float32'), name='add_2.weight'),
            ],
            nodes=[
                helper.make_node(
                    'MatMul',
                    inputs=['model_input', 'matmul_1.weight'],
                    outputs=['matmul_1.output'],
                    name='matmul_1'
                ),
                helper.make_node(
                    'Add',
                    inputs=['matmul_1.output', 'add_1.weight'],
                    outputs=['add_1.output'],
                    name='add_1'
                ),
                helper.make_node(
                    'MatMul',
                    inputs=['model_input', 'add_1.output'],
                    outputs=['matmul_2.output'],
                    name='matmul_2'
                ),
                helper.make_node(
                    'Add',
                    inputs=['matmul_2.output', 'add_2.weight'],
                    outputs=['model_output'],
                    name='add_2'
                )
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model

def softmax_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='SoftmaxModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[1, 1, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[1, 1, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.full((1, 1, 8, 8), 25.0).astype('float32'), name='matmul_1.weight'),
            ],
            nodes=[
                helper.make_node(
                    'MatMul',
                    inputs=['model_input', 'matmul_1.weight'],
                    outputs=['matmul.output'],
                    name='matmul'
                ),
                helper.make_node(
                    'Softmax',
                    inputs=['matmul.output'],
                    outputs=['softmax.output'],
                    name='softmax'
                ),
                helper.make_node(
                    'Sigmoid',
                    inputs=['softmax.output'],
                    outputs=['model_output'],
                    name='sigmoid'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def batchnorm_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='BatchnormModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32'), name='batchnorm.weight'),
                numpy_helper.from_array(np.random.randn(10, ).astype('float32'), name='batchnorm.bias'),
                numpy_helper.from_array(np.random.randn(10, ).astype('float32'), name='batchnorm.input_mean'),
                numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32'), name='batchnorm.input_var')
            ],
            nodes=[
                helper.make_node(
                    'BatchNormalization',
                    inputs=['model_input', 'batchnorm.weight', 'batchnorm.bias', 'batchnorm.input_mean', 'batchnorm.input_var'],
                    outputs=['model_output'],
                    name='batchnorm'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model

def batchnorm_model_constants():
    model = helper.make_model(
        graph=helper.make_graph(
            name='BatchnormModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            initializer=[],
            nodes=[
                helper.make_node('Constant', inputs=[], outputs=["batchnorm.weight"],
                                 value=numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32')), name='weight'),
                helper.make_node('Constant', inputs=[], outputs=["batchnorm.bias"],
                                 value=numpy_helper.from_array(np.random.randn(10, ).astype('float32')), name='bias'),
                helper.make_node('Constant', inputs=[], outputs=["batchnorm.input_mean"],
                                 value=numpy_helper.from_array(np.random.randn(10, ).astype('float32')), name="input_mean"),
                helper.make_node('Constant', inputs=[], outputs=["batchnorm.input_var"],
                                 value=numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32')), name='input_var'),
                helper.make_node(
                    'BatchNormalization',
                    inputs=['model_input', 'batchnorm.weight', 'batchnorm.bias', 'batchnorm.input_mean', 'batchnorm.input_var'],
                    outputs=['model_output'],
                    name='batchnorm'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model

def integer_concat_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='IntConcatModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[10, 10])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[10, 10, 1])],
            initializer=[],
            nodes=[
                helper.make_node('Constant', inputs=[], outputs=["constant_one"],
                                 value=numpy_helper.from_array(np.array([1]).astype('int64')), name='one'),
                helper.make_node(
                    'Shape',
                    inputs=['model_input'],
                    outputs=['input_shape'],
                    name='shape'
                ),
                helper.make_node(
                    'Concat',
                    inputs=['input_shape', 'constant_one'],
                    outputs=['output_shape'],
                    name='out_shape',
                    axis=0
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["model_input", "output_shape"],
                    outputs=["model_output"]
                )
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model


def shared_stat_batchnorm_model():
    model = helper.make_model(
        graph=helper.make_graph(
            name='BatchnormModel',
            inputs=[helper.make_tensor_value_info('model_input', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            outputs=[helper.make_tensor_value_info('model_output', TensorProto.FLOAT, shape=[10, 10, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.random.randn(10, 10, 1, 1).astype('float32'), name='conv_1.weight'),
                numpy_helper.from_array(np.random.randn(10, 10, 1, 1).astype('float32'), name='conv_2.weight'),
                numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32'), name='batchnorm.weight'),
                numpy_helper.from_array(np.random.randn(10, ).astype('float32'), name='batchnorm.bias'),
                numpy_helper.from_array(np.random.randn(10, ).astype('float32'), name='batchnorm.input_mean'),
                numpy_helper.from_array(np.abs(np.random.randn(10, )).astype('float32'), name='batchnorm.input_var')
            ],
            nodes=[
                helper.make_node(
                    'Conv',
                    inputs=['model_input', 'conv_1.weight'],
                    outputs=['conv_1.output'],
                    name='conv_1'
                ),
                helper.make_node(
                    'Identity',
                    inputs=['batchnorm.input_mean'],
                    outputs=['batchnorm1.input_mean'],
                    name='identity_1'
                ),
                helper.make_node(
                    'BatchNormalization',
                    inputs=['conv_1.output', 'batchnorm.weight', 'batchnorm.bias', 'batchnorm1.input_mean', 'batchnorm.input_var'],
                    outputs=['batch_norm_1.output'],
                    name='batchnorm_1'
                ),
                helper.make_node(
                    'Conv',
                    inputs=['batch_norm_1.output', 'conv_2.weight'],
                    outputs=['conv_2.output'],
                    name='conv_2'
                ),
                helper.make_node(
                    'Identity',
                    inputs=['batchnorm.input_mean'],
                    outputs=['batchnorm2.input_mean'],
                    name='identity_2'
                ),
                helper.make_node(
                    'BatchNormalization',
                    inputs=['conv_2.output', 'batchnorm.weight', 'batchnorm.bias', 'batchnorm2.input_mean',
                            'batchnorm.input_var'],
                    outputs=['model_output'],
                    name='batchnorm_2'
                ),
            ]
        )
    )
    onnx.checker.check_model(model, True)
    return model
