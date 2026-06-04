# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Models for use in unit testing"""

import functools
import platform
import sys
from typing import Any
import os
import tempfile
from pathlib import Path

import io
import onnx
import onnx_ir
import onnxscript
import torch.nn.functional as F
from torch import nn as nn
from torchvision.ops import roi_align
from torchvision import models
import numpy as np
import torch
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.batchnorm import _BatchNorm
from onnx import helper, numpy_helper, OperatorSetIdProto, TensorProto, load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_onnx.common import libquant_info
from .mobilenet import MockMobileNetV1, MockMobileNetV11

_DEFAULT_OPSET_VERSION = 13


def make_model(
    graph: onnx.GraphProto,
    *,
    opset_imports=None,
    ir_version=10,
    **kwargs,
):
    if opset_imports is None:
        opset_imports = [helper.make_operatorsetid("", 20)]

    return helper.make_model(
        graph=graph, **kwargs, opset_imports=opset_imports, ir_version=ir_version
    )


class Add(torch.nn.Module):
    """Add module for a functional add"""

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
    """Multiply module for a functional multiply"""

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
    """A model with a single residual connection.
    Use this model for unit testing purposes."""

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
    """A model with a single residual connection.
    Use this model for unit testing purposes."""

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
    """A model with multiple inputs.
    Use this model for unit testing purposes."""

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
    """Model with dictionary as input."""

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
        x1 = self.conv1(inputs[0]["inp_1"])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[0]["inp_2"])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConcatModel(nn.Module):
    """A model with concat op.
    Use this model for unit testing purposes.
    Expected inputs: 3 inputs, all of size (1, 3, 8, 8)"""

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
    """A model with modules defined using ModuleLists.
    Use this model for unit testing purposes.
    Expected inputs: 3 inputs, all of size (1, 3, 8, 8)"""

    def __init__(self, num_classes=3):
        super(ModuleListModel, self).__init__()
        self.mod_list = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # use 4th
                nn.ReLU(inplace=True),  # use 3rd
                nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),  # use 5th
                nn.ReLU(),  # dummy unused op
                nn.Conv2d(
                    3, 16, kernel_size=2, stride=2, padding=2, bias=False
                ),  # use 1st
            ]
        )
        self.seq_list = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),  # use 6th
            nn.ReLU(),  # dummy unused op
            nn.BatchNorm2d(16),  # use 2nd
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
    """Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32)"""

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
    """Use this model for quantsim_config unit testing purposes. Expect input shape (1, 3, 32, 32)"""

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
    """Use this model for unit testing purposes."""

    def __init__(self):
        super(ModelWithDropouts, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout2d(p=0.6)
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
    """Model that reuses a relu module. Expects input of shape (1, 3, 32, 32)"""

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
    """Model that uses functional modules instead of nn.Modules. Expects input of shape (1, 3, 32, 32)"""

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
    """A model with modules defined using nn.Sequential.
    Use this model for unit testing purposes.
    Expected inputs: 3 inputs, all of size (1, 3, 8, 8)"""

    def __init__(self, num_classes=3):
        super(SequentialModel, self).__init__()
        self.seq_list = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    """A Simple Super Node Model used as building block in Hierarchical Model"""

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
    """Sequential Model contains sequences of BasicConv2d Model"""

    def __init__(self):
        super(MultiConv2dModel, self).__init__()
        self.seq_list = nn.Sequential(
            BasicConv2d(kernel_size=3),
            BasicConv2d(kernel_size=1),
            BasicConv2d(kernel_size=3),
        )

    def forward(self, *inputs):
        return self.seq_list(inputs[0])


class NestedModel(nn.Module):
    """Aggregation Model contains two instance of Tiny Model"""

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
    """Aggregation Model contains multi-level of PyTorch Module
    Expected 5 inputs with shapes  in the following order:
        (1, 64, 32, 32)
        (1,  3, 32, 32)
        (1, 64, 32, 32)
        (1,  3, 32, 32)
        (1,  3,  8,  8)"""

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
    """Model with PassThroughOp as last layer. Expect input shape (1, 3, 32, 32)"""

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

    def forward(self, x, hx=None):
        return self.rnn(x, hx)


class SingleLayerBidirectionalLstmModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """

    def __init__(self):
        super(SingleLayerBidirectionalLstmModel, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=3, hidden_size=5, num_layers=1, bidirectional=True
        )

    def forward(self, x, hx=None):
        return self.lstm(x, hx)


class TwoLayerBidirectionalLSTMModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """

    def __init__(self):
        super(TwoLayerBidirectionalLSTMModel, self).__init__()
        self.recurrent = torch.nn.LSTM(
            input_size=3, hidden_size=5, num_layers=2, bidirectional=True
        )

    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionaRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """

    def __init__(self):
        super(TwoLayerBidirectionaRNNModel, self).__init__()
        self.recurrent = torch.nn.RNN(
            input_size=3, hidden_size=5, num_layers=2, bidirectional=True
        )

    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionalGRUModel(nn.Module):
    """
    Model using torch.nn.GRU module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """

    def __init__(self):
        super(TwoLayerBidirectionalGRUModel, self).__init__()
        self.recurrent = torch.nn.GRU(
            input_size=3, hidden_size=5, num_layers=2, bidirectional=True
        )

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
            nn.BatchNorm2d(16),
        )
        self.seq_list = nn.Sequential(
            self.inner_seq,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithFunctionalReLU(nn.Module):
    """Model that uses functional ReLU instead of nn.Modules. Expects input of shape (1, 3, 32, 32)"""

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
    """Model that uses single ReLU instances multiple times in the forward. Expects input of shape (1, 3, 32, 32)"""

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
        self.prelu1 = torch.nn.PReLU(init=0.3)
        self.prelu2 = torch.nn.PReLU(init=0.4)
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
        self.prelu = torch.nn.PReLU(init=0.3)
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
        return roi_align(
            input=features,
            boxes=rois,
            output_size=[self.aligned_height, self.aligned_width],
            spatial_scale=self.spatial_scale,
            sampling_ratio=0,
        )


class RoiModel(torch.nn.Module):
    def __init__(self, height, width, scale):
        super(RoiModel, self).__init__()
        self.roi = RoiAlignPyTorch(height, width, scale)

    def forward(self, *inputs):
        return self.roi(*inputs)


class LongSequentialModel(nn.Module):
    """A model with many back to back Conv -> Relus. Used to test for GraphSearchUtils maximum recursion error.
    Expected inputs: 3 inputs, all of size (1, 3, 1, 1)"""

    def __init__(self):
        super(LongSequentialModel, self).__init__()
        self.seq_list = nn.Sequential()
        for _ in range(1000):
            self.seq_list.append(nn.Conv2d(3, 3, (1, 1)))
            self.seq_list.append(nn.ReLU())

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        return x


def build_dummy_model():
    """BUild dummy ONNX model for testing"""
    op = OperatorSetIdProto()
    op.version = 13
    input_info = helper.make_tensor_value_info(
        name="input", elem_type=TensorProto.FLOAT, shape=[1, 3, 32, 32]
    )

    output_info = helper.make_tensor_value_info(
        name="output", elem_type=TensorProto.FLOAT, shape=[1, 10]
    )
    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_w", "conv_b"],
        ["3"],
        "conv",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu_node = helper.make_node("Relu", ["3"], ["4"], "relu")
    maxpool_node = helper.make_node(
        "MaxPool",
        ["4"],
        ["5"],
        "maxpool",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )

    flatten_node = helper.make_node("Flatten", ["5"], ["6"], "flatten")
    fc_node = helper.make_node("Gemm", ["6", "fc_w", "fc_b"], ["output"], "fc")

    conv_w_init = numpy_helper.from_array(
        np.random.rand(1, 3, 3, 3).astype(np.float32), "conv_w"
    )
    conv_b_init = numpy_helper.from_array(
        np.random.rand(1).astype(np.float32), "conv_b"
    )
    fc_w_init = numpy_helper.from_array(
        np.random.rand(256, 10).astype(np.float32), "fc_w"
    )
    fc_b_init = numpy_helper.from_array(np.random.rand(10).astype(np.float32), "fc_b")

    onnx_graph = helper.make_graph(
        [conv_node, relu_node, maxpool_node, flatten_node, fc_node],
        "dummy_graph",
        [input_info],
        [output_info],
        [conv_w_init, conv_b_init, fc_w_init, fc_b_init],
    )

    model = make_model(onnx_graph, opset_imports=[op])

    return model


def long_sequential_model(
    training=torch.onnx.TrainingMode.EVAL, opset_version=_DEFAULT_OPSET_VERSION
):
    x = torch.randn(1, 3, 1, 1)
    model = LongSequentialModel()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=training,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=min(opset_version, 20),  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )

    buffer.seek(0)
    model = load_model(buffer)
    if opset_version > 20:
        model = onnx.version_converter.convert_version(model, opset_version)

    model_onnx = ONNXModel(model)
    return model_onnx


def single_residual_model(
    training=torch.onnx.TrainingMode.EVAL,
    opset_version=_DEFAULT_OPSET_VERSION,
    dtype=torch.float32,
):
    x = torch.randn(1, 3, 32, 32).to(dtype)
    model = SingleResidualWithAvgPool().to(dtype)

    buffer = io.BytesIO()

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=training,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=min(opset_version, 20),  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = load_model(buffer)
    if opset_version > 20:
        model = onnx.version_converter.convert_version(model, opset_version)

    model_onnx = ONNXModel(model)
    return model_onnx


def multi_input_model(opset_version=_DEFAULT_OPSET_VERSION):
    x = (
        torch.rand(32, 1, 28, 28, requires_grad=True),
        torch.rand(32, 1, 28, 28, requires_grad=True),
    )
    model = ModelWithTwoInputs()
    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input1", "input2"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def multi_output_model():
    model = MultipleOutputModel()
    sample_input = np.random.rand(128, 3, 32, 32).astype(np.float32)

    buffer = io.BytesIO()
    input_names = ["input"]
    output_names = ["output_mul", "output_add"]
    torch.onnx.export(
        model,
        torch.as_tensor(sample_input),
        buffer,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
    )

    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def transposed_conv_model(opset_version=_DEFAULT_OPSET_VERSION):
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = TransposedConvModel()
    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=min(opset_version, 20),  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )

    buffer.seek(0)
    model = load_model(buffer)
    if opset_version > 20:
        model = onnx.version_converter.convert_version(model, opset_version)

    model = ONNXModel(model)
    return model


def depthwise_transposed_conv_model(opset_version=_DEFAULT_OPSET_VERSION):
    buffer = io.BytesIO()
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = DepthwiseTransposedConvModel()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def transposed_conv_model_without_bn(opset_version=_DEFAULT_OPSET_VERSION):
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = TransposedConvModelWithoutBN()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def linear_split_into_matmul_add(opset_version=_DEFAULT_OPSET_VERSION):
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
    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def unfusable_matmul_add(opset_version=_DEFAULT_OPSET_VERSION):
    # 3D input will split the linear layer in MatMul + Add in ONNX graph.
    dummy_input = torch.randn(1, 2, 4)

    class MatmulAdd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4))
            self.not_a_bias = torch.nn.Parameter(torch.zeros(2, 4))

        def forward(self, x):
            return x @ self.weight + self.not_a_bias

    model = MatmulAdd().eval()
    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def depthwise_conv_model(opset_version=_DEFAULT_OPSET_VERSION):
    torch.manual_seed(10)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MockMobileNetV1()
    initialize_bn_params(model)

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def depthwise_conv_model_with_relu6(opset_version=_DEFAULT_OPSET_VERSION):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MockMobileNetV11()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def concat_model(opset_version=_DEFAULT_OPSET_VERSION):
    x = (
        torch.rand(1, 3, 8, 8, requires_grad=True),
        torch.rand(1, 3, 8, 8, requires_grad=True),
        torch.rand(1, 3, 8, 8, requires_grad=True),
    )
    model = ConcatModel()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input1", "input2", "input3"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def hierarchical_model(opset_version=_DEFAULT_OPSET_VERSION):
    conv_shape = (1, 64, 32, 32)
    inp_shape = (1, 3, 32, 32)
    seq_shape = (1, 3, 8, 8)

    x = (
        torch.rand(1, 64, 32, 32, requires_grad=True),
        torch.rand(1, 3, 32, 32, requires_grad=True),
        torch.rand(1, 64, 32, 32, requires_grad=True),
        torch.rand(1, 3, 32, 32, requires_grad=True),
        torch.rand(1, 3, 8, 8, requires_grad=True),
    )
    model = HierarchicalModel()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


class BNBeforeConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNBeforeConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(
            20,
            20,
            3,
            bias=bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

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
        self.conv2 = torch.nn.Conv2d(
            20,
            20,
            3,
            bias=bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        return x


class BNAfterConvWithSharedWeights(torch.nn.Module):
    def __init__(self, padding=1, stride=1, dilation=1, groups=1, bias=True):
        super(BNAfterConvWithSharedWeights, self).__init__()
        # Two conv layers that share the same weight and bias tensors
        self.conv1 = torch.nn.Conv2d(
            10,
            20,
            3,
            bias=bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.conv2 = torch.nn.Conv2d(
            10,
            20,
            3,
            bias=bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        # Share weights and bias between conv1 and conv2
        self.conv2.weight = self.conv1.weight
        if bias:
            self.conv2.bias = self.conv1.bias

        # Two separate BN layers downstream from each conv
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.bn2 = torch.nn.BatchNorm2d(20)

        self.conv0a = torch.nn.Conv2d(10, 10, 1, bias=bias)
        self.conv0b = torch.nn.Conv2d(10, 10, 1, bias=bias)
        self.relu0a = torch.nn.ReLU()
        self.relu0b = torch.nn.ReLU()

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv0a(x)
        x1 = self.relu0a(x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv0b(x)
        x2 = self.relu0b(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        return x1 + x2


class ParallelConvSharedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0a = torch.nn.Conv2d(10, 20, 1, bias=True)
        self.conv0b = torch.nn.Conv2d(10, 20, 1, bias=True)
        self.relu0a = torch.nn.ReLU()
        self.relu0b = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(20, 20, 3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(20, 20, 3, padding=1, bias=True)
        self.conv2.weight = self.conv1.weight
        self.conv2.bias = self.conv1.bias
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv0a(x)
        x1 = self.relu0a(x1)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv0b(x)
        x2 = self.relu0b(x2)
        x2 = self.conv2(x2)
        x2 = self.relu2(x2)

        return x1 + x2


class BNAfterLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNAfterLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=bias)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        return x1 + x2


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
        self.conv1 = torch.nn.ConvTranspose2d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )
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
        self.conv1 = torch.nn.ConvTranspose1d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose1d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

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
        self.conv2 = torch.nn.ConvTranspose2d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        return x


class BNAfterConv1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterConv1d, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(
            10,
            10,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

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
        self.conv1d = torch.nn.Conv1d(
            10,
            20,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1d(x)

        return x


class BNAfterDynamicMatMul(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterDynamicMatMul, self).__init__()
        self.conv1d = torch.nn.Conv1d(
            10,
            20,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.fc1 = torch.nn.Linear(24 * 10, 20)
        self.flatten = torch.nn.Flatten()
        self.conv1d = torch.nn.Conv1d(
            10,
            20,
            3,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
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
        self.conv4 = torch.nn.Conv2d(
            32, 8, kernel_size=2, stride=2, padding=0, bias=True
        )
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


def _convert_to_onnx_no_fold(
    model: torch.nn.Module,
    dummy_input,
    filename="temp_model.onnx",
    opset_version=_DEFAULT_OPSET_VERSION,
):
    buffer = io.BytesIO()
    torch.onnx.export(
        model.eval(),
        dummy_input,
        buffer,
        training=torch.onnx.TrainingMode.PRESERVE,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def _convert_to_onnx(
    model: torch.nn.Module,
    dummy_input,
    opset_version=_DEFAULT_OPSET_VERSION,
):
    buffer = io.BytesIO()
    torch.onnx.export(
        model.eval(),
        dummy_input,
        buffer,
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def my_model_with_bns(opset_version=_DEFAULT_OPSET_VERSION):
    torch.manual_seed(10)
    model = MyModelFoldFoward().eval()
    initialize_bn_params(model)

    input_shape = (2, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.EVAL,  # whether to execute constant folding for optimization
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model_onnx = ONNXModel(load_model(buffer))
    return model_onnx


def initialize_bn_params(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
                module.bias.copy_(torch.randn_like(module.bias))
                module.running_mean.copy_(torch.randn_like(module.bias))
                module.running_var.add_(torch.randn_like(module.bias).abs())


def elementwise_op_model(opset_version=_DEFAULT_OPSET_VERSION):
    torch.manual_seed(10)
    model = ConstantElementwiseInputModel().eval()

    input_shape = (1, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        # where to save the model (can be a file or file-like object),
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model_onnx = ONNXModel(load_model(buffer))
    return model_onnx


class MultiInputWithConstant(torch.nn.Module):
    """A model with multiple inputs.
    Use this model for unit testing purposes."""

    def __init__(self, num_classes=3):
        super(MultiInputWithConstant, self).__init__()
        self.add0 = Add()
        self.conv1 = torch.nn.Conv2d(
            3, 16, kernel_size=2, stride=2, padding=3, bias=False
        )
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


def multi_input_with_constant_model(opset_version=_DEFAULT_OPSET_VERSION):
    torch.manual_seed(10)
    model = MultiInputWithConstant().eval()

    buffer = io.BytesIO()
    x = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model_onnx = ONNXModel(load_model(buffer))
    return model_onnx


def build_dummy_model_with_dynamic_input():
    """Build dummy ONNX model for testing. The batch-size dimension of the input is dynamic."""
    op = OperatorSetIdProto()
    op.version = 13
    input_info = helper.make_tensor_value_info(
        name="input", elem_type=TensorProto.FLOAT, shape=["batch_size", 3, 32, 32]
    )

    output_info = helper.make_tensor_value_info(
        name="output", elem_type=TensorProto.FLOAT, shape=["batch_size", 10]
    )
    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_w", "conv_b"],
        ["conv/output.3"],
        "conv",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu_node = helper.make_node("Relu", ["conv/output.3"], ["relu/output.4"], "relu")
    maxpool_node = helper.make_node(
        "MaxPool",
        ["relu/output.4"],
        ["maxpool/output.5"],
        "maxpool",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )

    flatten_node = helper.make_node(
        "Flatten", ["maxpool/output.5"], ["flatten/output.6"], "flatten"
    )
    fc_node = helper.make_node(
        "Gemm", ["flatten/output.6", "fc_w", "fc_b"], ["output"], "fc"
    )

    conv_w_init = numpy_helper.from_array(
        np.random.rand(1, 3, 3, 3).astype(np.float32), "conv_w"
    )
    conv_b_init = numpy_helper.from_array(
        np.random.rand(1).astype(np.float32), "conv_b"
    )
    fc_w_init = numpy_helper.from_array(
        np.random.rand(256, 10).astype(np.float32), "fc_w"
    )
    fc_b_init = numpy_helper.from_array(np.random.rand(10).astype(np.float32), "fc_b")

    onnx_graph = helper.make_graph(
        [conv_node, relu_node, maxpool_node, flatten_node, fc_node],
        "dummy_graph",
        [input_info],
        [output_info],
        [conv_w_init, conv_b_init, fc_w_init, fc_b_init],
    )

    model = make_model(onnx_graph, opset_imports=[op])

    return model


def conv_model_with_shared_bias(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(10, 10, 3)
            self.conv2 = torch.nn.Conv2d(10, 10, 3)
            self.conv1.bias = self.conv2.bias

        def forward(self, x):
            return self.conv2(self.conv1(x))

    model_path = os.path.join(tmp_path, "model.onnx")
    torch.onnx.export(Model(), torch.randn(1, 10, 32, 32), model_path, dynamo=False)
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def simple_relu_model(opset_version=_DEFAULT_OPSET_VERSION):
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

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        training=torch.onnx.TrainingMode.TRAINING,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model_onnx = ONNXModel(load_model(buffer))
    return model_onnx


def instance_norm_model(opset_version=_DEFAULT_OPSET_VERSION):
    model = InstanceNormModel().eval()
    for module in model.modules():
        if isinstance(module, _InstanceNorm) and not module.affine:
            with torch.no_grad():
                module.weight = torch.nn.Parameter(torch.randn(20))
                module.bias = torch.nn.Parameter(torch.randn(20))

    input_shape = (2, 10, 24, 24)
    x = torch.randn(*input_shape, requires_grad=True)

    # Export the model
    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.TRAINING,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=min(opset_version, 20),  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = load_model(buffer)
    if opset_version > 20:
        model = onnx.version_converter.convert_version(model, opset_version)

    model = ONNXModel(model)
    return model


def custom_add_model():
    model = make_model(
        graph=helper.make_graph(
            name="CustomAddModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 3, 64, 64]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 3, 64, 64]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(3, 3, 3, 3).astype(np.float32), name="weight"
                ),
                numpy_helper.from_array(
                    np.random.randn(3).astype(np.float32), name="bias"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["input", "weight", "bias"],
                    outputs=["y"],
                    pads=[1, 1, 1, 1],
                    name="conv",
                ),
                helper.make_node(
                    "CustomAdd",
                    inputs=["input", "y"],
                    outputs=["output"],
                    domain="ai.onnx.contrib",
                    name="add",
                ),
            ],
        ),
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.contrib", 1),
            helper.make_operatorsetid("", 13),
        ],
    )
    onnx.checker.check_model(model)
    return ONNXModel(model)


def conv_relu_model(opset_version=_DEFAULT_OPSET_VERSION):
    class ConvReluModel(torch.nn.Module):
        def __init__(self):
            super(ConvReluModel, self).__init__()
            self._conv_0 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1
            )
            self._relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self._relu(self._conv_0(x))

    torch.manual_seed(10)
    model = ConvReluModel().eval()
    x = torch.randn((1, 3, 8, 8))

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object),
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    buffer.seek(0)
    model = load_model(buffer)
    return model


def const_param_model():
    """ONNX model having constant tensors as op parameters"""

    model = make_model(
        graph=helper.make_graph(
            name="ConstantParamModel",
            inputs=[
                helper.make_tensor_value_info(
                    "latent", TensorProto.FLOAT, shape=[1, 4, 64, 64]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0",
                    TensorProto.FLOAT,
                    shape=[1, 32, 40960],
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(320, 4, 3, 3).astype("float32"),
                    name="conv_in.weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(320).astype("float32"), name="conv_in.bias"
                ),
            ],
            value_info=[
                helper.make_tensor_value_info(
                    "/conv_in/Conv_output_0", TensorProto.FLOAT, shape=[1, 320, 64, 64]
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Constant_output_0",
                    TensorProto.INT64,
                    shape=[3],
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Reshape_output_0",
                    TensorProto.FLOAT,
                    shape=[1, 32, 40960],
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Constant_1_output_0",
                    TensorProto.FLOAT,
                    shape=[32],
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Constant_2_output_0",
                    TensorProto.FLOAT,
                    shape=[32],
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["latent", "conv_in.weight", "conv_in.bias"],
                    outputs=["/conv_in/Conv_output_0"],
                    name="/conv_in/Conv",
                    doc_string="",
                    dilations=[1, 1],
                    group=1,
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/down_blocks.0/resnets.0/norm1/Constant_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Constant",
                    doc_string="",
                    value=numpy_helper.from_array(
                        np.array([0, 32, -1], dtype="int64"), name=""
                    ),
                ),
                helper.make_node(
                    "Reshape",
                    inputs=[
                        "/conv_in/Conv_output_0",
                        "/down_blocks.0/resnets.0/norm1/Constant_output_0",
                    ],
                    outputs=["/down_blocks.0/resnets.0/norm1/Reshape_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Reshape",
                    doc_string="",
                    allowzero=0,
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/down_blocks.0/resnets.0/norm1/Constant_1_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Constant_1",
                    doc_string="",
                    value=numpy_helper.from_array(
                        np.random.randn(32).astype("float32"), name=""
                    ),
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/down_blocks.0/resnets.0/norm1/Constant_2_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Constant_2",
                    doc_string="",
                    value=numpy_helper.from_array(
                        np.random.randn(32).astype("float32"), name=""
                    ),
                ),
                helper.make_node(
                    "InstanceNormalization",
                    inputs=[
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0",
                        "/down_blocks.0/resnets.0/norm1/Constant_1_output_0",
                        "/down_blocks.0/resnets.0/norm1/Constant_2_output_0",
                    ],
                    outputs=[
                        "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0"
                    ],
                    name="/down_blocks.0/resnets.0/norm1/InstanceNormalization",
                    doc_string="",
                    epsilon=9.999999747378752e-06,
                ),
            ],
        ),
    )

    return model


def weight_matmul_model(in_features=10, out_features=20):
    seq_len = 10
    matmul_layer = helper.make_node(
        "MatMul", inputs=["input", "weight"], name="matmul", outputs=["output"]
    )
    weight = numpy_helper.from_array(
        np.random.randn(in_features, out_features).astype(np.float32), name="weight"
    )
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, seq_len, in_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, seq_len, out_features]
    )
    graph = helper.make_graph(
        [matmul_layer],
        "matmul_graph",
        initializer=[weight],
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = make_model(graph)
    onnx.checker.check_model(model)
    return model


def weight_gemm_model(in_features, out_features, transposed_weight=False):
    matmul_layer = helper.make_node(
        "Gemm",
        inputs=["input", "weight"],
        name="matmul",
        outputs=["output"],
        transB=transposed_weight,
    )
    weight_shape = (
        (in_features, out_features)
        if not transposed_weight
        else (out_features, in_features)
    )
    weight = numpy_helper.from_array(
        np.random.randn(*weight_shape).astype(np.float32), name="weight"
    )
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, in_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, out_features]
    )
    graph = helper.make_graph(
        [matmul_layer],
        "matmul_graph",
        initializer=[weight],
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = make_model(graph)
    onnx.checker.check_model(model)
    return model


def conv_model(weight_shape, input_shape, output_shape, transpose=False, **kwargs):
    layer_type = "Conv" if not transpose else "ConvTranspose"
    conv_layer = helper.make_node(
        layer_type,
        inputs=["input", "weight"],
        name="conv",
        outputs=["output"],
        **kwargs,
    )
    weight = numpy_helper.from_array(
        np.random.randn(*weight_shape).astype(np.float32), name="weight"
    )
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, output_shape
    )
    graph = helper.make_graph(
        [conv_layer],
        "conv_graph",
        initializer=[weight],
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = make_model(graph)
    onnx.checker.check_model(model)
    return model


def pointwise_conv1d(input_shape):
    return conv_model(
        weight_shape=(input_shape[1], input_shape[1], 1),
        input_shape=input_shape,
        output_shape=input_shape,
        auto_pad="SAME_UPPER",
    )


def pointwise_conv3d(input_shape):
    return conv_model(
        weight_shape=(input_shape[1], input_shape[1], 1, 1, 1),
        input_shape=input_shape,
        output_shape=input_shape,
        auto_pad="SAME_UPPER",
    )


def pointwise_convtranspose1d(input_shape):
    return conv_model(
        weight_shape=(input_shape[1], input_shape[1], 1),
        input_shape=input_shape,
        output_shape=input_shape,
        transpose=True,
        auto_pad="SAME_UPPER",
    )


def pointwise_convtranspose3d(input_shape):
    return conv_model(
        weight_shape=(input_shape[1], input_shape[1], 1, 1, 1),
        input_shape=input_shape,
        output_shape=input_shape,
        transpose=True,
        auto_pad="SAME_UPPER",
    )


def padded_convtranspose2d(input_shape):
    pads = [1, 0, 1, 3]
    return conv_model(
        weight_shape=(input_shape[1], input_shape[1], 1, 1),
        input_shape=input_shape,
        output_shape=[
            input_shape[0],
            input_shape[1],
            input_shape[2] - pads[0] - pads[2],
            input_shape[3] - pads[1] - pads[3],
        ],
        transpose=True,
        pads=pads,
    )


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

    buffer = io.BytesIO()

    torch.onnx.export(
        model,
        torch.randn(batch_size, 10),
        buffer,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
        dynamo=False,
    )
    buffer.seek(0)
    onnx_model = onnx.load(buffer)
    return onnx_model


def simplifiable_model(batch_size=1):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(10, 10))
            self.weight2 = torch.nn.Parameter(torch.randn(20, 10))

        def forward(self, x):
            # Add some nonsense operations that will get folded in onnx simplifier
            weight3 = torch.nn.functional.linear(self.weight2, self.weight1)
            return torch.nn.functional.linear(x, weight3)

    model = Model()

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        torch.randn(batch_size, 10),
        buffer,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
        dynamo=False,
    )
    buffer.seek(0)
    onnx_model = onnx.load(buffer)
    return onnx_model


def layernorm_model():
    model = make_model(
        graph=helper.make_graph(
            name="LayerNormModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 4, 64, 64]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 32, 40960]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(320, 4, 3, 3).astype("float32"),
                    name="conv_in.weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(320).astype("float32"), name="conv_in.bias"
                ),
                numpy_helper.from_array(
                    np.random.randn(32).astype("float32"), name="layernorm.scale"
                ),
                numpy_helper.from_array(
                    np.random.randn(32).astype("float32"), name="layernorm.bias"
                ),
            ],
            value_info=[
                helper.make_tensor_value_info(
                    "/conv_in/Conv_output_0", TensorProto.FLOAT, shape=[1, 320, 64, 64]
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Constant_output_0",
                    TensorProto.INT64,
                    shape=[3],
                ),
                helper.make_tensor_value_info(
                    "/down_blocks.0/resnets.0/norm1/Reshape_output_0",
                    TensorProto.FLOAT,
                    shape=[1, 32, 40960],
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["input", "conv_in.weight", "conv_in.bias"],
                    outputs=["/conv_in/Conv_output_0"],
                    name="/conv_in/Conv",
                    doc_string="",
                    dilations=[1, 1],
                    group=1,
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/down_blocks.0/resnets.0/norm1/Constant_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Constant",
                    doc_string="",
                    value=numpy_helper.from_array(
                        np.array([-1, 32], dtype="int64"), name=""
                    ),
                ),
                helper.make_node(
                    "Reshape",
                    inputs=[
                        "/conv_in/Conv_output_0",
                        "/down_blocks.0/resnets.0/norm1/Constant_output_0",
                    ],
                    outputs=["/down_blocks.0/resnets.0/norm1/Reshape_output_0"],
                    name="/down_blocks.0/resnets.0/norm1/Reshape",
                    doc_string="",
                    allowzero=0,
                ),
                helper.make_node(
                    "LayerNormalization",
                    inputs=[
                        "/down_blocks.0/resnets.0/norm1/Reshape_output_0",
                        "layernorm.scale",
                        "layernorm.bias",
                    ],
                    outputs=["output"],
                    name="/down_blocks.0/resnets.0/norm1/LayerNormalization",
                    doc_string="",
                    epsilon=9.999999747378752e-06,
                ),
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def model_with_exceptional_ops():
    opset = OperatorSetIdProto()
    opset.version = 21
    model = make_model(
        graph=helper.make_graph(
            name="MatMul_GroupNorm_Model",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 12, 8, 8]
                ),
                helper.make_tensor_value_info(
                    "matmul_0.weight", TensorProto.FLOAT, shape=[1, 12, 6, 6]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 12, 6, 6]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(12, 12, 3, 3).astype("float32"),
                    name="conv_0.weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(12).astype("float32"), name="conv_0.bias"
                ),
                numpy_helper.from_array(
                    np.random.randn(12).astype("float32"), name="groupnorm_0.scale"
                ),
                numpy_helper.from_array(
                    np.random.randn(12).astype("float32"), name="groupnorm_0.bias"
                ),
            ],
            value_info=[
                helper.make_tensor_value_info(
                    "/conv_0/output_0", TensorProto.FLOAT, shape=[1, 12, 6, 6]
                ),
                helper.make_tensor_value_info(
                    "/matmul_0/output_0", TensorProto.FLOAT, shape=[1, 12, 6, 6]
                ),
                helper.make_tensor_value_info(
                    "/groupnorm_0/output_0", TensorProto.FLOAT, shape=[1, 12, 6, 6]
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv_0.weight", "conv_0.bias"],
                    outputs=["/conv_0/output_0"],
                    name="conv_0",
                    kernel_shape=[3, 3],
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["/conv_0/output_0", "matmul_0.weight"],
                    outputs=["/matmul_0/output_0"],
                    name="matmul_0",
                ),
                helper.make_node(
                    "GroupNormalization",
                    inputs=[
                        "/matmul_0/output_0",
                        "groupnorm_0.scale",
                        "groupnorm_0.bias",
                    ],
                    outputs=["/groupnorm_0/output_0"],
                    name="groupnorm_0",
                    num_groups=2,
                ),
                helper.make_node(
                    "Relu",
                    inputs=["/groupnorm_0/output_0"],
                    outputs=["model_output"],
                    name="relu_0",
                ),
            ],
        ),
        opset_imports=[opset],
    )
    onnx.checker.check_model(model)
    return model


def resize_op_model():
    model = make_model(
        graph=helper.make_graph(
            name="ResizeModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 3, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 1, 12, 12]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(1, 3, 3, 3).astype("float32"), name="conv_weight"
                )
            ],
            value_info=[
                helper.make_tensor_value_info(
                    "conv_output", TensorProto.FLOAT, shape=[1, 1, 6, 6]
                ),
                helper.make_tensor_value_info(
                    "const_scale", TensorProto.FLOAT, shape=[4]
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv_weight"],
                    outputs=["conv_output"],
                    kernel_shape=[3, 3],
                    name="conv",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["const_scale"],
                    value=numpy_helper.from_array(
                        np.array([1, 1, 2, 2]).astype("float32"), name=""
                    ),
                ),
                helper.make_node(
                    "Resize",
                    inputs=["conv_output", "", "const_scale"],
                    outputs=["model_output"],
                    name="resize",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def add_matmul_model():
    # Define the fixed shape input tensor
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [3, 3]
    )
    input_tensor_2 = helper.make_tensor_value_info(
        "input_2", onnx.TensorProto.FLOAT, [3, 3]
    )

    # Define the constant 1D vector to be added to the input
    constant_vector = np.random.randn(3).astype(np.float32)
    constant_tensor = helper.make_tensor(
        "constant_vector", onnx.TensorProto.FLOAT, [3], constant_vector
    )

    # Define the weights for both matmuls
    weight1 = helper.make_tensor(
        "weight1",
        onnx.TensorProto.FLOAT,
        [3, 3],
        np.random.randn(3, 3).astype(np.float32),
    )

    # Define the Add node
    add_node = helper.make_node(
        "Add",  # The operation is an addition
        ["input", "constant_vector"],  # The inputs: input and constant vector
        ["added_output"],  # The output name
        name="AddNode",
    )

    # Define the MatMul nodes
    matmul1 = helper.make_node(
        "MatMul",  # The operation is a matrix multiplication
        ["added_output", "weight1"],  # The inputs: added output and weight1
        ["output1"],  # The output name
        name="MatMul1",
    )

    matmul2 = helper.make_node(
        "MatMul",  # The operation is a matrix multiplication
        ["input_2", "added_output"],  # The inputs: added output and weight2
        ["output2"],  # The output name
        name="MatMul2",
    )

    # Define the output tensors
    output_tensor1 = helper.make_tensor_value_info(
        "output1", onnx.TensorProto.FLOAT, [3, 3]
    )
    output_tensor2 = helper.make_tensor_value_info(
        "output2", onnx.TensorProto.FLOAT, [3, 3]
    )

    # Create the graph
    graph_def = helper.make_graph(
        [add_node, matmul1, matmul2],  # The list of nodes in the graph
        "add_matmul_graph",  # A name for the graph
        [input_tensor, input_tensor_2],  # The list of inputs
        [output_tensor1, output_tensor2],  # The list of outputs
        [constant_tensor, weight1],  # Initializers (weights and constants)
    )

    # Create the model
    model_def = make_model(graph_def, producer_name="add_matmul")
    return model_def


def model_with_initializers_as_activations():
    model = make_model(
        graph=helper.make_graph(
            name="IntializersAsActivationsModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 3, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 3, 8, 8]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(
                        1,
                    ).astype("float32"),
                    name="mul_input2",
                ),
                numpy_helper.from_array(
                    np.random.randn(
                        1,
                    ).astype("float32"),
                    name="add_input2",
                ),
            ],
            value_info=[
                helper.make_tensor_value_info(
                    "mul_output", TensorProto.FLOAT, shape=[1, 3, 8, 8]
                ),
            ],
            nodes=[
                helper.make_node(
                    "Mul",
                    inputs=["model_input", "mul_input2"],
                    outputs=["mul_output"],
                    name="mul_0",
                ),
                helper.make_node(
                    "Add",
                    inputs=["mul_output", "add_input2"],
                    outputs=["model_output"],
                    name="add_0",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def gather_op_model():
    model = make_model(
        graph=helper.make_graph(
            name="GatherModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.INT64, shape=[1, 4]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 4, 3]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 3).astype("float32"), name="gather_weight"
                )
            ],
            nodes=[
                helper.make_node(
                    "Gather",
                    inputs=["gather_weight", "model_input"],
                    outputs=["model_output"],
                    name="gather",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def gather_op_with_int_data_model():
    model = make_model(
        graph=helper.make_graph(
            name="GatherModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.INT64, shape=[1, 4]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.INT64, shape=[1, 4]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.array([1, 3, 224, 224], dtype=np.int64), name="gather_weight"
                )
            ],
            nodes=[
                helper.make_node(
                    "Gather",
                    inputs=["gather_weight", "model_input"],
                    outputs=["model_output"],
                    name="gather",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def matmul_add_model():
    model = make_model(
        graph=helper.make_graph(
            name="MatMulAddModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 1, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 1, 8, 8]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.full((1, 1, 8, 8), 25.0).astype("float32"),
                    name="matmul_1.weight",
                ),
                numpy_helper.from_array(
                    np.full((1, 1, 8, 8), 100.0).astype("float32"), name="add_1.weight"
                ),
                numpy_helper.from_array(
                    np.full((1, 1, 8, 8), -50.0).astype("float32"), name="add_2.weight"
                ),
            ],
            nodes=[
                helper.make_node(
                    "MatMul",
                    inputs=["model_input", "matmul_1.weight"],
                    outputs=["matmul_1.output"],
                    name="matmul_1",
                ),
                helper.make_node(
                    "Add",
                    inputs=["matmul_1.output", "add_1.weight"],
                    outputs=["add_1.output"],
                    name="add_1",
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["model_input", "add_1.output"],
                    outputs=["matmul_2.output"],
                    name="matmul_2",
                ),
                helper.make_node(
                    "Add",
                    inputs=["matmul_2.output", "add_2.weight"],
                    outputs=["model_output"],
                    name="add_2",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def matmul_bias_add_model(bias_first=False):
    opset = OperatorSetIdProto()
    opset.version = 18
    model = make_model(
        graph=helper.make_graph(
            name="MatmulAdd",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 10).astype("float32"), name="matmul.weight"
                ),
                numpy_helper.from_array(
                    np.random.randn(10).astype("float32"), name="add.bias"
                ),
            ],
            nodes=[
                helper.make_node(
                    "MatMul",
                    inputs=["input", "matmul.weight"],
                    outputs=["matmul_out"],
                    name="matmul",
                ),
                helper.make_node(
                    "Add",
                    inputs=["add.bias", "matmul_out"]
                    if bias_first
                    else ["matmul_out", "add.bias"],
                    outputs=["output"],
                    name="add",
                ),
            ],
        ),
        opset_imports=[opset],
    )
    onnx.checker.check_model(model)
    return model


def matmul_add_with_transpose(include_perm_attr=False, dynamic_weight=False):
    """Create a model with explicit Transpose node (transB should be 1)."""
    from onnx import helper, TensorProto

    in_feat, out_feat = 64, 128

    # Create graph: input -> MatMul(input, Transpose(weight)) -> Add(bias, matmul) -> output
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 32, in_feat]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 32, out_feat]
    )

    bias = np.random.randn(out_feat).astype(np.float32)
    bias_init = onnx.numpy_helper.from_array(bias, "bias")

    if dynamic_weight:
        weight = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, [out_feat, in_feat]
        )
        inputs = [input_tensor, weight]
        inits = [bias_init]
    else:
        weight_np = np.random.randn(out_feat, in_feat).astype(np.float32)
        weight_init = onnx.numpy_helper.from_array(weight_np, "weight")
        inputs = [input_tensor]
        inits = [weight_init, bias_init]

    if include_perm_attr:
        transpose_node = helper.make_node(
            "Transpose", ["weight"], ["weight_t"], perm=[1, 0]
        )
    else:
        # Defaults to swapping axes
        transpose_node = helper.make_node("Transpose", ["weight"], ["weight_t"])

    matmul_node = helper.make_node("MatMul", ["input", "weight_t"], ["matmul_out"])
    add_node = helper.make_node("Add", ["bias", "matmul_out"], ["output"])

    graph = helper.make_graph(
        [transpose_node, matmul_node, add_node],
        "matmuladd_with_transpose",
        inputs=inputs,
        outputs=[output_tensor],
        initializer=inits,
    )

    model = make_model(graph)
    onnx.checker.check_model(model)
    return model


def softmax_model():
    model = make_model(
        graph=helper.make_graph(
            name="SoftmaxModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 1, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 1, 8, 8]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.full((1, 1, 8, 8), 25.0).astype("float32"),
                    name="matmul_1.weight",
                ),
            ],
            nodes=[
                helper.make_node(
                    "MatMul",
                    inputs=["model_input", "matmul_1.weight"],
                    outputs=["matmul.output"],
                    name="matmul",
                ),
                helper.make_node(
                    "Softmax",
                    inputs=["matmul.output"],
                    outputs=["softmax.output"],
                    name="softmax",
                ),
                helper.make_node(
                    "Tanh",
                    inputs=["softmax.output"],
                    outputs=["tanh.output"],
                    name="tanh",
                ),
                helper.make_node(
                    "Sigmoid",
                    inputs=["tanh.output"],
                    outputs=["model_output"],
                    name="sigmoid",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def standalone_batchnorm(input_shape: tuple[int, int, int, int]):
    _, num_channels, *_ = input_shape

    model = make_model(
        graph=helper.make_graph(
            name="BatchnormModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=input_shape
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=input_shape
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    (np.random.randint(2000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.weight",
                ),
                numpy_helper.from_array(
                    (np.random.randint(-1000, 1000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.bias",
                ),
                numpy_helper.from_array(
                    (np.random.randint(-1000, 1000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.input_mean",
                ),
                numpy_helper.from_array(
                    (np.random.randint(100, 2000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.input_var",
                ),
            ],
            nodes=[
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "input",
                        "batchnorm.weight",
                        "batchnorm.bias",
                        "batchnorm.input_mean",
                        "batchnorm.input_var",
                    ],
                    outputs=["output"],
                    name="batchnorm",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def standalone_dynamic_batchnorm(input_shape: tuple[int, int, int, int]):
    _, num_channels, *_ = input_shape

    model = make_model(
        graph=helper.make_graph(
            name="BatchnormModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=input_shape
                ),
                helper.make_tensor_value_info(
                    "batchnorm.input_mean", TensorProto.FLOAT, shape=[num_channels]
                ),
                helper.make_tensor_value_info(
                    "batchnorm.input_var", TensorProto.FLOAT, shape=[num_channels]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=input_shape
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    (np.random.randint(2000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.weight",
                ),
                numpy_helper.from_array(
                    (np.random.randint(-1000, 1000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="batchnorm.bias",
                ),
            ],
            nodes=[
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "input",
                        "batchnorm.weight",
                        "batchnorm.bias",
                        "batchnorm.input_mean",
                        "batchnorm.input_var",
                    ],
                    outputs=["output"],
                    name="batchnorm",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


batchnorm_model = functools.partial(standalone_batchnorm, (10, 10, 8, 8))
dynamic_batchnorm_model = functools.partial(
    standalone_dynamic_batchnorm, (10, 10, 8, 8)
)


def standalone_batchnorm_constants(
    input_shape, opset_version: int = _DEFAULT_OPSET_VERSION
):
    _, num_channels, *_ = input_shape

    model = make_model(
        opset_imports=[helper.make_operatorsetid("", opset_version)],
        graph=helper.make_graph(
            name="BatchnormModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=input_shape
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=input_shape
                )
            ],
            initializer=[],
            nodes=[
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["batchnorm.weight"],
                    value=numpy_helper.from_array(
                        (np.random.randint(2000, size=num_channels) / 1000).astype(
                            np.float32
                        )
                    ),
                    name="weight",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["batchnorm.bias"],
                    value=numpy_helper.from_array(
                        (
                            np.random.randint(-1000, 1000, size=num_channels) / 1000
                        ).astype(np.float32)
                    ),
                    name="bias",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["batchnorm.input_mean"],
                    value=numpy_helper.from_array(
                        (
                            np.random.randint(-1000, 1000, size=num_channels) / 1000
                        ).astype(np.float32)
                    ),
                    name="input_mean",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["batchnorm.input_var"],
                    value=numpy_helper.from_array(
                        (np.random.randint(100, 2000, size=num_channels) / 1000).astype(
                            np.float32
                        )
                    ),
                    name="input_var",
                ),
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "input",
                        "batchnorm.weight",
                        "batchnorm.bias",
                        "batchnorm.input_mean",
                        "batchnorm.input_var",
                    ],
                    outputs=["output"],
                    name="batchnorm",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


batchnorm_model_constants = functools.partial(
    standalone_batchnorm_constants, (10, 10, 8, 8)
)


def integer_concat_model():
    model = make_model(
        graph=helper.make_graph(
            name="IntConcatModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 10, 1]
                )
            ],
            initializer=[],
            nodes=[
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["constant_one"],
                    value=numpy_helper.from_array(np.array([1]).astype("int64")),
                    name="one",
                ),
                helper.make_node(
                    "Shape",
                    inputs=["model_input"],
                    outputs=["input_shape"],
                    name="shape",
                ),
                helper.make_node(
                    "Concat",
                    inputs=["input_shape", "constant_one"],
                    outputs=["output_shape"],
                    name="out_shape",
                    axis=0,
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["model_input", "output_shape"],
                    outputs=["model_output"],
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def shared_stat_batchnorm_model():
    return shared_tensor_batchnorm_model_with_identities(
        shared_stat=True, shared_bn_weight=True
    )


def shared_tensor_batchnorm_model_with_identities(
    *,
    shared_stat: bool = False,
    shared_conv_weight: bool = False,
    shared_bn_weight: bool = False,
):
    CHANNELS = 10

    def random_array(shape, name, abs=False):
        array = np.random.randn(*shape).astype("float32")
        if abs:
            array = np.abs(array)
        return numpy_helper.from_array(
            array,
            name=name,
        )

    conv1_weight = random_array((CHANNELS, CHANNELS, 1, 1), "conv_1.weight")
    bn1_weight = random_array((CHANNELS,), "batchnorm.weight")
    bn1_bias = random_array((CHANNELS,), "batchnorm.bias")
    bn1_mean = random_array((CHANNELS,), "batchnorm.input_mean")
    bn1_var = random_array((CHANNELS,), "batchnorm.input_var", abs=True)
    initializers = [conv1_weight, bn1_weight, bn1_bias, bn1_mean, bn1_var]

    if not shared_conv_weight:
        conv2_weight = random_array((CHANNELS, CHANNELS, 1, 1), "conv_2.weight")
        initializers.append(conv2_weight)
    else:
        conv2_weight = conv1_weight

    if not shared_bn_weight:
        bn2_weight = random_array((CHANNELS,), "batchnorm2.weight")
        bn2_bias = random_array((CHANNELS,), "batchnorm2.bias")
        initializers.extend([bn2_weight, bn2_bias])
    else:
        bn2_weight = bn1_weight
        bn2_bias = bn1_bias

    if not shared_stat:
        bn2_mean = random_array((CHANNELS,), "batchnorm2.input_mean")
        bn2_var = random_array((CHANNELS,), "batchnorm2.input_var", abs=True)
        initializers.extend([bn2_mean, bn2_var])
    else:
        bn2_mean = bn1_mean
        bn2_var = bn1_var

    model = make_model(
        graph=helper.make_graph(
            name="BatchnormModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[10, 10, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 10, 8, 8]
                )
            ],
            initializer=initializers,
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", conv1_weight.name],
                    outputs=["conv_1.output"],
                    name="conv_1",
                ),
                helper.make_node(
                    "Identity",
                    inputs=[bn1_mean.name],
                    outputs=["batchnorm1.input_mean_"],
                    name="identity_1",
                ),
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "conv_1.output",
                        bn1_weight.name,
                        bn1_bias.name,
                        "batchnorm1.input_mean_",
                        bn1_var.name,
                    ],
                    outputs=["batch_norm_1.output"],
                    name="batchnorm_1",
                ),
                helper.make_node(
                    "Identity",
                    inputs=[conv2_weight.name],
                    outputs=["conv_2.weight_"],
                    name="identity_3",
                ),
                helper.make_node(
                    "Conv",
                    inputs=["batch_norm_1.output", "conv_2.weight_"],
                    outputs=["conv_2.output"],
                    name="conv_2",
                ),
                helper.make_node(
                    "Identity",
                    inputs=[bn2_mean.name],
                    outputs=["batchnorm2.input_mean_"],
                    name="identity_2",
                ),
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "conv_2.output",
                        bn2_weight.name,
                        bn2_bias.name,
                        "batchnorm2.input_mean_",
                        bn2_var.name,
                    ],
                    outputs=["model_output"],
                    name="batchnorm_2",
                ),
            ],
        )
    )

    onnx.checker.check_model(model, True)
    return model


def matmul_with_constant_first_input():
    model = make_model(
        graph=helper.make_graph(
            name="MatMulModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[10]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 1]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 10).astype("float32"), name="matmul.weight"
                ),
                numpy_helper.from_array(np.array([1]).astype("int64"), name="axes"),
            ],
            nodes=[
                helper.make_node(
                    "Unsqueeze",
                    inputs=["model_input", "axes"],
                    outputs=["matmul_input"],
                    name="/Unsqueeze",
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["matmul.weight", "matmul_input"],
                    outputs=["model_output"],
                    name="matmul",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def conv_with_weight_identity_input():
    model = make_model(
        graph=helper.make_graph(
            name="IdentityConvModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[10, 10, 32, 32]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 10, 32, 32]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 10, 1, 1).astype("float32"),
                    name="identity.input",
                ),
            ],
            nodes=[
                helper.make_node(
                    "Identity",
                    inputs=["identity.input"],
                    outputs=["conv.weight"],
                ),
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv.weight"],
                    outputs=["model_output"],
                    name="conv",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def dynamic_conv_model(conv_transpose: bool = False):
    model = make_model(
        graph=helper.make_graph(
            name="DynamicConvModel",
            inputs=[
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, shape=[10, 10, 32, 32]
                ),
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, shape=[10, 10, 1, 1]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 10, 32, 32]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 10, 1, 1).astype("float32"), name="add.input"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Add",
                    inputs=["y", "add.input"],
                    outputs=["dynamic_conv.weight"],
                    name="add",
                ),
                helper.make_node(
                    "ConvTranspose" if conv_transpose else "Conv",
                    inputs=["x", "dynamic_conv.weight"],
                    outputs=["model_output"],
                    name="conv",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def squeezenet1_0(tmpdir):
    import torchvision

    filepath = os.path.join(os.path.join(tmpdir, "squeezenet1_0.onnx"))
    model = torchvision.models.squeezenet1_0()
    torch.onnx.export(
        model.eval(),
        torch.randn(1, 3, 224, 224),
        filepath,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    model = onnx.load(filepath)
    return ONNXModel(model)


if sys.platform != "win32" or platform.machine().lower() not in (
    "aarch64",
    "arm64",
):
    # onnxruntime_extensions is not available on Windows ARM64/ARM64EC
    from onnxruntime_extensions import PyOp, onnx_op

    @onnx_op(
        op_type="CustomAdd",
        inputs=[PyOp.dt_float, PyOp.dt_float],
        outputs=[PyOp.dt_float],
    )
    def add_op(x, y):
        return x + y


def custom_op_model():
    model = make_model(
        graph=helper.make_graph(
            name="CustomAddModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            initializer=[],
            nodes=[
                helper.make_node(
                    "Relu",
                    inputs=["model_input"],
                    outputs=["y"],
                    name="Relu",
                ),
                helper.make_node(
                    "CustomAdd",
                    inputs=["model_input", "y"],
                    outputs=["z"],
                    domain="ai.onnx.contrib",
                    name="CustomAdd_1",
                ),
                helper.make_node(
                    "CustomAdd",
                    inputs=["z", "y"],
                    outputs=["output"],
                    domain="ai.onnx.contrib",
                    name="CustomAdd_2",
                ),
                helper.make_node("Exp", inputs=["output"], outputs=["model_output"]),
            ],
        ),
        opset_imports=[helper.make_operatorsetid("ai.onnx.contrib", 1)],
    )
    return model


def model_with_split_matmul():
    model = make_model(
        graph=helper.make_graph(
            name="SplitMatMulModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 128, 8, 750]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "matmul_output", TensorProto.FLOAT, shape=[1, 15, 60, 6000]
                ),
                helper.make_tensor_value_info(
                    "split_output_0", TensorProto.FLOAT, shape=[1, 32, 8, 750]
                ),
                helper.make_tensor_value_info(
                    "split_output_1", TensorProto.FLOAT, shape=[1, 32, 8, 750]
                ),
                helper.make_tensor_value_info(
                    "split_output_3", TensorProto.FLOAT, shape=[1, 32, 8, 750]
                ),
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(128, 128, 1, 1).astype("float32"),
                    name="conv_weight",
                ),
                numpy_helper.from_array(
                    np.array([32, 32, 32, 32], dtype=np.int64), name="split_input"
                ),
                numpy_helper.from_array(
                    np.array([1, 1, 32, 6000], dtype=np.int64), name="reshape_input"
                ),
                numpy_helper.from_array(
                    np.random.randn(1, 15, 60, 32).astype("float32"),
                    name="matmul_weight",
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv_weight"],
                    outputs=["conv_output"],
                    name="conv",
                ),
                helper.make_node(
                    "Split",
                    inputs=["conv_output", "split_input"],
                    outputs=[
                        "split_output_0",
                        "split_output_1",
                        "split_output_2",
                        "split_output_3",
                    ],
                    name="split",
                    axis=1,
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["split_output_2", "reshape_input"],
                    outputs=["reshape_output"],
                    name="reshape",
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["matmul_weight", "reshape_output"],
                    outputs=["matmul_output"],
                    name="matmul",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def model_with_4d_matmul_weight():
    model = make_model(
        graph=helper.make_graph(
            name="SplitMatMulModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[5, 6000, 32]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "matmul_output", TensorProto.FLOAT, shape=[5, 6000, 60]
                ),
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(5, 32, 60).astype("float32"), name="matmul_weight"
                ),
            ],
            nodes=[
                helper.make_node(
                    "MatMul",
                    inputs=["model_input", "matmul_weight"],
                    outputs=["matmul_output"],
                    name="matmul",
                )
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def gather_concat_model():
    #
    # x -- Mul ---------------
    #                        |
    # y -- Mul -- Cast --- Gather -- Concat

    model = make_model(
        graph=helper.make_graph(
            name="GatherConcatModel",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, shape=[10, 10]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, shape=[2]),
            ],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.FLOAT, shape=[10, 2])
            ],
            initializer=[
                numpy_helper.from_array(np.array([2.0]).astype("float32"), name="scale")
            ],
            nodes=[
                helper.make_node("Mul", ["x", "scale"], ["x_2"], name="mul_1"),
                helper.make_node("Mul", ["y", "scale"], ["z"], name="mul_2"),
                helper.make_node(
                    "Cast", ["z"], ["z_int"], name="cast", to=TensorProto.INT64
                ),
                helper.make_node(
                    "Gather", ["x_2", "z_int"], ["w"], name="gather", axis=1
                ),
                helper.make_node("Concat", ["w"], ["out"], name="concat", axis=0),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def mobilenetv2():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model = models.MobileNetV2().eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.PRESERVE,
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def resnet18():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model = models.MobileNetV2().eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.PRESERVE,
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def model_with_initializers_in_graph_input():
    """
    Models with IR<4 have initializers in graph input compulsorily
    """
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=3,
        graph=helper.make_graph(
            name="InitInGraphInputModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
                helper.make_tensor_value_info(
                    "conv_weight", TensorProto.FLOAT, shape=[8, 3, 3, 3]
                ),
                helper.make_tensor_value_info(
                    "conv_bias", TensorProto.FLOAT, shape=[8]
                ),
                helper.make_tensor_value_info("bn_scale", TensorProto.FLOAT, shape=[8]),
                helper.make_tensor_value_info("bn_bias", TensorProto.FLOAT, shape=[8]),
                helper.make_tensor_value_info("bn_mean", TensorProto.FLOAT, shape=[8]),
                helper.make_tensor_value_info(
                    "bn_variance", TensorProto.FLOAT, shape=[8]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 8, 30, 30]
                ),
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(8, 3, 3, 3).astype("float32"), name="conv_weight"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="conv_bias"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="bn_scale"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="bn_bias"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="bn_mean"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="bn_variance"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv_weight", "conv_bias"],
                    outputs=["conv_output"],
                    name="conv",
                ),
                helper.make_node(
                    "BatchNormalization",
                    inputs=[
                        "conv_output",
                        "bn_scale",
                        "bn_bias",
                        "bn_mean",
                        "bn_variance",
                    ],
                    outputs=["bn_output"],
                    name="bn",
                ),
                helper.make_node(
                    "Relu", inputs=["bn_output"], outputs=["model_output"], name="relu"
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def standalone_layernorm(input_shape: tuple[int, int, int]):
    *_, num_channels = input_shape

    model = make_model(
        graph=helper.make_graph(
            name="LayerNormalization",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=input_shape
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=input_shape
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    (np.random.randint(2000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="scale",
                ),
                numpy_helper.from_array(
                    (np.random.randint(-1000, 1000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="bias",
                ),
            ],
            nodes=[
                helper.make_node(
                    "LayerNormalization",
                    inputs=["input", "scale", "bias"],
                    outputs=["output"],
                    name="LayerNormalization",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def standalone_instancenorm(input_shape: tuple[int, int, int]):
    _, num_channels, _ = input_shape

    model = make_model(
        graph=helper.make_graph(
            name="InstanceNormalization",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=input_shape
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=input_shape
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    (np.random.randint(2000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="scale",
                ),
                numpy_helper.from_array(
                    (np.random.randint(-1000, 1000, size=num_channels) / 1000).astype(
                        np.float32
                    ),
                    name="bias",
                ),
            ],
            nodes=[
                helper.make_node(
                    "InstanceNormalization",
                    inputs=["input", "scale", "bias"],
                    outputs=["output"],
                    name="InstanceNormalization",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def standalone_groupnorm(input_shape: tuple[int, int, int, int], num_groups: int = 2):
    _, num_channels, *_ = input_shape

    model = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels).eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        torch.randn(input_shape),
        buffer,
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=True,
    )
    buffer.seek(0)
    return load_model(buffer)


def model_with_ignore_ops(tmpdir):
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv = nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            )
            self.fc = nn.Linear(16 * 16 * 32, 10)

        def forward(self, x):
            x = self.conv(x)
            x = torch.squeeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
            x = torch.transpose(x, 1, 2)
            x1, x2 = torch.split(x, split_size_or_sections=16, dim=1)
            x1 = x1.reshape(1, -1)
            x2 = x2.reshape(1, -1)
            x1 = self.fc(x1)
            x2 = self.fc(x2)
            return x1, x2

    model = CustomModel()
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    model_file_path = os.path.join(tmpdir, "custom_model.onnx")
    torch.onnx.export(model, input_tensor, model_file_path, dynamo=False)

    return onnx.load(model_file_path)


def standalone_gemm(
    in_channels: int, out_channels: int, opset_version: int = _DEFAULT_OPSET_VERSION
):
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", opset_version)],
        graph=helper.make_graph(
            name="Gemm",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, in_channels]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, out_channels]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(in_channels, out_channels).astype(np.float32),
                    name="weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(out_channels).astype(np.float32),
                    name="bias",
                ),
            ],
            nodes=[
                helper.make_node(
                    "Gemm",
                    ["input", "weight", "bias"],
                    ["output"],
                    "Gemm",
                )
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def dynamic_gemm(in_channels: int, out_channels: int):
    model = make_model(
        graph=helper.make_graph(
            name="Gemm",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, in_channels]
                ),
                helper.make_tensor_value_info(
                    "weights", TensorProto.FLOAT, shape=[out_channels, in_channels]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, out_channels]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(out_channels).astype(np.float32),
                    name="bias",
                )
            ],
            nodes=[
                helper.make_node(
                    "Gemm",
                    ["input", "weights", "bias"],
                    ["output"],
                    "Gemm",
                )
            ],
        ),
    )
    onnx.checker.check_model(model)
    return model


def conv_relu():
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="ConvRelu",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 8, 30, 30]
                ),
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(8, 3, 3, 3).astype("float32"), name="conv_weight"
                ),
                numpy_helper.from_array(
                    np.random.randn(8).astype("float32"), name="conv_bias"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["input", "conv_weight", "conv_bias"],
                    outputs=["conv_output"],
                    name="conv",
                ),
                helper.make_node(
                    "Relu", inputs=["conv_output"], outputs=["output"], name="relu"
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def reshape_with_multiple_consumers():
    """
    input -> Relu -+-> Reshape ------> Add ->
                   +-> Sigmoid ---------^
    """
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="reshape_with_multiple_consumers",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[3, 1024]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 3, 1024]
                ),
            ],
            nodes=[
                helper.make_node(
                    "Relu",
                    inputs=["input"],
                    outputs=["relu_output"],
                    name="relu",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["shape"],
                    name="shape",
                    value_ints=[1, 3, 1024],
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["relu_output", "shape"],
                    outputs=["reshape_output"],
                    name="reshape",
                ),
                helper.make_node(
                    "Sigmoid",
                    inputs=["relu_output"],
                    outputs=["sigmoid_output"],
                    name="sigmoid",
                ),
                helper.make_node(
                    "Add",
                    inputs=["reshape_output", "sigmoid_output"],
                    outputs=["output"],
                    name="add",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def diverse_ops(dtype=TensorProto.FLOAT):
    """
    input -> Relu -+-> Reshape ------> MaxPool -> output
    """
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="reshape_with_multiple_consumers",
            inputs=[
                helper.make_tensor_value_info("input", dtype, shape=[3, 1024]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", dtype, shape=[1, 3, 1022]),
            ],
            nodes=[
                helper.make_node(
                    "Relu",
                    inputs=["input"],
                    outputs=["relu_output"],
                    name="relu",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["shape"],
                    name="shape",
                    value_ints=[1, 3, 1024],
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["relu_output", "shape"],
                    outputs=["reshape_output"],
                    name="reshape",
                ),
                helper.make_node(
                    "MaxPool",
                    inputs=["reshape_output"],
                    outputs=["output"],
                    name="maxpool",
                    kernel_shape=[3],
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def conv_prelu_model(
    training=torch.onnx.TrainingMode.EVAL, opset_version=_DEFAULT_OPSET_VERSION
):
    class ConvPrelu(nn.Module):
        """Simple Conv Prelu Model"""

        def __init__(self):
            super(ConvPrelu, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=(1, 1), bias=False)
            self.prelu1 = nn.PReLU()

        def forward(self, *inputs):
            return self.prelu1(self.conv1(inputs[0]))

    x = torch.randn(1, 3, 32, 32)
    model = ConvPrelu()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=training,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=min(opset_version, 20),  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )

    buffer.seek(0)
    model = load_model(buffer)
    if opset_version > 20:
        model = onnx.version_converter.convert_version(model, opset_version)

    model_onnx = ONNXModel(model)
    return model_onnx


def model_with_cast(tensor_type: onnx.TensorProto.DataType):
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="CastModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            initializer=[],
            nodes=[
                helper.make_node(
                    "Cast",
                    inputs=[
                        "input",
                    ],
                    outputs=["cast_output"],
                    name="cast_1",
                    to=tensor_type,
                ),
                helper.make_node(
                    "Relu", inputs=["cast_output"], outputs=["relu_output"], name="relu"
                ),
                helper.make_node(
                    "Cast",
                    inputs=[
                        "relu_output",
                    ],
                    outputs=["output"],
                    name="cast_2",
                    to=onnx.TensorProto.FLOAT,
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def model_with_constant(tensor_type: onnx.TensorProto.DataType):
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="CastModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            initializer=[],
            nodes=[
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["const_tensor"],
                    name="constant_1",
                    value=helper.make_tensor(
                        "const_value",
                        tensor_type,
                        [3, 3, 1, 1],
                        np.random.randn(3, 3).astype(
                            helper.tensor_dtype_to_np_dtype(tensor_type)
                        ),
                    ),
                ),
                helper.make_node(
                    "Cast",
                    inputs=[
                        "const_tensor",
                    ],
                    outputs=["cast_output"],
                    name="cast_1",
                    to=onnx.TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Conv",
                    inputs=["input", "cast_output"],
                    outputs=["output"],
                    name="conv_1",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def integer_output_model():
    model = make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="IntegerOutputModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[1, 3, 32, 32]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "conv_output", TensorProto.FLOAT, shape=[1, 3, 30, 30]
                ),
                helper.make_tensor_value_info(
                    "conv_output2", TensorProto.FLOAT, shape=[1, 8, 30, 30]
                ),
                helper.make_tensor_value_info(
                    "max_conv_output", TensorProto.INT64, shape=[1, 3, 30, 1]
                ),
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(3, 3, 3, 3).astype("float32"), name="weight"
                ),
                numpy_helper.from_array(
                    np.random.randn(8, 3, 3, 3).astype("float32"), name="weight2"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["input", "weight"],
                    outputs=["conv_output"],
                    name="conv",
                ),
                helper.make_node(
                    "Conv",
                    inputs=["input", "weight2"],
                    outputs=["conv_output2"],
                    name="conv2",
                ),
                helper.make_node(
                    "ArgMax",
                    inputs=[
                        "conv_output",
                    ],
                    outputs=["max_conv_output"],
                    name="max_",
                    axis=3,
                    keepdims=True,
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)
    return model


def conv_matmul_model():
    model = make_model(
        graph=helper.make_graph(
            name="ConvMatMulModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, shape=[1, 16, 32]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, shape=[1, 16, 64]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(16, 16, 3).astype("float32"),
                    name="conv1_weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(16, 16, 3).astype("float32"),
                    name="conv2_weight",
                ),
                numpy_helper.from_array(
                    np.random.randn(32, 64).astype("float32"),
                    name="matmul_weight",
                ),
            ],
            nodes=[
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "conv1_weight"],
                    outputs=["conv_output"],
                    name="conv1",
                    auto_pad="SAME_UPPER",
                ),
                helper.make_node(
                    "Relu",
                    inputs=["conv_output"],
                    outputs=["relu_output"],
                    name="relu",
                ),
                helper.make_node(
                    "Conv",
                    inputs=["relu_output", "conv2_weight"],
                    outputs=["conv2_output"],
                    name="conv2",
                    auto_pad="SAME_UPPER",
                ),
                helper.make_node(
                    "Relu",
                    inputs=["conv2_output"],
                    outputs=["relu2_output"],
                    name="relu2",
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["relu2_output", "matmul_weight"],
                    outputs=["model_output"],
                    name="matmul",
                ),
            ],
        )
    )
    onnx.checker.check_model(model, True)
    return model


def rmsnorm_model(tmpdir, elementwise_affine=True, eps=None, opset=17):
    class RmsNormModel(torch.nn.Module):
        def __init__(self, elementwise_affine, eps):
            super().__init__()
            self.rms_norm = torch.nn.RMSNorm(
                (10), eps=eps, elementwise_affine=elementwise_affine
            )
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(self.rms_norm(x))

    dummy_input = torch.randn(1, 5, 10)
    model = RmsNormModel(elementwise_affine, eps)
    fname = os.path.join(tmpdir, "rmsnorm.onnx")
    torch.onnx.export(
        model.eval(),
        dummy_input,
        fname,
        opset_version=opset,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
    )
    model = onnx.load(fname)
    model.graph.ClearField("value_info")
    return model


def llama_rmsnorm_model(tmpdir, opset=17, dtype=torch.float32):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    hidden_size = 32

    model = LlamaRMSNorm(hidden_size).to(dtype)
    dummy_input = torch.randn(1, 10, hidden_size).to(dtype)
    fname = os.path.join(tmpdir, "llama_rmsnorm.onnx")
    torch.onnx.export(
        model.eval(),
        dummy_input,
        fname,
        opset_version=opset,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
    )
    return onnx.load(fname)


def rmsnorm_invalid_multiple_axes(tmpdir, dim: int = 32, opset=16):
    """RMSNorm with multiple axes in ReduceMean - should NOT match"""

    class InvalidRMSNormMultipleAxes(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.randn(dim, dim)
            self.variance_epsilon = 0.003

        def forward(self, x):
            # Use mean over multiple axes instead of just the last one
            variance = x.pow(2).mean(dim=(-1, -2), keepdim=True)
            x = x / torch.sqrt(variance + self.variance_epsilon)
            return x * self.weight

    model = InvalidRMSNormMultipleAxes(dim=dim)
    fname = os.path.join(tmpdir, "rmsnorm_invalid_multiple_axes.onnx")
    x = torch.randn((1, 3, dim, dim))
    torch.onnx.export(
        model,
        x,
        fname,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return onnx.load(fname)


def rmsnorm_invalid_negative_epsilon(tmpdir, dim: int = 32, opset=16):
    """RMSNorm with negative epsilon - should NOT match"""

    class InvalidRMSNormNegativeEpsilon(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.randn(dim)
            self.variance_epsilon = -0.003  # Negative epsilon

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x / torch.sqrt(variance + self.variance_epsilon)
            return x * self.weight

    model = InvalidRMSNormNegativeEpsilon(dim=dim)
    fname = os.path.join(tmpdir, "rmsnorm_invalid_negative_epsilon.onnx")
    x = torch.randn((1, 3, dim, dim))
    torch.onnx.export(
        model,
        x,
        fname,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return onnx.load(fname)


def rmsnorm_invalid_wrong_power(tmpdir, dim: int = 32, opset=16):
    """RMSNorm with power 3 instead of 2 - should NOT match"""

    class InvalidRMSNormWrongPower(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.randn(dim)
            self.variance_epsilon = 0.003

        def forward(self, x):
            # Use power of 3 instead of 2
            variance = x.pow(3).mean(-1, keepdim=True)
            x = x / torch.sqrt(variance + self.variance_epsilon)
            return x * self.weight

    model = InvalidRMSNormWrongPower(dim=dim)
    fname = os.path.join(tmpdir, "rmsnorm_invalid_wrong_power.onnx")
    x = torch.randn((1, 3, dim, dim))
    torch.onnx.export(
        model,
        x,
        fname,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return onnx.load(fname)


def rmsnorm_invalid_intermediate_output(tmpdir, dim: int = 32, opset=16):
    """RMSNorm where intermediate variance is used elsewhere - should NOT match"""

    class InvalidRMSNormIntermediateOutput(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.randn(dim)
            self.variance_epsilon = 0.003

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x / torch.sqrt(variance + self.variance_epsilon)
            output = x * self.weight
            # Add the variance to the output (consuming intermediate result)
            # Broadcast variance to match output shape
            return output + variance

    torch.manual_seed(10)
    model = InvalidRMSNormIntermediateOutput(dim=dim)
    fname = os.path.join(tmpdir, "rmsnorm_invalid_intermediate_output.onnx")
    x = torch.randn((1, 3, dim, dim))
    torch.onnx.export(
        model,
        x,
        fname,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return onnx.load(fname)


def make_split_concat_model(num_splits=128, split_size=32, num_channels=64):
    import numpy as np
    from onnx import helper, TensorProto, numpy_helper

    total = num_splits * split_size
    split_outputs = [f"split_out_{i}" for i in range(num_splits)]

    unsqueeze_axis = numpy_helper.from_array(
        np.array([0], dtype=np.int64),
        name="unsqueeze_axis",
    )
    split_sizes = numpy_helper.from_array(
        np.full(num_splits, split_size, dtype=np.int64),
        name="split_sizes",
    )

    unsqueeze = helper.make_node(
        "Unsqueeze",
        ["input", "unsqueeze_axis"],
        ["unsq_out"],
        name="Unsqueeze_0",
    )
    split = helper.make_node(
        "Split",
        ["unsq_out", "split_sizes"],
        split_outputs,
        name="Split_0",
        axis=-1,
    )
    concat = helper.make_node(
        "Concat",
        split_outputs,
        ["output"],
        name="Concat_0",
        axis=1,
    )

    graph = helper.make_graph(
        [unsqueeze, split, concat],
        "split_concat_subgraph",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, num_channels, total]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, num_splits, num_channels, split_size]
            )
        ],
        initializer=[unsqueeze_axis, split_sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model)
    return model


def unsafe_concat_tie_model():
    """
            -- Add -+-----------+
    Input --+                   Concat ---
            -- Mul -- MaxPool --+
                                -- Add ---
    """
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Mul",
                inputs=["input", "input"],
                outputs=["mul_output"],
                name="mul",
            ),
            helper.make_node(
                "MaxPool",
                inputs=["mul_output"],
                outputs=["maxpool_output"],
                name="maxpool",
                kernel_shape=[3, 1],
                auto_pad="SAME_UPPER",
            ),
            helper.make_node(
                "Add",
                inputs=["input", "input"],
                outputs=["add_output"],
                name="add",
            ),
            helper.make_node(
                "Concat",
                inputs=["maxpool_output", "add_output"],
                outputs=["concat_output"],
                name="concat",
                axis=2,
            ),
            helper.make_node(
                "Add",
                inputs=["add_output", "add_output"],
                outputs=["add_2_output"],
                name="add_2",
            ),
            helper.make_node(
                "Add",
                inputs=["maxpool_output", "maxpool_output"],
                outputs=["add_3_output"],
                name="add_3",
            ),
        ],
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, shape=[1, 1, 32, 32]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "add_2_output", TensorProto.FLOAT, shape=[1, 1, 32, 32]
            ),
            helper.make_tensor_value_info(
                "concat_output", TensorProto.FLOAT, shape=[1, 1, 64, 32]
            ),
            helper.make_tensor_value_info(
                "add_3_output", TensorProto.FLOAT, shape=[1, 1, 32, 32]
            ),
        ],
        name="ReluMaxPoolConcatModel",
    )
    model = make_model(graph=graph)
    onnx.checker.check_model(model, True)
    return model


def relu_maxpool_concat_model():
    """
                    ----- Add -------------
            -- Add -+------------+
    Input --+                    Concat ---
            -- Relu -- MaxPool --+
    """
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Relu",
                inputs=["input"],
                outputs=["relu_output"],
                name="relu",
            ),
            helper.make_node(
                "MaxPool",
                inputs=["relu_output"],
                outputs=["maxpool_output"],
                name="maxpool",
                kernel_shape=[3, 1],
                auto_pad="SAME_UPPER",
            ),
            helper.make_node(
                "Add",
                inputs=["input", "input"],
                outputs=["add_output"],
                name="add",
            ),
            helper.make_node(
                "Concat",
                inputs=["maxpool_output", "add_output"],
                outputs=["concat_output"],
                name="concat",
                axis=2,
            ),
            helper.make_node(
                "Add",
                inputs=["add_output", "add_output"],
                outputs=["add_2_output"],
                name="add_2",
            ),
        ],
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, shape=[1, 1, 32, 32]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "add_2_output", TensorProto.FLOAT, shape=[1, 1, 32, 32]
            ),
            helper.make_tensor_value_info(
                "concat_output", TensorProto.FLOAT, shape=[1, 1, 64, 32]
            ),
        ],
        name="ReluMaxPoolConcatModel",
    )

    model = make_model(graph=graph)
    onnx.checker.check_model(model, True)
    return model


def decomposed_layernorm(
    bias: bool = True, eps: float = 1e-5, bias_first: bool = False
):
    import onnx_ir as ir

    reduce_mean_attrs = {"axes": [-1]}

    tape = ir.tape.Tape()
    x = ir.val("input", dtype=ir.DataType.FLOAT, shape=(1, 32, 10))
    scale = tape.initializer(ir.tensor([1.5] * 10, ir.DataType.FLOAT), name="scale")

    epsilon = tape.op(
        "Constant", [], attributes={"value": ir.tensor([eps], ir.DataType.FLOAT)}
    )
    pow_2 = tape.op(
        "Constant", [], attributes={"value": ir.tensor([2.0], ir.DataType.FLOAT)}
    )

    mean = tape.op("ReduceMean", [x], attributes=reduce_mean_attrs)
    centered = tape.op("Sub", [x, mean])
    squared = tape.op("Pow", [centered, pow_2])
    var = tape.op("ReduceMean", [squared], attributes=reduce_mean_attrs)
    var_eps = tape.op("Add", [var, epsilon])
    std_dev = tape.op("Sqrt", [var_eps])
    normalized = tape.op("Div", [centered, std_dev])
    scaled = tape.op("Mul", [scale, normalized])

    if bias:
        bias_val = tape.initializer(
            ir.tensor([0.5] * 10, ir.DataType.FLOAT), name="bias"
        )
        add_inputs = [bias_val, scaled] if bias_first else [scaled, bias_val]
        output = tape.op("Add", add_inputs)
    else:
        output = scaled

    output.shape = ir.Shape((1, 32, 10))
    output.dtype = ir.DataType.FLOAT
    output.name = "output"

    ir_model = ir.Model(
        ir.Graph(
            inputs=[x],
            outputs=[output],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 17},
            name="layernorm_graph",
        ),
        ir_version=10,
    )
    model = ir.to_proto(ir_model)
    onnx.checker.check_model(model)
    return model


def _scatter_op_params(use_scatter_nd, data_shape, updates_shape):
    if use_scatter_nd:
        indices_data = np.arange(updates_shape[0], dtype=np.int64).reshape(-1, 1)
        return "ScatterND", indices_data, {}
    indices_data = np.tile(
        np.arange(data_shape[1], dtype=np.int64), (data_shape[0], 1)
    )[: updates_shape[0], : updates_shape[1]]
    return "ScatterElements", indices_data, {"axis": 1}


def simple_scatter_model(use_scatter_nd=False):
    """
    data -------> Scatter --> [output]
    updates ---^     ^
                  indices (int64 initializer)
    """
    data_shape = [4, 4]
    updates_shape = [2, 4] if use_scatter_nd else [2, 2]
    op_type, indices_data, attrs = _scatter_op_params(
        use_scatter_nd, data_shape, updates_shape
    )
    indices = numpy_helper.from_array(indices_data, name="indices")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type,
                inputs=["data", "indices", "updates"],
                outputs=["scatter_output"],
                name="scatter",
                **attrs,
            ),
        ],
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, shape=data_shape),
            helper.make_tensor_value_info(
                "updates", TensorProto.FLOAT, shape=updates_shape
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "scatter_output", TensorProto.FLOAT, shape=data_shape
            ),
        ],
        initializer=[indices],
        name=f"Simple{op_type}Model",
    )
    model = make_model(graph=graph)
    onnx.checker.check_model(model, True)
    return model


def scatter_with_shared_input_model(use_scatter_nd=False):
    """
                (data)      (indices)
    Input1 -+  +---+         v
            Add|   Scatter --> Output
    Input2 -+  +---+
                (updates)

    Add output feeds both data and updates of Scatter op.
    """
    shape = [2, 4]
    op_type, indices_data, attrs = _scatter_op_params(use_scatter_nd, shape, shape)
    indices = numpy_helper.from_array(indices_data, name="indices")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Add",
                inputs=["input1", "input2"],
                outputs=["add_output"],
                name="add",
            ),
            helper.make_node(
                op_type,
                inputs=["add_output", "indices", "add_output"],
                outputs=["scatter_output"],
                name="scatter",
                **attrs,
            ),
        ],
        inputs=[
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape=shape),
            helper.make_tensor_value_info("input2", TensorProto.FLOAT, shape=shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "scatter_output", TensorProto.FLOAT, shape=shape
            ),
        ],
        initializer=[indices],
        name=f"{op_type}WithSharedInputModel",
    )
    model = make_model(graph=graph)
    onnx.checker.check_model(model, True)
    return model


def scatter_with_relu_data_model(use_scatter_nd=False):
    """
                          (indices)
                           v
    Input --> Relu --> Scatter --> [output]
          +------------^
                  (updates)

    Relu output feeds data. Raw input feeds updates.
    Relu has encoding constraint [0, None].
    Input has multiple consumers (Relu + Scatter).
    """
    shape = [4, 4]
    op_type, indices_data, attrs = _scatter_op_params(use_scatter_nd, shape, shape)
    indices = numpy_helper.from_array(indices_data, name="indices")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Relu",
                inputs=["input"],
                outputs=["relu_output"],
                name="relu",
            ),
            helper.make_node(
                op_type,
                inputs=["relu_output", "indices", "input"],
                outputs=["scatter_output"],
                name="scatter",
                **attrs,
            ),
        ],
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "scatter_output", TensorProto.FLOAT, shape=shape
            ),
        ],
        initializer=[indices],
        name=f"{op_type}WithReluDataModel",
    )
    model = make_model(graph=graph)
    onnx.checker.check_model(model, True)
    return model


def double_layernorm_model(tmpdir):
    """Two sequential LayerNorm instances (ln1, ln2) at opset 13 (decomposed)."""

    class DoubleLayerNormModel(torch.nn.Module):
        def __init__(self):
            super(DoubleLayerNormModel, self).__init__()
            self.ln1 = torch.nn.LayerNorm(128)
            self.ln2 = torch.nn.LayerNorm(128)

        def forward(self, x):
            x = self.ln1(x)
            x = self.ln2(x)
            return x

    model = DoubleLayerNormModel()
    dummy_input = torch.randn(1, 32, 128)
    model_path = os.path.join(tmpdir, "double_layernorm.onnx")
    torch.onnx.export(model, dummy_input, model_path, opset_version=13, dynamo=False)
    return onnx.load(model_path)


def matmul_add_with_shared_root_naming():
    opset = OperatorSetIdProto()
    opset.version = 18
    model = helper.make_model(
        graph=helper.make_graph(
            name="SharedRoot",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, shape=[10, 10]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.random.randn(10, 10).astype("float32"), name="w1"
                ),
                numpy_helper.from_array(
                    np.random.randn(10).astype("float32"), name="b1"
                ),
                numpy_helper.from_array(
                    np.random.randn(10, 10).astype("float32"), name="w2"
                ),
                numpy_helper.from_array(
                    np.random.randn(10).astype("float32"), name="b2"
                ),
            ],
            nodes=[
                helper.make_node(
                    "MatMul", ["input", "w1"], ["mm1_out"], name="/linear/MatMul"
                ),
                helper.make_node(
                    "Add", ["mm1_out", "b1"], ["add1_out"], name="/linear/Add"
                ),
                helper.make_node(
                    "MatMul", ["add1_out", "w2"], ["mm2_out"], name="/linear/MatMul_1"
                ),
                helper.make_node(
                    "Add", ["mm2_out", "b2"], ["output"], name="/linear/Add_1"
                ),
            ],
        ),
        opset_imports=[opset],
    )
    onnx.checker.check_model(model)
    return model


def model_with_transposed_and_non_transposed_gemm():
    class Model(torch.nn.Module):
        """
        W -> Transpose -----v        W2 -----v
        x --------------> MatMul -> Add -> MatMul -> Add
        """

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(in_features=10, out_features=5)
            self.weight_2 = torch.nn.Parameter(torch.ones(5, 3))
            self.bias_2 = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.linear(x) @ self.weight_2 + self.bias_2

    model = Model()
    x = torch.randn(1, 1, 10)
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (x,),
        buffer,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        optimize=False,  # prevent folding transB
    )
    buffer.seek(0)
    model = load_model(buffer)
    onnx.checker.check_model(model)
    return model
