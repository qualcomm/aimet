# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import contextlib
import copy
import logging
from itertools import chain
import json as json
import os
import tempfile
from pathlib import Path
import unittest.mock
from packaging import version
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from packaging.version import Version
from torchvision import models

import aimet_common.libpymo as libpymo

from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_common.utils import AimetLogger
from aimet_torch import elementwise_ops
from aimet_torch import onnx_utils
from aimet_torch import utils
import aimet_torch.v1.nn.modules.custom as aimet_modules
from aimet_torch.model_preparer import prepare_model
from models.test_models import TwoLayerBidirectionalLSTMModel, SingleLayerRNNModel, \
    ModelWithTwoInputs, SimpleConditional, RoiModel, InputOutputDictModel, Conv3dModel, SmallLinearModel
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper, QcQuantizeStandalone, \
    StaticGridQuantWrapper, QcQuantizeOpMode, LearnedGridQuantWrapper, enable_recompute, no_recompute
from aimet_torch.v1.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.v1.quantsim import QuantizationSimModel, check_accumulator_overflow, load_encodings_to_sim, \
    has_valid_encodings, compute_encodings_for_sims
from aimet_torch.v1.quantsim_straight_through_grad import compute_dloss_by_dx
from aimet_torch.v1.nn.modules.custom import DynamicConv2d

from models import test_models

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


@contextlib.contextmanager
def set_export_to_onnx_direct(export_to_onnx_direct):
    entry_state = onnx_utils.EXPORT_TO_ONNX_DIRECT
    onnx_utils.EXPORT_TO_ONNX_DIRECT = export_to_onnx_direct
    yield
    onnx_utils.EXPORT_TO_ONNX_DIRECT = entry_state


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    model.eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    with torch.no_grad():
        model(*dummy_input)


def dummy_forward_pass(model, args):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn((32, 1, 28, 28)))
    return output


class SmallMnistNoDropoutWithPassThrough(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropoutWithPassThrough, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pt1 = torch.nn.Identity()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pt2 = torch.nn.Identity()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.pt1(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu2(self.pt2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(self.conv2_drop(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnistNoDropout(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class SoftMaxAvgPoolModel(torch.nn.Module):
    def __init__(self):
        super(SoftMaxAvgPoolModel, self).__init__()
        self.sfmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(3)

    def forward(self, inp):
        x = self.sfmax(inp)
        return self.avgpool(x)


class ModelWithStandaloneOps(nn.Module):
    def __init__(self):
        super(ModelWithStandaloneOps, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.myquant = QcQuantizeStandalone(activation_bw=8, round_mode=MAP_ROUND_MODE_TO_PYMO['nearest'],
                                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                                            is_symmetric=False, data_type=QuantizationDataType.int)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = self.relu1(self.maxpool1(self.conv1(inputs[0])))
        x = self.conv2(x)
        x = self.myquant(x)
        x = self.relu2(self.maxpool2(self.conv2_drop(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithTwoInputsOneToAdd(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputsOneToAdd, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(10, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.add = aimet_modules.Add()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x1 = self.relu1_b(self.maxpool1_b(self.conv1_b(x1)))

        x = self.add(x1, x2)

        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class PreluModel(nn.Module):
    def __init__(self):
        super(PreluModel, self).__init__()
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(x)


class PixelShuffleModel(nn.Module):
    def __init__(self):
        super(PixelShuffleModel, self).__init__()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(x)


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


class ModuleListModel(nn.Module):
    def __init__(self):
        super(ModuleListModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
        self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                          nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                          StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8,
                                                                 activation_bw=8,
                                                                 round_mode='nearest',
                                                                 quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                 data_type=QuantizationDataType.int),
                                          nn.ModuleList([nn.MaxPool2d(2), nn.PReLU()])])

    def forward(self, *inputs):
        return self.layers[2](inputs[0])


class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=2, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.conv(inputs))


class ConvTransposeReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 5, kernel_size=2, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.conv(inputs))


class CustModelV1Simple(torch.nn.Module):
    def __init__(self):
        super(CustModelV1Simple, self).__init__()
        self.cust = CustomOp()

    def forward(self, x):
        k1, k2 = self.cust(x)
        return k1, k2


class CustomOp(torch.nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.size = 8
        self.mul1 = aimet_modules.Multiply()
        self.mul2 = aimet_modules.Multiply()

    def forward(self, x):
        y = self.mul1(x, self.size)
        z = self.mul2(x, 5)
        return y, z


class CustModelV2Simple(torch.nn.Module):
    def __init__(self):
        super(CustModelV2Simple, self).__init__()
        self.cust = CustomOpV2()

    def forward(self, x):
        k1, k2, k3, k4, k5 = self.cust(x)
        return k1, k2, k3, k4, k5


class CustomOpV2(torch.nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.size = 8
        self.mul = aimet_modules.Multiply()
        self.add = aimet_modules.Add()
        self.sub = aimet_modules.Subtract()
        self.clamp = Clamp()

    def forward(self, x):
        y = self.mul(x, self.size)
        z = x * 5 # functional op
        a = self.clamp(z)
        b = self.add(y, z)
        c = self.sub(y, z)
        return y, z, a, b, c


class Clamp(torch.nn.Module):
    """ Custom module for a functional clamp"""

    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for add op
        """
        return x.clamp(0)


class ModelInputsSharedConstantIntermediate(nn.Module):
    def __init__(self):
        super(ModelInputsSharedConstantIntermediate, self).__init__()
        self.add1 = aimet_modules.Add()
        self.add2 = aimet_modules.Add()
        self.mul = aimet_modules.Multiply()
        self.register_buffer('tensor1', torch.tensor([2.0]))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, a, b, c):
        x = self.add1(a, torch.tensor([2.0]))
        y = self.add2(self.tensor1, b)
        z = self.mul(c, x)

        x = self.relu1(x)
        y = self.relu2(y)
        z = self.relu3(z)
        return x, y, z


class HalfFloatTestModel(nn.Module):
    def __init__(self):
        super(HalfFloatTestModel, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class ModelWithConstantQuantization(torch.nn.Module):
    def __init__(self):
        super(ModelWithConstantQuantization, self).__init__()
        self.relu = torch.nn.ReLU()
        self.add = aimet_modules.Add()
        self.add2 = aimet_modules.Add()
        self.add3 = aimet_modules.Add()
        self.tensor1 = 1.0
        self.register_buffer('tensor2', torch.tensor([2.0]))

    def forward(self, inp):
        x = self.relu(inp)
        x = self.add(x, self.tensor1)
        x = self.add2(self.tensor2, x)
        x = self.add3(x, torch.tensor([1.0, 2.0]))
        return x


class TestQuantizationSimStaticGrad:
    def test_is_leaf_module_positive(self):
        """With an actual leaf module"""
        conv1 = nn.Conv2d(1, 10, 5)
        assert utils.is_leaf_module(conv1)

    # -------------------------------------------
    def test_is_leaf_module_negative(self):
        """With a non-leaf module"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        net = Net()
        model = net.to(torch.device('cpu'))

        assert not utils.is_leaf_module(model)

    # -------------------------------------------------------------
    def test_is_quantizable_module_positive(self):
        """With a quantizable module"""
        conv1 = nn.Conv2d(1, 10, 5)
        assert QuantizationSimModel._is_quantizable_module(conv1)

    # -------------------------------------------------------------
    def test_is_quantizable_module_negative(self):
        """With a non-quantizable module"""
        conv1 = StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8, round_mode='nearest',
                                       quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       data_type=QuantizationDataType.int)
        assert not QuantizationSimModel._is_quantizable_module(conv1)

    # ------------------------------------------------------------
    def verify_quantization_wrappers(self, original_model, quantized_model,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced):
        """Test utility to determine if quantization wrappers were added correctly"""

        # All leaf modules in the original model
        orig_modules = [(name, module) for name, module in original_model.named_modules()
                        if len(list(module.modules())) == 1]

        # All QcQuantized modules in the quantized model
        quant_modules = [(name, module) for name, module in quantized_model.named_modules()
                         if isinstance(module, QcQuantizeWrapper)]

        for i, orig_mod_tuple in enumerate(orig_modules):
            quant_mod_tuple = quant_modules[i]

            # If the original model has any QcQuantizeWrapper node, skip the checks
            if '_module_to_wrap' in orig_mod_tuple[0]:
                continue

            # Checks --------
            # Modules should be in the same order as before
            assert orig_mod_tuple[0] == quant_mod_tuple[0], "Quantized model has a incorrectly named module"

            if quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]:
                # For every leaf module in the first list, there is a corresponding QcQuantized model in the second list
                assert str(type(quant_mod_tuple[1]).__name__) == 'StaticGridQuantWrapper'

            # Each QcQuantized model has 1 child, that is the same type as the corresponding module in the original list
            assert len(list(quant_mod_tuple[1].modules())) == 2
            child = list(quant_mod_tuple[1].modules())[1]
            logger.debug("{} -> {}".format(type(child), type(orig_mod_tuple[1])))
            assert type(child) == type(orig_mod_tuple[1])

    # --------------------------------------------------------
    def test_add_quantization_wrappers_one_deep(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = Net()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # ------------------------------------------------------
    def test_add_quantization_wrappers_with_preexisting_quantization_layers(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                    round_mode='stochastic',
                                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                    data_type=QuantizationDataType.int)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12), in_place=True)

        # Add wrappers again, expect to be a nop
        sim._add_quantization_wrappers(model, num_inout_tensors={}, default_data_type=QuantizationDataType.int)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_two_deep(self):
        """With a one-deep model"""

        class SubNet(nn.Module):
            def __init__(self):
                super(SubNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = SubNet()
                self.fc1 = nn.Linear(320, 50)
                self.SubNet2 = SubNet()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequentials(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        print(model)
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))
        print(sim.model)
        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequential_two_deep(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Sequential(
                        nn.Conv2d(1, 10, 5),
                        nn.ReLU(),
                        nn.Conv2d(1, 10, 5)
                    ),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist(self):
        """With a one-deep model using ModuleList"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                             StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                                    round_mode='nearest',
                                                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                    data_type=QuantizationDataType.int)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_two_deep(self):
        """With a two-deep model using ModuleList"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
                self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                                  nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                                  StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8,
                                                                         activation_bw=8,
                                                                         round_mode='nearest',
                                                                         quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                         data_type=QuantizationDataType.int)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        print(model)
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12))
        print(sim.model)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_with_layers_to_ignore(self):
        """With a two-deep model using ModuleList and layers_to_ignore"""

        model = ModuleListModel()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12))
        layers_to_exclude = [sim.model.layers_deep[1], sim.model.layers_deep[3], sim.model.layers_deep[5]]
        sim.exclude_layers_from_quantization(layers_to_exclude)
        print(sim.model)

        assert isinstance(sim.model.layers[0]._module_to_wrap, nn.Linear)
        assert isinstance(sim.model.layers[1]._module_to_wrap, nn.Linear)
        assert isinstance(sim.model.layers[2]._module_to_wrap, nn.Conv2d)

        assert isinstance(sim.model.layers_deep[0][0]._module_to_wrap, nn.BatchNorm2d)
        assert isinstance(sim.model.layers_deep[0][1]._module_to_wrap, nn.ReLU)

        # layer ignored, so no QcQuantizeWrapper wrapper
        assert isinstance(sim.model.layers_deep[1], nn.Linear)
        assert isinstance(sim.model.layers_deep[2]._module_to_wrap, nn.Linear)

        # layer ignored, so no QcQuantizeWrapper wrapper
        assert isinstance(sim.model.layers_deep[3], nn.Conv2d)

        # non leaf layer specified, check that all submodules had wrappers removed
        assert isinstance(sim.model.layers_deep[5][0], nn.MaxPool2d)
        assert isinstance(sim.model.layers_deep[5][1], nn.PReLU)

        assert len(sim._excluded_layer_names) == 4
        assert 'layers_deep.1' in sim._excluded_layer_names
        assert 'layers_deep.3' in sim._excluded_layer_names
        assert 'layers_deep.5.0' in sim._excluded_layer_names
        assert 'layers_deep.5.1' in sim._excluded_layer_names

        with tempfile.TemporaryDirectory() as tmpdir, pytest.warns(DeprecationWarning):
            sim.export(tmpdir, 'modulelist_with_layers_to_ignore', dummy_input=torch.rand(1, 3, 12, 12))
            with open(os.path.join(tmpdir, "modulelist_with_layers_to_ignore.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)

            assert 'layers_deep.1' in encodings['excluded_layers']
            assert 'layers_deep.3' in encodings['excluded_layers']
            assert 'layers_deep.5.0' in encodings['excluded_layers']
            assert 'layers_deep.5.1' in encodings['excluded_layers']
            assert len(encodings['excluded_layers']) == 4

    # -------------------------------------------
    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)

        sim.model.conv1_a.param_quantizers['weight'].use_symmetric_encodings = True
        sim.model.conv1_a.param_quantizers['weight'].use_strict_symmetric = True

        # Quantize
        sim.compute_encodings(forward_pass, None)
        model(*dummy_input)

        # save encodings
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, 'two_input_model', dummy_input)

    # -------------------------------------------
    def test_model_with_two_inputs_fp16(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                   default_data_type=QuantizationDataType.float)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmpdir:
        # save encodings
            sim.export(tmpdir, 'two_input_model_fp16', dummy_input)
            encoding_file_path_pytorch = os.path.join(tmpdir, 'two_input_model_fp16' + '_torch' + '.encodings')
            load_encodings_to_sim(sim, encoding_file_path_pytorch)

            layer = sim.model.conv1_a
            if isinstance(layer, QcQuantizeWrapper):
                for input_quantizer in layer.input_quantizers:
                    if input_quantizer.enabled:
                        assert input_quantizer.encoding is None
                        assert input_quantizer.data_type is QuantizationDataType.float
                for output_quantizer in layer.output_quantizers:
                    if output_quantizer.enabled:
                        assert output_quantizer.encoding is None
                        assert output_quantizer.data_type is QuantizationDataType.float
                for name in layer.param_quantizers:
                    if layer.param_quantizers[name].enabled:
                        assert layer.param_quantizers[name].encoding is None
                        assert layer.param_quantizers[name].data_type is QuantizationDataType.float

    # -------------------------------------------
    def test_model_with_two_inputs_one_to_add(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        model = ModelWithTwoInputsOneToAdd()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        assert 2 == len(sim.model.add.input_quantizers)
        assert not sim.model.add.input_quantizers[0].enabled
        assert sim.model.add.input_quantizers[1].enabled

        # Quantize
        sim.compute_encodings(forward_pass, None)
        print(sim)

        with tempfile.TemporaryDirectory() as tmpdir:
            # save encodings
            sim.export(tmpdir, 'two_input_model_one_with_add', dummy_input)
            onnx_model = onnx.load(os.path.join(tmpdir, 'two_input_model_one_with_add.onnx'))
            for node in onnx_model.graph.node:
                if node.name == 'add':
                    break
            assert 2 == len(node.input)
            model_input_tensor = node.input[1]

            with open(os.path.join(tmpdir, "two_input_model_one_with_add.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)

            assert model_input_tensor in encodings['activation_encodings']
            activation_enc = encodings['activation_encodings'][model_input_tensor]
            assert isinstance(activation_enc[0]['offset'], int)
            param_enc = encodings["param_encodings"]["conv1_a.weight"]
            assert isinstance(param_enc[0]['offset'], int)

    def test_export_unified_encoding_format(self):
        """ test export functionality on ResNet18 """

        resnet18 = models.resnet18()
        resnet18.eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet18, dummy_input=dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'resnet18', dummy_input)
            with open(os.path.join(tmp_dir, 'resnet18.encodings')) as json_file:
                encoding_data = json.load(json_file)
                print(encoding_data)

            activation_keys = list(encoding_data["activation_encodings"].keys())

            assert len(activation_keys) == 24
            assert isinstance(encoding_data["activation_encodings"][activation_keys[0]], list)

            param_keys = list(encoding_data["param_encodings"].keys())
            assert "conv1.weight" in param_keys
            assert isinstance(encoding_data["param_encodings"]["conv1.weight"], list)

    def test_export_with_quantizer_args(self):
        """ test export functionality on ResNet18 """

        resnet18 = models.resnet18()
        resnet18.eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet18, dummy_input=dummy_input, default_output_bw=16, default_param_bw=16,
                                   quant_scheme=QuantScheme.post_training_tf)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'resnet18_with_quant_args', dummy_input)
            with open(os.path.join(tmp_dir, 'resnet18_with_quant_args.encodings')) as json_file:
                encoding_data = json.load(json_file)

            assert "quantizer_args" in encoding_data
            quantizer_args = encoding_data["quantizer_args"]
            assert quantizer_args["activation_bitwidth"] == 16
            assert quantizer_args["param_bitwidth"] == 16
            assert not quantizer_args["per_channel_quantization"]
            assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf.name
            assert quantizer_args["dtype"] == "int"
            assert "is_symmetric" in quantizer_args

    def test_export_to_torch_script(self):
        """ test export functionality on ResNet18 """
        resnet50 = models.resnet50()
        resnet50.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet50, dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as temp_dir:

            sim.export(temp_dir, 'resnet50', dummy_input, export_to_torchscript=True)
            with open(os.path.join(temp_dir, 'resnet50.encodings')) as json_file:
                encoding_data = json.load(json_file)

            activation_keys = list(encoding_data["activation_encodings"].keys())
            assert isinstance(encoding_data["activation_encodings"][activation_keys[0]], list)

            param_keys = list(encoding_data["param_encodings"].keys())
            assert param_keys[0] == "conv1.weight"
            assert isinstance(encoding_data["param_encodings"]["conv1.weight"], list)

    def test_export_to_onnx(self):
        """Exporting encodings and model"""
        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()
        sim = QuantizationSimModel(model, dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.model.conv1_a.param_quantizers['weight'].encoding.max = 10
        sim.model.conv1_a.output_quantizers[0].encoding.max = 30

        with tempfile.TemporaryDirectory() as temp_dir:

            # save encodings
            sim.export(temp_dir, 'two_input_model', dummy_input)

            # check the encodings
            with open(os.path.join(temp_dir, 'two_input_model.encodings'), 'r') as fp:
                encodings = json.load(fp)

                activation_encodings = encodings['activation_encodings']
                param_encodings = encodings['param_encodings']
                assert 16 == len(activation_encodings)
                assert 7 == len(param_encodings['conv1_a.weight'][0])
                assert 10 == param_encodings['conv1_a.weight'][0]['max']

            # check the exported model
            loaded_model = torch.load(os.path.join(temp_dir, 'two_input_model.pth'))
            loaded_model(torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))

    def test_no_fine_tuning_tf_enhanced(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(sim.model, None)

    @pytest.mark.cuda
    def test_multi_gpu_qat(self):
        """"""
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        class OneConvLayerModel(nn.Module):
            def __init__(self):
                super(OneConvLayerModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=2)

            def forward(self, x):
                x = self.conv1(x)
                return x

        model = OneConvLayerModel().to('cuda:0')
        model = model.to('cuda:0')

        dummy_input = torch.ones(2, 1, 3, 3).to('cuda:0')
        dummy_input[1] += 1

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=dummy_input)

        original_weight = sim.model.conv1._module_to_wrap.weight.clone().detach()

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1._module_to_wrap.weight.grad.clone().detach()
        sim.model.conv1._module_to_wrap.weight.grad = None

        sim.model = torch.nn.DataParallel(sim.model)

        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))

        weight = sim.model.module.conv1._module_to_wrap.weight.clone().detach()

        assert torch.allclose(output_multi_gpu, output_single_gpu)
        assert torch.allclose(original_weight, weight)

        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        assert torch.allclose(sim.model.module.conv1._module_to_wrap.weight.grad, grad_single_gpu)

    # -------------------------------------------
    def test_input_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizers[0].enabled = False
                module.input_quantizers[0].enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert not sim.model.conv1.output_quantizers[0].encoding
        assert sim.model.conv1.input_quantizers[0].encoding

        print(sim.model.conv1.input_quantizers[0])
        print(sim.model.conv1.output_quantizers[0])


    def test_inputs_shared_constant_intermediate_quantization(self):
        """"""
        model = ModelInputsSharedConstantIntermediate()

        dummy_input = (torch.randn(1, 10, 10, 10), torch.randn(1, 10, 10, 10), torch.randn(1, 10, 10, 10))

        def forward_pass(model, args):
            model(*dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input)

        sim.compute_encodings(forward_pass, None)

        # check mul's first input quantizer is real input(enable) , second input is intermediate (disable)
        assert sim.model.mul.input_quantizers[0].enabled
        assert not sim.model.mul.input_quantizers[1].enabled

        assert not sim.model.add1.input_quantizers[0].is_const
        assert sim.model.add1.input_quantizers[1].is_const
        assert sim.model.add2.input_quantizers[0].is_const
        assert not sim.model.add2.input_quantizers[1].is_const

        # save encodings
        input_names = ['a', 'b', 'c']

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'model_inputs_shared_constant_intermediate',
                       dummy_input,
                       onnx_export_args=OnnxExportApiArgs(input_names=input_names))
            with open(os.path.join(tmp_dir, "model_inputs_shared_constant_intermediate.encodings"), "r") as encodings_file:
                activation_encoding_tensors = set(json.load(encodings_file)['activation_encodings'].keys())
                assert set(input_names).issubset(activation_encoding_tensors)

    def test_constant_quantization(self):
        model = ModelWithConstantQuantization()
        dummy_input = torch.rand(1, 2)
        sim = QuantizationSimModel(model, dummy_input)
        assert not sim.model.add.input_quantizers[0].enabled
        assert sim.model.add.input_quantizers[1].enabled
        assert sim.model.add2.input_quantizers[0].enabled
        assert not sim.model.add2.input_quantizers[1].enabled
        assert not sim.model.add3.input_quantizers[0].enabled
        assert sim.model.add3.input_quantizers[1].enabled

        sim.compute_encodings(lambda m, _: m(dummy_input), None)
        # As add and add2 use constants with numel 1, we expect the quantizers to not have any encoding stats and thus
        # be disabled after compute encodings.
        assert not sim.model.add.input_quantizers[1].enabled
        assert not sim.model.add2.input_quantizers[1].enabled
        assert sim.model.add3.input_quantizers[1].enabled

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'model_with_constant_quantization', dummy_input)
            with open(os.path.join(tmp_dir, "model_with_constant_quantization.encodings"), "r") as encodings_file:
                activation_encoding_tensors = set(json.load(encodings_file)['activation_encodings'].keys())
                assert len(activation_encoding_tensors) == 6

    # -------------------------------------------
    def test_input_and_output_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizers[0].enabled = True
                module.input_quantizers[0].enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert sim.model.conv1.output_quantizers[0].encoding
        assert sim.model.conv1.input_quantizers[0].encoding

        print(sim.model.conv1.input_quantizers[0])
        print(sim.model.conv1.output_quantizers[0])

    def test_quantizing_models_with_funtional_add_ops(self):
        """
        Testing models with add functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = ya + yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_module_add_ops(self):
        """
        Testing models with add functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)
                self.add1 = aimet_modules.Add()
                self.add2 = aimet_modules.Add()

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = self.add1(ya, yb)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = self.add2(ya, yb)
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert not sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.add1.output_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_add_followed_by_split(self):
        """
        Testing models with add functional ops followed by a split
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv4a.input_quantizers[0].enabled
        assert sim.model.conv4b.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_add_followed_by_add(self):
        """
        Testing models with add functional ops followed by a split and then another add.
        This is similar to the resnet architecture where there are no ops on the residual connection
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5, padding=2)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = x
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv4a.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_model_with_input_leading_to_add(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.conv1(x1)
                x = x + x2
                return self.softmax(x)

        model = Net().eval()
        _ = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                 dummy_input=(torch.rand(1, 3, 28, 28), torch.rand(1, 3, 24, 24)))

    def test_quantizing_models_with_mul_ops(self):
        """
        Testing models with elementwise multiply functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya * yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya * yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_div_ops(self):
        """
        Testing models with elementwise division functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya / yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya / yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_concat_ops(self):
        """
        Testing models with concat functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 10, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 10, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 10, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 10, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_no_finetuning_tf(self):
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 28, 28))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

    def test_per_layer_bitwidths(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 28, 28))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)
        sim.model.conv1.set_output_bw(16)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(model, None)

        assert 16 == sim.model.conv1.output_quantizers[0].bitwidth
        assert 8 == sim.model.conv2.output_quantizers[0].bitwidth

    # -------------------------------------------
    def test_with_standalone_ops(self):

        model = ModelWithStandaloneOps()
        dummy_input = torch.rand(1, 1, 28, 28)

        sim = QuantizationSimModel(model=model, dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

        # Save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, "encodings_with_standalone_ops", dummy_input)
            with open(os.path.join(tmp_dir, 'encodings_with_standalone_ops.encodings')) as json_file:
                encoding_data = json.load(json_file)
            # in onnx definition tensor 16 is output of Reshape, to be ignored
            assert "32" not in encoding_data["activation_encodings"].keys()

    # -------------------------------------------------------------------------------
    def test_layers_to_ignore(self):
        """ Test the  capability to skip quantizing the layers specified by the user"""

        model = SmallMnist()

        dummy_input=torch.rand(1, 1, 28, 28)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        layers_to_ignore = [sim.model.conv1, sim.model.fc2]
        sim.exclude_layers_from_quantization(layers_to_ignore)

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # Check
        assert isinstance(sim.model.conv1, nn.Conv2d)
        assert not isinstance(sim.model.conv2, nn.Conv2d)
        assert isinstance(sim.model.fc2, nn.Linear)

        # export and check encodings file has excluded layers listed as string
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'excluded_layers', dummy_input, propagate_encodings=True)

            with open(os.path.join(tmp_dir, 'excluded_layers.encodings')) as f:
                encodings = json.load(f)
                assert 2 == len(encodings['excluded_layers'])

    def check_quant_params(self, model_layer, loaded_model_layer, check_weights):
        output_encoding1 = model_layer.output_quantizers[0].encoding
        output_encoding2 = loaded_model_layer.output_quantizers[0].encoding

        assert model_layer.output_quantizers[0].bitwidth == loaded_model_layer.output_quantizers[0].bitwidth
        assert np.allclose(output_encoding1.max, output_encoding2.max)
        assert np.allclose(output_encoding1.min, output_encoding2.min)
        assert np.allclose(output_encoding1.delta, output_encoding2.delta)
        assert np.allclose(output_encoding1.offset, output_encoding2.offset)

        if model_layer.param_quantizers:
            assert next(iter(model_layer.param_quantizers.values())).bitwidth == \
                   next(iter(loaded_model_layer.param_quantizers.values())).bitwidth

        assert model_layer.output_quantizers[0].round_mode == loaded_model_layer.output_quantizers[0].round_mode
        assert model_layer.output_quantizers[0].quant_scheme == loaded_model_layer.output_quantizers[0].quant_scheme

        if check_weights:
            assert np.allclose(model_layer._module_to_wrap.weight.detach().numpy(),
                               loaded_model_layer._module_to_wrap.weight.detach().numpy())

    def test_save_and_load(self):

        model = ModelWithStandaloneOps()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(32, 1, 28, 28))

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)

        # Run some inferences - mimic using a forward pass
        sim.model.eval()
        dummy_input = torch.randn((32, 1, 28, 28))
        output_before_save = sim.model(dummy_input)

        # Save quantized model
        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(sim.model, os.path.join(tmp_dir, 'xx'))

            loaded_model = torch.load(os.path.join(tmp_dir, 'xx'))
            loaded_model.eval()
            output_after_load = loaded_model(dummy_input)

            self.check_quant_params(sim.model.conv1, loaded_model.conv1, True)
            self.check_quant_params(sim.model.conv2, loaded_model.conv2, True)
            self.check_quant_params(sim.model.fc2, loaded_model.fc2, True)

            assert np.allclose(output_before_save.detach().numpy(),
                               output_after_load.detach().numpy())

    def test_ste_gradient_math_tensors(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = [-0.25, -0.25]
        c_enc_max = [1.0, 1.0]
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]])
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_1, grad_out_1)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]])
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]])
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_2, grad_out_2)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]])
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_3, grad_out_3)

    @pytest.mark.cuda
    def test_ste_gradient_math_tensors_cuda(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = [-0.25, -0.25]
        c_enc_max = [1.0, 1.0]
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]]).cuda()

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]]).cuda()
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]]).cuda()
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_1, grad_out_1)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]]).cuda()
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]]).cuda()
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_2, grad_out_2)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]]).cuda()
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]]).cuda()
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_3, grad_out_3)

    def test_ste_gradient_math(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = -0.25
        c_enc_max = 1.0
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]])
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]])
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]])
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]])
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)

        assert np.allclose(expected_grad_1, grad_out_1)
        assert np.allclose(expected_grad_2, grad_out_2)
        assert np.allclose(expected_grad_3, grad_out_3)

    def test_changing_param_quantizer_settings(self):
        """ Test that changing param quantizer settings takes effect after computing encodings is run """
        torch.random.manual_seed(10)
        model = SmallMnist()

        # Skew weights of conv1
        old_weight = model.conv1.weight.detach().clone()
        model.conv1.weight = torch.nn.Parameter(old_weight + .5 * torch.abs(torch.min(old_weight)), requires_grad=False)

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))

        # Check that no encoding is present for param quantizer
        assert not sim.model.conv1.param_quantizers['weight'].encoding

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)
        asym_min = sim.model.conv1.param_quantizers['weight'].encoding.min
        asym_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        assert 8 == sim.model.conv1.param_quantizers['weight'].encoding.bw

        # Check that offset is still symmetric
        assert sim.model.conv1.param_quantizers['weight'].encoding.offset == -128.0

        # Change param quantizer to symmetric and new bitwidth
        sim.model.conv1.param_quantizers['weight'].use_symmetric_encodings = False
        sim.model.conv1.param_quantizers['weight'].bitwidth = 4
        sim.compute_encodings(dummy_forward_pass, None)
        sym_min = sim.model.conv1.param_quantizers['weight'].encoding.min
        sym_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        assert 4 == sim.model.conv1.param_quantizers['weight'].encoding.bw

        # Check that offset is not relatively symmetric
        assert not sim.model.conv1.param_quantizers['weight'].encoding.offset in [-127, -128]

        # Check that mins and maxes have been recomputed
        assert not np.allclose(asym_min, sym_min)
        assert not np.allclose(asym_max, sym_max)

    def test_compute_encodings_on_subset_of_modules(self):
        """ Test that computing encodings on a subset of modules causes remaining quantized modules to be set to
            passThrough mode. """

        def dummy_forward_pass(model, _):
            conv1_out = model.conv1(torch.randn((1, 1, 28, 28)))
            relu1_out = model.relu1(conv1_out)

        model = SmallMnist()
        model.eval()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))
        sim.compute_encodings(dummy_forward_pass, None)
        for name, module in sim.model.named_modules():
            if isinstance(module, StaticGridQuantWrapper):
                assert QcQuantizeOpMode.ACTIVE == module._mode
                if name == 'relu1':
                    assert module.output_quantizers[0].enabled
                elif name in ['conv2', 'conv2_drop', 'relu2', 'relu3', 'dropout', 'fc2', 'log_softmax']:
                    assert not module.output_quantizers[0].enabled

    def test_connected_graph_is_none(self):
        """ Test that an assertion is thrown when connected graph is not able to be built. """

        def raise_trace_error(_self, _model, _inputs):
            raise torch.jit.TracingCheckError(None, None)

        model = SmallMnist()
        model.eval()
        with unittest.mock.patch.object(ConnectedGraph, '__init__', raise_trace_error):
            with unittest.mock.patch.object(ConnectedGraph, '__del__', lambda _self: None):
                with pytest.raises(AssertionError):
                    _ = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))

    def test_rnn_quantization(self):
        """ Test quantizing a model with rnn layer """
        model = SingleLayerRNNModel()
        dummy_input = torch.randn(10, 1, 3)

        sim = QuantizationSimModel(model, dummy_input)
        assert isinstance(sim.model.rnn, QcQuantizeRecurrent)

    def test_quantizing_qc_quantize_module(self):
        """ Test that qc_quantize_module is identified as not quantizable """
        qc_quantize_module = QcQuantizeRecurrent(torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1), weight_bw=16,
                                                 activation_bw=16, quant_scheme=QuantScheme.post_training_tf,
                                                 round_mode='nearest', data_type=QuantizationDataType.int)
        assert not QuantizationSimModel._is_quantizable_module(qc_quantize_module)

    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    @pytest.mark.parametrize('rounding_dtype', [torch.float32, torch.float64])
    def test_exported_weight(self, quant_scheme, rounding_dtype):
        """
        Test to check if the exported weight remains unchanged after quantization,
        regardless of the rounding method used.
        """
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.randn(2, 64, 8, 8)
        param_bitwidth = 8

        # Set model weight to odd numbers so that quantized weights fall between rounding border
        with torch.no_grad():
            model.conv.weight.data = torch.randint(-127, 127, model.conv.weight.shape, dtype=torch.float32) * 2 + 1

        sim = QuantizationSimModel(model, dummy_input, default_param_bw=param_bitwidth, quant_scheme=quant_scheme)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        # Adjust delta of the weight quantizer so that quantized weights fall between rounding border
        encoding = libpymo.TfEncoding()
        encoding.bw = 8
        encoding.max = 254.0
        encoding.min = -254.0
        encoding.offset = -254.0
        encoding.delta = 2.0
        sim.model.conv.param_quantizers['weight'].encoding = encoding

        with tempfile.TemporaryDirectory() as tmp_dir:
            conv_weights = []

            # Gather weights from exported .pth file
            sim.export(tmp_dir, 'model_for_weight_export', dummy_input)
            exported_model = torch.load(os.path.join(tmp_dir, 'model_for_weight_export.pth'))
            conv_weights.append(exported_model.conv.weight.detach())

            # Gather weights from exported .onnx file
            onnx_model = onnx.load(os.path.join(tmp_dir, 'model_for_weight_export.onnx'))
            for tensor in onnx_model.graph.initializer:
                if tensor.name == 'conv.weight':
                    conv_weights.append(torch.tensor(onnx.numpy_helper.to_array(tensor)))
                    break
            else:
                assert False, "Cannot find conv weight inside ONNX model"
            
            # Check exported weight remains unchanged after quantization,
            # regardless of the rounding method used.
            for conv_weight in conv_weights:
                delta = sim.model.conv.param_quantizers['weight'].encoding.delta
                assert torch.allclose(conv_weight, model.conv.weight.data, atol=delta)

                default_quant_result = torch.round(conv_weight / delta)
                conv_weight = conv_weight.to(rounding_dtype)
                delta = torch.tensor(delta, dtype=rounding_dtype)
                assert torch.equal(default_quant_result,
                                   torch.trunc(conv_weight / delta + torch.sign(conv_weight) * 0.5).to(torch.float32))


    def test_export_recurrent_model(self):
        """ Test export functionality with recurrent models """
        # models = [TwoLayerBidirectionaRNNModel(), TwoLayerBidirectionalLSTMModel(), TwoLayerBidirectionalGRUModel()]
        models = [TwoLayerBidirectionalLSTMModel()]
        dummy_input = torch.randn(10, 1, 3)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        for model in models:
            sim = QuantizationSimModel(model, dummy_input)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Edit part of weights tensor to compare with original model before and after removal of quantize module
            with torch.no_grad():
                sim.model.recurrent.weight_ih_l0[0][0] = 1
            edited_weight = sim.model.recurrent.weight_ih_l0.detach().clone()

            # Check that edited weight is different than original weight in module_to_quantize
            assert not torch.equal(edited_weight, sim.model.recurrent.module_to_quantize.weight_ih_l0)

            with tempfile.TemporaryDirectory() as tmp_dir:
                sim.export(tmp_dir, 'recurrent_save', dummy_input)
                exported_model = torch.load(os.path.join(tmp_dir, 'recurrent_save.pth'))

                # Check that weight from quantized module was copied to original module successfully
                assert isinstance(exported_model.recurrent, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
                assert torch.equal(edited_weight, exported_model.recurrent.weight_ih_l0)

                with open(os.path.join(tmp_dir, 'recurrent_save.encodings')) as f:
                    encodings = json.load(f)
                    # verifying the encoding against default eAI HW cfg
                    # activation encoding (input only w/o cell state) -- x_l0, h_l0, x_l1 & h_l1
                    assert 8 == len(encodings['activation_encodings'])
                    # param encoding (weight only w/o bias)  -- W_l0, R_l0, W_l1 & R_l1
                    assert 4 == len(encodings['param_encodings'])

    def test_export_dict_input_output(self):
        """ test export functionality on dictionary input and output """

        # Add an empty dictionary as the last element to not treat as named arguments.
        # see torch.onnx.export() API for more details.
        dummy_input = (
            {'a': torch.randn(1, 10, 10, 10),
             'b': torch.randn(1, 10, 10, 10),
             'c': torch.randn(1, 10, 10, 10)
             }, {}
        )

        model = InputOutputDictModel()

        def forward_pass(model, dummy_input):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input[0])
        sim.model.mul1.output_quantizers[0].enabled = True
        sim.model.mul2.output_quantizers[0].enabled = True
        sim.model.mul3.output_quantizers[0].enabled = True

        # Quantize
        sim.compute_encodings(forward_pass, dummy_input[0])

        o_names = ['ab', 'bc', 'ca']
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'dict_input_output_model', dummy_input,
                       onnx_export_args=OnnxExportApiArgs(input_names=list(dummy_input[0].keys()),
                                                          output_names=o_names,
                                                          opset_version=12
                                                          ))
            with open(os.path.join(tmp_dir, 'dict_input_output_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                print(encoding_data)

            onnx_model = onnx.load(os.path.join(tmp_dir, 'dict_input_output_model.onnx'))
            for inp in onnx_model.graph.input:
                assert inp.name in ['a', 'b', 'c']
            for exp, act in zip(o_names, onnx_model.graph.output):
                assert exp == act.name
            for tensor_name in encoding_data["activation_encodings"].keys():
                assert tensor_name in o_names

    def test_compute_encoding_fp16(self):
        """
        Test encodings generated for fp16
        """
        dummy_input = {'a': torch.randn(1, 10, 10, 10),
                       'b': torch.randn(1, 10, 10, 10),
                       'c': torch.randn(1, 10, 10, 10)}

        model = InputOutputDictModel()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                   default_data_type=QuantizationDataType.float)

        quantizer = sim.model.mul1.input_quantizers[0]
        enc_dict = sim._create_encoding_dict(encoding=None, quantizer=quantizer, propagate_encodings=False)
        assert enc_dict['dtype'] == 'float'
        assert enc_dict['bitwidth'] == 16
        assert 'min' not in enc_dict
        assert 'max' not in enc_dict
        assert 'scale' not in enc_dict
        assert 'offset' not in enc_dict
        assert 'is_symmetric' not in enc_dict

    def test_mapping_encoding_for_torch_module_with_multiple_onnx_ops(self):
        """
         Test the input and output encoding map to input/output at subgraph level when atorch module generates
         multiple onnx ops i.e. a sub-graph
        """

        dummy_input = torch.randn(1, 4, 256, 512)
        model = SoftMaxAvgPoolModel()

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.model.sfmax.output_quantizers[0].enabled = True
        sim.model.sfmax.input_quantizers[0].enabled = True
        sim.model.avgpool.output_quantizers[0].enabled = True
        sim.model.avgpool.input_quantizers[0].enabled = True
        sim.compute_encodings(forward_pass, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'sfmaxavgpool_model', dummy_input)

            with open(os.path.join(tmp_dir, 'sfmaxavgpool_model.encodings')) as json_file:
                encoding_data = json.load(json_file)

            assert len(encoding_data["activation_encodings"]) == 3

    def test_encodings_propagation_simple_model(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module
        """
        model = PixelShuffleModel()
        dummy_input = torch.randn(1, 4, 8, 8)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        # verifying the encodings propagation works well while using QcQuantizeRecurrent or QcQuantizeWrapper
        sim = QuantizationSimModel(model, dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'encodings_propagation_false', dummy_input)
            with open(os.path.join(tmp_dir, 'encodings_propagation_false.encodings')) as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 2

            # Save encodings again - now with propagate encodings flag enabled
            sim.export(tmp_dir, 'encodings_propagation_true', dummy_input, propagate_encodings=True)
            with open(os.path.join(tmp_dir, 'encodings_propagation_true.encodings')) as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 2

            pretty_data = json.dumps(encodings, indent=2)
            print(pretty_data)

            # verifying the encodings propagation is disabled if output quantizers are disabled.
            sim = QuantizationSimModel(model, dummy_input)
            sim.model.ps.output_quantizers[0].enabled = False
            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Save encodings again - now with propagate encodings flag enabled
            sim.export(tmp_dir, 'encodings_propagation_quant_disabled', dummy_input, propagate_encodings=True)
            with open(os.path.join(tmp_dir, 'encodings_propagation_quant_disabled.encodings')) as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 1

    def test_encodings_propagation_lstm_model(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module for LSTM layers
        """
        model = TwoLayerBidirectionalLSTMModel()
        dummy_input = torch.randn(10, 1, 3)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        # verifying the encodings propagation works well while using QcQuantizeRecurrent or QcQuantizeWrapper
        sim = QuantizationSimModel(model, dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'encodings_propagation_false', dummy_input)
            with open(os.path.join(tmp_dir, 'encodings_propagation_false.encodings')) as f:
                encodings = json.load(f)
            assert len(encodings['activation_encodings']) == 8

            # Save encodings again - now with propagate encodings flag enabled
            sim.export(tmp_dir, 'encodings_propagation_true', dummy_input, propagate_encodings=True)
            with open(os.path.join(tmp_dir, 'encodings_propagation_true.encodings')) as f:
                encodings = json.load(f)['activation_encodings']
                # Only eight entry should have min, max, delta and offset, remaining entries should be propagated
                # with bitwidth and dtype.
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 8

            pretty_data = json.dumps(encodings, indent=2)
            print(pretty_data)

    def test_change_quant_scheme_tf_enhanced_to_tf(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, 5)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(forward_pass, None)
        assert sim.model.conv2.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf_enhanced

        tensor = torch.rand(1, 10, 24, 24)
        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 1000
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv2.output_quantizers[0].encoding.max < 1.0

        sim.model.conv2.output_quantizers[0].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(1, 10, 24, 24)
        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 1000
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv2.output_quantizers[0].encoding.max == pytest.approx(1000.0, rel=0.1)

    def test_change_quant_scheme_tf_enhanced_to_tf_per_channel(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, 5)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim1 = QuantizationSimModel(model, dummy_input)
        sim1.model.conv2.enable_per_channel_quantization()
        sim1.compute_encodings(forward_pass, None)
        assert sim1.model.conv2.param_quantizers['weight'].quant_scheme == QuantScheme.post_training_tf_enhanced

        tensor = torch.rand(20, 10, 5, 5)
        sim1.model.conv2.param_quantizers['weight'].reset_encoding_stats()
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 100
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        sim1.model.conv2.param_quantizers['weight'].compute_encoding()

        sim1.model.conv2.param_quantizers['weight'].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(20, 10, 5, 5)
        sim1.model.conv2.param_quantizers['weight'].reset_encoding_stats()
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 100
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        sim1.model.conv2.param_quantizers['weight'].compute_encoding()
        assert sim1.model.conv2.param_quantizers['weight'].encoding[0].max == pytest.approx(100.0, rel=0.1)

        # Scenario: we set the scheme before a call to compute_encodings
        sim2 = QuantizationSimModel(model, dummy_input)
        sim2.model.conv2.enable_per_channel_quantization()
        sim2.model.conv2.param_quantizers['weight'].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(20, 10, 5, 5)
        tensor[0, 0, 0, 0] = 100
        sim2.model.conv2._module_to_wrap.weight.data = tensor
        sim2.compute_encodings(forward_pass, None)
        assert sim2.model.conv2.param_quantizers['weight'].encoding[0].max == pytest.approx(100.0, rel=0.1)

    @pytest.mark.skipif(version.parse(torch.__version__) >= version.parse("2.1.2"),
                        reason="Results in RuntimeError when exporting, needs further debugging.")
    def test_conditional_export(self):
        """ Test exporting a model with conditional paths """
        model = SimpleConditional()
        inp = torch.randn(1, 3)
        true_tensor = torch.tensor([1])
        false_tensor = torch.tensor([0])

        def forward_callback(model, _):
            model(inp, true_tensor)
            model(inp, false_tensor)

        qsim = QuantizationSimModel(model, dummy_input=(inp, true_tensor))
        qsim.compute_encodings(forward_callback, forward_pass_callback_args=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            qsim._export_conditional(tmp_dir, 'simple_cond', dummy_input=(inp, false_tensor),
                                     forward_pass_callback=forward_callback, forward_pass_callback_args=None)

            with open(os.path.join(tmp_dir, 'simple_cond.encodings')) as f:
                encodings = json.load(f)
                # verifying the encoding against default eAI HW cfg
                # activation encodings -- input, linear1 out, prelu1 out, linear2 out, prelu2 out, softmax out
                assert 6 == len(encodings['activation_encodings'])
                # param encoding -- linear 1 & 2 weight & bias, prelu 1 & 2 weight
                assert 4 == len(encodings['param_encodings'])

    def test_export_prelu_encoding_and_check_load_encodings(self):
        """ Test that prelu weight is exported correctly """
        model = PreluModel()
        dummy_input = torch.rand(1, 3, 8, 8)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'prelu_model', dummy_input=dummy_input)
            with open(os.path.join(tmp_dir, 'prelu_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
            assert 'prelu.weight' in encoding_data['param_encodings'].keys()

            output = sim.model(copy.deepcopy(dummy_input))
            del sim

            sim = QuantizationSimModel(model, dummy_input=dummy_input)
            encoding_file_path_pytorch = os.path.join(tmp_dir, 'prelu_model' + '_torch' + '.encodings')
            load_encodings_to_sim(sim, encoding_file_path_pytorch)

            layer = sim.model.prelu
            if isinstance(layer, QcQuantizeWrapper):
                for input_quantizer in layer.input_quantizers:
                    if input_quantizer.enabled:
                        assert input_quantizer.encoding is not None
                for output_quantizer in layer.output_quantizers:
                    if output_quantizer.enabled:
                        assert output_quantizer.encoding is not None
                for name in layer.param_quantizers:
                    if layer.param_quantizers[name].enabled:
                        assert layer.param_quantizers[name].encoding is not None

            output1 = sim.model(copy.deepcopy(dummy_input))
            assert sum(output1.flatten() - output.flatten()) == 0.0

    def test_fetching_varaible_from_module(self):
        class Model(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

            def forward(self, inputs):
                x = self.conv(inputs)
                return x

        in_channels = out_channels = 10
        model = Model(in_channels, out_channels, 5)
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(forward_pass, None)

        assert hasattr(sim.model.conv, 'in_channels')
        assert getattr(sim.model.conv, 'in_channels') == 10
        assert hasattr(sim.model.conv, 'out_channels')
        assert getattr(sim.model.conv, 'out_channels') == 10

        del sim

    def test_nested_input(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.add = aimet_modules.Add()

            def forward(self, x, y):
                return self.add(x, y)

        model = Model()

        length = 8
        shape = (256, 256)
        inputs_a = [torch.rand(shape) for _ in range(length)]
        inputs_b = [torch.rand(shape) for _ in range(length)]

        sim = QuantizationSimModel(model, (inputs_a, inputs_b))
        # Need to set these manually since in the connected graph, the Add is a no-op with no inputs.
        # The list concatenation happens outside of the Add node.
        # This means that Add will not be identified as an input module.
        sim.model.add.input_quantizers[0].enabled = True
        sim.model.add.input_quantizers[1].enabled = True

        def forward_pass(model, args):
            model.eval()
            model(inputs_a, inputs_b)
        sim.compute_encodings(forward_pass, None)

        assert sim.model.add.input_quantizers[0].encoding is not None
        assert sim.model.add.input_quantizers[1].encoding is not None

    def test_has_valid_encodings(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.relu1 = torch.nn.ReLU()
                self.conv = torch.nn.Conv2d(3, 8, (2, 2))
                self.relu2 = torch.nn.ReLU()
                self.unused_module = torch.nn.PReLU()

            def forward(self, *inputs):
                x = self.relu1(inputs[0])
                x = self.conv(x)
                x = self.relu2(x)
                return x

        model = Model()
        model.eval()
        qsim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 8, 8))
        modules = [qsim.model.relu1, qsim.model.conv, qsim.model.relu2, qsim.model.unused_module]
        for m in modules:
            assert not has_valid_encodings(m)
        qsim.compute_encodings(lambda m, _: m(torch.randn(1, 3, 8, 8)), None)
        for m in modules:
            if m == qsim.model.unused_module:
                assert not has_valid_encodings(m)
            else:
                assert has_valid_encodings(m)

    def test_save_model_with_embedded_quantization_nodes(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(*dummy_input)

            model = ModelWithTwoInputs()
            sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme = _quant_scheme)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Save model
            with tempfile.TemporaryDirectory() as tempdir:
                sim.export(tempdir, 'two_input_model', dummy_input,
                           onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                           use_embedded_encodings=True)
                onnx_model = onnx.load(os.path.join(tempdir, 'two_input_model' + '_embedded' + '.onnx'))
                onnx_type = set()
                for node in onnx_model.graph.node:
                    onnx_type.add(node.op_type)
                assert('QuantizeLinear' in onnx_type)
                assert('DequantizeLinear' in onnx_type)

    def test_save_model_with_embedded_quantization_nodes_fp16(self):
        """Model with more than 1 input"""

        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input=(torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(*dummy_input)

            model = ModelWithTwoInputs()

            sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                       quant_scheme = _quant_scheme, default_data_type=QuantizationDataType.float)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # save model
            with tempfile.TemporaryDirectory() as tempdir:
                sim.export(tempdir, 'two_input_model_fp16', dummy_input,
                           onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                           use_embedded_encodings=True)

                onnx_model = onnx.load(os.path.join(tempdir, 'two_input_model_fp16' + '_embedded' + '.onnx'))
                onnx_type = set()
                for node in onnx_model.graph.node:
                    onnx_type.add(node.op_type)
                assert('Cast' in onnx_type)

    def test_save_model_with_embedded_quantization_nodes_per_channel(self):
        """Model with more than 1 input"""

        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(*dummy_input)

            model = ModelWithTwoInputs()

            sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme = _quant_scheme)
            for _, wrapper in sim.quant_wrappers():
                wrapper.enable_per_channel_quantization()

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Export model with opset_vesrion 13
            with tempfile.TemporaryDirectory() as tempdir:
                sim.export(tempdir, 'two_input_model_perchannel', dummy_input,
                           onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=13)),
                           use_embedded_encodings=True)
                onnx_model = onnx.load(os.path.join(tempdir, 'two_input_model_perchannel' + '_embedded' + '.onnx'))
                onnx_type = set()
                for node in onnx_model.graph.node:
                    onnx_type.add(node.op_type)
                assert('QuantizeLinear' in onnx_type)
                assert('DequantizeLinear' in onnx_type)

    def test_save_model_with_embedded_quantization_nodes_using_torch_script(self):
        """Test export onnx model with embedded torch native quantization nodes using torch script"""

        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input = torch.rand(32, 1, 28, 28)

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(dummy_input)

            model = SmallMnist()
            sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme = _quant_scheme)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Save model
            with tempfile.TemporaryDirectory() as tempdir:
                QuantizationSimModel.save_model_with_embedded_quantization_nodes(sim.model, tempdir, 'two_input_model', dummy_input, export_to_torchscript = True)
                assert(os.path.exists(os.path.join(tempdir, 'two_input_model' + '_embedded' + '.torchscript.pth')))

    def test_native_pytorch_quantization_nodes_pertensor(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        torch.manual_seed(10)
        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input = torch.rand(32, 1, 28, 28)

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    return model(dummy_input)

            model = SmallMnist()
            sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme = _quant_scheme)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Generate model with native pytorch quantization nodes
            quant_sim_model = copy.deepcopy(sim.model)

            device = utils.get_device(quant_sim_model)
            QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device)

            # Inference AIMET quantization nodes
            aimet_res = forward_pass(sim.model, None)
            # Inference Native torch quantization nodes
            torch_res = forward_pass(quant_sim_model, None)
            assert torch.allclose(aimet_res, torch_res, rtol=1e-2)

    def test_native_pytorch_quantization_nodes_perchannel(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        torch.manual_seed(10)
        for _quant_scheme in [QuantScheme.post_training_tf,
                              QuantScheme.training_range_learning_with_tf_init]:
            dummy_input = torch.rand(32, 1, 28, 28)

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    return model(dummy_input)

            model = SmallMnist()
            sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme = _quant_scheme)
            for _, wrapper in sim.quant_wrappers():
                wrapper.enable_per_channel_quantization()

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Generate model with native pytorch quantization nodes
            quant_sim_model = copy.deepcopy(sim.model)

            device = utils.get_device(quant_sim_model)
            QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device)

            # Inference AIMET quantization nodes
            aimet_res = forward_pass(sim.model, None)
            # Inference Native torch quantization nodes
            torch_res = forward_pass(quant_sim_model, None)
            assert torch.allclose(aimet_res, torch_res, rtol=1e-2)

    def test_export_to_onnx_direct(self):
        if version.parse(torch.__version__) >= version.parse("1.13.0"):
            onnx_utils.EXPORT_TO_ONNX_DIRECT = True
            model = ModelWithTwoInputs()
            dummy_input = (torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda m, _: m(*dummy_input), None)
            with tempfile.TemporaryDirectory() as temp_dir:
                sim.export(temp_dir, 'direct_onnx_export', dummy_input)

                onnx_utils.EXPORT_TO_ONNX_DIRECT = False
                sim.export(temp_dir, 'onnxsaver_export', dummy_input)

                with open(os.path.join(temp_dir, 'direct_onnx_export.encodings')) as direct_onnx_json:
                    direct_onnx_encodings = json.load(direct_onnx_json)
                with open(os.path.join(temp_dir, 'onnxsaver_export.encodings')) as onnxsaver_json:
                    onnxsaver_encodings = json.load(onnxsaver_json)

                assert len(direct_onnx_encodings['activation_encodings']) == \
                       len(onnxsaver_encodings['activation_encodings'])
                assert len(direct_onnx_encodings['param_encodings']) == len(onnxsaver_encodings['param_encodings'])
                direct_onnx_act_names = direct_onnx_encodings['activation_encodings'].keys()
                onnxsaver_act_names = onnxsaver_encodings['activation_encodings'].keys()
                assert direct_onnx_act_names != onnxsaver_act_names

    def test_export_to_onnx_direct_fixed_param_names(self):
        torch.manual_seed(0)
        model = SmallLinearModel()
        dummy_input = torch.randn(1, 8, 3)
        with set_export_to_onnx_direct(True):
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda m, _: m(*dummy_input), None)

            with tempfile.TemporaryDirectory() as tmp_dir:
                sim.export(tmp_dir, 'single_linear', dummy_input)

                with open(os.path.join(tmp_dir, 'single_linear.encodings'), 'r') as encodings_file:
                    encodings = json.load(encodings_file)

                param_encodings_set = {name for name in encodings['param_encodings'].keys()}

                for name, _ in model.named_parameters():
                    if 'bias' not in name:
                        assert name in param_encodings_set

    @pytest.mark.parametrize("num_parameters", [1, 8])
    @pytest.mark.parametrize("config_file", [None, get_path_for_per_channel_config()])
    def test_export_to_onnx_for_multiple_p_relu_model(self, num_parameters, config_file):
        model = test_models.MultiplePReluModel(num_parameters)
        dummy_input = torch.randn(4, 3, 28, 28)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim = QuantizationSimModel(model, dummy_input, QuantScheme.post_training_tf,
                                       config_file=config_file)
            sim.compute_encodings(lambda m, _: m(dummy_input), None)
            filename_prefix = "multiple_p_relu_model"

            sim.export(tmp_dir, filename_prefix, dummy_input)
            with open(f"{tmp_dir}/{filename_prefix}.encodings") as encodings_file:
                encodings = json.load(encodings_file)

        param_encodings = encodings["param_encodings"]
        expected_param_names = ["act1.weight", "act2.weight", "act3.weight"]
        for param_name in expected_param_names:
            assert param_name in param_encodings

            if config_file: # Per-channel
                assert len(param_encodings[param_name]) == num_parameters
            else:           # Per-tensor
                assert len(param_encodings[param_name]) == 1

    def test_save_encodings_to_json(self):
        model = ModelWithTwoInputsOneToAdd()
        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))
        qsim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf)
        qsim.compute_encodings(lambda m, _: m(*dummy_input), None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            qsim.save_encodings_to_json(tmp_dir, 'saved_encodings')
            with open(os.path.join(tmp_dir, 'saved_encodings.json')) as encodings_file:
                encodings = json.load(encodings_file)
                assert len(encodings['activation_encodings']) == 13
                assert len(encodings['param_encodings']) == 5

    def test_compute_encodings_for_multiple_sims(self):
        class SecondModel(torch.nn.Module):
            def __init__(self, const_inp_shape):
                super(SecondModel, self).__init__()
                self.add = aimet_modules.Add()
                self.sub = aimet_modules.Subtract()
                self.batchnorm = torch.nn.BatchNorm1d(10)
                self.const_tensor = torch.randn(const_inp_shape)

            def forward(self, inp, inp2):
                x = self.add(inp, self.const_tensor)
                x = self.batchnorm(x)
                x = self.sub(x, inp2)
                return x

        model = ModelWithTwoInputsOneToAdd()
        model.eval()
        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))
        model_1_out = model(*dummy_input)
        model_2 = SecondModel(model_1_out.shape)
        model_2.eval()
        dummy_input_2 = torch.randn(model_1_out.shape)
        sim1 = QuantizationSimModel(model, dummy_input)
        sim2 = QuantizationSimModel(model_2, (dummy_input_2, dummy_input_2))

        def forward_pass_callback(model_list, _):
            x = model_list[0](*dummy_input)
            x = model_list[1](x, dummy_input_2)
            return x

        sim2.model.train()
        running_mean = sim2.model.batchnorm.running_mean.clone().detach()
        compute_encodings_for_sims([sim1, sim2], forward_pass_callback, None)

        # Check that even though sim2 was in training mode prior to compute encodings, it was placed in eval mode
        # during compute encodings, and that it was placed back to training mode afterwards.
        assert sim2.model.training
        assert torch.equal(running_mean, sim2.model.batchnorm.running_mean)
        assert sim1.model.conv1_a.output_quantizers[0].encoding is not None
        assert sim2.model.add.input_quantizers[0].encoding is not None
        assert sim2.model.add.input_quantizers[1].encoding is not None

    def test_load_and_freeze_encodings(self):
        model = SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        partial_torch_encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
                "conv2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.4923851788043976,
                            "min": -0.43767568469047546,
                            "offset": -120,
                            "scale": 0.0036472973879426718
                        }
                    }
                },
                "fc2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.1948324590921402,
                            "min": -0.15752412378787994,
                            "offset": -114,
                            "scale": 0.0013817904982715845
                        }
                    }
                },
                "relu1": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 1.0608084201812744,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.004160033073276281
                        }
                    }
                },
                "relu3": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.5247029066085815,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.0020576585084199905
                        }
                    }
                }
            },
            "excluded_layers": [],
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
                "fc2.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.13095608353614807,
                        "min": -0.14966410398483276,
                        "offset": -8,
                        "scale": 0.018708012998104095
                    }
                ]
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)
        originally_enabled = {q: q.enabled for q in chain(*utils.get_all_quantizers(qsim.model))}

        def forward_pass(model, dummy_input):
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

        with open("./temp_partial_torch_encodings.encodings", 'w') as fp:
            json.dump(partial_torch_encodings, fp)

        qsim.load_and_freeze_encodings("./temp_partial_torch_encodings.encodings")

        # ``load_and_freeze_encodings`` shouldn't additionally enable or disable any quantizers
        assert all(q.enabled == originally_enabled[q] for q in chain(*utils.get_all_quantizers(qsim.model)))

        loaded_quantizers = [
            qsim.model.conv1.input_quantizers[0],
            qsim.model.conv1.param_quantizers['weight'],
            qsim.model.conv2.output_quantizers[0],
            qsim.model.fc2.output_quantizers[0],
            qsim.model.fc2.param_quantizers['weight'],
            qsim.model.relu1.output_quantizers[0],
            qsim.model.relu3.output_quantizers[0],
        ]

        # ``load_and_freeze_encodings`` should freeze all the loaded quantizers
        for q in chain(*utils.get_all_quantizers(qsim.model)):
            if q in loaded_quantizers:
                assert q.is_encoding_frozen
            else:
                assert not q.is_encoding_frozen

        qsim.compute_encodings(forward_pass, dummy_input)
        decimal_point_check = 6

        def assert_input_output_quantizers(quantizers, quant_type):
            for idx, io_quant in enumerate(quantizers):
                assert io_quant.is_encoding_frozen == True
                qsim_encodings = io_quant.encoding
                actual_encodings = partial_torch_encodings['activation_encodings'][name][quant_type][str(idx)]
                assert qsim_encodings.bw == actual_encodings['bitwidth']
                np.testing.assert_almost_equal(qsim_encodings.delta, actual_encodings['scale'], decimal_point_check)
                np.testing.assert_almost_equal(qsim_encodings.max, actual_encodings['max'], decimal_point_check)
                np.testing.assert_almost_equal(qsim_encodings.min, actual_encodings['min'], decimal_point_check)
                assert qsim_encodings.offset == actual_encodings['offset']

        def assert_param_quantizers(param_quantizer, module_name, param_name):
            qsim_computed_encodings = param_quantizer[param_name].encoding
            qsim_computed_encodings = [qsim_computed_encodings] if not isinstance(qsim_computed_encodings, list) \
                else qsim_computed_encodings
            for idx, qsim_encoding in enumerate(qsim_computed_encodings):
                actual_encodings = partial_torch_encodings['param_encodings'][module_name+"."+param_name][idx]
                assert param_quantizer[param_name].is_encoding_frozen == True
                assert qsim_encoding.bw == actual_encodings['bitwidth']
                np.testing.assert_almost_equal(qsim_encoding.delta, actual_encodings['scale'], decimal_point_check)
                np.testing.assert_almost_equal(qsim_encoding.max, actual_encodings['max'], decimal_point_check)
                np.testing.assert_almost_equal(qsim_encoding.min, actual_encodings['min'], decimal_point_check)
                assert qsim_encoding.offset == actual_encodings['offset']

        input_quant_checked = []
        output_quant_checked = []
        param_quant_checked = []
        for name, quant in qsim.quant_wrappers():
            if name in partial_torch_encodings['activation_encodings']:
                if 'input' in partial_torch_encodings['activation_encodings'][name]:
                    assert_input_output_quantizers(quant.input_quantizers, 'input')
                    input_quant_checked.append(name)
                if 'output' in partial_torch_encodings['activation_encodings'][name]:
                    assert_input_output_quantizers(quant.output_quantizers, 'output')
                    output_quant_checked.append(name)

            param_types = quant.param_quantizers.keys()
            for param_type in param_types:
                module_param_name = name + "." + param_type
                if module_param_name in partial_torch_encodings['param_encodings']:
                    assert_param_quantizers(quant.param_quantizers, name, param_type)
                    param_quant_checked.append(module_param_name)

        actual_input_quant = {k for k, v in partial_torch_encodings['activation_encodings'].items() if 'input' in v}
        actual_output_quant = {k for k, v in partial_torch_encodings['activation_encodings'].items() if 'output' in v}
        actual_param_quant = set(partial_torch_encodings['param_encodings'].keys())

        assert actual_input_quant == set(input_quant_checked)
        assert actual_output_quant == set(output_quant_checked)
        assert actual_param_quant == set(param_quant_checked)

        os.remove("./temp_partial_torch_encodings.encodings")

    def test_logits_of_grouped_conv_net(self):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_channels, out_channels = 6, 12
        standard_grouped_conv_model = test_models.GroupedConvModel(
            in_channels, out_channels
        ).to(device).eval()

        custom_grouped_conv_model = test_models.CustomGroupedConvModel(
            in_channels // 2, out_channels // 2
        ).to(device).eval()
        with torch.no_grad():
            custom_grouped_conv_model.conv1.weight.copy_(
                standard_grouped_conv_model.conv.weight[: out_channels // 2]
            )
            custom_grouped_conv_model.conv2.weight.copy_(
                standard_grouped_conv_model.conv.weight[out_channels // 2 :]
            )

        standard_module_inputs = torch.randn(1, 6, 10, 10, device=device)
        custom_module_inputs = (
            standard_module_inputs[:, : in_channels // 2, :, :],
            standard_module_inputs[:, in_channels // 2 :, :, :],
        )

        pcq_config_path = get_path_for_per_channel_config()
        sim_from_standard = QuantizationSimModel(
            standard_grouped_conv_model, standard_module_inputs, config_file=pcq_config_path
        )
        sim_from_custom = QuantizationSimModel(
            custom_grouped_conv_model, custom_module_inputs, config_file=pcq_config_path
        )

        # Disable activation quantizers to measure impact of grouped conv weight
        def _disable_activation_quantizers(sim):
            for _, wrapper in sim.quant_wrappers():
                for q in wrapper.input_quantizers:
                    q.enabled = False

                for q in wrapper.output_quantizers:
                    q.enabled = False

        _disable_activation_quantizers(sim_from_standard)
        _disable_activation_quantizers(sim_from_custom)

        sim_from_standard.compute_encodings(lambda m, _: m(standard_module_inputs), None)
        sim_from_custom.compute_encodings(lambda m, _: m(*custom_module_inputs), None)
        with torch.inference_mode():
            standard_module_outputs = sim_from_standard.model(standard_module_inputs)
            custom_module_outputs = sim_from_custom.model(*custom_module_inputs)
        assert torch.allclose(standard_module_outputs, custom_module_outputs, atol=1e-5)

    @pytest.mark.parametrize(
        "test_model", [test_models.ModelWithSplitModule, test_models.ModelWithReluAfterSplit]
    )
    def test_quantizer_enable_with_split_module(self, test_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = test_model().to(device)
        model.eval()
        dummy_input = torch.randn(6, 2, device=device)

        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "strict_symmetric": "False",
            },
            "params": {},
            "op_type": {"Split": {"is_output_quantized": "False"}},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, dummy_input, config_file=config_path)

        if isinstance(model, test_models.ModelWithSplitModule):
            split_module = sim.model.split
        else:
            split_module = sim.model.split_module.split

        # is_input_quantized=True for model input
        # First input quantizer should be enabled, second input quantizer is disabled as it's constant
        assert split_module.input_quantizers[0].enabled
        assert not split_module.input_quantizers[1].enabled

        # All output quantizers of Split should be disabled because op specific config is set
        for output_quantizer in split_module.output_quantizers:
            assert not output_quantizer.enabled

class TestQuantizationSimLearnedGrid:

    # -------------------------------------------------------------------------------

    def test_replace_quant_wrapper(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.fc1 = nn.Linear(640, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                return x

        net = Net()
        model = net.to(torch.device('cpu'))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=torch.randn(1, 1, 12, 12))
        assert isinstance(sim.model.conv1, StaticGridQuantWrapper)
        assert isinstance(sim.model.fc1, StaticGridQuantWrapper)
        for _, module in sim.model._modules.items():
            module.input_quantizers[0].enabled = False
            module.output_quantizers[0].enabled = False
            module.param_quantizers['weight'].enabled = False
        sim._replace_quantization_wrapper(sim.model, device='cpu')

        assert isinstance(sim.model.conv1, LearnedGridQuantWrapper)
        assert isinstance(sim.model.fc1, LearnedGridQuantWrapper)

    @pytest.mark.cuda
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_range_learning_with_fp16_and_bw_32_quantizers(self, device):
        model = SmallMnistNoDropout()
        model.eval()
        model.to(device)
        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        sim.model.conv2.param_quantizers['weight'].data_type = QuantizationDataType.float
        sim.model.conv2.param_quantizers['weight'].bitwidth = 16
        sim.model.relu2.output_quantizers[0].bitwidth = 32
        sim.compute_encodings(lambda m, _: m(dummy_input), None)

        assert sim.model.conv2.param_quantizers['weight'].encoding is None
        assert sim.model.relu2.output_quantizers[0].encoding is None

        sim.model.train()
        output = sim.model(copy.deepcopy(dummy_input))
        loss = output.flatten().sum()

        orig_conv1_weight = sim.model.conv1._module_to_wrap.weight.clone().detach()
        orig_conv1_encoding_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        orig_conv2_weight = sim.model.conv2._module_to_wrap.weight.clone().detach()
        loss.backward()

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        optimizer.step()
        optimizer.zero_grad()

        new_conv1_weight = sim.model.conv1._module_to_wrap.weight.clone().detach()
        new_conv1_encoding_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        new_conv2_weight = sim.model.conv2._module_to_wrap.weight.clone().detach()
        assert not torch.equal(orig_conv1_weight, new_conv1_weight)
        assert orig_conv1_encoding_max != new_conv1_encoding_max
        assert not torch.equal(orig_conv2_weight, new_conv2_weight)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'rl_with_fp16_and_bw_32', dummy_input=dummy_input.to('cpu'))
            with open(os.path.join(tmp_dir, 'rl_with_fp16_and_bw_32_torch.encodings')) as json_file:
                encoding_data = json.load(json_file)
                assert encoding_data['param_encodings']['conv2.weight'][0] == {'bitwidth': 16, 'dtype': 'float'}
                assert 'relu2' not in encoding_data['activation_encodings']
                assert len(encoding_data['activation_encodings']) == 5

    @pytest.mark.cuda
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_learned_grid_with_fixed_encoding_min_max_vars(self, device):
        model = SmallMnistNoDropout()
        model.eval()
        model.to(device)
        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        sim.model.conv1.enable_per_channel_quantization()
        sim.model.relu1.output_quantizers[0].encoding_min_max_fixed_vals = (-5.0, 5.0)
        sim.model.conv1.param_quantizers['weight'].encoding_min_max_fixed_vals = (-10.0, 10.0)
        sim.compute_encodings(lambda m, _: m(dummy_input), None)

        assert sim.model.relu1.output_quantizers[0].is_encoding_frozen

        # Min and max will not be exactly what is specified due to 0 needing to be quantizable
        assert sim.model.relu1.output_quantizers[0].encoding.min > -5.5
        assert sim.model.relu1.output_quantizers[0].encoding.min < -4.5
        assert sim.model.relu1.output_quantizers[0].encoding.max > 4.5
        assert sim.model.relu1.output_quantizers[0].encoding.max < 5.5

        assert sim.model.conv1.param_quantizers['weight'].is_encoding_frozen
        for encoding in sim.model.conv1.param_quantizers['weight'].encoding:
            assert encoding.min > -10.5
            assert encoding.min < -9.5
            assert encoding.max > 9.5
            assert encoding.max < 10.5

    @pytest.mark.cuda
    @pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.device_count() > 1,
                        reason="Currently broken in multi-gpu environments")
    def test_multi_gpu_qat(self):
        """"""
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        class OneConvLayerModel(nn.Module):
            def __init__(self):
                super(OneConvLayerModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=2)

            def forward(self, x):
                x = self.conv1(x)
                return x

        model = OneConvLayerModel().to('cuda:0')
        model = model.to('cuda:0')

        dummy_input = torch.ones(2, 1, 3, 3).to('cuda:0')
        dummy_input[1] += 1

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        original_weight = sim.model.conv1._module_to_wrap.weight.clone().detach()

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1._module_to_wrap.weight.grad.clone().detach()
        sim.model.conv1._module_to_wrap.weight.grad = None

        sim.model = torch.nn.DataParallel(sim.model)

        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))

        weight = sim.model.module.conv1._module_to_wrap.weight.clone().detach()

        assert torch.allclose(output_multi_gpu, output_single_gpu)
        assert torch.allclose(original_weight, weight)

        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        assert torch.allclose(sim.model.module.conv1._module_to_wrap.weight.grad, grad_single_gpu)

    def test_copy_properties_between_wrappers_when_inp_output_is_1(self):
        conv1 = nn.Conv2d(3, 10, kernel_size=5)
        post_training_module = StaticGridQuantWrapper(conv1, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=False,
                                                      is_output_quantized=False, activation_bw=8, weight_bw=8)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2

        post_training_module.input_quantizers[0].enabled = True
        post_training_module.input_quantizers[0].encoding = encodings
        post_training_module.param_quantizers['weight'].enabled = False
        post_training_module.param_quantizers['bias'].enabled = False
        dummy_input = torch.randn(1, 3, 12, 12)
        sim = QuantizationSimModel(conv1, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)
        # sim.model.conv1.input_quantizer.enabled = True
        trainable_module = sim._construct_and_initialize_trainable_wrapper(post_training_module, device='cpu')

        assert trainable_module.output_quantizers[0].use_symmetric_encodings == False
        assert trainable_module.output_quantizers[0].enabled == False
        assert trainable_module.input0_encoding_min.item() == -5.0

    def test_copy_properties_for_elementwise_aimet_add_op(self):

        class ElementwiseAdd(nn.Module):
            def __init__(self):
                super(ElementwiseAdd, self).__init__()
                self.add = aimet_modules.Add()

            def forward(self, *inputs):
                return self.add(inputs[0], inputs[1])

        add = ElementwiseAdd()
        post_training_module = StaticGridQuantWrapper(add, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=False,
                                                      is_output_quantized=False, activation_bw=8, weight_bw=8,
                                                      num_inputs=2, num_outputs=1)
        encodings = libpymo.TfEncoding()
        encodings.bw, encodings.max, encodings.min, encodings.delta, encodings.offset = 8, 5.0, -1.0, 1, 0.2

        for inp_quantizer in post_training_module.input_quantizers:
            inp_quantizer.enabled = True
            inp_quantizer.encoding = encodings

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))
        sim = QuantizationSimModel(add, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        trainable_module = sim._construct_and_initialize_trainable_wrapper(post_training_module, device='cpu')

        assert trainable_module.output_quantizers[0].use_symmetric_encodings == False
        assert trainable_module.output_quantizers[0].enabled == False
        assert trainable_module.input0_encoding_min.item() == -1.0
        assert trainable_module.input1_encoding_max.item() == 5.0

    def test_qc_trainable_wrapper(self):
        torch.manual_seed(0)
        conv1 = nn.Conv2d(1, 32, kernel_size=5)

        trainable_module = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                   is_symmetric=False, is_output_quantized=True, activation_bw=8,
                                                   weight_bw=8, device='cpu', data_type=QuantizationDataType.int)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 3
        encodings.min = -2
        encodings.delta = 1
        encodings.offset = 0.2
        trainable_module.input_quantizers[0].enabled = True
        trainable_module.input_quantizers[0].encoding = encodings

        trainable_module.param_quantizers['weight'].enabled = True
        trainable_module.param_quantizers['weight'].encoding = encodings
        trainable_module.param_quantizers['bias'].enabled = True
        trainable_module.param_quantizers['bias'].encoding = encodings

        trainable_module.output_quantizers[0].enabled = True
        trainable_module.output_quantizers[0].encoding = encodings

        inp = torch.rand((1, 1, 5, 5), requires_grad=True)
        out = trainable_module(inp)
        optimizer = torch.optim.SGD(trainable_module.parameters(), lr=0.05, momentum=0.5)
        loss = out.flatten().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Checking if encoding min max have changed
        assert not trainable_module.input0_encoding_min.item() == -2.0
        assert not trainable_module.input0_encoding_max.item() == 3.0

        assert not trainable_module.output0_encoding_min.item() == -2.0
        assert not trainable_module.output0_encoding_max.item() == 3.0

        assert not trainable_module.weight_encoding_min.item() == -2.0
        assert not trainable_module.weight_encoding_max.item() == 3.0

        assert not trainable_module.bias_encoding_min.item() == -2.0
        assert not trainable_module.bias_encoding_max.item() == 3.0

    def test_qc_trainable_wrapper_for_model_with_multiple_inputs_with_one_add(self):
        # NOTE: Use asymmetric quantization for parameter, which have gradients both encoding min/max
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = Path(tmp_dir, "quantsim_config.json")
            with open(config_file_path, "w") as f:
                json.dump(quantsim_config, f)

            dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

            def forward_pass(sim_model, _):
                sim_model.eval()
                with torch.no_grad():
                    sim_model(*dummy_input)

            model = ModelWithTwoInputsOneToAdd()

            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    config_file=config_file_path)
            # Enable input parameters to add (multiple input parameter exist)
            sim.model.add.input_quantizers[0].enabled = True
            sim.model.add.input_quantizers[1].enabled = True

            sim.compute_encodings(forward_pass, forward_pass_callback_args=None)

            assert len(sim.model.add.input_quantizers) == 2

            out = sim.model(*dummy_input)
            for _, params in sim.model.named_parameters():
                assert params.grad is None

            optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
            optimizer.zero_grad()
            loss = out.flatten().sum()
            loss.backward()
            optimizer.step()

            # All parameters should have a gradient
            for params in sim.model.parameters():
                assert params.grad is not None

    def test_get_effective_encoding(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = ModelWithTwoInputsOneToAdd()
        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        sim = QuantizationSimModel(model,
                                   dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        sim.compute_encodings(forward_pass, None)
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.005, momentum=0.5)
        for _ in range(20):
            inputs = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))
            out = sim.model(*inputs)
            loss = out.flatten().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        def _helper(_module_name: str):
            wrapper = getattr(sim.model, _module_name)
            param_quantizers = [x for x in wrapper.param_quantizers.values()]
            quantizers = wrapper.output_quantizers + param_quantizers

            for quantizer in quantizers:
                if not quantizer.enabled:
                    continue

                encoding = quantizer.get_effective_encoding()
                delta = encoding.delta
                offset = encoding.offset
                encoding_min = encoding.min
                encoding_max = encoding.max

                assert np.isclose(encoding_min, delta * offset, atol=1e-5)
                assert np.isclose(encoding_max, encoding_min + delta * 255, atol=1e-5)

        module_names = ["conv1_a", "maxpool1_a", "relu1_a",
                        "conv1_b", "maxpool1_b", "relu1_b",
                        "add", "conv2", "maxpool2", "relu2",
                        "fc1", "relu3", "dropout", "fc2"]

        for module_name in module_names:
            _helper(module_name)

    def test_symmetry_characteristic_and_export_result(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = ModelWithTwoInputsOneToAdd()
        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        sim = QuantizationSimModel(model,
                                   dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        def forward_pass(sim_model, _):
            sim_model(*dummy_input)

        sim.compute_encodings(forward_pass, None)
        assert sim.model.conv1_a.weight_encoding_min == -sim.model.conv1_a.weight_encoding_max
        assert torch.allclose(sim.model.fc1.weight_encoding_min, -sim.model.fc1.weight_encoding_max)

        before_conv1_weight_encoding_min = sim.model.conv1_a.weight_encoding_min.clone().detach()
        before_fc_weight_encoding_min = sim.model.fc1.weight_encoding_min.clone().detach()

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.003, momentum=0.5)
        for _ in range(20):
            inputs = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))
            out = sim.model(*inputs)
            loss = out.flatten().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert torch.allclose(sim.model.conv1_a.weight_encoding_min, -sim.model.conv1_a.weight_encoding_max)
        assert torch.allclose(sim.model.fc1.weight_encoding_min, -sim.model.fc1.weight_encoding_max)

        after_conv1_weight_encoding_min = sim.model.conv1_a.weight_encoding_min.clone().detach()
        after_fc_weight_encoding_min = sim.model.fc1.weight_encoding_min.clone().detach()

        assert not torch.allclose(before_conv1_weight_encoding_min, after_conv1_weight_encoding_min)
        assert not torch.allclose(before_fc_weight_encoding_min, after_fc_weight_encoding_min)

        with tempfile.TemporaryDirectory() as tempdir:
            sim.export(tempdir, "results", dummy_input)
            with open(f"{tempdir}/results.encodings", "r") as encodings_file:
                encodings = json.load(encodings_file)

                param_encodings = encodings["param_encodings"]
                for layer in ["conv1_a", "conv2", "fc1"]:
                    encoding_info = param_encodings[f"{layer}.weight"][0]
                    encoding_min = encoding_info["min"]
                    encoding_max = encoding_info["max"]
                    scale = encoding_info["scale"]
                    offset = encoding_info["offset"]

                    # Default HTP config is non-strict symmetric when parameter quantization
                    # Non-strict symmetric should have
                    # encoding_min == -encoding_max - scale (one more bin)
                    # offset as -128
                    assert np.allclose(encoding_min, -encoding_max - scale)
                    assert offset == -128

    def test_set_and_get_encoding_properties(self):
        torch.manual_seed(0)
        conv1 = nn.Conv2d(1, 32, kernel_size=5)

        trainable_module = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                   is_symmetric=False, is_output_quantized=True, activation_bw=8,
                                                   weight_bw=8, device='cpu', data_type=QuantizationDataType.int)

        trainable_module.input_quantizers[0].enabled = True

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2

        # If enabled encoding cannot be None
        with pytest.raises(RuntimeError):
            trainable_module.input_quantizers[0].encoding = None
        trainable_module.input_quantizers[0].encoding = encodings

        # Check if quantizer.encoding is accessible
        print(trainable_module.input_quantizers[0].encoding)
        assert np.allclose(trainable_module.input0_encoding_min.detach().numpy(), encodings.min)

    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        assert sim.model.conv1_a.output_quantizers[0].encoding

        # self.assertAlmostEqual(sim.model.conv1_a.output_quantizer.encoding.min,
        #                        sim.model.conv1_a.output0_encoding_min.data)
        # self.assertAlmostEqual(sim.model.conv1_a.output_quantizer.encoding.max,
        #                        sim.model.conv1_a.output0_encoding_max.data)

        forward_pass(sim.model, None)
        print(sim)

    def test_memory_profiler(self):
        """ test using memory profiler """
        # checks PyTorch version before importing torch.profiler (introduced in version 1.8.0), for
        # older versions this test is passthrough.
        if version.parse(torch.__version__) >= version.parse("1.8"):

            from torch.profiler import profile, ProfilerActivity
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(3, 10, 5, 5)
                    self.conv2 = nn.Conv2d(10, 20, 5, 5)
                    self.conv3 = nn.Conv2d(20, 40, 5, 5)

                def forward(self, input):
                    x = self.conv1(input)
                    x = torch.nn.functional.relu(x)
                    x = self.conv2(x)
                    x = torch.nn.functional.relu(x)
                    x = self.conv3(x)
                    x = torch.nn.functional.relu(x)
                    return x

            model = Model()
            from aimet_torch.model_preparer import prepare_model
            model = prepare_model(model)
            in_tensor = torch.rand(32, 3, 224, 224)

            def forward_pass(model, args):
                model(in_tensor)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       dummy_input=in_tensor)
            sim.compute_encodings(forward_pass, None)

            print("Starting")
            sim.model.train()

            with profile(activities=[ProfilerActivity.CPU],
                         profile_memory=True, record_shapes=True) as prof:

                for _ in range(1):
                    out = sim.model(in_tensor)
                    out = out.sum().backward()

            memory_stats = [event for event in prof.key_averages() if event.key == '[memory]'][0]
            assert abs(memory_stats.cpu_memory_usage) < 1.1 * (100 * (10 ** 6))

            print(memory_stats.cpu_memory_usage)

    def test_accumulator_overflow(self):

        torch.manual_seed(0)
        model = models.resnet18()
        model = model.eval()
        layer, range_used = check_accumulator_overflow(model, 8, 32)

        assert layer == 'layer4.1.conv1'

        # self.assertAlmostEqual(100 * range_used, 0.263623, places=3)

    def test_export_prelu_encoding_and_check_load_encodings(self):
        """ Test that prelu weight is exported correctly """
        model = PreluModel()
        dummy_input = torch.rand(1, 3, 8, 8)
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'prelu_model', dummy_input=dummy_input)
            with open(os.path.join(tmp_dir, 'prelu_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
            assert 'prelu.weight' in encoding_data['param_encodings'].keys()

            output = sim.model(copy.deepcopy(dummy_input))
            del sim

            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                       quant_scheme=QuantScheme.training_range_learning_with_tf_init)
            encoding_file_path_pytorch = os.path.join(tmp_dir, 'prelu_model' + '_torch' + '.encodings')
            load_encodings_to_sim(sim, encoding_file_path_pytorch)

            layer = sim.model.prelu
            if isinstance(layer, QcQuantizeWrapper):
                for input_quantizer in layer.input_quantizers:
                    if input_quantizer.enabled:
                        assert input_quantizer.encoding is not None
                for output_quantizer in layer.output_quantizers:
                    if output_quantizer.enabled:
                        assert output_quantizer.encoding is not None
                for name in layer.param_quantizers:
                    if layer.param_quantizers[name].enabled:
                        assert layer.param_quantizers[name].encoding is not None

            output1 = sim.model(copy.deepcopy(dummy_input))
            assert sum(output1.flatten() - output.flatten()) == 0.0

    def test_load_encodings_multi_input_multi_output_model(self):
        net = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=4, default_output_bw=4)

        sim.model.cust.output_quantizers[0].enabled = False
        sim.compute_encodings(evaluate, dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'module_with_5_output', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                       propagate_encodings=False)

            del sim

            sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       default_param_bw=4, default_output_bw=4)
            sim.model.cust.output_quantizers[0].enabled = False
            encoding_file_path_pytorch = os.path.join(tmp_dir, 'module_with_5_output' + '_torch' + '.encodings')
            load_encodings_to_sim(sim, encoding_file_path_pytorch)

            layer = sim.model.cust
            if isinstance(layer, QcQuantizeWrapper):
                for input_quantizer in layer.input_quantizers:
                    if input_quantizer.enabled:
                        assert input_quantizer.encoding is not None
                for output_quantizer in layer.output_quantizers:
                    if output_quantizer.enabled:
                        assert output_quantizer.encoding is not None
                for name in layer.param_quantizers:
                    if layer.param_quantizers[name].enabled:
                        assert layer.param_quantizers[name].encoding is not None

    def test_load_encodings_to_sim_partial(self):
        model = SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        partial_torch_encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
                "conv2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.4923851788043976,
                            "min": -0.43767568469047546,
                            "offset": -120,
                            "scale": 0.0036472973879426718
                        }
                    }
                },
                "fc2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.1948324590921402,
                            "min": -0.15752412378787994,
                            "offset": -114,
                            "scale": 0.0013817904982715845
                        }
                    }
                },
                "relu1": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 1.0608084201812744,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.004160033073276281
                        }
                    }
                },
                "relu3": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.5247029066085815,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.0020576585084199905
                        }
                    }
                }
            },
            "excluded_layers": [],
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
                "fc2.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.13095608353614807,
                        "min": -0.14966410398483276,
                        "offset": -8,
                        "scale": 0.018708012998104095
                    }
                ]
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(partial_torch_encodings, f)

            load_encodings_to_sim(qsim, fname)

        loaded_quantizers = [
            qsim.model.conv1.input_quantizers[0],
            qsim.model.conv1.param_quantizers['weight'],
            qsim.model.conv2.output_quantizers[0],
            qsim.model.fc2.output_quantizers[0],
            qsim.model.fc2.param_quantizers['weight'],
            qsim.model.relu1.output_quantizers[0],
            qsim.model.relu3.output_quantizers[0],
        ]

        # ``load_encodings_to_sim`` should disable all the quantizers
        # whose corresponding encoding does not exist in the json file
        for q in chain(*utils.get_all_quantizers(qsim.model)):
            if q in loaded_quantizers:
                assert q.enabled
            else:
                assert not q.enabled

    def test_set_and_freeze_encodings_partial(self):
        model = SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        partial_torch_encodings = {
            "excluded_layers": [],
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
                "fc2.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.13095608353614807,
                        "min": -0.14966410398483276,
                        "offset": -8,
                        "scale": 0.018708012998104095
                    }
                ]
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)
        originally_enabled = {q: q.enabled for q in chain(*utils.get_all_quantizers(qsim.model))}

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(partial_torch_encodings, f)

            qsim.set_and_freeze_param_encodings(fname)

        # ``set_and_freeze_param_encodings`` shouldn't additionally enable or disable any quantizers
        assert all(q.enabled == originally_enabled[q] for q in chain(*utils.get_all_quantizers(qsim.model)))

        loaded_quantizers = [
            qsim.model.conv1.param_quantizers['weight'],
            qsim.model.fc2.param_quantizers['weight'],
        ]

        # ``set_and_freeze_param_encodings`` should freeze all the loaded quantizers
        for q in chain(*utils.get_all_quantizers(qsim.model)):
            if q in loaded_quantizers:
                assert q.is_encoding_frozen
            else:
                assert not q.is_encoding_frozen

    @pytest.mark.parametrize('load_encodings_fn', [load_encodings_to_sim,
                                                   QuantizationSimModel.load_and_freeze_encodings,
                                                   QuantizationSimModel.set_and_freeze_param_encodings])
    def test_legacy_load_encodings_mismatching_encoding(self, load_encodings_fn):
        model = SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv999": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
            },
            "param_encodings": {
                "conv999.weight": [ # NOTE: conv999 does not exist in the model
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)

        """
        When: Try to load encoding file some keys of which are missing in the model
              (Note that conv999 does not exist in the model)
        Then: Throw runtime error
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

    @pytest.mark.parametrize('load_encodings_fn', [load_encodings_to_sim,
                                                   QuantizationSimModel.load_and_freeze_encodings,
                                                   QuantizationSimModel.set_and_freeze_param_encodings])
    def test_legacy_load_encodings_to_disabled_quantizer(self, load_encodings_fn):
        model = SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
            },
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)

        """
        Given: Input/param quantizers of conv1 is disabled
        When: Try to load input/param quantizers to conv1
        Then: Throw runtime error
        """
        qsim.model.conv1.input_quantizers[0].enabled = False
        qsim.model.conv1.param_quantizers['weight'].enabled = False

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

    def test_inplace_modification_with_relu(self):
        """
        Test custom function behavior with view+in-place (relu)
         (output of custom function is view when returned as-is)
        """

        class ModelWithInPlaceReLU(torch.nn.Module):
            """ A model with in-place ReLU. Use this model for unit testing purposes.
                Expected inputs: 1 input, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithInPlaceReLU, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, inputs):
                ## These ops will wrapped as supergroup ##
                x = self.conv1(inputs)
                x = self.relu(x) # in-place relu
                ##########################################
                return x

        # Test with both QAT and QAT-range learning.
        input_shape = (1, 3, 8, 8)
        dummy_input = torch.randn(*input_shape)
        model = ModelWithInPlaceReLU().train() # In-place ops are problematic only during training
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]

        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme, default_param_bw=4,
                                             default_output_bw=4)
            quant_sim.compute_encodings(evaluate, dummy_input)

            optimizer = torch.optim.SGD(quant_sim.model.parameters(), lr=0.001, momentum=0.5)
            out = quant_sim.model(dummy_input)
            loss = out.flatten().sum()
            loss.backward() # Should not raise error
            optimizer.step()
            optimizer.zero_grad()

    def test_inplace_modification_with_add(self):
        """
        Test custom function behavior with view+in-place (add)
         (output of custom function is view when returned as-is)
        """

        class ModelWithInPlaceAdd(torch.nn.Module):
            """ A model with in-place Add. Use this model for unit testing purposes.
                Expected inputs: 2 inputs, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithInPlaceAdd, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x += 1 # in-place add
                return x

        # Test with both QAT and QAT-range learning.
        input_shape = (1, 3, 8, 8)
        dummy_input = torch.randn(*input_shape)
        model = ModelWithInPlaceAdd().train() # In-place ops are problematic only during backward pass
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]

        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme, default_param_bw=4,
                                             default_output_bw=4)
            quant_sim.compute_encodings(evaluate, dummy_input)

            optimizer = torch.optim.SGD(quant_sim.model.parameters(), lr=0.001, momentum=0.5)
            out = quant_sim.model(dummy_input)
            loss = out.flatten().sum()
            loss.backward() # Should not raise error
            optimizer.step()
            optimizer.zero_grad()

    def test_multi_output_onnx_op(self):
        """
        Test mapping and exporting of output encodings for multiple output onnx op.
        """

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        net = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=4, default_output_bw=4)

        sim.model.cust.output_quantizers[0].enabled = False
        sim.compute_encodings(evaluate, dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'module_with_5_output', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                       propagate_encodings=False)
            with open(os.path.join(tmp_dir, 'module_with_5_output.encodings')) as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert '7' not in activation_encodings
                assert set(['8', '9', '10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple(self):
        """

        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        cust_model = CustModelV1Simple()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                         default_param_bw=8, default_output_bw=8)
        quant_sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_sim.export(tmp_dir, 'cust_v1_simple', dummy_input,
                             onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                             propagate_encodings=True)
            with open(os.path.join(tmp_dir, 'cust_v1_simple.encodings')) as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert set(['10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple_v2(self):
        """

        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        cust_model = CustomOpV2()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                         default_param_bw=4, default_output_bw=4)

        quant_sim.compute_encodings(evaluate, dummy_input)

        print(quant_sim)
        print(quant_sim.model)

        a, b, c, d, e = quant_sim.model(dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_sim.export(tmp_dir, 'cust_v2_simple', dummy_input,
                             onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                             propagate_encodings=False)

            with open(os.path.join(tmp_dir, 'cust_v2_simple.encodings')) as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert len(activation_encodings) == 6

            module_names = { module_name for module_name, _ in cust_model.named_modules()}
            onnx_model = onnx.load(os.path.join(tmp_dir, 'cust_v2_simple.onnx'))

            for node in onnx_model.graph.node:
                if not node.name.startswith('.'):
                    name = node.name.split('#')[0]
                    assert '.'.join(name.split('.')[:-1]) in module_names
            onnx.checker.check_model(onnx_model)

    def test_quant_roi_model(self):
        roi_model = RoiModel(height=7, width=7, scale=0.25)
        x = torch.rand(1, 1, 6, 6)
        rois = torch.tensor([ [0, -2.0, -2.0, 22.0, 22.0], ])
        dummy_input = (x, rois)
        torch.onnx.export(roi_model, dummy_input, './roi.onnx', opset_version=11)
        sim = QuantizationSimModel(roi_model, dummy_input=dummy_input)
        for q in sim.model.roi.input_quantizers:
            q.enabled = False

        def forward_pass(model, _):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        sim.compute_encodings(forward_pass, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, "roi_model", dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),  propagate_encodings=True)

            with open(os.path.join(tmp_dir, 'roi_model.encodings')) as json_file:
                encodings = json.load(json_file)['activation_encodings']

                # Only one entry should have min, max, delta and offset, remaining entries should be propagated
                # with bitwidth and dtype.
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 1

    def test_attributes_mismatch_after_manual_change(self):
        """ Test to enusre that the attributes for quantizers are correctly set when modified manually """
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
            def forward(self, inputs):
                x = self.conv1(inputs)
                return x

        model = SimpleModel().eval()
        dummy_input = torch.randn(1, 3, 10, 10)
        quant_sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                         quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        # Manually change bitwidth, use_strict_symmetric and use_unsigned_symmetric attributes for quantizers.
        quant_sim.model.conv1.input_quantizers[0].bitwidth = 16
        quant_sim.model.conv1.output_quantizers[0].bitwidth = 16
        quant_sim.model.conv1.param_quantizers['weight'].bitwidth = 16
        quant_sim.model.conv1.input_quantizers[0].use_strict_symmetric = True
        quant_sim.model.conv1.input_quantizers[0].use_unsigned_symmetric = False
        quant_sim.model.conv1.input_quantizers[0].use_symmetric_encodings = True

        # Compute encodings.
        quant_sim.compute_encodings(evaluate, dummy_input)

        # Make sure the attributes are in sync after replacing StaticGridQuantWrapper by LearnedGridQuantWrapper.
        assert quant_sim.model.conv1.input_quantizers[0].bitwidth == 16
        assert quant_sim.model.conv1.output_quantizers[0].bitwidth == 16
        assert quant_sim.model.conv1.param_quantizers['weight'].bitwidth == 16
        assert quant_sim.model.conv1.input_quantizers[0].use_strict_symmetric
        assert not quant_sim.model.conv1.input_quantizers[0].use_unsigned_symmetric
        assert quant_sim.model.conv1.input_quantizers[0].use_symmetric_encodings

    def test_unused_module_handling(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=2, bias=False)

                # This is an unused module
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, inputs):
                x = self.conv1(inputs)
                return x

        model = SimpleModel().eval()
        dummy_input = torch.randn(1, 3, 10, 10)
        quant_sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                         quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        # Compute encodings.
        quant_sim.compute_encodings(evaluate, dummy_input)

        # Forward and backward should finish without runtime error
        out = quant_sim.model(dummy_input)
        out.sum().backward()

    @pytest.mark.parametrize("model", [ConvReluModel(), ConvTransposeReluModel()])
    def test_quantizer_flag_when_unsigned_symmetric_is_enabled(self, model):
        torch.manual_seed(116)

        # Use symmetric quantization both activation and parameter and
        #   enable unsigned symmetric and per channel quantization flag
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "True"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "strict_symmetric": "False",
                "unsigned_symmetric": "True",
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            # Force all weight values to have positive numbers
            with torch.no_grad():
                model.conv.weight.add_(1).clamp_min_(0)

            dummy_input = torch.rand(16, 3, 28, 28)
            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                       quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       config_file=os.path.join(temp_dir, "quantsim_config.json"))

            sim.compute_encodings(evaluate, dummy_input)
            # Check whether encoding of weight parameter is symmetric characteristic
            for encoding in sim.model.conv.param_quantizers["weight"].encoding:
                assert np.allclose(encoding.min, -encoding.max)
                assert encoding.offset == -128
                assert np.allclose(encoding.delta, encoding.max / 127.0)

            # Param quantizer should have is_unsigned_symmetric to False,
            #   even though unsigned_symmetric is True and encoding range is all positive
            assert not sim.model.conv.param_quantizers["weight"].is_unsigned_symmetric

            # Activation quantizer can have is_unsigned_symmetric to False,
            #   even though unsigned_symmetric is True and encoding range is all positive
            assert not sim.model.relu.output_quantizers[0].is_unsigned_symmetric

            def _validate_export_result(file_name: str) -> None:
                def _validate_encoding(_encoding_info):
                    encoding_min = encoding_info["min"]
                    encoding_max = encoding_info["max"]
                    scale = encoding_info["scale"]
                    offset = encoding_info["offset"]

                    assert np.allclose(encoding_min, -encoding_max - scale)
                    assert offset == -128
                    assert np.isclose(encoding_min, scale * offset, atol=1e-6)
                    assert np.isclose(encoding_max, encoding_min + scale * 255, atol=1e-6)

                with open(f"{temp_dir}/{file_name}.encodings", "r") as encodings_file:
                    encodings = json.load(encodings_file)

                    activation_encodings = encodings["activation_encodings"]
                    for _, encoding_info_list in activation_encodings.items():
                        for encoding_info in encoding_info_list:
                            _validate_encoding(encoding_info)

                    param_encodings = encodings["param_encodings"]
                    for encoding_info in param_encodings["conv.weight"]:
                        _validate_encoding(encoding_info)

            sim.export(temp_dir, "before_range_learning", dummy_input)
            _validate_export_result("before_range_learning")

            optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.001, momentum=0.5)
            for _ in range(20):
                inputs = torch.rand(32, 3, 28, 28)
                out = sim.model(inputs)
                loss = out.flatten().sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            sim.export(temp_dir, "after_range_learning", dummy_input)
            _validate_export_result("after_range_learning")

    def test_quantsim_conv3d_tf_int8_eval_train(self):

        torch.random.manual_seed(10)
        model = Conv3dModel()
        dummy_input = torch.randn(1, 3, 24, 24, 24)
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=8, default_output_bw=8)
        sim.compute_encodings(evaluate, dummy_input)
        print(sim)
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

        # Test inference
        sim.model.eval()
        with torch.no_grad():
            output = sim.model(dummy_input)

        # Test one iteration of training
        sim.model.train()
        output = sim.model(dummy_input)

        output = output.sum()
        output.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_quantsim_conv3d_tf_fp16_eval_train(self):

        torch.random.manual_seed(10)
        model = Conv3dModel()
        dummy_input = torch.randn(1, 3, 24, 24, 24)
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=16, default_output_bw=16,
                                   default_data_type=QuantizationDataType.float)
        sim.compute_encodings(evaluate, dummy_input)
        print(sim)
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

        # Test inference
        sim.model.eval()
        with torch.no_grad():
            output = sim.model(dummy_input)

        # Test one iteration of training
        sim.model.train()
        output = sim.model(dummy_input)

        output = output.sum()
        output.backward()
        optimizer.step()
        optimizer.zero_grad()

    @pytest.mark.skipif(
        Version(torch.__version__) >= Version('1.13.0'),
        reason="Error `addmm_impl_cpu_` not implemented for Half in PyTorch 1.13.0 or later"
    )
    def test_fp16_model_sim_eval_train_cpu(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.fc1 = nn.Linear(320, 50)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(50, 10)
                self.relu2 = nn.ReLU()

            def forward(self, x):
                x = x.view(-1, 320)
                x = self.fc1(x)
                x = x.float()
                x = self.relu1(x)
                x = x.half()
                x = self.fc2(x)
                x = x.float()
                x = self.relu2(x)
                x = x.half()
                return x

        model = DummyModel().half()
        dummy_input = torch.rand(1, 20, 4, 4).half()

        model(dummy_input)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=dummy_input)

        def dummy_forward(model, _):
            return model(dummy_input)

        sim.compute_encodings(dummy_forward, None)
        print(sim.model)

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        output = dummy_forward(model, None)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    @pytest.mark.cuda
    def test_fp16_model_sim_eval_train_gpu(self):

        model = HalfFloatTestModel().cuda().half()
        dummy_input = torch.rand(1, 20, 4, 4).half().cuda()
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        def dummy_forward(model, _):
            output = None
            for _ in range(100):
                output = model(dummy_input)
            return output

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        sim.compute_encodings(dummy_forward, None)
        output = dummy_forward(model, None)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    @pytest.mark.cuda
    def test_fp16_model_sim_eval_train_learned_grid_per_channel_gpu(self):

        quantsim_config = {
                                "defaults":
                                {
                                    "ops":
                                    {
                                        "is_output_quantized": "True",
                                        "is_symmetric": "True"
                                    },
                                    "params":
                                    {
                                        "is_quantized": "True",
                                        "is_symmetric": "True"
                                    },
                                    "strict_symmetric": "False",
                                    "unsigned_symmetric": "True",
                                    "per_channel_quantization": "True",
                                },
                                "params": {},
                                "op_type": {},
                                "supergroups": [],
                                "model_input": {},
                                "model_output": {}
                          }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), "w") as f:
                json.dump(quantsim_config, f)

            model = HalfFloatTestModel().cuda().half()
            dummy_input = torch.rand(1, 20, 4, 4).half().cuda()
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       dummy_input=dummy_input, config_file=os.path.join(tmp_dir, 'quantsim_config.json'))

            def dummy_forward(model, _):
                output = None
                for _ in range(100):
                    output = model(dummy_input)
                return output

            optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
            sim.compute_encodings(dummy_forward, None)
            output = dummy_forward(model, None)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    @pytest.mark.cuda
    def test_fp16_model_sim_eval_train_static_grid_per_channel_gpu(self):

        quantsim_config = {
            "defaults":
                {
                    "ops":
                        {
                            "is_output_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "True",
                    "per_channel_quantization": "True",
                },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "quantsim_config.json"), "w") as f:
                json.dump(quantsim_config, f)

            model = HalfFloatTestModel().cuda().half()
            dummy_input = torch.rand(1, 20, 4, 4).half().cuda()
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       dummy_input=dummy_input, config_file=os.path.join(tmp_dir, "quantsim_config.json"))

            def dummy_forward(model, _):
                output = None
                for _ in range(100):
                    output = model(dummy_input)
                return output

            optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
            sim.compute_encodings(dummy_forward, None)
            output = dummy_forward(model, None)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_tie_quantizers_for_concat(self):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 10, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 10, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 10, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 10, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv5(x)

                return x

        model = Net()
        model = prepare_model(model)

        dummy_input = torch.rand(1, 3, 28, 28)
        def dummy_forward(model, _):
            return model(dummy_input)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        sim.model.conv2b.output_quantizers[0] = sim.model.conv2a.output_quantizers[0]
        sim.model.conv2a.register_forward_hook(lambda layer, input, output: print(f"sim.model.conv2a"))
        sim.model.conv2b.register_forward_hook(lambda layer, input, output: print(f"sim.model.conv2b"))

        sim.compute_encodings(dummy_forward, None)

        assert sim.model.conv2a.output_quantizers[0].encoding.min == sim.model.conv2b.output_quantizers[0].encoding.min
        assert sim.model.conv2a.output_quantizers[0].encoding.max == sim.model.conv2b.output_quantizers[0].encoding.max

        # Couple of forward passes - to see if inference works
        print(sim)

        sim.model.conv2a.register_forward_hook(lambda layer, input, output: print(f"sim.model.conv2a"))
        sim.model.conv2b.register_forward_hook(lambda layer, input, output: print(f"sim.model.conv2b"))
        print("-" * 20)
        dummy_forward(sim.model, None)
        print("-" * 20)
        dummy_forward(sim.model, None)
        print("-" * 20)

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

        # A couple of forward-backward passes
        print("Train " + "-" * 20)
        output = dummy_forward(sim.model, None)
        output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Train " + "-" * 20)

        output = dummy_forward(sim.model, None)
        output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        print("Done")

    def test_gradient_accumulation(self):
        quantsim_config = {
            "defaults":
                {
                    "ops":
                        {
                            "is_output_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "True",
                    "per_channel_quantization": "True",
                },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file= Path(tmp_dir, "quantsim_config.json")
            with open(config_file, "w") as f:
                json.dump(quantsim_config, f)

            model = ConvReluModel()
            dummy_input = torch.rand(16, 3, 28, 28)
            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    config_file=config_file)
            sim.compute_encodings(evaluate, dummy_input)

            inputs = torch.rand(32, 3, 28, 28)
            grads = None
            for i in range(20):
                out = sim.model(inputs)
                loss = out.flatten().sum()
                loss.backward()
                if grads is None:
                    grads = {name: param.grad.clone() for name, param in sim.model.named_parameters()}

                for name, param in sim.model.named_parameters():
                    assert torch.allclose(param.grad, grads[name] * (i+1))

    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_no_hard_stop_upon_reused_modules(self, quant_scheme):
        """
        Forward & backward shouldn't hard-stop even if the model contains reused Modules
        """
        model = test_models.ModelWithReusedNodes()
        model.eval()
        shape = (1, 3, 32, 32)
        dummy_input = torch.randn(shape)
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=quant_scheme)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

        model.train()
        for _ in range(3):
            inp = torch.randn(shape)
            out = sim.model(inp)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_save_and_load_with_reused_weights(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))
        model[0].weight = model[1].weight
        sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 10))

        sim.compute_encodings(lambda model, _: model(torch.randn(1, 10)), None)

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.save_encodings_to_json(tmpdir, "model_enc")
            sim.load_encodings(os.path.join(tmpdir, "model_enc.json"))

    @pytest.mark.parametrize(
        'quant_scheme',
        [QuantScheme.training_range_learning_with_tf_init,
         QuantScheme.training_range_learning_with_tf_enhanced_init]
    )
    @pytest.mark.parametrize(
        'config_file',
        [None, get_path_for_per_channel_config()]
    )
    def test_module_with_list_input(self, quant_scheme, config_file):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = test_models.ModuleWithListInputModel()
        model.eval()
        model.to(device)

        shape = (1, 32, 128)
        dummy_inputs = torch.randn(shape, device=device)
        sim = QuantizationSimModel(model,
                                   dummy_input=dummy_inputs,
                                   quant_scheme=quant_scheme,
                                   config_file=config_file)
        sim.compute_encodings(forward_pass_callback=lambda m, _: m(dummy_inputs),
                              forward_pass_callback_args=None)

        sim.model.train()
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        for _ in range(3):
            inp = torch.randn(shape, device=device)
            out = sim.model(inp)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

    @pytest.mark.parametrize("hw_version", ['V69', 'V73', 'V75'])
    @pytest.mark.parametrize("quant_scheme", [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    def test_exception_for_embedding(self, hw_version, quant_scheme):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = test_models.ModelWithEmbedding().to(device)
        dummy_input = torch.tensor([[1, 4, 2, 5], [4, 3, 2, 7]], dtype=torch.int64, device=device)

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {
                    "Gather":
                    {
                      "is_output_quantized": "False",
                      "per_channel_quantization": "False"
                    },
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       default_param_bw=4,
                                       default_output_bw=16,
                                       config_file=config_path)

        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)

        wrapper = sim.model.embedding
        weight_quantizer = wrapper.param_quantizers["weight"]
        output_quantizer = wrapper.output_quantizers[0]
        if sim._hw_version in {"V73", "V75"}:
            assert weight_quantizer.bitwidth == output_quantizer.bitwidth
            assert not isinstance(weight_quantizer.encoding, list)
            assert weight_quantizer.use_symmetric_encodings == output_quantizer.use_symmetric_encodings
        else:
            assert weight_quantizer.bitwidth != output_quantizer.bitwidth
            assert weight_quantizer.use_symmetric_encodings != output_quantizer.use_symmetric_encodings

    @pytest.mark.parametrize('hw_version', ['V69', 'V73', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    def test_exception_for_groupnorm(self, hw_version, quant_scheme):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithGroupNorm().to(device)
        dummy_input = torch.randn((1, 6, 2, 2), device=device)

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {
                    "GroupNorm":
                    {
                      "per_channel_quantization": "False",
                      "params": {
                        "bias":
                        {
                          "is_quantized": "True"
                        }
                      }
                    },
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       default_param_bw=4,
                                       default_output_bw=16,
                                       config_file=config_path)

        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)

        wrapper = sim.model.gn
        weight_quantizer = wrapper.param_quantizers['weight']
        bias_quantizer = wrapper.param_quantizers['bias']
        output_quantizer = wrapper.output_quantizers[0]
        if sim._hw_version in {'V73', 'V75'}:
            assert weight_quantizer.bitwidth == output_quantizer.bitwidth
            assert weight_quantizer.use_symmetric_encodings == output_quantizer.use_symmetric_encodings
            assert not isinstance(weight_quantizer.encoding, list)
            assert bias_quantizer.enabled
            assert bias_quantizer.bitwidth == output_quantizer.bitwidth
            assert bias_quantizer.use_symmetric_encodings == output_quantizer.use_symmetric_encodings
        else:
            assert weight_quantizer.bitwidth != output_quantizer.bitwidth
            assert weight_quantizer.use_symmetric_encodings != output_quantizer.use_symmetric_encodings
            assert bias_quantizer.bitwidth != output_quantizer.bitwidth
            assert bias_quantizer.use_symmetric_encodings != output_quantizer.use_symmetric_encodings

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    def test_exception_for_matmul_if_input_quantization_disabled(self, hw_version, quant_scheme, default_output_bw):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul().to(device)
        dummy_input = (torch.randn(10, 3, 4, device=device), torch.randn(10, 5, 4, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=default_output_bw,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)
        closest_output_quantizer_of_second_input = sim.model.act2.output_quantizers[0]
        closest_output_quantizer_of_first_input = sim.model.act1.output_quantizers[0]

        if sim._hw_version in {'V73', 'V75'}:
            if closest_output_quantizer_of_second_input.bitwidth == 16:
                assert closest_output_quantizer_of_second_input.use_symmetric_encodings
                assert closest_output_quantizer_of_first_input.bitwidth == 16
            else:
                assert not closest_output_quantizer_of_second_input.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert closest_output_quantizer_of_second_input.bitwidth == 8
            assert closest_output_quantizer_of_second_input.use_symmetric_encodings
        else:
            assert not closest_output_quantizer_of_second_input.use_symmetric_encodings

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    @pytest.mark.parametrize('producer_output_quantization_enabled', [False, True])
    def test_exception_for_matmul_if_input_quantization_enabled(self, hw_version, quant_scheme, default_output_bw,
                                                                producer_output_quantization_enabled):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Temporarily add aimet_modules.MatMul entry to apply op_type config
        original_map_torch_types_to_onnx = copy.deepcopy(onnx_utils.map_torch_types_to_onnx)
        onnx_utils.map_torch_types_to_onnx[aimet_modules.MatMul] = ['MatMul']

        model = test_models.ModelWithMatMul().to(device)
        dummy_input = (torch.randn(10, 3, 4, device=device), torch.randn(10, 5, 4, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {}
            },
            "params": {},
            "op_type": {
                "MatMul": {"is_input_quantized": "True"},
                "Relu": {"is_output_quantized": str(producer_output_quantization_enabled)}
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=default_output_bw,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)
        first_input_quantizer, second_input_quantizer = sim.model.matmul.input_quantizers
        if sim._hw_version in {'V73', 'V75'}:
            if second_input_quantizer.bitwidth == 16:
                assert second_input_quantizer.use_symmetric_encodings
                assert first_input_quantizer.bitwidth == 16
            else:
                assert not second_input_quantizer.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert second_input_quantizer.use_symmetric_encodings
            assert second_input_quantizer.bitwidth == 8
        else:
            assert not second_input_quantizer.use_symmetric_encodings

        # Restore original mapping dictionary
        onnx_utils.map_torch_types_to_onnx = original_map_torch_types_to_onnx

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V69', 'V73', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    def test_exception_for_matmul_edge_case(self, hw_version, quant_scheme):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul().to(device)
        dummy_input = (torch.randn(10, 3, 4, device=device), torch.randn(10, 5, 4, device=device))
        quantsim_config = {
            "defaults": {"hw_version": hw_version, "ops": {"is_output_quantized": "True"}, "params": {}},
            "params": {},
            "op_type": {
                "Relu": {"is_output_quantized": "False"},
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=16,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)

        first_input_quantizer, second_input_quantizer = sim.model.matmul.input_quantizers
        closest_output_quantizer_of_second_input = sim.model.act2.output_quantizers[0]

        assert not second_input_quantizer.use_symmetric_encodings
        assert not closest_output_quantizer_of_second_input.use_symmetric_encodings

    @pytest.mark.parametrize("hw_version", ['default', 'V66', 'V68', 'V69', 'V73', 'V75'])
    @pytest.mark.parametrize(
        "quant_scheme",
        [
            QuantScheme.post_training_tf,
            QuantScheme.training_range_learning_with_tf_init,
            QuantScheme.post_training_tf_enhanced,
            QuantScheme.training_range_learning_with_tf_enhanced_init,
        ],
    )
    @pytest.mark.parametrize("default_output_bw", [8, 16])
    def test_exception_for_matmul_edge_case2(
            self, hw_version, quant_scheme, default_output_bw
    ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = test_models.ModelWithMatMul2().to(device)
        dummy_input = (
            torch.randn(10, 3, 4, device=device),
            torch.randn(10, 5, 4, device=device),
        )

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"},
                "params": {},
            },
            "params": {},
            "op_type": {
                "Relu": {"is_output_quantized": "False"},
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")

            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(
                model,
                dummy_input,
                quant_scheme,
                config_file=config_path,
                default_output_bw=default_output_bw,
                default_param_bw=4,
            )

        sim.compute_encodings(
            lambda sim_model, _: sim_model(*dummy_input), forward_pass_callback_args=None
        )

        closest_output_quantizer_of_second_input = sim.model.act3.output_quantizers[0]
        closest_output_quantizer_of_first_input = sim.model.act1.output_quantizers[0]
        if sim._hw_version in {'V73', 'V75'}:
            if closest_output_quantizer_of_second_input.bitwidth == 16:
                assert closest_output_quantizer_of_second_input.use_symmetric_encodings
                assert closest_output_quantizer_of_first_input.bitwidth == 16
            else:
                assert not closest_output_quantizer_of_second_input.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert closest_output_quantizer_of_second_input.bitwidth == 8
            assert closest_output_quantizer_of_second_input.use_symmetric_encodings
        else:
            assert not closest_output_quantizer_of_second_input.use_symmetric_encodings

    @pytest.mark.parametrize("hw_version", ['default', 'V75'])
    def test_exception_for_unused_matmul(self, hw_version):
        """
        Check that quantsim init doesn't fail with unused matmul ops
        """
        device = "cpu"
        model = test_models.ModelWithUnusedMatmul().to(device)
        dummy_input = (torch.randn(10, device=device), torch.randn(10, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input,
                                       config_file=config_path,
                                       default_output_bw=16,
                                       default_param_bw=4)

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    def test_exception_if_matmul_has_model_input(self, hw_version, quant_scheme, default_output_bw):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul4().to(device)
        dummy_input = (torch.randn(10, device=device), torch.randn(10, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=default_output_bw,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)

        closest_output_quantizer_of_first_input = sim.model.act1.output_quantizers[0]
        _, second_input_quantizer = sim.model.matmul.input_quantizers

        if sim._hw_version in {'V73', 'V75'}:
            if second_input_quantizer.bitwidth == 16:
                assert second_input_quantizer.use_symmetric_encodings
                assert closest_output_quantizer_of_first_input.bitwidth == 16
            else:
                assert not second_input_quantizer.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert second_input_quantizer.bitwidth == 8
            assert second_input_quantizer.use_symmetric_encodings
        else:
            assert not second_input_quantizer.use_symmetric_encodings

    @pytest.mark.parametrize('hw_version', ['V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    def test_exception_if_matmul_has_single_input_with_producer_op(self, hw_version, quant_scheme, default_output_bw):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul6().to(device)
        dummy_input = (torch.randn((2, 2), device=device), torch.randn((2, 2), device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {
                "Transpose":
                {
                  "is_output_quantized": "False"
                }
            },
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=default_output_bw,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)

        first_input_quantizer = sim.model.matmul.input_quantizers[0]
        closest_output_quantizer_of_second_input = sim.model.act1.output_quantizers[0]

        if sim._hw_version in {'V73', 'V75'}:
            if closest_output_quantizer_of_second_input.bitwidth == 16:
                assert closest_output_quantizer_of_second_input.use_symmetric_encodings
                assert first_input_quantizer.bitwidth == 16
            else:
                assert not closest_output_quantizer_of_second_input.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert closest_output_quantizer_of_second_input.bitwidth == 8
            assert closest_output_quantizer_of_second_input.use_symmetric_encodings
        else:
            assert not closest_output_quantizer_of_second_input.use_symmetric_encodings

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init,
                                              QuantScheme.post_training_tf_enhanced,
                                              QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    def test_exception_if_both_matmul_inputs_are_model_input(self, hw_version, quant_scheme, default_output_bw):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul5().to(device)
        dummy_input = (torch.randn(10, device=device), torch.randn(10, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme,
                                       config_file=config_path,
                                       default_output_bw=default_output_bw,
                                       default_param_bw=4)

        sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                              forward_pass_callback_args=None)

        first_input_quantizer, second_input_quantizer = sim.model.matmul.input_quantizers

        if sim._hw_version in {'V73', 'V75'}:
            if second_input_quantizer.bitwidth == 16:
                assert second_input_quantizer.use_symmetric_encodings
                assert first_input_quantizer.bitwidth == 16
            else:
                assert not second_input_quantizer.use_symmetric_encodings
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert second_input_quantizer.bitwidth == 8
            assert second_input_quantizer.use_symmetric_encodings
        else:
            assert not second_input_quantizer.use_symmetric_encodings

    def test_get_closest_producer_wrapper(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.permute = elementwise_ops.Permute()
                self.reshape = elementwise_ops.Reshape()
                self.relu = torch.nn.ReLU()

            def forward(self, inp):
                x = self.permute(inp, (1, 0))
                x = self.reshape(x, (3, 4))
                x = self.relu(x)
                return x

        model = Model()
        dummy_input = torch.randn(2, 6)

        qsim = QuantizationSimModel(model, dummy_input)
        qsim.model.reshape.output_quantizers[0].enabled = False
        qsim.model.permute.output_quantizers[0].enabled = False
        qsim.compute_encodings(lambda m, _: m(dummy_input), None)

        module_to_quant_wrapper = {}
        for _, wrapper in qsim.quant_wrappers():
            module_to_quant_wrapper[wrapper._module_to_wrap] = wrapper
        # Use connected graph op corresponding to reshape to test
        closest_wrapper = qsim._get_closest_producer_wrapper(qsim.connected_graph.ordered_ops[1], module_to_quant_wrapper)
        assert closest_wrapper == qsim.model.permute


def test_export_to_safetensors():
    torch.manual_seed(0)
    model = SmallMnistNoDropoutWithPassThrough()
    model.eval()
    dummy_data = torch.randn(1, 1, 32, 32)
    sim = QuantizationSimModel(model, dummy_data)
    sim.compute_encodings(lambda m, itr: m(dummy_data), None)
    with tempfile.TemporaryDirectory() as tempDir:
        sim.export_weights_to_safetensors(tempDir, 'sim_export')
        assert(os.path.exists(os.path.join(tempDir, 'sim_export'+'.safetensors')))


@pytest.mark.cuda
@pytest.mark.parametrize('input_dims', (2, 3, 4))
def test_fused_qdq_linear(input_dims):
    torch.manual_seed(2023)

    in_features = 200
    out_features = 100

    input_shape = tuple(in_features if dim == input_dims-1 else 4 for dim in range(input_dims))
    x = torch.randn(input_shape)
    weight = torch.randn((out_features, in_features))
    bias = torch.randn(out_features)
    linear = torch.nn.Linear(in_features, out_features).cuda()
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    linear_wrapper = LearnedGridQuantWrapper(linear,
                                             weight_bw=4,
                                             activation_bw=16,
                                             round_mode='round_nearest',
                                             quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                             device='cuda:0')
    param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(linear_wrapper)
    for quantizer in param_quantizers + input_quantizers + output_quantizers:
        quantizer.enabled = False

    linear_wrapper.param_quantizers['weight'].enabled = True
    linear_wrapper.weight_encoding_min = torch.nn.Parameter(torch.tensor([-1.0 for _ in range(out_features)]))
    linear_wrapper.weight_encoding_max = torch.nn.Parameter(torch.tensor([1.0 for _ in range(out_features)]))
    linear_wrapper.cuda()

    _assert_same_results_with_or_without_recompute(linear_wrapper, x)

def test_multiple_references_to_same_module():
    class TwoRelu(torch.nn.Module):
        def __init__(self):
            super(TwoRelu, self).__init__()
            self.relu1 = torch.nn.ReLU()
            self.relu2 = torch.nn.ReLU()

        def forward(self, inp):
            x = self.relu1(inp)
            x = self.relu2(x)
            return x

    class OuterModel(torch.nn.Module):
        def __init__(self):
            super(OuterModel, self).__init__()
            self.two_relu = TwoRelu()
            self.seq = torch.nn.Sequential(self.two_relu)

        def forward(self, inp):
            x = self.seq(inp)
            return x

    model = OuterModel()
    dummy_input = torch.randn(1, 2)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)
    assert qsim.model.two_relu is qsim.model.seq[0]
    assert isinstance(qsim.model.two_relu.relu1, StaticGridQuantWrapper)
    assert isinstance(qsim.model.two_relu.relu1._module_to_wrap, torch.nn.ReLU)


def _assert_same_results_with_or_without_recompute(wrapper: LearnedGridQuantWrapper, x):
    with no_recompute():
        # If recomputation is disabled, we use the default forward/backward functions
        # defined in torch.nn.Linear
        logits = wrapper(x.cuda())
        logits.sum().backward()

    weight_grad = wrapper._module_to_wrap.weight.grad.clone().detach()
    min_grad = wrapper.weight_encoding_min.grad.clone().detach()
    max_grad = wrapper.weight_encoding_max.grad.clone().detach()

    wrapper._module_to_wrap.weight.grad = None
    wrapper.weight_encoding_min.grad = None
    wrapper.weight_encoding_max.grad = None

    with enable_recompute():
        # If recomputation is enabled, AIMET uses its custom forward/backward functions
        # that perform recompute during bacward to reduce memory footprint
        logits_with_recompute = wrapper(x.cuda())
        logits_with_recompute.sum().backward()

    weight_grad_with_recompute = wrapper._module_to_wrap.weight.grad.clone().detach()
    min_grad_with_recompute = wrapper.weight_encoding_min.grad.clone().detach()
    max_grad_with_recompute = wrapper.weight_encoding_max.grad.clone().detach()

    # Assert logits and grads are equal with/without recomputation
    assert logits.equal(logits_with_recompute)
    assert weight_grad.equal(weight_grad_with_recompute)
    assert min_grad.equal(min_grad_with_recompute)
    assert max_grad.equal(max_grad_with_recompute)


class _DynamicConv2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = DynamicConv2d()

    def forward(self, input, weight, bias=None):
        return self.conv(input, weight, bias)

def test_dynamic_conv2d_export():
    model = _DynamicConv2dModel()
    dummy_input = (torch.randn(1, 3, 10, 10), # input
                   torch.randn(3, 3, 3, 3))   # weight
    sim = QuantizationSimModel(model, dummy_input)
    sim.compute_encodings(lambda model, _: model(*dummy_input), None)

    # Shouldn't throw error during export
    with tempfile.TemporaryDirectory() as tmp_dir:
        sim.export(tmp_dir, "dynamic_conv", dummy_input)



@pytest.mark.parametrize("dtype", [QuantizationDataType.float, QuantizationDataType.int])
def test_bool_quantization(dtype):
    """With an actual leaf module"""

    class Compare2Tensors(torch.nn.Module):
        def __init__(self):
            super(Compare2Tensors, self).__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x == y

    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.compare = Compare2Tensors()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.compare(x, y)

    def forward_pass(model, args):
        model(*dummy_input)

    dummy_input = torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12)
    model = MyModel()
    sim = QuantizationSimModel(model, dummy_input, default_output_bw=16, default_param_bw=16, default_data_type=dtype)

    sim.compute_encodings(forward_pass, None)

    assert sim.model.compare.output_quantizers[0].enabled == False