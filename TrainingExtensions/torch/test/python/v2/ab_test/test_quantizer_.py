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
import copy
import itertools
import logging
import json as json
import os
import tempfile
from packaging import version
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from packaging.version import Version
from torchvision import models

from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_common.utils import AimetLogger
from aimet_torch import onnx_utils
from aimet_torch import utils
import aimet_torch.v1.nn.modules.custom as aimet_modules
from aimet_torch.model_preparer import prepare_model
from ..models_.test_models import TwoLayerBidirectionalLSTMModel, SingleLayerRNNModel, \
    ModelWithTwoInputs, SimpleConditional, RoiModel, InputOutputDictModel, Conv3dModel
from ..models_.models_to_test import ModelWith5Output
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper, QcQuantizeStandalone, StaticGridQuantWrapper
from aimet_torch.v1.quantsim import check_accumulator_overflow, compute_encodings_for_sims
import aimet_torch.v2.nn as aimet_nn
from aimet_torch.v2.nn.fake_quant._legacy_impl import _FakeQuantizedUnaryOpMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.v2.quantsim import QuantizationSimModel

from ..models_ import test_models
from ..models_ import mnist_torch_model

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


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


FakeQuantizedClamp = _FakeQuantizedUnaryOpMixin.wrap(Clamp)


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


# From https://github.com/quic/aimet/blob/8ed479b24010834bfea09885cf6879b9bd916e8a/TrainingExtensions/torch/test/python/test_quantizer.py#L467
class TestQuantizationSimStaticGrad:
    def test_is_quantizable_module_negative(self):
        """With a non-quantizable module"""
        conv1 = aimet_nn.QuantizedConv2d(1, 10, 5)
        assert not QuantizationSimModel._is_quantizable_module(conv1)

    def verify_quantization_wrappers(self, original_model, quantized_model):
        """Test utility to determine if quantization wrappers were added correctly"""

        def is_leaf(module):
            if isinstance(module, aimet_nn.BaseQuantizationMixin):
                return True
            return len(list(module.modules())) == 1 and\
                    not isinstance(module, (nn.ModuleList, nn.ModuleDict))

        # All leaf modules in the original model
        orig_modules = {
            name: module for name, module in original_model.named_modules()
            if is_leaf(module) and not isinstance(module, QuantizeDequantize)
        }

        # All QcQuantized modules in the quantized model
        quant_modules = {
            name: module for name, module in quantized_model.named_modules()
            if isinstance(module, aimet_nn.BaseQuantizationMixin)
        }

        assert orig_modules.keys() == quant_modules.keys()

        for module_name, orig_module in orig_modules.items():
            # if original module class was a quantized module,
            # quantsim should not wrap it again with a quantization mixin
            quant_module = quant_modules[module_name]

            if isinstance(orig_module, aimet_nn.BaseQuantizationMixin):
                assert type(orig_module) == type(quant_module)
                continue

            orig_cls = type(orig_module)
            assert isinstance(quant_module, aimet_nn.BaseQuantizationMixin) and \
                   isinstance(quant_module, orig_cls)
            assert isinstance(quant_module.get_original_module(), orig_cls)

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
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12),
                                   quant_scheme=QuantScheme.post_training_tf)

        self.verify_quantization_wrappers(model, sim.model)

    @pytest.mark.skip("_add_quantization_wrappers not supported yet")
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

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12), in_place=True,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Add wrappers again, expect to be a nop
        sim._add_quantization_wrappers(model, num_inout_tensors={}, default_data_type=QuantizationDataType.int)

        self.verify_quantization_wrappers(model, sim.model)

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

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12),
                                   quant_scheme=QuantScheme.post_training_tf)

        self.verify_quantization_wrappers(model, sim.model)

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
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12),
                                   quant_scheme=QuantScheme.post_training_tf)
        self.verify_quantization_wrappers(model, sim.model)

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
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12),
                                   quant_scheme=QuantScheme.post_training_tf)

        self.verify_quantization_wrappers(model, sim.model)

    def test_add_quantization_wrappers_with_modulelist(self):
        """With a one-deep model using ModuleList"""
        q_conv2d = aimet_nn.QuantizedConv2d(1, 10, 5)
        q_conv2d.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                           bitwidth=8,
                                                           symmetric=False)
        q_conv2d.param_quantizers['weight'] = QuantizeDequantize(shape=(),
                                                                 bitwidth=8,
                                                                 symmetric=True)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1, 32),
                    nn.Linear(32, 64),
                    nn.Conv2d(1, 32, 5),
                    q_conv2d,
                ])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        with pytest.raises(RuntimeError):
            sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12),
                                       quant_scheme=QuantScheme.post_training_tf)


    def test_add_quantization_wrappers_with_modulelist_two_deep(self):
        """With a two-deep model using ModuleList"""
        q_conv2d = aimet_nn.QuantizedConv2d(1, 10, 5)
        q_conv2d.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                           bitwidth=8,
                                                           symmetric=False)
        q_conv2d.param_quantizers['weight'] = QuantizeDequantize(shape=(),
                                                                 bitwidth=8,
                                                                 symmetric=True)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1, 32),
                    nn.Linear(32, 64),
                    nn.Conv2d(3, 32, kernel_size=3),
                ])
                self.layers_deep = nn.ModuleList([
                    nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                    nn.Linear(3, 32),
                    nn.Linear(32, 64),
                    nn.Conv2d(1, 32, 5),
                    q_conv2d,
                ])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        with pytest.raises(RuntimeError):
            sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12),
                                       quant_scheme=QuantScheme.post_training_tf)

    def test_add_quantization_wrappers_with_modulelist_with_layers_to_ignore(self):
        """With a two-deep model using ModuleList and layers_to_ignore"""

        model = ModuleListModel()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12),
                                   quant_scheme=QuantScheme.post_training_tf)
        layers_to_exclude = [sim.model.layers_deep[1], sim.model.layers_deep[3], sim.model.layers_deep[5]]
        sim.exclude_layers_from_quantization(layers_to_exclude)

        assert isinstance(sim.model.layers[0], aimet_nn.QuantizedLinear)
        assert isinstance(sim.model.layers[1], aimet_nn.QuantizedLinear)
        assert isinstance(sim.model.layers[2], aimet_nn.QuantizedConv2d)

        assert isinstance(sim.model.layers_deep[0][0], aimet_nn.QuantizedBatchNorm2d)
        assert isinstance(sim.model.layers_deep[0][1], aimet_nn.QuantizedReLU)

        assert type(sim.model.layers_deep[1]) == nn.Linear # layer ignored, so no QcQuantizeWrapper wrapper
        assert isinstance(sim.model.layers_deep[2], aimet_nn.QuantizedLinear)

        assert type(sim.model.layers_deep[3]) == nn.Conv2d # layer ignored, so no QcQuantizeWrapper wrapper

        # non leaf layer specified, check that all submodules had wrappers removed
        assert type(sim.model.layers_deep[5][0]) == nn.MaxPool2d
        assert type(sim.model.layers_deep[5][1]) == nn.PReLU

        assert len(sim._excluded_layer_names) == 4
        assert 'layers_deep.1' in sim._excluded_layer_names
        assert 'layers_deep.3' in sim._excluded_layer_names
        assert 'layers_deep.5.0' in sim._excluded_layer_names
        assert 'layers_deep.5.1' in sim._excluded_layer_names

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'modulelist_with_layers_to_ignore', dummy_input=torch.rand(1, 3, 12, 12))
            with open(f"{tmp_dir}/modulelist_with_layers_to_ignore.encodings", "r") as encodings_file:
                encodings = json.load(encodings_file)

        assert 'layers_deep.1' in encodings['excluded_layers']
        assert 'layers_deep.3' in encodings['excluded_layers']
        assert 'layers_deep.5.0' in encodings['excluded_layers']
        assert 'layers_deep.5.1' in encodings['excluded_layers']
        assert len(encodings['excluded_layers']) == 4

    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""
        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        sim.model.conv1_a.param_quantizers['weight'].symmetric = True

        # Quantize
        sim.compute_encodings(forward_pass, None)
        model(*dummy_input)

        # save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'two_input_model', dummy_input)

    @pytest.mark.skip("load_encodings_to_sim not implemented")
    def test_model_with_two_inputs_fp16(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16,
                                   dummy_input=dummy_input,
                                   default_data_type=QuantizationDataType.float)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'two_input_model_fp16', dummy_input)
            encoding_file_path_pytorch = os.path.join(tmp_dir, 'two_input_model_fp16_torch.encodings')
            load_encodings_to_sim(sim, encoding_file_path_pytorch)

        layer = sim.model.conv1_a
        assert isinstance(layer, aimet_nn.QuantizedConv2d)

        assert isinstance(layer.input_quantizers[0], FloatQuantizeDequantize)
        assert layer.input_quantizers[0].is_float16()
        assert isinstance(layer.output_quantizers[0], FloatQuantizeDequantize)
        assert layer.output_quantizers[0].is_float16()
        assert isinstance(layer.param_quantizers['weight'], FloatQuantizeDequantize)
        assert layer.param_quantizers['weight'].is_float16()

    def test_model_with_two_inputs_one_to_add(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        model = ModelWithTwoInputsOneToAdd()

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        assert 2 == len(sim.model.add.input_quantizers)
        assert sim.model.add.input_quantizers[0] is None
        assert isinstance(sim.model.add.input_quantizers[1], QuantizeDequantize)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'two_input_model_one_with_add', dummy_input)
            onnx_model = onnx.load(f'{tmp_dir}/two_input_model_one_with_add.onnx')

            for node in onnx_model.graph.node:
                if node.name == 'add':
                    break
            assert 2 == len(node.input)
            model_input_tensor = node.input[1]

            with open(f"{tmp_dir}/two_input_model_one_with_add.encodings", "r") as encodings_file:
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
        sim = QuantizationSimModel(resnet18, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'resnet18', dummy_input)
            with open(f'{tmp_dir}/resnet18.encodings') as json_file:
                encoding_data = json.load(json_file)

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
            with open(f'{tmp_dir}/resnet18_with_quant_args.encodings') as json_file:
                encoding_data = json.load(json_file)

        assert "quantizer_args" in encoding_data
        quantizer_args = encoding_data["quantizer_args"]
        assert quantizer_args["activation_bitwidth"] == 16
        assert quantizer_args["param_bitwidth"] == 16
        assert not quantizer_args["per_channel_quantization"]
        assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf.name
        assert quantizer_args["dtype"] == "int"
        assert "is_symmetric" in quantizer_args

    @pytest.mark.skip('export to torchscript not supported yet')
    def test_export_to_torch_script(self):
        """ test export functionality on ResNet18 """

        resnet50 = models.resnet50()
        resnet50.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet50, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'resnet50', dummy_input, export_to_torchscript=True)
            with open(f'{tmp_dir}/resnet50.encodings') as json_file:
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
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with torch.no_grad():
            sim.model.conv1_a.param_quantizers['weight'].min.copy_(-10)
            sim.model.conv1_a.param_quantizers['weight'].max.copy_(10)
            sim.model.conv1_a.output_quantizers[0].max.copy_(30)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save encodings
            sim.export(tmp_dir, 'two_input_model', dummy_input)

            # check the encodings
            with open(os.path.join(tmp_dir, 'two_input_model.encodings'), 'r') as fp:
                encodings = json.load(fp)

                activation_encodings = encodings['activation_encodings']
                param_encodings = encodings['param_encodings']
                assert 16 == len(activation_encodings)
                assert 7 == len(param_encodings['conv1_a.weight'][0])
                min = param_encodings['conv1_a.weight'][0]['min']
                max = param_encodings['conv1_a.weight'][0]['max']
                scale = (max - min) / 255
                offset = round(min / scale)
                assert scale == pytest.approx(20/255)
                assert offset == -128

            # check the exported model
            loaded_model = torch.load(f'{tmp_dir}/two_input_model.pth')
            loaded_model(torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))

    def test_no_fine_tuning_tf(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        assert isinstance(sim.model.conv1, aimet_nn.QuantizedConv2d)

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

        dummy_input = torch.ones(2, 1, 3, 3).to('cuda:0')
        dummy_input[1] += 1

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=dummy_input)
        sim.model.to('cuda:0')

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        weight_single_gpu = sim.model.conv1.weight.clone().detach()
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1.weight.grad.clone().detach()

        sim.model.conv1.weight.grad = None

        sim.model = torch.nn.DataParallel(sim.model)
        weight_multi_gpu = sim.model.module.conv1.weight.clone().detach()
        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        grad_multi_gpu = sim.model.module.conv1.weight.grad.clone().detach()

        assert torch.allclose(weight_single_gpu, weight_multi_gpu)
        assert torch.allclose(output_single_gpu, output_multi_gpu)
        assert torch.allclose(grad_single_gpu, grad_multi_gpu)

    def test_input_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        sim.model.conv1.input_quantizers[0] = QuantizeDequantize((), 8, False)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert sim.model.conv1.input_quantizers[0].is_initialized()

    def test_inputs_shared_constant_intermediate_quantization(self):
        """"""
        model = ModelInputsSharedConstantIntermediate()

        dummy_input = (torch.randn(1, 10, 10, 10), torch.randn(1, 10, 10, 10), torch.randn(1, 10, 10, 10))

        def forward_pass(model, args):
            model(*dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        sim.compute_encodings(forward_pass, None)

        # check mul's first input quantizer is real input(enable) , second input is intermediate (disable)
        assert isinstance(sim.model.mul.input_quantizers[0], QuantizeDequantize)
        assert sim.model.mul.input_quantizers[1] is None

        assert isinstance(sim.model.add1.input_quantizers[0], QuantizeDequantize)
        assert sim.model.add1.input_quantizers[1] is not None
        assert sim.model.add2.input_quantizers[0] is not None
        assert isinstance(sim.model.add2.input_quantizers[1], QuantizeDequantize)

        # save encodings
        input_names = ['a', 'b', 'c']

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'model_inputs_shared_constant_intermediate',
                       dummy_input,
                       onnx_export_args=OnnxExportApiArgs(input_names=input_names))
            with open(f"{tmp_dir}/model_inputs_shared_constant_intermediate.encodings", "r") as encodings_file:
                activation_encoding_tensors = set(json.load(encodings_file)['activation_encodings'].keys())
                assert set(input_names).issubset(activation_encoding_tensors)

    @pytest.mark.skip('Disabling input quantizers of constant scalars is not supported yet')
    def test_constant_quantization(self):
        model = ModelWithConstantQuantization()
        dummy_input = torch.rand(1, 2)
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        assert sim.model.add.input_quantizers[0] is None
        assert isinstance(sim.model.add.input_quantizers[1], QuantizeDequantize)
        assert isinstance(sim.model.add2.input_quantizers[0], QuantizeDequantize)
        assert sim.model.add2.input_quantizers[1] is None
        assert sim.model.add3.input_quantizers[0] is None
        assert isinstance(sim.model.add3.input_quantizers[1], QuantizeDequantize)

        sim.compute_encodings(lambda m, _: m(dummy_input), None)

        # model.add/add2 take constant scalar as input.
        # We expect the quantizers to not have any encoding stats and thus
        assert not sim.model.add.input_quantizers[1].is_initialized()
        assert not sim.model.add2.input_quantizers[0].is_initialized()
        assert sim.model.add3.input_quantizers[1].is_initialized()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'model_with_constant_quantization', dummy_input)
            with open(f"{tmp_dir}/model_with_constant_quantization.encodings", "r") as encodings_file:
                activation_encoding_tensors = set(json.load(encodings_file)['activation_encodings'].keys())
                assert len(activation_encoding_tensors) == 6

    def test_input_and_output_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))

        sim.model.conv1.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                                  bitwidth=8,
                                                                  symmetric=False)
        sim.model.conv1.input_quantizers[0] = QuantizeDequantize(shape=(),
                                                                 bitwidth=8,
                                                                 symmetric=False)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert sim.model.conv1.output_quantizers[0].is_initialized()
        assert sim.model.conv1.input_quantizers[0].is_initialized()

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

        assert isinstance(sim.model.conv3.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

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

        assert sim.model.conv3.input_quantizers[0] is None
        assert isinstance(sim.model.add1.output_quantizers[0], QuantizeDequantize)

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

        assert isinstance(sim.model.conv4a.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv4b.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

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

        assert isinstance(sim.model.conv4a.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

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

        assert isinstance(sim.model.conv3.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

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

        assert isinstance(sim.model.conv3.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

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

        assert isinstance(sim.model.conv3.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.conv5.input_quantizers[0], QuantizeDequantize)

    def test_no_finetuning_tf(self):
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 28, 28))
        assert isinstance(sim.model.conv1, aimet_nn.QuantizedConv2d)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

    def test_with_standalone_ops(self):
        model = ModelWithStandaloneOps()
        dummy_input = torch.rand(1, 1, 28, 28)

        sim = QuantizationSimModel(model=model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

        # Save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(f"{tmp_dir}/", "encodings_with_standalone_ops", dummy_input)
            with open(f'{tmp_dir}/encodings_with_standalone_ops.encodings') as json_file:
                encoding_data = json.load(json_file)
            # in onnx definition tensor 16 is output of Reshape, to be ignored
            assert "32" not in encoding_data["activation_encodings"].keys()

    def test_layers_to_ignore(self):
        """ Test the  capability to skip quantizing the layers specified by the user"""

        model = SmallMnist()

        dummy_input=torch.rand(1, 1, 28, 28)
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        layers_to_ignore = [sim.model.conv1, sim.model.fc2]
        sim.exclude_layers_from_quantization(layers_to_ignore)

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # Check
        assert type(sim.model.conv1) == nn.Conv2d
        assert isinstance(sim.model.conv2, aimet_nn.QuantizedConv2d)
        assert type(sim.model.fc2) == nn.Linear

        # export and check encodings file has excluded layers listed as string
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'excluded_layers', dummy_input, propagate_encodings=True)

            with open(f'{tmp_dir}/excluded_layers.encodings') as f:
                encodings = json.load(f)
                assert 2 == len(encodings['excluded_layers'])

    def check_quant_params(self, model_layer, loaded_model_layer):
        def assert_equal(quantizer, other):
            if quantizer is None or other is None:
                assert quantizer == other
                return

            assert quantizer.bitwidth == other.bitwidth
            assert quantizer.shape == other.shape
            assert quantizer.symmetric == other.symmetric
            assert torch.equal(quantizer.max, other.max)
            assert torch.equal(quantizer.min, other.min)
            assert quantizer.is_initialized() == other.is_initialized()

        for quantizer, loaded_quantizer in zip(model_layer.input_quantizers,
                                               loaded_model_layer.input_quantizers):
            assert_equal(quantizer, loaded_quantizer)

        for quantizer, loaded_quantizer in zip(model_layer.output_quantizers,
                                               loaded_model_layer.output_quantizers):
            assert_equal(quantizer, loaded_quantizer)

        for quantizer, loaded_quantizer in zip(model_layer.param_quantizers.values(),
                                               loaded_model_layer.param_quantizers.values()):
            assert_equal(quantizer, loaded_quantizer)

    def test_save_and_load(self):
        model = ModelWithStandaloneOps()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(32, 1, 28, 28),
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)

        # Run some inferences - mimic using a forward pass
        sim.model.eval()
        dummy_input = torch.randn((32, 1, 28, 28))
        output_before_save = sim.model(dummy_input)

        # Save quantized model
        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(sim.model, f'{tmp_dir}/xx')
            loaded_model = torch.load(f'{tmp_dir}/xx')

        loaded_model.eval()
        output_after_load = loaded_model(dummy_input)

        self.check_quant_params(sim.model.conv1, loaded_model.conv1)
        self.check_quant_params(sim.model.conv2, loaded_model.conv2)
        self.check_quant_params(sim.model.conv2_drop, loaded_model.conv2_drop)
        self.check_quant_params(sim.model.fc2, loaded_model.fc2)

        assert torch.equal(output_before_save, output_after_load)

    def test_changing_param_quantizer_settings(self):
        """ Test that changing param quantizer settings takes effect after computing encodings is run """
        torch.random.manual_seed(10)
        model = SmallMnist()

        # Skew weights of conv1
        old_weight = model.conv1.weight.detach().clone()
        model.conv1.weight = torch.nn.Parameter(old_weight + .5 * torch.abs(torch.min(old_weight)), requires_grad=False)

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28),
                                   quant_scheme=QuantScheme.post_training_tf)

        # Check that no encoding is present for param quantizer
        assert not sim.model.conv1.param_quantizers['weight'].is_initialized()

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)
        asym_min = sim.model.conv1.param_quantizers['weight'].get_min()
        asym_max = sim.model.conv1.param_quantizers['weight'].get_max()
        assert 8 == sim.model.conv1.param_quantizers['weight'].bitwidth

        # Check that offset is still symmetric
        assert sim.model.conv1.param_quantizers['weight'].get_offset().item() == 0

        # Change param quantizer to symmetric and new bitwidth
        sim.model.conv1.param_quantizers['weight'].symmetric = False
        sim.model.conv1.param_quantizers['weight'].bitwidth = 4
        sim.compute_encodings(dummy_forward_pass, None)
        sym_min = sim.model.conv1.param_quantizers['weight'].get_min()
        sym_max = sim.model.conv1.param_quantizers['weight'].get_max()
        assert 4 == sim.model.conv1.param_quantizers['weight'].bitwidth

        # Check that offset is not relatively symmetric
        assert not sim.model.conv1.param_quantizers['weight'].get_offset().item() == 0

        # Check that mins and maxes have been recomputed
        assert not torch.allclose(asym_min, sym_min)
        assert not torch.allclose(asym_max, sym_max)

    def test_compute_encodings_on_subset_of_modules(self):
        def dummy_forward_pass(model, _):
            conv1_out = model.conv1(torch.randn((1, 1, 28, 28)))
            relu1_out = model.relu1(conv1_out)

        model = SmallMnist()
        model.eval()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28),
                                   quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(dummy_forward_pass, None)

        for name, module in sim.model.named_modules():
            if isinstance(module, aimet_nn.FakeQuantizationMixin):
                if name == 'relu1':
                    assert isinstance(module.output_quantizers[0], QuantizeDequantize)
                elif name in ['conv2', 'relu2', 'relu3', 'fc2', 'log_softmax']:
                    assert not module.output_quantizers[0].is_initialized()
                elif name in ['conv2_drop', 'dropout']:
                    assert module.output_quantizers[0] is None

    def test_rnn_quantization(self):
        """ Test quantizing a model with rnn layer """
        model = SingleLayerRNNModel()
        dummy_input = torch.randn(10, 1, 3)

        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        assert isinstance(sim.model.rnn, aimet_nn.QuantizedRNN)

        sim.compute_encodings(lambda model, _: model(dummy_input), None) # Should not throw error

    def test_lstm_quantization(self):
        """ Test quantizing a model with rnn layer """
        model = TwoLayerBidirectionalLSTMModel()
        dummy_input = torch.randn(10, 1, 3)

        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        assert isinstance(sim.model.recurrent, aimet_nn.QuantizedLSTM)

        sim.compute_encodings(lambda model, _: model(dummy_input), None) # Should not throw error

    def test_quantizing_qc_quantize_module(self):
        """ Test that qc_quantize_module is identified as not quantizable """
        q_rnn = aimet_nn.FakeQuantizedRNN(input_size=3, hidden_size=5, num_layers=1)
        assert not QuantizationSimModel._is_quantizable_module(q_rnn)

    @pytest.mark.skip("Exporting RNN is not supported yet")
    def test_export_recurrent_model(self):
        """ Test export functionality with recurrent models """
        model = TwoLayerBidirectionalLSTMModel()
        dummy_input = torch.randn(10, 1, 3)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Edit part of weights tensor to compare with original model before and after removal of quantize module
        with torch.no_grad():
            sim.model.recurrent.weight_ih_l0[0][0] = 1
        edited_weight = sim.model.recurrent.weight_ih_l0.detach().clone()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'recurrent_save', dummy_input)
            exported_model = torch.load(f'{tmp_dir}/recurrent_save.pth')

            # Check that weight from quantized module was copied to original module successfully
            assert isinstance(exported_model.recurrent, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
            assert torch.equal(edited_weight, exported_model.recurrent.weight_ih_l0)

            with open(f'{tmp_dir}/recurrent_save.encodings') as f:
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

        sim = QuantizationSimModel(model, dummy_input=dummy_input[0],
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, dummy_input[0])

        o_names = ['ab', 'bc', 'ca']
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'dict_input_output_model', dummy_input,
                       onnx_export_args=OnnxExportApiArgs(input_names=list(dummy_input[0].keys()),
                                                          output_names=o_names,
                                                          opset_version=12
                                                          ))
            with open(f'{tmp_dir}/dict_input_output_model.encodings') as json_file:
                encoding_data = json.load(json_file)

            onnx_model = onnx.load(f'{tmp_dir}/dict_input_output_model.onnx')

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

        quantizer = sim.model.mul1.output_quantizers[0]
        assert quantizer.get_legacy_encodings() == [{'dtype': 'float', 'bitwidth': 16}]

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

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        sim.model.sfmax.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                            bitwidth=8,
                                                            symmetric=False)
        sim.model.sfmax.input_quantizers[0] = QuantizeDequantize(shape=(),
                                                           bitwidth=8,
                                                           symmetric=False)
        sim.model.avgpool.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                            bitwidth=8,
                                                            symmetric=False)
        sim.model.avgpool.input_quantizers[0] = QuantizeDequantize(shape=(),
                                                            bitwidth=8,
                                                            symmetric=False)
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'sfmaxavgpool_model', dummy_input)

            with open(f'{tmp_dir}/sfmaxavgpool_model.encodings') as json_file:
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
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'encodings_propagation_false', dummy_input)
            with open(f'{tmp_dir}/encodings_propagation_false.encodings') as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 2

            # Save encodings again - now with propagate encodings flag enabled
            sim.export(tmp_dir, 'encodings_propagation_true', dummy_input, propagate_encodings=True)
            with open(f'{tmp_dir}/encodings_propagation_true.encodings') as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 2

        # verifying the encodings propagation is disabled if output quantizers are disabled.
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        sim.model.ps.output_quantizers[0] = None
        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings again - now with propagate encodings flag enabled
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'encodings_propagation_quant_disabled', dummy_input, propagate_encodings=True)
            with open(f'{tmp_dir}/encodings_propagation_quant_disabled.encodings') as f:
                encodings = json.load(f)['activation_encodings']
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 1

    @pytest.mark.skip("Exporting RNN is not supported yet")
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
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save encodings
            sim.export(tmp_dir, 'encodings_propagation_false', dummy_input)
            with open(f'{tmp_dir}/encodings_propagation_false.encodings') as f:
                encodings = json.load(f)
            assert len(encodings['activation_encodings']) == 8

            # Save encodings again - now with propagate encodings flag enabled
            sim.export(tmp_dir, 'encodings_propagation_true', dummy_input, propagate_encodings=True)
            with open(f'{tmp_dir}/encodings_propagation_true.encodings') as f:
                encodings = json.load(f)['activation_encodings']
                # Only eight entry should have min, max, delta and offset, remaining entries should be propagated
                # with bitwidth and dtype.
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 8

    @pytest.mark.skip("SQNR encoding analyzer not implemented yet")
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

    @pytest.mark.skip("SQNR encoding analyzer not implemented yet")
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

        qsim = QuantizationSimModel(model, dummy_input=(inp, true_tensor),
                                    quant_scheme=QuantScheme.post_training_tf)
        qsim.compute_encodings(forward_callback, forward_pass_callback_args=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            qsim._export_conditional(tmp_dir, 'simple_cond', dummy_input=(inp, false_tensor),
                                     forward_pass_callback=forward_callback, forward_pass_callback_args=None)

            with open(f'{tmp_dir}/simple_cond.encodings') as f:
                encodings = json.load(f)
                # verifying the encoding against default eAI HW cfg
                # activation encodings -- input, linear1 out, prelu1 out, linear2 out, prelu2 out, softmax out
                assert 6 == len(encodings['activation_encodings'])
                # param encoding -- linear 1 & 2 weight & bias, prelu 1 & 2 weight
                assert 4 == len(encodings['param_encodings'])

    @pytest.mark.skip("load_encodings_to_sim not implemented")
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

        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(forward_pass, None)

        assert sim.model.conv.in_channels == 10
        assert sim.model.conv.out_channels == 10

    def test_has_valid_encodings(self):
        def is_initialized(module):
            for quantizer in itertools.chain(module.input_quantizers,
                                             module.output_quantizers,
                                             module.param_quantizers.values()):
                if quantizer and not quantizer.is_initialized():
                    return False
            return True

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
        qsim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 8, 8),
                                    quant_scheme=QuantScheme.post_training_tf)
        modules = [qsim.model.relu1, qsim.model.conv, qsim.model.relu2, qsim.model.unused_module]
        for m in modules:
            assert not is_initialized(m)

        qsim.compute_encodings(lambda m, _: m(torch.randn(1, 3, 8, 8)), None)
        for m in modules:
            if m == qsim.model.unused_module:
                assert not is_initialized(m)
            else:
                assert is_initialized(m)

    @pytest.mark.skip('use_embedded_encodings argument not supported yet')
    def test_save_model_with_embedded_quantization_nodes__(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            sim.export(tmp_dir, 'two_input_model', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                       use_embedded_encodings=True)
            onnx_model = onnx.load(os.path.join(tmp_dir, 'two_input_model' + '_embedded' + '.onnx'))
            onnx_type = set()
            for node in onnx_model.graph.node:
                onnx_type.add(node.op_type)
            assert 'QuantizeLinear' in onnx_type
            assert 'DequantizeLinear' in onnx_type

    @pytest.mark.skip('use_embedded_encodings argument not supported yet')
    def test_save_model_with_embedded_quantization_nodes_fp16(self):
        """Model with more than 1 input"""

        dummy_input=(torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf,
                                   default_data_type=QuantizationDataType.float)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save model
            sim.export(tmp_dir, 'two_input_model_fp16', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                       use_embedded_encodings=True)

            onnx_model = onnx.load(os.path.join(tmp_dir, 'two_input_model_fp16' + '_embedded' + '.onnx'))
            onnx_type = set()
            for node in onnx_model.graph.node:
                onnx_type.add(node.op_type)
            assert 'Cast' in onnx_type

    @pytest.mark.skip('use_embedded_encodings argument not supported yet')
    def test_save_model_with_embedded_quantization_nodes_per_channel(self):
        """Model with more than 1 input"""
        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        for _, qmodule in sim.named_qmodules():
            qmodule.enable_per_channel_quantization()

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Export model with opset_vesrion 13
            sim.export(tmp_dir, 'two_input_model_perchannel', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=13)),
                       use_embedded_encodings=True)
            onnx_model = onnx.load(os.path.join(tmp_dir, 'two_input_model_perchannel' + '_embedded' + '.onnx'))
            onnx_type = set()
            for node in onnx_model.graph.node:
                onnx_type.add(node.op_type)
            assert 'QuantizeLinear' in onnx_type
            assert 'DequantizeLinear' in onnx_type

    @pytest.mark.skip('export to torchscript not supported yet')
    def test_save_model_with_embedded_quantization_nodes_using_torch_script(self):
        """Test export onnx model with embedded torch native quantization nodes using torch script"""

        dummy_input = torch.rand(32, 1, 28, 28)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        model = SmallMnist()
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            QuantizationSimModel.save_model_with_embedded_quantization_nodes(sim.model, tmp_dir, 'two_input_model',
                                                                             dummy_input, export_to_torchscript=True)
            assert(os.path.exists(os.path.join(tmp_dir, 'two_input_model' + '_embedded' + '.torchscript.pth')))

    @pytest.mark.skip("_replace_quantization_wrapper_with_native_torch_quantization_nodes not supported")
    def test_native_pytorch_quantization_nodes_pertensor(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        torch.manual_seed(10)
        dummy_input = torch.rand(32, 1, 28, 28)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                return model(dummy_input)

        model = SmallMnist()
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)

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

    @pytest.mark.skip("_replace_quantization_wrapper_with_native_torch_quantization_nodes not supported")
    def test_native_pytorch_quantization_nodes_perchannel(self):
        """Test export onnx model with embedded torch native quantization nodes"""

        torch.manual_seed(10)
        dummy_input = torch.rand(32, 1, 28, 28)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                return model(dummy_input)

        model = SmallMnist()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=quant_scheme)
        for _, qmodule in sim.named_qmodules():
            qmodule.enable_per_channel_quantization()

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

    @pytest.mark.skipif(version.parse(torch.__version__) >= version.parse("1.13.0"), reason='Not supported in torch >= 1.13')
    def test_export_to_onnx_direct(self):
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
            with open(f'{tmp_dir}/saved_encodings.json') as encodings_file:
                encodings = json.load(encodings_file)
                assert len(encodings['activation_encodings']) == 13
                assert len(encodings['param_encodings']) == 5

    @pytest.mark.skip('compute_encodings_for_sims not supported yet')
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

    @pytest.mark.skip('load_and_freeze_encodings not supported yet')
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

        qsim = QuantizationSimModel(model=model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                    rounding_mode='nearest', default_output_bw=16, default_param_bw=8, in_place=False,
                                    config_file=None)
        def forward_pass(model, dummy_input):
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/temp_partial_torch_encodings.encodings", 'w') as fp:
                json.dump(partial_torch_encodings, fp)

            qsim.load_and_freeze_encodings(f"{tmp_dir}/temp_partial_torch_encodings.encodings")

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
            for name, qmodule in qsim.named_qmodules():
                if name in partial_torch_encodings['activation_encodings']:
                    if 'input' in partial_torch_encodings['activation_encodings'][name]:
                        assert_input_output_quantizers(qmodule.input_quantizers, 'input')
                        input_quant_checked.append(name)
                    if 'output' in partial_torch_encodings['activation_encodings'][name]:
                        assert_input_output_quantizers(qmodule.output_quantizers, 'output')
                        output_quant_checked.append(name)

                param_types = qmodule.param_quantizers.keys()
                for param_type in param_types:
                    module_param_name = name + "." + param_type
                    if module_param_name in partial_torch_encodings['param_encodings']:
                        assert_param_quantizers(qmodule.param_quantizers, name, param_type)
                        param_quant_checked.append(module_param_name)

            actual_input_quant = {k for k, v in partial_torch_encodings['activation_encodings'].items() if 'input' in v}
            actual_output_quant = {k for k, v in partial_torch_encodings['activation_encodings'].items() if 'output' in v}
            actual_param_quant = set(partial_torch_encodings['param_encodings'].keys())

            assert actual_input_quant == set(input_quant_checked)
            assert actual_output_quant == set(output_quant_checked)
            assert actual_param_quant == set(param_quant_checked)

            os.remove("./temp_partial_torch_encodings.encodings")

    @pytest.mark.cuda
    def test_and_compare_quantizer_no_fine_tuning_CPU_and_GPU(self):

        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        dummy_input = torch.rand(1, 1, 28, 28)
        dummy_input_cuda = dummy_input.cuda()

        # create model on CPU
        model_cpu = mnist_torch_model.Net().to('cpu').eval()

        model_gpu = copy.deepcopy(model_cpu).to('cuda')
        cpu_sim_model = QuantizationSimModel(model_cpu, quant_scheme='tf', in_place=True,
                                             dummy_input=dummy_input)
        # Quantize
        cpu_sim_model.compute_encodings(lambda model, input: model(input), dummy_input)

        # create model on GPU
        gpu_sim_model = QuantizationSimModel(model_gpu, quant_scheme='tf', in_place=True,
                                             dummy_input=dummy_input_cuda)
        gpu_sim_model.model.cuda()
        # Quantize
        gpu_sim_model.compute_encodings(lambda model, input: model(input), dummy_input_cuda)

        # check the encodings only min and max
        # Test that first and second are approximately (or not approximately)
        # equal by computing the difference, rounding to the given number of
        # decimal places (default 7), and comparing to zero. Note that these
        # methods round the values to the given number of decimal places
        # (i.e. like the round() function) and not significant digits
        # excluding fc1 since it is part of Matmul->Relu supergroup
        # can't use assertEqual for FC2, so using assertAlmostEquals for FC2
        assert torch.allclose(model_gpu.conv1.output_quantizers[0].get_min().cpu(),
                              model_cpu.conv1.output_quantizers[0].get_min(), rtol=1e-4)
        assert torch.allclose(model_gpu.conv1.output_quantizers[0].get_max().cpu(),
                              model_cpu.conv1.output_quantizers[0].get_max(), rtol=1e-4)

        assert torch.allclose(model_gpu.conv2.output_quantizers[0].get_min().cpu(),
                              model_cpu.conv2.output_quantizers[0].get_min(), rtol=1e-4)
        assert torch.allclose(model_gpu.conv2.output_quantizers[0].get_max().cpu(),
                              model_cpu.conv2.output_quantizers[0].get_max(), rtol=1e-4)

        assert torch.allclose(model_gpu.fc2.output_quantizers[0].get_min().cpu(),
                              model_cpu.fc2.output_quantizers[0].get_min(), rtol=1e-4)
        assert torch.allclose(model_gpu.fc2.output_quantizers[0].get_max().cpu(),
                              model_cpu.fc2.output_quantizers[0].get_max(), rtol=1e-4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            gpu_sim_model.export(tmp_dir, "quantizer_no_fine_tuning__GPU", dummy_input)
            cpu_sim_model.export(tmp_dir, "quantizer_no_fine_tuning__CPU", dummy_input)

        assert torch.device('cuda:0') == next(model_gpu.parameters()).device
        assert torch.device('cpu') == next(model_cpu.parameters()).device


# From https://github.com/quic/aimet/blob/8ed479b24010834bfea09885cf6879b9bd916e8a/TrainingExtensions/torch/test/python/test_quantizer.py#L3015
class TestQuantizationSimLearnedGrid:
    @pytest.mark.cuda
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_range_learning_with_fp16_and_bw_32_quantizers(self, device):
        model = SmallMnistNoDropout()
        model.eval()
        model.to(device)
        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        sim.model.to(device)
        sim.model.conv2.param_quantizers['weight'] = FloatQuantizeDequantize(dtype=torch.float16)
        sim.model.relu2.output_quantizers[0] = None
        sim.compute_encodings(lambda m, _: m(dummy_input), None)

        sim.model.train()
        output = sim.model(copy.deepcopy(dummy_input))
        loss = output.flatten().sum()

        orig_conv1_weight = sim.model.conv1.weight.clone().detach()
        orig_conv1_encoding_max = sim.model.conv1.param_quantizers['weight'].get_max()
        orig_conv2_weight = sim.model.conv2.weight.clone().detach()
        loss.backward()

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        optimizer.step()
        optimizer.zero_grad()

        new_conv1_weight = sim.model.conv1.weight.clone().detach()
        new_conv1_encoding_max = sim.model.conv1.param_quantizers['weight'].get_max()
        new_conv2_weight = sim.model.conv2.weight.clone().detach()
        assert not torch.equal(orig_conv1_weight, new_conv1_weight)
        assert orig_conv1_encoding_max != new_conv1_encoding_max
        assert not torch.equal(orig_conv2_weight, new_conv2_weight)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'rl_with_fp16_and_bw_32', dummy_input=dummy_input.to('cpu'))
            with open(f'{tmp_dir}/rl_with_fp16_and_bw_32_torch.encodings') as json_file:
                encoding_data = json.load(json_file)
                assert encoding_data['param_encodings']['conv2.weight'][0] == {'bitwidth': 16, 'dtype': 'float'}
                assert 'relu2' not in encoding_data['activation_encodings']
                assert len(encoding_data['activation_encodings']) == 5

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

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)
        sim.model.to('cuda:0')

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        weight_single_gpu = sim.model.conv1.weight.clone().detach()
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1.weight.grad.clone().detach()

        sim.model.conv1.weight.grad = None

        sim.model = torch.nn.DataParallel(sim.model)
        weight_multi_gpu = sim.model.module.conv1.weight.clone().detach()
        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        grad_multi_gpu = sim.model.module.conv1.weight.grad.clone().detach()

        assert torch.allclose(weight_single_gpu, weight_multi_gpu)
        assert torch.allclose(output_multi_gpu, output_single_gpu)
        assert torch.allclose(grad_single_gpu, grad_multi_gpu)

    def test_qc_trainable_wrapper(self):
        torch.manual_seed(0)
        q_conv1 = aimet_nn.QuantizedConv2d(1, 32, kernel_size=5)
        q_conv1.param_quantizers['weight'] = QuantizeDequantize(shape=(),
                                                                bitwidth=8,
                                                                symmetric=False)
        q_conv1.param_quantizers['bias'] = QuantizeDequantize(shape=(),
                                                              bitwidth=8,
                                                              symmetric=False)
        q_conv1.input_quantizers[0] = QuantizeDequantize(shape=(),
                                                          bitwidth=8,
                                                          symmetric=False)
        q_conv1.output_quantizers[0] = QuantizeDequantize(shape=(),
                                                           bitwidth=8,
                                                           symmetric=False)
        q_conv1.param_quantizers['weight'].set_range(-1, 1)
        q_conv1.param_quantizers['bias'].set_range(-1, 1)
        q_conv1.input_quantizers[0].set_range(-1, 1)
        q_conv1.output_quantizers[0].set_range(-1, 1)

        inp = torch.rand((1, 1, 5, 5))
        optimizer = torch.optim.SGD(q_conv1.parameters(), lr=0.05, momentum=0.5)
        out = q_conv1(inp)
        loss = out.flatten().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Checking if encoding min max have changed
        assert not torch.any(q_conv1.input_quantizers[0].min == -1.0)
        assert not torch.any(q_conv1.input_quantizers[0].max == 1.0)

        assert not torch.any(q_conv1.input_quantizers[0].min == -1.0)
        assert not torch.any(q_conv1.input_quantizers[0].max == 1.0)

        assert not torch.any(q_conv1.param_quantizers['weight'].min == -1.0)
        assert not torch.any(q_conv1.param_quantizers['weight'].max == 1.0)

        assert not torch.any(q_conv1.param_quantizers['bias'].min == -1.0)
        assert not torch.any(q_conv1.param_quantizers['bias'].max == 1.0)

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

        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        model = ModelWithTwoInputsOneToAdd()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = f'{tmp_dir}/quantsim_config.json'
            with open(config_file_path, "w") as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                       quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       config_file=config_file_path)

        # Enable input parameters to add (multiple input parameter exist)
        sim.model.add.input_quantizers[0] = QuantizeDequantize(shape=(),
                                                               bitwidth=8,
                                                               symmetric=False)
        sim.model.add.input_quantizers[1] = QuantizeDequantize(shape=(),
                                                               bitwidth=8,
                                                               symmetric=False)

        sim.compute_encodings(forward_pass, forward_pass_callback_args=None)

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
        quantizer = sim.model.conv1_a.param_quantizers['weight']
        assert torch.allclose(quantizer.get_min(), -quantizer.get_max() - quantizer.get_scale())
        quantizer = sim.model.fc1.param_quantizers['weight']
        assert torch.allclose(quantizer.get_min(), -quantizer.get_max() - quantizer.get_scale())

        before_conv1_weight_encoding_min = sim.model.conv1_a.param_quantizers['weight'].get_min()
        before_fc_weight_encoding_min = sim.model.fc1.param_quantizers['weight'].get_min()

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.003, momentum=0.5)
        for _ in range(20):
            inputs = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))
            out = sim.model(*inputs)
            loss = out.flatten().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        quantizer = sim.model.conv1_a.param_quantizers['weight']
        assert torch.allclose(quantizer.get_min(), -quantizer.get_max() - quantizer.get_scale())
        quantizer = sim.model.fc1.param_quantizers['weight']
        assert torch.allclose(quantizer.get_min(), -quantizer.get_max() - quantizer.get_scale())

        after_conv1_weight_encoding_min = sim.model.conv1_a.param_quantizers['weight'].get_min()
        after_fc_weight_encoding_min = sim.model.fc1.param_quantizers['weight'].get_min()

        assert not torch.allclose(before_conv1_weight_encoding_min, after_conv1_weight_encoding_min)
        assert not torch.allclose(before_fc_weight_encoding_min, after_fc_weight_encoding_min)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, "results", dummy_input)
            with open(f"{tmp_dir}/results.encodings", "r") as encodings_file:
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

        assert sim.model.conv1_a.output_quantizers[0].is_initialized()

        forward_pass(sim.model, None)

    @pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.8"),
                        reason="torch.profiler is not supported in torch<1.8")
    def test_memory_profiler(self):
        """ test using memory profiler """
        # checks PyTorch version before importing torch.profiler (introduced in version 1.8.0), for
        # older versions this test is passthrough.
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

        sim.model.train()

        with profile(activities=[ProfilerActivity.CPU],
                     profile_memory=True, record_shapes=True) as prof:

            for _ in range(1):
                out = sim.model(in_tensor)
                out = out.sum().backward()

        memory_stats = [event for event in prof.key_averages() if event.key == '[memory]'][0]
        assert abs(memory_stats.cpu_memory_usage) < 1.1 * (100 * (10 ** 6))

    def test_accumulator_overflow(self):

        torch.manual_seed(0)
        model = models.resnet18()
        model = model.eval()
        layer, range_used = check_accumulator_overflow(model, 8, 32)

        assert layer == 'layer4.1.conv1'

        # self.assertAlmostEqual(100 * range_used, 0.263623, places=3)

    @pytest.mark.skip("load_encodings_to_sim not implemented")
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

    @pytest.mark.skip("load_encodings_to_sim not implemented")
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
        quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                         default_param_bw=4, default_output_bw=4)
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
        quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                         default_param_bw=4, default_output_bw=4)
        quant_sim.compute_encodings(evaluate, dummy_input)
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

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=4, default_output_bw=4)

        sim.model.cust.output_quantizers[0] = None
        sim.compute_encodings(evaluate, dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'module_with_5_output', dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                       propagate_encodings=False)
            with open(f'{tmp_dir}/module_with_5_output.encodings') as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert '7' not in activation_encodings
                assert set(['8', '9', '10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple(self):
        cust_model = CustModelV1Simple()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                         default_param_bw=8, default_output_bw=8)

        quant_sim.compute_encodings(evaluate, dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_sim.export(tmp_dir, 'cust_v1_simple', dummy_input,
                             onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                             propagate_encodings=True)
            with open(f'{tmp_dir}/cust_v1_simple.encodings') as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert set(['10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple_v2(self):
        cust_model = CustomOpV2()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                         default_param_bw=4, default_output_bw=4)

        quant_sim.compute_encodings(evaluate, dummy_input)

        a, b, c, d, e = quant_sim.model(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_sim.export(tmp_dir, 'cust_v2_simple', dummy_input,
                             onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                             propagate_encodings=False)

            with open(f'{tmp_dir}/cust_v2_simple.encodings') as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert len(activation_encodings) == 6

            module_names = { module_name for module_name, _ in cust_model.named_modules()}
            onnx_model = onnx.load(f'{tmp_dir}/cust_v2_simple.onnx')

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
        sim = QuantizationSimModel(roi_model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf)
        for i, _ in enumerate(list(sim.model.roi.input_quantizers)):
            sim.model.roi.input_quantizers[i] = None

        def forward_pass(model, _):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, "roi_model", dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),  propagate_encodings=True)

            with open(f'{tmp_dir}/roi_model.encodings') as json_file:
                encodings = json.load(json_file)['activation_encodings']

                # Only one entry should have min, max, delta and offset, remaining entries should be propagated
                # with bitwidth and dtype.
                encodings = [{key: val} for key, val in encodings.items() if 'scale' in val[0]]
                assert len(encodings) == 1

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

    def test_quantsim_conv3d_tf_int8_eval_train(self):

        torch.random.manual_seed(10)
        model = Conv3dModel()
        dummy_input = torch.randn(1, 3, 24, 24, 24)
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=8, default_output_bw=8)
        sim.compute_encodings(evaluate, dummy_input)
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
        sim.model = sim.model.cuda().half()

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
                                    "unsigned_symmetric": "False",
                                    "per_channel_quantization": "True",
                                },
                                "params": {},
                                "op_type": {},
                                "supergroups": [],
                                "model_input": {},
                                "model_output": {}
                          }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/quantsim_config.json", "w") as f:
                json.dump(quantsim_config, f)

            model = HalfFloatTestModel().cuda().half()
            dummy_input = torch.rand(1, 20, 4, 4).half().cuda()
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       dummy_input=dummy_input, config_file=f"{tmp_dir}/quantsim_config.json")
            sim.model.cuda().half()

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
                    "unsigned_symmetric": "False",
                    "per_channel_quantization": "True",
                },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/quantsim_config.json", "w") as f:
                json.dump(quantsim_config, f)

            model = HalfFloatTestModel().cuda().half()
            dummy_input = torch.rand(1, 20, 4, 4).half().cuda()
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       dummy_input=dummy_input, config_file=f"{tmp_dir}/quantsim_config.json")
            sim.model.cuda().half()

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

        sim.compute_encodings(dummy_forward, None)

        assert torch.equal(sim.model.conv2a.output_quantizers[0].get_min(),
                           sim.model.conv2b.output_quantizers[0].get_min())
        assert torch.equal(sim.model.conv2a.output_quantizers[0].get_max(),
                           sim.model.conv2b.output_quantizers[0].get_max())

        dummy_forward(sim.model, None)
        dummy_forward(sim.model, None)

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

        # A couple of forward-backward passes
        output = dummy_forward(sim.model, None)
        output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        output = dummy_forward(sim.model, None)
        output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

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
                    "unsigned_symmetric": "False",
                    "per_channel_quantization": "True",
                },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = f"{tmp_dir}/quantsim_config.json"
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

    @pytest.mark.parametrize('quant_scheme', [QuantScheme.training_range_learning_with_tf_init])
                                              # QuantScheme.training_range_learning_with_tf_enhanced_init])
    @pytest.mark.parametrize('config_file', [None, get_path_for_per_channel_config()])
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
        sim.model.to(device)
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

    @pytest.mark.parametrize("hw_version", ['V69', 'V73', 'V75', 'V79'])
    @pytest.mark.parametrize("quant_scheme", [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_exception_for_embedding(self, hw_version, quant_scheme):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = test_models.ModelWithEmbedding().to(device)
        dummy_input = torch.tensor([[1, 4, 2, 5], [4, 3, 2, 7]], dtype=torch.int64, device=device)

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"},
                "params": {"is_symmetric": "True", "is_quantized": "True"},
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

        qembedding = sim.model.embedding
        weight_quantizer = qembedding.param_quantizers["weight"]
        assert weight_quantizer.min.numel() == weight_quantizer.max.numel() == 1

        if sim._hw_version in {"V73", "V75", "V79"}:
            assert weight_quantizer.bitwidth == 16
            assert not weight_quantizer.symmetric
        elif sim._hw_version == "V69":
            assert weight_quantizer.bitwidth == 4
            assert weight_quantizer.symmetric
        else:
            raise

    @pytest.mark.parametrize('hw_version', ['V69', 'V73', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_exception_for_groupnorm(self, hw_version, quant_scheme):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithGroupNorm().to(device)
        dummy_input = torch.randn((1, 6, 2, 2), device=device)

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"},
                "params": {"is_symmetric": "True", "is_quantized": "True"},
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
        assert weight_quantizer
        assert bias_quantizer

        assert weight_quantizer.min.numel() == weight_quantizer.max.numel() == 1

        if sim._hw_version in {'V73', 'V75'}:
            assert weight_quantizer.bitwidth == 16
            assert not weight_quantizer.symmetric
            assert bias_quantizer.bitwidth == 16
            assert not bias_quantizer.symmetric
        else:
            assert weight_quantizer.bitwidth == 4
            assert weight_quantizer.symmetric
            assert bias_quantizer.bitwidth == 4
            assert bias_quantizer.symmetric

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    @pytest.mark.parametrize('default_output_bw', [8, 16])
    def test_exception_for_matmul_if_input_quantization_disabled(self, hw_version, quant_scheme, default_output_bw):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = test_models.ModelWithMatMul().to(device)
        dummy_input = (torch.randn(10, 3, 4, device=device), torch.randn(10, 5, 4, device=device))

        quantsim_config = {
            "defaults": {
                "hw_version": hw_version,
                "ops": {"is_output_quantized": "True"}, "params": {"is_symmetric": "True"},
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
                assert closest_output_quantizer_of_second_input.symmetric
                assert closest_output_quantizer_of_first_input.bitwidth == 16
            else:
                assert not closest_output_quantizer_of_second_input.symmetric
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert closest_output_quantizer_of_second_input.bitwidth == 8
            assert closest_output_quantizer_of_second_input.symmetric
        else:
            assert not closest_output_quantizer_of_second_input.symmetric

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V73', 'V69', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
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
                assert second_input_quantizer.symmetric
                assert first_input_quantizer.bitwidth == 16
            else:
                assert not second_input_quantizer.symmetric
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert second_input_quantizer.symmetric
            assert second_input_quantizer.bitwidth == 8
        else:
            assert not second_input_quantizer.symmetric

        # Restore original mapping dictionary
        onnx_utils.map_torch_types_to_onnx = original_map_torch_types_to_onnx

    @pytest.mark.parametrize('hw_version', ['default', 'V66', 'V68', 'V69', 'V73', 'V75'])
    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    @pytest.mark.parametrize("default_output_bw", [8, 16])
    def test_exception_for_matmul_edge_case(
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
                assert closest_output_quantizer_of_second_input.symmetric
                assert closest_output_quantizer_of_first_input.bitwidth == 16
            else:
                assert not closest_output_quantizer_of_second_input.symmetric
        elif sim._hw_version in {'V66', 'V68', 'V69'}:
            assert closest_output_quantizer_of_second_input.bitwidth == 8
            assert closest_output_quantizer_of_second_input.symmetric
        else:
            assert not closest_output_quantizer_of_second_input.symmetric
