# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import math
import pytest
import json
import tempfile
from pathlib import Path
from contextlib import contextmanager
import torch
from torchvision import models

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.v1.batch_norm_fold import (
    fold_given_batch_norms,
    fold_all_batch_norms,
    fold_all_batch_norms_to_scale,
    _is_valid_bn_fold,
    _find_all_batch_norms_to_fold,
)
from models.test_models import TransposedConvModel, Conv3dModel, Conv3dModel1
from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
from aimet_common.defs import QuantScheme
import aimet_torch.v1.batch_norm_fold as batch_norm_fold
from torch.nn.modules.batchnorm import _BatchNorm
torch.manual_seed(1228)


def _initialize_bn_params(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
                module.bias.copy_(torch.randn_like(module.bias))
                module.running_mean.copy_(torch.randn_like(module.bias))
                module.running_var.add_(torch.randn_like(module.bias).abs())


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


class TwoInputs(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(TwoInputs, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2)
        self.ada = torch.nn.AdaptiveAvgPool2d(18)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = torch.nn.Linear(1600, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs[1])
        x2 = self.bn2(x2)
        x2 = self.conv3(x2)
        x2 = self.ada(x2)
        x = x1 + x2
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@contextmanager
def _use_python_impl(flag: bool):
    orig_flag = batch_norm_fold.USE_PYTHON_IMPL
    try:
        batch_norm_fold.USE_PYTHON_IMPL = flag
        yield
    finally:
        batch_norm_fold.USE_PYTHON_IMPL = orig_flag


@pytest.fixture(params=[True, False])
def use_python_impl(request):
    param: bool = request.param

    with _use_python_impl(param):
        yield


class TestTrainingExtensionBnFold:

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_fold_resnet18(self, use_python_impl, device):
        torch.manual_seed(10)
        model = models.resnet18().to(device)
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(1, 3, 224, 224).to(device)

        baseline_output = model(random_input)

        layer_list = [(model.layer2[0].conv1, model.layer2[0].bn1)]
        params_before = model.layer2[0].conv1.weight.clone()
        fold_given_batch_norms(model, layer_list)
        params_after = model.layer2[0].conv1.weight
        output_after_fold = model(random_input)

        assert not torch.equal(params_before, params_after)
        assert not isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, atol=1.e-3)

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_python_impl(self, device):
        try:
            flag = batch_norm_fold.USE_PYTHON_IMPL
            torch.manual_seed(10)
            model = models.resnet18().eval().to(device)
            _initialize_bn_params(model)
            model_copy = copy.deepcopy(model)

            batch_norm_fold.USE_PYTHON_IMPL = True
            layer_list = [(model.layer2[0].conv1, model.layer2[0].bn1)]
            fold_given_batch_norms(model, layer_list)

            batch_norm_fold.USE_PYTHON_IMPL = False
            layer_list = [(model_copy.layer2[0].conv1, model_copy.layer2[0].bn1)]
            fold_given_batch_norms(model_copy, layer_list)

            # Ensure that the weight parameter is updated correctly after bn fold.
            assert torch.allclose(model.layer2[0].conv1.weight, model_copy.layer2[0].conv1.weight)
            assert not isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d)
            assert not isinstance(model_copy.layer2[0].bn1, torch.nn.BatchNorm2d)
        finally:
            batch_norm_fold.USE_PYTHON_IMPL = flag

    def test_fold_bn_before_conv_no_bias(self, use_python_impl):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 2, bias=False)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 2, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(20, 10, 4, 4)

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.conv2)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv2.weight.requires_grad == model.conv2.bias.requires_grad
        assert model.conv2.weight.device == model.conv2.bias.device
        assert model.conv2.weight.dtype == model.conv2.bias.dtype

    def test_fold_bn_before_conv_with_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 30, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.conv2)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-1)

    def test_fold_bn_before_conv_with_padding(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, padding=1, bias=False)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 3, padding=1, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(20, 10, 6, 6)

        baseline_output = model(random_input)

        conv_bn = fold_all_batch_norms(model, (20, 10, 6, 6))

        output_after_fold = model(random_input)

        assert not conv_bn
        assert isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_conv_transpose(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, padding=1, bias=False)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.ConvTranspose2d(20, 40, 3, padding=0, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(20, 10, 6, 6)

        baseline_output = model(random_input)

        conv_bn = fold_all_batch_norms(model, (20, 10, 6, 6))

        output_after_fold = model(random_input)

        assert not conv_bn
        assert isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_filter_conv_bn_pair(self, use_python_impl):
        invalid_fold_forward = [torch.nn.Conv2d(10, 20, 3, padding=1),
                                torch.nn.Conv2d(10, 10, 2, groups=10),
                                torch.nn.Conv2d(10, 20, 2, groups=2),
                                torch.nn.Conv1d(10, 20, 3, padding=1),
                                torch.nn.Conv1d(10, 10, 2, groups=10),
                                torch.nn.Conv1d(10, 20, 2, groups=2),
                                torch.nn.ConvTranspose2d(10, 20, 3),
                                ]
        is_invalid = [not _is_valid_bn_fold(layer, False) for layer in invalid_fold_forward]
        assert all(is_invalid)

        invalid_fold_backward = [torch.nn.ConvTranspose2d(10, 20, 2, groups=2)]
        is_invalid = [not _is_valid_bn_fold(layer, True) for layer in invalid_fold_backward]
        assert all(is_invalid)

        valid_fold_forward = [torch.nn.Conv2d(10, 20, 3, padding=0),
                              torch.nn.Linear(10, 10)]
        is_valid = [_is_valid_bn_fold(layer, False) for layer in valid_fold_forward]
        assert all(is_valid)

        valid_fold_backward = [torch.nn.Conv2d(10, 20, 2, padding=0),
                               torch.nn.Conv2d(10, 20, 2, padding=1),
                               torch.nn.Conv2d(10, 20, 2, groups=2),
                               torch.nn.Conv2d(10, 10, 2, groups=10),
                               torch.nn.Conv1d(10, 20, 2, padding=0),
                               torch.nn.Conv1d(10, 20, 2, padding=1),
                               torch.nn.Conv1d(10, 20, 2, groups=2),
                               torch.nn.Conv1d(10, 10, 2, groups=10),
                               torch.nn.ConvTranspose2d(10, 20, 2, padding=0),
                               torch.nn.ConvTranspose2d(10, 20, 2, padding=1),
                               torch.nn.ConvTranspose2d(10, 10, 2, groups=10),
                               torch.nn.Linear(10, 10),
                               ]
        is_valid = [_is_valid_bn_fold(layer, True) for layer in valid_fold_backward]
        assert all(is_valid)

    def test_fold_bn_after_conv_no_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1.weight.requires_grad == model.conv1.bias.requires_grad
        assert model.conv1.weight.device == model.conv1.bias.device
        assert model.conv1.weight.dtype == model.conv1.bias.dtype

    def test_fold_bn_after_conv_depthwise(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_transposed_conv_depthwise(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_conv_with_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_linear_layer_no_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.fc1.weight.requires_grad == model.fc1.bias.requires_grad
        assert model.fc1.weight.device == model.fc1.bias.device
        assert model.fc1.weight.dtype == model.fc1.bias.dtype

    def test_fold_bn_before_linear_layer_with_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_linear_layer_no_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        baseline_output = model(random_input)

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.fc1.weight.requires_grad == model.fc1.bias.requires_grad
        assert model.fc1.weight.device == model.fc1.bias.device
        assert model.fc1.weight.dtype == model.fc1.bias.dtype

    def test_fold_bn_after_linear_layer_with_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        baseline_output = model(random_input)

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_find_batch_norms_to_fold(self, use_python_impl):
        model = MyModel().eval()
        _initialize_bn_params(model)

        input_shape = (2, 10, 24, 24)
        connected_graph = ConnectedGraph(model,
                                         create_rand_tensors_given_shapes(input_shape, get_device(model)))

        conv_bn_pairs, bn_conv_pairs, bn_picked = _find_all_batch_norms_to_fold(connected_graph)

        assert len(conv_bn_pairs) == len(bn_conv_pairs) == 1
        assert (model.conv1, model.bn1) in conv_bn_pairs
        assert (model.bn2, model.conv3) in bn_conv_pairs
        assert len(bn_picked) == 2

    def test_bn_fold_auto_mode_transposed_conv2d(self, use_python_impl):

        torch.manual_seed(10)
        model = TransposedConvModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand((10, 10, 4, 4))

        baseline_output = model(random_input)

        folded_pairs = fold_all_batch_norms(model, (10, 10, 4, 4))

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)

        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert len(folded_pairs) == 2

    def test_find_batch_norms_to_fold_multi_input(self, use_python_impl):
        model = TwoInputs().eval()
        _initialize_bn_params(model)
        inp_shapes = [(1, 3, 32, 32), (1, 3, 20, 20)]

        connected_graph = ConnectedGraph(model,
                                         create_rand_tensors_given_shapes(inp_shapes, get_device(model)))

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(connected_graph)


        assert len(conv_bn_pairs) == 2
        assert not bn_conv_pairs
        assert (model.conv1, model.bn1) in conv_bn_pairs
        assert (model.conv2, model.bn2) in conv_bn_pairs

    def test_bn_fold_auto_mode(self, use_python_impl):
        torch.manual_seed(10)

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input)

        folded_pairs = fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert len(folded_pairs) == 2

    def test_fold_auto_mode_with_bn_after_Conv1d_layer(self, use_python_impl):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))

        baseline_output = model(random_input)
        orig_bn = model.bn1

        bn_pairs = fold_all_batch_norms(model, (2, 10, 32))
        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs

    def test_bn_conversion(self, use_python_impl):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)
                self.relu = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.relu(x)
                x = self.bn1(x)

                return x

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))

        baseline_output = model(random_input)
        orig_bn = model.bn1

        bn_pairs = fold_all_batch_norms(model, (2, 10, 32))
        output_after_fold = model(random_input)

        assert isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        assert 0 == len(bn_pairs)
        assert (model.conv1d, orig_bn) not in bn_pairs

    def test_fold_manual_with_bn_after_Conv1d_layer_no_bias(self, use_python_impl):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))

        baseline_output = model(random_input)

        layer_list = [(model.conv1d, model.bn1)]
        fold_given_batch_norms(model, layer_list)
        output_after_fold = (model.conv1d(random_input))

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1d.weight.requires_grad == model.conv1d.bias.requires_grad
        assert model.conv1d.weight.device == model.conv1d.bias.device
        assert model.conv1d.weight.dtype == model.conv1d.bias.dtype

    @pytest.mark.cuda
    def test_multi_gpu(self, use_python_impl):
        torch.manual_seed(10)
        model = MyModel()
        model.eval()
        model = torch.nn.DataParallel(model)
        model.to(device='cuda:0')
        random_input = torch.rand(2, 10, 24, 24).to(device='cuda:0')
        output_before = model(random_input)

        # BN fold
        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after = model(random_input)
        assert torch.allclose(output_before, output_after, rtol=1.e-2)

    def test_fold_bn_before_Conv1d_with_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))

        baseline_output = model(random_input)
        orig_bn = model.bn1
        bn_pairs = fold_all_batch_norms(model, (2, 10, 32))

        output_after_fold = model(random_input)

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_Conv1d_no_bias(self, use_python_impl):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.conv1d = torch.nn.Conv1d(4, 4, kernel_size=2, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 4, 4))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 4, 4)))
        model.eval()

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.conv1d)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1d.weight.requires_grad == model.conv1d.bias.requires_grad
        assert model.conv1d.weight.device == model.conv1d.bias.device
        assert model.conv1d.weight.dtype == model.conv1d.bias.dtype

    def test_bn_fold_conv3d_fold_backward(self, use_python_impl):

        torch.random.manual_seed(10)
        model = Conv3dModel()
        inp = torch.randn(1, 3, 24, 24, 24)
        model.bn1.weight.data = torch.randn(model.bn1.weight.shape)
        model.bn2.weight.data = torch.randn(model.bn2.weight.shape)

        # eval
        model = model.eval()
        orig_out = model(inp)
        _ = fold_all_batch_norms(model, input_shapes=(1, 3, 24, 24, 24), dummy_input=inp)
        new_out = model(inp)

        assert torch.allclose(orig_out, new_out, atol=1e-5)
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm3d)]
        assert not bn_modules

    def test_bn_fold_conv3d_fold_forward(self, use_python_impl):

        torch.random.manual_seed(10)
        model = Conv3dModel1()
        inp = torch.randn(1, 3, 24, 24, 24)
        model.bn1.weight.data = torch.randn(model.bn1.weight.shape)
        model.bn2.weight.data = torch.randn(model.bn2.weight.shape)

        # eval
        model = model.eval()
        orig_out = model(inp)
        _ = fold_all_batch_norms(model, input_shapes=(1, 3, 24, 24, 24), dummy_input=inp)
        new_out = model(inp)

        assert torch.allclose(orig_out, new_out, atol=1e-5)
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm3d)]
        assert len(bn_modules) == 1
        assert isinstance(model.bn1, torch.nn.Identity)
        assert isinstance(model.bn2, torch.nn.BatchNorm3d)


symmetric_quantsim_config ={
    "defaults": {
        "ops": { "is_output_quantized": "True" },
        "params": { "is_quantized": "True", "is_symmetric": "True"},
        "strict_symmetric": "False",
        "unsigned_symmetric": "True",
        "per_channel_quantization": "True"
    },
    "params": {
        "bias": { "is_quantized": "False" }
    },
    "op_type": {},
    "supergroups": [
        { "op_list": ["Conv", "Relu"] },
        { "op_list": ["Conv", "Clip"] },
        { "op_list": ["Add", "Relu"] },
        { "op_list": ["Gemm", "Relu"] },
    ],
    "model_input": { "is_input_quantized": "True" },
    "model_output": {}
}

asymmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
asymmetric_quantsim_config["defaults"]["params"]["is_symmetric"] = "False"

strict_symmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
strict_symmetric_quantsim_config["defaults"]["strict_symmetric"] = "True"

quantsim_config_map = {
    "symmetric": symmetric_quantsim_config,
    "asymmetric": asymmetric_quantsim_config,
    # "strict_symmetric": strict_symmetric_quantsim_config,
}

def quantsim(model, dummy_input, quantsim_config=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file_path = Path(tmp_dir, "quantsim_config.json")

        quantsim_config = quantsim_config or symmetric_quantsim_config
        with open(config_file_path, 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model,
                                dummy_input.clone(),
                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                config_file=config_file_path)

        def forward_pass_callback(model, _):
            model(dummy_input.clone())

        sim.compute_encodings(forward_pass_callback, None)
        return sim


class TestTrainingExtensionBnFoldToScale:
    @pytest.mark.parametrize("config", quantsim_config_map.keys())
    @pytest.mark.parametrize("seed", range(10))
    def test_fold_resnet18(self, seed, config):
        quantsim_config = quantsim_config_map[config]

        torch.manual_seed(seed)
        model = models.resnet18().eval()
        _initialize_bn_params(model)
        model = prepare_model(model)

        random_input = torch.rand((1, 3, 224, 224))
        sim = quantsim(model, random_input.clone(), quantsim_config)
        layer2_0 = getattr(sim.model.layer2, "0")

        # NOTE: layer2[0] is structured as below.
        #  input --> layer2[0].conv1 --------> layer2[0].bn1 ----------> layer2[0].relu -> output
        #         - param: quantized        - param: not quantized    - param: N/A
        #         - output: not quantized   - output: not quantized   - output: quantized

        # Check quantizers are enabled/disabled properly
        assert not layer2_0.conv1.output_quantizers[0].enabled
        assert layer2_0.conv1.param_quantizers["weight"].enabled
        assert not layer2_0.bn1.output_quantizers[0].enabled
        assert not layer2_0.bn1.param_quantizers["weight"].enabled
        assert layer2_0.relu.output_quantizers[0].enabled

        buffer = None
        def collect_output(module, inp, output):
            # Forward hook for collecting
            nonlocal buffer
            buffer = output.clone().detach()

        def int8_repr(x, quantizer):
            # Return fake-quantized output in INT8 representation
            encoding = quantizer.encoding
            delta = float((encoding.max - encoding.min)/255)
            offset = float(round(-encoding.min/255))
            return x / delta + offset

        ### Outputs before batchnorm folding
        with layer2_0.relu.register_forward_hook(collect_output):
            fakequant_output = sim.model(random_input.clone()).clone().detach()
            int8_output = int8_repr(fakequant_output, sim.model.fc.output_quantizers[0])
            fakequant_relu_output = buffer
            int8_relu_output = int8_repr(fakequant_relu_output, layer2_0.relu.output_quantizers[0])

        ### Apply batchnorm folding
        layer_list = [(layer2_0.conv1, layer2_0.bn1)]
        fold_given_batch_norms(sim.model, layer_list)

        ### Outputs after batchnorm folding
        with layer2_0.relu.register_forward_hook(collect_output):
            fakequant_output_after_folding = sim.model(random_input.clone()).clone().detach()
            int8_output_after_folding = int8_repr(fakequant_output_after_folding, sim.model.fc.output_quantizers[0])
            fakequant_relu_output_after_folding = buffer
            int8_relu_output_after_folding = int8_repr(fakequant_relu_output_after_folding, layer2_0.relu.output_quantizers[0])

        # Check batchnorm is replaced with identity
        assert isinstance(layer2_0.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert not layer2_0.conv1.output_quantizers[0].enabled
        assert layer2_0.conv1.param_quantizers["weight"].enabled
        assert not layer2_0.bn1.output_quantizers[0].enabled
        assert not layer2_0.bn1.param_quantizers["weight"].enabled
        assert layer2_0.relu.output_quantizers[0].enabled

        # test 1: All final outputs should be contained within 3-tick difference
        last_output_encoding = sim.model.fc.output_quantizers[0].encoding
        delta = float((last_output_encoding.max - last_output_encoding.min)/255)
        assert torch.allclose(fakequant_output, fakequant_output_after_folding, atol=3*delta) # Allow 3-tick difference
        assert torch.allclose(int8_output, int8_output_after_folding, atol=3) # Allow 3-tick difference

        # test 2: At least 99% of the final outputs should be contained within 1-tick difference
        assert torch.isclose(fakequant_output, fakequant_output_after_folding, atol=1*delta).sum() >= math.floor(fakequant_output.numel() * 0.99)
        assert torch.isclose(int8_output, int8_output_after_folding, atol=1).sum() >= math.floor(int8_output.numel() * 0.99)

        # test 3: All ReLU outputs should be contained within 1-tick difference
        relu_output_encoding = layer2_0.relu.output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert torch.allclose(fakequant_relu_output, fakequant_relu_output_after_folding, atol=1*delta) # Allow 1-tick difference
        assert torch.allclose(int8_relu_output, int8_relu_output_after_folding, atol=1) # Allow 1-tick difference

        # test 4: At least 99% of the ReLU outputs should be almost exactly equal
        assert torch.isclose(fakequant_relu_output, fakequant_relu_output_after_folding).sum() >= math.floor(fakequant_relu_output.numel() * 0.99)
        assert torch.isclose(int8_relu_output, int8_relu_output_after_folding).sum() >= math.floor(int8_relu_output.numel() * 0.99)

    def test_fold_bn_before_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 2, bias=False)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 2, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((20, 10, 4, 4)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        layer_list = [(model.bn1, model.conv2)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 30, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((2, 10, 24, 24)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        layer_list = [(model.bn1, model.conv2)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        baseline_output = model(random_input)

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        relu_output_encoding = model.relu1.output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

        conv1 = model.conv1._module_to_wrap
        assert conv1.weight.requires_grad == conv1.bias.requires_grad
        assert conv1.weight.device == conv1.bias.device
        assert conv1.weight.dtype == conv1.bias.dtype

    def test_fold_bn_after_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        baseline_output = model(random_input)

        fold_all_batch_norms_to_scale(sim)

        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        relu_output_encoding = model.relu1.output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

    def test_fold_bn_after_transposed_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((2, 10, 24, 24)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        # NOTE: Bathcnorm quantizers should be enabled since batchnorm folding
        #       is not supported for transposed depthwise conv.
        assert model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        fold_all_batch_norms_to_scale(sim)
        # Folding BatchNorm to transposed depthwise convolution is not supported
        assert isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)

        # Check quantizers are enabled/disabled properly
        # NOTE: Bathcnorm quantizers should be enabled since batchnorm folding
        #       is not supported for transposed depthwise conv.
        assert model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

    def test_fold_bn_after_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand(2, 10, 24, 24)

        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        baseline_output = model(random_input)

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert not model.conv1.output_quantizers[0].enabled
        assert model.conv1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled
        assert model.relu1.output_quantizers[0].enabled

        relu_output_encoding = sim.model.relu1.output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

    def test_fold_bn_before_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((32, 10)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        layer_list = [(model.bn1, model.fc1)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((32, 10)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        layer_list = [(model.bn1, model.fc1)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))
        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        baseline_output = model(random_input)

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        fc_output_encoding = model.fc1.output_quantizers[0].encoding
        bn_output_encoding = model.bn1.output_quantizers[0]._compute_updated_encoding()
        assert fc_output_encoding.max == bn_output_encoding.max and\
                fc_output_encoding.min == bn_output_encoding.min and\
                fc_output_encoding.delta == bn_output_encoding.delta and\
                fc_output_encoding.offset == bn_output_encoding.offset and\
                fc_output_encoding.bw == bn_output_encoding.bw

        fc_output_encoding = model.fc1.output_quantizers[0].encoding
        delta = float((fc_output_encoding.max - fc_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

        fc1 = model.fc1._module_to_wrap
        assert fc1.weight.requires_grad == fc1.bias.requires_grad
        assert fc1.weight.device == fc1.bias.device
        assert fc1.weight.dtype == fc1.bias.dtype

    def test_fold_bn_after_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))
        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        baseline_output = model(random_input)

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert model.fc1.output_quantizers[0].enabled
        assert model.fc1.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        fc_output_encoding = model.fc1.output_quantizers[0].encoding
        bn_output_encoding = model.bn1.output_quantizers[0]._compute_updated_encoding()
        assert fc_output_encoding.max == bn_output_encoding.max and\
                fc_output_encoding.min == bn_output_encoding.min and\
                fc_output_encoding.delta == bn_output_encoding.delta and\
                fc_output_encoding.offset == bn_output_encoding.offset and\
                fc_output_encoding.bw == bn_output_encoding.bw

        fc_output_encoding = model.fc1.output_quantizers[0].encoding
        delta = float((fc_output_encoding.max - fc_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

    def test_bn_fold_auto_mode_transposed_conv2d(self):
        torch.manual_seed(10)
        model = TransposedConvModel().eval()
        _initialize_bn_params(model)

        random_input = torch.rand((10, 10, 4, 4))
        sim = quantsim(model, random_input)
        model = sim.model

        baseline_output = model(random_input)
        folded_pairs = fold_all_batch_norms_to_scale(sim)
        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        conv2_output_encoding = model.conv2.output_quantizers[0].encoding
        delta = float((conv2_output_encoding.max - conv2_output_encoding.min) / 255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference
        assert len(folded_pairs) == 2

    def test_bn_fold_auto_mode(self):
        torch.manual_seed(10)

        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((2, 10, 24, 24)))

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim)

    def test_fold_auto_mode_with_bn_after_Conv1d_layer(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))
        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        baseline_output = model(random_input)
        orig_bn = model.bn1

        bn_pairs = fold_all_batch_norms_to_scale(sim)
        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        conv_output_encoding = model.conv1d.output_quantizers[0].encoding
        bn_output_encoding = model.bn1.output_quantizers[0]._compute_updated_encoding()
        assert conv_output_encoding.max == bn_output_encoding.max and\
                conv_output_encoding.min == bn_output_encoding.min and\
                conv_output_encoding.delta == bn_output_encoding.delta and\
                conv_output_encoding.offset == bn_output_encoding.offset and\
                conv_output_encoding.bw == bn_output_encoding.bw

        conv_output_encoding = model.conv1d.output_quantizers[0].encoding
        delta = float((conv_output_encoding.max - conv_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs

    def test_fold_manual_with_bn_after_Conv1d_layer_no_bias(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel().eval()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))
        sim = quantsim(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        baseline_output = model(random_input)

        layer_list = [(model.conv1d, model.bn1)]
        fold_given_batch_norms(model, layer_list)
        output_after_fold = model(random_input)

        assert isinstance(model.bn1._module_to_wrap, torch.nn.Identity)

        # Check quantizers are enabled/disabled properly
        assert model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert not model.bn1.output_quantizers[0].enabled
        assert not model.bn1.param_quantizers["weight"].enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        conv_output_encoding = model.conv1d.output_quantizers[0].encoding
        bn_output_encoding = model.bn1.output_quantizers[0]._compute_updated_encoding()
        assert conv_output_encoding.max == bn_output_encoding.max and\
                conv_output_encoding.min == bn_output_encoding.min and\
                conv_output_encoding.delta == bn_output_encoding.delta and\
                conv_output_encoding.offset == bn_output_encoding.offset and\
                conv_output_encoding.bw == bn_output_encoding.bw

        conv_output_encoding = model.conv1d.output_quantizers[0].encoding
        delta = float((conv_output_encoding.max - conv_output_encoding.min)/255)
        assert torch.allclose(baseline_output, output_after_fold, atol=delta) # Allow 1-tick difference

        conv1d = model.conv1d._module_to_wrap
        assert conv1d.weight.requires_grad == conv1d.bias.requires_grad
        assert conv1d.weight.device == conv1d.bias.device
        assert conv1d.weight.dtype == conv1d.bias.dtype

    def test_fold_bn_before_Conv1d_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((2, 10, 32)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim)

    def test_fold_bn_before_Conv1d_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.conv1d = torch.nn.Conv1d(4, 4, kernel_size=2, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel().eval()
        _initialize_bn_params(model)

        sim = quantsim(model, torch.randn((2, 4, 4)))
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.conv1d.output_quantizers[0].enabled
        assert model.conv1d.param_quantizers["weight"].enabled
        assert model.bn1.output_quantizers[0].enabled
        assert model.bn1.param_quantizers["weight"].enabled

        layer_list = [(model.bn1, model.conv1d)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)
