# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

from decimal import Decimal
import torch
import torch.nn as nn
import torch.nn.functional as functional
import pytest
import copy
from contextlib import contextmanager

import aimet_common.defs
from aimet_common.defs import LayerCompRatioPair
from aimet_common.utils import AimetLogger
from .models import mnist_torch_model as mnist_model
import aimet_torch.compression_factory as cf_svd
from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.layer_database import LayerDatabase, Layer
from aimet_torch.svd.svd_pruner import WeightSvdPruner, PyWeightSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class MnistSequentialModel(nn.Module):
    def __init__(self):
        super(MnistSequentialModel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.classfier = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.classfier(x)
        return x


class TestTrainingExtensionsSvd:

    def test_set_parent_attribute_two_deep(self):
        """With a two-deep model"""
        class SubNet(nn.Module):
            def __init__(self):
                super(SubNet, self).__init__()
                self.conv1 = nn.Conv2d(30, 40, 5)
                self.conv2 = nn.Conv2d(40, 50, kernel_size=5)

            def forward(self, *inputs):
                pass

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = SubNet()
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = SubNet()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1.conv2): Layer(model.subnet1.conv2, id(model.subnet1.conv2),
                                                 output_activation_shape),
                  id(model.subnet2.conv1): Layer(model.subnet2.conv1, id(model.subnet2.conv1),
                                                 output_activation_shape),
                  id(model.fc2):           Layer(model.fc2, id(model.fc2),
                                                 output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)

        # child : model.subnet1.conv2 --> parent : model.subnet1
        assert model.subnet1 == layers[id(model.subnet1.conv2)].parent_module
        # child : model.subnet2.conv1 --> parent : model.subnet2
        assert model.subnet2 == layers[id(model.subnet2.conv1)].parent_module
        # child : model.fc2 --> parent : model
        assert model == layers[id(model.fc2)].parent_module

    def test_set_attributes_with_sequentials(self):
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
                    nn.Conv2d(10, 20, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1[2]): Layer(model.subnet1[2], id(model.subnet1[2]),
                                              output_activation_shape),
                  id(model.subnet2[0]): Layer(model.subnet2[0], id(model.subnet2[0]),
                                              output_activation_shape),
                  id(model.fc2): Layer(model.fc2, id(model.fc2),
                                       output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)

        # child : model.subnet1.2 --> parent : model.subnet1
        assert model.subnet1 == layers[id(model.subnet1[2])].parent_module
        # child : model.subnet2.1 --> parent : model.subnet2
        assert model.subnet2 == layers[id(model.subnet2[0])].parent_module
        # child : model.fc2 --> parent : model
        assert model == layers[id(model.fc2)].parent_module

    def test_set_parent_attribute_with_sequential_two_deep(self):
        """With a two-deep model"""
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
                        nn.Conv2d(20, 50, 5)
                    ),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1[0]):     Layer(model.subnet1[0], None, output_activation_shape),
                  id(model.subnet1[2][0]):  Layer(model.subnet1[2][0], None, output_activation_shape),
                  id(model.subnet1[2][2]):  Layer(model.subnet1[2][2], None, output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)
        # child : model.subnet1.0 --> parent : model.subnet1
        assert model.subnet1 == layers[id(model.subnet1[0])].parent_module
        # child : model.subnet1.2.0 --> parent : model.subnet1.2
        assert model.subnet1[2] == layers[id(model.subnet1[2][0])].parent_module
        # child : model.subnet1.2.2 --> parent : model.subnet1.2
        assert model.subnet1[2] == layers[id(model.subnet1[2][2])].parent_module


class TestWeightSvdPruning:

    def test_prune_layer(self):

        model = mnist_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        conv2 = comp_layer_db.find_layer_by_name('conv2')
        weight_svd_pruner = WeightSvdPruner()
        weight_svd_pruner._prune_layer(orig_layer_db, comp_layer_db, conv2, 0.5, aimet_common.defs.CostMetric.mac)

        conv2_a = comp_layer_db.find_layer_by_name('conv2.0')
        conv2_b = comp_layer_db.find_layer_by_name('conv2.1')

        assert (1, 1) == conv2_a.module.kernel_size
        assert 32 == conv2_a.module.in_channels
        assert 15 == conv2_a.module.out_channels

        assert (5, 5) == conv2_b.module.kernel_size
        assert 15 == conv2_b.module.in_channels
        assert 64 == conv2_b.module.out_channels

        assert isinstance(comp_layer_db.model.conv2, nn.Sequential)

        for layer in comp_layer_db:
            print("Layer: " + layer.name)
            print("   Module: " + str(layer.module))

        print(comp_layer_db.model)

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    @pytest.mark.parametrize("channels", [(16, 32), (32, 16)])
    @pytest.mark.parametrize("comp_ratio", [Decimal(0.25), Decimal(0.5), Decimal(0.75)])
    @pytest.mark.parametrize("bias", [True, False])
    def test_prune_model_fc(self, device, channels, comp_ratio, bias):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = nn.Linear(channels[0], channels[1], bias=bias)
                self.fc2 = nn.Linear(channels[1], 12, bias=bias)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = Model().eval().to(device)
        dummy_input = torch.randn(1, channels[0]).to(device)
        layer_db = LayerDatabase(model, dummy_input)
        fc1 = layer_db.find_layer_by_name('fc1')
        layer_comp_ratio_list = [LayerCompRatioPair(fc1, comp_ratio)]
        # Using MO implementation
        pruner = WeightSvdPruner()
        mo_layer_db = pruner.prune_model(layer_db, layer_comp_ratio_list, aimet_common.defs.CostMetric.mac,
                                         trainer=None)
        # Using python implementation
        pruner = PyWeightSvdPruner()
        py_layer_db = pruner.prune_model(layer_db, layer_comp_ratio_list, aimet_common.defs.CostMetric.mac,
                                         trainer=None)

        assert id(mo_layer_db.model) != id(py_layer_db.model)
        with torch.no_grad():
            atol = torch.finfo(torch.float16).eps
            assert torch.allclose(mo_layer_db.model(dummy_input), py_layer_db.model(dummy_input), atol=atol)


    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    @pytest.mark.parametrize("channels", [(16, 32), (32, 16)])
    @pytest.mark.parametrize("comp_ratio", [Decimal(0.25), Decimal(0.5), Decimal(0.75)])
    @pytest.mark.parametrize("bias", [True, False])
    def test_prune_model_conv(self, device, channels, comp_ratio, bias):
        torch.manual_seed(0)
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=bias)
                self.conv2 = nn.Conv2d(channels[1], 6, kernel_size=1, bias=bias)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = Model().eval().to(device)
        dummy_input = torch.randn(1, channels[0], 10, 10).to(device)
        layer_db = LayerDatabase(model, dummy_input)
        conv1 = layer_db.find_layer_by_name('conv1')
        layer_comp_ratio_list = [LayerCompRatioPair(conv1, comp_ratio)]
        # Using MO implementation
        pruner = WeightSvdPruner()
        mo_layer_db = pruner.prune_model(layer_db, layer_comp_ratio_list, aimet_common.defs.CostMetric.mac,
                                         trainer=None)
        # Using python implementation
        pruner = PyWeightSvdPruner()
        py_layer_db = pruner.prune_model(layer_db, layer_comp_ratio_list, aimet_common.defs.CostMetric.mac,
                                         trainer=None)

        assert id(mo_layer_db.model) != id(py_layer_db.model)
        with torch.no_grad():
            atol = torch.finfo(torch.float16).eps
            assert torch.allclose(mo_layer_db.model(dummy_input), py_layer_db.model(dummy_input), atol=atol)
