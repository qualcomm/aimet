# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import numpy as np
import unittest
import copy
from decimal import Decimal

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.winnow.winnow_utils import to_numpy
from .models import mnist_torch_model
from aimet_torch.svd.svd_splitter import SpatialSvdModuleSplitter
from aimet_torch.svd.svd_pruner import SpatialSvdPruner
from aimet_torch.layer_database import LayerDatabase
from aimet_common.defs import CostMetric, LayerCompRatioPair


def get_data_loader(data_set_size, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_loader = torch.utils.data.DataLoader(
        datasets.FakeData(size=data_set_size, image_size=(1, 28, 28), num_classes=10,
                          transform=transform, target_transform=None, random_offset=0),
        batch_size=batch_size, shuffle=False)
    return data_loader


class _TestNet(torch.nn.Module):
    def __init__(self):
        super(_TestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = torch.nn.Linear(800, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class _TestNetStrided(torch.nn.Module):
    def __init__(self):
        super(_TestNetStrided, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5, stride=(2, 2))
        self.fc1 = torch.nn.Linear(200, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TestSpatialSvdLayerSplit(unittest.TestCase):

    def test_split_layer(self):
        model = _TestNet()

        layer = model.conv2

        num_examples = 2000

        input_data = np.random.normal(size=[num_examples,
                                            layer.in_channels,
                                            *layer.kernel_size])

        output_data = F.conv2d(torch.FloatTensor(input_data), layer.weight, bias=layer.bias)

        first_layer, second_layer = SpatialSvdModuleSplitter.split_module(module=layer, rank=100)
        seq_layers = torch.nn.Sequential(first_layer, second_layer)
        new_output = to_numpy(seq_layers.forward(torch.FloatTensor(input_data)))

        output_data = to_numpy(output_data)

        assert np.allclose(new_output, output_data, atol=1e-5)

    def test_split_layer_with_stride(self):
        model = _TestNetStrided()

        layer = model.conv2
        num_examples = 2000
        k_x, k_t = layer.kernel_size

        input_data = np.random.normal(size=[num_examples,
                                            layer.in_channels,
                                            k_x + 2,
                                            k_t + 2])

        output_data = layer(torch.FloatTensor(input_data))

        first_layer, second_layer = SpatialSvdModuleSplitter.split_module(module=layer, rank=100)
        seq_layers = torch.nn.Sequential(first_layer, second_layer)

        new_output = seq_layers.forward(torch.FloatTensor(input_data))

        assert np.allclose(new_output.detach(), output_data.detach(), atol=1e-5)

    def test_split_layer_rank_reduced(self):
        model = _TestNet()

        layer = model.conv2
        num_examples = 2000
        input_data = np.random.normal(size=[num_examples,
                                            layer.in_channels,
                                            *layer.kernel_size])

        layer.weight.data[:, 0] = 0

        output_data = F.conv2d(torch.FloatTensor(input_data), layer.weight, bias=layer.bias)
        first_layer, second_layer = SpatialSvdModuleSplitter.split_module(module=layer,
                                                                          rank=96)
        seq_layers = torch.nn.Sequential(first_layer, second_layer)

        new_output = torch.relu(seq_layers.forward(torch.FloatTensor(input_data)))
        output_data = torch.relu(output_data).detach()

        assert np.allclose(new_output.detach(), output_data, atol=1e-5)


class TestSpatialSvdPruning(unittest.TestCase):

    def test_prune_layer(self):

        model = mnist_torch_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        conv1 = comp_layer_db.find_layer_by_name('conv1')
        spatial_svd_pruner = SpatialSvdPruner()
        spatial_svd_pruner._prune_layer(orig_layer_db, comp_layer_db, conv1, 0.5, CostMetric.mac)

        conv1_a = comp_layer_db.find_layer_by_name('conv1.0')
        conv1_b = comp_layer_db.find_layer_by_name('conv1.1')

        self.assertEqual((5, 1), conv1_a.module.kernel_size)
        self.assertEqual(1, conv1_a.module.in_channels)
        self.assertEqual(2, conv1_a.module.out_channels)

        self.assertEqual((1, 5), conv1_b.module.kernel_size)
        self.assertEqual(2, conv1_b.module.in_channels)
        self.assertEqual(32, conv1_b.module.out_channels)

        self.assertTrue(isinstance(comp_layer_db.model.conv1, torch.nn.Sequential))

        for layer in comp_layer_db:
            print("Layer: " + layer.name)
            print("   Module: " + str(layer.module))

        print(comp_layer_db.model)

        # check the output shapes of two newly created split layers
        # first split layer output
        conv1_a_output = comp_layer_db.model.conv1[0](torch.rand(1, 1, 28, 28))

        # second split layer output
        conv1_b_output = comp_layer_db.model.conv1[1](conv1_a_output)

        self.assertEqual(conv1_a.output_shape, list(conv1_a_output.shape))
        self.assertEqual(conv1_b.output_shape, list(conv1_b_output.shape))

    def test_prune_model_2_layers(self):

        model = mnist_torch_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        conv1 = comp_layer_db.find_layer_by_name('conv1')
        conv2 = comp_layer_db.find_layer_by_name('conv2')
        pruner = SpatialSvdPruner()

        layer_db = pruner.prune_model(orig_layer_db, [LayerCompRatioPair(conv1, Decimal(0.5)),
                                                      LayerCompRatioPair(conv2, Decimal(0.5))], CostMetric.mac,
                                      trainer=None)

        conv1_a = layer_db.find_layer_by_name('conv1.0')
        conv1_b = layer_db.find_layer_by_name('conv1.1')

        self.assertEqual((5, 1), conv1_a.module.kernel_size)
        self.assertEqual(1, conv1_a.module.in_channels)
        self.assertEqual(2, conv1_a.module.out_channels)

        self.assertEqual((1, 5), conv1_b.module.kernel_size)
        self.assertEqual(2, conv1_b.module.in_channels)
        self.assertEqual(32, conv1_b.module.out_channels)

        conv2_a = layer_db.find_layer_by_name('conv2.0')
        conv2_b = layer_db.find_layer_by_name('conv2.1')

        self.assertEqual((5, 1), conv2_a.module.kernel_size)
        self.assertEqual(32, conv2_a.module.in_channels)
        self.assertEqual(53, conv2_a.module.out_channels)

        self.assertEqual((1, 5), conv2_b.module.kernel_size)
        self.assertEqual(53, conv2_b.module.in_channels)
        self.assertEqual(64, conv2_b.module.out_channels)

        self.assertTrue(isinstance(layer_db.model.conv1, torch.nn.Sequential))
        self.assertTrue(isinstance(layer_db.model.conv2, torch.nn.Sequential))

        for layer in layer_db:
            print("Layer: " + layer.name)
            print("   Module: " + str(layer.module))

        print(layer_db.model)
