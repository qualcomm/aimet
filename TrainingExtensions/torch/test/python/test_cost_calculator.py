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
import math
import unittest
from decimal import Decimal

import torch
import torch.nn as nn

from aimet_common import cost_calculator as common_cost_calculator
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.utils import AimetLogger
import aimet_torch._base.nn.modules.custom as aimet_modules
from aimet_torch.channel_pruning.channel_pruner import (
    InputChannelPruner,
    ChannelPruningCostCalculator,
)
from aimet_torch.layer_database import Layer, LayerDatabase
from aimet_torch.utils import (
    create_rand_tensors_given_shapes,
    create_fake_data_loader,
    get_device,
)
from aimet_torch import cost_calculator as pt_cost_calculator

from .models import mnist_torch_model, test_models

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


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


class TestTrainingExtensionsCostCalculator(unittest.TestCase):

    def test_compute_layer_cost(self):

        logger.debug(self.id())

        conv1 = nn.Conv2d(1, 32, kernel_size=5)
        layer1 = Layer(conv1, "conv1", (1, 32, 28, 28))
        cost1 = common_cost_calculator.CostCalculator.compute_layer_cost(layer1)

        self.assertEqual(32 * 1 * 5 * 5, cost1.memory)
        self.assertEqual(32 * 1 * 5 * 5 * 28 * 28, cost1.mac)

        conv2 = nn.Conv2d(32, 64, kernel_size=5)
        layer2 = Layer(conv2, "conv2", (1, 64, 14, 14))
        cost2 = common_cost_calculator.CostCalculator.compute_layer_cost(layer2)

        self.assertEqual(64 * 32 * 5 * 5, cost2.memory)
        self.assertEqual(64 * 32 * 5 * 5 * 14 * 14, cost2.mac)

        conv2 = nn.Conv2d(32, 32, kernel_size=5, groups=32)
        layer2 = Layer(conv2, "conv2", (1, 32, 14, 14))
        cost2 = common_cost_calculator.CostCalculator.compute_layer_cost(layer2)

        self.assertEqual(32 * 1 * 5 * 5, cost2.memory)
        self.assertEqual(32 * 1 * 5 * 5 * 14 * 14, cost2.mac)


    def test_compute_layer_cost_pt(self):
        logger.debug(self.id())

        conv1 = nn.Conv2d(1, 32, kernel_size=5)
        layer1 = Layer(conv1, "conv1", (1, 32, 28, 28))
        cost1 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer1)

        self.assertEqual(32 * 1 * 5 * 5, cost1.memory)
        self.assertEqual(32 * 1 * 5 * 5 * 28 * 28, cost1.mac)

        conv2 = nn.Conv2d(32, 64, kernel_size=5)
        layer2 = Layer(conv2, "conv2", (1, 64, 14, 14))
        cost2 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer2)

        self.assertEqual(64 * 32 * 5 * 5, cost2.memory)
        self.assertEqual(64 * 32 * 5 * 5 * 14 * 14, cost2.mac)

        conv2 = nn.Conv2d(32, 32, kernel_size=5, groups=32)
        layer2 = Layer(conv2, "conv2", (1, 32, 14, 14))
        cost2 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer2)

        self.assertEqual(32 * 1 * 5 * 5, cost2.memory)
        self.assertEqual(32 * 1 * 5 * 5 * 14 * 14, cost2.mac)

        linear1 = nn.Linear(10, 15, bias=False)
        layer3 = Layer(linear1, "linear1", (32, 10, 10))
        cost3 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer3)

        self.assertEqual(15 * 10 * 1 * 1 * 10 * 10, cost3.mac)
        self.assertEqual(15 * 10, cost3.memory)

        sigmoid1 = nn.Sigmoid()
        layer4 = Layer(sigmoid1, "sigmoid1", (32, 10, 15), (32, 10, 15))
        cost4 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer4)

        self.assertEqual(32 * 10 * 15, cost4.mac)
        self.assertEqual(32 * 10 * 15, cost4.memory)

        multiply1 = aimet_modules.Multiply()
        layer5 = Layer(multiply1, "multiply1", (32, 10, 15), [(32, 10, 15), (32, 10, 15)])
        cost5 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer5)

        self.assertEqual(32 * 10 * 15, cost5.mac)
        self.assertEqual(32 * 10 * 15, cost5.memory)

        layer6 = Layer(torch.nn.AvgPool2d(4), "AvgPool1", (32, 10, 15), (32, 10, 15))
        cost6 = pt_cost_calculator.CostCalculator.compute_layer_cost(layer6)

        self.assertEqual(32 * 10 * 15, cost6.mac)
        self.assertEqual(32 * 10 * 15, cost6.memory)


    def test_total_model_cost(self):

        logger.debug(self.id())
        model = MnistSequentialModel().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        cost_calc = common_cost_calculator.CostCalculator()
        network_cost = cost_calc.compute_model_cost(layer_db)

        self.assertEqual(800 + 51200 + 3211264 + 10240, network_cost.memory)
        self.assertEqual(627200 + 10035200 + 3211264 + 10240, network_cost.mac)

    def test_mac_count_of_grouped_conv_net(self):
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

        with torch.inference_mode():
            standard_module_outputs = standard_grouped_conv_model(standard_module_inputs)
            custom_module_outputs = custom_grouped_conv_model(*custom_module_inputs)
        self.assertTrue(torch.allclose(standard_module_outputs, custom_module_outputs, atol=1e-5))

        standard_grouped_conv_db = LayerDatabase(standard_grouped_conv_model, standard_module_inputs)
        custom_grouped_conv_db = LayerDatabase(custom_grouped_conv_model, custom_module_inputs)
        cost_calc = common_cost_calculator.CostCalculator()

        standard_grouped_conv_cost = cost_calc.compute_model_cost(standard_grouped_conv_db)
        custom_grouped_conv_cost = cost_calc.compute_model_cost(custom_grouped_conv_db)
        self.assertEqual(standard_grouped_conv_cost.mac, custom_grouped_conv_cost.mac)


class TestTrainingExtensionsSpatialSvdCostCalculator(unittest.TestCase):

    def test_calculate_spatial_svd_cost(self):

        conv = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2))
        layer = Layer(conv, "conv", output_shape=[1, 64, 28, 28])

        self.assertEqual(32 * 5, common_cost_calculator.SpatialSvdCostCalculator.calculate_max_rank(layer))

        comp_ratios_to_check = [0.8, 0.75, 0.5, 0.25, 0.125]

        original_cost = common_cost_calculator.CostCalculator.compute_layer_cost(layer)

        for comp_ratio in comp_ratios_to_check:

            rank = common_cost_calculator.SpatialSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, CostMetric.mac)
            print('Rank = {}, for compression_ratio={}'.format(rank, comp_ratio))
            compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_cost_given_rank(layer, rank)

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.01))

        # Higher level API
        for comp_ratio in comp_ratios_to_check:

            compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_per_layer_compressed_cost(layer, comp_ratio,
                                                                                              CostMetric.mac)

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.01))

    def test_calculate_spatial_svd_cost_linear_layer(self):

        linear = nn.Linear(128, 256)

        input_shape=(1, 128)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(linear))
        layer_db = LayerDatabase(linear, dummy_input)

        layer = layer_db.find_layer_by_module(linear)

        self.assertEqual(128, common_cost_calculator.SpatialSvdCostCalculator.calculate_max_rank(layer))

        comp_ratios_to_check = [1.0, 0.8, 0.75, 0.5, 0.25, 0.125]

        original_cost = common_cost_calculator.CostCalculator.compute_layer_cost(layer)
        self.assertEqual(128 * 256, original_cost.mac)
        self.assertEqual(128 * 256, original_cost.memory)

        for comp_ratio in comp_ratios_to_check:
            rank = common_cost_calculator.SpatialSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, CostMetric.mac)
            print('Rank = {}, for compression_ratio={}'.format(rank, comp_ratio))
            compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_cost_given_rank(layer, rank)

            self.assertTrue(math.isclose(compressed_cost.mac / original_cost.mac, comp_ratio, abs_tol=0.01))

        # Higher level API
        for comp_ratio in comp_ratios_to_check:
            compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_per_layer_compressed_cost(layer, comp_ratio,
                                                                                              CostMetric.mac)

            self.assertTrue(math.isclose(compressed_cost.mac / original_cost.mac, comp_ratio, abs_tol=0.01))

    def test_calculate_spatial_svd_cost_with_stride(self):

        conv = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2), stride=2)
        layer = Layer(conv, "conv", output_shape=[1, 64, 14, 14])

        original_cost = common_cost_calculator.CostCalculator.compute_layer_cost(layer)
        compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_cost_given_rank(layer, 40)

        self.assertEqual(10035200, original_cost.mac)
        self.assertEqual(5017600, compressed_cost.mac)
        print(original_cost)
        print(compressed_cost)

    def test_calculate_spatial_svd_cost_all_layers(self):

        model = mnist_torch_model.Net().to("cpu")
        print(model)

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        model_cost = common_cost_calculator.SpatialSvdCostCalculator.compute_model_cost(layer_db)
        self.assertEqual(627200 + 10035200 + 3211264 + 10240, model_cost.mac)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        for layer in layer_db:
            layer_ratio_list.append(LayerCompRatioPair(layer, Decimal(0.5)))

        compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_compressed_cost(layer_db,
                                                                                layer_ratio_list, CostMetric.mac)
        self.assertEqual(5244960
                         + (3136 * 385 + 385 * 1024)
                         + (1024 * 4 + 4 * 10),
                         compressed_cost.mac)

    def test_calculate_spatial_svd_cost_all_layers_given_ranks(self):

        model = mnist_torch_model.Net().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_rank_list = [(layer_db.find_layer_by_module(model.conv1), 2),
                           (layer_db.find_layer_by_module(model.conv2), 53),
                           (layer_db.find_layer_by_module(model.fc1), 385),
                           (layer_db.find_layer_by_module(model.fc2), 4)]

        compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_compressed_cost_given_ranks(layer_db,
                                                                                            layer_rank_list)
        self.assertEqual(5244960
                         + (3136 * 385 + 385 * 1024)
                         + (1024 * 4 + 4 * 10),
                         compressed_cost.mac)

        # Create a list of tuples of (layer, comp_ratio)
        layer_rank_list = [(layer_db.find_layer_by_module(model.conv1), 2),
                           (layer_db.find_layer_by_module(model.conv2), 53),
                           (layer_db.find_layer_by_module(model.fc1), 385),
                           (layer_db.find_layer_by_module(model.fc2), None)]

        compressed_cost = common_cost_calculator.SpatialSvdCostCalculator.calculate_compressed_cost_given_ranks(layer_db,
                                                                                            layer_rank_list)
        self.assertEqual(5244960
                         + (3136 * 385 + 385 * 1024)
                         + (1024 * 10),
                         compressed_cost.mac)


class TestTrainingExtensionsWeightSvdCostCalculator(unittest.TestCase):

    def test_calculate_weight_svd_cost(self):

        conv = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2))
        layer = Layer(conv, "conv", output_shape=[1, 64, 28, 28])

        self.assertEqual(32, common_cost_calculator.WeightSvdCostCalculator.calculate_max_rank(layer))

        comp_ratios_to_check = [0.8, 0.75, 0.5, 0.25, 0.125]

        original_cost = common_cost_calculator.CostCalculator.compute_layer_cost(layer)

        for comp_ratio in comp_ratios_to_check:

            rank = common_cost_calculator.WeightSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, CostMetric.mac)
            print('Rank = {}, for compression_ratio={}'.format(rank, comp_ratio))
            compressed_cost = common_cost_calculator.WeightSvdCostCalculator.calculate_cost_given_rank(layer, rank)
            print('Compressed cost={}, compression_ratio={}'.format(compressed_cost,
                                                                    compressed_cost.mac/original_cost.mac))

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.03))

        # Higher level API
        for comp_ratio in comp_ratios_to_check:

            compressed_cost = common_cost_calculator.WeightSvdCostCalculator.calculate_per_layer_compressed_cost(layer, comp_ratio,
                                                                                             CostMetric.mac)

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.03))

    def test_calculate_weight_svd_cost_all_layers(self):

        model = mnist_torch_model.Net().to("cpu")
        print(model)

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        for layer in layer_db:
            if isinstance(layer.module, nn.Conv2d):
                layer_ratio_list.append(LayerCompRatioPair(layer, Decimal('0.5')))
            else:
                layer_ratio_list.append(LayerCompRatioPair(layer, Decimal('0.5')))

        compressed_cost = common_cost_calculator.WeightSvdCostCalculator.calculate_compressed_cost(layer_db,
                                                                               layer_ratio_list, CostMetric.mac)

        self.assertEqual(7031800, compressed_cost.mac)


class TestTrainingExtensionsChannelPruningCostCalculator(unittest.TestCase):

    def test_calculate_channel_pruning_cost_all_layers(self):

        model = mnist_torch_model.Net().to("cpu")
        print(model)

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        # Unfortunately in mnist we can only input channel prune conv2
        for layer in layer_db:
            if layer.module is model.conv2:
                layer_ratio_list.append(LayerCompRatioPair(layer, Decimal('0.5')))
            else:
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        # Create the Input channel pruner
        dataset_size = 1000
        batch_size = 10

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 1, 28, 28),
                                    num_reconstruction_samples=10,
                                    allow_custom_downsample_ops=True)

        cost_calculator = ChannelPruningCostCalculator(pruner)

        compressed_cost = cost_calculator.calculate_compressed_cost(layer_db,
                                                                    layer_ratio_list, CostMetric.mac)

        self.assertEqual(8552704, compressed_cost.mac)
