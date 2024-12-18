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

import unittest
import unittest.mock
from decimal import Decimal
import torch
import torch.nn as nn
import torch.nn.functional as F

from aimet_common.comp_ratio_select import GreedyCompRatioSelectAlgo
from aimet_common.defs import CostMetric
from aimet_common.cost_calculator import SpatialSvdCostCalculator
from aimet_common.compression_algo import CompressionAlgo

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from .models import mnist_torch_model
from aimet_torch.layer_database import LayerDatabase
from aimet_torch.svd.svd_pruner import SpatialSvdPruner
from aimet_torch.layer_selector import ConvNoDepthwiseLayerSelector


class ModelWithTwoInputs(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputs, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(F.max_pool2d(self.conv1_a(x1), 2))
        x2 = F.relu(F.max_pool2d(self.conv1_b(x2), 2))
        x = x1 + x2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TestTrainingExtensionsCompressionAlgo(unittest.TestCase):

    def testSpatialSvd(self):

        torch.manual_seed(1)

        model = mnist_torch_model.Net()

        rounding_algo = unittest.mock.MagicMock()
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        mock_eval = unittest.mock.MagicMock()
        mock_eval.side_effect = [100,
                                 90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 50]

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        pruner = SpatialSvdPruner()
        comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                           mock_eval, 20, CostMetric.mac, Decimal(0.5),
                                                           10, True, None, rounding_algo, True, bokeh_session=None)

        layer_selector = ConvNoDepthwiseLayerSelector()
        spatial_svd_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner,
                                           mock_eval,
                                           layer_selector, modules_to_ignore=[],
                                           cost_calculator=SpatialSvdCostCalculator(),
                                           use_cuda=next(model.parameters()).is_cuda)

        compressed_layer_db, stats = spatial_svd_algo.compress_model(CostMetric.mac, trainer=None)

        self.assertTrue(isinstance(compressed_layer_db.model.conv1, torch.nn.Sequential))
        self.assertTrue(isinstance(compressed_layer_db.model.conv2, torch.nn.Sequential))
        self.assertTrue(stats.per_layer_stats[0].compression_ratio <= 0.5)
        self.assertEqual(0.3, stats.per_layer_stats[1].compression_ratio)

        print("Compressed model:")
        print(compressed_layer_db.model)

        print(stats)

    def testSpatialSvdMultiInputModel(self):

        torch.manual_seed(1)

        model = ModelWithTwoInputs()

        rounding_algo = unittest.mock.MagicMock()
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        mock_eval = unittest.mock.MagicMock()
        mock_eval.side_effect = [100,
                                 90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 50]

        input_shape=[(1, 1, 28, 28), (1, 1, 28, 28)]
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        pruner = SpatialSvdPruner()
        comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                           mock_eval, 20, CostMetric.mac, Decimal(0.5),
                                                           10, True, None, rounding_algo, True, bokeh_session=None)

        layer_selector = ConvNoDepthwiseLayerSelector()
        spatial_svd_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner,
                                           mock_eval,
                                           layer_selector, modules_to_ignore=[],
                                           cost_calculator=SpatialSvdCostCalculator(),
                                           use_cuda=next(model.parameters()).is_cuda)

        compressed_layer_db, stats = spatial_svd_algo.compress_model(CostMetric.mac, trainer=None)

        self.assertTrue(isinstance(compressed_layer_db.model.conv1_a, torch.nn.Sequential))
        self.assertTrue(isinstance(compressed_layer_db.model.conv2, torch.nn.Sequential))
        self.assertTrue(stats.per_layer_stats[0].compression_ratio <= 0.5)

        print("Compressed model:")
        print(compressed_layer_db.model)

        print(stats)
