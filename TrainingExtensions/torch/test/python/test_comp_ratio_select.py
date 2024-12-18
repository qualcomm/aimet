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
from unittest.mock import create_autospec
from decimal import Decimal
import math
import os
import signal

from torch import nn
import torch.nn.functional as functional
import aimet_common.libpymo as pymo

from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.cost_calculator import SpatialSvdCostCalculator,WeightSvdCostCalculator
from aimet_common import comp_ratio_select
from aimet_common.bokeh_plots import BokehServerSession
from aimet_common.bokeh_plots import DataTable
from aimet_common.bokeh_plots import ProgressBar
from aimet_common.utils import start_bokeh_server_session

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from .models import mnist_torch_model
from aimet_torch.layer_database import Layer, LayerDatabase
from aimet_torch.svd.svd_pruner import SpatialSvdPruner
from aimet_torch import pymo_utils

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

class TestTrainingExtensionsCompRatioSelect(unittest.TestCase):

    def test_per_layer_eval_scores(self):

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()

        model = mnist_torch_model.Net().to('cpu')

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        layer1 = layer_db.find_layer_by_name('conv1')
        layer_db.mark_picked_layers([layer1])

        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10]

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, 0.5, 10, True, None,
                                                                  None, False, bokeh_session=None)
        progress_bar = ProgressBar(1, "eval scores", "green", bokeh_document=bokeh_session)
        data_table = DataTable(num_columns=3, num_rows=1,
                               column_names=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
                               row_index_names= [layer1.name], bokeh_document=bokeh_session)
        pruner.prune_model.return_value = layer_db
        eval_dict = greedy_algo._compute_layerwise_eval_score_per_comp_ratio_candidate(data_table, progress_bar, layer1)

        self.assertEqual(90, eval_dict[Decimal('0.1')])
        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_eval_scores(self):

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()
        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 91, 81, 71, 61, 51, 41, 31, 21, 11]

        model = mnist_torch_model.Net().to('cpu')

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        layer1 = layer_db.find_layer_by_name('conv1')
        layer2 = layer_db.find_layer_by_name('conv2')
        layer_db.mark_picked_layers([layer1, layer2])

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, 0.5, 10, True, None,
                                                                  None, False, bokeh_session=None)

        eval_dict = greedy_algo._compute_eval_scores_for_all_comp_ratio_candidates()

        self.assertEqual(50, eval_dict['conv1'][Decimal('0.5')])
        self.assertEqual(60, eval_dict['conv1'][Decimal('0.4')])

        self.assertEqual(11, eval_dict['conv2'][Decimal('0.9')])

    def test_eval_scores_with_spatial_svd_pruner(self):

        pruner = SpatialSvdPruner()
        eval_func = unittest.mock.MagicMock()
        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 91, 81, 71, 61, 51, 41, 31, 21, 11]

        model = mnist_torch_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        layer1 = layer_db.find_layer_by_name('conv1')
        layer2 = layer_db.find_layer_by_name('conv2')
        layer_db.mark_picked_layers([layer1, layer2])

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, 0.5, 10, True, None,
                                                                  None, True, bokeh_session=None)
        eval_dict = greedy_algo._compute_eval_scores_for_all_comp_ratio_candidates()

        print()
        print(eval_dict)
        self.assertEqual(90, eval_dict['conv1'][Decimal('0.1')])

        self.assertEqual(51, eval_dict['conv2'][Decimal('0.5')])
        self.assertEqual(21, eval_dict['conv2'][Decimal('0.8')])

    def test_find_min_max_eval_scores(self):

        eval_scores_dict = {'layer1': {Decimal('0.1'): 90, Decimal('0.5'): 50, Decimal('0.7'): 30, Decimal('0.8'): 20},
                            'layer2': {Decimal('0.2'): 91, Decimal('0.3'): 45, Decimal('0.7'): 30, Decimal('0.9'): 11}}

        min_score, max_score = comp_ratio_select.GreedyCompRatioSelectAlgo._find_min_max_eval_scores(eval_scores_dict)

        self.assertEqual(11, min_score)
        self.assertEqual(91, max_score)

        eval_scores_dict = {'layer1': {Decimal('0.1'): 10, Decimal('0.5'): 92, Decimal('0.7'): 30, Decimal('0.8'): 20},
                            'layer2': {Decimal('0.2'): 91, Decimal('0.3'): 45, Decimal('0.7'): 30, Decimal('0.9'): 11}}

        min_score, max_score = comp_ratio_select.GreedyCompRatioSelectAlgo._find_min_max_eval_scores(eval_scores_dict)

        self.assertEqual(10, min_score)
        self.assertEqual(92, max_score)

    def test_find_layer_comp_ratio_given_eval_score(self):

        eval_scores_dict = {'layer1': {Decimal('0.1'): 90, Decimal('0.5'): 50, Decimal('0.7'): 30, Decimal('0.8'): 20},

                            'layer2': {Decimal('0.1'): 11,
                                       Decimal('0.3'): 23,
                                       Decimal('0.5'): 47,
                                       Decimal('0.7'): 85,
                                       Decimal('0.9'): 89}
                            }

        layer2 = Layer(nn.Conv2d(32, 64, 3), "layer2", None)
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo
        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         45,
                                                                         layer2)
        self.assertEqual(Decimal('0.5'), comp_ratio)

        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         48,
                                                                         layer2)
        self.assertEqual(Decimal('0.7'), comp_ratio)

        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         90,
                                                                         layer2)
        self.assertEqual(None, comp_ratio)

    def test_select_per_layer_comp_ratios(self):

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()
        rounding_algo = unittest.mock.MagicMock()
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        eval_func.side_effect = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                                 11, 21, 31, 35, 40, 45, 50, 55, 60]

        model = mnist_torch_model.Net()
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        layer1 = layer_db.find_layer_by_name('conv1')
        layer2 = layer_db.find_layer_by_name('conv2')
        selected_layers = [layer1, layer2]
        layer_db.mark_picked_layers([layer1, layer2])

        try:
            os.remove('./data/greedy_selection_eval_scores_dict.pkl')
        except OSError:
            pass

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, Decimal(0.6), 10, True,
                                                                  None, rounding_algo, False, bokeh_session=None)

        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.6), actual_compression_ratio, abs_tol=0.05))
        self.assertTrue(os.path.isfile('./data/greedy_selection_eval_scores_dict.pkl'))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)

        # lets repeat with a saved eval_dict
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, Decimal(0.6), 10, True,
                                                                  './data/greedy_selection_eval_scores_dict.pkl',
                                                                  rounding_algo, False, bokeh_session=None)
        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)

        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.6), actual_compression_ratio, abs_tol=0.05))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)

    def test_select_per_layer_comp_ratios_with_spatial_svd_pruner(self):

        pruner = SpatialSvdPruner()
        eval_func = unittest.mock.MagicMock()
        rounding_algo = unittest.mock.MagicMock()
        eval_func.side_effect = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                                 11, 21, 31, 35, 40, 45, 50, 55, 60]
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        model = mnist_torch_model.Net()
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        selected_layers = [layer for layer in layer_db if isinstance(layer.module, nn.Conv2d)]
        layer_db.mark_picked_layers(selected_layers)

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db, pruner, SpatialSvdCostCalculator(),
                                                                  eval_func, 20, CostMetric.mac, Decimal(0.4), 10, True,
                                                                  None, rounding_algo, False, bokeh_session=None)
        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)

        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.3), actual_compression_ratio, abs_tol=0.8))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)
