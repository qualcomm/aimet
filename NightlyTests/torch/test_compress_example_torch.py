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

import itertools
import logging
import os
import pytest
import unittest
from unittest.mock import MagicMock
from decimal import Decimal

import math
import numpy
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import models

import aimet_common.defs
import aimet_torch.defs
from aimet_common.defs import RankSelectScheme
from aimet_common.utils import AimetLogger, start_bokeh_server_session
from aimet_common.data_cache_utility import is_cache_env_set, is_mnist_cache_present, copy_mnist_to_cache,\
    copy_cache_mnist_to_local_build
from aimet_common.compression_algo import CompressionAlgo

import aimet_torch.utils
from aimet_torch.compress import ModelCompressor
from aimet_torch.defs import ModuleCompRatioPair, ChannelPruningParameters
from models import mnist_torch_model
from models.imagenet_dataloader import ImageNetDataLoader
from models.supervised_classification_pipeline import \
    create_stand_alone_supervised_classification_evaluator
from aimet_torch.visualize_serialized_data import VisualizeCompression


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

image_dir = './data/tiny-imagenet-200'
image_size = 224
batch_size = 5
num_workers = 1


class ModelWithTwoInputs(torch.nn.Module):

    def __init__(self):
        super(ModelWithTwoInputs, self).__init__()
        self.conv1_a = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_b = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x1, x2):
        F = torch.nn.functional
        x1 = F.relu(F.max_pool2d(self.conv1_a(x1), 2))
        x2 = F.relu(F.max_pool2d(self.conv1_b(x2), 2))
        x = x1 + x2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DataLoaderMnist:
    """ A dataloader for the MNIST dataset """

    def __init__(self, cuda, seed, shuffle, train_batch_size=64, test_batch_size=100):
        """
        Constructor

        :param cuda: If True, data will be loaded in GPU memory
        :param seed: Seed to use for randomization (to help with reproducibility)
        :param shuffle: If we want data to be shuffled
        :param train_batch_size: Batch size for train data
        :param test_batch_size: Batch size for test data
        """

        self._cuda = cuda
        self._seed = seed
        self._shuffle = shuffle
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        # set the GPU flags appropriately
        # to allocate data on GPU or CPU
        self._use_cuda = self._cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self._use_cuda else "cpu")
        self._kwargs = {'num_workers': 1, 'pin_memory': True} if self._use_cuda else {}
        # set the seed value
        torch.manual_seed(self._seed)
        mnist_download = True

        if is_mnist_cache_present():
            copy_cache_mnist_to_local_build()
            mnist_download = False

        # train loader
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=mnist_download,
                                                                       transform=transforms.Compose(
                                                                           [transforms.ToTensor(),
                                                                            transforms.Normalize((0.5307,),
                                                                                                 (0.9081,))])),
                                                        batch_size=self._train_batch_size, shuffle=self._shuffle,
                                                        **self._kwargs)

        # test loader
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=False,
                                                                      transform=transforms.Compose(
                                                                          [transforms.ToTensor(),
                                                                           transforms.Normalize((0.5307,),
                                                                                                (0.9081,))])),
                                                       batch_size=self._test_batch_size, shuffle=self._shuffle,
                                                       **self._kwargs)
        """
        If MNIST is not present in the cache, copy the data to Cache, so from next time
        we do not have to download it again, and data would be used from Cache.
        """
        if not is_mnist_cache_present() and is_cache_env_set():
            copy_mnist_to_cache()


class SvdAcceptanceTests(unittest.TestCase):

    def test_spatial_svd_compress_manual(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18()

        manual_params = aimet_torch.defs.SpatialSvdParameters.ManualModeParams(
            [ModuleCompRatioPair(model.layer1[0].conv1, 0.5),
             ModuleCompRatioPair(model.layer2[1].conv2, 0.4)])

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.manual, manual_params,
                                                       multiplicity=8)

        # Only used in this test for baseline and final accuracy - essentially a don't care
        mock_eval = MagicMock()
        mock_eval.return_value = 50

        results = ModelCompressor.compress_model(model=model, eval_callback=mock_eval, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        # Check that indeed weight svd was applied to some layer
        self.assertTrue(isinstance(compressed_model.layer1[0].conv1, torch.nn.Sequential))
        self.assertEqual(48, compressed_model.layer1[0].conv1[0].out_channels)
        self.assertEqual((3, 1), compressed_model.layer1[0].conv1[0].kernel_size)
        self.assertEqual((1, 0), compressed_model.layer1[0].conv1[0].padding)

        self.assertTrue(isinstance(compressed_model.layer2[1].conv2, torch.nn.Sequential))
        self.assertEqual(80, compressed_model.layer2[1].conv2[0].out_channels)

    def test_spatial_svd_compress_auto(self):

        torch.manual_seed(1)
        numpy.random.seed(1)

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18()

        modules_to_ignore = [model.conv1,
                             model.layer1[0].conv1, model.layer1[0].conv2,
                             model.layer1[1].conv1, model.layer1[1].conv2,
                             model.layer2[0].downsample[0],

                             model.layer3[0].conv1, model.layer3[0].conv2,
                             model.layer3[1].conv1, model.layer3[1].conv2,
                             model.layer3[0].downsample[0],

                             model.layer4[0].conv1, model.layer4[0].conv2,
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=4)

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=8)

        mock_eval = FakeEvaluator(input_shape)
        mock_eval.return_values = [0.75, 0.50, 0.25, 0.75, 0.50, 0.25, 0.75, 0.50, 0.25, 0.75, 0.50, 0.25,
                                   0.50, 0.50]

        results = ModelCompressor.compress_model(model=model, eval_callback=mock_eval, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)
        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertEqual(48, compressed_model.layer2[0].conv2[0].out_channels)
        self.assertEqual((3, 1), compressed_model.layer2[0].conv2[0].kernel_size)
        self.assertTrue(math.isclose(float(stats.mac_compression_ratio), 0.83, abs_tol=0.01))

    def test_spatial_svd_compress_auto_multi_input_model(self):

        torch.manual_seed(1)
        numpy.random.seed(1)

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = [(1, 1, 28, 28), (1, 1, 28, 28)]

        model = ModelWithTwoInputs()

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=4)

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params)

        mock_eval = FakeEvaluator(input_shape)
        mock_eval.return_values = [0.75, 0.50, 0.25, 0.75, 0.50, 0.25, 0.75, 0.50, 0.25,
                                   0.50, 0.50]

        results = ModelCompressor.compress_model(model=model, eval_callback=mock_eval, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)
        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertEqual(8, compressed_model.conv2[0].out_channels)
        self.assertEqual((5, 1), compressed_model.conv2[0].kernel_size)
        self.assertTrue(math.isclose(float(stats.mac_compression_ratio), 0.25, abs_tol=0.01))

    def test_spatial_svd_compress_auto_with_high_multiplicity(self):

        torch.manual_seed(1)
        numpy.random.seed(1)

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18()

        modules_to_ignore = [model.conv1,
                             model.layer1[0].conv1, model.layer1[0].conv2,
                             model.layer1[1].conv1, model.layer1[1].conv2,
                             model.layer2[0].downsample[0],

                             model.layer3[0].conv1, model.layer3[0].conv2,
                             model.layer3[1].conv1, model.layer3[1].conv2,
                             model.layer3[0].downsample[0],

                             model.layer4[0].conv1, model.layer4[0].conv2,
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=4)

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=64)

        mock_eval = FakeEvaluator(input_shape)
        mock_eval.return_values = [0.75, 0.50, 0.25, 0.75, 0.50, 0.25, 0.75, 0.50, 0.25, 0.75, 0.50, 0.25,
                                   0.50, 0.50]

        results = ModelCompressor.compress_model(model=model, eval_callback=mock_eval, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)
        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertEqual(64, compressed_model.layer2[0].conv2[0].out_channels)
        self.assertEqual((3, 1), compressed_model.layer2[0].conv2[0].kernel_size)
        self.assertTrue(math.isclose(float(stats.mac_compression_ratio), 0.87, abs_tol=0.01))

    @pytest.mark.cuda
    def test_weight_svd_compress_manual(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))

        manual_params = aimet_torch.defs.WeightSvdParameters.ManualModeParams(
            [ModuleCompRatioPair(model.layer1[0].conv1, 0.5),
             ModuleCompRatioPair(model.layer2[1].conv2, 0.4), ModuleCompRatioPair(model.fc, 0.4)])

        params = aimet_torch.defs.WeightSvdParameters(aimet_torch.defs.WeightSvdParameters.Mode.manual, manual_params)

        results = ModelCompressor.compress_model(model, evaluate, 10, input_shape,
                                                 aimet_common.defs.CompressionScheme.weight_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertTrue(isinstance(compressed_model.layer1[0].conv1, torch.nn.Sequential))
        self.assertEqual(math.floor(64 * 64 * 3 * 3 * 0.5 / (64 + 64 * 3 * 3)),
                         compressed_model.layer1[0].conv1[0].out_channels)

        self.assertTrue(isinstance(compressed_model.fc, torch.nn.Sequential))
        self.assertEqual(math.floor(512 * 1000 * 0.4 / (512 + 1000)), compressed_model.fc[0].out_features)

    @pytest.mark.cuda
    def test_weight_svd_compress_auto_greedy(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))
        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2,
                             model.fc
                             ]
        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.7),
                                                                    num_comp_ratio_candidates=4,
                                                                    saved_eval_scores_dict=
                                                                   './data/resnet18_eval_scores.pkl')
        rank_select = RankSelectScheme.greedy
        auto_params = aimet_torch.defs.WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                                          select_params=greedy_params,
                                                                          modules_to_ignore=modules_to_ignore)
        params = aimet_torch.defs.WeightSvdParameters(aimet_torch.defs.WeightSvdParameters.Mode.auto, auto_params,
                                                      multiplicity=8)

        results = ModelCompressor.compress_model(model, evaluate, 10, input_shape,
                                                 aimet_common.defs.CompressionScheme.weight_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertFalse(isinstance(compressed_model.conv1, torch.nn.Sequential))
        self.assertFalse(isinstance(compressed_model.fc, torch.nn.Sequential))

    @pytest.mark.cuda
    def test_weight_svd_compress_auto_high_multiplicity(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))
        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2,
                             model.fc
                             ]
        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.7),
                                                                    num_comp_ratio_candidates=4,
                                                                    saved_eval_scores_dict=
                                                                   './data/resnet18_eval_scores.pkl')
        rank_select = RankSelectScheme.greedy
        auto_params = aimet_torch.defs.WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                                          select_params=greedy_params,
                                                                          modules_to_ignore=modules_to_ignore)
        params = aimet_torch.defs.WeightSvdParameters(aimet_torch.defs.WeightSvdParameters.Mode.auto, auto_params,
                                                      multiplicity=64)

        results = ModelCompressor.compress_model(model, evaluate, 10, input_shape,
                                                 aimet_common.defs.CompressionScheme.weight_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)

        self.assertFalse(isinstance(compressed_model.conv1, torch.nn.Sequential))
        self.assertFalse(isinstance(compressed_model.fc, torch.nn.Sequential))

    @pytest.mark.cuda
    def test_spatial_svd_with_fine_tuning(self):
        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        # load trained MNIST model
        data_loader = DataLoaderMnist(cuda=True, seed=1, shuffle=False, train_batch_size=64, test_batch_size=100)
        input_shape = (1, 1, 28, 28)
        model = torch.load(os.path.join('./', 'data', 'mnist_trained_on_GPU.pth'))
        modules_to_ignore = [model.conv1]
        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                                                    num_comp_ratio_candidates=10,
                                                                    use_monotonic_fit=True)
        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=1)
        results = ModelCompressor.compress_model(model, mnist_torch_model.evaluate, 10, input_shape,
                                                 aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 trainer=Trainer(), visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)
        self.assertTrue(math.isclose(float(stats.mac_compression_ratio), 0.7, abs_tol=0.1))

    @unittest.skip
    def test_pickled_compression_ratios(self):

        visualization_url, process = start_bokeh_server_session(8002)

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True

        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))

        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]
        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10)

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=8)

        ModelCompressor.compress_model(model=model, eval_callback=evaluate, eval_iterations=5,
                                       input_shape=input_shape,
                                       compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                       cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                       visualization_url=None)

        comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
        eval_scores_path = './data/greedy_selection_eval_scores_dict.pkl'
        self.assertTrue(os.path.exists(comp_ratios_file_path))
        layer_comp_ratio_list = CompressionAlgo.unpickle_comp_ratios_list(comp_ratios_file_path)
        self.assertEqual(type(layer_comp_ratio_list), list)
        compression_visualizations = VisualizeCompression(visualization_url)
        compression_visualizations.display_eval_scores(eval_scores_path)
        compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


class ChannelPruningAcceptanceTests(unittest.TestCase):

    @pytest.mark.cuda
    def test_channel_pruning_manual(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))
        manual_params = ChannelPruningParameters.ManualModeParams([ModuleCompRatioPair(model.layer1[0].conv2, 0.3),
                                                                   ModuleCompRatioPair(model.layer2[1].conv1, 0.5)])
        params = ChannelPruningParameters(data_loader.train_loader, 5000,
                                          True,
                                          aimet_torch.defs.ChannelPruningParameters.Mode.manual,
                                          manual_params, multiplicity=8)

        compressed_model, stats = ModelCompressor.compress_model(model, evaluate, 10, input_shape,
                                                                 aimet_common.defs.CompressionScheme.channel_pruning,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params, visualization_url=None)
        baseline_model_accuracy = stats.baseline_model_accuracy
        compressed_best_model_accuracy = stats.compressed_model_accuracy
        self.assertTrue(baseline_model_accuracy >= compressed_best_model_accuracy)
        self.assertEqual(24, compressed_model.layer1[0].conv2.in_channels)

    @pytest.mark.cuda
    def test_channel_pruning_compress_auto_resnet(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        input_shape = (1, 3, 224, 224)
        model = models.resnet18(pretrained=False).to(torch.device('cuda'))
        model.eval()

        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10,
                                                                    use_monotonic_fit=True,
                                                                    saved_eval_scores_dict=
                                                                   './data/resnet18_eval_scores.pkl')
        auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                              modules_to_ignore=modules_to_ignore)

        # selecting single batch for reconstruction
        # num_reconstruction_samples = 50
        # 50 / 10 (samples_per_image) = 5 = batch size

        params = ChannelPruningParameters(data_loader=data_loader.train_loader,
                                          num_reconstruction_samples=50,
                                          allow_custom_downsample_ops=True,
                                          mode=aimet_torch.defs.ChannelPruningParameters.Mode.auto,
                                          params=auto_params, multiplicity=8)

        results = ModelCompressor.compress_model(model=model, eval_callback=evaluate, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.channel_pruning,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)
        self.assertNotEqual(model, compressed_model)
        self.assertTrue(0.6 < float(stats.mac_compression_ratio) < 0.65)

    @pytest.mark.cuda
    def test_channel_pruning_compress_auto_resnet_custom_downsample_ops_not_allowed(self):

        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))
        model.eval()

        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10,
                                                                    saved_eval_scores_dict=
                                                                   './data/resnet18_eval_scores.pkl')
        auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                              modules_to_ignore=modules_to_ignore)

        # selecting single batch for reconstruction
        # num_reconstruction_samples = 50
        # 50 / 10 (samples_per_image) = 5 = batch size

        params = ChannelPruningParameters(data_loader=data_loader.train_loader,
                                          num_reconstruction_samples=50,
                                          allow_custom_downsample_ops=False,
                                          mode=aimet_torch.defs.ChannelPruningParameters.Mode.auto,
                                          params=auto_params)

        results = ModelCompressor.compress_model(model=model, eval_callback=evaluate, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.channel_pruning,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)
        self.assertNotEqual(model, compressed_model)
        self.assertTrue(0.5 < float(stats.mac_compression_ratio) < 0.65)

    @pytest.mark.cuda
    def test_channel_pruning_compress_auto_resnet_with_very_high_multiplicity(self):
        torch.cuda.empty_cache()
        torch.manual_seed(1)
        numpy.random.seed(1)
        torch.backends.cudnn.deterministic = True

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        input_shape = (1, 3, 224, 224)
        model = models.resnet18().to(torch.device('cuda'))
        model.eval()

        modules_to_ignore = [model.conv1,
                             model.layer2[0].downsample[0],
                             model.layer3[0].downsample[0],
                             model.layer4[0].downsample[0],
                             model.layer4[1].conv1,
                             model.layer4[1].conv2
                             ]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10,
                                                                    saved_eval_scores_dict=
                                                                   './data/resnet18_eval_scores.pkl')
        auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                              modules_to_ignore=modules_to_ignore)

        # selecting single batch for reconstruction
        # num_reconstruction_samples = 50
        # 50 / 10 (samples_per_image) = 5 = batch size

        params = ChannelPruningParameters(data_loader=data_loader.train_loader,
                                          num_reconstruction_samples=50,
                                          allow_custom_downsample_ops=True,
                                          mode=aimet_torch.defs.ChannelPruningParameters.Mode.auto,
                                          params=auto_params, multiplicity=64)

        results = ModelCompressor.compress_model(model=model, eval_callback=evaluate, eval_iterations=5,
                                                 input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.channel_pruning,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 visualization_url=None)

        compressed_model, stats = results
        print(compressed_model)
        print(stats)
        self.assertNotEqual(model, compressed_model)
        print("\n\n mac_compression_ratio: %s" % float(stats.mac_compression_ratio))
        self.assertTrue(0.6 < float(stats.mac_compression_ratio) < 0.65)


class FakeEvaluator:

    def __init__(self, input_shape):
        self.index = -1
        self.return_values = []

        self.rand_inputs = aimet_torch.utils.create_rand_tensors_given_shapes(input_shape, torch.device('cpu'))

    @property
    def return_value(self, return_values):
        self.return_values = return_values

    def __call__(self, model, iterations, use_cuda):

        # Just one forward pass, to make sure the forward pass still works
        _ = model(*self.rand_inputs)

        self.index += 1
        return self.return_values[self.index]


def evaluate(model, early_stopping_iterations, use_cuda):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: top_1_accuracy on validation data
    """

    data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
    if early_stopping_iterations is not None:
        # wrapper around validation data loader to run only 'X' iterations to save time
        val_loader = itertools.islice(data_loader.val_loader, early_stopping_iterations)
    else:
        # iterate over entire validation data set
        val_loader = data_loader.val_loader

    criterion = torch.nn.CrossEntropyLoss().cuda()
    evaluator = create_stand_alone_supervised_classification_evaluator(model, criterion, use_cuda=use_cuda)
    evaluator.run(val_loader)
    return evaluator.state.metrics['top_1_accuracy']


class Trainer:
    def __init__(self):
        self._layer_db = []

    def train_model(self, model, layer, train_flag=True):
        """
        Trains a model
        :param model: Model to be trained
        :param layer: layer which has to be fine tuned
        :param train_flag: Default: True. If ture the model gets trained
        :return:
        """
        if train_flag:
            mnist_torch_model.train(model, epochs=1, use_cuda=True, batch_size=50, batch_callback=None)
        self._layer_db.append(layer)
