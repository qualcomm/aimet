# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=missing-docstring
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import os
from decimal import Decimal
import torch


# Compression-related imports
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters, RankSelectScheme
from aimet_torch.defs import WeightSvdParameters, SpatialSvdParameters, ChannelPruningParameters, \
    ModuleCompRatioPair
from aimet_torch.compress import ModelCompressor

# Quantization related import
from aimet_torch.v1.quantsim import QuantizationSimModel

# Both compression and quantization related imports


def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """
    return .5


class Trainer:
    """ Example trainer class """

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


def spatial_svd_manual_mode():

    # Load a trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    manual_params = SpatialSvdParameters.ManualModeParams([ModuleCompRatioPair(model.conv1, 0.5),
                                                           ModuleCompRatioPair(model.conv2, 0.4)])
    params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.manual,
                                  params=manual_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.spatial_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)    # Stats object can be pretty-printed easily


def spatial_svd_auto_mode():

    # load trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10)
    auto_params = SpatialSvdParameters.AutoModeParams(greedy_params,
                                                      modules_to_ignore=[model.conv1])

    params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                  params=auto_params, multiplicity=8)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.spatial_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


def spatial_svd_auto_mode_with_layerwise_finetuning():

    # load trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10)
    auto_params = SpatialSvdParameters.AutoModeParams(greedy_params,
                                                      modules_to_ignore=[model.conv1])

    params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                  params=auto_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.spatial_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params, trainer=Trainer())

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


def weight_svd_manual_mode():

    # Load a trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    manual_params = WeightSvdParameters.ManualModeParams([ModuleCompRatioPair(model.conv1, 0.5),
                                                          ModuleCompRatioPair(model.conv2, 0.4)])
    params = WeightSvdParameters(mode=WeightSvdParameters.Mode.manual,
                                 params=manual_params, multiplicity=8)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.weight_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)    # Stats object can be pretty-printed easily


def weight_svd_auto_mode():

    # Load trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10)
    rank_select = RankSelectScheme.greedy
    auto_params = WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                     select_params=greedy_params,
                                                     modules_to_ignore=[model.conv1])

    params = WeightSvdParameters(mode=WeightSvdParameters.Mode.auto,
                                 params=auto_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.weight_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


def channel_pruning_auto_mode():

    # Load trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10)
    auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                          modules_to_ignore=[model.conv1])

    data_loader = mnist_torch_model.DataLoaderMnist(cuda=True, seed=1, shuffle=True)
    params = ChannelPruningParameters(data_loader=data_loader.train_loader,
                                      num_reconstruction_samples=500,
                                      allow_custom_downsample_ops=True,
                                      mode=ChannelPruningParameters.Mode.auto,
                                      params=auto_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.channel_pruning,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


def channel_pruning_manual_mode():

    # Load a trained MNIST model
    model = torch.load(os.path.join('../', 'data', 'mnist_trained_on_GPU.pth'))

    # Specify the necessary parameters
    manual_params = ChannelPruningParameters.ManualModeParams([ModuleCompRatioPair(model.conv2, 0.4)])

    data_loader = mnist_torch_model.DataLoaderMnist(cuda=True, seed=1, shuffle=True)
    params = ChannelPruningParameters(data_loader=data_loader.train_loader,
                                      num_reconstruction_samples=500,
                                      allow_custom_downsample_ops=True,
                                      mode=ChannelPruningParameters.Mode.manual,
                                      params=manual_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.channel_pruning,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)    # Stats object can be pretty-printed easily


if __name__ == '__main__':
    spatial_svd_manual_mode()
    spatial_svd_auto_mode()

    weight_svd_manual_mode()
    weight_svd_auto_mode()

    channel_pruning_manual_mode()
    channel_pruning_auto_mode()

