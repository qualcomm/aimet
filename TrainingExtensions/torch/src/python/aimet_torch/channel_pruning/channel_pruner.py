# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Prunes layers using Channel Pruning scheme """
from typing import Iterator, List
import copy

import torch
import numpy as np

# Import AIMET specific modules
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.cost_calculator import CostCalculator, Cost
from aimet_common.pruner import Pruner
from aimet_common.channel_pruner import select_channels_to_prune
from aimet_torch.layer_database import LayerDatabase, Layer
from aimet_torch.data_subsampler import DataSubSampler
from aimet_torch.channel_pruning.weight_reconstruction import WeightReconstructor
from aimet_torch import utils
from aimet_torch.winnow.winnow import winnow_model


class InputChannelPruner(Pruner):
    """
    Pruner for Channel Pruning method
    """

    def __init__(self, data_loader: Iterator, input_shape, num_reconstruction_samples: int,
                 allow_custom_downsample_ops: bool):
        """
        Input Channel Pruner with given data_loader, input shape, number of batches and samples per image.

        :param data_loader: data loader
        :param input_shape: input shape
        :param num_reconstruction_samples: number of reconstruction samples
        """
        self._data_loader = data_loader
        self._input_shape = input_shape
        self._num_reconstruction_samples = num_reconstruction_samples
        self._allow_custom_downsample_ops = allow_custom_downsample_ops

    @staticmethod
    def _select_inp_channels(layer: torch.nn.Module, comp_ratio: float) -> list:
        """
        L2 magnitude filter selection

        :param layer: torch.nn.Conv2d
        :param comp_ratio: the ratio of costs after pruning has taken place
                           0 < comp_ratio <= 1.
        :return:
            prune_channels_indices: list of input channels indices to prune.
        """

        assert isinstance(layer, torch.nn.Conv2d)

        # weight data is of shape [Noc, Nic, k_h, k_w]
        weight_data = layer.weight.data

        if weight_data.is_cuda:
            weight_data = weight_data.cpu().detach().numpy()
        else:
            weight_data = np.array(layer.weight.data)

        num_in_channels = layer.in_channels

        prune_indices = select_channels_to_prune(weight_data, comp_ratio, num_in_channels)

        return prune_indices

    def _data_subsample_and_reconstruction(self, orig_layer: torch.nn.Conv2d, pruned_layer: torch.nn.Conv2d,
                                           orig_model: torch.nn.Module, comp_model: torch.nn.Module):
        """
        Collect and sub sampled output data from original layer and input from pruned layer and set
        reconstructed weight and bias to pruned layer in pruned model

        :param orig_layer: layer from original model
        :param pruned_layer: layer from potentially compressed model
        :param orig_model: original model without any compression
        :param comp_model: compressed model
        :return: Nothing
        """
        inp_data, out_data = DataSubSampler.get_sub_sampled_data(orig_layer, pruned_layer, orig_model, comp_model,
                                                                 self._data_loader, self._num_reconstruction_samples)

        WeightReconstructor.reconstruct_params_for_conv2d(pruned_layer, inp_data, out_data)

    def _sort_on_occurrence(self, model: torch.nn.Module, layer_comp_ratio_list: List[LayerCompRatioPair]) -> \
            List[LayerCompRatioPair]:
        """
        Function takes model and list of conv layer-comp ratio to sort, and sorts them based on
        occurrence in the model.

        :param model: model
        :param layer_comp_ratio_list: layer compression ratio list
        :return: sorted_layer_comp_ratio_List
        """

        sorted_layer_comp_ratio_list = []
        input_data = torch.randn(self._input_shape)

        # check if the layer weight is on GPU
        if layer_comp_ratio_list[0].layer.module.weight.is_cuda:
            input_data = input_data.cuda()

        def sorting_hook(module, _inp, _out):
            """
            hook to sort modules based on occurrence
            """
            for pair in layer_comp_ratio_list:
                if pair.layer.module == module:
                    sorted_layer_comp_ratio_list.append(LayerCompRatioPair(pair.layer, pair.comp_ratio))

        handles = []

        for pair in layer_comp_ratio_list:
            handles.append(pair.layer.module.register_forward_hook(sorting_hook))

        # run one forward pass with hooks
        with utils.in_eval_mode(model), torch.no_grad():
            _ = model(input_data)

        # remove hooks
        for handle in handles:
            handle.remove()

        return sorted_layer_comp_ratio_list

    def _winnow_and_reconstruct_layer(self, orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase,
                                      layer: Layer, comp_ratio: float, perform_reconstruction: bool):
        """
        Replaces a given layer within the comp_layer_db with a pruned version of the layer

        :param orig_layer_db: original Layer database
        :param comp_layer_db: Layer database, will be modified
        :param layer: Layer to prune
        :param comp_ratio: compression - ratio
        :return:
        """
        # 1) channel selection
        prune_indices = self._select_inp_channels(layer.module, comp_ratio)

        # 2) winnow - in place API
        _, module_list = winnow_model(comp_layer_db.model, self._input_shape,
                                      [(layer.module, prune_indices)],
                                      reshape=self._allow_custom_downsample_ops,
                                      in_place=True)

        # 3) data sub sampling and reconstruction
        if perform_reconstruction:
            # get original layer reference
            orig_layer = orig_layer_db.find_layer_by_name(layer.name)
            self._data_subsample_and_reconstruction(orig_layer.module, layer.module, orig_layer_db.model,
                                                    comp_layer_db.model)

        # 4) update layer database
        if module_list:
            self._update_layer_database_after_winnowing(comp_layer_db, module_list)

    @staticmethod
    def _update_layer_database_after_winnowing(comp_layer_db, updated_module_list):

        # Get reference to the new module to update in the layer database
        for old_module_name, new_module in updated_module_list:

            try:
                old_layer = comp_layer_db.find_layer_by_name(old_module_name)
            except KeyError:
                # Nothing to update, LayerDatabase was not tracking this layer
                continue

            if isinstance(new_module, torch.nn.Sequential):
                comp_layer_db.update_layer_with_module_in_sequential(old_layer, new_module)

            else:
                # Determine new output shape
                new_output_shape = [old_layer.output_shape[0], new_module.out_channels,
                                    old_layer.output_shape[2], old_layer.output_shape[3]]

                new_layer = Layer(new_module, old_layer.name, new_output_shape)
                comp_layer_db.replace_layer(old_layer, new_layer)

    def _prune_layer(self, orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase,
                     layer: Layer, comp_ratio: float, cost_metric: CostMetric):
        self._winnow_and_reconstruct_layer(orig_layer_db, comp_layer_db, layer, comp_ratio, True)

    def calculate_compressed_cost(self, layer_db: LayerDatabase,
                                  layer_comp_ratio_list: List[LayerCompRatioPair]) -> Cost:
        """
        Calculate cost of a compressed model given a set of layers and corresponding comp-ratios
        :param layer_db: Layer database for original model
        :param layer_comp_ratio_list: List of (layer + comp-ratio) pairs
        :return: Estimated cost of the compressed model
        """
        # Copy the db
        comp_layer_db = copy.deepcopy(layer_db)

        # create a compressed model
        for layer_comp_ratio in layer_comp_ratio_list:
            if layer_comp_ratio.comp_ratio is not None:
                layer = comp_layer_db.find_layer_by_name(layer_comp_ratio.layer.name)
                comp_ratio = layer_comp_ratio.comp_ratio
                if comp_ratio == 1.0:
                    continue
                self._winnow_and_reconstruct_layer(layer_db, comp_layer_db, layer,
                                                   comp_ratio, False)

        # calculate and return the cost of this model
        return CostCalculator.compute_model_cost(comp_layer_db)

    def prune_model(self, layer_db: LayerDatabase, layer_comp_ratio_list: List[LayerCompRatioPair],
                    cost_metric: CostMetric, trainer):

        # sort all the layers in layer_comp_ratio_list based on occurrence
        layer_comp_ratio_list = self._sort_on_occurrence(layer_db.model, layer_comp_ratio_list)
        # call the base class method
        comp_layer_db = Pruner.prune_model(self, layer_db,
                                           layer_comp_ratio_list, cost_metric, trainer)

        return comp_layer_db


class ChannelPruningCostCalculator(CostCalculator):
    """ Cost calculation utilities for Channel Pruning """

    def __init__(self, pruner: InputChannelPruner):
        self._pruner = pruner

    def calculate_compressed_cost(self, layer_db: LayerDatabase, # pylint: disable=arguments-differ
                                  layer_ratio_list: List[LayerCompRatioPair], cost_metric: CostMetric) -> Cost:
        """
        Calculate compressed cost of a model given a list of layer-compression-ratio pairs
        :param layer_db: Layer database for the original model
        :param layer_ratio_list: List of layer, compression-ratio
        :param cost_metric: Cost metric to use for compression (mac or memory)
        :return: Compressed cost
        """

        # Special logic for channel pruning - we first actually prune the model and then determine its cost
        # Because it is not easy to estimate it otherwise
        compressed_cost = self._pruner.calculate_compressed_cost(layer_db, layer_ratio_list)

        return compressed_cost
