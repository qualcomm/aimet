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

"""Network and per layer cost calculator"""
from decimal import Decimal
from functools import reduce
from typing import List, Tuple

from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.layer_database import Layer, Conv2dTypeSpecificParams, LayerDatabase
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class Cost:
    """
    Models cost of a layer or a collection of layers
    """

    def __init__(self, mem_cost: int, mac_cost: int):
        self.memory = mem_cost
        self.mac = mac_cost

    def __str__(self):
        return '(Cost: memory={}, mac={})'.format(self.memory, self.mac)

    def __add__(self, another_cost):
        return Cost(self.memory + another_cost.memory,
                    self.mac + another_cost.mac)

    def __sub__(self, another_cost):
        return Cost(self.memory - another_cost.memory,
                    self.mac - another_cost.mac)


class CostCalculator:
    """
    Utility for calculating per layer cost and network cost
    """
    @classmethod
    def get_compressed_model_cost(cls, layer_db, layer_ratio_list, original_model_cost, cost_metric):
        """
        computes compressed model cost metric with all layers included
        :param layer: layer data base
        :param layer: layer ratio list
        :param layer: original model cost
        :param layer: cost metric
        :return: comp ratio for compressed model
        """

        # Add the layers that were not selected to this list to get the accurate cost of the compressed model
        for layer in layer_db:
            if layer not in layer_db.get_selected_layers():
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        # Calculate compressed model cost
        compressed_model_cost = cls.calculate_compressed_cost(layer_db,
                                                              layer_ratio_list,
                                                              cost_metric)

        if cost_metric == CostMetric.memory:
            current_comp_ratio = Decimal(compressed_model_cost.memory / original_model_cost.memory)
        else:
            current_comp_ratio = Decimal(compressed_model_cost.mac / original_model_cost.mac)

        return current_comp_ratio

    @staticmethod
    def compute_layer_cost(layer: Layer):
        """
        Computes per layer cost
        :param layer: Attributes for a layer
        :return: Cost of the layer
        """
        weight_dim = list(layer.weight_shape)
        additional_act_dim = [layer.output_shape[-1], layer.output_shape[-2]]

        mem_cost = reduce(lambda x, y: x*y, weight_dim)
        mac_dim = weight_dim + additional_act_dim
        mac_cost = reduce(lambda x, y: x*y, mac_dim)

        return Cost(mem_cost, mac_cost)

    @classmethod
    def compute_network_cost(cls, layers):
        """
        Function to get the total cost of the model in terms of Memory and MAC metric
        :return: total cost (Memory), total cost (MAC)
        """
        network_cost = Cost(0, 0)

        for layer in layers.values():

            cost = cls.compute_layer_cost(layer)
            network_cost += cost

        return network_cost

    @classmethod
    def compute_model_cost(cls, layer_db: LayerDatabase):
        """
        Function to get the total cost of the model in terms of Memory and MAC metric
        :return: total cost (Memory), total cost (MAC)
        """
        network_cost_memory = 0
        network_cost_mac = 0
        for layer in layer_db:

            cost = cls.compute_layer_cost(layer)

            network_cost_memory += cost.memory
            network_cost_mac += cost.mac

        return Cost(network_cost_memory, network_cost_mac)

    @classmethod
    def calculate_comp_ratio_given_rank(cls, layer: Layer, rank: int, cost_metric: CostMetric):
        """
        Finds compression ratio for the rounded rank
        :param layer:
        :param rank:
        :param cost_metric: Cost metric (mac or memory)
        :return:
        """

        original_cost = CostCalculator.compute_layer_cost(layer)
        if cost_metric == CostMetric.memory:
            compressed_cost = cls.calculate_cost_given_rank(layer, rank).memory
            updated_comp_ratio = Decimal(compressed_cost)/Decimal(original_cost.memory)
        else:
            compressed_cost = cls.calculate_cost_given_rank(layer, rank).mac
            updated_comp_ratio = Decimal(compressed_cost)/Decimal(original_cost.mac)
        return updated_comp_ratio

    @classmethod
    def calculate_rank_given_comp_ratio(cls, layer: Layer, comp_ratio: float, cost_metric: CostMetric) -> int:
        """
        Calculates rank to be used for spatial svd splitting to achieve a given compression-ratio
        Note since both mac and memory counts are scaled versions of each other, cost-metric is not a concern
        :param layer: Layer reference
        :param comp_ratio: Compression-ratio
        :param cost_metric: Cost metric (mac or memory)
        :return: Rank
        """

        orig_cost = CostCalculator.compute_layer_cost(layer)
        if cost_metric == CostMetric.mac:
            target_cost = orig_cost.mac * comp_ratio
        else:
            target_cost = orig_cost.memory * comp_ratio

        # Invoke via use of strategy pattern
        current_rank_candidate = cls.calculate_max_rank(layer)

        if cost_metric == CostMetric.memory:
            running_cost = cls.calculate_cost_given_rank(layer, current_rank_candidate).memory
        else:
            running_cost = cls.calculate_cost_given_rank(layer, current_rank_candidate).mac

        while (running_cost > target_cost) and (current_rank_candidate > 0):

            current_rank_candidate -= 1

            # Invoke via use of strategy pattern
            cost = cls.calculate_cost_given_rank(layer, current_rank_candidate)

            if cost_metric == CostMetric.memory:
                running_cost = cost.memory
            else:
                running_cost = cost.mac

        if current_rank_candidate <= 0:
            current_rank_candidate = 1

        return current_rank_candidate

    @classmethod
    def calculate_per_layer_compressed_cost(cls, layer: Layer, comp_ratio: float, cost_metric: CostMetric) -> Cost:
        """
        Calculate compressed cost for a layer given a compression ratio
        :param layer: Layer
        :param comp_ratio: Compression ratio (between 0 and 1)
        :param cost_metric: Cost metric (mac or memory)
        :return: Compressed cost
        """

        # Invoke using the strategy pattern
        rank = cls.calculate_rank_given_comp_ratio(layer, comp_ratio, cost_metric)
        cost = cls.calculate_cost_given_rank(layer, rank)

        return cost

    @classmethod
    def calculate_compressed_cost(cls, _layer_db: LayerDatabase,
                                  layer_ratio_list: List[LayerCompRatioPair], cost_metric: CostMetric) -> Cost:
        """
        Calculate compressed cost of a model given a list of layer-compression-ratio pairs
        :param _layer_db: Layer database for the original model
        :param layer_ratio_list: List of layer, compression-ratio
        :param cost_metric: Cost metric to use for compression (mac or memory)
        :return: Compressed cost
        """

        running_cost = Cost(0, 0)
        for layer_comp_ratio_pair in layer_ratio_list:
            if layer_comp_ratio_pair.comp_ratio is not None:
                cost = cls.calculate_per_layer_compressed_cost(layer_comp_ratio_pair.layer,
                                                               layer_comp_ratio_pair.comp_ratio, cost_metric)
            else:
                cost = cls.compute_layer_cost(layer_comp_ratio_pair.layer)

            running_cost += cost

        return running_cost

    @classmethod
    def calculate_compressed_cost_given_ranks(cls, _layer_db: LayerDatabase,
                                              layer_rank_list: List[Tuple[Layer, int]]) -> Cost:
        """
        Calculate compressed cost of a model given a list of layer and rank pairs
        :param _layer_db: Layer database for the original model
        :param layer_rank_list: List of layer, corresponding rank
        :return: Compressed cost
        """

        running_cost = Cost(0, 0)
        for layer, rank in layer_rank_list:
            if rank:
                cost = cls.calculate_cost_given_rank(layer, rank)
            else:
                cost = cls.compute_layer_cost(layer)

            running_cost += cost

        return running_cost

    @staticmethod
    def calculate_cost_given_rank(layer: Layer, rank: int) -> Cost:
        """
        Give a rank for splitting a given layer, calculate the compressed cost
        :param layer: Layer
        :param rank: Rank to split the layer with
        :return: Compressed cost of the layer after splitting
        """

    @staticmethod
    def calculate_max_rank(layer: Layer) -> int:
        """
        Given a layer, calculate the max rank (only applies for SVD-based decomposition schemes)
        :param layer: Layer
        :return: Maximum rank
        """


class SpatialSvdCostCalculator(CostCalculator):
    """ Cost calculation utilities for Spatial SVD """

    @staticmethod
    def calculate_cost_given_rank(layer: Layer, rank: int) -> Cost:

        m = layer.weight_shape[1]
        n = layer.weight_shape[0]

        if isinstance(layer.type_specific_params, Conv2dTypeSpecificParams):
            kh = layer.weight_shape[2]
            kw = layer.weight_shape[3]

            # (m, n, kh, kw) is split into (m, rank, kh, 1) and (rank, n, 1, kw)
            mem_cost = (m * rank * kh) + (rank * n * kw)
            output_dim_cost = layer.output_shape[2] * layer.output_shape[3]
            mac_cost = (m * rank * kh * layer.type_specific_params.stride[1] +
                        rank * n * kw) * output_dim_cost

        else:
            mem_cost = m * rank + rank * n
            mac_cost = (m * rank + rank * n) * layer.weight_shape[2] * layer.weight_shape[3]

        return Cost(mem_cost, mac_cost)

    @staticmethod
    def calculate_max_rank(layer: Layer):

        if isinstance(layer.type_specific_params, Conv2dTypeSpecificParams):
            # min(Nic * Kh, Noc * Kw)
            max_rank = min(layer.weight_shape[1] * layer.weight_shape[2], layer.weight_shape[0] * layer.weight_shape[3])

        else:
            max_rank = min(layer.weight_shape[1], layer.weight_shape[0])

        return max_rank


class WeightSvdCostCalculator(CostCalculator):
    """ Cost calculation utilities for Weight SVD """

    @staticmethod
    def calculate_cost_given_rank(layer: Layer, rank: int) -> Cost:

        if isinstance(layer.type_specific_params, Conv2dTypeSpecificParams):
            m = layer.weight_shape[1]
            n = layer.weight_shape[0]
            kernel_size = layer.weight_shape[2] * layer.weight_shape[3]
            stride_factor = layer.type_specific_params.stride[0] * layer.type_specific_params.stride[1]

        else:
            m = layer.weight_shape[1]
            n = layer.weight_shape[0]
            kernel_size = 1
            stride_factor = 1

        # (m, n, kh, kw) is split into (m, rank, 1, 1) and (rank, n, kh, kw)
        # stride only applied to the 1st layer. And stride has no effect on memory cost
        mem_cost = (m * rank) + (rank * n * kernel_size)
        mac_cost = (m * rank * stride_factor + rank * n * kernel_size) * layer.output_shape[2] * layer.output_shape[3]

        return Cost(mem_cost, mac_cost)

    @staticmethod
    def calculate_max_rank(layer: Layer):

        max_rank = layer.weight_shape[1]

        return max_rank
