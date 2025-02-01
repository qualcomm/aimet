# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Prunes layers using SpatialSvdModuleSplitter SVD scheme """

import copy

import aimet_common.libpymo as pymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import CostMetric
from aimet_common import cost_calculator
import aimet_common.svd_pruner

from aimet_torch import pymo_utils
from aimet_torch.svd.svd_splitter import (
    SpatialSvdModuleSplitter,
    MoWeightSvdModuleSplitter,
    PyWeightSvdModuleSplitter
)
from aimet_torch.layer_database import LayerDatabase, Layer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdPruner(aimet_common.svd_pruner.SpatialSvdPruner):
    """
    Pruner for Spatial-SVD method
    """
    def _perform_svd_and_split_layer(self, layer: Layer, rank: int, comp_layer_db: LayerDatabase):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for spatial svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """
        # Split module using Spatial SVD
        module_a, module_b = SpatialSvdModuleSplitter.split_module(layer.module, rank)

        first_layer_shape = copy.copy(layer.output_shape)

        # Set the second dimension (output_channels) to the rank being used for splitting first layer
        first_layer_shape[1] = rank
        # Special logic for strided layers
        first_layer_shape[3] = first_layer_shape[3] * layer.module.stride[1]

        # Create two new layers and return them
        layer_a = Layer(module_a, layer.name + '.0', first_layer_shape)
        layer_b = Layer(module_b, layer.name + '.1', layer.output_shape)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)


class WeightSvdPruner(aimet_common.svd_pruner.WeightSvdPruner):
    """
    Pruner for Weight-SVD method
    """
    # pylint: disable=no-self-use
    def _perform_svd_and_split_layer(self, layer: Layer, rank: int, cost_metric: CostMetric,
                                     comp_layer_db: LayerDatabase):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for weight svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """
        # For the rounded rank compute the new compression ratio
        comp_ratio = cost_calculator.WeightSvdCostCalculator.calculate_comp_ratio_given_rank(layer, rank,
                                                                                             cost_metric)
        # Create a new instance of libpymo and register layers with it
        svd_lib_ref = pymo.GetSVDInstance()
        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd([layer], cost_metric, svd_lib_ref)

        # Split module using Weight SVD
        logger.info("Splitting module: %s with rank: %r", layer.name, rank)
        module_a, module_b = MoWeightSvdModuleSplitter.split_module(layer.module, rank, name=layer.name,
                                                                    svd_lib_ref=svd_lib_ref)

        layer_a = Layer(module_a, layer.name + '.0', layer.output_shape)
        layer_b = Layer(module_b, layer.name + '.1', layer.output_shape)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)
        return comp_ratio

class PyWeightSvdPruner(aimet_common.svd_pruner.WeightSvdPruner):
    """
    Pruner for Weight-SVD method using numpy.
    """
    # pylint: disable=no-self-use
    def _perform_svd_and_split_layer(self, layer: Layer, rank: int, cost_metric: CostMetric,
                                     comp_layer_db: LayerDatabase):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for weight svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """
        # For the rounded rank compute the new compression ratio
        comp_ratio = cost_calculator.WeightSvdCostCalculator.calculate_comp_ratio_given_rank(layer, rank,
                                                                                             cost_metric)
        # Split module using Weight SVD
        logger.info("Splitting module: %s with rank: %r", layer.name, rank)
        module_a, module_b = PyWeightSvdModuleSplitter.split_module(layer.module, rank)

        layer_a = Layer(module_a, layer.name + '.0', layer.output_shape)
        layer_b = Layer(module_b, layer.name + '.1', layer.output_shape)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)

        return comp_ratio
