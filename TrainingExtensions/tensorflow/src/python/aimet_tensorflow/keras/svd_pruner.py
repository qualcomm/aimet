# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Prunes layers using Spatial SVD schemes """
from typing import Tuple

import tensorflow as tf
from aimet_common import libpymo as pymo

from aimet_common.utils import AimetLogger
from aimet_common import cost_calculator
from aimet_common.defs import CostMetric
import aimet_common.svd_pruner
from aimet_common.pruner import Pruner

from aimet_tensorflow.keras.utils import pymo_utils
from aimet_tensorflow.keras.layer_database import LayerDatabase, Layer
from aimet_tensorflow.keras.svd_spiltter import SpatialSvdModuleSplitter, WeightSvdModuleSplitter

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
        module_a, module_b = SpatialSvdModuleSplitter.split_module(comp_layer_db.model, layer, rank)

        # get the output activation shape for first conv op
        output_shape_a = _get_layer_output_shape(module_a)

        # get the output activation shape for second conv op
        output_shape_b = _get_layer_output_shape(module_b)

        # Create two new layers and return them
        layer_a = Layer(layer=module_a, name=module_a.name, output_shape=output_shape_a)
        layer_b = Layer(layer=module_b, name=module_b.name, output_shape=output_shape_b)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)


class WeightSvdPruner(Pruner):
    """
    Pruner for Weight-SVD method
    """

    def _prune_layer(self, orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase, layer: Layer, comp_ratio: float,
                     cost_metric: CostMetric) -> float:
        """
        Replaces a given layer within the comp_layer_db with a pruned version of the layer
        In this case, the replaced layer will be a sequential of two spatial-svd-decomposed layers

        :param orig_layer_db: original Layer database
        :param comp_layer_db: Layer database, which will get modified
        :param layer: Layer to prune
        :param comp_ratio: Compression-ratio
        :param cost_metric: Cost metric to used for compression
        :return: updated compression ratio
        """

        # Given a compression ratio, find the appropriate rounded rank
        rank = cost_calculator.WeightSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, cost_metric)

        logger.info("Weight SVD splitting layer: %s using rank: %s", layer.name, rank)

        # For the rounded rank compute the new compression ratio
        comp_ratio = cost_calculator.WeightSvdCostCalculator.calculate_comp_ratio_given_rank(layer, rank,
                                                                                             cost_metric)

        # Create a new instance of libpymo and register layers with it
        svd_lib_ref = pymo.GetSVDInstance()
        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd([layer], cost_metric, svd_lib_ref, pymo.TYPE_SINGLE)

        # Split module using Weight SVD
        logger.info("Splitting module: %s with rank: %r", layer.name, rank)
        module_a, module_b = WeightSvdModuleSplitter.split_module(comp_layer_db.model, layer.module, rank, svd_lib_ref)

        # get the output activation shape for first conv/fc op
        output_shape_a = _get_layer_output_shape(module_a)

        # get the output activation shape for second conv/fc op
        output_shape_b = _get_layer_output_shape(module_b)

        # Create two new layers and return them
        layer_a = Layer(layer=module_a, name=module_a.name, output_shape=output_shape_a)
        layer_b = Layer(layer=module_b, name=module_b.name, output_shape=output_shape_b)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)
        return comp_ratio


def _get_layer_output_shape(layer: tf.keras.layers) -> Tuple:
    """
    Returns the output shape of the layer as required by the Layer class

    :param layer: Keras Layer for which output shape needs to be retrieved.
    :return: Tuple containing the output shape
    """
    output_activation_shape = list(layer.output_shape)
    if len(output_activation_shape) == 4:
        reorder = [0, 3, 1, 2]
        output_activation_shape = [output_activation_shape[idx] for idx in reorder]
    # activation dimension for FC layer is (1,1)
    if isinstance(layer, tf.keras.layers.Dense):
        output_activation_shape.extend([1, 1])

    return tuple(output_activation_shape)
