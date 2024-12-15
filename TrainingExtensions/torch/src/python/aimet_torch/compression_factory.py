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

""" Provides a factory to construct various AIMET model compression classes based on a scheme """

from typing import Tuple, List
import torch

from aimet_common.defs import CostMetric, RankSelectScheme, EvalFunction, LayerCompRatioPair
from aimet_common.cost_calculator import SpatialSvdCostCalculator, WeightSvdCostCalculator
from aimet_common.comp_ratio_select import GreedyCompRatioSelectAlgo, ManualCompRatioSelectAlgo
from aimet_common.comp_ratio_rounder import RankRounder, ChannelRounder
from aimet_common.compression_algo import CompressionAlgo
from aimet_common.bokeh_plots import BokehServerSession

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.defs import SpatialSvdParameters, WeightSvdParameters, ChannelPruningParameters, ModuleCompRatioPair
from aimet_torch.layer_selector import ConvFcLayerSelector, ConvNoDepthwiseLayerSelector, ManualLayerSelector
from aimet_torch.layer_database import LayerDatabase
from aimet_torch.svd.svd_pruner import SpatialSvdPruner, PyWeightSvdPruner
from aimet_torch.channel_pruning.channel_pruner import InputChannelPruner, ChannelPruningCostCalculator

class CompressionFactory:
    """ Factory to construct various AIMET model compression classes based on a scheme """

    @classmethod
    def create_spatial_svd_algo(cls, model: torch.nn.Module, eval_callback: EvalFunction, eval_iterations,
                                input_shape: Tuple, cost_metric: CostMetric,
                                params: SpatialSvdParameters, bokeh_session: BokehServerSession) -> CompressionAlgo:
        """
        Factory method to construct SpatialSvdCompressionAlgo

        :param model: Model to compress
        :param eval_callback: Evaluation callback for the model
        :param eval_iterations: Evaluation iterations
        :param input_shape: Shape of the input tensor for model
        :param cost_metric: Cost metric (mac or memory)
        :param params: Spatial SVD compression parameters
        :param bokeh_session: The Bokeh Session to display plots
        :return: An instance of SpatialSvdCompressionAlgo
        """

        # pylint: disable=too-many-locals
        # Rationale: Factory functions unfortunately need to deal with a lot of parameters

        device = get_device(model)
        dummy_input = create_rand_tensors_given_shapes(input_shape, device)

        # Create a layer database
        layer_db = LayerDatabase(model, dummy_input)
        use_cuda = next(model.parameters()).is_cuda

        # Create a pruner
        pruner = SpatialSvdPruner()
        cost_calculator = SpatialSvdCostCalculator()
        comp_ratio_rounding_algo = RankRounder(params.multiplicity, cost_calculator)

        # Create a comp-ratio selection algorithm
        if params.mode == SpatialSvdParameters.Mode.auto:
            greedy_params = params.mode_params.greedy_params
            comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, cost_calculator, eval_callback,
                                                               eval_iterations, cost_metric,
                                                               greedy_params.target_comp_ratio,
                                                               greedy_params.num_comp_ratio_candidates,
                                                               greedy_params.use_monotonic_fit,
                                                               greedy_params.saved_eval_scores_dict,
                                                               comp_ratio_rounding_algo, use_cuda,
                                                               bokeh_session=bokeh_session)
            layer_selector = ConvNoDepthwiseLayerSelector()
            modules_to_ignore = params.mode_params.modules_to_ignore
        else:
            # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
            layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)

            comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                               layer_comp_ratio_pairs,
                                                               comp_ratio_rounding_algo, cost_metric=cost_metric)

            layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
            modules_to_ignore = []

        # Create the overall Spatial SVD compression algorithm
        spatial_svd_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                           layer_selector, modules_to_ignore, cost_calculator, use_cuda)

        return spatial_svd_algo

    @classmethod
    def create_channel_pruning_algo(cls, model: torch.nn.Module, eval_callback: EvalFunction, eval_iterations,
                                    input_shape: Tuple, cost_metric: CostMetric,
                                    params: ChannelPruningParameters, bokeh_session: BokehServerSession) -> CompressionAlgo:
        """
        Factory method to construct ChannelPruningCompressionAlgo

        :param model: Model to compress
        :param eval_callback: Evaluation callback for the model
        :param eval_iterations: Evaluation iterations
        :param input_shape: Shape of the input tensor for model
        :param cost_metric: Cost metric (mac or memory)
        :param params: Channel Pruning compression parameters
        :param bokeh_session: The Bokeh session to display plots
        :return: An instance of ChannelPruningCompressionAlgo
        """

        # pylint: disable=too-many-locals
        # Rationale: Factory functions unfortunately need to deal with a lot of parameters

        device = get_device(model)
        dummy_input = create_rand_tensors_given_shapes(input_shape, device)

        # Create a layer database
        layer_db = LayerDatabase(model, dummy_input)
        use_cuda = next(model.parameters()).is_cuda

        # Create a pruner
        pruner = InputChannelPruner(data_loader=params.data_loader, input_shape=input_shape,
                                    num_reconstruction_samples=params.num_reconstruction_samples,
                                    allow_custom_downsample_ops=params.allow_custom_downsample_ops)
        comp_ratio_rounding_algo = ChannelRounder(params.multiplicity)

        # Create a comp-ratio selection algorithm
        cost_calculator = ChannelPruningCostCalculator(pruner)

        if params.mode == ChannelPruningParameters.Mode.auto:
            greedy_params = params.mode_params.greedy_params
            comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, cost_calculator, eval_callback,
                                                               eval_iterations, cost_metric,
                                                               greedy_params.target_comp_ratio,
                                                               greedy_params.num_comp_ratio_candidates,
                                                               greedy_params.use_monotonic_fit,
                                                               greedy_params.saved_eval_scores_dict,
                                                               comp_ratio_rounding_algo, use_cuda,
                                                               bokeh_session=bokeh_session)
            layer_selector = ConvNoDepthwiseLayerSelector()
            modules_to_ignore = params.mode_params.modules_to_ignore

        else:
            # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
            layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)

            comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                               layer_comp_ratio_pairs,
                                                               comp_ratio_rounding_algo, cost_metric=cost_metric)

            layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
            modules_to_ignore = []

        # Create the overall Channel Pruning compression algorithm
        channel_pruning_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                               layer_selector, modules_to_ignore, cost_calculator, use_cuda)

        return channel_pruning_algo

    @classmethod
    def create_weight_svd_algo(cls, model: torch.nn.Module, eval_callback: EvalFunction, eval_iterations,
                               input_shape: Tuple, cost_metric: CostMetric,
                               params: WeightSvdParameters, bokeh_session) -> CompressionAlgo:
        """
        Factory method to construct WeightSvdCompressionAlgo

        :param model: Model to compress
        :param eval_callback: Evaluation callback for the model
        :param eval_iterations: Evaluation iterations
        :param input_shape: Shape of the input tensor for model
        :param cost_metric: Cost metric (mac or memory)
        :param params: Weight SVD compression parameters
        :param bokeh_session: The Bokeh session to display plots
        :return: An instance of WeightSvdCompressionAlgo
        """

        # pylint: disable=too-many-locals
        # Rationale: Factory functions unfortunately need to deal with a lot of parameters

        device = get_device(model)
        dummy_input = create_rand_tensors_given_shapes(input_shape, device)

        # Create a layer database
        layer_db = LayerDatabase(model, dummy_input)
        use_cuda = next(model.parameters()).is_cuda

        # Create a pruner
        pruner = PyWeightSvdPruner()
        cost_calculator = WeightSvdCostCalculator()
        comp_ratio_rounding_algo = RankRounder(params.multiplicity, cost_calculator)

        # Create a comp-ratio selection algorithm
        if params.mode == WeightSvdParameters.Mode.auto:
            # greedy
            if params.mode_params.rank_select_scheme is RankSelectScheme.greedy:
                greedy_params = params.mode_params.select_params
                comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db=layer_db,
                                                                   pruner=pruner,
                                                                   cost_calculator=cost_calculator,
                                                                   eval_func=eval_callback,
                                                                   eval_iterations=eval_iterations,
                                                                   cost_metric=cost_metric,
                                                                   target_comp_ratio=greedy_params.target_comp_ratio,
                                                                   num_candidates=greedy_params.num_comp_ratio_candidates,
                                                                   use_monotonic_fit=greedy_params.use_monotonic_fit,
                                                                   saved_eval_scores_dict=greedy_params.saved_eval_scores_dict,
                                                                   comp_ratio_rounding_algo=comp_ratio_rounding_algo,
                                                                   use_cuda=use_cuda,
                                                                   bokeh_session=bokeh_session)
            else:
                raise ValueError("Unknown Rank selection scheme: {}".format(params.AutoModeParams.rank_select_scheme))

            layer_selector = ConvFcLayerSelector()
            modules_to_ignore = params.mode_params.modules_to_ignore

        else:
            # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
            layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)

            comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                               layer_comp_ratio_pairs,
                                                               comp_ratio_rounding_algo, cost_metric=cost_metric)

            layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
            modules_to_ignore = []

        # Create the overall Weight SVD compression algorithm
        weight_svd_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                          layer_selector, modules_to_ignore, cost_calculator, use_cuda)

        return weight_svd_algo

    @staticmethod
    def _get_layer_pairs(layer_db: LayerDatabase, module_comp_ratio_pairs: List[ModuleCompRatioPair]):
        layer_comp_ratio_pairs = []

        for pair in module_comp_ratio_pairs:
            layer_comp_ratio_pair = LayerCompRatioPair(layer_db.find_layer_by_module(pair.module),
                                                       pair.comp_ratio)
            layer_comp_ratio_pairs.append(layer_comp_ratio_pair)

        return layer_comp_ratio_pairs
