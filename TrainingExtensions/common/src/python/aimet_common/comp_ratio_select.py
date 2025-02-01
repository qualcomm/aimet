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

""" Implements different compression-ratio selection algorithms and a common interface to them """

import abc
from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional
import math
import pickle
import statistics
import os

from aimet_common.bokeh_plots import DataTable
from aimet_common.bokeh_plots import LinePlot
from aimet_common.bokeh_plots import ProgressBar
from aimet_common.utils import AimetLogger
from aimet_common.curve_fit import MonotonicIncreasingCurveFit
from aimet_common.defs import CostMetric, LayerCompRatioPair, GreedyCompressionRatioSelectionStats, EvalFunction
from aimet_common.pruner import Pruner
from aimet_common import cost_calculator as cc
from aimet_common.layer_database import Layer, LayerDatabase
from aimet_common.comp_ratio_rounder import CompRatioRounder


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CompRatioSelect)


class CompRatioSelectAlgo(metaclass=abc.ABCMeta):
    """
    Abstract interface for all compression-ratio selection algorithms
    """

    def __init__(self, layer_db: LayerDatabase, cost_calculator: cc.CostCalculator,
                 cost_metric: CostMetric, comp_ratio_rounding_algo: Optional[CompRatioRounder]):
        """
        Constructor
        :param layer_db: Database of compressible layers
        """
        # pylint: disable=too-many-arguments

        self._layer_db = layer_db
        self._cost_calculator = cost_calculator
        self._cost_metric = cost_metric
        self._rounding_algo = comp_ratio_rounding_algo

    @abc.abstractmethod
    def select_per_layer_comp_ratios(self, ) -> Tuple[List[LayerCompRatioPair],
                                                      Any]:
        """
        Runs the compression-ratio algorithm to determine optimal compression ratios for each layer
        :return: List of layer and compression ratio pairs, and stats
        """


class GreedyCompRatioSelectAlgo(CompRatioSelectAlgo):
    """
    Implements the greedy compression-ratio select algorithm
    """

    PICKLE_FILE_EVAL_DICT = './data/greedy_selection_eval_scores_dict.pkl'

    # pylint: disable=too-many-locals
    def __init__(self, layer_db: LayerDatabase, pruner: Pruner, cost_calculator: cc.CostCalculator,
                 eval_func: EvalFunction, eval_iterations, cost_metric: CostMetric, target_comp_ratio: float,
                 num_candidates: int, use_monotonic_fit: bool, saved_eval_scores_dict: Optional[str],
                 comp_ratio_rounding_algo: CompRatioRounder, use_cuda: bool, bokeh_session):

        # pylint: disable=too-many-arguments
        CompRatioSelectAlgo.__init__(self, layer_db, cost_calculator, cost_metric, comp_ratio_rounding_algo)

        self._eval_func = eval_func
        self.bokeh_session = bokeh_session
        self._eval_iter = eval_iterations
        self._is_cuda = use_cuda
        self._pruner = pruner
        self._saved_eval_scores_dict = saved_eval_scores_dict
        self._target_comp_ratio = target_comp_ratio
        self._use_monotonic_fit = use_monotonic_fit

        if saved_eval_scores_dict:
            self._comp_ratio_candidates = 0

        else:
            self._comp_ratio_candidates = []
            for index in range(1, num_candidates):
                self._comp_ratio_candidates.append((Decimal(1) / Decimal(num_candidates)) * index)

    def _pickle_eval_scores_dict(self, eval_scores_dict):

        if not os.path.exists('./data'):
            os.makedirs('./data')

        with open(self.PICKLE_FILE_EVAL_DICT, 'wb') as file:
            pickle.dump(eval_scores_dict, file)

        logger.info("Greedy selection: Saved eval dict to %s", self.PICKLE_FILE_EVAL_DICT)

    @staticmethod
    def _unpickle_eval_scores_dict(saved_eval_scores_dict_path: str):

        with open(saved_eval_scores_dict_path, 'rb') as f:
            eval_dict = pickle.load(f)

        logger.info("Greedy selection: Read eval dict from %s", saved_eval_scores_dict_path)
        return eval_dict

    @staticmethod
    def _calculate_function_value_by_interpolation(comp_ratio: Decimal, layer_eval_score_dict: dict,
                                                   comp_ratio_list: List):
        """
        Calculates eval score for a comp ratio by interpolation
        :param comp_ratio:
        :param layer_eval_score_dict:
        :param comp_ratio_list:
        :return:
        """
        if comp_ratio in comp_ratio_list:
            eval_score = layer_eval_score_dict[comp_ratio]
        else:
            ind = 0
            for ind, c in enumerate(comp_ratio_list, start=0):
                if comp_ratio < c:
                    break

            if ind == len(comp_ratio_list) - 1:
                eval_score = layer_eval_score_dict[comp_ratio_list[-1]]
            else:
                x1 = comp_ratio_list[ind]
                y1 = layer_eval_score_dict[comp_ratio_list[ind]]
                x2 = comp_ratio_list[ind - 1]
                y2 = layer_eval_score_dict[comp_ratio_list[ind - 1]]
                eval_score = (float(comp_ratio) - float(x1)) * (y1 - y2) / (float(x1) - float(x2)) + y1
        return eval_score

    def _update_eval_dict_with_rounding(self, eval_scores_dict, rounding_algo, cost_metric):
        updated_eval_dict = {}

        for layer_name in eval_scores_dict:
            layer_eval_dict = eval_scores_dict[layer_name]
            eval_dict_per_layer = {}

            layer = self._layer_db.find_layer_by_name(layer_name)
            comp_ratio_list = sorted(list(layer_eval_dict.keys()), key=float)
            for comp_ratio in layer_eval_dict:
                rounded_comp_ratio = rounding_algo.round(layer, comp_ratio, cost_metric)

                eval_score = self._calculate_function_value_by_interpolation(rounded_comp_ratio, layer_eval_dict,
                                                                             comp_ratio_list)
                eval_dict_per_layer[Decimal(rounded_comp_ratio)] = eval_score
            updated_eval_dict[layer_name] = eval_dict_per_layer
        return updated_eval_dict

    @staticmethod
    def _fit_eval_dict_to_monotonic_function(eval_scores_dict):

        for layer in eval_scores_dict:
            layer_eval_dict = eval_scores_dict[layer]
            # Convert dict of eval-scores and comp-ratios to lists
            eval_scores = list(layer_eval_dict.values())
            comp_ratios = list(layer_eval_dict.keys())

            eval_scores, polynomial_coefficients = MonotonicIncreasingCurveFit.fit(comp_ratios, eval_scores)
            logger.debug("The coefficients for layer %s are %s", layer, str(polynomial_coefficients))
            # Update the layer_eval_dict
            for index, comp_ratio in enumerate(comp_ratios):
                layer_eval_dict[comp_ratio] = eval_scores[index]

    def _construct_eval_dict(self):
        #  If the user already passed in a previously saved eval scores dict, we just use that
        if self._saved_eval_scores_dict:
            eval_scores_dict = self._unpickle_eval_scores_dict(self._saved_eval_scores_dict)

        else:
            # Create the eval scores dictionary
            eval_scores_dict = self._compute_eval_scores_for_all_comp_ratio_candidates()

            # save the dictionary to file (in case the user wants to reuse the dictionary in the future)
            self._pickle_eval_scores_dict(eval_scores_dict)
        return eval_scores_dict

    def select_per_layer_comp_ratios(self):

        # Compute eval scores for each candidate comp-ratio in each layer
        eval_scores_dict = self._construct_eval_dict()

        # Fit the scores to a monotonically increasing function
        if self._use_monotonic_fit:
            self._fit_eval_dict_to_monotonic_function(eval_scores_dict)

        updated_eval_scores_dict = self._update_eval_dict_with_rounding(eval_scores_dict, self._rounding_algo,
                                                                        self._cost_metric)

        # Get the overall min and max scores
        current_min_score, current_max_score = self._find_min_max_eval_scores(updated_eval_scores_dict)
        exit_threshold = (current_max_score - current_min_score) * 0.0001
        logger.info("Greedy selection: overall_min_score=%f, overall_max_score=%f",
                    current_min_score, current_max_score)

        # Base cost
        original_model_cost = self._cost_calculator.compute_model_cost(self._layer_db)
        logger.info("Greedy selection: Original model cost=%s", original_model_cost)

        while True:

            # Current mid-point score
            current_mid_score = statistics.mean([current_max_score, current_min_score])
            current_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(current_mid_score,
                                                                                       updated_eval_scores_dict,
                                                                                       original_model_cost)

            logger.debug("Greedy selection: current candidate - comp_ratio=%f, score=%f, search-window=[%f:%f]",
                         current_comp_ratio, current_mid_score, current_min_score, current_max_score)

            # Exit condition: is the binary search window too small to continue?
            should_exit, selected_score = self._evaluate_exit_condition(current_min_score, current_max_score,
                                                                        exit_threshold,
                                                                        current_comp_ratio, self._target_comp_ratio)

            if should_exit:
                break

            if current_comp_ratio > self._target_comp_ratio:
                # Not enough compression: Binary search the lower half of the scores
                current_max_score = current_mid_score
            else:
                # Too much compression: Binary search the upper half of the scores
                current_min_score = current_mid_score

        # Search finished, return the selected comp ratios per layer
        # Calculate the compression ratios for each layer based on this score
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(selected_score, updated_eval_scores_dict)
        selected_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(selected_score,
                                                                                    updated_eval_scores_dict,
                                                                                    original_model_cost)

        logger.info("Greedy selection: final choice - comp_ratio=%f, score=%f",
                    selected_comp_ratio, selected_score)

        return layer_ratio_list, GreedyCompressionRatioSelectionStats(updated_eval_scores_dict)

    @staticmethod
    def _evaluate_exit_condition(min_score, max_score, exit_threshold, current_comp_ratio, target_comp_ratio):

        if math.isclose(min_score, max_score, abs_tol=exit_threshold):
            return True, min_score

        if math.isclose(current_comp_ratio, target_comp_ratio, abs_tol=0.001):
            return True, statistics.mean([max_score, min_score])

        return False, None

    def _calculate_model_comp_ratio_for_given_eval_score(self, eval_score, eval_scores_dict,
                                                         original_model_cost):

        # Calculate the compression ratios for each layer based on this score
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(eval_score, eval_scores_dict)
        for layer in self._layer_db:
            if layer not in self._layer_db.get_selected_layers():
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        # Calculate compressed model cost
        compressed_model_cost = self._cost_calculator.calculate_compressed_cost(self._layer_db,
                                                                                layer_ratio_list,
                                                                                self._cost_metric)

        if self._cost_metric == CostMetric.memory:
            current_comp_ratio = Decimal(compressed_model_cost.memory / original_model_cost.memory)
        else:
            current_comp_ratio = Decimal(compressed_model_cost.mac / original_model_cost.mac)

        return current_comp_ratio

    def _find_all_comp_ratios_given_eval_score(self, given_eval_score, eval_scores_dict):
        layer_ratio_list = []
        for layer in self._layer_db.get_selected_layers():
            comp_ratio = self._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                      given_eval_score, layer)
            layer_ratio_list.append(LayerCompRatioPair(layer, comp_ratio))

        return layer_ratio_list

    @staticmethod
    def _find_layer_comp_ratio_given_eval_score(eval_scores_dict: Dict[str, Dict[Decimal, float]],
                                                given_eval_score, layer: Layer):

        # Find the closest comp ratio candidate for the current eval score
        eval_scores_for_layer = eval_scores_dict[layer.name]

        # Sort the eval scores by increasing order of compression
        comp_ratios = list(eval_scores_for_layer.keys())
        sorted_comp_ratios = sorted(comp_ratios, reverse=True)

        # Special cases
        # Case1: Eval score is higher than even our most conservative comp ratio: then no compression
        if given_eval_score > eval_scores_for_layer[sorted_comp_ratios[0]]:
            return None

        if given_eval_score < eval_scores_for_layer[sorted_comp_ratios[-1]]:
            return sorted_comp_ratios[-1]

        # Start with a default of no compression
        selected_comp_ratio = None

        for index, comp_ratio in enumerate(sorted_comp_ratios[1:]):

            if given_eval_score > eval_scores_for_layer[comp_ratio]:
                selected_comp_ratio = sorted_comp_ratios[index]
                break

        return selected_comp_ratio

    @staticmethod
    def _find_min_max_eval_scores(eval_scores_dict: Dict[str, Dict[Decimal, float]]):
        first_layer_scores = list(eval_scores_dict.values())[0]
        first_score = list(first_layer_scores.values())[0]

        min_score = first_score
        max_score = first_score

        for layer_scores in eval_scores_dict.values():
            for eval_score in layer_scores.values():

                if eval_score < min_score:
                    min_score = eval_score

                if eval_score > max_score:
                    max_score = eval_score

        return min_score, max_score

    def _compute_eval_scores_for_all_comp_ratio_candidates(self) -> Dict[str, Dict[Decimal, float]]:
        """
        Creates and returns the eval scores dictionary

        :return: Dictionary of {layer_name: {compression_ratio: eval_score}}  for all selected layers
                 and all compression-ratio candidates
        """

        selected_layers = self._layer_db.get_selected_layers()

        # inputs to initialize a TabularProgress object
        num_candidates = len(self._comp_ratio_candidates)
        num_layers = len(selected_layers)

        if self.bokeh_session:
            column_names = [str(i) for i in self._comp_ratio_candidates]
            layer_names = [i.name for i in selected_layers]

            progress_bar = ProgressBar(total=num_layers * num_candidates, title="Eval Scores Table", color="green",
                                       bokeh_document=self.bokeh_session)
            data_table = DataTable(num_layers, num_candidates, column_names, bokeh_document=self.bokeh_session,
                                   row_index_names=layer_names)
        else:
            data_table = None
            progress_bar = None

        eval_scores_dict = {}
        for layer in selected_layers:

            layer_wise_eval_scores = self._compute_layerwise_eval_score_per_comp_ratio_candidate(data_table,
                                                                                                 progress_bar, layer)
            eval_scores_dict[layer.name] = layer_wise_eval_scores

        return eval_scores_dict

    def _compute_layerwise_eval_score_per_comp_ratio_candidate(self, tabular_progress_object, progress_bar,
                                                               layer: Layer) -> Dict[Decimal, float]:
        """
        Computes eval scores for each compression-ratio candidate for a given layer
        :param layer: Layer for which to calculate eval scores
        :return: Dictionary of {compression_ratio: eval_score} for each compression-ratio candidate
        """

        layer_wise_eval_scores_dict = {}

        # Only publish plots to a document if a bokeh server session exists
        if self.bokeh_session:

            # plot to visualize the evaluation scores as they update for each layer
            layer_wise_eval_scores_plot = LinePlot(x_axis_label="Compression Ratios", y_axis_label="Eval Scores",
                                                   title=layer.name, bokeh_document=self.bokeh_session)
        # Loop over each candidate
        for comp_ratio in self._comp_ratio_candidates:
            logger.info("Analyzing compression ratio: %s =====================>", comp_ratio)

            # Prune layer given this comp ratio
            pruned_layer_db = self._pruner.prune_model(self._layer_db,
                                                       [LayerCompRatioPair(layer, comp_ratio)],
                                                       self._cost_metric,
                                                       trainer=None)

            eval_score = self._eval_func(pruned_layer_db.model, self._eval_iter, use_cuda=self._is_cuda)
            layer_wise_eval_scores_dict[comp_ratio] = eval_score

            # destroy the layer database
            pruned_layer_db.destroy()
            pruned_layer_db = None

            logger.info("Layer %s, comp_ratio %f ==> eval_score=%f", layer.name, comp_ratio,
                        eval_score)

            if self.bokeh_session:
                layer_wise_eval_scores_plot.update(new_x_coordinate=comp_ratio, new_y_coordinate=eval_score)
                # Update the data table by adding the computed eval score
                tabular_progress_object.update_table(str(comp_ratio), layer.name, eval_score)
                # Update the progress bar
                progress_bar.update()

        # remove plot so that we have a fresh figure to visualize for the next layer.
        if self.bokeh_session:
            layer_wise_eval_scores_plot.remove_plot()

        return layer_wise_eval_scores_dict


class ManualCompRatioSelectAlgo(CompRatioSelectAlgo):
    """
    Implements the manual compression-ratio select algorithm. Just reflects back the user-selected
    layer and comp-ratio pairs.
    """

    def __init__(self, layer_db: LayerDatabase, layer_comp_ratio_pairs: List[LayerCompRatioPair],
                 comp_ratio_rounding_algo: CompRatioRounder, cost_metric: CostMetric):
        CompRatioSelectAlgo.__init__(self, layer_db, cost_calculator=None, cost_metric=cost_metric,
                                     comp_ratio_rounding_algo=None)

        self._layer_comp_ratio_pairs = layer_comp_ratio_pairs
        self._rounding_algo = comp_ratio_rounding_algo

    def select_per_layer_comp_ratios(self):
        for pair in self._layer_comp_ratio_pairs:
            rounded_comp_ratio = self._rounding_algo.round(pair.layer, pair.comp_ratio, self._cost_metric)
            pair.comp_ratio = rounded_comp_ratio

        return self._layer_comp_ratio_pairs, None
