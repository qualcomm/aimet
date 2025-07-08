# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Common Algorithm between PyTorch & TensorFlow for Auto Mixed precision"""

import os
import copy
import io
import abc
import json
import time
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Tuple, List, Dict, Union
import pickle
import functools
import math

from aimet_common.defs import QuantizationDataType
from aimet_common.utils import AimetLogger
from aimet_common.amp.quantizer_groups import QuantizerGroupBase
from aimet_common.amp.utils import (
    AMPSearchAlgo,
    CandAttr,
    CandParam,
    calculate_starting_bit_ops,
    sort_accuracy_list,
    CANDIDATE_WITH_DTYPE,
    ACCURACY_LIST,
    disable_quantizers,
    enable_quantizers,
    create_quant_group_to_candidate_dict,
    modify_candidate_in_accuracy_list,
    export_list,
    binary_search,
    interpolation_search,
    brute_force_search,
)


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


class GreedyMixedPrecisionAlgoParams:
    """Bundle parameters needed for GreedyMixedPrecisionAlgo together for reducing amount of function parameters"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        candidates: List[CANDIDATE_WITH_DTYPE],
        forward_pass_callback: Callable[[Any], Any],
        eval_callback_for_phase1: Callable[[Any], float],
        eval_callback_for_phase2: Callable[[Any], float],
    ):
        """
        :param candidates: List of Tuples of all possible bitwidth values to quantize to (excluding max bitwidth)
        :param forward_pass_callback: Callable object used to compute quantization encodings
        :param eval_callback_for_phase1: Callable object used to measure sensitivity of each
                                     quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                     module. Therefore, a user might want to run the phase 1 with a smaller dataset
        :param eval_callback_for_phase2: Callale object used to get accuracy of quantized model
                                     for phase 2 calculations. The phase 2 involves finding pareto front curve
        """
        self.candidates = candidates
        self.forward_pass_callback = forward_pass_callback.func
        self.forward_pass_callback_args = forward_pass_callback.args
        self.eval_callback_for_phase1 = eval_callback_for_phase1
        self.eval_callback_for_phase2 = eval_callback_for_phase2


class GreedyMixedPrecisionAlgo(abc.ABC):  # pylint: disable=too-many-instance-attributes
    """Base class for Naive Greedy MixedPrecisionAlgo"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        sim,
        candidates: List[CANDIDATE_WITH_DTYPE],
        eval_callback_for_phase1: Callable[[Any], float],
        eval_callback_for_phase2: Callable[[Any], float],
        forward_pass_callback: Callable[[Any], Any],
        mac_dict: Dict,
        results_dir: str,
        clean_start: bool,
        phase2_reverse: bool = False,
    ):
        """
        :param sim: Quantized sim model
        :param candidates: List of Tuple of all possible bitwidth values to quantize to
        :param eval_callback_for_phase1: Callable object used to measure sensitivity of each
                                     quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                     module. Therefore, a user might want to run the phase 1 with a smaller dataset
        :param eval_callback_for_phase2: Callale object used to get accuracy of quantized model
                                     for phase 2 calculations. The phase 2 involves finding pareto front curve
        :param forward_pass_callback: Callable object used to compute quantization encodings
        :param mac_dict: Dictionary mapping modules to mac counts
        :param results_dir: Path to save results and cache intermediate results
        :param clean_start: If true, any cached information from previous runs will be deleted prior to starting the
                            mixed-precision analysis. If false, prior cached information will be used if applicable. Note
                            it is the user's responsibility to set this flag to true if anything in the model or
                            quantization parameters changes compared to the previous run.
        :param phase2_reverse: If user will set this parameter to True, then phase1 of amp algo, that is calculating accuracy list will not be changed,
                            whereas the phase2 algo of amp, which generate the pareto list will be changed. In phase2, algo will start, model with all quantizer groups in least candidate, and
                            one by one, it will put nodes in higher candidate till target accuracy does not meet.
        """
        self._validate_inputs(candidates)

        self._sim = sim
        self.phase2_reverse = phase2_reverse
        self.time_taken_phase1 = None
        self.time_taken_phase2 = None
        self._results_dir = results_dir
        self._clean_start = clean_start
        self._module_name_dict, self.quantizer_groups = self._find_quantizer_group(sim)

        self.algo_params = GreedyMixedPrecisionAlgoParams(
            candidates,
            forward_pass_callback,
            eval_callback_for_phase1,
            eval_callback_for_phase2,
        )
        # Get the mac dict
        self._mac_dict = mac_dict

        self.fp32_accuracy = None
        self.baseline_candidate = None
        self.accuracy_list = None
        self.pareto_list = None
        self.min_accuracy = None
        self.min_candidate = None
        # Populate final eval score within pareto list
        self._final_eval_score = None

        # dict of lists to hold the supported candidates for all the quantizers
        self._supported_candidates_per_quantizer_group = defaultdict(list)

        # List of eval scores and corresponding candidates.
        self._eval_scores = []

        # Dict to store candidate for each quantizer group
        self._candidate_mapping_dict = {}

    @abc.abstractmethod
    def _find_quantizer_group(self, sim) -> Tuple[Dict, List[QuantizerGroupBase]]:
        """
        Finds quantizer groups in a quantization sim

        :param sim: Quantization sim
        :return: Dictionary mapping quantized op name to sim.quantizer_config,
            and a List of quantizer groups
        """

    @property
    @abc.abstractmethod
    def baseline_candidate_options(self) -> List[CANDIDATE_WITH_DTYPE]:
        """
        Returns the _baseline_candidate_options which is the intersection of amp candidates and candidates supported by
        all the quantizer groups
        """

    def __str__(self):
        if self.baseline_candidate is None:
            raise RuntimeError("Baseline bitwidth is not set. Call run() first.")

        (
            percentage_act_quantizers_flipped,
            percentage_param_quantizers_flipped,
            percentage_quantizers_flipped,
            act_quantizers_flipped,
            param_quantizers_flipped,
            total_act_quantizers,
            total_param_quantizers,
            total_quantizers,
        ) = self._count_and_get_quantizers_flipped()

        stream = io.StringIO(newline="\n")

        stream.write(
            "\n**********************************************************************************************\n"
        )
        stream.write("Mixed Precision Statistics\n")
        stream.write(f"Mixed precision model accuracy {self._final_eval_score}\n")
        stream.write("\n")
        stream.write(
            "**********************************************************************************************\n"
        )

        max_candidate = self.baseline_candidate
        stream.write(
            f"\nActivation Quantizers flipped from baseline candidate - {max_candidate[CandAttr.activation]}\n"
        )

        for row in act_quantizers_flipped:
            stream.write(f"{row}\n")

        stream.write(
            f"Percentage of Activation Quantizers flipped from baseline  {percentage_act_quantizers_flipped} \n"
        )

        stream.write(
            "**********************************************************************************************\n"
        )

        stream.write(
            f"\nParameter Quantizers flipped from baseline candidate: - {max_candidate[CandAttr.parameter]}\n"
        )

        for row in param_quantizers_flipped:
            stream.write(f"{row}\n")

        stream.write(
            f"Percentage of Parameter Quantizers flipped from baseline  {percentage_param_quantizers_flipped} \n"
        )

        stream.write(
            "**********************************************************************************************\n"
        )

        stream.write(
            f"Percentage of all Quantizers flipped from baseline  {percentage_quantizers_flipped} \n"
        )

        stream.write(
            "**********************************************************************************************\n"
        )

        stream.write(
            f"Total Number of activation quantizers are  {total_act_quantizers} \n"
        )

        stream.write(
            "**********************************************************************************************\n"
        )

        stream.write(
            f"Total Number of Param quantizers are  {total_param_quantizers} \n"
        )

        stream.write(
            "**********************************************************************************************\n"
        )

        stream.write(f"Total Number of quantizers are  {total_quantizers} \n")

        stream.write(
            "**********************************************************************************************\n"
        )

        return stream.getvalue()

    def _count_and_get_quantizers_flipped(self):
        # pylint: disable=too-many-locals
        (
            total_act_quantizers,
            total_param_quantizers,
            count_act_quantizers_flipped,
            count_param_quantizers_flipped,
        ) = 0, 0, 0, 0
        act_quantizers_flipped, param_quantizers_flipped = [], []
        for quantizer_group in self.quantizer_groups:
            candidate = quantizer_group.get_candidate(self._module_name_dict)
            max_candidate = self.baseline_candidate
            quantizer_group_list = quantizer_group.to_list()
            for type_of_op, module_name in quantizer_group_list:
                if type_of_op in ("input", "output", "activation"):
                    total_act_quantizers += 1
                    if (
                        candidate[CandAttr.activation][CandParam.bitwdith]
                        != max_candidate[CandAttr.activation][CandParam.bitwdith]
                        or candidate[CandAttr.activation][CandParam.data_type]
                        != max_candidate[CandAttr.activation][CandParam.data_type]
                    ):
                        count_act_quantizers_flipped += 1
                        act_quantizers_flipped.append(
                            (
                                type_of_op,
                                module_name,
                                (
                                    candidate[CandAttr.activation][CandParam.bitwdith],
                                    candidate[CandAttr.activation][CandParam.data_type],
                                ),
                            )
                        )
                else:
                    total_param_quantizers += 1
                    if (
                        candidate[CandAttr.parameter][CandParam.bitwdith]
                        != max_candidate[CandAttr.parameter][CandParam.bitwdith]
                        or candidate[CandAttr.parameter][CandParam.data_type]
                        != max_candidate[CandAttr.parameter][CandParam.data_type]
                    ):
                        count_param_quantizers_flipped += 1
                        param_quantizers_flipped.append(
                            (
                                type_of_op,
                                module_name,
                                (
                                    candidate[CandAttr.parameter][CandParam.bitwdith],
                                    candidate[CandAttr.parameter][CandParam.data_type],
                                ),
                            )
                        )

        percentage_act_quantizers_flipped = (
            (count_act_quantizers_flipped * 100) / total_act_quantizers
            if total_act_quantizers
            else 0
        )
        percentage_param_quantizers_flipped = (
            (count_param_quantizers_flipped * 100) / total_param_quantizers
            if total_param_quantizers
            else 0
        )
        percentage_quantizers_flipped = (
            (count_param_quantizers_flipped + count_act_quantizers_flipped) * 100
        ) / (total_param_quantizers + total_act_quantizers)
        total_quantizers = total_param_quantizers + total_act_quantizers
        return (
            percentage_act_quantizers_flipped,
            percentage_param_quantizers_flipped,
            percentage_quantizers_flipped,
            act_quantizers_flipped,
            param_quantizers_flipped,
            total_act_quantizers,
            total_param_quantizers,
            total_quantizers,
        )

    @staticmethod
    def _export_accuracy_list(
        accuracy_list: ACCURACY_LIST, results_dir: str, file_name: str = "accuracy_list"
    ):
        """
        Exports accuracy list as a json file where dotted name for each quantizer group is saved

        :param accuracy_list: List of Tuple of accuracy and bitwidths for each quantizer group
        :param results_dir: Path to save accuracy list
        :param: file_name: File name of the accuracy list.
        """

        results_dir = os.path.join(results_dir, ".cache")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## Json dump will not work with accuracy_list as it has objects inside it, so converting the objects to list or to string.
        acc_list_with_quant_groups_as_list = []
        for quantizer_group, candidate, eval_score, bit_ops_reduction in accuracy_list:
            acc_list_with_quant_groups_as_list.append(
                (
                    quantizer_group.to_list(),
                    str(candidate),
                    eval_score,
                    bit_ops_reduction,
                )
            )

        file_path = os.path.join(results_dir, file_name + ".pkl")
        file_path_json = os.path.join(results_dir, file_name + ".json")

        with open(file_path, "wb") as file:
            pickle.dump(accuracy_list, file)
        # Dumping in json format as it is readible without mapping to any class.
        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump(acc_list_with_quant_groups_as_list, f, indent=1)

    @staticmethod
    def find_max_eval(
        eval_results: List[Tuple[float, CANDIDATE_WITH_DTYPE]],
    ) -> Tuple[float, CANDIDATE_WITH_DTYPE]:
        """
        Find the candidate corresponding to max accuracy

        :eval_results: List of tuples consisting of accuracy scores and corresponding candidates
        :return: The tuple with highest accuracy
        """
        max_idx = 0
        max_acc = eval_results[0][0]
        for idx, (acc, _) in enumerate(eval_results):
            if acc > max_acc:
                max_acc = acc
                max_idx = idx
        return max_acc, eval_results[max_idx][1]

    def _calc_baseline_fp32_accuracy(self):
        """Calculates baseline FP32 accuracy"""
        disabled_quantizers = OrderedDict()
        # Disable all quantizers
        for quantizer_group in self.quantizer_groups:
            quantizers = quantizer_group.get_active_quantizers(self._module_name_dict)
            disable_quantizers(quantizers)
            disabled_quantizers[quantizer_group] = quantizers

        fp32_accuracy = self.evaluate_model(self.algo_params.eval_callback_for_phase2)

        # Enable the disabled quantizers
        for quantizers in disabled_quantizers.values():
            enable_quantizers(quantizers)

        logger.info("Baseline FP32 accuracy: %f", fp32_accuracy)
        return fp32_accuracy

    def _get_best_candidate(self):
        """Gets best candidate from list of provided candidates"""
        if not self._eval_scores:
            forward_pass_callback = self.algo_params.forward_pass_callback
            forward_pass_callback_args = self.algo_params.forward_pass_callback_args
            eval_callback = self.algo_params.eval_callback_for_phase2

            logger.info("Computing accuracy for maximum candidate")

            for candidate in self.baseline_candidate_options:
                candidate_map_dict = {}
                for quantizer_group in self.quantizer_groups:
                    valid_candidate = candidate
                    if (
                        len(
                            self._supported_candidates_per_quantizer_group[
                                quantizer_group
                            ][0]
                        )
                        == 2
                        and valid_candidate
                        not in self._supported_candidates_per_quantizer_group[
                            quantizer_group
                        ]
                    ):
                        # quantizer_group supports both activation & params bw and candidate is not valid for the given
                        # quantizer group then consider the first valid candidate from supported candidate dict
                        valid_candidate = (
                            self._supported_candidates_per_quantizer_group[
                                quantizer_group
                            ][0]
                        )
                    elif (
                        len(
                            self._supported_candidates_per_quantizer_group[
                                quantizer_group
                            ][0]
                        )
                        == 1
                    ):
                        # quantizer_group supports only activation bw and candidate is not valid for the given
                        # quantizer group then consider the first valid candidate from supported candidate dict
                        valid_candidate = (valid_candidate[0],)
                        if (
                            valid_candidate
                            not in self._supported_candidates_per_quantizer_group[
                                quantizer_group
                            ]
                        ):
                            valid_candidate = (
                                self._supported_candidates_per_quantizer_group[
                                    quantizer_group
                                ][0]
                            )
                    quantizer_group.set_quantizers_to_candidate(
                        self._module_name_dict, valid_candidate
                    )
                    candidate_map_dict[quantizer_group] = valid_candidate
                # Recompute encodings
                self._sim.compute_encodings(
                    forward_pass_callback, forward_pass_callback_args
                )
                # Compute accuracy of model with new bitwidth
                eval_score = self.evaluate_model(eval_callback)
                logger.info(
                    "QuantSim accuracy with candidate %s : %f", candidate, eval_score
                )
                self._eval_scores.append((eval_score, candidate))
                self._candidate_mapping_dict[candidate] = copy.deepcopy(
                    candidate_map_dict
                )

        return self.find_max_eval(self._eval_scores)

    def _set_all_quantizer_groups_to_candidate(self, candidate: CANDIDATE_WITH_DTYPE):
        """
        Sets all quantizer groups to bitwidth of baseline

        :param candidate: Bitwidth to set the quantizer groups to
        """

        for quantizer_group in self.quantizer_groups:
            valid_candidate = candidate
            if (
                candidate in self._candidate_mapping_dict
                and quantizer_group in self._candidate_mapping_dict[candidate]
            ):
                valid_candidate = self._candidate_mapping_dict[candidate][
                    quantizer_group
                ]
            else:
                logger.warning(
                    "Either %s or %s not found in candidate mapping dict. Setting %s as valid candidate",
                    str(candidate),
                    str(quantizer_group),
                    str(candidate),
                )
            quantizer_group.set_quantizers_to_candidate(
                self._module_name_dict, valid_candidate
            )

        self._sim.compute_encodings(
            self.algo_params.forward_pass_callback,
            self.algo_params.forward_pass_callback_args,
        )

    def evaluate_model(self, eval_callback: Callable[[Any], float]) -> float:
        """
        Evaluates a model and assert that the eval score is non-negative.

        :param eval_callback: Callback function that contains eval function and eval args.
        :return: Eval score.
        :raises:
          - RuntimeError if eval score is neither float nor convertible to float.
        """
        eval_score = self._evaluate_model(eval_callback)

        try:
            eval_score = float(eval_score)
        except TypeError as e:
            msg = (
                "eval_callback is expected to return float "
                "or a value that can be cast to float, "
                "but returned {}.".format(type(eval_score))
            )
            raise RuntimeError(msg) from e

        return eval_score

    @abc.abstractmethod
    def _evaluate_model(self, eval_callback: Callable[[Any], float]) -> float:
        """
        Evaluates a model

        :param eval_callback: Callback function that contains eval function and eval args
        :return: Eval score
        """

    @abc.abstractmethod
    def _find_bit_ops_reduction_for_acc_list(
        self,
        quantizer_group,
        max_candidate: CANDIDATE_WITH_DTYPE,
        candidate: CANDIDATE_WITH_DTYPE,
    ) -> int:
        """
        Finds reduction in bit ops from max candidate to new candidate

        :param quantizer_group: Quantizer group
        :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
        :param candidate: Activation bitwidth, parameter bitwidth
        :return: Bit ops reduction
        """

    @abc.abstractmethod
    def calculate_running_bit_ops(
        self,
        quantizer_group,
        module_bitwidth_dict: Dict,
        max_candidate: CANDIDATE_WITH_DTYPE,
        candidate: CANDIDATE_WITH_DTYPE,
        running_bit_ops: int,
    ) -> int:
        """
        Calculates running bit ops value for every quantizer group

        :param quantizer_group: A group of activation & parameter quantizers
        :param module_bitwidth_dict: Dict; Key: Module name value: Activation, parameter bitwidth of module
        :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
        :param candidate: Bitwidth to change the quantizer group to
        :param running_bit_ops: Running bit ops value calculated uptil the quantizer group
        :return: Running bit ops value
        """

    def set_baseline(
        self,
        fp32_accuracy: float = None,
        baseline_candidate: CANDIDATE_WITH_DTYPE = None,
    ) -> Tuple[float, CANDIDATE_WITH_DTYPE]:
        """
        Set baseline accuracy and baseline bitwidth.

        :param fp32_accuracy: FP32 accuracy
        :param baseline_candidate: The candidate [bitwidth, datatype] with max accuracy
        :return: The new baseline accuracy and baseline bitwidth.
        """

        if fp32_accuracy is None:
            fp32_accuracy = self._calc_baseline_fp32_accuracy()

        if baseline_candidate is None:
            _, baseline_candidate = self._get_best_candidate()

        self.fp32_accuracy = fp32_accuracy
        self.baseline_candidate = baseline_candidate

        # If _candidate_mapping_dict is empty, consider baseline_candidate is valid for all quantizer groups
        if not self._candidate_mapping_dict:
            self._candidate_mapping_dict = {
                baseline_candidate: {
                    quantizer_group: baseline_candidate
                    for quantizer_group in self.quantizer_groups
                }
            }
            logger.warning(
                "candidate mapping is found to be empty. Setting %s for all quantizer groups as valid",
                str(baseline_candidate),
            )

        # Set all quantizers to baseline's bitwidth
        self._set_all_quantizer_groups_to_candidate(self.baseline_candidate)

        return self.fp32_accuracy, self.baseline_candidate

    def _export_amp_execution_info(self):
        (
            percentage_act_quantizers_flipped,
            percentage_param_quantizers_flipped,
            percentage_quantizers_flipped,
            act_quantizers_flipped,
            param_quantizers_flipped,
            total_act_quantizers,
            total_param_quantizers,
            total_quantizers,
        ) = self._count_and_get_quantizers_flipped()

        execution_info = [
            ("fp32 accuray", self.fp32_accuracy),
            ("Phase1 Time", self.time_taken_phase1),
            ("Phase2 Time", self.time_taken_phase2),
            ("final score", self._final_eval_score),
            ("phase2 eval count", len(self.pareto_list)),
            ("percentage_act_quantizers_flipped", percentage_act_quantizers_flipped),
            (
                "percentage_param_quantizers_flipped",
                percentage_param_quantizers_flipped,
            ),
            ("percentage_quantizers_flipped", percentage_quantizers_flipped),
            ("count_act_quantizers_flipped", len(act_quantizers_flipped)),
            ("count_param_quantizers_flipped", len(param_quantizers_flipped)),
            ("total_act_quantizers", total_act_quantizers),
            ("total_param_quantizers", total_param_quantizers),
            ("total_quantizers", total_quantizers),
        ]

        export_list(execution_info, self._results_dir, "amp_info_list")

    def run(
        self,
        allowed_accuracy_drop: Union[None, float],
        search_algo: AMPSearchAlgo = AMPSearchAlgo.Binary,
    ):
        """
        Run mixed precision main algorithm.

        :param allowed_accuracy_drop: Maximum allowed drop in accuracy from baseline. If None then complete pareto
            curve will be constructed
        :param search_algo: TODO
        """
        if search_algo == AMPSearchAlgo.BruteForce:
            search_algo_fn = brute_force_search
        elif search_algo == AMPSearchAlgo.Interpolation:
            search_algo_fn = interpolation_search
        elif search_algo == AMPSearchAlgo.Binary:
            search_algo_fn = binary_search
        else:
            raise ValueError(
                f"Invalid search algo. Expected AMPSearchAlgo object, but got {search_algo}"
            )

        if self.fp32_accuracy is None or self.baseline_candidate is None:
            self.set_baseline()

        if self.min_accuracy is None or self.min_candidate is None:
            self.min_accuracy, self.min_candidate = (
                self._choose_lowest_from_candidates()
            )

        if allowed_accuracy_drop is None:
            allowed_accuracy_drop = math.inf
        else:
            # if allowed_accuracy_drop is not None:
            max_candidate_accuracy, _ = self._get_best_candidate()
            if self.fp32_accuracy - max_candidate_accuracy > allowed_accuracy_drop:
                # Running AMP cannot produce any meaningful result as the accuracy drop seen with
                # the best candidate is larger than the allowed drop in accuracy
                logger.info(
                    "The difference between baseline (%0.4f) and candidate with best accuracy (%0.4f) is"
                    " higher than the allowed accuracy drop (%0.4f)",
                    self.fp32_accuracy,
                    max_candidate_accuracy,
                    allowed_accuracy_drop,
                )
                # Set all quantizers to highest bitwidth candidate and compute encodings.
                self._set_all_quantizer_groups_to_candidate(self.baseline_candidate)
                # Compute final eval score.
                self._final_eval_score = max_candidate_accuracy
                return

            if self.fp32_accuracy - self.min_accuracy < allowed_accuracy_drop:
                # Set all quantizers to lowest bitwidth candidate and compute encodings.
                self._set_all_quantizer_groups_to_candidate(self.min_candidate)
                # Compute final eval score.
                self._final_eval_score = self.evaluate_model(
                    self.algo_params.eval_callback_for_phase2
                )
                logger.info(
                    "The difference between baseline (%0.4f) and candidate with the least accuracy (%0.4f) "
                    "is less than allowed accuracy drop (%0.4f). Early-exiting mixed precision algorithm. "
                    "No need to further create quantizer group sensitivity list and pareto front list. ",
                    self.fp32_accuracy,
                    self.min_accuracy,
                    allowed_accuracy_drop,
                )
                return

        # Perform phase 1
        # Create sorted accuracy list (phase 1)
        start_phase1 = time.time()
        self.accuracy_list = self._create_and_save_accuracy_list(
            self.baseline_candidate
        )
        end_phase1 = time.time()

        # calculating time taken by phase1 execution
        self.time_taken_phase1 = (end_phase1 - start_phase1) / 60
        logger.info("Time taken by phase1 is (%0.4f) mins", self.time_taken_phase1)
        # Phase 2
        start_phase2 = time.time()
        # Create pareto front list and quantize model as much as possible while still meeting accuracy condition
        self.pareto_list = self._create_pareto_front_list(
            allowed_accuracy_drop,
            self.accuracy_list,
            self.fp32_accuracy,
            self.baseline_candidate,
            self.min_candidate,
            search_algo_fn,
            self.phase2_reverse,
        )
        end_phase2 = time.time()

        # calculating time taken by phase2 execution
        self.time_taken_phase2 = (end_phase2 - start_phase2) / 60
        logger.info("Time taken by phase2 is (%0.4f) mins", self.time_taken_phase2)

        # Run Phase 3 if supported and enabled to reduce the number of converts in the graph
        self._reduce_mp_convert_ops()

        self._export_amp_execution_info()

    def _create_and_save_accuracy_list(
        self, baseline_candidate: CANDIDATE_WITH_DTYPE
    ) -> ACCURACY_LIST:
        """
        Create a list of tuples of (quantizer_group, bitwidth, accuracy score)

        :param baseline_candidate: Candidate [bitwidth, dtype] which yields max accuracy
        :return: Sorted accuracy list containing tuples of (quantizer, candidate, accuracy score, bit ops reduction)
        """

        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        index_of_quantizer_group = {}
        for index, quantizer_group in enumerate(self.quantizer_groups):
            index_of_quantizer_group[quantizer_group] = index

        accuracy_list: ACCURACY_LIST = []

        file = os.path.join(self._results_dir, ".cache", "accuracy_list.pkl")
        combinations_already_computed = set()

        if os.path.isfile(file):
            if self._clean_start:
                os.remove(file)
                logger.info("Removed old cached files and restarting computation")
            else:
                with open(file, "rb") as f:
                    accuracy_list = pickle.load(f)

                combinations_already_computed.update(
                    (quantizer_group, candidate)
                    for quantizer_group, candidate, _, _ in accuracy_list
                )

        disabled_quantizers = OrderedDict()
        try:
            # Disable all quantizers
            for quantizer_group in self.quantizer_groups:
                quantizers = quantizer_group.get_active_quantizers(
                    self._module_name_dict
                )
                disable_quantizers(quantizers)
                disabled_quantizers[quantizer_group] = quantizers

            # Loop through all possible bitwidths and all quantizers.  Set each quantizer in turn to the lower bitwidth,
            # calculate resulting model accuracy, and reset the quantizer back to default bitwidth.
            # Accuracy list will contain tuples of the quantizer, bitwidth, and accuracy score
            for (
                quantizer_group,
                candidates,
            ) in self._supported_candidates_per_quantizer_group.items():
                quantizers = disabled_quantizers[quantizer_group]
                try:
                    enable_quantizers(
                        quantizers
                    )  # Temporarily enable quantizers in the current quantizer group

                    for candidate in candidates:
                        if candidate == baseline_candidate:
                            continue

                        # If starting the computation from an already existing state, then check if that combination
                        # has already been executed
                        if (
                            quantizer_group,
                            candidate,
                        ) in combinations_already_computed:
                            continue

                        # Set quantizer bitwidth to lower candidate (bitwidth)
                        quantizer_group.set_quantizers_to_candidate(
                            self._module_name_dict, candidate
                        )

                        # Recompute encodings for new candidate (bitwidth)
                        self._sim.compute_encodings(
                            self.algo_params.forward_pass_callback,
                            self.algo_params.forward_pass_callback_args,
                        )
                        # Compute accuracy of model with new candidate (bitwidth)
                        eval_score = self.evaluate_model(
                            self.algo_params.eval_callback_for_phase1
                        )

                        bit_ops_reduction = self._find_bit_ops_reduction_for_acc_list(
                            quantizer_group, baseline_candidate, candidate
                        )
                        accuracy_list.append(
                            (quantizer_group, candidate, eval_score, bit_ops_reduction)
                        )
                        # Sort accuracy list, first by descending accuracy score, then by descending order of addition of bitwidths if accuracy
                        # scores are identical, if that is also identical we sort by relative bit ops change in descending order
                        # If bit ops reduction is also the same, then we sort in ascending order based on occurence of
                        # quantizer group in the model
                        accuracy_list = sort_accuracy_list(
                            accuracy_list, index_of_quantizer_group
                        )
                        self._export_accuracy_list(accuracy_list, self._results_dir)
                        logger.info(
                            "\n Quantizer: %s candidate: %s eval_score: %f \n",
                            quantizer_group,
                            candidate,
                            eval_score,
                        )
                finally:
                    # Getting candidate which is valid for the quantizer group for a given baseline_candidate
                    valid_baseline_candidate = baseline_candidate
                    if (
                        baseline_candidate in self._candidate_mapping_dict
                        and quantizer_group
                        in self._candidate_mapping_dict[baseline_candidate]
                    ):
                        valid_baseline_candidate = self._candidate_mapping_dict[
                            baseline_candidate
                        ][quantizer_group]
                    else:
                        logger.warning(
                            "Either %s or %s not found in candidate mapping dict. Setting %s as valid "
                            "baseline candidate",
                            str(baseline_candidate),
                            str(quantizer_group),
                            str(baseline_candidate),
                        )

                    # Reset bitwidth back to default
                    quantizer_group.set_quantizers_to_candidate(
                        self._module_name_dict, valid_baseline_candidate
                    )
                    disable_quantizers(quantizers)
        finally:
            # Enable the disabled quantizers
            for quantizers in disabled_quantizers.values():
                enable_quantizers(quantizers)

        logger.info("Completed Accuracy list computation")

        # Recompute encodings after last quantizer's bitwidth is set back to self._max_bitwidth
        self._sim.compute_encodings(
            self.algo_params.forward_pass_callback,
            self.algo_params.forward_pass_callback_args,
        )

        return accuracy_list

    @staticmethod
    def _export_pareto_list(
        results_dir: str, pareto_front: List, file_name: str = "pareto_list"
    ):
        """
        Exports pareto list as pickle file under cache folder and as a json file

        :param results_dir: Path in which pareto list is stored
        :param pareto_front: Pareto front list
        :param: file_name: File name of the pareto list.
        """
        pareto_path_json = os.path.join(results_dir, "pareto_list.json")
        results_dir = os.path.join(results_dir, ".cache")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file_path_pickle = os.path.join(results_dir, file_name + ".pkl")
        file_path_json = os.path.join(results_dir, file_name + ".json")

        pareto_list_with_quant_groups_as_list = []
        for bitops, acc, quant_group, candidate in pareto_front:
            pareto_list_with_quant_groups_as_list.append(
                (bitops, acc, quant_group.to_list(), str(candidate))
            )

        with open(file_path_pickle, "wb") as f:
            pickle.dump(pareto_front, f)

        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump(pareto_list_with_quant_groups_as_list, f, indent=1)

        with open(pareto_path_json, "w", encoding="utf-8") as f:
            json.dump(pareto_list_with_quant_groups_as_list, f, indent=1)

    def _create_pareto_front_list(
        self,
        allowed_accuracy_drop: float,
        accuracy_list: ACCURACY_LIST,
        fp32_accuracy: float,
        baseline_candidate: CANDIDATE_WITH_DTYPE,
        lowest_candidate: CANDIDATE_WITH_DTYPE,
        search_algo: Callable,
        phase2_reverse: bool,
    ) -> List:
        """
        Create pareto front list comprised of tuples of bitops count, accuracy score, next quantizer that was
        quantized, and the new bitwidth used.  Entries in this list are quantized cumulatively, meaning the bitops
        count and accuracy score for one entry is calculated by quantizing all previous quantizers to specified
        bitwidths, along with the current quantizer.

        :param allowed_accuracy_drop: Maximum allowed drop in accuracy from baseline
            (using max candidate bitwidth for all quantizers)
        :param accuracy_list: Sensitivity list per quantizer group
        :param fp32_accuracy: Accuracy corresponding to maximum bitwidth candidate
        :param baseline_candidate: The maximum [bitwidth, dtype] candidate
        :param lowest_candidate: The lowest [bitwidth, dtype] candidate
        :param phase2_reverse: If user will set this parameter to True, then phase1 of amp algo, that is calculating accuracy
                            list will not be changed, whereas the phase2 algo of amp, which generate the pareto list will be
                            changed. In phase2, algo will start, model with all quantizer groups in least candidate, and
                            one by one, it will put nodes in higher candidate till target accuracy does not meet.
        :return: Pareto front list containing the following information:
                 - Relative bit ops wrt baseline candidate
                 - eval score
                 - quantizer group
                 - candidate
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        if phase2_reverse:
            # starting candidate would be the lowest candidate
            starting_candidate = lowest_candidate
            # As we will change bitwidths from lowest precision to the higher precision to achieve target
            # this implementation need the order of the list from the high to low sensitivity, hence accuracy list
            # from the phase1 need to be reversed.

            accuracy_list.reverse()
            self._export_accuracy_list(
                accuracy_list, self._results_dir, "accuracy_list_reverse"
            )
            # quant_group_to_candidate_dict will be having the order of candidates as occured in accuracy_list_reverse for every quantizer group
            quant_group_to_candidate_dict = create_quant_group_to_candidate_dict(
                accuracy_list
            )
            # candidate in accuracy_list_reverse give information about the sensitivity of the quantizer group when turning
            # the quantizer group into that particular candidate, modify_candidate_in_accuracy_list function will replace
            # the candidate by the target candidate, that the quantizer group should be converted to, which is the next higher candidate
            accuracy_list = modify_candidate_in_accuracy_list(
                accuracy_list, quant_group_to_candidate_dict, baseline_candidate
            )
            # Exporting Modified accuray list, upon which phase2 search algo will run.
            self._export_accuracy_list(
                accuracy_list, self._results_dir, "accuracy_list_reverse_modified"
            )
        else:
            # starting candidate would be the maximum candidate
            starting_candidate = baseline_candidate

        self._set_all_quantizer_groups_to_candidate(starting_candidate)
        pareto_front = [None for _ in accuracy_list]
        file_path = os.path.join(self._results_dir, ".cache", "pareto_list.pkl")

        if os.path.isfile(file_path):
            if self._clean_start:
                os.remove(file_path)
                logger.info(
                    "Removed old pareto list pickled file and restarting computation"
                )
            else:
                with open(file_path, "rb") as f:
                    _pareto_front = pickle.load(f)
                    for (
                        relative_bit_ops,
                        eval_score,
                        quantizer_group,
                        candidate,
                    ) in _pareto_front:
                        for i, (_quantizer_group, _candidate, _, _) in enumerate(
                            accuracy_list
                        ):
                            if (
                                _quantizer_group == quantizer_group
                                and _candidate == candidate
                            ):
                                pareto_front[i] = (
                                    relative_bit_ops,
                                    eval_score,
                                    quantizer_group,
                                    candidate,
                                )

        # Running bit ops will store the bit ops for each iteration in the loop below.  In each iteration, a new
        # quantizer is affected.  To avoid recomputing bitops for every module again, only subtract bitops for the
        # module that the quantizer corresponds to for the quantizer's previous bitwidth, and add bitops for the module
        # using the quantizer's new bitwidth.
        starting_bit_ops = calculate_starting_bit_ops(
            self._mac_dict, starting_candidate
        )
        # Module bitwidth dict maps modules to a tuple of two bitwidths, corresponding to the most recently used
        # bitwidths for the module's input quantizer and weight quantizer.

        def evaluate_pareto_point(i: int) -> float:
            """Evaluate i-th point of pareto curve"""
            if pareto_front[i] is not None:
                _, eval_score, _, _ = pareto_front[i]
                return eval_score

            module_bitwidth_dict = {}
            bit_ops = starting_bit_ops
            for quantizer_group, candidate, _, _ in accuracy_list[: i + 1]:
                self._set_quantizer_group_to_candidate(quantizer_group, candidate)
                bit_ops = self.calculate_running_bit_ops(
                    quantizer_group,
                    module_bitwidth_dict,
                    starting_candidate,
                    candidate,
                    bit_ops,
                )
            # Find bit ops value relative to starting bit ops
            relative_bit_ops = bit_ops / starting_bit_ops

            # optimize the mixed precision profile if enabled and op graph is available. Once optimized, this method
            # computes the eval score, sets the model back in the unoptimized state
            eval_score = self._optimize_mp_profile_and_evaluate_model()

            quantizer_group, candidate, _, _ = accuracy_list[i]

            pareto_front[i] = (relative_bit_ops, eval_score, quantizer_group, candidate)

            logger.info(
                "\n Quantizer group: %s candidate: %s eval_score: %f Relative BitOps %f \n",
                quantizer_group,
                str(
                    (
                        (
                            candidate[CandAttr.activation][CandParam.bitwdith],
                            candidate[CandAttr.activation][CandParam.data_type],
                        ),
                        (
                            candidate[CandAttr.parameter][CandParam.bitwdith]
                            if len(candidate) == 2
                            else None,
                            candidate[CandAttr.parameter][CandParam.data_type]
                            if len(candidate) == 2
                            else None,
                        ),
                    )
                ),
                eval_score,
                relative_bit_ops,
            )

            self._export_pareto_list(
                self._results_dir, [x for x in pareto_front if x is not None]
            )

            for quantizer_group, candidate, _, _ in accuracy_list[: i + 1]:
                self._set_quantizer_group_to_candidate(
                    quantizer_group, starting_candidate
                )

            return eval_score

        values = [
            functools.partial(evaluate_pareto_point, i)
            for i, _ in enumerate(accuracy_list)
        ]
        target_accuracy = fp32_accuracy - allowed_accuracy_drop

        i = search_algo(values, target_accuracy, self.phase2_reverse)

        if i == 0 and pareto_front[i][1] < target_accuracy and not phase2_reverse:
            # If no entry in pareto front meets the target eval score, fall back to baseline candidate
            pass
        elif (
            (i == len(values) - 1)
            and (pareto_front[len(values) - 1][1] < target_accuracy)
            and phase2_reverse
        ):
            # If no entry in pareto front meets the target eval score, fall back to baseline candidate
            self._set_all_quantizer_groups_to_candidate(baseline_candidate)
        else:
            for quantizer_group, candidate, _, _ in accuracy_list[: i + 1]:
                self._set_quantizer_group_to_candidate(quantizer_group, candidate)

        logger.info("Completed Pareto list computation")

        # optimize the mixed precision profile if op graph is available
        self._final_eval_score = self._optimize_mp_profile_and_evaluate_model()
        logger.info("AMP phase-2 final accuracy:%f", self._final_eval_score)

        return [x for x in pareto_front if x is not None]

    def _set_quantizer_group_to_candidate(
        self, quantizer_group: QuantizerGroupBase, candidate: CANDIDATE_WITH_DTYPE
    ):
        quantizer_group.set_quantizers_to_candidate(self._module_name_dict, candidate)

    @staticmethod
    def _validate_inputs(candidates: List[CANDIDATE_WITH_DTYPE]):
        for candidate in candidates:
            (act_bw, act_data_type), (param_bw, param_data_type) = candidate
            assert isinstance(act_bw, int), (
                "candidate's activation bitwidth is expected to be of type int"
            )
            assert isinstance(act_data_type, QuantizationDataType), (
                "candidate's activation data type is expected to be of type QuantizationDataType"
            )
            assert isinstance(param_bw, int), (
                "candidate's param bitwidth is expected to be of type int"
            )
            assert isinstance(param_data_type, QuantizationDataType), (
                "candidate's param data type is expected to be of type QuantizationDataType"
            )

    def _choose_lowest_from_candidates(self) -> Tuple[float, CANDIDATE_WITH_DTYPE]:
        """
        Choose the lowest bitwidth candidate among all the candidates.
        If there is tie, then candidate with the higher accuracy is returned.

        :return: Lowest bitwidth candidate and corresponding accuracy.
        """
        # Set the first candidate as minimum.
        min_accuracy, min_candidate = self._eval_scores[0]
        min_bit_ops = calculate_starting_bit_ops(self._mac_dict, min_candidate)

        for index, (_, candidate) in enumerate(self._eval_scores):
            bit_ops = calculate_starting_bit_ops(self._mac_dict, candidate)
            if bit_ops < min_bit_ops:
                min_bit_ops = bit_ops
                min_accuracy, min_candidate = self._eval_scores[index]
            elif bit_ops == min_bit_ops:
                accuracy, candidate = self._eval_scores[index]
                if accuracy > min_accuracy:
                    min_bit_ops = bit_ops
                    min_accuracy, min_candidate = self._eval_scores[index]

        return min_accuracy, min_candidate

    @abc.abstractmethod
    def _create_op_graph(self, sim):
        """
        Creates op graph

        :param sim: QuantizationSimModel object
        """

    @abc.abstractmethod
    def _optimize_mp_profile_and_evaluate_model(self):
        """
        Uses OpGraph if available to optimize the mixed precision profile in the sim object
        """

    @abc.abstractmethod
    def _reduce_mp_convert_ops(self):
        """
        Reduce mixed precision convert ops if enabled and supported
        """
