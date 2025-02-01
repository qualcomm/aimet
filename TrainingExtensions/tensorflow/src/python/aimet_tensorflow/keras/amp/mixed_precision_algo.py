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

""" Evaluator class for mixed precision """
import functools
import os
from collections import defaultdict, OrderedDict
import pickle
from typing import Tuple, List, Dict, Callable, Union, Any
import contextlib

import numpy
import tensorflow as tf

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.defs import CallbackFunc
from aimet_common.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo as MixedPrecisionAlgo
from aimet_common.amp.utils import (
    sort_accuracy_list,
    CANDIDATE_WITH_DTYPE,
    ACCURACY_LIST,
    disable_quantizers,
    enable_quantizers,
)
from aimet_common.amp.convert_ops_reduction import SamplingStrategy

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.amp import utils as mixed_precision_utils
from aimet_tensorflow.keras.amp.quantizer_groups import QuantizerGroup, find_quantizer_group
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_tensorflow.keras.amp.convert_ops_reduction import ReduceConvertOps

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


class EvalCallbackFactory:
    """
    Factory class for various built-in eval callbacks
    """

    def __init__(self,
                 data_loader_wrapper: Any,
                 forward_fn: Callable[[tf.keras.Model, Any], numpy.ndarray] = None):
        """
        :param data_loader_wrapper: Callable function which gives Data loader to be used for evaluation
        :param forward_fn: Function that runs forward pass and returns the output tensor.
                           i.e. f: (model, input yielded from data loader) -> output tensor
        """
        self._data_loader_wrapper = data_loader_wrapper
        self._forward_fn = forward_fn or _default_forward_fn
        # storing batchwise fp32 outputs in the list
        self._batchwise_fp32_outputs_list = []

    def _forward_fn_wrapper(self, *args, **kwargs):
        output = self._forward_fn(*args, **kwargs)
        if not isinstance(output, (numpy.ndarray, tf.Tensor)):
            raise RuntimeError(
                "Forward pass was expected to return a tf.Tensor or numpy.ndarray, "
                f"but returned an object of type {type(output)}. "
                "Try specifying `forward_fn` to adapt the output."
            )
        return output

    _DEFAULT_SQNR_NUM_SAMPLES = 128

    def sqnr(self, original_model: tf.keras.Model = None, num_samples: int = _DEFAULT_SQNR_NUM_SAMPLES) -> CallbackFunc:
        """
        Returns SQNR eval callback.
        :param original_model: Original Keras Model against whose output SQNR will be calculated.
                In case of None quantized model will be used after disabling the quantizers
                as a reference to compute SQNR.
        :param num_samples: Number of samples used for evaluation
        :return: SQNR eval callback
        """
        evaluate_sqnr = functools.partial(_evaluate_sqnr,
                                          original_model=original_model,
                                          data_loader_wrapper=self._data_loader_wrapper,
                                          forward_fn=self._forward_fn_wrapper,
                                          num_samples=num_samples,
                                          batchwise_fp32_outputs_list=self._batchwise_fp32_outputs_list)
        return CallbackFunc(evaluate_sqnr)


def _default_forward_fn(model, inputs):
    return model(inputs)


@contextlib.contextmanager
def disable_all_quantizers(quantized_model: tf.keras.Model):
    """
    Helper function to disable the quantizers temporarily.
    :param quantized_model: Quantized model whose quantizers needs to be disabled.
    :return: Intermediately it provides an quantizer disabled version of the provided model.
    """
    active_quantizers = []

    for layer in quantized_model.layers:
        if not isinstance(layer, QcQuantizeWrapper):
            continue

        for quantizer in layer.param_quantizers + \
                         layer.input_quantizers + \
                         layer.output_quantizers:
            if quantizer.is_enabled():
                active_quantizers.append(quantizer)

    try:
        for quantizer in active_quantizers:
            quantizer.disable()
        yield
    finally:
        for quantizer in active_quantizers:
            quantizer.enable()


def _evaluate_sqnr(quantized_model: tf.keras.Model,
                   _: Any,
                   original_model: tf.keras.Model,
                   data_loader_wrapper: Any,
                   forward_fn: Callable[[tf.keras.Model, Any], Union[tf.Tensor, numpy.ndarray]],
                   num_samples: int,
                   batchwise_fp32_outputs_list: list) -> float:
    """
    Compute SQNR given a model and a data loader.
    :param quantized_model: Keras Model for which the sqnr needs to be computed
    :param _: To make it compatible to be used with CallbackFunc Class
    :param original_model: Keras Model agarinst which the sqnr needs to be computed
    :param data_loader_wrapper: Callable function which will return the Data loader to evaluate SQNR from
    :param forward_fn: Function that runs forward pass and returns the output tensor.
    :param num_samples: Number of samples used for evaluation
    :param batchwise_fp32_outputs_list: List to store the model FP32 final outputs
    :return: SQNR in dB scale
    """
    capture_fp32_output_only_once = False
    if not batchwise_fp32_outputs_list:
        capture_fp32_output_only_once = True

    sqnr = 0.0
    data_loader = data_loader_wrapper()
    count = 0
    index = 0
    for x in data_loader:
        if count + len(x) <= num_samples:

            if original_model is not None:
                if capture_fp32_output_only_once:
                    fp32_output = forward_fn(original_model, x)
                    batchwise_fp32_outputs_list.append(fp32_output)
                else:
                    fp32_output = batchwise_fp32_outputs_list[index]
                    index += 1
            else:
                if capture_fp32_output_only_once:
                    with disable_all_quantizers(quantized_model):
                        fp32_output = forward_fn(quantized_model, x)
                        batchwise_fp32_outputs_list.append(fp32_output)
                else:
                    fp32_output = batchwise_fp32_outputs_list[index]
                    index += 1
            quantized_output = forward_fn(quantized_model, x)

            # Accumulate signal by noise ratio
            sqnr += _compute_sqnr(fp32_output, quantized_output)
            count += len(x)
        else:
            break

    # Convert SQNR into dB scale
    sqnr_db = 10 * numpy.log10(sqnr / count)
    return sqnr_db


def _compute_sqnr(orig_tensor: Union[tf.Tensor, numpy.ndarray],
                  noisy_tensor: Union[tf.Tensor, numpy.ndarray]) -> float:
    """
    Compute SQNR between two tensors.

    :param orig_tensor: Original tensor
    :param noisy_tensor: Noisy tensor
    :return: SQNR
    """
    # pylint: disable=unidiomatic-typecheck
    if (type(orig_tensor) != type(noisy_tensor)) \
            or not isinstance(orig_tensor, (numpy.ndarray, tf.Tensor)) \
            or not isinstance(noisy_tensor, (numpy.ndarray, tf.Tensor)):
        raise ValueError("Only tf.Tensor and numpy.ndarray is supported for "
                         "computing SQNR and both original and noisy should have be or same type. "
                         f"Instead got orig_tensor of type {type(orig_tensor)} "
                         f"and noisy_tensor of type {type(noisy_tensor)}")

    assert orig_tensor.shape == noisy_tensor.shape

    # SQNR := E[signal**2] / E[noise**2]

    signal = orig_tensor
    noise = orig_tensor - noisy_tensor
    if isinstance(orig_tensor, numpy.ndarray):
        sqnr = (numpy.power(signal, 2).mean()) / ((numpy.power(noise, 2).mean()) + 0.0001)
    elif isinstance(orig_tensor, tf.Tensor):
        sqnr = tf.reduce_mean(tf.square(signal)) / (tf.reduce_mean(tf.square(noise)) + 0.0001)
    return float(sqnr)


class GreedyMixedPrecisionAlgo(MixedPrecisionAlgo):
    """ Naive Greedy MixedPrecisionAlgo class """

    ENABLE_CONVERT_OP_REDUCTION = False # Run phase-3

    # pylint: disable=too-many-arguments
    def __init__(self, sim: QuantizationSimModel,
                 candidates: List[CANDIDATE_WITH_DTYPE], eval_callback_for_phase1: CallbackFunc,
                 eval_callback_for_phase2: CallbackFunc, results_dir: str,
                 clean_start: bool, forward_pass_callback: CallbackFunc, phase1_optimize: bool = True):
        """
        :param sim: Quantized sim model
        :param candidates: List of Tuple of all possible [bitwidth, QuantizationDataType] values to quantize to
        :param eval_callback_for_phase1: An object of CallbackFunc class which takes in Eval function (callable) and eval
                                     function parameters. This evaluation callback used to measure sensitivity of each
                                     quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                     module. Therefore, a user might want to run the phase 1 with a smaller dataset
        :param eval_callback_for_phase2: An object of CallbackFunc class which takes in Eval function (callable) and eval
                                     function parameters. Evaluation callback used to get accuracy of quantized model
                                     for phase 2 calculations. The phase 2 involves finding pareto front curve
        :param results_dir: Path to save results and cache intermediate results
        :param clean_start: If true, any cached information from previous runs will be deleted prior to starting the
                            mixed-precision analysis. If false, prior cached information will be used if applicable. Note
                            it is the user's responsibility to set this flag to true if anything in the model or
                            quantization parameters changes compared to the previous run.
        :param forward_pass_callback: An object of CallbackFunc class which takes in Forward pass function (callable) and its
                                  function parameters. Forward pass callback used to compute quantization encodings
        :param phase1_optimize: If user set this parameter to true then phase1 optimized logic will be executed else default code will be executed
        """
        # pylint: disable=protected-access
        original_model = sim._model_without_wrappers
        mac_dict = mixed_precision_utils.create_mac_dict(sim._model_without_wrappers)
        super().__init__(sim, candidates, eval_callback_for_phase1, eval_callback_for_phase2, forward_pass_callback,
                         mac_dict, results_dir, clean_start)

        self._read_var_op_parent_op_dict = \
            mixed_precision_utils.find_read_var_op_parent_op_dict(original_model)

        self._baseline_candidate_options = candidates.copy()
        self.phase1_optimize = phase1_optimize

        # set all the quantizer_groups to support all the candidates
        for quantizer_group in self.quantizer_groups:
            self._supported_candidates_per_quantizer_group[quantizer_group] = candidates.copy()

    def _create_and_save_accuracy_list_optimized(self, baseline_candidate) -> ACCURACY_LIST:
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

        file = os.path.join(self._results_dir, '.cache', 'accuracy_list.pkl')
        combinations_already_computed = set()

        if os.path.isfile(file):
            if self._clean_start:
                os.remove(file)
                logger.info("Removed old cached files and restarting computation")
            else:
                with open(file, 'rb') as f:
                    accuracy_list = pickle.load(f)

                combinations_already_computed.update(
                    (quantizer_group, candidate)
                    for quantizer_group, candidate, _, _ in accuracy_list
                )

        disabled_quantizers = OrderedDict()

        try:
            # Disable all quantizers
            for quantizer_group in self.quantizer_groups:
                quantizers = quantizer_group.get_active_quantizers(self._module_name_dict)
                disable_quantizers(quantizers)
                disabled_quantizers[quantizer_group] = quantizers

            # quantizer_groups_per_candidate = {"candidate1":[quantizer_group1,quantizer_group2,...]}
            # quantizer_groups_per_candidate is the dictionary with keys as candidates and values as quantizer groups that supports the corresponding candidate
            # quantizer_groups_per_candidate is like reverse mapping to self._supported_candidates_per_quantizer_group
            quantizer_groups_per_candidate = defaultdict(list)
            for quantizer_group, candidates in self._supported_candidates_per_quantizer_group.items():
                for candidate in candidates:
                    quantizer_groups_per_candidate[candidate].append(quantizer_group)

            # Loop through all possible bitwidths(candidates). Set all the quantizer groups to the corresponding bitwidth(candidate)
            # Compute encodings by disabling the parameters and  reuse the encodings
            for candidate, quantizer_groups in quantizer_groups_per_candidate.items():
                if candidate == baseline_candidate:
                    continue

                # configure the sim model with the candidate by enabling the quantizers and set quantizers to corresponding candidate
                for quantizer_group in quantizer_groups:
                    quantizers = disabled_quantizers[quantizer_group]
                    try:
                        enable_quantizers(quantizers)
                        # Set quantizer bitwidth to candidate (bitwidth)
                        quantizer_group.set_quantizers_to_candidate(self._module_name_dict, candidate)
                    except RuntimeError as e:
                        logger.info("Exception occured while setting Quantizers to Candidate: %s", e)

                disabled_param_quantizers = OrderedDict()
                for quantizer_group in quantizer_groups:
                    quantizers = quantizer_group.get_active_param_quantizers(self._module_name_dict)
                    disable_quantizers(quantizers)
                    disabled_param_quantizers[quantizer_group] = quantizers


                # compute encodings with out parameter quantization
                self._sim.compute_encodings(self.algo_params.forward_pass_callback,
                                            self.algo_params.forward_pass_callback_args)


                # enable the parameter quantization
                for quantizer_group in quantizer_groups:
                    quantizers = disabled_param_quantizers[quantizer_group]
                    enable_quantizers(quantizers)
                # compute parameter encodings and set op_mode for parameters
                # pylint: disable=protected-access
                self._sim._compute_and_set_parameter_encodings([])
                op_mode = self._sim._param_op_mode_after_analysis(self._sim.quant_scheme)
                self._sim._set_op_mode_parameters(op_mode)

                # Disable all the quantizers
                for quantizer_group in quantizer_groups:
                    quantizers = quantizer_group.get_active_quantizers(self._module_name_dict)
                    disable_quantizers(quantizers)
                    disabled_quantizers[quantizer_group] = quantizers

                # Loop over all the quantizer groups and enable one at a time and calculate resulting model accuracy and disable the enabled quantizer
                # Accuracy list will contain tuples of the quantizer, bitwidth, and accuracy score
                for quantizer_group in quantizer_groups:
                    quantizers = disabled_quantizers[quantizer_group]
                    try:
                        enable_quantizers(quantizers)
                        # If starting the computation from an already existing state, then check if that combination
                        # has already been executed
                        if (quantizer_group, candidate) in combinations_already_computed:
                            continue
                        # Compute accuracy of model with new candidate (bitwidth)
                        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase1)

                        bit_ops_reduction = self._find_bit_ops_reduction_for_acc_list(quantizer_group, baseline_candidate, candidate)
                        accuracy_list.append((quantizer_group, candidate, eval_score, bit_ops_reduction))
                        # Sort accuracy list, first by descending accuracy score, then by descending order of addition of bitwidths if accuracy
                        # scores are identical, if that is also identical we sort by relative bit ops change in descending order
                        # If bit ops reduction is also the same, then we sort in ascending order based on occurence of
                        # quantizer group in the model
                        accuracy_list = sort_accuracy_list(accuracy_list, index_of_quantizer_group)
                        self._export_accuracy_list(accuracy_list, self._results_dir)
                        logger.info('\n Quantizer: %s candidate: %s eval_score: %f \n', quantizer_group,
                                    candidate, eval_score)
                    finally:
                        # Disable the quantizer
                        disable_quantizers(quantizers)
        finally:
            # set all quantizers to baseline candidate
            for quantizer_group in self.quantizer_groups:
                quantizers = disabled_quantizers[quantizer_group]
                try:
                    # Enable the disabled quantizers
                    enable_quantizers(quantizers)
                    quantizer_group.set_quantizers_to_candidate(self._module_name_dict, baseline_candidate)
                except RuntimeError as e:
                    logger.info("Exception occured while setting Quantizers to Candidate: %s", e)

        logger.info('Completed Accuracy list computation')
        # Recompute encodings after quantizer's bitwidth is set back to self._max_bitwidth
        self._sim.compute_encodings(self.algo_params.forward_pass_callback, self.algo_params.forward_pass_callback_args)
        return accuracy_list

    def _evaluate_model(self, eval_callback: CallbackFunc) -> float:
        """
        Evaluates a model
        :param eval_callback: Callback function that contains eval function and eval args
        :return: Eval score
        """
        return eval_callback.func(self._sim.model, eval_callback.args)

    @property
    def baseline_candidate_options(self) -> List[CANDIDATE_WITH_DTYPE]:
        """
        Returns the _baseline_candidate_options which is the intersection of amp candidates and candidates supported by
        all the quantizer groups
        """
        return self._baseline_candidate_options

    def _find_quantizer_group(self, sim) -> Tuple[Dict, List[QuantizerGroup]]:
        """
        Finds quantizer groups in a quantization sim
        :param sim: Quantization sim
        :return: Dictionary mapping quantized op name to sim.quantizer_config,
            and a List of quantizer groups
        """
        return find_quantizer_group(sim)

    def _find_bit_ops_reduction_for_acc_list(self,
                                             quantizer_group: QuantizerGroup,
                                             max_candidate: CANDIDATE_WITH_DTYPE,
                                             candidate: CANDIDATE_WITH_DTYPE) -> int:
        """
        Finds reduction in bit ops from max candidate to new candidate
        :param quantizer_group: Quantizer group
        :param candidate: Activation bitwidth, parameter bitwidth
        :return: Bit ops reduction
        """
        return mixed_precision_utils.find_bit_ops_reduction(quantizer_group, self._mac_dict,
                                                            self._read_var_op_parent_op_dict,
                                                            max_candidate, candidate)

    def calculate_running_bit_ops(self,
                                  quantizer_group: QuantizerGroup,
                                  module_bitwidth_dict: Dict,
                                  max_candidate: CANDIDATE_WITH_DTYPE,
                                  candidate: CANDIDATE_WITH_DTYPE,
                                  running_bit_ops: int) -> int:
        """
        Calculates running bit ops value for every quantizer group
        :param quantizer_group: A group of activation & parameter quantizers
        :param module_bitwidth_dict: Dict; Key: Module name value: Activation, parameter bitwidth of module
        :param candidate: Bitwidth to change the quantizer group to
        :param running_bit_ops: Running bit ops value calculated uptil the quantizer group
        :return: Running bit ops value
        """
        running_bit_ops = mixed_precision_utils.calculate_running_bit_ops(self._mac_dict, quantizer_group,
                                                                          self._read_var_op_parent_op_dict,
                                                                          module_bitwidth_dict,
                                                                          max_candidate,
                                                                          candidate,
                                                                          running_bit_ops)
        return running_bit_ops

    def _create_and_save_accuracy_list(self, baseline_candidate):
        try:
            if self.phase1_optimize:
                return self._create_and_save_accuracy_list_optimized(baseline_candidate)
            return super()._create_and_save_accuracy_list(baseline_candidate)
        finally:
            pass

    def _create_op_graph(self, sim):
        """
        Creates op graph

        :param sim: QuantizationSimModel object
        """
        return None


    def _optimize_mp_profile_and_evaluate_model(self):
        """
        Uses OpGraph if available to optimize the mixed precision profile in the sim object
        """
        # Recompute quantizer encodings
        self._sim.compute_encodings(self.algo_params.forward_pass_callback,
                                    self.algo_params.forward_pass_callback_args)
        # Compute new accuracy score
        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase2)
        return eval_score

    def set_quantizer_groups_candidates(self, quantizer_group_candidates: List[Tuple]) -> None:
        """
        Setter function to set quantizer groups to given candidate. This method also computes the encodings following
        the change in quantizer groups
        :param quantizer_group_candidates: list of quantizer groups and their candidates
        """
        def validate_quantizer_candidate(qg: QuantizerGroup, qg_candidate) -> bool:
            """
            Helper method to validate whether candidate can be applied
            :param qg: quantizer group whose candidate needs to be changed
            :param qg_candidate: the new candidate which needs to be applied to the quantizer group
            :return: boolean value True if success else False
            """
            supported_candidates = self._supported_candidates_per_quantizer_group.get(qg)
            if not qg.parameter_quantizers:
                (activation_bw, activation_data_type), _ = qg_candidate
                # Since only activation quantizers are present, validate activation candidate
                for supported_candidate in supported_candidates:
                    if supported_candidate[0] == (activation_bw, activation_data_type):
                        return True
                return False
            # both activation and param quantizers are present in the quantizer group
            return qg_candidate in supported_candidates

        for quantizer_group, candidate in quantizer_group_candidates:
            assert validate_quantizer_candidate(quantizer_group, candidate)
            quantizer_group.set_quantizers_to_candidate(self._module_name_dict, candidate)

        self._sim.compute_encodings(self.algo_params.forward_pass_callback, self.algo_params.forward_pass_callback_args)

    # pylint: disable=protected-access
    def _reduce_mp_convert_ops(self):
        """
        Reduce mixed precision convert ops overhead if enabled and supported
        """
        if self.ENABLE_CONVERT_OP_REDUCTION:
            reduce_convert_ops_algo = ReduceConvertOps(self._sim, self.quantizer_groups, self.algo_params.candidates, self._mac_dict)

            # Check if phase 2 solution is all 8 bits
            phase2_all_8bits = all(
                sol == 8 for sol in
                reduce_convert_ops_algo._phase_two_sol.values()
            )

            # Check if phase 2 solution is all 16 bits
            phase2_all_16bits = all(
                sol == 16 for sol in
                reduce_convert_ops_algo._phase_two_sol.values()
            )

            if phase2_all_8bits or phase2_all_16bits:
                logger.warning('Skipping phase3 because there is no scope to reduce convert-op overhead')
            else:
                for alpha in reduce_convert_ops_algo.DEFAULT_ALPHA_OPTIONS:
                    phase_three_sol, solve_data_dict = \
                        reduce_convert_ops_algo.run_amp_phase_3(alpha, SamplingStrategy.weighted_with_predicted_convert_cost)
                    qg_candidates = reduce_convert_ops_algo.generate_qg_solution(phase_three_sol)

                    # save sim state with the new quantizer settings
                    self.set_quantizer_groups_candidates(qg_candidates)

                    # export encodings; convert to pb has to be disabled as it invalidates all the models present in the
                    # current thread that were created outside tf.compat.v1.Graph; sim.model is still needed for other alpha
                    self._sim.export(path=self._results_dir, filename_prefix=f'AMP_ph3_export_alpha_{alpha}', convert_to_pb=False)

                    reduce_convert_ops_algo.save_phase_3_data_as_json(solve_data_dict, self._results_dir,
                                                                      filename_suffix="alpha_{}".format(alpha))
