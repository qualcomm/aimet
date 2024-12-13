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
import os
from collections import defaultdict, OrderedDict
import pickle
import functools
from typing import Any, Callable, Union, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from aimet_common.utils import AimetLogger
from aimet_common.defs import CallbackFunc
from aimet_common.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo as MixedPrecisionAlgo
from aimet_common.amp.quantizer_groups import reformat_supported_kernels
from aimet_common.amp.utils import (
    sort_accuracy_list,
    CANDIDATE_WITH_DTYPE,
    ACCURACY_LIST,
    disable_quantizers,
    enable_quantizers,
)
from aimet_common.amp.convert_ops_reduction import SamplingStrategy
from aimet_torch import utils
from aimet_torch._base.amp import utils as mixed_precision_utils
from aimet_torch._base.amp.convert_ops_reduction import ReduceConvertOps
from aimet_torch._base.amp.quantizer_groups import find_quantizer_group, QuantizerGroup, get_module_name_to_module_dict, find_supported_candidates
from aimet_torch._base.quantsim import _QuantizationSimModelInterface


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


class EvalCallbackFactory:
    """
    Factory class for various built-in eval callbacks
    """
    def __init__(self,
                 data_loader: DataLoader,
                 forward_fn: Callable[[torch.nn.Module, Any], torch.Tensor] = None):
        """

        :param data_loader: Data loader to be used for evaluation
        :param forward_fn: Function that runs forward pass and returns the output tensor.
                           This function is expected to take 1) a model and 2) a single batch
                           yielded from the data loader, and return a single torch.Tensor object
                           which represents the output of the model.
                           The default forward function is roughly equivalent to
                           ``lambda model, batch: model(batch)``
        """
        self._data_loader = data_loader
        self._forward_fn = forward_fn or _default_forward_fn

        # storing batchwise fp32 outputs in the list
        self._batchwise_fp32_outputs_list = []

    def _forward_fn_wrapper(self, *args, **kwargs):
        output = self._forward_fn(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "Forward pass was expected to return a torch.Tensor, "
                f"but returned an object of type {type(output)}. "
                "Try specifying `forward_fn` to adapt the output."
            )
        return output

    _DEFAULT_SQNR_NUM_SAMPLES = 128


    def sqnr(self, num_samples: int = _DEFAULT_SQNR_NUM_SAMPLES) -> CallbackFunc:
        """
        Returns SQNR eval callback.

        :param num_samples: Number of samples used for evaluation
        :return: A callback function that evaluates the input model's SQNR
                 between fp32 outputs and fake-quantized outputs
        """
        evaluate_sqnr = functools.partial(_evaluate_sqnr,
                                          data_loader=self._data_loader,
                                          forward_fn=self._forward_fn_wrapper,
                                          num_samples=num_samples,
                                          batchwise_fp32_outputs_list=self._batchwise_fp32_outputs_list)
        return CallbackFunc(evaluate_sqnr)


def _default_forward_fn(model, inputs):
    if isinstance(inputs, torch.Tensor):
        return model(inputs)

    assert isinstance(inputs, (tuple, list))
    return model(*inputs)


def _evaluate_sqnr(model: torch.nn.Module, _: Any,
                   data_loader: DataLoader,
                   forward_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
                   num_samples: int,
                   batchwise_fp32_outputs_list: list) -> float:
    """
    Compute SQNR given a model and a data loader.

    :param model: Root module
    :param _: Placeholder for CallbackFunc
    :param data_loader: Data loader to evaluate SQNR from
    :param forward_fn: Function that runs forward pass and returns the output tensor.
    :param num_samples: Number of samples used for evaluation
    :param batchwise_fp32_outputs_list: List to store the model FP32 final outputs
    :return: SQNR in dB scale
    """
    # If capture_fp32_output_only_once is true then batchwise_fp32_outputs_list would be empty so we can calculate outputs by passing through model
    # we can store those results and use instead of passing to model
    # Initially capture_fp32_output_only_once is false and we make it to true if batchwise_fp32_outputs_list is empty to store the outputs in the list

    capture_fp32_output_only_once = False
    if not batchwise_fp32_outputs_list:
        capture_fp32_output_only_once = True

    sqnr = 0.0
    batch_size = data_loader.batch_size or 1
    device = utils.get_device(model)
    with utils.in_eval_mode(model), torch.no_grad():
        for i, x in enumerate(data_loader):
            if i * batch_size < num_samples:
                x = utils.change_tensor_device_placement(x, device)
                # First time we pass input to model and store the outputs in batchwise_fp32_outputs_list
                if capture_fp32_output_only_once:
                    with utils.disable_all_quantizers(model):
                        fp32_output = forward_fn(model, x)
                    batchwise_fp32_outputs_list.append(fp32_output)
                else:
                    # Re-using the stored outputs instead of computing again
                    fp32_output = batchwise_fp32_outputs_list[i]

                quantized_output = forward_fn(model, x)

                # Accumulate signal by noise ratio
                # changed in_place to False otherwise it overwrites the batchwise_fp32_outputs_list
                sqnr += _compute_sqnr(fp32_output, quantized_output, in_place=False)
            else:
                break

    # Convert SQNR into dB scale
    sqnr_db = 10 * np.log10(sqnr / num_samples)
    return sqnr_db


def _compute_sqnr(orig_tensor: torch.Tensor,
                  noisy_tensor: torch.Tensor,
                  in_place: bool = False) -> float:
    """
    Compute SQNR between two tensors.

    IMPORTANT: If in_place is True, the input tensors will be used as in-place buffer
               and hence will have been corrupted when this function returns.

    :param orig_tensor: Original tensor
    :param noisy_tensor: Noisy tensor
    :param in_place: If True, use the input tensors as in-place buffer
    :return: SQNR
    """
    assert orig_tensor.shape == noisy_tensor.shape
    assert orig_tensor.dtype == noisy_tensor.dtype

    # SQNR := E[signal**2] / E[noise**2]
    if in_place:
        signal = orig_tensor
        noise = noisy_tensor.negative_().add_(orig_tensor)
    else:
        signal = orig_tensor.detach().clone()
        noise = orig_tensor - noisy_tensor

    sqnr = signal.square_().mean() / (noise.square_().mean() + 0.0001)
    return float(sqnr)


class GreedyMixedPrecisionAlgo(MixedPrecisionAlgo):
    """ Naive Greedy MixedPrecisionAlgo class """

    ENABLE_MP_PROFILE_OPTIMIZE = False # Enables op_graph.optimize() during Phase-2
    ENABLE_CONVERT_OP_REDUCTION = False # Run phase-3

    # pylint: disable=too-many-arguments
    def __init__(self, sim: _QuantizationSimModelInterface,
                 dummy_input: Union[torch.Tensor, Tuple],
                 candidates: List[CANDIDATE_WITH_DTYPE],
                 eval_callback_for_phase1: CallbackFunc,
                 eval_callback_for_phase2: CallbackFunc,
                 results_dir: str, clean_start: bool,
                 forward_pass_callback: CallbackFunc,
                 use_all_amp_candidates: bool = False,
                 phase2_reverse: bool = False,
                 phase1_optimize: bool = False):
        """

        :param sim: Quantized sim model
        :param dummy_input: Dummy input to the model. If the model has more than one input, pass a tuple.
                            User is expected to place the tensors on the appropriate device.
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
        :param use_all_amp_candidates: Using the “supported_kernels” field in the config file (under defaults
                    and op_type sections), a list of supported candidates can be specified. All the AMP candidates
                    which are passed through the “candidates” field may not be supported based on the data passed
                    through “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP algo
                    will ignore the "supported_kernels" in the config file and will continue to use all the candidates.
        :param phase2_reverse: If user will set this parameter to True, then phase1 of amp algo, that is calculating accuracy list will not be changed,
                            whereas the phase2 algo of amp, which generate the pareto list will be changed. In phase2, algo will start, model with all quantizer groups in least candidate, and
                            one by one, it will put nodes in higher candidate till target accuracy does not meet.
        :phase1_optimize: If user set this parameter to True then phase1 optimized logic will be executed else common code will be executed
        """
        mac_dict = mixed_precision_utils.create_mac_dict(sim.model, dummy_input)
        self.phase1_optimize = phase1_optimize
        self.dummy_input = dummy_input

        super().__init__(sim, candidates,
                                                       eval_callback_for_phase1,
                                                       eval_callback_for_phase2,
                                                       forward_pass_callback,
                                                       mac_dict,
                                                       results_dir, clean_start, phase2_reverse)

        supported_kernels = reformat_supported_kernels(sim.get_supported_kernels())

        # Find 1. candidates for each of the quantizers by using supported_kernels, candidates and
        # use_all_amp_candidates (flag)
        # 2. max_candidate_options based on the candidates which are present in all the quantizers
        self._supported_candidates_per_quantizer_group, self._baseline_candidate_options = find_supported_candidates(
            self.quantizer_groups,
            candidates,
            supported_kernels,
            get_module_name_to_module_dict(sim),
            use_all_amp_candidates)


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

                # list to store all the param quantizers
                param_quantizers = []

                for _, wrapper in self._sim.quant_wrappers():
                    for _, param_quantizer in wrapper.param_quantizers.items():
                        if param_quantizer.enabled:
                            param_quantizers.append(param_quantizer)

                # disable the parameter quantization
                disable_quantizers(param_quantizers)


                # compute encodings with out parameter quantization
                self._sim.compute_encodings(self.algo_params.forward_pass_callback,
                                            self.algo_params.forward_pass_callback_args)

                # enable the parameter quantization
                enable_quantizers(param_quantizers)


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

                        bit_ops_reduction = self._find_bit_ops_reduction_for_acc_list(quantizer_group,
                                                                                      baseline_candidate,
                                                                                      candidate)
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



    def _evaluate_model(self, eval_callback) -> float:
        """
        Evaluates a model

        :param eval_callback: Callback function that contains eval function and eval args
        :return: Eval score
        """
        return eval_callback.func(self._sim.model, eval_callback.args)

    def _find_quantizer_group(self, sim) -> Tuple[Dict, List[QuantizerGroup]]:
        """
        Finds quantizer groups in a quantization sim

        :param sim: Quantization sim
        :return: Dictionary mapping quantized op name to sim.quantizer_config,
            and a List of quantizer groups
        """
        return find_quantizer_group(sim)

    @property
    def baseline_candidate_options(self) -> List[CANDIDATE_WITH_DTYPE]:
        """
        Returns the _baseline_candidate_options which is the intersection of amp candidates and candidates supported by
        all the quantizer groups
        """
        return self._baseline_candidate_options

    def _find_bit_ops_reduction_for_acc_list(
            self,
            quantizer_group: QuantizerGroup,
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
        return mixed_precision_utils.find_bit_ops_reduction(quantizer_group, self._mac_dict,
                                                            max_candidate, candidate)

    def calculate_running_bit_ops(
            self,
            quantizer_group: QuantizerGroup,
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
        :param candidate: candidate to change the quantizer group to
        :param running_bit_ops: Running bit ops value calculated uptil the quantizer group
        :return: Running bit ops value
        """
        running_bit_ops = mixed_precision_utils.calculate_running_bit_ops(self._mac_dict, quantizer_group,
                                                                          module_bitwidth_dict,
                                                                          max_candidate,
                                                                          candidate,
                                                                          running_bit_ops)
        return running_bit_ops

    def _create_and_save_accuracy_list(self, baseline_candidate):
        def disable_all_quantizers():
            return utils.disable_all_quantizers(self._sim.model)

        # Note: "disable_all_quantizers" will be migrated to the open source codew
        #        as a method of QuantizationSimModel.
        try:
            setattr(self._sim, "disable_all_quantizers", disable_all_quantizers)
            if self.phase1_optimize:
                return self._create_and_save_accuracy_list_optimized(baseline_candidate)
            return super()._create_and_save_accuracy_list(baseline_candidate)
        finally:
            delattr(self._sim, "disable_all_quantizers")

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

    def _create_op_graph(self, sim):
        """
        Creates op graph

        :param sim: QuantizationSimModel object
        """
        return None

    def _optimize_mp_profile_and_evaluate_model(self):
        """
        Get the eval score of the model based on whether ENABLE_MP_PROFILE_OPTIMIZE is enabled.
        """

        # Recompute quantizer encodings
        self._sim.compute_encodings(self.algo_params.forward_pass_callback,
                                    self.algo_params.forward_pass_callback_args)
        # Compute new accuracy score
        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase2)

        return eval_score

    # pylint: disable=protected-access
    def _reduce_mp_convert_ops(self):
        """
        Reduce mixed precision convert ops if enabled and supported
        """
        if self.ENABLE_CONVERT_OP_REDUCTION:
            reduce_convert_ops_algo = ReduceConvertOps(self._sim, self.quantizer_groups, self.algo_params.candidates, self._mac_dict)

            # Check if phase 2 solution is all 8 bits
            phase2_all_8bits = all(
                reduce_convert_ops_algo._phase_two_sol[qg] == 8 for qg in
                reduce_convert_ops_algo._phase_two_sol)

            # Check if phase 2 solution is all 16 bits
            phase2_all_16bits = all(
                reduce_convert_ops_algo._phase_two_sol[qg] == 16 for qg in
                reduce_convert_ops_algo._phase_two_sol)

            dummy_input_cpu = reduce_convert_ops_algo.convert_tensor_to_cpu(self.dummy_input)

            if phase2_all_8bits or phase2_all_16bits:
                logger.warning('Skipping phase3 because there is no scope to reduce convert-op overhead')
            else:
                for alpha in reduce_convert_ops_algo.DEFAULT_ALPHA_OPTIONS:
                    phase_three_sol, solve_data_dict = \
                        reduce_convert_ops_algo.run_amp_phase_3(alpha, SamplingStrategy.weighted_with_predicted_convert_cost)
                    qg_candidates = reduce_convert_ops_algo.generate_qg_solution(phase_three_sol)

                    # save sim state with the new quantizer settings
                    self.set_quantizer_groups_candidates(qg_candidates)

                    # export
                    self._sim.export(path=self._results_dir, filename_prefix=f'AMP_ph3_export_alpha_{alpha}',
                                     dummy_input=dummy_input_cpu)

                    reduce_convert_ops_algo.save_phase_3_data_as_json(solve_data_dict, self._results_dir,
                                                                      filename_suffix="alpha_{}".format(alpha))
