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
"""Evaluator class for mixed precision"""

import contextlib
import os
from collections import defaultdict, OrderedDict
import pickle
import functools
import itertools
from typing import Any, Callable, Tuple, List, Dict
import json
import numpy as np
import onnxruntime as ort

from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_common.defs import CallbackFunc, qtype
from aimet_common.amp.mixed_precision_algo import (
    GreedyMixedPrecisionAlgo as MixedPrecisionAlgo,
)
from aimet_common.amp.quantizer_groups import reformat_supported_kernels
from aimet_common.amp.utils import (
    sort_accuracy_list,
    CANDIDATE_WITH_DTYPE,
    ACCURACY_LIST,
    disable_quantizers,
    enable_quantizers,
    candidate_cost,
)

from aimet_onnx.amp import utils as mixed_precision_utils
from aimet_onnx.amp.quantizer_groups import (
    find_quantizer_group,
    QuantizerGroup,
    find_supported_candidates,
)
from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.defs import DataLoader

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


@contextlib.contextmanager
def _disable_all_quantizers(sim: QuantizationSimModel):
    """
    Temporarily disable all quantizers in the model within with-as block.

    :param sim: Quantized sim model
    """
    active_quantizers = set(
        quantizer for quantizer in sim.qc_quantize_op_dict.values() if quantizer.enabled
    )

    try:
        for quantizer in active_quantizers:
            quantizer.enabled = False
        yield
    finally:
        for quantizer in active_quantizers:
            quantizer.enabled = True


class EvalCallbackFactory:
    """
    Factory class for various built-in eval callbacks
    """

    def __init__(
        self,
        data_loader: DataLoader,
        forward_fn: Callable[[ort.InferenceSession, Any], np.ndarray] = None,
    ):
        """
        :param data_loader: Data loader to be used for evaluation
        :param forward_fn: Function that runs forward pass and returns the output tensor.
                           This function is expected to take 1) a model 2) List of starting op names
                           3) List of output op names and 4) batch yielded from the data set, and
                           return a single tf.Tensor (or np.ndarray) object which represents the output of the model.
        """
        self._data_loader = data_loader
        self._forward_fn = forward_fn or _default_forward_fn

        # storing batchwise fp32 outputs in the list
        self._batchwise_fp32_outputs_list = []

    def _forward_fn_wrapper(self, *args, **kwargs):
        output = self._forward_fn(*args, **kwargs)
        if not isinstance(output, np.ndarray):
            raise RuntimeError(
                "Forward pass was expected to return a numpy.ndarray, "
                f"but returned an object of type {type(output)}. "
                "Try specifying `forward_fn` to adapt the output."
            )
        return output

    _DEFAULT_SQNR_NUM_SAMPLES = 128

    def sqnr(
        self, sim: QuantizationSimModel, num_samples: int = _DEFAULT_SQNR_NUM_SAMPLES
    ) -> Callable[[ort.InferenceSession], float]:
        """
        Returns SQNR eval callback.
        NOTE: sim object is required to enable/disable quantizer_info objects associated with quant ops.

        :param sim: Quantized sim model
        :param num_samples: Number of samples used for evaluation
        :return: A callback function that evaluates model SQNR between fp32_outputs and quantized outputs.
        """
        evaluate_sqnr = functools.partial(
            _evaluate_sqnr,
            sim=sim,
            data_loader=self._data_loader,
            forward_fn=self._forward_fn_wrapper,
            num_samples=num_samples,
            batchwise_fp32_outputs_list=self._batchwise_fp32_outputs_list,
        )
        return CallbackFunc(evaluate_sqnr)


def _default_forward_fn(sess: ort.InferenceSession, inputs: Any) -> np.ndarray:
    output_tensors = sess.run(None, {"input": inputs})[0]
    return output_tensors


def _evaluate_sqnr(
    session: ort.InferenceSession,
    _: Any,
    sim: QuantizationSimModel,
    data_loader: DataLoader,
    forward_fn: Callable,
    num_samples: int,
    batchwise_fp32_outputs_list: list,
) -> float:
    """
    Compute SQNR given a model and a data loader.

    :param session: sim session
    :param sim: Quantization sim model
    :param data_loader: Data loader to evaluate SQNR from
    :param forward_fn: Function that runs forward pass and returns the output tensor.
    :param num_samples: Number of samples used for evaluation
    :return: SQNR in dB scale
    """
    assert sim.session == session, (
        "session associated with sim and session passed to this callback should be same."
    )
    capture_fp32_output_only_once = False
    if not batchwise_fp32_outputs_list:
        capture_fp32_output_only_once = True

    sqnr = 0.0
    batch_size = data_loader.batch_size or 1
    for i, x in enumerate(data_loader):
        if i * batch_size < num_samples:
            if capture_fp32_output_only_once:
                with _disable_all_quantizers(sim):
                    fp32_output = forward_fn(session, x)
                batchwise_fp32_outputs_list.append(fp32_output)
            else:
                fp32_output = batchwise_fp32_outputs_list[i]

            quantized_output = forward_fn(session, x)
            # Accumulate signal by noise ratio
            sqnr += _compute_sqnr(fp32_output, quantized_output)
        else:
            break

    # Convert SQNR into dB scale
    sqnr_db = 10 * np.log10(sqnr / num_samples)
    return sqnr_db


def _compute_sqnr(orig_tensor: np.ndarray, noisy_tensor: np.ndarray) -> float:
    """
    Compute SQNR between two tensors.

    :param orig_tensor: Original tensor
    :param noisy_tensor: Noisy tensor
    :return: SQNR
    """
    assert orig_tensor.shape == noisy_tensor.shape

    # SQNR := E[signal**2] / E[noise**2]
    signal = orig_tensor
    noise = orig_tensor - noisy_tensor
    sqnr = (np.power(signal, 2).mean()) / ((np.power(noise, 2).mean()) + 0.0001)
    return float(sqnr)


class GreedyMixedPrecisionAlgo(MixedPrecisionAlgo):
    """Naive Greedy MixedPrecisionAlgo class"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sim: QuantizationSimModel,
        candidates: List[CANDIDATE_WITH_DTYPE],
        eval_callback_for_phase1: Callable[[ort.InferenceSession], float],
        eval_callback_for_phase2: Callable[[ort.InferenceSession], float],
        results_dir: str,
        clean_start: bool,
        forward_pass_callback: Callable[[ort.InferenceSession], Any],
        use_all_amp_candidates: bool = False,
        phase1_optimize: bool = False,
    ):
        """
        :param sim: Quantized sim model
        :param candidates: List of Tuple of all possible [bitwidth, QuantizationDataType] values to quantize to
        :param eval_callback_for_phase1: Callable object used to measure sensitivity of each
                                     quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                     module. Therefore, a user might want to run the phase 1 with a smaller dataset
        :param eval_callback_for_phase2: Callale object used to get accuracy of quantized model
                                     for phase 2 calculations. The phase 2 involves finding pareto front curve
        :param results_dir: Path to save results and cache intermediate results
        :param clean_start: If true, any cached information from previous runs will be deleted prior to starting the
                            mixed-precision analysis. If false, prior cached information will be used if applicable. Note
                            it is the user's responsibility to set this flag to true if anything in the model or
                            quantization parameters changes compared to the previous run.
        :param forward_pass_callback: Callable object used to compute quantization encodings
        :param use_all_amp_candidates: Using the “supported_kernels” field in the config file (under defaults
                    and op_type sections), a list of supported candidates can be specified. All the AMP candidates
                    which are passed through the “candidates” field may not be supported based on the data passed
                    through “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP algo
                    will ignore the "supported_kernels" in the config file and will continue to use all the candidates.
        :phase1_optimize: If user set this parameter to true then phase1 optimized logic will be executed else default code will be executed
        """
        mac_dict = mixed_precision_utils.create_mac_dict(sim)
        self.phase1_optimize = phase1_optimize

        super().__init__(
            sim,
            candidates,
            eval_callback_for_phase1,
            eval_callback_for_phase2,
            forward_pass_callback,
            mac_dict,
            results_dir,
            clean_start,
        )
        self._param_name_to_op_name_dict = (
            mixed_precision_utils.find_param_name_to_parent_name_dict(
                sim.connected_graph
            )
        )

        supported_kernels = reformat_supported_kernels(sim.get_supported_kernels())

        (
            self._supported_candidates_per_quantizer_group,
            self._baseline_candidate_options,
        ) = find_supported_candidates(
            self.quantizer_groups,
            candidates,
            supported_kernels,
            mixed_precision_utils.get_quantizer_to_op_type_dict(sim),
            use_all_amp_candidates,
        )

    def _create_and_save_accuracy_list_optimized(
        self, baseline_candidate
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

            # quantizer_groups_per_candidate = {"candidate1":[quantizer_group1,quantizer_group2,...]}
            # quantizer_groups_per_candidate is the dictionary with keys as candidates and values as quantizer groups that supports the corresponding candidate
            # quantizer_groups_per_candidate is like reverse mapping to self._supported_candidates_per_quantizer_group
            quantizer_groups_per_candidate = defaultdict(list)
            for (
                quantizer_group,
                candidates,
            ) in self._supported_candidates_per_quantizer_group.items():
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
                        quantizer_group.set_quantizers_to_candidate(
                            self._module_name_dict, candidate
                        )
                    except RuntimeError as e:
                        logger.info(
                            "Exception occured while setting Quantizers to Candidate: %s",
                            e,
                        )

                # list to store all the param quantizers
                param_quantizers_qgp = []

                for quantizer_group in quantizer_groups:
                    for quantizer in quantizer_group.get_param_quantizers(
                        self._module_name_dict
                    ):
                        if quantizer.enabled:
                            param_quantizers_qgp.append(quantizer)

                # compute encodings
                self._sim.compute_encodings(
                    self.algo_params.forward_pass_callback,
                    self.algo_params.forward_pass_callback_args,
                )
                # export encodings
                self._export_encodings(self._results_dir)

                # disable the parameter quantization
                disable_quantizers(param_quantizers_qgp)

                # compute encodings with out parameter quantization
                self._sim.compute_encodings(
                    self.algo_params.forward_pass_callback,
                    self.algo_params.forward_pass_callback_args,
                )

                # export activation encodings
                self._export_activation_encodings(self._results_dir)
                # enable the parameter quantization
                enable_quantizers(param_quantizers_qgp)
                self._load_param_encodings(self._results_dir)

                # Disable all the quantizers
                for quantizer_group in quantizer_groups:
                    quantizers = quantizer_group.get_active_quantizers(
                        self._module_name_dict
                    )
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
                        if (
                            quantizer_group,
                            candidate,
                        ) in combinations_already_computed:
                            continue
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
                        # Disable the quantizer
                        disable_quantizers(quantizers)
        finally:
            # set all quantizers to baseline candidate
            for quantizer_group in self.quantizer_groups:
                quantizers = disabled_quantizers[quantizer_group]
                try:
                    # Enable the disabled quantizers
                    enable_quantizers(quantizers)
                    quantizer_group.set_quantizers_to_candidate(
                        self._module_name_dict, baseline_candidate
                    )
                except RuntimeError as e:
                    logger.info(
                        "Exception occured while setting Quantizers to Candidate: %s", e
                    )

        logger.info("Completed Accuracy list computation")
        # Recompute encodings after quantizer's bitwidth is set back to self._max_bitwidth
        self._sim.compute_encodings(
            self.algo_params.forward_pass_callback,
            self.algo_params.forward_pass_callback_args,
        )
        return accuracy_list

    def _export_encodings(self, path: str):
        """
        Export encodings of the sim model to the given path

        :param path: Encodings will store in the given path/.cache folder
        """
        results_dir = os.path.join(path, ".cache")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        path = os.path.join(results_dir, "encodings_with_param.encodings")
        # pylint: disable=protected-access
        self._sim._export_encodings(path)

    def _export_activation_encodings(self, path: str):
        """
        Export encodings of the sim model to the given path

        :param path: Encodings will store in the given path/.cache folder
        """
        results_dir = os.path.join(path, ".cache")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        path = os.path.join(results_dir, "encodings_with_act.encodings")
        # pylint: disable=protected-access
        self._sim._export_encodings(path)

    def _load_param_encodings(self, path: str):
        """
        Loads parameter encodings to the sim model

        :param path: Folder Path where encodings file is present
        """
        param_path = os.path.join(path, ".cache/encodings_with_param.encodings")

        # Load encodings file
        with open(param_path) as json_file:
            param_encodings = json.load(json_file)

        # Loading activation encodings also along with param encodings to get compatible with load_encodings_to_sim function
        # These activation encodings are already present in quantsim model
        act_path = os.path.join(path, ".cache/encodings_with_act.encodings")
        with open(act_path) as json_file:
            act_encodings = json.load(json_file)

        param_encodings["activation_encodings"] = act_encodings["activation_encodings"]

        # Save the updated encodings to the file
        save_json_yaml(param_path, param_encodings)
        # load encodings
        load_encodings_to_sim(self._sim, param_path, strict=True)

        # Removing the files created by _export_encodings function
        os.remove(param_path)
        os.remove(act_path)

    def _evaluate_model(
        self, eval_callback: Callable[[ort.InferenceSession], float]
    ) -> float:
        """
        Evaluates a model

        :param eval_callback: Callback function that contains eval function and eval args
        :return: Eval score
        """
        return eval_callback(self._sim.session)

    def _find_quantizer_group(
        self, sim: QuantizationSimModel
    ) -> Tuple[Dict[str, QcQuantizeOp], List[QuantizerGroup]]:
        """
        Finds quantizer groups in a quantization sim
        :param sim: Quantization sim
        :return: Dictionary mapping quantized op name to quantizer,
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
        return mixed_precision_utils.find_bit_ops_reduction(
            quantizer_group,
            self._mac_dict,
            self._param_name_to_op_name_dict,
            max_candidate,
            candidate,
        )

    def calculate_running_bit_ops(
        self,
        quantizer_group: QuantizerGroup,
        module_bitwidth_dict: Dict[str, int],
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
        running_bit_ops = mixed_precision_utils.calculate_running_bit_ops(
            self._mac_dict,
            quantizer_group,
            self._param_name_to_op_name_dict,
            module_bitwidth_dict,
            max_candidate,
            candidate,
            running_bit_ops,
        )
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
        # Apply exception rules logic to enforce a valid quantizer configuration
        self._sim._apply_exception_rules()  # pylint: disable = protected-access

        # Recompute quantizer encodings
        self._sim.compute_encodings(
            self.algo_params.forward_pass_callback,
            self.algo_params.forward_pass_callback_args,
        )
        # Compute new accuracy score
        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase2)
        return eval_score

    def _reduce_mp_convert_ops(self):
        """
        Reduce mixed precision convert ops if enabled and supported
        """


class _GreedyMixedPrecisionFromDict(GreedyMixedPrecisionAlgo):
    def __init__(
        self,
        sim: QuantizationSimModel,
        mp_accuracy_results: Dict[
            Tuple[str, str], Tuple[Dict[str, Any], Dict[str, float]]
        ],
        eval_fn: Callable[[ort.InferenceSession], float],
        results_dir,
    ):
        self._encoding_dicts = {}
        self._score_dicts = {}
        for precision, (score_dict, enc_dict) in mp_accuracy_results.items():
            candidate = _precision_to_candidate(precision)
            self._encoding_dicts[candidate] = enc_dict
            self._score_dicts[candidate] = score_dict

        super().__init__(
            sim,
            [
                _precision_to_candidate(precision)
                for precision in mp_accuracy_results.keys()
            ],
            eval_callback_for_phase1=CallbackFunc(lambda _: RuntimeError()),
            eval_callback_for_phase2=CallbackFunc(
                lambda session, _: float(eval_fn(session))
            ),
            results_dir=results_dir,
            clean_start=True,
            forward_pass_callback=CallbackFunc(lambda *_: RuntimeError()),
        )

    def _create_and_save_accuracy_list(
        self, baseline_candidate: CANDIDATE_WITH_DTYPE
    ) -> ACCURACY_LIST:
        """
        Creates accuracy list from sensitivity dict info. Does not actually save the list.
        """
        # Skips phase 1 of AMP algorithm
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        index_of_quantizer_group = {}
        for index, quantizer_group in enumerate(self.quantizer_groups):
            index_of_quantizer_group[quantizer_group] = index

        # Tuple["QuantizerGroup", CANDIDATE_WITH_DTYPE, float, int]
        accuracy_list: ACCURACY_LIST = []

        for candidate, score_dict in self._score_dicts.items():
            if candidate == baseline_candidate:
                continue

            for quantizer_group in self.quantizer_groups:
                quantizer_group: QuantizerGroup = quantizer_group
                # All scores should be the same for a group since the same logic is used to
                # Create quantizer groups in phase one. Use min reduction in case this changes
                score = min(
                    score_dict[name]
                    for name in itertools.chain(
                        quantizer_group.activation_quantizers,
                        quantizer_group.parameter_quantizers,
                    )
                )
                bit_ops_reduction = self._find_bit_ops_reduction_for_acc_list(
                    quantizer_group, baseline_candidate, candidate
                )
                accuracy_list.append(
                    (quantizer_group, candidate, score, bit_ops_reduction)
                )

                accuracy_list = sort_accuracy_list(
                    accuracy_list, index_of_quantizer_group
                )

        return accuracy_list

    def _optimize_mp_profile_and_evaluate_model(self):
        # Apply exception rules
        self._sim._apply_exception_rules()  # pylint:disable = protected-access

        # Note: We do not have samples to re-calibrate the sim after applying exception rules
        #   Instead, must find a valid encoding for the layer with matching bitwidth
        for name, quantizer in self._sim.qc_quantize_op_dict.items():
            if quantizer.is_initialized() or not quantizer.enabled:
                continue
            self._load_valid_encoding(name)

        return self.evaluate_model(self.algo_params.eval_callback_for_phase2)

    def _get_best_candidate(self) -> Tuple[float, CANDIDATE_WITH_DTYPE]:
        """Gets best candidate from list of provided candidates"""

        # TODO: Remove this. Extra state that gets changed silently in super()._get_best_candidate
        for candidate in self.baseline_candidate_options:
            self._candidate_mapping_dict[candidate] = {
                qg: candidate for qg in self.quantizer_groups
            }

        highest_prec = sorted(
            self.baseline_candidate_options,
            key=lambda candidate: candidate_cost(*candidate),
        )[-1]
        self._set_all_quantizer_groups_to_candidate(highest_prec)
        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase2)
        return eval_score, highest_prec

    def _choose_lowest_from_candidates(self) -> Tuple[float, CANDIDATE_WITH_DTYPE]:
        """
        Choose the lowest bitwidth candidate among all the candidates.

        :return: Lowest bitwidth candidate and corresponding accuracy.
        """
        lowest_prec = sorted(
            self.baseline_candidate_options,
            key=lambda candidate: candidate_cost(*candidate),
        )[0]
        self._set_all_quantizer_groups_to_candidate(lowest_prec)
        eval_score = self.evaluate_model(self.algo_params.eval_callback_for_phase2)
        return eval_score, lowest_prec

    def _set_quantizer_group_to_candidate(
        self, quantizer_group: QuantizerGroup, candidate: CANDIDATE_WITH_DTYPE
    ):
        # pylint: disable = protected-access
        # Load cached encoding for the candidate
        for quantizer_name in (
            quantizer_group.activation_quantizers + quantizer_group.parameter_quantizers
        ):
            encoding = self._encoding_dicts[candidate][quantizer_name]
            self._sim.qc_quantize_op_dict[quantizer_name]._load_encodings_dict(encoding)

    def _load_valid_encoding(self, quantizer_name: str):
        # pylint: disable = protected-access
        if quantizer_name in self._sim.param_names:
            self._sim._compute_param_encodings(overwrite=False)
            return

        quantizer: QcQuantizeOp = self._sim.qc_quantize_op_dict[quantizer_name]

        for encoding_dict in self._encoding_dicts.values():
            enc = encoding_dict[quantizer_name]
            if (
                enc["bw"] == quantizer.bitwidth
                and enc["dtype"] == quantizer.data_type.name.upper()
            ):
                quantizer._load_encodings_dict(enc)
                return

        raise RuntimeError(f"No valid encoding found for quantizer: {enc}")

    def _set_all_quantizer_groups_to_candidate(self, candidate):
        """
        Sets all quantizer groups to bitwidth/dtype of candidate

        :param candidate: Bitwidth and dtype to set the quantizer groups to
        """
        for quantizer_group in self.quantizer_groups:
            self._set_quantizer_group_to_candidate(quantizer_group, candidate)


def _precision_to_candidate(precision: set[Tuple[str, str]]) -> CANDIDATE_WITH_DTYPE:
    return tuple(
        tuple(reversed(qtype.from_string(prec).to_legacy_repr()))
        for prec in reversed(precision.split(","))
    )
