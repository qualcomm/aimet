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
"""Mixed precision inference"""

from typing import Any, Callable, Union, Tuple, List, Dict, Iterable
import tempfile
from tqdm import tqdm

import onnxruntime as ort

from aimet_common.defs import qtype, int8, int16
from aimet_common.utils import AimetLogger
from aimet_common.amp.utils import (
    visualize_quantizer_group_sensitivity,
    visualize_pareto_curve,
    CANDIDATE_WITH_DTYPE,
    AMPSearchAlgo,
)
from aimet_onnx.utils import disable_quantizers
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.mixed_precision_algo import (
    GreedyMixedPrecisionAlgo,
    _GreedyMixedPrecisionFromDict,
)
from aimet_onnx.amp.quantizer_groups import QuantizerGroup, find_quantizer_group

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


# (param_type, activation_type)
Precision = Tuple[qtype, qtype]

w8a8 = (int8, int8)
w8a16 = (int8, int16)
w16a16 = (int16, int16)

# Dictionary of Precision: (Encoding dict, Sensitivity dict)
_MPSensitivityResults = Dict[Precision, Tuple[Dict[str, Dict], Dict[str, float]]]


def analyze_mixed_precision_sensitivity(
    sim: QuantizationSimModel,
    precisions: List[Precision],
    eval_fn: Callable[[ort.InferenceSession], float],
    calibration_input: Union[Callable, Iterable],
) -> _MPSensitivityResults:
    """
    Runs per-layer sensitivity analysis on sim for each of the specified precisions. The result should be passed
    to :func:`apply_amp` to optimize model precisions.

    .. warning::
        The contents of the output dictionary is subject to change between versions and should only be used as
        input to :func:`apply_amp`.

    Args:
        sim: QuantizationSimModel to analyze
        precisions: List of (param_type, activation_type) tuples to analyze
        eval_fn: Function which takes in an InferenceSession and returns an evaluation score (higher being better)
        calibration_input: Callable or iterable to be passed to sim.compute_encodings() for calibration

    Returns:
        Dictionary containing mixed precision sensitivity results
    """
    # TODO: Restore sim state after running this
    # pylint: disable=protected-access
    _, quantizer_groups = find_quantizer_group(sim)

    results = {}
    for precision in precisions:
        logger.info("Analyzing sensitivity for precision: %s", precision)
        _set_precision(sim, *precision)

        # Note: For consistency with legacy API, compute activation encodings w/out param quantization
        with disable_quantizers(sim, sim.param_names):
            sim.compute_encodings(calibration_input)
        sim._compute_param_encodings()

        sens_dict = _analyze_group_sensitivities(sim, quantizer_groups, eval_fn)

        encoding_list = sim._get_encodings(sim.qc_quantize_op_dict.keys(), "1.0.0")
        encoding_dict = {enc.pop("name"): enc for enc in encoding_list}

        results[",".join(str(qt) for qt in precision)] = (sens_dict, encoding_dict)

    return results


def apply_amp(
    sim: QuantizationSimModel,
    sensitivity_dict: _MPSensitivityResults,
    acceptance_fn: Callable[[ort.InferenceSession], bool],
):
    """
    Applies automatic mixed precision algorithm to optimize QuantizationSimModel bitwidth configuration.

    Args:
        sim: QuantizationSimModel to optimize
        sensitivity_dict: The set of mixed precision sensitivity results returned by :func:`analyze_mixed_precision_sensitivity`
        acceptance_fn: Callable which returns True if the input session meets the target task performance
    """
    with tempfile.TemporaryDirectory() as tmp:
        mixed_precision_algo = _GreedyMixedPrecisionFromDict(
            sim, sensitivity_dict, acceptance_fn, tmp
        )
        mixed_precision_algo.run(0.5)


def _analyze_group_sensitivities(
    sim: QuantizationSimModel,
    quantizer_groups: List[QuantizerGroup],
    eval_fn: Callable[[ort.InferenceSession], float],
) -> Dict[str, float]:
    """
    Performs group-wise sensitivity analysis for all quantizer groups, returning sensitivity as a flattened dict of
    tensor names to group score.
    """
    quantizer_sensitivities = {}
    with disable_quantizers(sim, sim.qc_quantize_op_dict.keys()):
        for group in tqdm(quantizer_groups):
            quantizer_names = group.activation_quantizers + group.parameter_quantizers

            # Enable group quantizers
            for name in quantizer_names:
                sim.qc_quantize_op_dict[name].enabled = True

            group_sens = eval_fn(sim.session)

            for name in quantizer_names:
                # Add to sensitivity dictionary
                quantizer_sensitivities[name] = group_sens
                # Disable the quantizer
                sim.qc_quantize_op_dict[name].enabled = False

    return quantizer_sensitivities


def _set_precision(
    sim: QuantizationSimModel, param_type: qtype, activation_type: qtype
):
    """
    Sets all quantizers to the specified param_type, activation_type
    """
    param_type = (
        qtype.from_string(param_type) if isinstance(param_type, str) else param_type
    )
    activation_type = (
        qtype.from_string(activation_type)
        if isinstance(activation_type, str)
        else activation_type
    )
    param_dtype, param_bw = param_type.to_legacy_repr()
    for name in sim.param_names:
        quantizer = sim.qc_quantize_op_dict.get(name)
        if quantizer and quantizer.enabled:
            quantizer.set_bitwidth(param_bw)
            quantizer.data_type = param_dtype

    act_dtype, act_bw = activation_type.to_legacy_repr()
    for name in sim.activation_names:
        quantizer = sim.qc_quantize_op_dict.get(name)
        if quantizer and quantizer.enabled:
            quantizer.set_bitwidth(act_bw)
            quantizer.data_type = act_dtype

    sim._apply_exception_rules()  # pylint: disable=protected-access


# pylint: disable=too-many-arguments
def choose_mixed_precision(
    sim: QuantizationSimModel,
    candidates: List[CANDIDATE_WITH_DTYPE],
    eval_callback_for_phase1: Callable[[ort.InferenceSession], float],
    eval_callback_for_phase2: Callable[[ort.InferenceSession], float],
    allowed_accuracy_drop: Union[None, float],
    results_dir: str,
    clean_start: bool,
    forward_pass_callback: Callable[[ort.InferenceSession], Any],
    use_all_amp_candidates: bool = False,
    phase1_optimize: bool = True,
    amp_search_algo: AMPSearchAlgo = AMPSearchAlgo.Binary,
) -> Union[List[Tuple[int, float, QuantizerGroup, int]], None]:
    """
    High-level API to perform in place Mixed Precision evaluation on the given sim model. A pareto list is created and
    a curve for Accuracy vs BitOps is saved under the results directory

    :param sim: Quantized sim model
    :param candidates: List of tuples for all possible bitwidth values for activations and parameters
                    Suppose the possible combinations are-
                    ((Activation bitwidth - 8, Activation data type - int), (Parameter bitwidth - 16, parameter data type - int))
                    ((Activation bitwidth - 16, Activation data type - float), (Parameter bitwidth - 16, parameter data type - float))
                    candidates will be [((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                        ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
    :param eval_callback_for_phase1: Callable object used to measure sensitivity of each
                                 quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                 module. Therefore, a user might want to run the phase 1 with a smaller dataset
    :param eval_callback_for_phase2: Callale object used to get accuracy of quantized model
                                 for phase 2 calculations. The phase 2 involves finding pareto front curve
    :param allowed_accuracy_drop: Maximum allowed drop in accuracy from FP32 baseline. The pareto front curve is plotted only till the point where the allowable
                                  accuracy drop is met. To get a complete plot for picking points on the curve, the user
                                  can set the allowable accuracy drop to None.
    :param results_dir: Path to save results and cache intermediate results
    :param clean_start: If true, any cached information from previous runs will be deleted prior to starting the
                        mixed-precision analysis. If false, prior cached information will be used if applicable. Note
                        it is the user's responsibility to set this flag to true if anything in the model or
                        quantization parameters changes compared to the previous run.
    :param forward_pass_callback: Callable object used to compute quantization encodings
    :param use_all_amp_candidates: Using the “supported_kernels” field in the config file (under defaults
                    and op_type sections), a list of supported candidates can be specified. All the AMP candidates
                    which are passed through the “candidates” field may not be supported based on the data passed
                    through “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP
                    algorithm will ignore the "supported_kernels" in the config file and continue to use all candidates.
    :phase1_optimize: If user set this parameter to false then phase1 default logic will be executed else optimized logic will be executed.
    :param amp_search_algo: A valid value from the Enum AMPSearchAlgo. Defines the search algorithm to be used for
                            the phase 2 of AMP.

    :return: Pareto front list containing information including Bitops, QuantizerGroup candidates and
             corresponding eval scores. The Pareto front list can be used for plotting a pareto front curve which
             provides information regarding how bit ops vary w.r.t. accuracy. If the allowable accuracy drop is set to
             100% then a user can use the pareto front curve to pick points and re-run,
             None if we early exit the mixed precision algorithm.
    """
    mixed_precision_algo = GreedyMixedPrecisionAlgo(
        sim,
        candidates,
        eval_callback_for_phase1,
        eval_callback_for_phase2,
        results_dir,
        clean_start,
        forward_pass_callback,
        use_all_amp_candidates,
        phase1_optimize,
    )
    mixed_precision_algo.run(allowed_accuracy_drop, amp_search_algo)

    if (
        mixed_precision_algo.accuracy_list is not None
        and mixed_precision_algo.pareto_list is not None
    ):
        # Print mixed precision stats
        logger.info(mixed_precision_algo)

        # Visualize quantizer group sensitivity
        visualize_quantizer_group_sensitivity(
            mixed_precision_algo.accuracy_list,
            mixed_precision_algo.baseline_candidate,
            mixed_precision_algo.fp32_accuracy,
            results_dir=results_dir,
        )
        # Create pareto list curve
        visualize_pareto_curve(mixed_precision_algo.pareto_list, results_dir)
        return mixed_precision_algo.pareto_list

    return None
