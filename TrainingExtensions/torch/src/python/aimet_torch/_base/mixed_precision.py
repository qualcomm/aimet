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

""" Mixed precision inference """

from typing import Union, Tuple, List
import torch

from aimet_common.utils import AimetLogger
from aimet_common.amp.utils import (
    visualize_quantizer_group_sensitivity,
    visualize_pareto_curve,
    CANDIDATE_WITH_DTYPE as TORCH_CANDIDATE,
    AMPSearchAlgo,
)
from aimet_common.defs import CallbackFunc
from .quantsim import _QuantizationSimModelInterface
from .amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
from .amp.quantizer_groups import QuantizerGroup

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


# pylint: disable=too-many-arguments
def choose_mixed_precision(sim: _QuantizationSimModelInterface, dummy_input: Union[torch.Tensor, Tuple],
                           candidates: List[TORCH_CANDIDATE], eval_callback_for_phase1: CallbackFunc,
                           eval_callback_for_phase2: CallbackFunc, allowed_accuracy_drop: Union[None, float],
                           results_dir: str, clean_start: bool, forward_pass_callback: CallbackFunc,
                           use_all_amp_candidates: bool = False, phase2_reverse: bool = False,
                           phase1_optimize: bool = True, amp_search_algo: AMPSearchAlgo = AMPSearchAlgo.Binary) -> \
        Union[List[Tuple[int, float, QuantizerGroup, int]], None]:
    """
    High-level API to perform in place Mixed Precision evaluation on the given sim model. A pareto list is created and
    a curve for Accuracy vs BitOps is saved under the results directory

    :param sim: Quantized sim model
    :param dummy_input: Dummy input to the model. If the model has more than one input, pass a tuple.
                        User is expected to place the tensors on the appropriate device.
    :param candidates: List of tuples for all possible bitwidth values for activations and parameters
                    Suppose the possible combinations are-
                    ((Activation bitwidth - 8, Activation data type - int), (Parameter bitwidth - 16, parameter data type - int))
                    ((Activation bitwidth - 16, Activation data type - float), (Parameter bitwidth - 16, parameter data type - float))
                    candidates will be [((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                        ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
    :param eval_callback_for_phase1: An object of CallbackFunc class which takes in Eval function (callable) and eval
                                     function parameters. This evaluation callback used to measure sensitivity of each
                                     quantizer group during phase 1. The phase 1 involves finding accuracy list/sensitivity of each
                                     module. Therefore, a user might want to run the phase 1 with a smaller dataset
    :param eval_callback_for_phase2: An object of CallbackFunc class which takes in Eval function (callable) and eval
                                     function parameters. Evaluation callback used to get accuracy of quantized model
                                     for phase 2 calculations. The phase 2 involves finding pareto front curve
    :param allowed_accuracy_drop: Maximum allowed drop in accuracy from FP32 baseline. The pareto front curve is plotted only till the point where the allowable
                                  accuracy drop is met. To get a complete plot for picking points on the curve, the user
                                  can set the allowable accuracy drop to None.
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
                    through “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP
                    algorithm will ignore the "supported_kernels" in the config file and continue to use all candidates.
    :param phase2_reverse: If user will set this parameter to True, then phase1 of amp algo, that is calculating accuracy list
                            will not be changed, whereas the phase2 algo of amp, which generate the pareto list will be changed. In phase2, algo will start,
                            model with all quantizer groups in least candidate, and one by one, it will put nodes in higher candidate till target accuracy does not meet.
    :param phase1_optimize: If user set this parameter to false then phase1 default logic will be executed else optimized logic will be executed.
    :param amp_search_algo: A valid value from the Enum AMPSearchAlgo. Defines the search algorithm to be used for
                            the phase 2 of AMP.

    :return: Pareto front list containing information including Bitops, QuantizerGroup candidates and
             corresponding eval scores. The Pareto front list can be used for plotting a pareto front curve which
             provides information regarding how bit ops vary w.r.t. accuracy. If the allowable accuracy drop is set to
             100% then a user can use the pareto front curve to pick points and re-run,
             None if we early exit the mixed precision algorithm.
    """
    mixed_precision_algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_for_phase1,
                                                    eval_callback_for_phase2, results_dir, clean_start,
                                                    forward_pass_callback, use_all_amp_candidates, phase2_reverse, phase1_optimize)
    mixed_precision_algo.run(allowed_accuracy_drop, amp_search_algo)

    if mixed_precision_algo.accuracy_list is not None and mixed_precision_algo.pareto_list is not None:
        # Print mixed precision stats
        logger.info(mixed_precision_algo)

        # Visualize quantizer group sensitivity
        visualize_quantizer_group_sensitivity(mixed_precision_algo.accuracy_list,
                                              mixed_precision_algo.baseline_candidate,
                                              mixed_precision_algo.fp32_accuracy,
                                              results_dir=results_dir)
        # Create pareto list curve
        visualize_pareto_curve(mixed_precision_algo.pareto_list, results_dir)
        return mixed_precision_algo.pareto_list

    return None
