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
# pylint: disable=too-many-lines, disable=protected-access

""" Implementation of AIMET AutoQuantBase and v1 AutoQuant """
import copy
import functools
import itertools
import os
from unittest.mock import patch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

from aimet_torch import utils
from aimet_torch._base.auto_quant import (
    AutoQuantBase,
    _EvalManager,
    _QuantSchemePair,
    _EvalSession,
    cache,
    _MixedPrecisionArgs,
    _MixedPrecisionResult,
    ParetoFrontType,
)
from aimet_torch.v1.adaround.adaround_weight import Adaround
from aimet_torch._base.adaround.adaround_weight import AdaroundParameters
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.utils import get_all_quantizers
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo, EvalCallbackFactory, _default_forward_fn

from aimet_common.defs import QuantScheme, CallbackFunc, QuantizationDataType
from aimet_common.utils import AimetLogger
from aimet_common.amp.utils import (
    create_sensitivity_plot,
    create_pareto_curve,
    CANDIDATE_WITH_DTYPE,
    AmpCandidate,
)


__all__ = [
    'AutoQuant',
    'AutoQuantWithAutoMixedPrecision',
]


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)


class AutoQuant(AutoQuantBase): # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    @staticmethod
    def _get_adaround():
        """ returns AdaRound """
        return Adaround

    @staticmethod
    def _get_quantsim(model, dummy_input, **kwargs):
        return QuantizationSimModel(model, dummy_input, **kwargs)

    def _configure_quantsim(self, # pylint: disable=too-many-arguments
                            sim,
                            output_bw,
                            output_quant_scheme,
                            output_percentile,
                            param_bw,
                            param_quant_scheme,
                            param_percentile,
                            encoding_path):

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        # Set input/output quantizers' quant schemes
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.quant_scheme = output_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    output_percentile is not None:
                quantizer.set_percentile_value(output_percentile)

        # Set param quantizers' quant schemes
        for quantizer in param_quantizers:
            quantizer.quant_scheme = param_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    param_percentile is not None:
                quantizer.set_percentile_value(param_percentile)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        # Disable input/output quantizers, using fp32 to simulate int32.
        if output_bw == 32:
            for quantizer in input_quantizers + output_quantizers:
                quantizer.enabled = False

        # Disable param quantizers, using fp32 to simulate int32.
        if param_bw == 32:
            for quantizer in param_quantizers:
                quantizer.enabled = False

    @staticmethod
    def _has_enabled_quantizers(sim):
        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)
        return any(quantizer.enabled for quantizer in param_quantizers +\
                                                      input_quantizers +\
                                                      output_quantizers)

    @staticmethod
    def _disable_activation_quantizers(sim):
        _, input_quantizers, output_quantizers = get_all_quantizers(sim.model)
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.enabled = False



# The number of samples to be used for performance evaluation and AMP.
# NOTE: None means "all".
DEFAULT_NUM_SAMPLES_FOR_AMP_PHASE_1 = EvalCallbackFactory._DEFAULT_SQNR_NUM_SAMPLES
DEFAULT_NUM_SAMPLES_FOR_AMP_PHASE_2 = None


class AutoQuantWithAutoMixedPrecision:
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    3) Adaround, and 4) Automatic Mixed Precision (if enabled).
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """
    def __init__( # pylint: disable=too-many-arguments, too-many-function-args
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            data_loader: DataLoader,
            eval_callback: Callable[[torch.nn.Module], float],
            param_bw: int = 8,
            output_bw: int = 8,
            quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            rounding_mode: str = 'nearest',
            config_file: str = None,
            results_dir: str = "/tmp",
            cache_id: str = None,
            strict_validation: bool = True,
            model_prepare_required: bool = True,) -> None:
        """
        :param model: Model to be quantized. Assumes model is on the correct device
        :param dummy_input: Dummy input for the model. Assumes that dummy_input is on the correct device
        :param data_loader: A collection that iterates over an unlabeled dataset, used for computing encodings
        :param eval_callback: Function that calculates the evaluation score
        :param param_bw: Parameter bitwidth
        :param output_bw: Output bitwidth
        :param quant_scheme: Quantization scheme
        :param rounding_mode: Rounding mode
        :param config_file: Path to configuration file for model quantizers
        :param results_dir: Directory to save the results of PTQ techniques
        :param cache_id: ID associated with cache results
        :param strict_validation: Flag set to True by default.hen False, AutoQuant will proceed with execution and handle errors internally if possible. This may produce unideal or unintuitive results.
        :param model_prepare_required: Flag set to True by default.If False, AutoQuant will skip model prepare block in the pipeline.
        """
        self._auto_quant_base = AutoQuant(model,
                                          dummy_input,
                                          data_loader,
                                          eval_callback,
                                          param_bw,
                                          output_bw,
                                          quant_scheme,
                                          rounding_mode,
                                          config_file,
                                          results_dir,
                                          cache_id,
                                          strict_validation,
                                          model_prepare_required)
        self._data_loader = data_loader
        self._amp_args = None

    def run_inference(self) -> Tuple[QuantizationSimModel, float]:
        '''
        Creates a quantization model and performs inference

        :return: QuantizationSimModel, model accuracy as float
        '''
        return self._auto_quant_base.run_inference()

    def optimize(self, allowed_accuracy_drop: float = 0.0)\
            -> Tuple[torch.nn.Module, float, str, ParetoFrontType]:
        """
        Integrate and apply post-training quantization techniques.

        :param allowed_accuracy_drop: Maximum allowed accuracy drop
        :return: Tuple of  (best model, eval score, encoding path, pareto front).
            Pareto front is None if AMP is not enabled or AutoQuant exits
            without performing AMP.
        """
        html_template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "auto_quant_diagnostics_template_with_amp.html",
        )
        with patch.object(_EvalManager, "HTML_TEMPLATE_FILE", html_template_file):
            result = self._auto_quant_base._optimize_helper(self._optimize_main,
                                                            allowed_accuracy_drop)
            return result["model"],\
                   result["accuracy"],\
                   result["encoding_path"],\
                   result["pareto_list"]

    def set_adaround_params(self, adaround_params: AdaroundParameters) -> None:
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `data_loader` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        return self._auto_quant_base.set_adaround_params(adaround_params)

    def set_export_params(self,
                          onnx_export_args: OnnxExportApiArgs = -1,
                          propagate_encodings: bool = None) -> None:
        """
        Set parameters for QuantizationSimModel.export.

        :param onnx_export_args: optional export argument with onnx specific overrides
                if not provide export via torchscript graph
        :param propagate_encodings: If True, encoding entries for intermediate ops
                (when one PyTorch ops results in multiple ONNX nodes) are filled with
                the same BW and data_type as the output tensor for that series of ops.
        """
        return self._auto_quant_base.set_export_params(onnx_export_args, propagate_encodings)

    def set_mixed_precision_params(
            self,
            candidates: List[CANDIDATE_WITH_DTYPE],
            num_samples_for_phase_1: Optional[int] = DEFAULT_NUM_SAMPLES_FOR_AMP_PHASE_1,
            forward_fn: Callable = _default_forward_fn,
            num_samples_for_phase_2: Optional[int] = DEFAULT_NUM_SAMPLES_FOR_AMP_PHASE_2,
    ) -> None:
        """
        Set mixed precision parameters.
        NOTE: Automatic mixed precision will NOT be enabled unless this method
        is explicitly called by the user.

        :param candidates: List of tuples of candidate bitwidths and datatypes.
        :param num_samples_for_phase_1: Number of samples to be used for performance
                evaluation in AMP phase 1.
        :param forward_fn: Function that runs forward pass and returns the output tensor.
                which will be used for SQNR compuatation in phase 1.
                This function is expected to take 1) a model and 2) a single batch
                yielded from the data loader, and return a single torch.Tensor object
                which represents the output of the model.
                The default forward function is roughly equivalent to
                ``lambda model, batch: model(batch)``
        :param num_samples_for_phase_2: Number of samples to be used for performance
                evaluation in AMP phase 2.
        """
        if len(candidates) < 2:
            raise ValueError(f"AMP requires at least two candidates. Got {len(candidates)}.")

        baseline_param_bw = self._auto_quant_base._quantsim_params["param_bw"]
        baseline_output_bw = self._auto_quant_base._quantsim_params["output_bw"]
        baseline_candidate = (
            (baseline_output_bw, QuantizationDataType.int),
            (baseline_param_bw, QuantizationDataType.int),
        )

        if baseline_candidate not in candidates:
            raise ValueError(
                f"AMP candidate must contain W{baseline_param_bw}A{baseline_output_bw}, "
                "which was passed to the constructor of AutoQuant as `param_bw` and `output_bw`."
            )

        for candidate in candidates:
            ((output_bw, output_dtype), (param_bw, param_dtype)) = candidate

            if output_dtype != param_dtype:
                raise ValueError(
                    "The data types of parameters and outputs should be the same. "
                    f"Got {output_dtype} output and {param_dtype} for parameter."
                )

            if output_dtype == QuantizationDataType.float:
                continue

            # The param/output_bw passed to the constructor of AutoQuant
            # must be the baseline-bitwidth candidate among all AMP candidates.
            if output_bw < baseline_output_bw or param_bw < baseline_param_bw:
                raise ValueError(
                    "All AMP candidates should be strictly superior to the baseline "
                    f"W{baseline_param_bw}A{baseline_output_bw}, which was passed "
                    "to the constructor of AutoQuant. Please make sure that all the INT candidates "
                    f"satisfy param_bw >= {baseline_param_bw} and output_bw >= {baseline_param_bw}."
                )

        factory = EvalCallbackFactory(self._data_loader, forward_fn=forward_fn)
        sqnr_eval_callback = factory.sqnr(num_samples_for_phase_1)

        candidates = [AmpCandidate(candidate) for candidate in set(candidates)]

        self._amp_args = _MixedPrecisionArgs(
            candidates=candidates,
            forward_pass_callback=CallbackFunc(self._auto_quant_base.forward_pass_callback, None),
            eval_callback_for_phase1=sqnr_eval_callback,
            eval_callback_for_phase2=CallbackFunc(self._auto_quant_base.eval_callback,
                                                  num_samples_for_phase_2),
        )

    def set_model_preparer_params(
            self,
            modules_to_exclude: List[torch.nn.Module] = None,
            concrete_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Set parameters for model preparer.

        :param modules_to_exclude: List of modules to exclude when tracing.
        :param concrete_args: Parameter for model preparer. Allows you to partially specialize
            your function, whether it's to remove control flow or data structures. If the
            model has control flow, torch.fx won't be able to trace the model. Check
            torch.fx.symbolic_trace API in detail.
        """
        return self._auto_quant_base.set_model_preparer_params(modules_to_exclude, concrete_args)

    def get_quant_scheme_candidates(self) -> Tuple[_QuantSchemePair, ...]:
        """
        Return the candidates for quant scheme search.
        During :meth:`~AutoQuant.optimize`, the candidate with the highest accuracy
        will be selected among them.

        :return: Candidates for quant scheme search
        """
        return self._auto_quant_base.get_quant_scheme_candidates()

    def set_quant_scheme_candidates(self, candidates: Tuple[_QuantSchemePair, ...]):
        """
        Set candidates for quant scheme search.
        During :meth:`~AutoQuant.optimize`, the candidate with the highest accuracy
        will be selected among them.

        :param candidates: Candidates for quant scheme search
        """
        return self._auto_quant_base.set_quant_scheme_candidates(candidates)

    @cache.mark("mixed_precision")
    def _apply_mixed_precision(
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            target_acc: float,
            amp_args: _MixedPrecisionArgs,
            results_dir: str,
            encoding_path: str = None,
    ) -> _MixedPrecisionResult:
        """
        Apply mixed-precision and return the highest accuracy.

        NOTE1: Input model is not mutated.
        NOTE2: Parameter `clean_start` is always set to True.

        :param model: Model to apply mixed precision.
        :param dummy_input: Dummy input to the model.
        :param target_acc: Minimum evaluation score required.
        :param encoding_path: Path to parameter encodings file.
        :param results_dir: Directory to save the results of AdaRound and mixed precision.
        :return: MixedPrecisionAlgo object.
        """
        if not amp_args:
            raise RuntimeError

        sim = self._auto_quant_base._create_quantsim_and_encodings(model,
                                                                   encoding_path=encoding_path)

        algo = GreedyMixedPrecisionAlgo(
            sim,
            dummy_input,
            amp_args.candidates,
            amp_args.eval_callback_for_phase1,
            amp_args.eval_callback_for_phase2,
            results_dir=results_dir,
            clean_start=True,
            forward_pass_callback=amp_args.forward_pass_callback
        )

        # Find baseline accuracy and bw corresponding to baseline accuracy
        algo.set_baseline(fp32_accuracy=self._auto_quant_base._fp32_acc)
        allowed_accuracy_drop = algo.fp32_accuracy - target_acc

        algo.run(allowed_accuracy_drop)

        sensitivity_plot = None
        if algo.accuracy_list is not None:
            # Visualize quantizer group sensitivity
            sensitivity_plot = create_sensitivity_plot(algo.accuracy_list,
                                                       algo.baseline_candidate,
                                                       algo.fp32_accuracy)
        pareto_plot = None
        if algo.pareto_list is not None:
            # Create pareto list curve
            pareto_plot = create_pareto_curve(algo.pareto_list)

        return _MixedPrecisionResult(algo.pareto_list,
                                     algo._sim,
                                     algo._final_eval_score,
                                     sensitivity_plot,
                                     pareto_plot)

    def _optimize_main(self, fp32_model: torch.nn.Module, target_acc: float) -> Dict[str, Any]:
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :return: The best ptq result as a dictionary.
        """
        # pylint: disable=broad-except, too-many-locals, too-many-statements, too-many-branches

        if self._amp_args:
            candidates = copy.copy(self._amp_args.candidates)
        else:
            candidates = []

        eval_manager = self._auto_quant_base.eval_manager
        dummy_input = self._auto_quant_base.dummy_input
        results_dir = self._auto_quant_base.results_dir
        strict_validation = eval_manager._strict_validation

        sess = eval_manager.session("")
        _multiconfig_adaround_fn = _adaround_wrapper(self._auto_quant_base._apply_adaround,
                                                     self._auto_quant_base,
                                                     candidates,
                                                     target_acc,
                                                     sess.eval)
        sess_eval_fn = _EvalSession.eval
        def eval_fn(_, model, param_bw=None, output_bw=None, **kwargs):
            if param_bw == 32:
                # For W32 evaluation, use the highest output bitwidth
                # among all the AMP candidates
                output_bitwidths = [
                    output_bw for (output_bw, output_dtype), _ in candidates
                    if output_dtype == QuantizationDataType.int
                ]
                output_bitwidths.append(self._auto_quant_base._quantsim_params["output_bw"])
                output_bw = max(output_bitwidths)
            return sess_eval_fn(_, model, param_bw=param_bw, output_bw=output_bw, **kwargs)

        with patch.object(self._auto_quant_base, "_apply_adaround", _multiconfig_adaround_fn),\
                patch.object(_EvalSession, "eval", eval_fn):
            try:
                result = self._auto_quant_base._optimize_main(fp32_model, target_acc)

                # Automatic Mixed Precision
                result["pareto_list"] = None

                # An empty `result` dict means AutoQuant early-exited
                # because W32 eval score didn't meet the target accuracy.
                # In this case, do not proceed to AMP and exit immediately.
                if result["model"] is None and\
                        result["accuracy"] is None and\
                        result["encoding_path"] is None and\
                        result["applied_techniques"] is None:
                    return result

                if result["accuracy"] >= target_acc or not self._amp_args:
                    return result

                if len(candidates) < 2:
                    _logger.info(
                        "After Adaround, we have only one Adarond-compatible candidate left for AMP (W%dA%d). "
                        "Return without proceeding to AMP", candidates[0].param_bw, candidates[0].output_bw
                    )
                    return result

                model = result["model"]
                applied_techniques = result["applied_techniques"]
                # Freeze weight encoding to adaround weight encoding
                encoding_path = result["encoding_path"] if "adaround" in applied_techniques else None
            except Exception:
                if strict_validation:
                    raise
                result = {}
                model = fp32_model
                applied_techniques = []
                encoding_path = None

            amp_args = copy.copy(self._amp_args)
            if amp_args:
                amp_args.candidates = candidates

        with eval_manager.session("Automatic Mixed Precision", ptq=True) as sess:
            amp_result = self._apply_mixed_precision(
                model, dummy_input, target_acc, amp_args, results_dir, encoding_path=encoding_path
            )
            result["pareto_list"] = amp_result.pareto_list

            if amp_result.sensitivity_plot is not None:
                sess.diagnostics.add(amp_result.sensitivity_plot)

            if amp_result.pareto_plot is not None:
                sess.diagnostics.add(amp_result.pareto_plot)

            sess.set_ptq_result(sim=amp_result.sim, acc=amp_result.final_eval_score,
                                applied_techniques=[*applied_techniques, "automatic_mixed_precision"])

        best_result = eval_manager.get_best_ptq_result()
        if best_result:
            if "automatic_mixed_precision" not in best_result.applied_techniques:
                sess.result["effective"] = False
            if best_result.accuracy >= target_acc:
                sess.result["target_satisfied"] = True
            result.update(best_result.as_dict())
            return result

        raise RuntimeError("None of batchnorm folding, CLE, or Adaround "
                           "has been finished successfully.")

def _adaround_wrapper(apply_adaround_fn: Callable,
                      auto_quant: AutoQuantBase,
                      amp_candidates: List[AmpCandidate],
                      target_acc: float,
                      eval_fn: Callable):
    @functools.wraps(apply_adaround_fn)
    def _apply_adaround_wrapper(*args, **kwargs): # pylint: disable=too-many-locals
        # If AMP candidates are empty (i.e. AMP is disabled),
        # perform normal (single-round) adaround.
        if not amp_candidates:
            return apply_adaround_fn(*args, **kwargs)

        def apply_adaround(param_bw: int):
            _logger.info("Running Adaround with W%d", param_bw)

            orig_param_bw = auto_quant._quantsim_params["param_bw"]
            try:
                auto_quant._quantsim_params["param_bw"] = param_bw
                return apply_adaround_fn(*args, **kwargs)
            finally:
                auto_quant._quantsim_params["param_bw"] = orig_param_bw

        int_candidates = [
            candidate for candidate in amp_candidates
            if candidate.param_dtype == QuantizationDataType.int
        ]
        sorted_int_candidates = sorted(int_candidates,
                                       key=lambda candidate: (candidate.param_bw, candidate.output_bw))
        # Run Adaround with the lowest-bitwidth candidate
        lowest_candidate = sorted_int_candidates[0]
        model, encoding_path = apply_adaround(param_bw=lowest_candidate.param_bw)

        # If the lowest candidate is the only INT candidate, return immediately
        if len(sorted_int_candidates) == 1:
            return model, encoding_path

        eval_score = eval_fn(model,
                             param_bw=lowest_candidate.param_bw,
                             output_bw=lowest_candidate.output_bw,
                             encoding_path=encoding_path)
        _logger.info("W%dA%d eval score after Adaround: %f",
                     lowest_candidate.param_bw,
                     lowest_candidate.output_bw,
                     eval_score)

        # If the lowest candidate satisfy the target accuracy, return immediately
        if eval_score >= target_acc:
            return model, encoding_path

        # If the lowest candidate fails to meet the target accuracy,
        # discard the lowest candidate, apply Adaround to the second-lowest candidate,
        # and use it as the baseline for AMP.
        second_lowest_candidate = sorted_int_candidates[1]

        if second_lowest_candidate.param_bw != lowest_candidate.param_bw:
            model = None
            model, encoding_path = apply_adaround(param_bw=second_lowest_candidate.param_bw)
            eval_score = eval_fn(model,
                                 param_bw=second_lowest_candidate.param_bw,
                                 output_bw=second_lowest_candidate.output_bw,
                                 encoding_path=encoding_path)
            _logger.info("W%dA%d eval score after Adaround: %f",
                         second_lowest_candidate.param_bw,
                         second_lowest_candidate.output_bw,
                         eval_score)

        # Only the candidates that are compatible with adaround can be used for AMP
        adaround_compatible_amp_candidates = [
            candidate for candidate in amp_candidates
            if candidate.param_bw == second_lowest_candidate.param_bw or\
                    candidate.param_dtype == QuantizationDataType.float
        ]

        # Fill in AMP candidates with Adaround-compatible candidates only
        amp_candidates.clear()
        amp_candidates.extend(adaround_compatible_amp_candidates)

        return model, encoding_path

    return _apply_adaround_wrapper
