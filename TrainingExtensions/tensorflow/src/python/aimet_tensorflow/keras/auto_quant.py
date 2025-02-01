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

"""Automatic Post-Training Quantization"""
import os
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Tuple, Optional

import jinja2
from bokeh.resources import CDN
from tqdm import tqdm
import tensorflow as tf


from aimet_common.auto_quant import Diagnostics
from aimet_common.cache import Cache
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import validate_quantsim_inputs
from aimet_common.utils import AimetLogger, Spinner
from aimet_tensorflow.keras.adaround_weight import AdaroundParameters
from aimet_tensorflow.keras.adaround_weight import Adaround
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.cache import KerasModelSerializationProtocol
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model
from aimet_tensorflow.keras.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)

cache = Cache()

# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None


class AutoQuant: # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 allowed_accuracy_drop: float,
                 eval_callback: Callable[[tf.keras.Model, Optional[int]], float],
                 unlabeled_dataset: tf.data.Dataset,
                 default_param_bw: int = 8,
                 default_output_bw: int = 8,
                 default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 default_rounding_mode: str = "nearest",
                 default_config_file: str = None):
        """
        :param allowed_accuracy_drop: Maximum allowed accuracy drop.
        :param unlabeled_dataset: An unlabeled dataset for encoding computation.
                By default, this dataset will be also used for Adaround unless
                otherwise specified by `self.set_adaround_params`
        :param eval_callback: A function that maps model and the number samples
                to the evaluation score. This callback is expected to return a
                scalar value representing the model performance evaluated
                against exactly `N` samples, where `N` is the number of samples
                passed as the second argument of this callback.
                NOTE: If `N` is None, the model is expected to be evaluated against
                the whole evaluation dataset.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_config_file: Path to configuration file for model quantizers
        """
        if allowed_accuracy_drop < 0:
            raise ValueError(
                "`allowed_accuracy_drop` must be a positive value. Got {:.2f}"
                .format(allowed_accuracy_drop)
            )

        validate_quantsim_inputs(default_quant_scheme,
                                 default_rounding_mode,
                                 default_output_bw,
                                 default_param_bw)

        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.eval_callback = eval_callback
        self.default_param_bw = default_param_bw
        self.default_output_bw = default_output_bw
        self.default_quant_scheme = default_quant_scheme
        self.default_rounding_mode = default_rounding_mode
        self.default_config_file = default_config_file
        self._fp_32_acc = None

        def forward_pass_callback(model, _: Any = None):
            for input_data in tqdm(unlabeled_dataset):
                model(input_data)

        self.forward_pass_callback = forward_pass_callback
        self._unlabeled_dataset = unlabeled_dataset
        self.adaround_params = AdaroundParameters(unlabeled_dataset,
                                                  len(unlabeled_dataset))

    def apply(self,
              fp32_model: tf.keras.Model,
              results_dir: str = "/tmp",
              cache_id: str = None,
              custom_objects: Dict = None) -> Tuple[tf.keras.Model, float, str]:
        """
        Apply post-training quantization techniques.

        :param fp32_model: Model to apply PTQ techniques.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :param custom_objects: Custom objects of model.
        :return: Tuple of (best model, eval score, encoding path).
        """
        result = self._apply_helper(self._auto_quant_main,
                                    fp32_model,
                                    results_dir,
                                    cache_id, custom_objects)

        return result["model"], result["accuracy"], result["encoding_path"]

    def _apply_helper(self,
                      auto_quant_main_fn: Callable,
                      fp32_model: tf.keras.Model,
                      results_dir: str = "/tmp",
                      cache_id: str = None,
                      custom_objects: Dict = None) -> Dict[str, Any]:
        """

        :param auto_quant_main_fn: Function that implements the main logic of AutoQuant.
        :param fp32_model: Model to apply PTQ techniques.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :param custom_objects: Custom objects of model.
        :return: The best ptq result as a dictionary.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        if cache_id is None:
            cache_dir = None
        else:
            cache_dir = os.path.join(results_dir, ".auto_quant_cache", cache_id)

        with cache.enable(cache_dir):
            _logger.info("Starting AutoQuant")

            self._fp_32_acc = self._evaluate_model_performance(fp32_model)
            target_acc = self._fp_32_acc - self.allowed_accuracy_drop

            _logger.info("Target eval score: %f", target_acc)
            _logger.info("FP32 eval score (W32A32): %f", self._fp_32_acc)

            eval_manager = _EvalManager(
                quantsim_factory=self._create_quantsim_and_encodings,
                eval_func=self._evaluate_model_performance,
                results_dir=results_dir
            )

            ret = auto_quant_main_fn(fp32_model, target_acc,
                                     eval_manager, results_dir, custom_objects)

            acc = ret["accuracy"]
            encoding_path = ret["encoding_path"]
            applied_techniques = ", ".join(ret["applied_techniques"])
            _logger.info("Best eval score: %f. Encoding path: %s. Applied techniques %s",
                         acc, encoding_path, applied_techniques)

            if acc < target_acc:
                _logger.info(
                    "AutoQuant is unable to match the target accuracy. "
                    "Consider Quantization Aware Training."
                )

            eval_manager.export_diagnostics()

            return ret

    def _evaluate_model_performance(self, model: tf.keras.Model) -> float:
        """
        Evaluate the model performance

        :param model: Model to evaluate
        :return: Evaluation score
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def set_adaround_params(self, adaround_params: AdaroundParameters):
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `unlabeled_dataset_iterable` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        self.adaround_params = adaround_params

    def _create_quantsim_and_encodings(self,
                                       model: tf.keras.Model,
                                       quant_scheme: QuantScheme = None,
                                       rounding_mode: str = None,
                                       default_output_bw: int = None,
                                       default_param_bw: int = None,
                                       config_file: str = None,
                                       encoding_path: str = None) -> QuantizationSimModel:
        """

        :param model: Model to quantize
        :param quant_scheme: Quantization scheme. Defaults to self.default_quant_scheme.
        :param rounding_mode: Rounding mode. Defaults to self.default_rounding_mode.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
                                  Defaults to self.default_output_bw.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
                                 Defaults to self.default_param_bw.
        :param config_file: Path to configuration file for model quantizers.
                            Defaults to self.default_config_file.
        :param encoding_path: Path to parameter encodings file.
        :return: Quantsim model.
        """
        kwargs = {
            "quant_scheme": (quant_scheme or self.default_quant_scheme),
            "rounding_mode": (rounding_mode or self.default_rounding_mode),
            "default_output_bw": (default_output_bw or self.default_output_bw),
            "default_param_bw": (default_param_bw or self.default_param_bw),
            "config_file": (config_file or self.default_config_file),
        }
        sim = QuantizationSimModel(model, **kwargs)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        sim.compute_encodings(self.forward_pass_callback, None)

        return sim

    def _apply_batchnorm_folding(self, model: tf.keras.Model) -> Tuple[tf.keras.Model, List[Tuple]]:
        """
        Apply batchnorm folding
        Note: Input model is not mutated
        :param model: Model to apply batchnorm folding
        :return: Output model and folded pairs
        """
        original_weight = model.get_weights()
        model = tf.keras.models.clone_model(model)
        model.set_weights(original_weight)
        folded_pairs, model = fold_all_batch_norms(model)
        return model, folded_pairs

    @cache.mark("cle", KerasModelSerializationProtocol())
    def _apply_cross_layer_equalization(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Apply cross-layer equalization
        Note: Input model is not mutated
        :param model: Model to apply cross-layer-equalization
        :return: CLE applied model
        """
        return equalize_model(model)

    def _apply_adaround(self,
                        model: tf.keras.Model,
                        results_dir: str) -> Tuple[tf.keras.Model, str]:
        """
        Apply adaround

        NOTE: Input model is not mutated
        :param model: Model to apply adaround
        :param results_dir: Directory to save the results of AdaRound
        :return: Output model and the path to the parameter encoding file
        """
        filename_prefix = "adaround"
        adaround_encoding_path = os.path.join(results_dir,
                                              f"{filename_prefix}.encodings")

        _apply_adaround_cached = cache.mark("adaround", KerasModelSerializationProtocol())\
            (Adaround.apply_adaround)

        model = _apply_adaround_cached(model,
                                       self.adaround_params,
                                       path=results_dir,
                                       filename_prefix=filename_prefix,
                                       default_param_bw=self.default_param_bw,
                                       default_quant_scheme=self.default_quant_scheme,
                                       config_file=self.default_config_file)

        return model, adaround_encoding_path

    def _auto_quant_main(self,
                         fp32_model: tf.keras.Model,
                         target_acc: float,
                         eval_manager: "_EvalManager",
                         results_dir: str = "/tmp",
                         custom_objects: Dict = None) -> Dict[str, Any]:
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :param eval_manager: _Evalmanager object.
        :param results_dir: Directory to save the results.
        :param custom_objects: Custom objects of model.
        :return: The best ptq result as a dictionary.
        """
        with eval_manager.analysis_session("Weight Quantization Sensitivity") as sess:
            acc = sess.eval(fp32_model, default_output_bw=32)
            sess.diagnostics.add(
                f"Weight-quantized eval score (W{self.default_param_bw}A32): {acc:f}"
            )
            _logger.info("Weight-quantized eval score (W%dA32): %f", self.default_param_bw, acc)

        with eval_manager.analysis_session("Activation Quantization Sensitivity") as sess:
            acc = sess.eval(fp32_model, default_param_bw=32)
            sess.diagnostics.add(
                f"Activation-quantized eval score (W32A{self.default_output_bw}): {acc:f}"
            )
            _logger.info("Activation-quantized eval score (W32A%d): %f", self.default_output_bw, acc)

        # Batchnorm Folding
        with eval_manager.ptq_session("Batchnorm Folding") as sess:
            model, folded_pairs = self._apply_batchnorm_folding(fp32_model)
            for conv, bn in folded_pairs:
                sess.diagnostics.add(f"{conv} was merged with {bn}.")
            sess.set_ptq_result(model=model, applied_techniques=["batchnorm_folding"], custom_objects=custom_objects)

        best_result = eval_manager.get_best_ptq_result()
        if best_result.accuracy >= target_acc:
            return best_result.as_dict(custom_objects=custom_objects)

        # Cross-Layer Equalization
        with eval_manager.ptq_session("Cross-Layer Equalization") as sess:
            model = self._apply_cross_layer_equalization(fp32_model)
            sess.set_ptq_result(model=model, applied_techniques=["cross_layer_equalization"], custom_objects=custom_objects)

        best_result = eval_manager.get_best_ptq_result()
        if best_result.accuracy >= target_acc:
            return best_result.as_dict()

        # Adaround
        with eval_manager.ptq_session("AdaRound") as sess:
            model, encoding_path = self._apply_adaround(best_result.load_model(custom_objects=custom_objects),
                                                        results_dir)
            sess.set_ptq_result(model=model,
                                encoding_path=encoding_path,
                                applied_techniques=[*best_result.applied_techniques, "adaround"],
                                custom_objects=custom_objects)

        return eval_manager.get_best_ptq_result().as_dict()


@dataclass
class PtqResult:
    """
    Evaluation results
    :param model_path: Path to the serialized model.
    :param encoding_path: Path to the encoding file.
    :param accuracy: Accuracy of the model.
    :param applied_techniques: Applied ptq techniques.
    """
    model_path: str
    encoding_path: str
    accuracy: float
    applied_techniques: List[str]

    def load_model(self, custom_objects: Dict = None):
        """
        Load model
        :param custom_objects: Custom objects of model.
        :return: Loaded model
        """
        return tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)

    def as_dict(self, custom_objects: Dict = None):
        """
        Convert to dictionary
        :param custom_objects: Custom objects of model.
        """
        return {
            "model": self.load_model(custom_objects=custom_objects),
            "accuracy": self.accuracy,
            "encoding_path": self.encoding_path,
            "applied_techniques": self.applied_techniques,
        }


class _EvalManager:
    """
    Evaluation manager for AutoQuant.
    """

    def __init__(self,
                 quantsim_factory: Callable,
                 eval_func: Callable[[tf.keras.Model], float],
                 results_dir: str):
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._results_dir = results_dir

        os.makedirs(self._results_dir, exist_ok=True)

        self._all_sessions: List[_EvalSession] = []
        self._ptq_sessions: List[_PtqSession] = []

    def get_best_ptq_result(self) -> PtqResult:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far.
        """
        if not self._ptq_sessions:
            raise RuntimeError

        ptq_results = [sess.ptq_result for sess in self._ptq_sessions]
        return max(ptq_results, key=lambda ptq_result: ptq_result.accuracy)

    def analysis_session(self, title: str) -> "_EvalSession":
        """
        Return a session for analysis only.
        :param title: Title of the session.
        :return: Analysis session.
        """
        return self._get_session(title, _EvalSession)

    def ptq_session(self, title: str) -> "_PtqSession":
        """
        Return a session for analysis only.
        :param title: Title of the session.
        :return: PTQ session.
        """
        sess = self._get_session(title, _PtqSession)
        self._ptq_sessions.append(sess)
        return sess

    def _get_session(self, title: str, session_cls: type):
        """
        Session factory.
        :param title: Title of the session.
        :session_cls: Class of the session.
        :return: Session object.
        """
        session = session_cls(title,
                              self._quantsim_factory,
                              self._eval_func,
                              results_dir=os.path.join(self._results_dir, ".trace"))
        self._all_sessions.append(session)
        return session

    def export_diagnostics(self) -> str:
        """
        Export diagnostics in html format.
        :return: Diagnostics string in html format.
        """
        loader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        env = jinja2.Environment(loader=loader)
        template = env.get_template("auto_quant_diagnostics_template.html")

        if any(sess.diagnostics.contains_bokeh() for sess in self._all_sessions):
            head = CDN.render()
        else:
            head = ""

        body = {
            sess.title: sess.diagnostics
            for sess in self._all_sessions
            if not sess.diagnostics.is_empty()
        }

        html = template.render(head=head, body=body)
        filename = os.path.join(self._results_dir, "diagnostics.html")
        with open(filename, "w") as f:
            f.write(html)
        return html


class _EvalSession:
    """
    Evaluation session for AutoQuant.

    Each session object contains a title and diagnostics produced during the session.
    The collected diagnostics will be exported into a html file by _EvalManager.
    """
    def __init__(self,
                 title: str,
                 quantsim_factory: Callable,
                 eval_func: Callable[[tf.keras.Model], float],
                 results_dir: str):
        """
        :param title:
        :param quantsim_factory:
        :param eval_func:
        :param results_dir:
        """
        self._title = title
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._results_dir = results_dir
        self._spinner = None

        os.makedirs(self._results_dir, exist_ok=True)

        self._diagnostics = Diagnostics()

        # Map session title to file name.
        # e.g. title: "Cross-Layer Equalization" -> filename: "cross_layer_equalization"
        self._filename = self._title.lower().replace("-", " ")
        self._filename = "_".join(self._filename.split())

    def eval(self, model: tf.keras.Model, **kwargs):
        """
        Evaluate the model
        :param model: Model to evaluate.
        :param kwargs: Additional arguments to the quantsim factory.
        :return: Eval score
        """
        sim = self._quantsim_factory(model, **kwargs)
        acc = self._eval_func(sim.model)
        return acc

    @property
    def title(self):
        """Getter of self._title."""
        return self._title

    @property
    def diagnostics(self):
        """Getter of self._diagnostics."""
        return self._diagnostics

    def __enter__(self):
        self._spinner = Spinner(self._title)
        self._spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._spinner is not None:
            self._spinner.__exit__(exc_type, exc_val, exc_tb)


class _PtqSession(_EvalSession):
    """
    PTQ session.

    Each PTQ session object should call `set_ptq_result` exactly once
    inside a with-as block.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptq_result = None

    @property
    def ptq_result(self) -> PtqResult:
        """Getter of self._ptq_result."""
        if self._ptq_result is None:
            raise RuntimeError
        return self._ptq_result

    def set_ptq_result(self,
                       applied_techniques: List[str],
                       model: tf.keras.Model = None,
                       sim: QuantizationSimModel = None,
                       acc: float = None,
                       custom_objects: Dict = None,
                       **kwargs):
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block

        Exactly one among model and (sim, acc) pair should be specified
        1) If sim and acc is specified, save them as the result of this session
        2) If model is specified, evaluate the quantized accuracy of the model and save the result
        :param applied_techniques: List of applied technique names
        :param model: Result of PTQ
        :param sim: Result of PTQ. The quantization encoding (compute_encodings()) is
                    assumed to have been computed in advance
        :param acc: Eval score
        :param custom_objects: Custom objects of model.
        :param kwargs: Additional arguments to the quantsim factory
        """
        if sim is None:
            assert acc is None
            assert model is not None
            sim = self._quantsim_factory(model, **kwargs)
            acc = self._eval_func(sim.model)
        else:
            assert acc is not None
            assert model is None

        self._set_ptq_result(sim, acc, applied_techniques, custom_objects)

    def _set_ptq_result(self,
                        sim: QuantizationSimModel,
                        acc: float,
                        applied_techniques: List[str],
                        custom_objects: Dict = None) -> PtqResult:
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block
        :param sim: Result of PTQ. The quantization encoding (compute_encodings()) is
                    assumed to have been computed in advance
        :param acc: Eval score
        :param applied_techniques: List of applied technique names
        :param custom_objects: Custom objects of model.
        :return: PtqResult object
        """
        if self._ptq_result is not None:
            raise RuntimeError(
                "sess.eval() can be called only once per each _EvalSession instance."
            )
        model_path, encoding_path = self._export(sim, custom_objects)
        self._ptq_result = PtqResult(model_path=model_path,
                                     encoding_path=encoding_path,
                                     accuracy=acc,
                                     applied_techniques=applied_techniques)
        _logger.info(self._ptq_result)
        return self._ptq_result

    def _export(self, sim: QuantizationSimModel, custom_objects: Dict = None) -> Tuple[str, str]:
        """
        Export quantsim
        :param sim: QuantizationSimModel object to export
        :return: The paths where model and encoding are saved
        """
        sim.export(path=self._results_dir, filename_prefix=self._filename, custom_objects=custom_objects)
        model_path = os.path.join(self._results_dir, f"{self._filename}")
        encoding_path = os.path.join(self._results_dir, f"{self._filename}.encodings")
        _logger.info("The results of %s is saved in %s and %s.",
                     self._title, model_path, encoding_path)
        return model_path, encoding_path
