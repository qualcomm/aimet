# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Quant Analyzer"""
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_common.quant_analyzer import export_per_layer_sensitivity_analysis_plot, save_json, \
    create_and_export_min_max_ranges_plot, export_stats_histogram_plot, export_per_layer_mse_plot
from aimet_common.utils import CallbackFunc, AimetLogger, Spinner
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.graphsearchtuils import GraphSearchUtils
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.utils.quantizer_utils import get_enabled_activation_quantizers, enable_disable_quantizers, \
    get_enabled_param_quantizers

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def _sort_quant_wrappers_based_on_occurrence(sim: QuantizationSimModel) -> Dict[str, QcQuantizeWrapper]:
    """
    Sort quant wrappers based on occurrence for given quantsim model.

    :param sim: Quantsim model.
    :return: Ordered dictionary which maps wrapped layer name to quant wrapper.
    """
    sorted_quant_wrappers_dict = OrderedDict()
    for wrapper in sim.model.layers:
        if not isinstance(wrapper, QcQuantizeWrapper):
            continue

        sorted_quant_wrappers_dict[wrapper.original_layer.name] = wrapper

    return sorted_quant_wrappers_dict


def _get_enabled_quantizers(sorted_quant_wrappers: Dict[str, QcQuantizeWrapper]) -> \
        Dict[QcQuantizeWrapper, List[TensorQuantizer]]:
    """
    For given sorted quant wrappers dict, get enabled quantizers.

    :param sorted_quant_wrappers: Dictionary containing quant wrappers sorted based on occurrence.
    :return: Dictionary which maps a quant wrapper to a list of enabled quantizers in it.
    """
    enabled_quant_wrappers = defaultdict(list)

    for quant_wrapper in sorted_quant_wrappers.values():
        for quantizer in quant_wrapper.param_quantizers:
            if quantizer.is_enabled():
                enabled_quant_wrappers[quant_wrapper].append(quantizer)

        for quantizer in quant_wrapper.output_quantizers:
            if quantizer.is_enabled():
                enabled_quant_wrappers[quant_wrapper].append(quantizer)

        for quantizer in quant_wrapper.input_quantizers:
            if quantizer.is_enabled():
                enabled_quant_wrappers[quant_wrapper].append(quantizer)

    return enabled_quant_wrappers


def _get_output_of_intermediate_layer(model: tf.keras.Model,
                                      input_tensor: tf.Tensor,
                                      layer_index: int) -> tf.Tensor:
    """
    Return output tensor from model extracted up to target intermediate layer

    :param model: tf.keras.Model
    :param input_tensor: Input tensor to feed
    :param layer_index: Index of layer
    :return: Output tensor from intermediate layer
    """
    layer_output = model.get_layer(index=layer_index).output
    extracted_model = tf.keras.Model(inputs=model.inputs, outputs=layer_output)

    return extracted_model(input_tensor)


class QuantAnalyzer:
    """
    QuantAnalyzer tool provides

    1) model sensitivity to weight and activation quantization
    2) per layer sensitivity analysis
    3) per layer encoding (min - max range)
    4) per PDF analysis and
    5) per layer MSE analysis
    """

    def __init__(self,
                 model: tf.keras.Model,
                 forward_pass_callback: CallbackFunc,
                 eval_callback: CallbackFunc):
        """
        :param model: FP32 model to analyze for quantization.
        :param forward_pass_callback: A callback function for model calibration that simply runs
                forward passes on the model to compute encoding (delta/offset). This
                callback function should use representative data and should be subset of
                entire train/validation dataset (~1000 images/samples).
        :param eval_callback: A callback function for model evaluation that determines model
                performance. This callback function is expected to return scalar value
                representing the model performance evaluated against entire test/evaluation dataset.
        """
        if not isinstance(forward_pass_callback, CallbackFunc):
            raise ValueError('forward_pass_callback and its argument(s) are not encapsulated by CallbackFunc class.')
        if not isinstance(eval_callback, CallbackFunc):
            raise ValueError('eval_callback and its argument(s) are not encapsulated by CallbackFunc class.')

        self._model = model
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback
        self._unlabeled_dataset = None
        self._num_batches = None

    # pylint: disable=unused-argument, no-self-use
    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                rounding_mode: str = "nearest",
                default_param_bw: int = 8,
                default_output_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/"):
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quant wrappers,
            3) export per layer encodings min - max ranges,
            4) export per layer statistics histogram (PDF) when quant scheme is TF-Enhanced,
            5) per layer MSE analysis

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        sim = self._create_quantsim_and_encodings(quant_scheme,
                                                  rounding_mode,
                                                  default_param_bw,
                                                  default_output_bw,
                                                  config_file)

        # Check model sensitivity to weight and activation quantization individually.
        self.check_model_sensitivity_to_quantization(sim, default_param_bw, default_output_bw)

        # Perform per layer analysis by enabling each quant wrapper (OPTION-1).
        self.perform_per_layer_analysis_by_enabling_quant_wrappers(sim, results_dir)

        # Perform per layer analysis by disabling each quant wrapper (OPTION-2).
        self.perform_per_layer_analysis_by_disabling_quant_wrappers(sim, results_dir)

        # Export encoding min-max range.
        self.export_per_layer_encoding_min_max_range(sim, results_dir)

        # Export PDF of statistics
        if quant_scheme == QuantScheme.post_training_tf_enhanced:
            self.export_per_layer_stats_histogram(sim, results_dir)

        # Export per layer MSE loss between fp32 and quantized output activations.
        if self._unlabeled_dataset and self._num_batches:
            self.export_per_layer_mse_loss(sim, results_dir)

    def _create_quantsim_and_encodings(self,
                                       quant_scheme: QuantScheme,
                                       rounding_mode: str,
                                       default_param_bw: int,
                                       default_output_bw: int,
                                       config_file: str) -> QuantizationSimModel:
        """
        Create Quantsim and compute encodings.

        :param quant_scheme: Quantization scheme.
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :return: Quantsim model.
        """
        _, self._model = fold_all_batch_norms(self._model) # pylint: disable=attribute-defined-outside-init
        sim = QuantizationSimModel(self._model,
                                   quant_scheme=quant_scheme,
                                   rounding_mode=rounding_mode,
                                   default_output_bw=default_output_bw,
                                   default_param_bw=default_param_bw,
                                   config_file=config_file)

        sim.compute_encodings(forward_pass_callback=self._forward_pass_callback.func,
                              forward_pass_callback_args=self._forward_pass_callback.args)

        return sim

    def check_model_sensitivity_to_quantization(self,
                                                sim: QuantizationSimModel,
                                                default_param_bw: int,
                                                default_output_bw: int):
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        fp32_eval_score = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", default_output_bw,
                     act_quantized_eval_score)

    def _eval_model(self, model: tf.keras.Model) -> float:
        """
        Evaluate the model performance.
        :param model: tf.keras.Model to be evaluated
        :return: Scalar value representing model performance
        """
        return self._eval_callback.func(model, self._eval_callback.args)

    def _eval_weight_quantized_model(self, sim: QuantizationSimModel) -> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = get_enabled_activation_quantizers(sim)
        enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        enable_disable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(self, sim: QuantizationSimModel) -> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = get_enabled_param_quantizers(sim)
        enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        enable_disable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score

    def perform_per_layer_analysis_by_enabling_quant_wrappers(self,
                                                              sim: QuantizationSimModel,
                                                              results_dir: str) -> Dict[str, float]:
        """
        NOTE: Option 1

        1. All quant wrappers' parameters and activations quantizers are disabled.
        2. For every quant wrappers, based on occurrence:
            i. Each quant wrapper's parameters and activations quantizers are enabled as per JSON config file and set to bit-width specified.
            ii. Measure and record eval score on subset of dataset.
            iii. Disable enabled quantizers in step i.
        3. Returns dictionary containing quant wrapper name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer-wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-1:\nAll the quant wrappers are disabled.\n"
                     "Starting per-layer analysis by enabling quant wrappers as per config file.")

        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=True,
                                                                      enabled_before=True,
                                                                      enabled_after=False)
        export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_layer_quant_enabled")
        save_json(layer_wise_eval_score_dict,
                  results_dir,
                  title="per_layer_quant_enabled.json")
        _logger.info("Exported per-layer quant analysis (enabled) plot.")
        return layer_wise_eval_score_dict

    def perform_per_layer_analysis_by_disabling_quant_wrappers(self,
                                                               sim: QuantizationSimModel,
                                                               results_dir: str) -> Dict[str, float]:
        """
        NOTE: Option 2

        1. All quant wrappers' parameters and activations quantizers are enabled as per JSON config file and set to bit-width specified.
        2. For every quant wrappers, based on occurrence:
            i. Each quant wrapper's parameters and activations quantizers are disabled.
            ii. Measure and record eval score on subset of dataset.
            iii. Enable disabled quantizers in step i.
        3. Returns dictionary containing quant wrapper name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-2:\nAll the quant wrappers are enabled as per config file.\n"
                     "Starting per-layer analysis by disabling quant wrappers.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=False,
                                                                      enabled_before=False,
                                                                      enabled_after=True)
        export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_layer_quant_disabled")
        save_json(layer_wise_eval_score_dict,
                  results_dir,
                  title="per_layer_quant_disabled.json")
        _logger.info("Exported per-layer quant analysis (disabled) plot.")
        return layer_wise_eval_score_dict

    def _perform_per_layer_analysis(self,
                                    sim: QuantizationSimModel,
                                    disable_all_quantizers: bool,
                                    enabled_before: bool,
                                    enabled_after: bool) -> Dict[str, float]:
        """
        Helper function for perform_per_layer_analysis_by_enabling_quant_wrappers() and
        perform_per_layer_analysis_by_disabling_quant_wrappers()

        :param sim: Quantsim model.
        :param disable_all_quantizers: Flag to disable all the quantizers before per-layer analysis.
        :param enabled_before: Flag to set enabled for quantizers before computing encodings.
        :param enabled_after: Flag to set enabled for quantizers after computing encodings.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score.
        """
        # Sorted quant wrappers based on occurrence.
        # maps wrapped module name to a quant wrapper.
        sorted_quant_wrappers = _sort_quant_wrappers_based_on_occurrence(sim)

        # quant wrappers and it's enabled quantizers.
        # maps quant wrapper to a list of enabled quantizers in it.
        enabled_quant_wrappers = _get_enabled_quantizers(sorted_quant_wrappers)

        if disable_all_quantizers:
            for enabled_quantizers in enabled_quant_wrappers.values():
                enable_disable_quantizers(enabled_quantizers, enabled=False)

        eval_score_dict = {}
        for name, quant_wrapper in sorted_quant_wrappers.items():
            if quant_wrapper in enabled_quant_wrappers:
                enabled_quantizers = enabled_quant_wrappers[quant_wrapper]
                enable_disable_quantizers(enabled_quantizers, enabled=enabled_before)

                # Record eval score.
                eval_score_dict[name] = self._eval_model(sim.model)
                _logger.debug("For layer: %s, the eval score is: %f", name, eval_score_dict[name])

                enable_disable_quantizers(enabled_quantizers, enabled=enabled_after)

        if disable_all_quantizers:
            for enabled_quantizers in enabled_quant_wrappers.values():
                enable_disable_quantizers(enabled_quantizers, enabled=True)

        return eval_score_dict

    # pylint: disable=no-self-use
    def export_per_layer_encoding_min_max_range(self,
                                                sim: QuantizationSimModel,
                                                results_dir: str) -> Tuple[Dict, Dict]:
        """
        Export encoding min and max range for all weights and activations. results_dir should have
        html files in following format.

        -results_dir
            -activations.html
            -weights.html

        If per channel quantization(PCQ) is enabled then,

        -results_dir
            -activations.html
            -{wrapped_module_name}_{param_name}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise min-max range for weights and activations.
        """
        min_max_ranges_dir = os.path.join(results_dir, "min_max_ranges")

        min_max_range_for_activations_dict = {}
        min_max_range_for_weights_dict = {}
        for quant_wrapper in sim.quant_wrappers():
            wrapped_layer_name = quant_wrapper.original_layer.name

            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                if quantizer.is_enabled():
                    name = f"{wrapped_layer_name}_input_{index}"
                    min_max_range_for_activations_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)

            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                if quantizer.is_enabled():
                    name = f"{wrapped_layer_name}_output_{index}"
                    min_max_range_for_activations_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)

            for quantizer in quant_wrapper.param_quantizers:
                if quantizer.is_enabled():
                    # Keras parameter name usually contains slash (/) and it can cause incorrect file path when saving
                    #   Replace slash (/) with dash (-) to avoid it
                    quantizer_name = quantizer.name.replace("/", "-")
                    name = f"{wrapped_layer_name}_{quantizer_name}"

                    if isinstance(quantizer.encoding, List): # per-channel
                        per_channel_encodings = {}
                        for index, encoding in enumerate(quantizer.encoding):
                            per_channel_encodings[f"{name}_{index}"] = (encoding.min, encoding.max)
                        min_max_range_for_weights_dict[name] = per_channel_encodings
                    else: # per-tensor
                        min_max_range_for_weights_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)

        create_and_export_min_max_ranges_plot(min_max_range_for_weights_dict,
                                              min_max_ranges_dir,
                                              title="weights")
        create_and_export_min_max_ranges_plot(min_max_range_for_activations_dict,
                                              min_max_ranges_dir,
                                              title="activations")
        save_json(min_max_range_for_weights_dict, min_max_ranges_dir, title="weights.json")
        save_json(min_max_range_for_activations_dict, min_max_ranges_dir, title="activations.json")
        _logger.info("Exported per layer encodings min-max ranges plot(s).")
        return min_max_range_for_weights_dict, min_max_range_for_activations_dict

    def export_per_layer_stats_histogram(self, sim: QuantizationSimModel, results_dir: str) -> None:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Export histogram that represents a PDF of collected statistics by a quantizer for every
        quant wrapper. After invoking this API, results_dir should have html files in following
        format for every quantizers of quant wrappers.

        -results_dir
            -activations_pdf
                name_{input/output}_{index}.html
            -weights_pdf
                -name
                    param_name_{channel_index}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        """
        weights_pdf_dir = os.path.join(results_dir, "weights_pdf")
        activations_pdf_dir = os.path.join(results_dir, "activations_pdf")

        for quant_wrapper in sim.quant_wrappers():
            wrapped_layer_name = quant_wrapper.original_layer.name

            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                if quantizer.encoding:
                    self._create_and_export_stats_histogram_plot(quantizer, activations_pdf_dir,
                                                                 title=f"{wrapped_layer_name}_input_q{index}")

            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                if quantizer.encoding:
                    self._create_and_export_stats_histogram_plot(quantizer, activations_pdf_dir,
                                                                 title=f"{wrapped_layer_name}_output_q{index}")

            for quantizer in quant_wrapper.param_quantizers:
                if quantizer.encoding:
                    # Keras parameter name usually contains slash (/) and it can cause incorrect file path when saving
                    #   Replace slash (/) with dash (-) to avoid it
                    param_name = quantizer.name.replace("/", "-")
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 os.path.join(weights_pdf_dir, wrapped_layer_name),
                                                                 title=f"{wrapped_layer_name}_{param_name}")
        _logger.info("Exported per layer stats histogram plot(s).")

    @staticmethod
    def _create_and_export_stats_histogram_plot(quantizer: TensorQuantizer,
                                                results_dir: str,
                                                title: str) -> None:
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        histograms = quantizer.get_stats_histogram()
        encodings = quantizer.encoding
        if not isinstance(encodings, List):
            encodings = [encodings]

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            export_stats_histogram_plot(histogram, encoding, results_dir, title=f"{title}_{index}")

    def export_per_layer_mse_loss(self,
                                  sim: QuantizationSimModel,
                                  results_dir: str) -> Dict[str, float]:
        """
        NOTE: Need to pass same model input data through both fp32 and quantsim model to
        tap output activations of each layer.

        Export MSE loss between fp32 and quantized output activations for each layer.
        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return layer wise MSE loss. dict[layer_name] = MSE loss.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        mse_loss_dict = {}
        with Spinner("Calculating per-layer MSE loss"):
            for index, layer in enumerate(self._model.layers):
                if isinstance(layer, tf.keras.layers.InputLayer) or \
                        GraphSearchUtils.is_folded_batch_normalization(layer):
                    continue

                loss = self._compute_mse_loss(sim, index)
                mse_loss_dict[layer.name] = loss

        export_per_layer_mse_plot(mse_loss_dict,
                                  results_dir,
                                  title="per_layer_mse_loss")
        save_json(mse_loss_dict, results_dir, title="per_layer_mse_loss.json")
        _logger.info("Exported per layer MSE loss plot.")
        return mse_loss_dict

    def _compute_mse_loss(self,
                          sim: QuantizationSimModel,
                          index: int) -> float:
        """
        Compute MSE loss between fp32 and quantized output activations for each batch, add for
        all the batches and return averaged mse loss.

        :param sim: Quantsim model.
        :param index: Index of layer
        :return: MSE loss between fp32 and quantized output activations.
        """
        loss = 0.0
        total = 0
        mse = tf.keras.losses.MeanSquaredError()
        for tensor in self._unlabeled_dataset.take(self._num_batches):
            quantized_output = _get_output_of_intermediate_layer(sim.model, tensor, index)
            fp32_output = _get_output_of_intermediate_layer(self._model, tensor, index)

            loss += mse(quantized_output, fp32_output).numpy()
            total += tensor.shape[0]

        return loss / total

    def enable_per_layer_mse_loss(self, unlabeled_dataset: tf.data.Dataset, num_batches: int) -> None:
        """
        Enable per layer MSE loss analysis.

        :param unlabeled_dataset: tf.data.Dataset provided as input to the model
            and used to calculate mse loss
        :param num_batches: Maximum number of batches to be used for MSE loss calculation
        """
        self._unlabeled_dataset = unlabeled_dataset
        self._num_batches = num_batches
