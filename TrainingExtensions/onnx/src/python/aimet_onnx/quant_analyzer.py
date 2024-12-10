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

""" Quant Analyzer """

import os
import re
from typing import Union, Tuple, Dict, List, Iterable
import copy
from collections import defaultdict

import numpy as np
from onnx import ModelProto
import onnxruntime as ort
from onnxruntime.quantization.onnx_model import ONNXModel
from sklearn.metrics import mean_squared_error

from aimet_common.utils import AimetLogger, CallbackFunc
from aimet_common.defs import QuantScheme
from aimet_common.quant_analyzer import save_json, export_per_layer_sensitivity_analysis_plot, \
    create_and_export_min_max_ranges_plot, export_per_layer_mse_plot, export_stats_histogram_plot

from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.adaround.activation_sampler import ModuleData
from aimet_onnx import utils
from aimet_onnx.meta.operations import Op

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.QuantAnalyzer)


class QuantAnalyzer:
    """
    QuantAnalyzer provides following utilities:

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per layer quantizer historgram analysis and
     5) per layer MSE analysis
    """
    def __init__(self,
                 model: Union[ModelProto, ONNXModel],
                 dummy_input: Dict[str, np.ndarray],
                 forward_pass_callback: CallbackFunc,
                 eval_callback: CallbackFunc,
                 ):
        """
        :param model: FP32 model to analyze for quantization.
        :param dummy_input: Dummy input to model.
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

        self._onnx_model = model
        if not isinstance(self._onnx_model, ONNXModel):
            self._onnx_model = ONNXModel(self._onnx_model)
        self._dummy_input = dummy_input
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback
        self._unlabeled_dataset_iterable = None
        self._num_batches = None

    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                default_param_bw: int = 8,
                default_activation_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/",
                ):
        """
        Analyzes model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quantizers,
            3) export per layer encodings min - max ranges,
            4) export per layer quantizer stats histogram,
            5) per layer MSE analysis

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_activation_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        sim = self.create_quantsim_and_encodings(quant_scheme,
                                                 default_param_bw,
                                                 default_activation_bw,
                                                 config_file)

        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Check model sensitivity to weight and activation quantization individually.
        self.check_model_sensitivity_to_quantization(sim)

        # Perform per layer analysis by enabling its quantizers (OPTION-1).
        self.perform_per_layer_analysis_by_enabling_quantizers(sim, results_dir)

        # Perform per layer analysis by disabling its quantizers (OPTION-2).
        self.perform_per_layer_analysis_by_disabling_quantizers(sim, results_dir)

        # Export encoding min-max range.
        self.export_per_layer_encoding_min_max_range(sim, results_dir)

        # Export PDF of statistics.
        if quant_scheme == QuantScheme.post_training_tf_enhanced:
            self.export_per_layer_stats_histogram(sim, results_dir)

        # Export per layer MSE loss between fp32 and quantized output activations.
        if self._unlabeled_dataset_iterable:
            self.export_per_layer_mse_loss(sim, results_dir)

    def create_quantsim_and_encodings(self, quant_scheme: QuantScheme, default_param_bw: int,
                                      default_activation_bw: int, config_file: str) \
            -> QuantizationSimModel:
        """
        Creates quantsim object and computes encodings.

        :param quant_scheme: Quantization scheme.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_activation_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :return: Quantsim object.
        """
        _ = fold_all_batch_norms_to_weight(self._onnx_model)
        kwargs = dict(
            quant_scheme=quant_scheme,
            default_activation_bw=default_activation_bw,
            default_param_bw=default_param_bw,
            config_file=config_file,
        )
        sim = QuantizationSimModel(copy.deepcopy(self._onnx_model), self._dummy_input, **kwargs)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
        return sim

    def check_model_sensitivity_to_quantization(self,
                                                sim: QuantizationSimModel,
                                                ) -> Tuple[float, float, float]:
        """
        Performs model sensitivity analysis to weight and activation quantization individually.

        :param sim: Quantsim model.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        # pylint: disable=protected-access
        fp32_eval_score = self._eval_model(self._onnx_model.model)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", sim._default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", sim._default_activation_bw,
                     act_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def _eval_model(self, model):
        """
        Evaluate the model performance.

        :param model: ONNX model to be evaluated.
        :return: Scaler value representing model performance.
        """
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(
            path_or_bytes=model.SerializeToString(),
            providers=providers,
        )
        return self._eval_callback.func(session, self._eval_callback.args)

    def _eval_weight_quantized_model(self, sim: QuantizationSimModel)-> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = self._get_enabled_activation_quantizers(sim)
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_callback.func(sim.session, self._eval_callback.args)
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(self, sim: QuantizationSimModel)-> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = self._get_enabled_param_quantizers(sim)
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_callback.func(sim.session, self._eval_callback.args)
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score

    @staticmethod
    def _get_enabled_param_quantizers(sim: QuantizationSimModel) -> List[QcQuantizeOp]:
        """
        For given quantsim model, get all enabled param quantizers.
        :param sim: Quantsim model.
        :return: List of enabled param quantizers.
        """
        param_quantizers, _ = sim.get_all_quantizers()
        enabled_param_quantizers = [quantizer for quantizer in param_quantizers if quantizer.enabled]
        return enabled_param_quantizers

    @staticmethod
    def _get_enabled_activation_quantizers(sim: QuantizationSimModel) -> List[QcQuantizeOp]:
        """
        For given quantsim model, get all enabled activation quantizers.
        :param sim: Quantsim model.
        :return: List of enabled activation quantizers.
        """
        _, act_quantizers = sim.get_all_quantizers()
        enabled_activation_quantizers = [quantizer for quantizer in act_quantizers if quantizer.enabled]
        return enabled_activation_quantizers

    @staticmethod
    def _enable_disable_quantizers(quantizers: List[QcQuantizeOp], enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """
        for quantizer in quantizers:
            quantizer.enabled = enabled

    def perform_per_layer_analysis_by_enabling_quantizers(self,
                                                          sim: QuantizationSimModel,
                                                          results_dir: str,
                                                          ) -> Dict:
        """
        Performs layer-wise quantization sensitivity analysis by enabling its quantizers

        1. All parameter and activation quantizers are disabled.
        2. For every layer, based on occurrence:
              a. Each layer's parameters and activations quantizers are enabled as per JSON config file
                 and set to bit-width specified.
              b. Measure and record eval score on subset of dataset.
              c. Disable enabled quantizers in step a.
        3. Returns dictionary containing layer name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("OPTION-1: All the quantizers are disabled. "
                     "Starting per-layer analysis by enabling layer-specific quantizers as per config file.")
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

    def perform_per_layer_analysis_by_disabling_quantizers(self,
                                                           sim: QuantizationSimModel,
                                                           results_dir: str,
                                                           ) -> Dict:
        """
        Performs layer-wise quantization sensitivity analysis by disabling its quantizers

        1. All parameter and activation quantizers are enabled as per JSON config file
           and set to bit-width specified.
        2. For every layer, based on occurrence:
              a. Each layer's parameters and activations quantizers are disabled.
              b. Measure and record eval score on subset of dataset.
              c. Enable disabled quantizers in step a.
        3. Returns dictionary containing layer name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("OPTION-2: All the quantizers are enabled as per config file. "
                     "Starting per-layer analysis by disabling layer-specific quantizers.")
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
                                    enabled_after: bool,
                                    ) -> Dict:
        """
        Helper function for perform_per_layer_analysis_by_enabling_quantizers() and
        perform_per_layer_analysis_by_disabling_quantizers()

        :param sim: Quantsim model.
        :param disable_all_quantizers: Flag to disable all the quantizers before per-layer analysis.
        :param enabled_before: Flag to set enabled for quantizers before computing encodings.
        :param enabled_after: Flag to set enabled for quantizers after computing encodings.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score.
        """

        # Mapping of ops and their enabled quantizers
        op_to_quantizers_dict = defaultdict(list)
        for op in sim.connected_graph.ordered_ops:
            input_quantizers, output_quantizers, param_quantizers = self._get_op_quantizers(op, sim)
            if not input_quantizers and not output_quantizers and not param_quantizers:
                continue
            op_to_quantizers_dict[op.name_op].extend(input_quantizers)
            op_to_quantizers_dict[op.name_op].extend(output_quantizers)
            op_to_quantizers_dict[op.name_op].extend(list(param_quantizers.values()))

        if disable_all_quantizers:
            for enabled_quantizers in op_to_quantizers_dict.values():
                self._enable_disable_quantizers(enabled_quantizers, enabled=False)

        eval_score_dict = {}
        for op_name, enabled_quantizers in op_to_quantizers_dict.items():
            self._enable_disable_quantizers(enabled_quantizers, enabled=enabled_before)

            # Record eval score.
            eval_score_dict[op_name] = self._eval_callback.func(sim.session, self._eval_callback.args)
            _logger.debug("For layer: %s, the eval score is: %f", op_name, eval_score_dict[op_name])

            self._enable_disable_quantizers(enabled_quantizers, enabled=enabled_after)

        if disable_all_quantizers:
            for enabled_quantizers in op_to_quantizers_dict.values():
                self._enable_disable_quantizers(enabled_quantizers, enabled=True)

        return eval_score_dict

    @staticmethod
    def _get_op_quantizers(op: Op, sim: QuantizationSimModel) -> (List, List, Dict):
        """
        This function returns the enabled input, output and param quantizers of the given connected graph op.

        :param op: Connected Graph Op
        :param sim: QuantSim object
        :return: list of input quantizers, list of output quantizers and dictionary of param quantizers
        """
        input_quantizers, output_quantizers, param_quantizers = sim.get_op_quantizers(op)
        input_quantizers = [q for q in input_quantizers if q.enabled]
        output_quantizers = [q for q in output_quantizers if q.enabled]
        param_quantizers = {name: q for name, q in param_quantizers.items() if q.enabled}
        return input_quantizers, output_quantizers, param_quantizers

    # pylint: disable=no-self-use, too-many-branches, too-many-locals
    def export_per_layer_encoding_min_max_range(self, sim: QuantizationSimModel, results_dir: str) -> Tuple[Dict, Dict]:
        """
        Exports encoding min and max range for all weights and activations. results_dir has
        html files in following format.

        -results_dir
            -activations.html,
            -weights.html

        If per channel quantization(PCQ) is enabled then,

        -results_dir
            -activations.html,
            -{layer_name}_{param_name}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise min-max range for weights and activations.
        """
        min_max_ranges_dir = os.path.join(results_dir, "min_max_ranges")

        min_max_range_for_activations_dict = {}
        min_max_range_for_weights_dict = {}

        for op in sim.connected_graph.ordered_ops:
            input_quantizers, output_quantizers, param_quantizers = self._get_op_quantizers(op, sim)
            op_name = re.sub(r'\W+', '_', op.name_op)

            # Get input activations' encodings if starting op
            for index, quantizer in enumerate(input_quantizers):
                name = f"{op_name}_input_{index}"
                encodings = quantizer.get_encodings()
                min_max_range_for_activations_dict[name] = (encodings[0].min, encodings[0].max)

            # Get output activations' encodings
            for index, quantizer in enumerate(output_quantizers):
                name = f"{op_name}_output_{index}"
                encodings = quantizer.get_encodings()
                min_max_range_for_activations_dict[name] = (encodings[0].min, encodings[0].max)

            # Get parameters' encodings
            for param_name, quantizer in param_quantizers.items():
                name = re.sub(r'\W+', '_', f"{op_name}_{param_name}")
                encodings = quantizer.get_encodings()
                if len(encodings) > 1: # per-channel
                    per_channel_encodings = {}
                    for index, encoding in enumerate(encodings):
                        per_channel_encodings[f"{name}_{index}"] = (encoding.min, encoding.max)
                    min_max_range_for_weights_dict[name] = per_channel_encodings
                else: # per-tensor
                    min_max_range_for_weights_dict[name] = (encodings[0].min, encodings[0].max)

        create_and_export_min_max_ranges_plot(min_max_range_for_weights_dict, min_max_ranges_dir, title="weights")
        create_and_export_min_max_ranges_plot(min_max_range_for_activations_dict, min_max_ranges_dir, title="activations")

        save_json(min_max_range_for_weights_dict, min_max_ranges_dir, title="weights.json")
        save_json(min_max_range_for_activations_dict, min_max_ranges_dir, title="activations.json")

        _logger.info("Exported per layer encodings min-max ranges plot(s).")

        return min_max_range_for_weights_dict, min_max_range_for_activations_dict

    # pylint: disable=too-many-branches, too-many-locals
    def export_per_layer_stats_histogram(self,
                                         sim: QuantizationSimModel,
                                         results_dir: str,
                                         ):
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Exports histogram that represents a PDF of collected statistics by a quantizer.
        After invoking this API, results_dir should have html files in following
        format for every quantizers in the model.

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

        for op in sim.connected_graph.ordered_ops:
            input_quantizers, output_quantizers, param_quantizers = self._get_op_quantizers(op, sim)
            op_name = re.sub(r'\W+', '_', op.name_op)

            # Collect stats histogram of input activation quantizers
            for index, quantizer in enumerate(input_quantizers):
                self._create_and_export_stats_histogram_plot(quantizer,
                                                             activations_pdf_dir,
                                                             title=f"{op_name}_input_q{index}")

            # Collect stats histogram of output activation quantizers
            for index, quantizer in enumerate(output_quantizers):
                self._create_and_export_stats_histogram_plot(quantizer,
                                                             activations_pdf_dir,
                                                             title=f"{op_name}_output_q{index}")

            # Collect stats histogram of param quantizers
            for param_name, quantizer in param_quantizers.items():
                sanitized_param_name = re.sub(r'\W+', '_', param_name)
                self._create_and_export_stats_histogram_plot(quantizer,
                                                             os.path.join(weights_pdf_dir, op_name),
                                                             title=f"{op_name}_{sanitized_param_name}")

        _logger.info("Exported per layer stats histogram plot(s).")

    @staticmethod
    def _create_and_export_stats_histogram_plot(quantizer: QcQuantizeOp, results_dir: str, title: str):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        histograms = quantizer.get_stats_histogram()
        encodings = quantizer.get_encodings()

        if not isinstance(encodings, List):
            encodings = [encodings]

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            export_stats_histogram_plot(histogram, encoding, results_dir, title=f"{title}_{index}")

    def enable_per_layer_mse_loss(self, unlabeled_dataset_iterable: Iterable, num_batches: int):
        """
        Enables per layer MSE loss analysis.

        :param unlabeled_dataset_iterable: A collection (i.e. iterable with `__len__`)
                that iterates over an unlabeled dataset. The values yielded by this iterable are expected
                to be able to be passed directly to the model.
        :param num_batches: Number of batches. Approximately 256 samples/images are recommended,
                so if batch size of data loader is 64, then 4 number of batches leads to 256 samples/images.
        """
        if len(unlabeled_dataset_iterable) < num_batches:
            raise ValueError(f'Can not fetch {num_batches} batches from '
                             f'a data loader of length {len(unlabeled_dataset_iterable)}.')

        self._unlabeled_dataset_iterable = unlabeled_dataset_iterable
        self._num_batches = num_batches

    def export_per_layer_mse_loss(self, sim: QuantizationSimModel, results_dir: str) -> Dict:
        """
        Exports MSE loss between fp32 and quantized output activations for each layer.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise MSE loss. dict[layer_name] = MSE loss.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        mse_loss_dict = {}
        for op_node in self._onnx_model.nodes():
            if op_node.op_type == 'Constant':
                continue
            op_output = op_node.output[0]
            if op_output in sim.qc_quantize_op_dict:
                quantized_op_output = op_output + '_updated'
                loss = self._compute_mse_loss(op_output, quantized_op_output, self._onnx_model, sim.model)
                mse_loss_dict[op_node.name] = loss

        export_per_layer_mse_plot(mse_loss_dict,
                                  results_dir,
                                  title="per_layer_mse_loss")
        save_json(mse_loss_dict, results_dir, title="per_layer_mse_loss.json")
        _logger.info("Exported per layer MSE loss plot.")
        return mse_loss_dict

    # pylint: disable=too-many-locals
    def _compute_mse_loss(self, fp32_act_name: str, quantized_act_name: str,
                          fp32_model: ONNXModel, quantized_model: ONNXModel) -> float:
        """
        Compute MSE loss between fp32 and quantized output activations for each batch, add for
        all the batches and return averaged mse loss.

        :param fp32_act_name: module from the fp32_model.
        :param quantized_act_name: Corresponding quant wrapper from the QuantSim model.
        :param fp32_model: PyTorch model.
        :param sim: Quantsim model.
        :return: MSE loss between fp32 and quantized output activations.
        """

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']

        # output activations collector.
        orig_module_collector = ModuleData(fp32_model, fp32_act_name, providers)
        quant_module_collector = ModuleData(quantized_model, quantized_act_name, providers)

        total = 0
        loss = 0.0
        batch_index = 0
        for model_inputs in self._unlabeled_dataset_iterable:
            model_inputs = utils.create_input_dict(fp32_model.model, model_inputs)
            _, quantized_out_acts = quant_module_collector.collect_inp_out_data(model_inputs,
                                                                                collect_input=False,
                                                                                collect_output=True)
            _, fp32_out_acts = orig_module_collector.collect_inp_out_data(model_inputs,
                                                                          collect_input=False,
                                                                          collect_output=True)
            loss += mean_squared_error(fp32_out_acts[0].reshape(fp32_out_acts[0].shape[0], -1),
                                       quantized_out_acts[0].reshape(fp32_out_acts[0].shape[0], -1))
            total += fp32_out_acts[0].shape[0]
            batch_index += 1
            if batch_index == self._num_batches:
                break

        average_loss = loss/total
        return float(average_loss)
