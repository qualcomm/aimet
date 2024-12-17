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

""" Quant Analyzer base class """

import os
import contextlib
from collections import OrderedDict, defaultdict
from typing import Union, Tuple, Dict, List, Collection, Type, Generator
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

from aimet_common.quant_analyzer import save_json, export_per_layer_sensitivity_analysis_plot,\
    create_and_export_min_max_ranges_plot, export_per_layer_mse_plot
from aimet_common.utils import AimetLogger, CallbackFunc
from aimet_common.defs import QuantScheme
from aimet_torch import utils
from aimet_torch._base.quantsim import _QuantizationSimModelInterface

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.QuantAnalyzer)

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


class QuantAnalyzerBase(ABC):
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per PDF analysis and
     5) per layer MSE analysis
    """
    def __init__(self,
                 model: torch.nn.Module,
                 dummy_input: Union[torch.Tensor, Tuple],
                 forward_pass_callback: CallbackFunc,
                 eval_callback: CallbackFunc,
                 modules_to_ignore: List[torch.nn.Module] = None,
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
        :param modules_to_ignore: Excludes certain modules from being analyzed.
        """
        if not isinstance(forward_pass_callback, CallbackFunc):
            raise ValueError('forward_pass_callback and its argument(s) are not encapsulated by CallbackFunc class.')
        if not isinstance(eval_callback, CallbackFunc):
            raise ValueError('eval_callback and its argument(s) are not encapsulated by CallbackFunc class.')

        self._model = model
        self._dummy_input = dummy_input
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback
        self._unlabeled_dataset_iterable = None
        self._num_batches = None
        self._modules_to_ignore = modules_to_ignore

    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                default_param_bw: int = 8,
                default_output_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/",
                ):
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quant wrappers,
            3) export per layer encodings min - max ranges,
            4) export per layer statistics histogram (PDF) when quant scheme is TF-Enhanced,
            5) per layer MSE analysis

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        sim = self._create_quantsim_and_encodings(quant_scheme,
                                                  default_param_bw,
                                                  default_output_bw,
                                                  config_file)

        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Check model sensitivity to weight and activation quantization individually.
        self.check_model_sensitivity_to_quantization(sim)

        # Perform per layer analysis by enabling each quant wrapper (OPTION-1).
        self.perform_per_layer_analysis_by_enabling_quant_wrappers(sim, results_dir)

        # Perform per layer analysis by disabling each quant wrapper (OPTION-2).
        self.perform_per_layer_analysis_by_disabling_quant_wrappers(sim, results_dir)

        # Export encoding min-max range.
        self.export_per_layer_encoding_min_max_range(sim, results_dir)

        # Export PDF of statistics.
        if quant_scheme == QuantScheme.post_training_tf_enhanced:
            self.export_per_layer_stats_histogram(sim, results_dir)

        # Export per layer MSE loss between fp32 and quantized output activations.
        if self._unlabeled_dataset_iterable:
            self.export_per_layer_mse_loss(sim, results_dir)

    def enable_per_layer_mse_loss(self, unlabeled_dataset_iterable: Union[DataLoader, Collection], num_batches: int):
        """
        Enable per layer MSE loss analysis.

        :param unlabeled_dataset_iterable: A collection (i.e. iterable with `__len__`)
                that iterates over an unlabeled dataset. The values yielded by this iterable are expected
                to be able to be passed directly to the model.
        :param num_batches: Number of batches. Approximately 256 samples/images are recommended,
                so if batch size of data loader is 64, then 4 number of batches leads to 256 samples/images.
        """
        # TODO: Make per layer MSE loss analysis as part of top level API.
        if len(unlabeled_dataset_iterable) < num_batches:
            raise ValueError(f'Can not fetch {num_batches} batches from '
                             f'a data loader of length {len(unlabeled_dataset_iterable)}.')

        self._unlabeled_dataset_iterable = unlabeled_dataset_iterable
        self._num_batches = num_batches

    def _create_quantsim_and_encodings(self, quant_scheme: QuantScheme, default_param_bw: int,
                                       default_output_bw: int, config_file: str) \
            -> _QuantizationSimModelInterface:
        """
        Create Quantsim and compute encodings.

        :param quant_scheme: Quantization scheme.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :return: Quantsim model.
        """
        if isinstance(self._dummy_input, torch.Tensor):
            input_shape = tuple(self._dummy_input.shape)
        else:
            input_shape = [tuple(x.shape) for x in self._dummy_input]
        _ = self._fold_all_batch_norms(self._model, input_shape, dummy_input=self._dummy_input)

        kwargs = dict(
            quant_scheme=quant_scheme,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=config_file,
        )
        sim = self._get_quantsim_cls()(self._model, self._dummy_input, **kwargs)
        if self._modules_to_ignore:
            self._exclude_modules_from_quantization(self._model, sim, self._modules_to_ignore)

        self.patch_quantsim_to_store_histogram(sim)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
        return sim

    def _eval_weight_quantized_model(self, sim: _QuantizationSimModelInterface)-> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        with self._disable_activation_quantizers(sim):
            eval_score = self._eval_model(sim.model)
            return eval_score

    def _eval_activation_quantized_model(self, sim: _QuantizationSimModelInterface)-> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        with self._disable_param_quantizers(sim):
            eval_score = self._eval_model(sim.model)
            return eval_score

    def _eval_model(self, model: torch.nn.Module) -> float:
        """
        Evaluate the model performance.

        :param model: PyTorch model to be evaluated.
        :return: Scaler value representing model performance.
        """
        with utils.in_eval_mode(model), torch.no_grad():
            return self._eval_callback.func(model, self._eval_callback.args)

    def _sort_quant_wrappers_based_on_occurrence(self, sim: _QuantizationSimModelInterface) -> Dict:
        """
        Sort quant wrappers based on occurrence for given quantsim model.

        :param sim: Quantsim model.
        :return: Ordered dictionary which maps wrapped module name to quant wrapper.
        """
        def sorting_hook(quant_wrapper: torch.nn.Module, *_):
            """
            Hook-function to sort quant wrappers based on occurrence.

            :param quant_wrapper: Quant wrapper.
            :param _: Additional args.
            """
            quant_wrapper_name = module_to_name_dict[quant_wrapper]
            sorted_quant_wrappers_dict[quant_wrapper_name] = quant_wrapper

        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        sorted_quant_wrappers_dict = OrderedDict()
        utils.run_hook_for_layers_with_given_input(sim.model, self._dummy_input, sorting_hook,
                                                   module_type_for_attaching_hook=self._get_quant_wrapper_type(),
                                                   leaf_node_only=False)
        return sorted_quant_wrappers_dict

    @classmethod
    def _get_enabled_quantizers(cls, sorted_quant_wrappers: Dict)\
            -> Dict:
        """
        For given sorted quant wrappers dict, get enabled quantizers.

        :param sorted_quant_wrappers: Dictionary containing quant wrappers sorted based on occurrence.
        :return: Dictionary which maps a quant wrapper to a list of enabled quantizers in it.
        """
        enabled_quant_wrappers = defaultdict(list)

        for quant_wrapper in sorted_quant_wrappers.values():
            for quantizer in quant_wrapper.param_quantizers.values():
                if cls._is_quantizer_enabled(quantizer):
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)
            for quantizer in quant_wrapper.output_quantizers:
                if cls._is_quantizer_enabled(quantizer):
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)
            for quantizer in quant_wrapper.input_quantizers:
                if cls._is_quantizer_enabled(quantizer):
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)

        return enabled_quant_wrappers

    @classmethod
    def _get_enabled_param_quantizers(cls, sim: _QuantizationSimModelInterface) -> List:
        """
        For given quantsim model, get all enabled param quantizers.
        :param sim: Quantsim model.
        :return: List of enabled param quantizers.
        """
        enabled_param_quantizers = []
        for quant_wrapper in cls._get_quantized_modules(sim):
            for quantizer in quant_wrapper.param_quantizers.values():
                if cls._is_quantizer_enabled(quantizer):
                    enabled_param_quantizers.append(quantizer)

        return enabled_param_quantizers

    @classmethod
    def _get_enabled_activation_quantizers(cls, sim: _QuantizationSimModelInterface) -> List:
        """
        For given quantsim model, get all enabled activation quantizers.
        :param sim: Quantsim model.
        :return: List of enabled activation quantizers.
        """
        enabled_activation_quantizers = []
        for quant_wrapper in cls._get_quantized_modules(sim):
            for quantizer in quant_wrapper.input_quantizers:
                if cls._is_quantizer_enabled(quantizer):
                    enabled_activation_quantizers.append(quantizer)
            for quantizer in quant_wrapper.output_quantizers:
                if cls._is_quantizer_enabled(quantizer):
                    enabled_activation_quantizers.append(quantizer)

        return enabled_activation_quantizers

    def _perform_per_layer_analysis(self,
                                    sim: _QuantizationSimModelInterface,
                                    disable_all_quantizers: bool,
                                    enabled_before: bool,
                                    enabled_after: bool,
                                    ) -> Dict:
        """
        Helper function for perform_per_layer_analysis_by_enabling_quant_wrappers() and
        perform_per_layer_analysis_by_disabling_quant_wrappers()

        :param sim: Quantsim model.
        :param disable_all_quantizers: Flag to disable all the quantizers before per-layer analysis.
        :param enabled_before: Flag to set enabled for quantizers before computing encodings.
        :param enabled_after: Flag to set enabled for quantizers after computing encodings.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score.
        """
        # Validate input arguments
        assert (disable_all_quantizers, enabled_before, enabled_after) in \
            ((True, True, False), (False, False, True))

        # Sorted quant wrappers based on occurrence.
        # maps wrapped module name to a quant wrapper.
        sorted_quant_wrappers = self._sort_quant_wrappers_based_on_occurrence(sim)

        # quant wrappers and it's enabled quantizers.
        # maps quant wrapper to a list of enabled quantizers in it.
        enabled_quant_wrappers = self._get_enabled_quantizers(sorted_quant_wrappers)

        eval_score_dict = {}
        for name, quant_wrapper in sorted_quant_wrappers.items():
            if quant_wrapper not in enabled_quant_wrappers:
                continue

            with contextlib.ExitStack() as stack:
                if disable_all_quantizers and enabled_before:
                    # Disable all quantizers except quant_wrapper
                    for enabled_quant_wrapper in enabled_quant_wrappers.keys():
                        if enabled_quant_wrapper == quant_wrapper:
                            continue
                        stack.enter_context(self._disable_quant_wrapper(enabled_quant_wrapper))
                else:
                    # Disable only quant_wrapper
                    stack.enter_context(self._disable_quant_wrapper(quant_wrapper))

                # Record eval score.
                eval_score_dict[name] = self._eval_model(sim.model)
                _logger.debug("For layer: %s, the eval score is: %f", name, eval_score_dict[name])

        return eval_score_dict

    # pylint: disable=no-self-use
    @abstractmethod
    def _create_and_export_stats_histogram_plot(self,
                                                quantizer,
                                                results_dir: str,
                                                title: str,
                                                ):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """

    def check_model_sensitivity_to_quantization(self,
                                                sim: _QuantizationSimModelInterface,
                                                ) -> Tuple[float, float, float]:
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        # pylint: disable=protected-access
        fp32_eval_score = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", sim._default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", sim._default_output_bw,
                     act_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def perform_per_layer_analysis_by_enabling_quant_wrappers(self,
                                                              sim: _QuantizationSimModelInterface,
                                                              results_dir: str,
                                                              ) -> Dict:
        """
        NOTE: Option 1

        1. All quant wrappers' parameters and activations quantizers are disabled.
        2. Based on occurrence for every quant wrappers
            - Each quant wrapper's parameters and activations quantizers are enabled as per JSON config file and set to bit-width specified.
            - Measure and record eval score on subset of dataset.
            - Disable enabled quantizers in step 1.
        3. Returns dictionary containing quant wrapper name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
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
                                                               sim: _QuantizationSimModelInterface,
                                                               results_dir: str,
                                                               ) -> Dict:
        """
        NOTE: Option 2

        1. All quant wrappers' parameters and activations quantizers are enabled as per JSON config file and set to bit-width specified.
        2. Based on occurrence for every quant wrappers
            - Each quant wrapper's parameters and activations quantizers are disabled.
            - Measure and record eval score on subset of dataset.
            - Enable disabled quantizers in step 1.
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

    # pylint: disable=no-self-use
    def export_per_layer_encoding_min_max_range(self,
                                                sim: _QuantizationSimModelInterface,
                                                results_dir: str,
                                                ) -> Tuple[Dict, Dict]:
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
        # pylint: disable=too-many-locals
        min_max_ranges_dir = os.path.join(results_dir, "min_max_ranges")

        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        min_max_range_for_activations_dict = {}
        min_max_range_for_weights_dict = {}
        for quant_wrapper in self._get_quantized_modules(sim):
            wrapped_module_name = module_to_name_dict[quant_wrapper]
            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                if self._is_quantizer_enabled(quantizer):
                    name = f"{wrapped_module_name}_input_{index}"
                    encoding = self._get_quantizer_encodings(quantizer)[0]
                    min_max_range_for_activations_dict[name] = (encoding.min, encoding.max)
            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                if self._is_quantizer_enabled(quantizer):
                    name = f"{wrapped_module_name}_output_{index}"
                    encoding = self._get_quantizer_encodings(quantizer)[0]
                    min_max_range_for_activations_dict[name] = (encoding.min, encoding.max)
            for param_name, quantizer in quant_wrapper.param_quantizers.items():
                if self._is_quantizer_enabled(quantizer):
                    name = f"{wrapped_module_name}_{param_name}"
                    encodings = self._get_quantizer_encodings(quantizer)
                    if len(encodings) > 1: # per-channel
                        per_channel_encodings = {}
                        for index, encoding in enumerate(encodings):
                            per_channel_encodings[f"{name}_{index}"] = (encoding.min, encoding.max)
                        min_max_range_for_weights_dict[name] = per_channel_encodings
                    else: # per-tensor
                        min_max_range_for_weights_dict[name] = (encodings[0].min, encodings[0].max)

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

    def export_per_layer_stats_histogram(self,
                                         sim: _QuantizationSimModelInterface,
                                         results_dir: str,
                                         ):
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

        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        for quant_wrapper in self._get_quantized_modules(sim):
            wrapped_module_name = module_to_name_dict[quant_wrapper]
            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                if quantizer is not None and self._get_quantizer_encodings(quantizer):
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 activations_pdf_dir,
                                                                 title=f"{wrapped_module_name}_input_q{index}")
            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                if quantizer is not None and self._get_quantizer_encodings(quantizer):
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 activations_pdf_dir,
                                                                 title=f"{wrapped_module_name}_output_q{index}")
            for param_name, quantizer in quant_wrapper.param_quantizers.items():
                if quantizer is not None and self._get_quantizer_encodings(quantizer):
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 os.path.join(weights_pdf_dir, wrapped_module_name),
                                                                 title=f"{wrapped_module_name}_{param_name}")
        _logger.info("Exported per layer stats histogram plot(s).")

    def export_per_layer_mse_loss(self,
                                  sim: _QuantizationSimModelInterface,
                                  results_dir: str,
                                  ) -> Dict:
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

        name_to_quant_wrapper_dict = {}
        for name, module in sim.model.named_modules():
            name_to_quant_wrapper_dict[name] = module

        modules = utils.get_ordered_list_of_modules(self._model, self._dummy_input)
        mse_loss_dict = {}
        for name, module in modules:
            quant_wrapper = name_to_quant_wrapper_dict[name]
            loss = self._compute_mse_loss(module, quant_wrapper, self._model, sim)
            mse_loss_dict[name] = loss

        export_per_layer_mse_plot(mse_loss_dict,
                                  results_dir,
                                  title="per_layer_mse_loss")
        save_json(mse_loss_dict, results_dir, title="per_layer_mse_loss.json")
        _logger.info("Exported per layer MSE loss plot.")
        return mse_loss_dict

    def _compute_mse_loss(self, module: torch.nn.Module, quant_wrapper: torch.nn.Module,
                          fp32_model: torch.nn.Module, sim: _QuantizationSimModelInterface) -> float:
        """
        Compute MSE loss between fp32 and quantized output activations for each batch, add for
        all the batches and return averaged mse loss.

        :param module: module from the fp32_model.
        :param quant_wrapper: Corresponding quant wrapper from the QuantSim model.
        :param fp32_model: PyTorch model.
        :param sim: Quantsim model.
        :return: MSE loss between fp32 and quantized output activations.
        """
        # output activations collector.
        orig_module_collector = utils.ModuleData(fp32_model, module)
        quant_module_collector = utils.ModuleData(sim.model, quant_wrapper)

        total = 0
        loss = 0.0
        batch_index = 0
        for model_inputs in self._unlabeled_dataset_iterable:
            assert isinstance(model_inputs, (torch.Tensor, tuple, list))
            _, quantized_out_acts = quant_module_collector.collect_inp_out_data(model_inputs,
                                                                                collect_input=False,
                                                                                collect_output=True)
            _, fp32_out_acts = orig_module_collector.collect_inp_out_data(model_inputs,
                                                                          collect_input=False,
                                                                          collect_output=True)
            loss += torch.nn.functional.mse_loss(fp32_out_acts, quantized_out_acts).item()
            total += fp32_out_acts.size(0)
            batch_index += 1
            if batch_index == self._num_batches:
                break

        average_loss = loss/total
        return average_loss

    @staticmethod
    def _exclude_modules_from_quantization(model: torch.nn.Module, sim: _QuantizationSimModelInterface,
                                           modules_to_ignore: List[torch.nn.Module]):
        """
        For the modules in the modules_to_ignore, remove the corresponding quant wrappers.

        :param model: Original model.
        :param sim: Quantsim model.
        :param modules_to_ignore: The list of modules for which the quant wrappers are removed.
        """
        name_to_quant_wrapper_dict = {}
        for name, module in sim.model.named_modules():
            name_to_quant_wrapper_dict[name] = module

        module_to_name_dict = {}
        for name, module in model.named_modules():
            module_to_name_dict[module] = name

        quant_wrappers_to_ignore = []
        for module in modules_to_ignore:
            name = module_to_name_dict[module]
            quant_wrapper = name_to_quant_wrapper_dict[name]
            quant_wrappers_to_ignore.append(quant_wrapper)

        sim.exclude_layers_from_quantization(quant_wrappers_to_ignore)

    @staticmethod
    def patch_quantsim_to_store_histogram(_):
        """
        Placeholder function to prevent patching v1 quantsim
        """

    @staticmethod
    @abstractmethod
    def _get_quantsim_cls() -> Type[_QuantizationSimModelInterface]:
        ...

    @staticmethod
    @abstractmethod
    def _get_quant_wrapper_type() -> Tuple[Type]:
        ...

    @staticmethod
    @abstractmethod
    def _is_quantizer_enabled(quantizer):
        ...

    @staticmethod
    @abstractmethod
    def _get_quantizer_encodings(quantizer):
        ...

    @staticmethod
    @abstractmethod
    def _enable_disable_quantizers(quantizers, enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """

    @classmethod
    @abstractmethod
    def _disable_param_quantizers(cls, sim: _QuantizationSimModelInterface):
        ...

    @classmethod
    @abstractmethod
    def _disable_activation_quantizers(cls, sim: _QuantizationSimModelInterface):
        ...

    @staticmethod
    @abstractmethod
    def _disable_quant_wrapper(module):
        ...

    @staticmethod
    @abstractmethod
    def _get_quantized_modules(sim: _QuantizationSimModelInterface) -> Generator:
        ...

    @staticmethod
    @abstractmethod
    def _fold_all_batch_norms(*args, **kwargs):
        ...
