# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implementation for simulating models running on Quantized hardware """
import contextlib
import os
import io
import copy
from typing import Tuple, List, Union, Dict, Callable, Optional, Any
import torch

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.utils import deprecated

from aimet_torch.v1.qc_quantize_op import QcQuantizeStandAloneBase, QcQuantizeWrapper, QcQuantizeOpMode, \
    StaticGridQuantWrapper, LearnedGridQuantWrapper, NativeTorchQuantWrapper
from aimet_torch.v1.tensor_quantizer import initialize_learned_grid_quantizer_attributes
from aimet_torch.v1.qc_quantize_op import get_encoding_by_quantizer as _get_encoding_by_quantizer
from aimet_torch import utils
from aimet_torch.v1.utils import create_encoding_dict
from aimet_torch.onnx_utils import OnnxSaver, OnnxExportApiArgs
from aimet_torch.v1.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper
from aimet_torch.v1._builder import _V1LazyQuantizeWrapper
from aimet_torch._base.quantsim import (
    _QuantizationSimModelBase,
    _QuantizedModuleProtocol,
    unquantizable_modules,
    QuantParams,
    ExportableQuantModule,
    save_checkpoint,
    load_checkpoint,
    check_accumulator_overflow,
)

__all__ = [
    'QuantizationSimModel',
    'QuantParams',
    'ExportableQuantModule',
    'save_checkpoint',
    'load_checkpoint',
    'check_accumulator_overflow',
    'load_encodings_to_sim',
    'compute_encodings_for_sims',
]


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# If a torch module type is in this dictionary, call the corresponding quantized module constructor instead of wrapping
# it with QcQuantizeWrapper.
qc_quantize_modules_dict = {
    torch.nn.RNN: QcQuantizeRecurrent,
    torch.nn.LSTM: QcQuantizeRecurrent,
    torch.nn.GRU: QcQuantizeRecurrent
}


# Types of modules which cannot be quantized
quantized_modules = (
    QcQuantizeWrapper,
    QcQuantizeStandAloneBase,
    QcQuantizeRecurrent,
    _QuantizedModuleProtocol,
    LazyQuantizeWrapper,
)


class QuantizationSimModel(_QuantizationSimModelBase):
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-public-methods
    _quantized_modules = quantized_modules

    def _realize_quant_wrappers_in_model(self, model: torch.nn.Module):
        """
        Prepare QuantSim for compute encodings. Resets encodings for each quantizable layer and sets mode to Analysis.
        Realize quant wrappers using collected information in LazyQuantWrapper.

        :param model: model containing modules wrapped with LazyQuantWrapper
        """
        for module_name, module_ref in model.named_children():
            if isinstance(module_ref, LazyQuantizeWrapper):
                quantized_module = module_ref.realize()
                setattr(model, module_name, quantized_module)

            elif not utils.is_leaf_module(module_ref):
                self._realize_quant_wrappers_in_model(module_ref)

    def __str__(self):
        """
        Pretty-printed output indicating where in the model, quantizers have been activated
        :return:
        """

        def print_quantizer_state(stream, quantizer, prefix_string):
            if quantizer.enabled:
                stream.write(f'  {prefix_string}: bw={quantizer.bitwidth}, '
                             f'encoding-present={bool(quantizer.encoding)}\n')

                if quantizer.encoding:
                    stream.write(f'    {quantizer}')
            else:
                stream.write(f'  {prefix_string}: Not quantized\n')

            stream.write('  -------\n')

        stream = io.StringIO(newline='\n')
        stream.write("-------------------------\n")
        stream.write("Quantized Model Report\n")
        stream.write("-------------------------\n")

        for layer_name, layer in self._get_qc_quantized_layers(self.model):
            stream.write('----------------------------------------------------------\n')
            stream.write('Layer: {}\n'.format(layer_name))

            # Inputs
            if isinstance(layer.input_quantizers, dict):
                for name, quantizer in layer.input_quantizers.items():
                    print_quantizer_state(stream, quantizer, prefix_string=f"Input[{name}]")
            else:
                for index, quantizer in enumerate(layer.input_quantizers):
                    print_quantizer_state(stream, quantizer, prefix_string=f"Input[{index}]")

            # Params
            for param_name, quantizer in layer.param_quantizers.items():
                print_quantizer_state(stream, quantizer, prefix_string=f"Param[{param_name}]")

            # Outputs
            if isinstance(layer.output_quantizers, dict):
                for name, quantizer in layer.output_quantizers.items():
                    print_quantizer_state(stream, quantizer, prefix_string=f"Output[{name}]")
            else:
                for index, quantizer in enumerate(layer.output_quantizers):
                    print_quantizer_state(stream, quantizer, prefix_string=f"Output[{index}]")

        return stream.getvalue()

    @staticmethod
    def prepare_sim_for_compute_encodings(sim: 'QuantizationSimModel'):
        """
        Prepare QuantSim for compute encodings. Resets encodings for each quantizable layer and sets mode to Analysis.

        :param sim: QuantSim to prepare
        """
        # pylint: disable=protected-access
        quantized_layers = sim._get_qc_quantized_layers(sim.model)

        for _, layer in quantized_layers:
            # Clear stats and encodings if they are present
            layer.reset_encodings()

            # And set the mode to analysis
            layer.set_mode(QcQuantizeOpMode.ANALYSIS)

        for _, layer in quantized_layers:
            # call only when quant scheme is percentile
            if sim._quant_scheme == QuantScheme.post_training_percentile:
                layer.set_percentile_value(sim._percentile_value)

    @staticmethod
    def compute_layer_encodings_for_sim(sim: 'QuantizationSimModel'):
        """
        Compute encodings for each quantizable layer in sim after forward pass has been called.

        :param sim: QuantSim to compute encodings for
        """
        # pylint: disable=protected-access
        quantized_layers = sim._get_qc_quantized_layers(sim.model)
        # Get the computed per-layer encodings and log them
        for name, layer in quantized_layers:
            layer.compute_encoding()

            # Before we return we set the mode to active - meaning ready for quantize/de-quantize
            # for layers with valid_encoding, otherwise we set to pass through
            if isinstance(layer, QcQuantizeRecurrent):
                sim.set_mode_for_recurrent_module(layer, name)
            else:
                # By default we want to set the Quantization wrappers to ACTIVE mode
                layer.set_mode(QcQuantizeOpMode.ACTIVE)

        sim.replace_wrappers_for_quantize_dequantize()

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args): # pylint: disable=arguments-differ
        """
        Computes encodings for all quantization sim nodes in the model. It is also used to find initial encodings for
        Range Learning

        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        :return: None

        """

        QuantizationSimModel.prepare_sim_for_compute_encodings(self)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        with utils.in_eval_mode(self.model), torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        QuantizationSimModel.compute_layer_encodings_for_sim(self)

    @classmethod
    def set_mode_for_recurrent_module(cls, layer: QcQuantizeRecurrent, name: str):
        """
        Sets Recurrent module to active or pass through mode based on quantizer state

        :param layer:  Qc Quantizer layer for recurrent module
        :param name:  layer name
        :return: True if the encoding is invalid

        """
        for quantizer_name, output_quantizer in layer.output_quantizers.items():
            if output_quantizer.enabled:
                if output_quantizer.encoding:
                    encoding = output_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)

        for quantizer_name, input_quantizer in layer.input_quantizers.items():
            if input_quantizer.enabled:
                if input_quantizer.encoding:
                    encoding = input_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)

        layer.set_mode(QcQuantizeOpMode.ACTIVE)

    def set_percentile_value(self, percentile_value: float):
        """
        Set the percentile value to be used while computing encodings
        """
        if percentile_value < 90 or percentile_value > 100:
            raise ValueError("Percentile value must be in range [90, 100]")
        self._percentile_value = percentile_value

    def _replace_quantization_wrapper(self, model, device):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param model: model for which PostTrainingWrapper gets replaced with Trainable wrapped module
        :param device: device on which model is present
        :return: None
        """
        for module_name, module_ref in model.named_children():

            if isinstance(module_ref, StaticGridQuantWrapper):
                # Create a Trainable wrapper and copy properties of PostTrainingWrapper to the Trainable wrapper
                quantized_module = self._construct_and_initialize_trainable_wrapper(module_ref, device)
                setattr(model, module_name, quantized_module)

            elif isinstance(module_ref, QcQuantizeRecurrent):
                # Set Recurrent layer for training mode
                module_ref.construct_and_initialize_trainable_quantizers(self._quant_scheme)

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                self._replace_quantization_wrapper(module_ref, device)

    def _construct_and_initialize_trainable_wrapper(self, post_training_module: StaticGridQuantWrapper,
                                                    device: torch.device) -> LearnedGridQuantWrapper:
        """
        Copies following tensor quantizer attributes from StaticGridQuantWrapper to LearnedGridQuantWrapper
        to avoid any mismatch.
            - enabled
            - bitwidth
            - encoding
            - use_symmetric_encodings
            - use_strict_symmetric
            - use_unsigned_symmetric

        :param post_training_module: StaticGridQuantWrapper wrapped module
        :param device: device on which model is present
        :return: trainable_module: QcTrainable wrapper module
        """

        # pylint: disable=protected-access
        module = post_training_module._module_to_wrap

        num_inputs = len(post_training_module.input_quantizers)
        num_outputs = len(post_training_module.output_quantizers)

        # Creating a LearnedGridQuantWrapper module
        trainable_module = LearnedGridQuantWrapper(module, self._default_param_bw,
                                                   self._default_output_bw, self._rounding_mode, self._quant_scheme,
                                                   device=device, num_inputs=num_inputs, num_outputs=num_outputs,
                                                   data_type=QuantizationDataType.int)
        # Copy user settable attributes for outputs
        for index, quantizer in enumerate(post_training_module.output_quantizers):
            initialize_learned_grid_quantizer_attributes(trainable_module.output_quantizers[index], quantizer)
            if trainable_module.output_quantizers[index].encoding_min_max_fixed_vals is not None:
                trainable_module.output_quantizers[index].freeze_encoding()
        # Copy user settable attributes for inputs
        for index, quantizer in enumerate(post_training_module.input_quantizers):
            initialize_learned_grid_quantizer_attributes(trainable_module.input_quantizers[index], quantizer)
            if trainable_module.input_quantizers[index].encoding_min_max_fixed_vals is not None:
                trainable_module.input_quantizers[index].freeze_encoding()
        # Copy user settable attributes for params
        for name, quantizer in post_training_module.param_quantizers.items():
            learned_grid_quantizer = trainable_module.param_quantizers[name]
            initialize_learned_grid_quantizer_attributes(learned_grid_quantizer, quantizer)
            if learned_grid_quantizer.encoding_min_max_fixed_vals is not None:
                learned_grid_quantizer.freeze_encoding()

        return trainable_module

    def replace_wrappers_for_quantize_dequantize(self):
        """
        Replaces StaticGridWrapper with LearnedGridWrapper
        """
        if self._quant_scheme == QuantScheme.training_range_learning_with_tf_init or self._quant_scheme == \
                QuantScheme.training_range_learning_with_tf_enhanced_init:
            try:
                device = utils.get_device(self.model)
            except StopIteration:
                # Model doesn't have any parameter.
                # Set device to cpu by default.
                device = torch.device('cpu')

            self._replace_quantization_wrapper(self.model, device)

    def _create_quantizer_module(self, module_to_quantize: torch.nn.Module, num_inout_tensors: Dict,
                                 data_type: QuantizationDataType) -> torch.nn.Module:
        """Instantiates wrapper based on quant scheme
        """
        assert self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced,
                                      QuantScheme.training_range_learning_with_tf_enhanced_init,
                                      QuantScheme.training_range_learning_with_tf_init,
                                      QuantScheme.post_training_percentile]

        # We lookup the number of input and output tensors already determined
        # Special case, we are adding a wrapper for a module not in the forward pass: Use default of 1, 1
        num_in_tensors, num_out_tensors = num_inout_tensors.get(module_to_quantize, (1, 1))

        # Set quantizer to be a module replacer if it is in qc_quantize_modules_dict, otherwise set as
        # StaticGridQuantWrapper.
        quantizer_wrapper_type = qc_quantize_modules_dict.get(type(module_to_quantize), _V1LazyQuantizeWrapper)

        if issubclass(quantizer_wrapper_type, LazyQuantizeWrapper):
            quant_scheme_for_initialization = self._quant_scheme
        else:
            quant_scheme_for_initialization = utils.get_v1_quant_scheme_for_initialization(self._quant_scheme)

        # TODO add quant_scheme_for_initialization for FP8 case
        quantized_module = quantizer_wrapper_type(module_to_quantize, self._default_param_bw, self._default_output_bw,
                                                  self._rounding_mode, quant_scheme_for_initialization, num_inputs=num_in_tensors,
                                                  num_outputs=num_out_tensors, data_type=data_type)

        return quantized_module

    @classmethod
    def _is_quantizable_module(cls, module: torch.nn.Module):
        # pylint: disable=unidiomatic-typecheck
        return type(module) != torch.nn.Module and\
               not isinstance(module, unquantizable_modules) and\
               not cls._is_quantized_module(module)

    @classmethod
    def _is_quantized_module(cls, module: torch.nn.Module):
        return isinstance(module, quantized_modules)

    def _add_quantization_wrappers(self, module, num_inout_tensors, default_data_type: QuantizationDataType):
        """Recursively add quantization wrappers to all appropriate modules starting with module
        """
        if self._is_quantized_module(module):
            return

        for module_name, module_ref in module.named_children():
            logger.debug("nn.Module found : %s", module_ref)

            if self._is_quantizable_module(module_ref) and utils.is_leaf_module(module_ref):
                # Create a new QcQuantize wrapper module
                quantized_module = self._create_quantizer_module(module_ref, num_inout_tensors, default_data_type)
                setattr(module, module_name, quantized_module)
            else:
                self._add_quantization_wrappers(module_ref, num_inout_tensors, default_data_type)

    # pylint: disable=too-many-arguments
    @classmethod
    def _update_encoding_dicts_for_layer(cls, layer: _QuantizedModuleProtocol, layer_name: str, activation_encodings_onnx: Dict,
                                         activation_encodings_torch: Dict, param_encodings: Dict,
                                         op_to_io_tensor_map: Dict, valid_param_set: set, propagate_encodings: bool,
                                         tensor_to_consumer_map: Dict[str, str],
                                         layers_to_onnx_op_names: Dict[str, str],
                                         tensor_to_quantizer_map: Dict):
        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param activation_encodings_onnx: dictionary of activation encodings which maps onnx attribute to encodings
        :param activation_encodings_torch: dictionary of activation encodings which maps pytorch names to encodings
        :param param_encodings: dictionary of param encodings
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param valid_param_set: a set of valid param input names in model
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        """
        if isinstance(layer, QcQuantizeRecurrent):
            # Update encodings for Recurrent layers
            QuantizationSimModel._update_encoding_dict_for_recurrent_layers(layer, layer_name, op_to_io_tensor_map,
                                                                            activation_encodings_onnx,
                                                                            param_encodings, propagate_encodings,
                                                                            tensor_to_quantizer_map)
        else:
            super()._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings_onnx,
                                                     activation_encodings_torch,
                                                     param_encodings, op_to_io_tensor_map,
                                                     valid_param_set, propagate_encodings,
                                                     tensor_to_consumer_map, layers_to_onnx_op_names,
                                                     tensor_to_quantizer_map)

    @staticmethod
    def _update_encoding_dict_for_recurrent_layers(layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                                   activation_encodings_onnx: Dict, param_encodings: Dict,
                                                   propagate_encodings: bool, tensor_to_quantizer_map: Dict):
        """

        :param layer:
        :param layer_name:
        :param op_to_io_tensor_map:
        :param activation_encodings_onnx:
        :param param_encodings:
        :param propagate_encodings:
        :return:
        """

        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-locals

        onnx_activations_to_quantizers, onnx_params_to_quantizers = \
            layer.get_activation_param_quantizers_for_onnx_tensors(op_to_io_tensor_map[layer_name +
                                                                                       '#root_node'])
        # ------------------
        # Activations
        # ------------------
        quantizer = None
        for tensor, quantizer in onnx_activations_to_quantizers.items():
            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
            encoding = create_encoding_dict(quantizer_encoding, quantizer, propagate_encodings=False)
            activation_encodings_onnx[tensor] = [encoding]
            tensor_to_quantizer_map[tensor] = quantizer

        if propagate_encodings and quantizer:
            _, op_names = QuantizationSimModel.find_op_names_for_layer(layer_name, op_to_io_tensor_map, None, None)
            for op_name in op_names:
                io_tensor_list = op_to_io_tensor_map[op_name]
                if not isinstance(io_tensor_list, list):
                    io_tensor_list = [io_tensor_list]

                for io_tensors in io_tensor_list:

                    if io_tensors.outputs:
                        for output_tensor in io_tensors.outputs:
                            if output_tensor in onnx_activations_to_quantizers:
                                continue
                            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
                            encoding = create_encoding_dict(quantizer_encoding, quantizer, True)

                            activation_encodings_onnx[output_tensor] = [encoding]
                            tensor_to_quantizer_map[output_tensor] = quantizer

        # ------------------
        # Params
        # ------------------
        for tensor, quantizer in onnx_params_to_quantizers.items():
            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
            encoding = create_encoding_dict(quantizer_encoding, quantizer, propagate_encodings=False)
            param_encodings[tensor] = [encoding]
            tensor_to_quantizer_map[tensor] = quantizer

    @staticmethod
    def _get_qc_quantized_layers(model) -> List[Tuple[str, QcQuantizeWrapper]]:
        quantized_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (QcQuantizeRecurrent, LazyQuantizeWrapper, _QuantizedModuleProtocol)):
                quantized_layers.append((name, module))
        return quantized_layers

    @classmethod
    def _remove_quantization_wrappers(cls, starting_module, list_of_modules_to_exclude):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param starting_module: Module to recursive search downstream from
        :param list_of_modules_to_exclude: List of torch modules to remove quantization wrappers from (if present)
        :return: None
        """
        for module_name, module_ref in starting_module.named_children():

            # If modules is in the exclude list, remove the wrapper
            if module_ref in list_of_modules_to_exclude:
                if isinstance(module_ref, (_QuantizedModuleProtocol, QcQuantizeRecurrent)):
                    orig_module = module_ref.get_original_module()
                elif isinstance(module_ref, QcQuantizeStandAloneBase):
                    orig_module = torch.nn.Identity()
                else:
                    orig_module = None

                if orig_module:
                    setattr(starting_module, module_name, orig_module)
                    module_ref = orig_module

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                cls._remove_quantization_wrappers(module_ref, list_of_modules_to_exclude)

    @classmethod
    @torch.no_grad()
    def _apply_qdq_to_model_parameters(cls, model: torch.nn.Module):
        """
        Applies quant-dequant to the parameters of a PyTorch model
        to avoid rounding error during weight quantization.

        :param model: The PyTorch model whose parameters will be quant-dequantized.
        """
        # pylint: disable=protected-access
        for module in model.modules():
            if isinstance(module, (QcQuantizeRecurrent, StaticGridQuantWrapper)):
                with utils.in_eval_mode(module):
                    module._quantize_dequantize_params()
            elif isinstance(module, (LearnedGridQuantWrapper)):
                with utils.in_eval_mode(module):
                    module._quantize_params()
                    cls._update_parameters_by_attr(module._module_to_wrap)

    def named_qmodules(self):
        """Generator that yields all quantized modules in the model and their names
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent, LazyQuantizeWrapper)):
                yield name, module

    quant_wrappers = named_qmodules

    @staticmethod
    def _replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device: torch.device):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param quant_sim_model: model for which QcQuantizeWrapper gets replaced with wrapped module using
        native torch quantization nodes
        :param device: device on which model is present
        :return:
        """
        # Recursively replace quantization wrappers to native torch quantization nodes
        for module_name, module_ref in quant_sim_model.named_children():
            # Create a native torch quantization node
            if isinstance(module_ref, QcQuantizeWrapper):
                embedded_module = NativeTorchQuantWrapper(module_ref, '_module_to_wrap', device)
                setattr(quant_sim_model, module_name, embedded_module)

            elif isinstance(module_ref, QcQuantizeRecurrent):
                logger.error('Do not support save model embedded native torch quantization nodes using QcQuantizeRecurrent.')
                raise AssertionError

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(module_ref, device)

    @classmethod
    def save_model_with_embedded_quantization_nodes(cls, sim_model, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
                                                    onnx_export_args: Optional[Union[OnnxExportApiArgs, Dict]] = None,
                                                    export_to_torchscript: bool = False, is_conditional: bool = False):
        """
        Export model embedded with native torch quantization nodes. These nodes will be exported
        as default onnx or torch script quantized nodes.
        :param sim_model: model with the quantsim wrappers
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph
        :param onnx_export_args: optional export argument with onnx specific overrides if not provide export via
                torchscript graph. Int16 can only be exported by torchscript
        :param export_to_torchscript: If True, export to torchscript. Export to onnx otherwise. Defaults to False.
        :param is_conditional: True if model is conditional, False otherwise
        :return:
        """
        def _validate_torchquantizer(quant_sim_model):
            # To avoid non 8 bit TorchQuantizer are exported to ONNX
            for _, module in quant_sim_model.named_modules():
                if isinstance(module, NativeTorchQuantWrapper):
                    quantizers = module.input_quantizers + module.output_quantizers
                    if 'weight' in module.param_quantizers:
                        quantizers += [module.param_quantizers['weight']]
                    if 'bias' in module.param_quantizers:
                        quantizers += [module.param_quantizers['bias']]

                    for quantizer in quantizers:
                        if quantizer.enabled and quantizer.data_type == QuantizationDataType.int and quantizer.bitwidth != 8:
                            raise ValueError('Only 8 bit quantizers are supported by exporting to ONNX model.'
                                             'Please enable export_to_torchscript if you want to export non 8 bit quantizers.')

        model_filename = filename_prefix + '_embedded' + '.onnx'
        model_path = os.path.join(path, model_filename)
        quant_sim_model = copy.deepcopy(sim_model)

        device = utils.get_device(quant_sim_model)
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.to(device)
        else:
            dummy_input = tuple([input.to(device) for input in dummy_input])  # pylint: disable=consider-using-generator
        QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device)

        if export_to_torchscript:
            with utils.in_eval_mode(quant_sim_model), torch.no_grad():
                trace = torch.jit.trace(quant_sim_model, dummy_input)
                ts_path = os.path.join(path, filename_prefix + '_embedded' + '.torchscript.pth')
                trace.save(ts_path)
        else:
            _validate_torchquantizer(quant_sim_model)
            OnnxSaver._export_model_to_onnx(quant_sim_model, dummy_input, model_path, is_conditional, onnx_export_args) # pylint: disable=protected-access



@deprecated("Use QuantizationSimModel.load_encodings instead.")
def load_encodings_to_sim(quant_sim_model: _QuantizationSimModelBase, pytorch_encoding_path: str):
    """
    Loads the saved encodings to quant sim model. The encoding filename to load should end in _torch.encodings,
    generated as part of quantsim export.

    :param quant_sim_model: Quantized model to load encodings for. Note: The model configuration should be the same as
        when encodings were exported.
    :param pytorch_encoding_path: Path of the encodings file to load.
    """
    for module in quant_sim_model.model.modules():
        if isinstance(module, QcQuantizeWrapper):
            module.set_mode(QcQuantizeOpMode.ACTIVE)

    quant_sim_model.load_encodings(pytorch_encoding_path,
                                   strict=True,
                                   partial=False,
                                   requires_grad=None,
                                   allow_overwrite=None)

    if isinstance(quant_sim_model, QuantizationSimModel):
        # Only for V1 quantsim
        quant_sim_model.replace_wrappers_for_quantize_dequantize()


def compute_encodings_for_sims(sim_list: List[QuantizationSimModel], forward_pass_callback: Callable,
                               forward_pass_callback_args: Any):
    """
    Compute encodings for a list of QuantSims.

    :param sim_list: List of QuantSims to compute encodings for.
    :param forward_pass_callback: A callback function that simply runs forward passes on the models. This callback
        function should use representative data for the forward pass, so the calculated encodings work for all
        data samples. This callback internally chooses the number of data samples it wants to use for calculating
        encodings.
        The callback expects exactly two inputs:
            - List of models which are involved in the forward pass. The models are taken directly from calling
            sim.model for each sim in sim_list, passed in the same order in which the sims appear in sim_list.
            - Forward pass callback args
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
        the user to determine the type of this parameter. E.g. could be simply an integer representing the number
        of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
        If set to None, forward_pass_callback will be invoked with no parameters.
    """
    ctx_managers = [torch.no_grad()]
    for sim in sim_list:
        ctx_managers.append(utils.in_eval_mode(sim.model))
        QuantizationSimModel.prepare_sim_for_compute_encodings(sim)

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        _ = forward_pass_callback([sim.model for sim in sim_list], forward_pass_callback_args)

    for sim in sim_list:
        QuantizationSimModel.compute_layer_encodings_for_sim(sim)
