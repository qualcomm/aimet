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
""" Quantsim for Keras """
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import os
from typing import Union, Dict, Tuple, Optional, List

import tensorflow as tf
from aimet_common import libpymo

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_common import quantsim
from aimet_common.quantsim import extract_global_quantizer_args
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.graphsearchtuils import GraphSearchUtils
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper, QuantizerSettings
from aimet_tensorflow.keras.quant_sim.qc_mha_wrapper import QcQuantizableMultiHeadAttention
from aimet_tensorflow.keras.rnn.qc_quant_LSTM import QuantizedLSTM
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer, ActivationTensorQuantizer, \
    ParamPerTensorQuantizer, StaticGridPerChannelQuantizer, ParamPerChannelQuantizer
from aimet_tensorflow.keras.quantsim_config.quantsim_config import QuantSimConfigurator, INPUT_QUANTIZERS, \
    OUTPUT_QUANTIZERS, PARAM_QUANTIZERS
from aimet_tensorflow.keras.utils.common import convert_h5_model_to_pb_model

from aimet_tensorflow.keras.defs import AxisHandling
import aimet_tensorflow.keras.utils.common as keras_common_utils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

unquantizable_modules = (tf.keras.layers.InputLayer, QcQuantizeWrapper)
substitutable_modules = {
    tf.keras.layers.MultiHeadAttention: QcQuantizableMultiHeadAttention,
    tf.keras.layers.LSTM: QuantizedLSTM
}


@dataclass
class QuantizationSimModelParams:
    """
    Data class that holds parameters for QuantizationSimModel. Used specifically to rebuild after converting to TF frozen pb
    """
    quant_scheme: Union[QuantScheme, str] = 'tf_enhanced'
    rounding_mode: str = 'nearest'
    default_output_bw: int = 8
    default_param_bw: int = 8
    in_place: bool = False
    config_file: str = None
    default_data_type: QuantizationDataType = QuantizationDataType.int


# pylint: disable=too-many-ancestors
# pylint: disable=too-many-instance-attributes
class QuantizationSimModel(tf.keras.Model):
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def __init__(self, model, quant_scheme: Union[QuantScheme, str] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, in_place: bool = False,
                 config_file: str = None, default_data_type: QuantizationDataType = QuantizationDataType.int):
        """
        :param model: Model to quantize
        :param quant_scheme: Quantization Scheme, currently supported schemes are post_training_tf and
               post_training_tf_enhanced, defaults to post_training_tf_enhanced
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'.
        :param default_output_bw: bitwidth to use for activation tensors, defaults to 8
        :param default_param_bw: bitwidth to use for parameter tensors, defaults to 8
        :param in_place: If True, then the given 'model' is modified in-place to add quant-sim nodes.
                Only suggested use of this option is when the user wants to avoid creating a copy of the model
        :param config_file: Path to a config file to use to specify rules for placing quant ops in the model
        :param default_data_type: Default data type to use for quantizing all layer parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16
        """
        super().__init__()

        self._model_without_wrappers = model
        if not in_place:
            self._model_without_wrappers = tf.keras.models.clone_model(model)
            n_weights = len(self._model_without_wrappers.weights)
            self._model_without_wrappers.set_weights(model.get_weights()[:n_weights])
        self._layer_name_to_quant_wrapper = {}
        self._substituted_layer = {}  # to hold the substituted layers
        self._validate_model()
        self.connected_graph = ConnectedGraph(self._model_without_wrappers)
        self._quantsim_configurator = self._initialize_quantsim_configurator(quant_scheme, rounding_mode,
                                                                             default_output_bw, default_param_bw,
                                                                             default_data_type, config_file)
        self.quant_scheme = quant_scheme
        self._percentile_value = 100  # default percentile value
        self.per_channel_quantization_enabled = self._quantsim_configurator.per_channel_quantization_flag
        self.model = self._add_quantization_wrappers(quant_scheme, rounding_mode,
                                                     default_output_bw, default_param_bw, default_data_type)
        self.quant_args = extract_global_quantizer_args(quant_scheme, self._quantsim_configurator)

        self._params = QuantizationSimModelParams(quant_scheme, rounding_mode, default_output_bw, default_param_bw,
                                                  in_place, config_file, default_data_type)

    def _validate_model(self):
        """
        Check that model is appropriate for quantsim.
        """
        multiple_inbound_node_layers = []

        for layer in self._model_without_wrappers.layers:
            if len(layer.inbound_nodes) > 1:
                multiple_inbound_node_layers.append(layer.name)

        if multiple_inbound_node_layers:
            error_msg = (f'Layers with more than one inbound nodes are unsupported. This may occur if a layer is '
                         f'reused multiple times in the model definition.\n'
                         f'Layers with multiple inbound nodes: {multiple_inbound_node_layers}')
            _logger.error(error_msg)
            raise NotImplementedError(error_msg)

        sep_conv_found = self.check_separable_conv(self._model_without_wrappers)
        if sep_conv_found:
            # Raising an assertion error incase there's SeparableConv2D in the model because in this case we have two sets of weights: Depthwise
            # and Pointwise. For depthwise kernels, LAST TWO AXIS should be considered and for pointwise kernels LAST AXIS
            # should be considered, which is not handled here. Running model preparer beforehand will resolve this as there the
            # SeparableConv2D is splitted into two layers Depthwise and Pointwise seperately.
            raise AssertionError("SeparableConv2D found in the model. Please run model preparer before calling QuantizationSimModel")

    def check_separable_conv(self, model: tf.keras.models.Model | tf.keras.Sequential) -> bool:
        """
        Checks for SeparableConv2D layer in the model
        :param model: Keras Model
        :return: Boolean value, True if SeperableConv layer is found else False
        """
        for layer in model.layers:
            if isinstance(layer, tf.keras.Sequential):
                if self.check_separable_conv(layer):
                    return True
            elif isinstance(layer, tf.keras.layers.SeparableConv2D):
                return True
        return False

    def _get_quantizer_list(self) -> Tuple[List, List, List]:
        """
        Method to provide a list of input, output and parameter quantizers
        :return: Three lists containing input, paramater and output quantizers respectively
        """
        input_quantizers = []
        parameter_quantizers = []
        output_quantizers = []

        for wrapper in self.quant_wrappers():
            for quantizer in wrapper.input_quantizers:
                input_quantizers.append(quantizer)

            for quantizer in wrapper.param_quantizers:
                parameter_quantizers.append(quantizer)

            for quantizer in wrapper.output_quantizers:
                output_quantizers.append(quantizer)

        return input_quantizers, parameter_quantizers, output_quantizers

    def set_percentile_value(self, percentile_value: float):
        """
        Set the percentile value to be used while computing encodings for quantizers having percentile quant scheme.

        :param percentile_value: Percentile value to be set to
        """
        if percentile_value < 90 or percentile_value > 100:
            raise ValueError("Percentile value must be in range [90, 100]")
        self._percentile_value = percentile_value

        # Set the percentile value to the activation quantizers
        input_quantizers, _, output_quantizers = self._get_quantizer_list()
        for quantizer in input_quantizers + output_quantizers:
            if quantizer.quant_scheme == QuantScheme.post_training_percentile:
                quantizer.set_percentile_value(self._percentile_value)

    def _initialize_quantsim_configurator(self, quant_scheme: Union[QuantScheme, str], rounding_mode: str,
                                          default_output_bw: int, default_param_bw: int,
                                          default_data_type: QuantizationDataType = QuantizationDataType.int,
                                          config_file: str = None) -> QuantSimConfigurator:
        """
        Initialize quantsim configurator
        :param quant_scheme: Quantization Scheme
        :param rounding_mode: The round scheme to used
        :param default_output_bw: bitwidth to use for activation tensors
        :param default_param_bw: bitwidth to use for parameter tensors
        :param default_data_type: data type to use for the parameter tensors
        :param config_file: Path to a config file to use to specify rules for placing quant ops in the model
        :return: QuantSimConfigurator
        """
        return QuantSimConfigurator(self.connected_graph, quant_scheme, rounding_mode,
                                    default_output_bw, default_param_bw, default_data_type, config_file)

    def _add_quantization_wrappers(self, quant_scheme, rounding_mode,
                                   default_output_bw, default_param_bw, default_data_type):
        """
        Add quantization wrappers to the model and return a new model with the wrappers inserted.
        :param quant_scheme: Quantization scheme to use
        :param rounding_mode: Rounding mode to use
        :param default_output_bw: Default bitwidth for activation quantizers
        :param default_param_bw: Default bitwidth for param quantizers
        :param default_data_type: data type to use for param quantizers
        """

        def wrap_layer(layer) -> tf.keras.layers.Layer:
            """
            Function to wrap layers with QcQuantizeWrappers, used by keras clone_model()
            :param layer: Layer to wrap
            :return: Wrapped layer, or original layer if layer is not to be wrapped
            """
            if isinstance(layer, tuple(substitutable_modules.keys())):
                new_class = substitutable_modules[type(layer)]
                config = layer.get_config()
                config["copy_source_weights"] = layer.get_weights()

                if isinstance(layer, tf.keras.layers.LSTM):
                    if isinstance(self._model_without_wrappers, tf.keras.Sequential):
                        config["is_sequential_model"] = True

                    # pylint: disable=protected-access
                    if self._quantsim_configurator._layer_to_config_dict[layer]["is_input_quantized"]["setting"]:
                        config["is_input_quantized"] = True
                    config["quant_scheme"] = quant_scheme
                    config["rounding_mode"] = rounding_mode
                    config["default_output_bw"] = default_output_bw
                    config["default_param_bw"] = default_param_bw
                    config["default_data_type"] = default_data_type

                wrapped_layer = new_class.from_config(config)
                self._substituted_layer[layer] = wrapped_layer
                return wrapped_layer

            if isinstance(layer, tf.keras.Sequential):
                return tf.keras.models.clone_model(layer, clone_function=wrap_layer)

            if isinstance(layer, unquantizable_modules) or layer.submodules:
                return layer

            activation_quant_settings = QuantizerSettings(default_output_bw, default_data_type, rounding_mode,
                                                          quant_scheme, False, False, False)
            param_quant_settings = QuantizerSettings(default_param_bw, default_data_type, rounding_mode,
                                                     quant_scheme, False, False, False)

            input_quantizers, output_quantizers, param_quantizers = self._get_quantizers_by_layer(layer)
            wrapper = QcQuantizeWrapper(layer, activation_quant_settings, param_quant_settings,
                                        num_inputs=len(layer.inbound_nodes[0].keras_inputs),
                                        input_quantizers=input_quantizers,
                                        output_quantizers=output_quantizers,
                                        param_quantizers=param_quantizers,
                                        per_channel_quantization_enabled=self.per_channel_quantization_enabled)
            self._layer_name_to_quant_wrapper[layer.name] = wrapper
            return wrapper

        return tf.keras.models.clone_model(self._model_without_wrappers, clone_function=wrap_layer)

    def _get_quantizers_by_layer(self, layer: tf.keras.layers.Layer) -> Tuple[Optional[ActivationTensorQuantizer],
                                                                              Optional[ActivationTensorQuantizer],
                                                                              Union[ParamPerTensorQuantizer,
                                                                                    ParamPerChannelQuantizer]]:
        """
        Get input/output/param quantizers from quantizers dictionary or initialize quantizers if layer is not found
        :param layer: Target layer
        :return: tuple of input, output, param quantizers
        """
        quantizers_dict = self._quantsim_configurator.get_quantizers_dict(layer)
        if quantizers_dict is None:
            _logger.warning("%s not found in quantizers dict, will generate quantizers automatically", layer.name)
            input_quantizers = None
            output_quantizers = None
            param_quantizers = None
        else:
            input_quantizers = quantizers_dict.get(INPUT_QUANTIZERS)
            output_quantizers = quantizers_dict.get(OUTPUT_QUANTIZERS)
            param_quantizers = quantizers_dict.get(PARAM_QUANTIZERS)

        return input_quantizers, output_quantizers, param_quantizers

    @staticmethod
    def _quantizer_to_name_tuple(quantizers: List[TensorQuantizer]) -> Tuple[Optional[List[str]]]:
        """
        Converts a list of quantizers to a tuple of quantizer names
        :param quantizers: quantizers
        :return: tuple of quantizer names
        """
        quant_list = []
        if not quantizers:
            return None

        for quantizer in quantizers:
            quant_list.append(quantizer.name)
        return tuple(quant_list)

    def get_quantizer_name_by_layer(self, layer: tf.keras.layers.Layer) -> Tuple[Optional[List[str]],
                                                                                 Optional[List[str]],
                                                                                 Optional[List[str]]]:
        """
        Get the names of input, output and param quantizers
        :param layer: the keras layer
        :return: Tuple of quantizer names
        """
        input_quantizers, output_quantizers, param_quantizers = self._get_quantizers_by_layer(layer)
        output_quantizers_names = self._quantizer_to_name_tuple(output_quantizers)
        input_quantizers_names = self._quantizer_to_name_tuple(input_quantizers)
        parameter_quantizers_names = self._quantizer_to_name_tuple(param_quantizers)

        return input_quantizers_names, output_quantizers_names, parameter_quantizers_names

    def _disable_quantizers_in_folded_batchnorm(self):
        """
        Disable input/output/param quantizers if layer is folded batch normalization
        """
        for quantsim_wrapper in self._layer_name_to_quant_wrapper.values():
            if GraphSearchUtils.is_folded_batch_normalization(quantsim_wrapper.original_layer):
                for q in quantsim_wrapper.input_quantizers:
                    q.disable()
                for q in quantsim_wrapper.output_quantizers:
                    q.disable()
                for q in quantsim_wrapper.param_quantizers:
                    q.disable()

    @staticmethod
    def _get_encoding_dict_for_quantizer(quantizer: TensorQuantizer) -> Union[List[Dict[str, Union[str, int, float]]],
                                                                              Dict[str, Union[str, int, float]]]:
        """
        Get encoding dict for a tensor quantizer.

        :param quantizer: Quantizer to get encoding info from
        :return: Dictionary or List of dictionaries containing encodings info for the tensor quantizer
        """
        if not isinstance(quantizer, ParamPerChannelQuantizer) or quantizer.data_type == QuantizationDataType.float:
            quantizer_encodings = [quantizer.encoding]
        else:
            quantizer_encodings = quantizer.encoding
        return [
            {
                'min': encoding.min,
                'max': encoding.max,
                'scale': encoding.delta,
                'offset': int(encoding.offset),
                'bitwidth': encoding.bw,
                'is_symmetric': str(quantizer.is_symmetric),
                'dtype': 'int'
            } if quantizer.data_type == QuantizationDataType.int
            else {'dtype': 'float', 'bitwidth': int(quantizer.bitwidth)}
            for encoding in quantizer_encodings
        ]

    def get_encodings_dict(self) -> Dict[str, Union[str, Dict]]:
        """
        Get encodings dict containing all activation and parameter encodings info in the model
        :return: Dictionary containing all activation and parameter encodings info in the model
        """
        # pylint: disable=protected-access, too-many-branches
        model_input_tensor_names = [inp.name for inp in self.model.inputs]
        activation_encodings = {}
        param_encodings = {}
        for wrapper in self.quant_wrappers():
            for idx, input_quantizer in enumerate(wrapper.input_quantizers):
                if input_quantizer.is_encoding_valid() or input_quantizer.data_type == QuantizationDataType.float:
                    # because dense layers in quantizable MHA are not explicitly sublayers, they don't have their
                    # inbound_nodes parameter populated, so the name of the quantizer is used instead
                    if not wrapper._layer_to_wrap.inbound_nodes:
                        tensor_name = wrapper.name + "/" + input_quantizer.name + ":0"
                    else:
                        tensor_name = wrapper._layer_to_wrap.inbound_nodes[0].keras_inputs[idx].name
                    encoding_dict = self._get_encoding_dict_for_quantizer(input_quantizer)
                    if tensor_name in model_input_tensor_names:
                        tensor_name += ":0"
                    activation_encodings[tensor_name] = encoding_dict
            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                if param_quantizer.is_encoding_valid() or param_quantizer.data_type == QuantizationDataType.float:
                    param_name = wrapper._layer_to_wrap.weights[idx].name
                    encoding_dict = self._get_encoding_dict_for_quantizer(param_quantizer)
                    param_encodings[param_name] = encoding_dict
            for idx, output_quantizer in enumerate(wrapper.output_quantizers):
                if output_quantizer.is_encoding_valid() or output_quantizer.data_type == QuantizationDataType.float:
                    # because dense layers in quantizable MHA are not explicitly sublayers, they don't have their
                    # inbound_nodes parameter populated, so the name of the quantizer is used instead
                    if not wrapper._layer_to_wrap.inbound_nodes:
                        tensor_name = wrapper.name + ":0"
                    elif isinstance(wrapper._layer_to_wrap.output, List):
                        tensor_name = wrapper._layer_to_wrap.output[idx].name
                    else:
                        tensor_name = wrapper._layer_to_wrap.output.name
                    encoding_dict = self._get_encoding_dict_for_quantizer(output_quantizer)
                    activation_encodings[tensor_name] = encoding_dict
        return {
            'version': quantsim.encoding_version,
            'activation_encodings': activation_encodings,
            'param_encodings': param_encodings,
            'quantizer_args': self.quant_args if hasattr(self, "quant_args") else {}
        }

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model.

        :param forward_pass_callback: A callback function that is expected to run forward passes on a model.
            This callback function should use representative data for the forward pass, so the calculated encodings work for all data samples.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is.
            Up to the user to determine the type of this parameter. E.g. could be simply an integer representing the number of data samples to use.
            Or could be a tuple of parameters or an object representing something more complex.
        """
        ops_with_invalid_encodings = []
        self._compute_and_set_parameter_encodings(ops_with_invalid_encodings)

        self._set_op_mode_parameters(libpymo.TensorQuantizerOpMode.quantizeDequantize)

        forward_pass_callback(self.model, forward_pass_callback_args)
        for quant_wrapper in self.quant_wrappers():
            quant_wrapper.compute_encoding(ops_with_invalid_encodings)

        op_mode = self._param_op_mode_after_analysis(self.quant_scheme)

        self._set_op_mode_parameters(op_mode)

        if ops_with_invalid_encodings:
            _logger.info('The following quantizers did not have valid encodings and have been set to passThrough mode: '
                         '%s', ops_with_invalid_encodings)
            _logger.info('This can be due to the quantizers not having been evaluated during the forward pass in '
                         'compute encodings. Evaluation is required to collect statistics needed to compute valid '
                         'encodings.\n'
                         'As a result, the quantizers have been set to passThrough mode, meaning no quantization noise '
                         'will be simulated for these ops if they are evaluated in the future.\n'
                         'If this is not desired, amend the forward pass to evaluate tensors which require these ops '
                         'to be evaluated, and recompute encodings.')

    def _set_op_mode_parameters(self, op_mode: libpymo.TensorQuantizerOpMode):
        """
        Sets quant mode for parameters and if the encodings are invalid, then adds those wrappers
        to wrappers_with_invalid_encodings
        :param op_mode: Quant mode to set to
        """

        for quantizer_info in self.quant_wrappers():
            for param_quantizer in quantizer_info.param_quantizers:
                if param_quantizer.is_enabled():
                    param_quantizer.quant_mode = op_mode

    @staticmethod
    @contextlib.contextmanager
    def _set_encoding_version_to_0_6_1():
        assert quantsim.encoding_version in {'0.6.1', '1.0.0'}
        if quantsim.encoding_version == '1.0.0':
            _logger.info('Exporting to encoding version 1.0.0 is not yet supported. Exporting using version 0.6.1 '
                         'instead.')
        old_encoding_version = quantsim.encoding_version
        quantsim.encoding_version = '0.6.1'

        yield

        quantsim.encoding_version = old_encoding_version

    def export(self, path, filename_prefix, custom_objects=None, convert_to_pb=True):
        """
        This method exports out the quant-sim model so it is ready to be run on-target. Specifically, the following are saved

        1. The sim-model is exported to a regular Keras model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can then be imported by the on-target runtime (if desired)

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param custom_objects: If there are custom objects to load, Keras needs a dict of them to map them
        """
        with self._set_encoding_version_to_0_6_1():
            model_path = os.path.join(path, filename_prefix)

            #TF Version 2.4 has bug i.e. save() in tf format don't work for unrolled LSTM.
            for layer in self._model_without_wrappers.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    break
            else:
                self._model_without_wrappers.save(model_path)

            self._model_without_wrappers.save(model_path + '.h5', save_format='h5')

            # Conversion of saved h5 model to pb model for consumption by SNPE/QNN
            try:
                if convert_to_pb:
                    convert_h5_model_to_pb_model(f'{model_path}.h5', custom_objects=custom_objects)
            except ValueError:
                _logger.error("Could not convert h5 to frozen pb. "
                              "Please call export() again with custom_objects defined.")
                raise
            finally:
                encodings_dict = self.get_encodings_dict()
                encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
                save_json_yaml(encoding_file_path, encodings_dict)

    def _compute_and_set_parameter_encodings(self, ops_with_invalid_encodings: List):
        # pylint: disable=too-many-nested-blocks
        for quantizer_wrapper in self.quant_wrappers():
            for idx, param_quantizer in enumerate(quantizer_wrapper.param_quantizers):
                if param_quantizer.is_enabled() and param_quantizer.data_type == QuantizationDataType.int:
                    # 0th input to our quant wrapper is the tensor being quantized
                    weight_tensor = quantizer_wrapper.original_layer.get_weights()[idx]

                    # Per-channel
                    if isinstance(param_quantizer, StaticGridPerChannelQuantizer):
                        for index, tensor_quantizer in enumerate(param_quantizer.tensor_quantizer):
                            if param_quantizer.axis_handling == AxisHandling.LAST_TWO_AXES.value:
                                last_two_axes_combined_shape = list(weight_tensor.shape[:-2]) + [-1]
                                channel_slice = weight_tensor.reshape(*last_two_axes_combined_shape)
                                channel_slice = channel_slice.take(index, channel_slice.ndim - 1)
                            elif isinstance(quantizer_wrapper.original_layer, tf.keras.layers.Conv2DTranspose):
                                if weight_tensor.ndim == 4:
                                    channel_slice = weight_tensor.take(index, weight_tensor.ndim - 2)
                                else:
                                    # For bias in Transpose layers
                                    channel_slice = weight_tensor.take(index, weight_tensor.ndim - 1)
                            else:
                                channel_slice = weight_tensor.take(index, weight_tensor.ndim - 1)
                            tensor_quantizer.updateStats(channel_slice, False)

                    # Per-tensor
                    else:
                        tensor_quantizer = param_quantizer.tensor_quantizer
                        tensor_quantizer.updateStats(weight_tensor, False)

                    param_quantizer.compute_encoding(ops_with_invalid_encodings)

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file
        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)

        for quant_wrapper in self.quant_wrappers():
            quant_wrapper.set_and_freeze_param_encoding(param_encodings)

    # pylint: disable=too-many-nested-blocks
    def load_encodings_to_sim(self, encoding_file_path: str):
        """
        Loads the saved encodings to quant sim model

        :param encoding_file_path: path from where to load encodings file
        :return:
        """
        # pylint: disable=protected-access, too-many-branches, too-many-locals, too-many-statements
        # Load encodings file
        with open(encoding_file_path) as json_file:
            encodings = json.load(json_file)

        param_encodings = encodings['param_encodings']
        activation_encodings = encodings['activation_encodings']

        model_input_tensor_names = [inp.name for inp in self.model.inputs]

        for wrapper in self.quant_wrappers():
            for idx, input_quantizer in enumerate(wrapper.input_quantizers):
                # because dense layers in quantizable MHA and RNN are not explicitly sublayers, they don't have their
                # inbound_nodes parameter populated, so the name of the quantizer is used instead
                if not wrapper._layer_to_wrap.inbound_nodes:
                    tensor_name = wrapper.name + "/" + input_quantizer.name + ":0"
                else:
                    tensor_name = wrapper._layer_to_wrap.inbound_nodes[0].keras_inputs[idx].name
                if tensor_name in model_input_tensor_names:
                    tensor_name += ":0"

                if tensor_name in activation_encodings:
                    if not input_quantizer.is_enabled():
                        _logger.info("Not loading encodings for quantizer: %s as it is disabled", tensor_name)
                        continue
                    encoding_dict = activation_encodings[tensor_name][0]
                    if encoding_dict['dtype'] == 'int':
                        encoding, is_symmetric = keras_common_utils.create_encoding_from_dict(encoding_dict)
                        input_quantizer.tensor_quantizer.isEncodingValid = True
                        input_quantizer.set_quantizer_encodings(encoding.bw, is_symmetric, encoding,
                                                                libpymo.TensorQuantizerOpMode.quantizeDequantize)
                        _logger.info("Setting encodings for : %s", tensor_name)
                    elif encoding_dict['dtype'] == 'float':
                        input_quantizer.data_type = QuantizationDataType.float
                        input_quantizer.bitwidth = encoding_dict['bitwidth']
                        _logger.info("Setting quantizer dtype to float for : %s", tensor_name)
                    else:
                        raise RuntimeError("Unrecognized dtype %s for: %s" % (encoding_dict['dtype'], tensor_name))
                else:
                    if input_quantizer.is_enabled():
                        input_quantizer.disable()
                        _logger.info("Encoding for quantizer: %s is not present thus disabling it.", tensor_name)

            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                param_name = wrapper._layer_to_wrap.weights[idx].name

                if param_name in param_encodings:
                    if not param_quantizer.is_enabled():
                        _logger.info("Not loading encodings for parameter: %s as quantizer is disabled", param_name)
                        continue
                    if isinstance(param_quantizer, StaticGridPerChannelQuantizer):
                        if param_encodings[param_name][0]['dtype'] == 'float':
                            wrapper.param_quantizers[idx] = ParamPerTensorQuantizer(layer=param_quantizer._original_layer,
                                                                                    name=param_quantizer.name,
                                                                                    quant_scheme=param_quantizer._quant_scheme,
                                                                                    round_mode='nearest',
                                                                                    bitwidth=param_encodings[param_name][0]['bitwidth'],
                                                                                    data_type=QuantizationDataType.float,
                                                                                    is_symmetric=param_quantizer.is_symmetric,
                                                                                    use_strict_symmetric=param_quantizer.use_strict_symmetric,
                                                                                    use_unsigned_symmetric=param_quantizer.use_unsigned_symmetric,
                                                                                    enabled=False)
                        else:
                            encoding, is_symmetric = keras_common_utils.create_encoding_from_dict(
                                param_encodings[param_name])
                            for tensor_quantizer in param_quantizer.tensor_quantizer:
                                tensor_quantizer.isEncodingValid = True
                            bw = encoding[0].bw
                            param_quantizer.set_quantizer_encodings(bw, is_symmetric, encoding,
                                                                    libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
                        _logger.info("Setting encodings for : %s", param_name)
                    else:
                        encoding_dict = param_encodings[param_name][0]
                        if encoding_dict['dtype'] == 'int':
                            encoding, is_symmetric = keras_common_utils.create_encoding_from_dict(encoding_dict)
                            param_quantizer.tensor_quantizer.isEncodingValid = True
                            bw = encoding.bw
                            param_quantizer.set_quantizer_encodings(bw, is_symmetric, encoding,
                                                                    libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
                            _logger.info("Setting encodings for : %s", param_name)
                        elif encoding_dict['dtype'] == 'float':
                            param_quantizer.data_type = QuantizationDataType.float
                            param_quantizer.bitwidth = encoding_dict['bitwidth']
                            _logger.info("Setting quantizer to float for : %s", param_name)
                        else:
                            raise RuntimeError("Unrecognized dtype %s for: %s" % (encoding_dict['dtype'], tensor_name))
                else:
                    if param_quantizer.is_enabled():
                        param_quantizer.disable()
                        _logger.info("Encoding for parameter: %s not present thus disabling this quantizer.",
                                     param_name)

            # Loading encodings means that compute encodings was called. Therefore, these two lines set the correct
            # op mode for the correct quant scheme and if the quantization was per channel or not.
            op_mode = self._param_op_mode_after_analysis(self.quant_scheme)
            self._set_op_mode_parameters(op_mode)

            for idx, output_quantizer in enumerate(wrapper.output_quantizers):
                # because dense layers in quantizable MHA are not explicitly sublayers, they don't have their
                # inbound_nodes parameter populated, so the name of the quantizer is used instead
                if not wrapper._layer_to_wrap.inbound_nodes:
                    tensor_names = [wrapper.name + ":0"]
                else:
                    # There can be multiple outputs if there is a
                    # `tf.split` in the model.
                    if isinstance(wrapper._layer_to_wrap.output, list):
                        tensor_names = [
                            output.name
                            for output in wrapper._layer_to_wrap.output
                        ]
                    else:
                        tensor_names = [wrapper._layer_to_wrap.output.name]

                for tensor_name in tensor_names:
                    if tensor_name in activation_encodings:
                        if not output_quantizer.is_enabled():
                            _logger.info("Not loading encodings for quantizer: %s as it is disabled", tensor_name)
                            continue
                        encoding_dict = activation_encodings[tensor_name][0]
                        if encoding_dict['dtype'] == 'int':
                            encoding, is_symmetric = keras_common_utils.create_encoding_from_dict(encoding_dict)
                            output_quantizer.tensor_quantizer.isEncodingValid = True
                            output_quantizer.set_quantizer_encodings(encoding.bw, is_symmetric, encoding,
                                                                     libpymo.TensorQuantizerOpMode.quantizeDequantize)
                            _logger.info("Setting encodings for : %s", tensor_name)
                        elif encoding_dict['dtype'] == 'float':
                            output_quantizer.data_type = QuantizationDataType.float
                            output_quantizer.bitwidth = encoding_dict['bitwidth']
                            _logger.info("Setting quantizer dtype to float for : %s", tensor_name)
                        else:
                            raise RuntimeError("Unrecognized dtype %s for: %s" % (encoding_dict['dtype'], tensor_name))
                    else:
                        if output_quantizer.is_enabled():
                            output_quantizer.disable()
                            _logger.info("Encoding for quantizer: %s is not present thus disabling it.", tensor_name)

    def _param_op_mode_after_analysis(self, quant_scheme) -> libpymo.TensorQuantizerOpMode:
        """
        Returns quant mode to use for parameters after encodings have been computed
        :param quant_scheme: Quantization scheme to use
        :return: Quant mode to use
        """
        if quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                            QuantScheme.training_range_learning_with_tf_enhanced_init] \
                or self.per_channel_quantization_enabled:
            return libpymo.TensorQuantizerOpMode.quantizeDequantize
        return libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

    def quant_wrappers(self):
        """
        Generator for yielding all quantization wrappers
        """
        for layer in self.model.layers:
            if isinstance(layer, QcQuantizeWrapper):
                yield layer
            if isinstance(layer, tuple(substitutable_modules.values())):
                yield from layer.quant_wrappers()

            # For Getting Quantizers from Sequantial Block
            if isinstance(layer, tf.keras.Sequential):
                yield from quant_wrappers_for_sequential_block(layer)

    def get_quant_wrapper_for_layer_name(self, layer_name: str) -> QcQuantizeWrapper:
        """
        Return qc quant wrapper corresponding to a layer name
        :param layer_name: Layer name to get quantize wrapper for
        :return: Qc quant wrapper corresponding to a layer name
        """
        return self._layer_name_to_quant_wrapper.get(layer_name)

    # pylint: disable=too-many-locals
    def _fill_missing_encoding_min_max_gradients(self, gradients: list):
        """
        Computes the encoding min/max gradients and populates the gradients list
        :param gradients: gradients computed using GradientTape(gradients for encoding min/max will be `None`)
        """

        def _find_weight_in_layer(weight_name: str, model_layer: tf.keras.layers.Layer):

            for weight in model_layer.weights:
                if weight.name.split(":")[0] == weight_name:
                    return weight

            return None

        # Mapping used to get the gradients of weights(kernel, bias etc)
        weight_name_to_gradient = dict(zip([weight.name.split(":")[0] for weight in self.model.trainable_weights],
                                           gradients))

        # Mapping used to get index of encoding min/max gradients (which would be `None`) and fill them
        weight_name_to_index = dict(zip([weight.name for weight in self.model.trainable_weights],
                                        range(len(self.model.trainable_weights))))

        # Only process layers where 'param_quantizers' is defined (i.e. QcQuantizeWrapper layers)
        for layer in filter(lambda _layer: hasattr(_layer, 'param_quantizers'), self.model.layers):
            for param_quantizer in layer.param_quantizers:
                if param_quantizer.name in weight_name_to_gradient:
                    # Value of weight associated with this param quantizer
                    weight_tensor = _find_weight_in_layer(param_quantizer.name, layer.original_layer)

                    # Gradients of the weights
                    grad = weight_name_to_gradient[param_quantizer.name]

                    # Using the weights and it's gradients, compute gradients for encoding min/max
                    dloss_by_dmin, dloss_by_dmax = param_quantizer.get_gradients_for_encoding_min_max(weight_tensor,
                                                                                                      grad)

                    enc_min_index = weight_name_to_index[param_quantizer.encoding_min.name]
                    enc_max_index = weight_name_to_index[param_quantizer.encoding_max.name]

                    gradients[enc_min_index] = dloss_by_dmin
                    gradients[enc_max_index] = dloss_by_dmax

        # TODO: Remove this logic once this has been resolved in QNN/SNPE
        # Go through activation quantizers (Input/Output) and set any ReLU's encoding min to 0
        relu_quantize_wrappers = [
            _layer for _layer in self.model.layers
            if isinstance(_layer, QcQuantizeWrapper) and isinstance(_layer.original_layer, tf.keras.layers.ReLU)
        ]

        def _set_encoding_min_grad_to_None(quantizer):
            enc_min_index = weight_name_to_index[quantizer.encoding_min.name]
            gradients[enc_min_index] = None

        for relu_quantizer in relu_quantize_wrappers:
            for output_quantizer in relu_quantizer.output_quantizers:
                _set_encoding_min_grad_to_None(output_quantizer)

    # pylint: disable=useless-super-delegation
    def get_config(self):
        return super().get_config()

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)

    def train_step(self, data):
        """
        Custom training loop, equivalent to overriding `keras.Model.fit` function
        Reference: https://keras.io/guides/customizing_what_happens_in_fit/
        Only relevant when using range-learning, otherwise equivalent to `keras.Model.fit`
        Param quantizers are disconnected in the op graph of the wrapped model
        Because of this, the gradients are not computed for encoding min/max(when range learning is enabled)
        This custom train_step function computes the missing gradients for encoding min/max of param quantizers
        """
        x, y = data

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_weights)

        # Manually compute missing gradients for encoding min/max when using range learning
        if self.quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init]:
            self._fill_missing_encoding_min_max_gradients(gradients)

        gradients_to_apply = [(gradient, weight) for gradient, weight in zip(gradients, self.model.trainable_weights)
                              if gradient is not None]

        self.optimizer.apply_gradients(gradients_to_apply)

        self.compiled_metrics.update_state(y, predictions)

        return {m.name: m.result() for m in self.metrics}


def quant_wrappers_for_sequential_block(seq_block: tf.keras.Sequential):
    """
        Generator for yielding all quantization wrappers for a Sequantial Block
    """
    for layer in seq_block.layers:
        if isinstance(layer, QcQuantizeWrapper):
            yield layer
        if isinstance(layer, tuple(substitutable_modules.values())):
            yield from layer.quant_wrappers()

        # in cases of nested Sequential Block
        if isinstance(layer, tf.keras.Sequential):
            yield from quant_wrappers_for_sequential_block(layer)
