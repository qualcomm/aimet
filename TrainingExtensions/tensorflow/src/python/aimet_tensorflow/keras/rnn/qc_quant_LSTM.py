# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Code to override Keras LSTM classes"""

import sys
from typing import Union
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from packaging import version

if version.parse(tf.version.VERSION) >= version.parse("2.10"):
    # pylint: disable=ungrouped-imports
    from tensorflow.compat.v1 import executing_eagerly_outside_functions, assign
    from tensorflow import config

    # Ignore pylint errors as keras module is not available in TF 2.4
    from keras import activations
    from keras import backend
    from keras import regularizers
    from keras.engine.input_spec import InputSpec
    from keras.layers.rnn import LSTM
else:
    from tensorflow.python.framework import config # pylint: disable=ungrouped-imports
    from tensorflow.python.framework import constant_op # pylint: disable=ungrouped-imports
    from tensorflow.python.framework import dtypes # pylint: disable=ungrouped-imports
    from tensorflow.python.framework.ops import executing_eagerly_outside_functions, device # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import activations # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import backend # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import regularizers # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.engine.input_spec import InputSpec # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, LSTM # pylint: disable=ungrouped-imports
    from tensorflow.python.ops import nn # pylint: disable=ungrouped-imports
    from tensorflow.python.ops.state_ops import assign # pylint: disable=ungrouped-imports

# pylint: disable=wrong-import-position
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.quantsim import QuantizerSettings, QcQuantizeWrapper
from aimet_tensorflow.keras.rnn.qc_quant_LSTMCell import QuantizedLSTMCell

#Manage inheritance as per version
if version.parse(tf.version.VERSION) >= version.parse("2.10"):
    base_list = [LSTM]
else:
    base_list = [DropoutRNNCellMixin, LSTM]

# pylint: disable=too-many-ancestors
# pylint: disable=abstract-method
class QuantizedLSTM(*base_list):

    """Class merging LSTM class from recurrent.py and recurrent_v1.py in TF2.4
    , to be in sync with latest tf 2.10 version"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False,
            is_sequential_model=False,
            is_input_quantized=False,
            quant_scheme: Union[QuantScheme, str] = 'tf_enhanced',
            rounding_mode: str = 'nearest',
            default_output_bw: int = 8,
            default_param_bw: int = 8,
            default_data_type: QuantizationDataType = QuantizationDataType.int,
            copy_source_weights=None,
            **kwargs,
    ):
        """Overriden LSTM __init__ function"""

        #Quantization is supported for urolled LSTM only.
        if not unroll:
            logging.error("Only unrolled LSTM can be quantized, as of now. Need to use flag 'unroll=True'")
            sys.exit()

        self._wrapped_layers = []
        self.is_sequential_model = is_sequential_model
        self.is_input_quantized = is_input_quantized
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.default_output_bw = default_output_bw
        self.default_param_bw = default_param_bw
        self.default_data_type = default_data_type
        self.copy_source_weights = copy_source_weights

        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self.return_runtime = kwargs.pop("return_runtime", False)
        implementation = kwargs.pop("implementation", 2)
        if implementation == 0:
            logging.warning(
                "`implementation=0` has been deprecated, "
                "and now defaults to `implementation=1`."
                "Please update your layer call."
            )
        if "enable_caching_device" in kwargs:
            cell_kwargs = {
                "enable_caching_device": kwargs.pop("enable_caching_device")
            }
        else:
            cell_kwargs = {}

        cell = QuantizedLSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
            quant_scheme=quant_scheme,
            rounding_mode=rounding_mode,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            default_data_type=default_data_type,
            copy_source_weights=copy_source_weights,
            **cell_kwargs,
        )

        # pylint: disable=bad-super-call
        super(LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            time_major=time_major,
            unroll=unroll,
            **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

        #As implementation need all activations and paramencodings from every LSTM cell (unrolled LSTM), create different memory for each cell
        self._cell_dict = {}
        self.step_count = 0

        self.state_spec = [
            InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
        ]

        (tf_version_sigmoid, tf_version_tanh) = (tf.sigmoid, tf.tanh) if version.parse(tf.version.VERSION) >= version.parse("2.10") \
                                                                      else (nn.sigmoid, nn.tanh)
        self._could_use_gpu_kernel = (
            self.activation in (activations.tanh, tf_version_tanh)
            and self.recurrent_activation in (activations.sigmoid, tf_version_sigmoid)
            and recurrent_dropout == 0
            and not unroll
            and use_bias
            and executing_eagerly_outside_functions())

        if config.list_logical_devices('GPU'):
            # Only show the message when there is GPU available, user will not care
            # about the cuDNN if there isn't any GPU.
            if self._could_use_gpu_kernel:
                logging.debug("CUDNN is not supported")

    def get_config(self):

        """ Override get_config """

        # pylint: disable=redefined-outer-name
        config = {
            "is_sequential_model":
                self.is_sequential_model,
            "is_input_quantized":
                self.is_input_quantized,
            "quant_scheme":
                self.quant_scheme,
            "rounding_mode":
                self.rounding_mode,
            "default_output_bw":
                self.default_output_bw,
            "default_param_bw":
                self.default_param_bw,
            "default_data_type":
                self.default_data_type,
            "copy_source_weights":
                self.copy_source_weights
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config

    def build(self, input_shape):

        """RNN build method overriden"""

        # pylint: disable=bad-super-call
        super().build(input_shape)

        # pylint: disable=attribute-defined-outside-init
        if self.is_input_quantized:
            self._wrapped_lstm_input = self._wrap_layer(tf.keras.layers.Lambda(
                lambda x: x, name="lstm_input" if self.is_sequential_model else "input_1"), 1)
            self._wrapped_layers.append(self._wrapped_lstm_input)

        self.built = True

    def _wrap_layer(self, layer: tf.keras.layers.Layer, num_inputs: int) -> tf.keras.layers.Layer:

        """
        Function to wrap layers with QcQuantizeWrappers, used by keras clone_model()
        :param layer: Layer to wrap
        :return: Wrapped layer, or original layer if layer is not to be wrapped
        """

        activation_quant_settings = QuantizerSettings(self.default_output_bw, self.default_data_type, self.rounding_mode,
                                                      self.quant_scheme, False, False, False)
        param_quant_settings = QuantizerSettings(self.default_param_bw, self.default_data_type, self.rounding_mode,
                                                 self.quant_scheme, False, False, False)

        input_quantizers, output_quantizers, param_quantizers = None, None, None
        wrapper = QcQuantizeWrapper(layer, activation_quant_settings, param_quant_settings,
                                    num_inputs=num_inputs,
                                    input_quantizers=input_quantizers,
                                    output_quantizers=output_quantizers,
                                    param_quantizers=param_quantizers, name=layer.name, in_quant_enabled=False)
        return wrapper

    if version.parse(tf.version.VERSION) >= version.parse("2.10"):
        @classmethod
        def runtime(cls, runtime_name):

            """Used by overriden code to determine runtime"""

            with tf.device('/cpu:0'):
                return tf.constant(runtime_name, dtype=tf.float32, name='runtime')
    else:
        @classmethod
        def runtime(cls, runtime_name):

            """Used by overriden code to determine runtime"""

            with device('/cpu:0'):
                return constant_op.constant(runtime_name, dtype=dtypes.float32, name='runtime')

    def call(self, inputs, mask=None, training=None, initial_state=None):

        """Overriden LSTM recurrent_v2 call function"""

        #Initialize count to go through the same built network, when call is again hit
        self.step_count = 0

        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
        is_ragged_input = row_lengths is not None
        self._validate_args_if_ragged(is_ragged_input, mask)

        # LSTM does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

        if isinstance(mask, list):
            mask = mask[0]

        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]

        # TODO(b/156447398) Investigate why the cuDNN kernel kernel fails with
        # ragged inputs.
        if is_ragged_input or not self._could_use_gpu_kernel:
            # Fall back to use the normal LSTM.
            kwargs = {'training': training}
            self._maybe_reset_cell_dropout_mask(self.cell)

            def step(inputs, states):
                self.step_count += 1
                #Clone cell to have different instance and hence data
                layer_var = "lstm_cell_"+str(self.step_count)
                if self._cell_dict.get(layer_var) is None:
                    self._cell_dict[layer_var] = self.cell.__class__.from_config(self.cell.get_config())
                    if self.step_count == 1:
                        self._cell_dict[layer_var].is_first_step = True
                    h, [h, c] = self._cell_dict[layer_var](inputs, states, **kwargs)

                    layers = self._cell_dict[layer_var].get_wrapped_layers()
                    # pylint: disable=protected-access
                    for layer in layers:
                        if layer._name in ["MatMul", "MatMul_1"]:
                            layer._layer_to_wrap.kernel._handle_name = "lstm/lstm_cell"+"/"+layer._layer_to_wrap.kernel._handle_name.split("/")[-1]

                        if layer._name == "BiasAdd":
                            layer._layer_to_wrap.bias._handle_name = "lstm/lstm_cell"+"/"+layer._layer_to_wrap.bias._handle_name.split("/")[-1]

                        layer._name = "lstm/lstm_cell_"+str(self.step_count)+"/"+layer._name

                    self._wrapped_layers.extend(layers)
                else:
                    #Initialization completed while first call(). Need to only run the step.
                    h, [h, c] = self._cell_dict[layer_var](inputs, states, **kwargs)

                return h, [h, c]

            #Wrap LSTM Input to get input encodings. Output of this layer will be input itself. Get output encoding itself
            #for model input encoding, if LSTM is first layer.
            if self.is_input_quantized:
                inputs = self._wrapped_lstm_input(inputs)

            last_output, outputs, states = backend.rnn(
                step,
                inputs,
                initial_state,
                constants=None,
                go_backwards=self.go_backwards,
                mask=mask,
                unroll=self.unroll,
                input_length=row_lengths if row_lengths is not None else timesteps,
                time_major=self.time_major,
                zero_output_for_mask=self.zero_output_for_mask)

            runtime = self.runtime(0)

        if self.stateful:
            updates = [
                assign(self_state, state)
                for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)

        if self.return_sequences:
            output = backend.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        else:
            output = last_output

        return_val = output
        if self.return_state:
            return_val = [output] + list(states)
        elif self.return_runtime:
            return_val = (output, runtime)

        return return_val

    def quant_wrappers(self):

        """Function to allow QuantizationSimModel to access local quantization wrappers"""

        for layer in self._wrapped_layers:
            yield layer
