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

"""Code to override Keras LSTMCell class"""

from typing import Union
import tensorflow as tf
from packaging import version

if version.parse(tf.version.VERSION) >= version.parse("2.10"):
    # Ignore pylint errors as keras module is not available in TF 2.4
    from keras import backend
    from keras import constraints
    from keras import initializers
    from keras import regularizers
    from keras.engine.base_layer import Layer
    from keras.engine.input_spec import InputSpec
    from keras.layers import Dense
    from keras.layers.rnn import LSTMCell
    from keras.layers.rnn.rnn_utils import caching_device
else:
    from tensorflow.python.keras import backend # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import constraints # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import initializers # pylint: disable=ungrouped-imports
    from tensorflow.python.keras import regularizers # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.engine.base_layer import Layer # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.engine.input_spec import InputSpec # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers import Dense # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers.recurrent import LSTMCell, _caching_device as caching_device # pylint: disable=ungrouped-imports

# pylint: disable=wrong-import-position
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.quantsim import QuantizerSettings, QcQuantizeWrapper

#General vs SNPE names map. Values to key are SNPE names
aimet_SNPE_name_map = {"kernel":"kernel",
                       "recurrent_kernel":"recurrent_kernel",
                       "input_matmul":"MatMul",
                       "hidden_st_matmul":"MatMul_1",
                       "in_hidden_st_add":"add",
                       "bias_add":"BiasAdd",
                       "input_gate":"Sigmoid",
                       "forget_gate":"Sigmoid_1",
                       "candidate_cell_state":"Tanh",
                       "ig_ct_matmul":"mul_1",
                       "prev_c_fg_matmul":"mul",
                       "new_cell_state":"add_1",
                       "output_gate":"Sigmoid_2",
                       "act_new_cell_state":"Tanh_1",
                       "new_hidden_state":"mul_2"}

# pylint: disable=abstract-method
# pylint: disable=too-many-ancestors
class CustomDense(Dense):

    """Class using Keras dense with customzations for weight cache"""

    def build(self, input_shape):

        """Overridden build method, to sync with standard LSTM way of building weights"""

        # pylint: disable=protected-access
        default_caching_device = caching_device(self)

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        # pylint: disable=attribute-defined-outside-init
        if self.name == aimet_SNPE_name_map["input_matmul"]:
            wt_name = aimet_SNPE_name_map["kernel"]
        elif self.name == aimet_SNPE_name_map["hidden_st_matmul"]:
            wt_name = aimet_SNPE_name_map["recurrent_kernel"]
        else:
            wt_name = "unknown"

        # pylint: disable=unexpected-keyword-arg
        self.kernel = self.add_weight(
            name=wt_name,
            shape=[last_dim, self.units*4],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
            trainable=True,
        )

        self.bias = None
        self.built = True

# pylint: disable=too-many-ancestors
class BiasAdd(Layer):

    """Layer to add bias"""

    def __init__(self,
                 units,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):

        "Initialize method"

        super().__init__(**kwargs)
        self.units = units
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        """build method"""

        # pylint: disable=protected-access
        default_caching_device = caching_device(self)

        # pylint: disable=attribute-defined-outside-init
        # pylint: disable=unexpected-keyword-arg
        self.bias = self.add_weight(
            shape=(self.units*4,),
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            caching_device=default_caching_device,
        )

    # pylint: disable=arguments-differ
    def call(self, inputs, **kwargs):

        """call method"""

        return backend.bias_add(inputs, self.bias)

    def get_config(self):

        """ Override get_config """

        # pylint: disable=redefined-outer-name
        config = {
            'units':
                self.units,
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        # pylint: disable=bad-super-call
        base_config = super().get_config()
        base_config.update(config)
        return base_config

# pylint: disable=too-many-ancestors
# pylint: disable=too-many-instance-attributes
# pylint: disable=abstract-method
class QuantizedLSTMCell(LSTMCell):

    """Class overriding LSTM Cell from Keras"""

    def __init__(self,
                 units,
                 quant_scheme: Union[QuantScheme, str] = 'tf_enhanced',
                 rounding_mode: str = 'nearest',
                 default_output_bw: int = 8,
                 default_param_bw: int = 8,
                 default_data_type: QuantizationDataType = QuantizationDataType.int,
                 copy_source_weights=None,
                 **kwargs):

        """Overriden LSTM Cell __init__"""

        super().__init__(units, **kwargs)
        self.wrapped_layers = []

        self.is_first_step = False
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.default_output_bw = default_output_bw
        self.default_param_bw = default_param_bw
        self.default_data_type = default_data_type
        self.copy_source_weights = copy_source_weights

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-statements
    def wrap_LSTM_internals(self, input_shape):

        """Method wrapping LSTM Internal Splits"""

        #Do Input weight matmul
        self.input_matmul = CustomDense(self.units, input_shape=input_shape, name=aimet_SNPE_name_map["input_matmul"], use_bias=False,
                                        kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer,
                                        kernel_constraint=self.kernel_constraint)

        #Call build() to initialize weights.
        self.input_matmul.build(input_shape)
        self.input_matmul.set_weights([self.copy_source_weights[0]])

        #Wrap Matmul
        self._wrapped_input_matmul = self._wrap_layer(self.input_matmul, 1)

        #Do Hidden State Matmul
        _input_shape = (1, self.units)
        self.hidden_st_matmul = CustomDense(self.units, input_shape=_input_shape, name=aimet_SNPE_name_map["hidden_st_matmul"], use_bias=False,
                                            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint)

        #Call build() to initialize weights.
        self.hidden_st_matmul.build(_input_shape)
        self.hidden_st_matmul.set_weights([self.copy_source_weights[1]])

        #Wrap Hidden State Matmul
        self._wrapped_hidden_st_matmul = self._wrap_layer(self.hidden_st_matmul, 1)

        #Add Input and hidden state outputs and wrap same
        self.in_hidden_st_add = tf.keras.layers.Add(name=aimet_SNPE_name_map["in_hidden_st_add"])
        self._wrapped_in_hidden_st_add = self._wrap_layer(self.in_hidden_st_add, 2)

        #Wrap Bias Add
        if self.use_bias:
            self.bias_add = BiasAdd(self.units, name=aimet_SNPE_name_map["bias_add"], bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer, bias_constraint=self.bias_constraint)

            #Call build() to initialize weights.
            self.bias_add.build(_input_shape)
            self.bias_add.set_weights([self.copy_source_weights[2]])

            #Wrap Bias Add
            self._wrapped_bias_add = self._wrap_layer(self.bias_add, 1)

        self.input_gate = tf.keras.layers.Lambda(self.recurrent_activation, name=aimet_SNPE_name_map["input_gate"])
        self._wrapped_input_gate = self._wrap_layer(self.input_gate, 1)

        self.forget_gate = tf.keras.layers.Lambda(self.recurrent_activation, name=aimet_SNPE_name_map["forget_gate"])
        self._wrapped_forget_gate = self._wrap_layer(self.forget_gate, 1)

        self.candidate_cell_state = tf.keras.layers.Lambda(self.activation, name=aimet_SNPE_name_map["candidate_cell_state"])
        self._wrapped_candidate_cell_state = self._wrap_layer(self.candidate_cell_state, 1)

        self.ig_ct_matmul = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name=aimet_SNPE_name_map["ig_ct_matmul"])
        self._wrapped_ig_ct_matmul = self._wrap_layer(self.ig_ct_matmul, 2)

        self.prev_c_fg_matmul = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name=aimet_SNPE_name_map["prev_c_fg_matmul"])
        self._wrapped_prev_c_fg_matmul = self._wrap_layer(self.prev_c_fg_matmul, 2)

        self.new_cell_state = tf.keras.layers.Lambda(lambda x: x[0] + x[1], name=aimet_SNPE_name_map["new_cell_state"])
        self._wrapped_new_cell_state = self._wrap_layer(self.new_cell_state, 2)

        self.output_gate = tf.keras.layers.Lambda(self.recurrent_activation, name=aimet_SNPE_name_map["output_gate"])
        self._wrapped_output_gate = self._wrap_layer(self.output_gate, 1)

        self.act_new_cell_state = tf.keras.layers.Lambda(self.activation, name=aimet_SNPE_name_map["act_new_cell_state"])
        self._wrapped_act_new_cell_state = self._wrap_layer(self.act_new_cell_state, 1)

        self.new_hidden_state = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name=aimet_SNPE_name_map["new_hidden_state"])
        self._wrapped_new_hidden_state = self._wrap_layer(self.new_hidden_state, 2)

        if self.is_first_step is False:
            self.wrapped_layers.extend([self._wrapped_input_matmul, self._wrapped_hidden_st_matmul])

        self.wrapped_layers.extend([self._wrapped_in_hidden_st_add,
                                    self._wrapped_bias_add, self._wrapped_input_gate, self._wrapped_forget_gate,
                                    self._wrapped_candidate_cell_state, self._wrapped_ig_ct_matmul, self._wrapped_prev_c_fg_matmul,
                                    self._wrapped_new_cell_state, self._wrapped_output_gate, self._wrapped_act_new_cell_state, self._wrapped_new_hidden_state])

    def build(self, input_shape):

        """Overridden build to call wrap method"""

        # pylint: disable=bad-super-call,super-with-arguments
        super(LSTMCell, self).build(input_shape)

        self.wrap_LSTM_internals(input_shape)

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

    # pylint: disable=too-many-locals
    def call(self, inputs, states, training=None):

        """Overridden call function, using wrapped layers"""

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
            inputs = inputs * dp_mask[0]

        x = self._wrapped_input_matmul(inputs)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1 = h_tm1 * rec_dp_mask[0]

        h = self._wrapped_hidden_st_matmul(h_tm1)
        z = self._wrapped_in_hidden_st_add([x, h])

        if self.use_bias:
            z = self._wrapped_bias_add(z)

        # pylint: disable=unexpected-keyword-arg,redundant-keyword-arg,no-value-for-parameter
        z = tf.split(z, num_or_size_splits=4, axis=1)
        z0, z1, z2, z3 = z

        i = self._wrapped_input_gate(z0)
        f = self._wrapped_forget_gate(z1)

        inter_ct = self._wrapped_candidate_cell_state(z2)
        prev_c_fg_mul = self._wrapped_prev_c_fg_matmul((c_tm1, f))
        c = self._wrapped_new_cell_state((prev_c_fg_mul, self._wrapped_ig_ct_matmul((i, inter_ct))))

        o = self._wrapped_output_gate(z3)

        h = self._wrapped_new_hidden_state((o, self._wrapped_act_new_cell_state(c)))

        return h, [h, c]

    def get_wrapped_layers(self):

        """Function to return wrapped layers"""
        return self.wrapped_layers

    def get_config(self):

        """ Override get_config """

        # pylint: disable=redefined-outer-name
        config = {
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
        # pylint: disable=bad-super-call
        base_config = super().get_config()
        base_config.update(config)
        return base_config
