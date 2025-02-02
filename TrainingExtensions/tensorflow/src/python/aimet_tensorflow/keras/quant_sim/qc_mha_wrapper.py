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
""" Qc Quantize wrapper for tf 2 keras MultiHeadAttention Layer"""

from typing import Union
import math
import tensorflow as tf
from packaging import version

# Disable pylint errors because of conditional imports
# pylint: disable=ungrouped-imports
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.advanced_activations import _large_compatible_negative
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops, special_math_ops, array_ops
if version.parse(tf.version.VERSION) < version.parse("2.10"):
    from tensorflow.python.keras.layers.multi_head_attention import MultiHeadAttention, _build_proj_equation, _get_output_shape
else:
    from tensorflow.keras.layers import MultiHeadAttention
    from keras.layers.attention.multi_head_attention import _build_proj_equation, _get_output_shape

# Disable pylint warning since there is a conditional import above
# pylint: disable=wrong-import-position
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.quantsim import QuantizerSettings, QcQuantizeWrapper

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint:disable=too-many-instance-attributes
# pylint:disable=attribute-defined-outside-init

class QcQuantizableMultiHeadAttention(MultiHeadAttention):
    """
    Quantize Keras layers.MultiHeadAttention
    """
    def __init__(self,
                 quant_scheme: Union[QuantScheme, str] = 'tf_enhanced',
                 rounding_mode: str = 'nearest',
                 default_output_bw: int = 8,
                 default_param_bw: int = 8,
                 default_data_type: QuantizationDataType = QuantizationDataType.int,
                 copy_source_weights=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.default_output_bw = default_output_bw
        self.default_param_bw = default_param_bw
        self.default_data_type = default_data_type
        self.copy_source_weights = copy_source_weights
        self._remove_quantizers = False

    def get_config(self):
        """ Override get_config """
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
        base_config = super().get_config()
        base_config.update(config)
        return base_config

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
                                    param_quantizers=param_quantizers, name=layer.name)
        return wrapper

    def _build_from_signature(self, query, value, key=None):
        """ Invokes base class version of function to build layers and variables and
        then wraps them with QcQuantizeWrappers
        :param query: query tensor or TensorShape
        :param value: value tensor or TensorShape
        :param key: key tensor or TensorShape
        """
        super()._build_from_signature(query, value, key)

        if key is None:
            key = value

        query_shape = tensor_shape.TensorShape(query.shape) if hasattr(query, "shape") else query
        value_shape = tensor_shape.TensorShape(value.shape) if hasattr(value, "shape") else value
        key_shape = tensor_shape.TensorShape(key.shape) if hasattr(key, "shape") else key

        with tf_utils.maybe_init_scope(self):
            _, _, output_rank = _build_proj_equation(query_shape.rank - 1, bound_dims=1, output_dims=2)
            output_shape = _get_output_shape(output_rank, [self._num_heads, self._key_dim])

            with tf.name_scope("query"):
                self._query_dense.build(query_shape)
            with tf.name_scope("value"):
                self._value_dense.build(value_shape)
            with tf.name_scope("key"):
                self._key_dense.build(key_shape)
            with tf.name_scope("attention_output"):
                self._output_dense.build(output_shape)

            if self.copy_source_weights is not None:
                new_weights = self.get_weights()
                # Weights 0-5 in QcQuantizableMultiHeadAttention correspond to the weights 0-5 in Keras MHA, and
                # represent the weights and biases associated with the query, key, and value feedforward layers
                new_weights[0:6] = self.copy_source_weights[0:6]
                # Weights 32-33 in QcQuantizableMultiHeadAttention correspond to the weights 6-7 in Keras MHA, and
                # represent the output feedforward layer weights and biases
                new_weights[32:34] = self.copy_source_weights[6:8]
                self.set_weights(new_weights)

            self._wrapped_query_dense = self._wrap_layer(self._query_dense, 1)
            self._wrapped_key_dense = self._wrap_layer(self._key_dense, 1)
            self._wrapped_value_dense = self._wrap_layer(self._value_dense, 1)
            self._wrapped_output_dense = self._wrap_layer(self._output_dense, 1)

            self._wrapped_layers = [self._wrapped_query_dense, self._wrapped_key_dense, self._wrapped_value_dense,
                                    self._wrapped_attention_score_layer, self._wrapped_identity_layer,
                                    self._wrapped_addition, self._wrapped_masked_softmax,
                                    self._wrapped_combine_qkv_layer, self._wrapped_output_dense]

    def _build_attention(self, rank: int):
        """Invokes base class version of  function to build multi-head dot-product attention computations. Creates
        lambda layers for all operations that need quantized inputs or outputs, and wraps them with QcQuantizeWrappers
        :param rank: the rank of query, key, value tensors.
        """
        super()._build_attention(rank)

        def scale_and_multiply(inputs):
            return special_math_ops.einsum(self._dot_product_equation,
                                           inputs[0],
                                           math_ops.multiply(inputs[1], 1.0 / math.sqrt(float(self._key_dim))))

        self._attention_score_layer = tf.keras.layers.Lambda(scale_and_multiply, name="scale_and_multiply")
        self._wrapped_attention_score_layer = self._wrap_layer(self._attention_score_layer, 2)

        self._identity_layer = tf.keras.layers.Lambda(lambda x: x, name="identity")
        self._wrapped_identity_layer = self._wrap_layer(self._identity_layer, 1)

        def masked_add(inputs):
            adder = (1.0 - math_ops.cast(inputs[1], inputs[0].dtype)) * (_large_compatible_negative(inputs[0].dtype))
            return inputs[0] + adder

        self._add_layer = tf.keras.layers.Lambda(masked_add, name="masked_add")
        self._wrapped_addition = self._wrap_layer(self._add_layer, 2)

        def softmax_func(inputs):
            return self._masked_softmax(inputs)

        self._softmax_layer = tf.keras.layers.Lambda(softmax_func, name="softmax")
        self._wrapped_masked_softmax = self._wrap_layer(self._softmax_layer, 1)

        def combine_qkv(inputs):
            return special_math_ops.einsum(self._combine_equation, inputs[0], inputs[1])

        self._combine_qkv_layer = tf.keras.layers.Lambda(combine_qkv, name="combine_qkv")
        self._wrapped_combine_qkv_layer = self._wrap_layer(self._combine_qkv_layer, 2)

    def _compute_attention(self,
                           query,
                           key,
                           value,
                           attention_mask=None,
                           training=None):
        attention_scores = self._wrapped_attention_score_layer([key, query])

        if attention_mask is not None:
            attention_mask = self._wrapped_identity_layer(attention_mask)

            mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = array_ops.expand_dims(attention_mask, axis=mask_expansion_axes)

            attention_scores = self._wrapped_addition([attention_scores, attention_mask])

        attention_scores = self._wrapped_masked_softmax(attention_scores)

        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        attention_output = self._wrapped_combine_qkv_layer([attention_scores_dropout, value])
        return attention_output, attention_scores

    def call(self,
             query,
             value,
             key=None,
             attention_mask=None,
             return_attention_scores=False,
             training=None):
        """
        Call function adapted from parent class to use wrapped layers
        :param query: query tensor
        :param value: value tensor
        :param key: optional key tensor. If a key is not provided, then the value tensor is used
        :param attention_mask: optional attention mask
        :param return_attention_scores: boolean indicating whether attention scores should also be returned
        :param training: boolean indicating whether this layer should behave as in training mode, or in regular mode
        """
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        if not self._remove_quantizers:
            query_layer = self._wrapped_query_dense
            key_layer = self._wrapped_key_dense
            value_layer = self._wrapped_value_dense
            output_layer = self._wrapped_output_dense
            attn_func = self._compute_attention
        else:
            query_layer = self._query_dense
            key_layer = self._key_dense
            value_layer = self._value_dense
            output_layer = self._output_dense
            attn_func = super()._compute_attention

        query = query_layer(query)
        key = key_layer(key)
        value = value_layer(value)

        attention_output, attention_scores = attn_func(query, key, value, attention_mask, training)
        attention_output = output_layer(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def deactivate_quantizers(self):
        """Function to deactivate quantizers during forward pass"""
        self._remove_quantizers = True

    def reactivate_quantizers(self):
        """Function to reactivate quantizers during forward pass"""
        self._remove_quantizers = False

    def quant_wrappers(self):
        """Function to allow QuantizationSimModel to access local quantization wrappers"""
        for layer in self._wrapped_layers:
            yield layer
