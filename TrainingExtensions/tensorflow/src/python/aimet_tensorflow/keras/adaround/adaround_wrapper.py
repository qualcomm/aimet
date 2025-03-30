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

""" Adaround wrapper """

import typing
from typing import Dict, Union, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import AIMET specific modules
from aimet_common import libpymo
from aimet_common.defs import QuantScheme
from aimet_common.defs import AdaroundConstants

BATCH_SIZE = 32
QUANT_SCHEME_TO_PYMO = {QuantScheme.post_training_tf_enhanced: libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                        QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF}

ConvType = typing.Union[tf.keras.layers.Conv2D,
                        tf.keras.layers.Conv2DTranspose,
                        tf.keras.layers.DepthwiseConv2D]

_supported_convs = ConvType.__args__


class AdaroundWrapper(keras.layers.Layer):
    """
    Adaround Wrapper base class
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, layer: tf.keras.layers.Layer, param_bw: int, quant_scheme: QuantScheme, is_symmetric: bool,
                 strict_symmetric: bool, unsigned_symmetric: bool, per_channel_enabled: bool,
                 output_height: Union[int, None], output_width: Union[int, None], output_channels: Union[int, None]):
        """
        :param layer: Tf keras layer.
        :param param_bw: Bitwidth for weight quantization
        :param quant_scheme: Quantization scheme
        :param is_symmetric: Symmetric vs Asymmetric encodings
        :param strict_symmetric: Strict symmetric flag
        :param unsigned_symmetric: Unsigned symmetric flag
        :param per_channel_enabled: Flag set for per channel encodings
        """
        super().__init__()

        self._layer = layer
        self.per_channel_enabled = per_channel_enabled
        self._orig_weight_tensor_shape, self._weight_tensor, self._bias_tensor = \
            self._get_weight_and_bias_tensors(layer)

        self.use_soft_rounding = self.add_weight(layer.name + '_use_soft_rounding', dtype=tf.bool,
                                                 initializer=tf.constant_initializer(True), trainable=False)
        self.ch_axis = self._get_channel_axis(self._layer, self._weight_tensor.shape)
        self.encoding = self.compute_encodings(self._weight_tensor, param_bw, quant_scheme, is_symmetric,
                                               strict_symmetric=strict_symmetric, unsigned_symmetric=unsigned_symmetric,
                                               enable_per_channel=self.per_channel_enabled, ch_axis=self.ch_axis)
        alpha = self._calculate_alpha(self._weight_tensor, self.encoding, self.per_channel_enabled, self.ch_axis)
        self.alpha = self.add_weight(self._layer.name + '_alpha', trainable=True, shape=alpha.shape)
        self.alpha.assign(alpha)
        self._output_height = output_height
        self._output_width = output_width
        self._output_channels = output_channels

    # Different 'call' method signatures between TF 2.4 and TF 2.10
    # pylint: disable=arguments-differ
    def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
        """
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments
        :return: Adarounded output tensor
        """
        adaround_weight_tensor = self.adaround_weights()
        adaround_out_tensor = self._compute_output_with_adarounded_weights(inputs, adaround_weight_tensor)

        if self._bias_tensor is not None:
            adaround_out_tensor = adaround_out_tensor + self._bias_tensor

        return adaround_out_tensor

    @staticmethod
    def compute_encodings(weight_data: np.ndarray, param_bw: int, quant_scheme: QuantScheme, is_symmetric: bool,
                          strict_symmetric: bool, unsigned_symmetric: bool, enable_per_channel: bool, ch_axis: int) \
            -> Union[libpymo.TfEncoding, List[libpymo.TfEncoding]]:
        """
        :param weight_data: Weight data of Adaround supported ops
        :param param_bw: bitwidth (4-31) to use for quantizing weight data
        :param quant_scheme: Quantization scheme
        :param is_symmetric: True if symmetric encodings is used, else asymmetric encodings.
        :param strict_symmetric: If true, and if is_symmetric is true, calculate encodings exactly centered
        around 0. E.g. if bw==8, then this results in quantized int values (-127:127). If this is not set, then
        quantized int values would be (-128:127) to use the entire range.
        :param unsigned_symmetric: If true, and if is_symmetric is true, check if the entire statistics we
        have collected are for +ve numbers. If yes, use quantized int values (0:255). This is a special case,
        where we have double the resolution for the computed encodings while still preserving the zero-point to
        be absolute 0.
        :param enable_per_channel: if set to True, use per channel quantization
        :param ch_axis: dimension to be used for per channel quantization. This field is unused for per-tensor flow
        :return: Encodings object for per-tensor flow or list of Encoding objects for per-channel flow
                Encoding object to contain (bw, max, min, delta and offset)
        """
        # pylint: disable=too-many-locals
        quant_scheme = QUANT_SCHEME_TO_PYMO[quant_scheme]
        # Create Encodings Analyzer and collect statistical data to compute encodings
        # Since the weight data is numpy and on CPU memory, useCuda is False
        if enable_per_channel:
            encoding = []
            shape = list(weight_data.shape)
            assert ch_axis < len(shape), 'ch_axis is pointing to an incorrect dimension'
            num_channels = shape.pop(ch_axis)

            # reshape weights based on the ch_axis - ch_axis has to be the first index to slice and be used for encoding
            weight_data = weight_data.transpose(
                AdaroundWrapper._generate_weight_transpose_perm(weight_data.shape, ch_axis))
            weight_data = np.ascontiguousarray(weight_data, weight_data.dtype)

            for ch_idx in range(num_channels):
                analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)
                analyzer.updateStats(weight_data[ch_idx], False)
                channel_encoding, _ = analyzer.computeEncoding(param_bw, is_symmetric, strict_symmetric,
                                                               unsigned_symmetric)
                encoding.append(channel_encoding)

        else:
            # Compute the encodings for the weight data using collected stats
            analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)
            analyzer.updateStats(weight_data, False)
            encoding, _ = analyzer.computeEncoding(param_bw, is_symmetric, strict_symmetric, unsigned_symmetric)

        return encoding

    def adaround_weights(self) -> tf.Tensor:
        """
        Adaround the weight tensor. Extra post-processing step if the layer is a DepthwiseConv2D
        :return: AdaRounded weight tensor
        """
        adaround_tensor = self._get_adarounded_weight(self.alpha, self._weight_tensor, self.encoding,
                                                      self.use_soft_rounding,
                                                      enable_per_channel=self.per_channel_enabled,
                                                      ch_axis=self.ch_axis)
        if self.per_channel_enabled and isinstance(self._layer, tf.keras.layers.DepthwiseConv2D):
            return tf.reshape(adaround_tensor, self._orig_weight_tensor_shape)
        return adaround_tensor

    @staticmethod
    def _get_adarounded_weight(alpha, weight_tensor, encoding, use_soft_rounding, enable_per_channel: bool,
                               ch_axis: int) -> tf.Tensor:
        """
        Get the adarounded weight

        :param alpha: Alpha parameter
        :param weight_tensor: Weight to adaround
        :param encoding: Encodings corresponding to weights
        :param use_soft_rounding: True if soft rounding is to be used, False if hard rounding is to be used
        :param enable_per_channel: True if per-channel mode, else False
        :param ch_axis: channel axis to be used in the per-channel mode
        :return: Adarounded weight tensor
        """

        # Soft rounding maps alpha parameter between zero and one using rectified sigmoid function
        def compute_soft_rounding():
            return tf.clip_by_value(tf.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                    AdaroundConstants.GAMMA, 0, 1)

        # Hard rounding maps alpha to exact zero or one
        def compute_hard_rounding():
            return tf.cast(alpha > 0, dtype=alpha.dtype)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

        if enable_per_channel:
            assert isinstance(encoding, list), "Per-channel expects encoding to be a list"

            delta = AdaroundWrapper._broadcast_to_tensor(weight_tensor, [enc.delta for enc in encoding], ch_axis)
            offset = AdaroundWrapper._broadcast_to_tensor(weight_tensor, [enc.offset for enc in encoding], ch_axis)
            bw = encoding[0].bw
        else:
            delta = encoding.delta
            offset = encoding.offset
            bw = encoding.bw

        # Scale the tensor
        tensor = tf.floor(weight_tensor / delta)

        # Compute h_alpha depending on soft or hard rounding
        h_alpha = tf.cond(use_soft_rounding, compute_soft_rounding, compute_hard_rounding)

        # Adaround the tensor
        tensor = tf.add(tensor, h_alpha)

        # Quantize and de-quantize the tensor
        tensor_quant = tf.clip_by_value(tensor - offset, 0, 2 ** bw - 1)
        tensor_dequant = (tensor_quant + offset) * delta

        return tensor_dequant

    @staticmethod
    def _get_channel_axis(layer: tf.keras.layers.Layer, shape: Tuple) -> int:
        """
        Get channel axis corresponding to the Keras layer
        :param layer: Keras layer to get channel axis for
        :param shape: Shape of the weight tensor of the layer
        :return: Channel axis for the Keras layer
        """
        ch_axis = len(shape) - 1
        if isinstance(layer, tf.keras.layers.Conv2DTranspose):
            ch_axis = 2
        return ch_axis

    def _compute_output_with_adarounded_weights(self, inp_tensor: tf.Tensor, adaround_weight_tensor: tf.Tensor) -> \
            tf.Tensor:
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param inp_tensor: The input tensor to be used for computing the output
        :param adaround_weight_tensor: The adarounded weight
        :return: output of the op computed with AdaRounded weights
        """
        if isinstance(self._layer, _supported_convs):
            kwargs = self._get_conv_args(self._layer)
            if isinstance(self._layer, tf.keras.layers.DepthwiseConv2D):
                adaround_out_tensor = tf.nn.depthwise_conv2d(inp_tensor, adaround_weight_tensor, **kwargs)
            elif isinstance(self._layer, tf.keras.layers.Conv2DTranspose):
                adaround_out_tensor = self._compute_output_with_adaround_weights_conv2d_transpose_helper(
                    self._output_height,
                    self._output_width,
                    self._output_channels,
                    inp_tensor,
                    adaround_weight_tensor,
                    **kwargs)
            else:
                adaround_out_tensor = tf.nn.conv2d(inp_tensor, adaround_weight_tensor, **kwargs)
        elif isinstance(self._layer, tf.keras.layers.Dense):
            adaround_out_tensor = tf.matmul(inp_tensor, adaround_weight_tensor)
        else:
            raise ValueError('Keras Layer: {} not supported'.format(self._layer))

        return adaround_out_tensor

    def _get_weight_and_bias_tensors(self, layer: tf.keras.layers.Layer) -> Tuple[Tuple, np.ndarray, np.ndarray]:
        """
        Function to properly grab the weight and bias tensor of a given Keras layer, as well as transform
        weights if needed.
        :param layer: Keras layer to have weights and bias extracted from
        :return: weight and bias tensor of Keras layer
        """
        weights = layer.get_weights()
        weight_tensor = weights[0]
        orig_weight_shape = weight_tensor.shape
        if self.per_channel_enabled and isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            weight_tensor = self._transform_input_ndarray_for_depthwise_conv_2d(weight_tensor)

        bias_tensor = None
        if len(weights) > 1:
            bias_tensor = weights[1]

        return orig_weight_shape, weight_tensor, bias_tensor

    @staticmethod
    def _get_conv_args(layer: tf.keras.layers.Conv2D) -> Dict:
        """
        :param op: Tf op of type Conv2d, Depthwise_Conv2d
        :return: keyword arguments
        """

        if layer.data_format == 'channels_last':
            data_format = 'NHWC'
            strides = [1, layer.strides[0], layer.strides[1], 1]
        else:
            data_format = 'NCHW'
            strides = [1, 1, layer.strides[0], layer.strides[1]]

        if layer.padding == 'valid':
            padding = 'VALID'
        else:
            padding = 'SAME'

        kwargs = {'data_format': data_format,
                  'strides': strides,
                  'padding': padding,
                  'dilations': layer.dilation_rate}
        return kwargs

    @staticmethod
    def _compute_output_with_adaround_weights_conv2d_transpose_helper(
            output_height, output_width, output_channels,
            inp_tensor, adaround_weight_tensor, **kwargs) -> tf.Tensor:
        """
        Compute output specificly for Conv2DTranpose layers for both tensorflow and keras
        (i.e. Conv2DTranspose, Conv2DBackpropInput)
        :param output_height: output height of the layer/op
        :param output_width: output width of the layer/op
        :param output_channels: output channels of layer/op
        :param inp_tensor: The input tensor to be used for computing output
        :param adaround_weight_tensor: the adarounded weight
        :param **kwargs: Other kwargs needed
        :return: output of the op computed with AdaROunded weights
        """
        assert output_height is not None, 'Output height required for conv2d transpose'
        assert output_width is not None, 'Output width required for conv2d transpose'
        assert output_channels is not None, 'Output channels required for conv2d transpose'

        if kwargs['data_format'] == 'NCHW':
            output_shape = (BATCH_SIZE, output_channels, output_height, output_width)
        else:
            output_shape = (BATCH_SIZE, output_height, output_width, output_channels)

        kwargs['output_shape'] = output_shape

        adaround_out_tensor = tf.nn.conv2d_transpose(inp_tensor, adaround_weight_tensor, **kwargs)
        return adaround_out_tensor

    @staticmethod
    def _calculate_alpha(tensor: tf.Tensor, encoding: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]],
                         enable_per_channel: bool, ch_axis: int) -> tf.Tensor:
        """
        Calculate alpha parameter for either per tensor or per channel
        :param tensor: The tensor to be ada rounded
        :param encoding: Encoding(s) for the tensor
        :param enable_per_channel: Flag for per channel to broadcoast the tensor
        :param ch_axis: Axis to broadcast if per channel is enabled
        :return: Adarounded output tensor
        """
        if enable_per_channel:
            assert isinstance(encoding, list), "Per-channel expects encoding to be a list"
            delta = AdaroundWrapper._broadcast_to_tensor(tensor, [enc.delta for enc in encoding], ch_axis)
        else:
            delta = encoding.delta

        tensor_floor = tf.floor(tensor / delta)
        tensor = (tensor / delta) - tensor_floor
        # pylint: disable=invalid-unary-operand-type
        return -tf.math.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

    @staticmethod
    def _transform_input_ndarray_for_depthwise_conv_2d(input_arr: Union[np.ndarray]) -> Union[np.ndarray]:
        """
        For DepthwiseConv2d op, if per-channel is enabled, we need to use the last two axes as channel axis.
        This helper function basically merges the last two dimensions into one, so that the rest of the flow would work
        just fine
        Example: if tf.shape = (2, 2, 3, 8), output tensor = (2, 2, 24)

        :param input_arr: input array of type np.ndarray which needs to be transformed to the new shape
        """

        assert len(input_arr.shape) >= 3
        shape = list(input_arr.shape[:-2])
        shape.append(input_arr.shape[-2] * input_arr.shape[-1])
        return input_arr.reshape(tuple(shape))

    @staticmethod
    def _generate_weight_transpose_perm(shape: tuple, ch_axis: int) -> List:
        """
        Given shape of tensor/np.ndarray and channel axis, this function generates the permutation list to be used for
        the transpose operation of the tensor/np.ndarray
        shape = (A, B, C, D)
        ch_axis = 2
        return = (C, A, B, D)
        :param shape: tuple representing the shape of the tensor/np.ndarray
        :ch_axis: dimension to be used for per channel quantization
        :return permutation list
        """
        perm = list(range(len(shape)))
        ch_dim = perm.pop(ch_axis)
        # make ch_idx dimension the first one
        perm.insert(0, ch_dim)
        return perm

    @staticmethod
    def _broadcast_to_tensor(tensor: tf.Tensor, encoding: list, ch_axis: int) -> tf.constant:
        """
        Broadcast per-channel delta/offset using the encodings array

        :param tensor: The weight tensor to be ada-rounded
        :param encoding: list of per-channel encoding delta/offset to generate broadcasted encoding
        :param ch_axis: dimension to be used for per channel quantization
        """
        def _get_broadcast_shape() -> List:
            """
            compute the broadcast shape based on the channel index
            """
            shape = list(tensor.shape)
            channels = shape.pop(ch_axis)
            broadcast_shape = shape + [channels]
            return broadcast_shape

        def _get_encoding_rotate_perm() -> List:
            """
            Generate the permutation list to apply on delta/offset(which is broadcasted) to match the original shape
            """
            length = len(list(tensor.shape))
            ret_perm = list(range(length))
            channel_swap = ret_perm.pop()
            ret_perm.insert(ch_axis, channel_swap)
            return ret_perm

        tensor_encoding = tf.constant(encoding, dtype=tensor.dtype)
        # broadcast delta/offset of shape (num_channels,) to broadcast_shape
        tensor_encoding = tf.broadcast_to(tensor_encoding, _get_broadcast_shape())
        tensor_encoding = tf.transpose(tensor_encoding, perm=_get_encoding_rotate_perm())
        return tensor_encoding
