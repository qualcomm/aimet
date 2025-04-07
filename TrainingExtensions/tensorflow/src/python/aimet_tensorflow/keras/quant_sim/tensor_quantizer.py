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
# pylint: disable=too-many-lines
""" Tensor quantizer for tf 2 keras """
import abc
import functools
from typing import List, Optional, Tuple, Callable
import tensorflow as tf
import tensorflow.keras.backend as K

from aimet_common import libpymo
import aimet_common.libaimet_tf_ops as qcops

from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantScheme, QuantizationDataType, \
    RANGE_LEARNING_SCHEMES
from aimet_common.quantsim import calculate_delta_offset, compute_min_max_given_delta_offset
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.defs import AxisHandling
from aimet_tensorflow.keras.quant_sim.quantsim_straight_through_grad import qc_straight_through_estimator_grad, \
    quantsim_custom_grad_learned_grid, quantsim_per_channel_custom_grad_learned_grid
import aimet_tensorflow.keras.utils.common as keras_common_utils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def _handle_conv2d_transpose(callback):
    @functools.wraps(callback)
    def _handle(cls, tensor):
        if isinstance(cls.original_layer, tf.keras.layers.Conv2DTranspose):
            if len(tensor.shape) == 4:
                # Transpose input tensor, pass to qc_quantize_per_channel, transpose result back
                # Permute dimensions used to transpose the input tensor to a dimensionality that
                # the underlying C++ Op is expecting for this type of axis handling.
                # HWOI -> HWIO
                permute = [0, 1, 3, 2]
                tensor = K.permute_dimensions(tensor, permute)

                # TODO: Workaround until gradient support for non-range learning is implemented
                return_val = callback(cls, tensor)
                if isinstance(return_val, tuple):
                    # If the function returns both quantized tensor and gradient
                    # In the case of call_quantsim_custom_grad_learned_grid
                    return K.permute_dimensions(return_val[0], permute), return_val[1]

                # If the function returns just the quantized tensor
                # In the case of call_per_channel_quantize_dequantize
                return K.permute_dimensions(return_val, permute)

        return callback(cls, tensor)

    return _handle


def _update_grad_to_tf_tensor_if_needed(grad_func: Callable):
    """
    Decorator function to convert gradient tensors that are represented as tf.IndexedSlices into
    tf.Tensor's. This typically occurs with operations such as tf.gather and keras.Embedding layers.

    :param grad_func: The gradient function that will be called after conversion
    """
    def wrapper(*args, **kwargs):
        grad = args[0]
        if isinstance(grad, tf.IndexedSlices):
            _logger.debug("Converting %s from tf.IndexedSlices to tf.Tensor", grad.name)
            args = (tf.convert_to_tensor(grad),) + args[1:]
        return grad_func(*args, **kwargs)
    return wrapper

class TensorQuantizer(tf.keras.layers.Layer, abc.ABC):
    """Tensor quantizer class containing the bare bones of a given Quantizer"""

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    # pylint: disable=unused-argument
    def __init__(self, layer: tf.keras.layers.Layer, name: str, op_mode: libpymo.TensorQuantizerOpMode,
                 quant_scheme: QuantScheme,
                 bitwidth: int, data_type: QuantizationDataType, is_symmetric: bool, **kwargs):
        super().__init__(name=name)
        # Original layer is here to handle the case if the wrapped layer is a Conv2DTranspose and therefore
        # needs its tensor to be transposed before given to the QcPerChannelOp
        self._original_layer = layer
        self._quant_scheme = quant_scheme
        self._bitwidth = self.add_weight(name + '.bitwidth', dtype=tf.int8,
                                         initializer=tf.constant_initializer(bitwidth), trainable=False)
        self._is_int_data_type = self.add_weight(name + '.data_type', dtype=tf.bool,
                                                 initializer=tf.constant_initializer(
                                                     (data_type == QuantizationDataType.int)),
                                                 trainable=False)
        self._is_symmetric = self.add_weight(name + '.is_symmetric', dtype=tf.bool,
                                             initializer=tf.constant_initializer(is_symmetric), trainable=False)

        self._quantizer_mode = self.add_weight(name + '.op_mode', dtype=tf.int32, trainable=False,
                                               initializer=tf.constant_initializer(int(op_mode)))

        # Use this flag to determine if encoding min and max values are fit to be used. Ex. Can be set to True after
        # compute encodings has been called, or after encodings have been set by passing in a libpymo TfEncoding object.
        # Can set to False upon changing things like quant scheme, bitwidth, is symmetric, etc.
        self._is_encoding_valid = False

        # Behavior of frozen encoding:
        # - quant_scheme, round_mode, bitwidth, is_symmetric, use_strict_symmetric, use_unsigned_symmetric, encoding,
        #       and quant_mode cannot be changed.
        # - compute_encoding(), reset_quant_mode() will take no effect.
        # - Tensor quantizer cannot be disabled.
        self._is_encoding_frozen = False

    @property
    def original_layer(self):
        """ Original layer wrapped by quantizer """
        return self._original_layer

    @property
    def quant_scheme(self):
        """ Quant scheme getter """
        return self._quant_scheme

    @quant_scheme.setter
    @abc.abstractmethod
    def quant_scheme(self, quant_scheme: QuantScheme):
        """ Quant scheme setter """

    @property
    def bitwidth(self):
        """ Bitwidth getter """
        return tf.keras.backend.get_value(self._bitwidth)

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """ Bitwidth setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting bitwidth.', self.name)
            return
        self._bitwidth.assign(bitwidth)
        self.reset_quant_mode()

    @property
    def data_type(self):
        """ Bitwidth getter """
        return QuantizationDataType.int if tf.keras.backend.get_value(self._is_int_data_type) \
            else QuantizationDataType.float

    @data_type.setter
    def data_type(self, data_type: QuantizationDataType):
        """ Bitwidth setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting data type.', self.name)
            return

        self._is_int_data_type.assign(data_type == QuantizationDataType.int)
        self.reset_quant_mode()

    @property
    def is_symmetric(self):
        """ Is symmetric getter """
        return tf.keras.backend.get_value(self._is_symmetric)

    @is_symmetric.setter
    def is_symmetric(self, is_symmetric: bool):
        """ Is symmetric setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting is_symmetric.', self.name)
            return
        self._is_symmetric.assign(is_symmetric)
        self.reset_quant_mode()

    def freeze_encoding(self):
        """
        Freeze the encoding
        """
        self._is_encoding_frozen = True

    @property
    def quant_mode(self):
        """ Get quant mode """
        return tf.keras.backend.get_value(self._quantizer_mode)

    @quant_mode.setter
    def quant_mode(self, quant_mode: libpymo.TensorQuantizerOpMode):
        """ Quant mode setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting quant_mode.', self.name)
            return
        self._quantizer_mode.assign(int(quant_mode))

    def disable(self):
        """ Disable the tensor quantizer """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not disabling quantizer.', self.name)
            return
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.passThrough))

    def is_enabled(self) -> bool:
        """ Return True if the tensor quantizer is enabled, False otherwise """
        return self.quant_mode != int(libpymo.TensorQuantizerOpMode.passThrough)

    def is_encoding_valid(self):
        """ Returns the status of the encodings validity"""
        return self._is_encoding_valid

    def reset_quant_mode(self):
        """ Reset quantizer mode if applicable """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not resetting quant_mode.', self.name)
            return
        if self.quant_mode == int(libpymo.TensorQuantizerOpMode.quantizeDequantize):
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.updateStats))
        self._is_encoding_valid = False

    @abc.abstractmethod
    def _call_handler(self, tensor):
        """
        Equivalent to `tf.keras.Layers.call` function as it is called from `call` function
        `call` function handles passThrough at the top level
        """

    # pylint: disable=arguments-differ, arguments-renamed
    def call(self, tensor):
        """
        Forward pass for the quantizer
        """
        if self.quant_mode == libpymo.TensorQuantizerOpMode.passThrough.value:
            return tensor
        return self._call_handler(tensor)

    def set_quantizer_encodings(self, bitwidth: int, is_symmetric: bool, encoding: libpymo.TfEncoding,
                                opmode: libpymo.TensorQuantizerOpMode):
        """
        Helper Function to set encodings, bitwidth, opmode, symmetric flag and opmode to tensor quantizer
        :param bitwidth: Bitwidth for the tensor quantizer
        :param is_symmetric: True if symmetric encoding is used. False otherwise.
        :param encoding: encodings which needs to be applied to the quantizer
        :param opmode: operation mode for the quantizer
        """
        # pylint: disable  = attribute-defined-outside-init
        self.is_symmetric = is_symmetric
        self.bitwidth = bitwidth
        self.encoding = encoding
        self.quant_mode = opmode


# pylint: disable=too-many-ancestors
class StaticGridPerTensorQuantizer(TensorQuantizer):
    """ Class that represents encodings on a per-tensor basis """

    # pylint: disable=too-many-arguments
    def __init__(self, layer: tf.keras.layers.Layer, name: str, op_mode: libpymo.TensorQuantizerOpMode,
                 quant_scheme: QuantScheme, round_mode: str, bitwidth: int, data_type: QuantizationDataType,
                 is_symmetric: bool, use_strict_symmetric: bool, use_unsigned_symmetric: bool):

        super().__init__(layer, name, op_mode, quant_scheme, bitwidth, data_type, is_symmetric)

        self._encoding_min = self.add_weight(name + '.encoding_min', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.))
        self._encoding_max = self.add_weight(name + '.encoding_max', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.))

        self._tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme],
                                                         MAP_ROUND_MODE_TO_PYMO[round_mode])
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._encoding_min_max_fixed_vals = None

    @TensorQuantizer.quant_scheme.setter
    def quant_scheme(self, quant_scheme: QuantScheme):
        """ Quant scheme setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting quant_scheme.', self.name)
            return
        self._tensor_quantizer.setQuantScheme(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme])
        self._quant_scheme = quant_scheme
        self.reset_quant_mode()

    @property
    def tensor_quantizer(self):
        """ Tensor quantizer getter """
        return self._tensor_quantizer

    @property
    def use_strict_symmetric(self):
        """ Use strict symmetric getter """
        return self._tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """ Use strict symmetric setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting use_strict_symmetric.', self.name)
            return
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self.reset_quant_mode()

    @property
    def use_unsigned_symmetric(self):
        """ Use unsigned symmetric getter """
        return self._tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """ Use unsigned symmetric setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting use_unsigned_symmetric.', self.name)
            return
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self.reset_quant_mode()

    @property
    def round_mode(self):
        """ Quant scheme getter """
        return self._tensor_quantizer.roundingMode

    @round_mode.setter
    def round_mode(self, round_mode: str):
        """ Round mode setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting round_mode.', self.name)
            return
        self._tensor_quantizer.roundingMode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        self.reset_quant_mode()

    @property
    def encoding(self) -> Optional[libpymo.TfEncoding]:
        """ Get encodings in libpymo form """
        if self._is_encoding_valid:
            encodings = libpymo.TfEncoding()
            # pylint: disable = protected-access
            encodings_min = tf.keras.backend.get_value(self._encoding_min)
            encodings_max = tf.keras.backend.get_value(self._encoding_max)
            encodings.delta, encodings.offset = calculate_delta_offset(encodings_min, encodings_max,
                                                                       self.bitwidth, self.is_symmetric,
                                                                       self.use_strict_symmetric)
            encodings.min, encodings.max = compute_min_max_given_delta_offset(encodings.delta, encodings.offset,
                                                                              self.bitwidth, self.is_symmetric,
                                                                              self.use_strict_symmetric)
            encodings.bw = self.bitwidth
            return encodings
        return None

    @encoding.setter
    @abc.abstractmethod
    def encoding(self, encoding: libpymo.TfEncoding):
        pass

    @abc.abstractmethod
    def enable(self):
        """ Enable the tensor quantizer """

    @property
    def encoding_min(self) -> tf.Variable:
        """Return the encoding_min variable"""
        return self._encoding_min

    @property
    def encoding_max(self) -> tf.Variable:
        """Return the encoding_max variable"""
        return self._encoding_max

    def _set_encoding_values(self, encoding: libpymo.TfEncoding):
        """
        Set encoding values.
        :param encoding: Encoding containing values to set
        """
        assert encoding is not None, "Encodings cannot be None if Quantizer is enabled"
        assert isinstance(encoding, libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
        self.bitwidth = encoding.bw
        self._encoding_min.assign(encoding.min)
        self._encoding_max.assign(encoding.max)
        self._is_encoding_valid = True

    def set_fixed_encoding_range(self, fixed_range: Tuple[float, float]):
        """
        Set the min/max values to be used when computing encodings

        :param fixed_range: Tuple of (min, max) value to use in-place of observer statistics when computing encodings
        """
        self._encoding_min_max_fixed_vals = fixed_range

    def compute_encoding(self, ops_with_invalid_encodings: List = None):
        """ Compute encoding for the tensor quantizer """
        if self.is_enabled() and not self._is_encoding_frozen:

            if self._encoding_min_max_fixed_vals is None:
                encoding = self._tensor_quantizer.computeEncoding(self.bitwidth, self.is_symmetric)
                is_valid_encoding = self._tensor_quantizer.isEncodingValid
            else:
                min_val, max_val = self._encoding_min_max_fixed_vals
                encoding = libpymo.TfEncoding()
                encoding.bw = self.bitwidth
                encoding.min = min_val
                encoding.max = max_val
                partial_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF, self.round_mode)
                partial_quantizer.computePartialEncoding(encoding.bw, encoding, self.is_symmetric,
                                                         self.use_unsigned_symmetric, self.use_strict_symmetric)
                is_valid_encoding = True

            # TODO: remove last two parameters after fixing PyModelOptimizations
            if self.data_type == QuantizationDataType.float:
                self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
            else:
                if is_valid_encoding:  # pylint: disable=using-constant-test
                    if self._quant_scheme in RANGE_LEARNING_SCHEMES and self.is_symmetric and encoding.min != 0:
                        self._encoding_min.assign(-encoding.max)
                    else:
                        self._encoding_min.assign(encoding.min)
                    self._encoding_max.assign(encoding.max)

                    if self.quant_mode == int(libpymo.TensorQuantizerOpMode.updateStats):
                        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
                    self._is_encoding_valid = True
                else:
                    _logger.info('Tensor quantizer %s did not have a valid encoding calculated, and has been set to '
                                 'passThrough mode.', self.name)
                    ops_with_invalid_encodings.append(self.name)
                    self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.passThrough))

    # pylint: disable=arguments-differ
    def _call_handler(self, tensor: tf.Tensor):
        if self.quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init]:
            return self.call_quantsim_custom_grad_learned_grid(tensor)
        return self.call_quantize_straight_through_estimator_grad(tensor)

    @tf.custom_gradient
    def call_quantize_straight_through_estimator_grad(self, tensor: tf.Tensor):
        """
        Quantizes tensor with straight through estimator grad
        :param tensor: Tensor to quantize
        """

        @_update_grad_to_tf_tensor_if_needed
        def grad(upstream: tf.Tensor, variables: List):
            """
            Straight through estimator grad function
            :param upstream: Gradient from child layers
            :param variables: Variables used in forward pass to return gradients for
            """
            assert len(variables) == 2, 'len variables is ' + str(len(variables))

            return qc_straight_through_estimator_grad(tensor, self._encoding_min, self._encoding_max,
                                                      self._quantizer_mode, upstream)

        return qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                 op_mode=self._quantizer_mode,
                                 tensor_quantizer_reference=libpymo.PtrToInt64(self._tensor_quantizer),
                                 encoding_min=self._encoding_min,
                                 encoding_max=self._encoding_max,
                                 bit_width=self._bitwidth,
                                 use_symmetric_encoding=self._is_symmetric,
                                 is_int_data_type=self._is_int_data_type), grad

    @tf.custom_gradient
    def call_quantsim_custom_grad_learned_grid(self, tensor: tf.Tensor):
        """
        Quantizes tensor with range learning grad
        :param tensor: Tensor to quantize
        """

        @_update_grad_to_tf_tensor_if_needed
        def grad(upstream: tf.Tensor, variables: List):
            """
            Range learning grad function
            :param upstream: Gradient from child layers
            :param variables: Variables used in forward pass to return gradients for
            """
            assert len(variables) == 2, 'len variables is ' + str(len(variables))

            (dloss_by_dx, (dloss_by_dmin, dloss_by_dmax)) = quantsim_custom_grad_learned_grid(tensor,
                                                                                              self._encoding_min,
                                                                                              self._encoding_max,
                                                                                              self._quantizer_mode,
                                                                                              self._bitwidth,
                                                                                              self._is_symmetric,
                                                                                              upstream)
            # To account for the difference in the order of variables between TF 2.4 and TF 2.10
            if 'encoding_max' in variables[0].name:
                return dloss_by_dx, [dloss_by_dmax, dloss_by_dmin]

            return dloss_by_dx, [dloss_by_dmin, dloss_by_dmax]

        return qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                 op_mode=self._quantizer_mode,
                                 tensor_quantizer_reference=libpymo.PtrToInt64(self._tensor_quantizer),
                                 encoding_min=self._encoding_min,
                                 encoding_max=self._encoding_max,
                                 bit_width=self._bitwidth,
                                 use_symmetric_encoding=self._is_symmetric,
                                 is_int_data_type=self._is_int_data_type), grad

    def get_gradients_for_encoding_min_max(self, weight_tensor: tf.Tensor, grad: tf.Tensor):
        """
        Compute the gradients for encoding min/max using weights and their gradients
        :param weight_tensor: Weight for which encoding min/max gradients are needed
        :param grad: Gradients of the weights
        """
        _, [dloss_by_dmin, dloss_by_dmax] = quantsim_custom_grad_learned_grid(weight_tensor, self._encoding_min,
                                                                              self._encoding_max, self._quantizer_mode,
                                                                              self._bitwidth, self.is_symmetric, grad)

        return dloss_by_dmin, dloss_by_dmax

    def get_stats_histogram(self) -> List[List[Tuple]]:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Get histogram of statistics. Returns list of buckets where each bucket is
        tuple of two values - the float value representing the left edge of the
        bucket and a PDF of the values in this bucket relative to all the values
        seen across all buckets.

        :return: List of buckets where each bucket is (xLeft, PDF).
        """
        if self._quant_scheme != QuantScheme.post_training_tf_enhanced:
            raise RuntimeError("get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced.")

        if not self.is_encoding_valid():
            raise RuntimeError("get_stats_histogram() can be invoked only when encoding is computed.")

        # Return a list of histograms for compatability with per-channel case
        histograms = [self.tensor_quantizer.getStatsHistogram()]
        return histograms

    def set_percentile_value(self, percentile: float):
        """
        Sets the percentile value to the tensor quantizer only in case of percentile quant scheme

        :param percentile: Percentile value to set
        """
        if self._quant_scheme != QuantScheme.post_training_percentile:
            raise RuntimeError("set_percentile_value() can be invoked only when quantization scheme is Percentile.")

        self.tensor_quantizer.setPercentileValue(percentile)

    def get_percentile_value(self) -> float:
        """
        Fetches the percentile value to the tensor quantizer only in case of percentile quant scheme.

        :return Percentile value of the quantizer
        """
        if self._quant_scheme != QuantScheme.post_training_percentile:
            raise RuntimeError("get_percentile_value() can be invoked only when quantization scheme is Percentile.")

        return self.tensor_quantizer.getPercentileValue()


# pylint: disable=too-many-ancestors
class ParamPerTensorQuantizer(StaticGridPerTensorQuantizer):
    """ Parameter tensor quantizer definition """

    # pylint: disable=too-many-arguments
    def __init__(self, layer: tf.keras.layers.Layer, name: str, quant_scheme: QuantScheme, round_mode: str,
                 bitwidth: int, data_type: QuantizationDataType, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, enabled: bool):
        keras_common_utils.log_param_quantizer_wrapper_details(layer)
        if enabled:
            op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize
        else:
            op_mode = libpymo.TensorQuantizerOpMode.passThrough

        super().__init__(layer, name, op_mode, quant_scheme, round_mode, bitwidth, data_type, is_symmetric,
                         use_strict_symmetric, use_unsigned_symmetric)

    def enable(self):
        """ Enable the parameter tensor quantizer """
        # If encoding is frozen, no need to do anything (and quant mode should already be set to quantizeDequantize,
        # instead of oneShotQuantizeDequantize)
        if not self._is_encoding_frozen:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))

    @StaticGridPerTensorQuantizer.encoding.setter
    def encoding(self, encoding: libpymo.TfEncoding):
        """
        Sets encoding parameter using values obtained from encodings
        :param encoding: encodings value
        """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting new encoding values.', self.name)
            return
        self._set_encoding_values(encoding)
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))


# pylint: disable=too-many-ancestors
class ActivationTensorQuantizer(StaticGridPerTensorQuantizer):
    """ Activation tensor quantizer definition """

    # pylint: disable=too-many-arguments
    def __init__(self, layer: tf.keras.layers.Layer, name: str, quant_scheme: QuantScheme, round_mode: str,
                 bitwidth: int, data_type: QuantizationDataType, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, enabled: bool):

        if enabled:
            op_mode = libpymo.TensorQuantizerOpMode.updateStats
        else:
            op_mode = libpymo.TensorQuantizerOpMode.passThrough

        super().__init__(layer, name, op_mode, quant_scheme, round_mode, bitwidth, data_type, is_symmetric,
                         use_strict_symmetric, use_unsigned_symmetric)

    def enable(self):
        """ Enable the activation tensor quantizer """
        if self._is_encoding_valid:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
        else:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.updateStats))

    @StaticGridPerTensorQuantizer.encoding.setter
    def encoding(self, encoding: libpymo.TfEncoding):
        """
        Sets encoding parameter using values obtained from encodings
        :param encoding: encodings value
        """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting new encoding values.', self.name)
            return
        self._set_encoding_values(encoding)
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))


class StaticGridPerChannelQuantizer(TensorQuantizer):
    """ Class that represents encodings on a per-channel basis """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def __init__(self, layer: tf.keras.layers.Layer, name: str, op_mode: libpymo.TensorQuantizerOpMode,
                 quant_scheme: QuantScheme, round_mode: str, bitwidth: int, data_type: QuantizationDataType,
                 is_symmetric: bool, use_strict_symmetric: bool, use_unsigned_symmetric: bool,
                 axis_handling: AxisHandling, num_output_channels: int):

        super().__init__(layer, name, op_mode, quant_scheme, bitwidth, data_type, is_symmetric)

        self.axis_handling = axis_handling.value
        self._encoding_max = self.add_weight(name + '.encoding_max', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.), shape=(num_output_channels,))
        self._encoding_min = self.add_weight(name + '.encoding_min', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.), shape=(num_output_channels,))

        tensor_quantizer_int64 = []
        tensor_quantizers = []
        for _ in range(num_output_channels):
            tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme],
                                                       MAP_ROUND_MODE_TO_PYMO[round_mode])
            tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
            tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)

            tensor_quantizers.append(tensor_quantizer)
            tensor_quantizer_int64.append(libpymo.PtrToInt64(tensor_quantizer))

        self._tensor_quantizer = tensor_quantizers
        self._tensor_quantizer_int64 = tensor_quantizer_int64

    @TensorQuantizer.quant_scheme.setter
    def quant_scheme(self, quant_scheme: QuantScheme):
        """ Quant scheme setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting quant_scheme.', self.name)
            return
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setQuantScheme(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme])
        self._quant_scheme = quant_scheme
        self.reset_quant_mode()

    @property
    def tensor_quantizer(self):
        """ Tensor quantizer getter """
        return self._tensor_quantizer

    @property
    def round_mode(self):
        """ Rounding mode for each tensor quantizer """
        # All the tensor quantizers have the same roundingMode.
        round_mode = self._tensor_quantizer[0].roundingMode
        for tensor_quantizer in self._tensor_quantizer:
            assert tensor_quantizer.roundingMode == round_mode, \
                f"Not all libpymo.TensorQuantizer's have the same round_mode for original layer {self._original_layer.name}. \
                    Expected: {round_mode}. Got {tensor_quantizer.roundingMode}"
        return round_mode

    @round_mode.setter
    def round_mode(self, round_mode: str):
        """ Round mode setter for each tensor quantizer"""
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encodings are frozen, not setting round_mode', self.name)
            return
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.roundingMode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        self.reset_quant_mode()

    @property
    def use_strict_symmetric(self):
        """ Use strict symmetric getter """
        # All the tensor quantizers have the same strictSymmetric.
        use_strict_symmetric = self._tensor_quantizer[0].getStrictSymmetric()
        for tensor_quantizer in self._tensor_quantizer:
            assert tensor_quantizer.getStrictSymmetric() == use_strict_symmetric, \
                f"Not all libpymo.TensorQuantizer's have the same strictSymmetric setting for original layer {self._original_layer.name}. \
                    Expected: {use_strict_symmetric}. Got {tensor_quantizer.getStrictSymmetric()}"
        return use_strict_symmetric

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """ Use strict symmetric setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting use_strict_symmetric.', self.name)
            return
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self.reset_quant_mode()

    @property
    def use_unsigned_symmetric(self):
        """ Use unsigned symmetric getter """
        # All the tensor quantizers have the same unsignedSymmetric.
        use_unsigned_symmetric = self._tensor_quantizer[0].getUnsignedSymmetric()
        for tensor_quantizer in self._tensor_quantizer:
            assert tensor_quantizer.getUnsignedSymmetric() == use_unsigned_symmetric, \
                f"Not all libpymo.TensorQuantizer's have the same unsignedSymmetric setting for original layer {self._original_layer.name}. \
                    Expected: {use_unsigned_symmetric}. Got {tensor_quantizer.getUnsignedSymmetric()}"
        return use_unsigned_symmetric

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """ Use unsigned symmetric setter """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting use_unsigned_symmetric.', self.name)
            return
        for tensor_quantizer in self.tensor_quantizer:
            tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self.reset_quant_mode()

    @property
    def encoding(self) -> Optional[List[libpymo.TfEncoding]]:
        """ Get encodings in libpymo form """
        if self._is_encoding_valid:
            total_number_of_encodings = self._encoding_min.shape[0]

            # Get all backend tensors up front
            all_keras_backend_encoding_mins, all_keras_backend_encoding_maxs = \
                tf.keras.backend.batch_get_value([self._encoding_min, self._encoding_max])

            # Create all encoding objects upfront and then update each one's properties
            all_tf_encoding_objects = list(libpymo.TfEncoding() for _ in range(total_number_of_encodings))

            # Get all use_strict_symmetric up front, we loop through all the tensorQuantizers to validate they have the
            # same setting. Here, we call it once to reduce computation.
            use_strict_symmetric = self.use_strict_symmetric

            for idx, encoding in enumerate(all_tf_encoding_objects):
                encoding.min = all_keras_backend_encoding_mins[idx]
                encoding.max = all_keras_backend_encoding_maxs[idx]
                encoding.delta, encoding.offset = calculate_delta_offset(encoding.min, encoding.max,
                                                                         self.bitwidth, self.is_symmetric,
                                                                         use_strict_symmetric)

                encoding.min, encoding.max = compute_min_max_given_delta_offset(encoding.delta, encoding.offset,
                                                                                self.bitwidth, self.is_symmetric,
                                                                                use_strict_symmetric)
                encoding.bw = self.bitwidth

            return all_tf_encoding_objects

        return None

    @encoding.setter
    @abc.abstractmethod
    def encoding(self, encoding: libpymo.TfEncoding):
        pass

    @property
    def encoding_min(self) -> tf.Variable:
        """Return the encoding_min variable"""
        return self._encoding_min

    @property
    def encoding_max(self) -> tf.Variable:
        """Return the encoding_min variable"""
        return self._encoding_max

    @abc.abstractmethod
    def enable(self):
        """ Enable the tensor quantizer """

    def _set_encoding_values(self, encodings: List[libpymo.TfEncoding]):
        """
        Set encodings values.
        :param encoding: Encoding containing values to set
        """
        for i, encoding in enumerate(encodings):
            assert encoding is not None, "Encodings cannot be None if Quantizer is enabled"
            assert isinstance(encoding, libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
            self.bitwidth = encoding.bw
            self._encoding_min[i].assign(encoding.min)
            self._encoding_max[i].assign(encoding.max)
            self._is_encoding_valid = True

    def compute_encoding(self, ops_with_invalid_encodings: List = None):
        """ Compute encodings per tensor quantizer representing each channel """
        for i, tensor_quantizer, in enumerate(self._tensor_quantizer):
            if self.is_enabled() and not self._is_encoding_frozen:
                # TODO: remove last two parameters after fixing PyModelOptimizations
                encoding = tensor_quantizer.computeEncoding(self.bitwidth, self.is_symmetric)
                if tensor_quantizer.isEncodingValid:
                    # In the case of range learning signed symmetric, set min = max for symmetric updates.
                    if self._quant_scheme in RANGE_LEARNING_SCHEMES and self.is_symmetric and encoding.min != 0:
                        self._encoding_min[i].assign(-encoding.max)
                    else:
                        self._encoding_min[i].assign(encoding.min)
                    self._encoding_max[i].assign(encoding.max)
                    if self.quant_mode == int(libpymo.TensorQuantizerOpMode.updateStats):
                        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
                    self._is_encoding_valid = True
                else:
                    _logger.info(
                        'Tensor quantizer %s did not have a valid encoding calculated, and has been set to '
                        'passThrough mode.', self.name)
                    ops_with_invalid_encodings.append(self.name)
                    self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.passThrough))

    def _call_handler(self, tensor):
        """
        :param tensor: Tensor to quantize
        """
        if self.quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init]:
            return self.call_quantsim_custom_grad_learned_grid(tensor)
        return self.call_per_channel_quantize_dequantize(tensor)

    @tf.custom_gradient
    @_handle_conv2d_transpose
    def call_quantsim_custom_grad_learned_grid(self, tensor: tf.Tensor):
        """
        Per-channel quantization with range learning
        :param tensor: Tensor to quantize
        :return: Per-channel quantized tensor, gradient function
        """

        # pylint: disable=unused-argument
        def grad(upstream, variables):
            """
            :param upstream: Gradient from child layers
            :param variables: Variables used in forward pass to return gradients for
            :return: Per-channel quantized tensor, gradient function
            """
            return quantsim_per_channel_custom_grad_learned_grid(inputs=tensor,
                                                                 encoding_min=self._encoding_min,
                                                                 encoding_max=self._encoding_max,
                                                                 op_mode=self._quantizer_mode,
                                                                 bitwidth=self._bitwidth,
                                                                 is_symmetric=self.is_symmetric,
                                                                 is_int_data_type=self._is_int_data_type,
                                                                 axis_handling=self.axis_handling,
                                                                 grad=upstream)

        return qcops.qc_quantize_per_channel(name='qc_quantize_per_channel_op',
                                             in_tensor=tensor,
                                             op_mode=self._quantizer_mode,
                                             tensor_quantizer_reference=self._tensor_quantizer_int64,
                                             encoding_min=self._encoding_min,
                                             encoding_max=self._encoding_max,
                                             bit_width=self._bitwidth,
                                             use_symmetric_encoding=self._is_symmetric,
                                             is_int_data_type=self._is_int_data_type,
                                             axis_handling=self.axis_handling,
                                             # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                                             is_training=tf.cast(tf.keras.backend.learning_phase(),
                                                                 dtype=tf.bool)), grad

    @_handle_conv2d_transpose
    def call_per_channel_quantize_dequantize(self, tensor: tf.Tensor):
        """
        Quantize tensor with range learning grad
        :param tensor: Tensor to quantize
        """

        return qcops.qc_quantize_per_channel(name='qc_quantize_per_channel_op', in_tensor=tensor,
                                             op_mode=self._quantizer_mode,
                                             tensor_quantizer_reference=self._tensor_quantizer_int64,
                                             encoding_min=self._encoding_min,
                                             encoding_max=self._encoding_max,
                                             bit_width=self._bitwidth,
                                             use_symmetric_encoding=self._is_symmetric,
                                             is_int_data_type=self._is_int_data_type,
                                             axis_handling=self.axis_handling,
                                             is_training=bool(tf.keras.backend.learning_phase()))

    def get_gradients_for_encoding_min_max(self, weight_tensor: tf.Tensor, grad: tf.Tensor):
        """
        Compute the gradients for encoding min/max using weights and their gradients
        :param weight_tensor: Weight for which encoding min/max gradients are needed
        :param grad: Gradients of the weights
        """
        gradients = quantsim_per_channel_custom_grad_learned_grid(weight_tensor,
                                                                  self._encoding_min,
                                                                  self._encoding_max,
                                                                  self._quantizer_mode,
                                                                  self._bitwidth,
                                                                  self.is_symmetric,
                                                                  self._is_int_data_type,
                                                                  self.axis_handling,
                                                                  grad)

        return gradients[3], gradients[4]

    def get_stats_histogram(self) -> List[List[Tuple]]:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Get histogram of statistics. Returns list of buckets where each bucket is
        tuple of two values - the float value representing the left edge of the
        bucket and a PDF of the values in this bucket relative to all the values
        seen across all buckets.

        :return: List of buckets where each bucket is (xLeft, PDF).
        """
        if self._quant_scheme != QuantScheme.post_training_tf_enhanced:
            raise RuntimeError("get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced.")

        if not self.is_encoding_valid():
            raise RuntimeError("get_stats_histogram() can be invoked only when encoding is computed.")

        histograms = [quantizer.getStatsHistogram() for quantizer in self.tensor_quantizer]
        return histograms

    def set_percentile_value(self, percentile: float):
        """
        Sets the percentile value to the tensor quantizer only in case of percentile quant scheme

        :param percentile: Percentile value to set
        """
        if self._quant_scheme != QuantScheme.post_training_percentile:
            raise RuntimeError("set_percentile_value() can be invoked only when quantization scheme is Percentile.")
        for quantizer in self.tensor_quantizer:
            quantizer.setPercentileValue(percentile)

    def get_percentile_value(self) -> List[float]:
        """
        Fetches the percentile value to the tensor quantizer only in case of percentile quant scheme.

        :return Percentile value of the quantizer
        """

        if self._quant_scheme != QuantScheme.post_training_percentile:
            raise RuntimeError("set_percentile_value() can be invoked only when quantization scheme is Percentile.")

        return [quantizer.getPercentileValue() for quantizer in self.tensor_quantizer]


# pylint: disable=too-many-ancestors
class ParamPerChannelQuantizer(StaticGridPerChannelQuantizer):
    """ Parameter per channel quantizer definition """

    # pylint: disable=too-many-arguments
    def __init__(self, layer: tf.keras.layers.Layer, name: str, quant_scheme: QuantScheme, round_mode: str,
                 bitwidth: int, data_type: QuantizationDataType, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, axis_handling: AxisHandling, num_output_channels: int, enabled: bool):
        keras_common_utils.log_param_quantizer_wrapper_details(layer, axis_handling.value, num_output_channels)
        if enabled:
            op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize
        else:
            op_mode = libpymo.TensorQuantizerOpMode.passThrough

        super().__init__(layer, name, op_mode, quant_scheme, round_mode, bitwidth,data_type, is_symmetric,
                         use_strict_symmetric, use_unsigned_symmetric, axis_handling, num_output_channels)

    def enable(self):
        """
            Enable the paramter tensor quantizer
            If encoding is frozen, no need to do anything (and quant mode should already be set to quantizeDequantize,
            instead of oneShotQuantizeDequantize)
            """
        if not self._is_encoding_frozen:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))

    @StaticGridPerChannelQuantizer.encoding.setter
    def encoding(self, encoding: libpymo.TfEncoding):
        """
            Sets encoding parameter using values obtained from encodings
            :param encoding: encodings value
            """
        if self._is_encoding_frozen:
            _logger.info('Tensor quantizer %s encoding is frozen, not setting new encoding value', self.name)
            return
        self._set_encoding_values(encoding)
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
