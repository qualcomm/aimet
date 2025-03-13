# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Common utility for Quantization """

from typing import Union, Tuple, Dict
import numpy as np

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_common import libpymo

# Defined below is a quantization encoding format version, which will follow XX.YY.ZZ versioning as described below,
#
#    XX = Major Revision
#    YY = Minor Revision
#    ZZ = Patching version
#
# Change in major revision should indicate substantial change to the format, updates to minor version indicates
# additional information element being added to encoding format and might require update to fully consume the encodings.
# The patching version shall be updated to indicate minor updates to quantization simulation e.g. bug fix etc.
encoding_version = '1.0.0'
ALLOW_EXPERIMENTAL = False
VALID_ENCODING_VERSIONS = {'0.6.1', '1.0.0', '2.0.0.beta'}


def gate_min_max(min_val: float, max_val: float) -> Tuple[float, float]:
    """
    Gates min and max encoding values to retain zero in the range representation.
    Rules : min at maximum can be zero, max at minimum can be zero and
    if max and min are equal, adds epsilon to maintain range.
    :param min_val: min encoding value
    :param max_val: max encoding value
    :return: gated min and max values
    """

    epsilon = 1e-5
    # For per channel quantization
    if isinstance(min_val, np.ndarray):
        gated_min = np.clip(min_val, None, 0.0)
        gated_max = np.clip(max_val, 0.0, None)
        gated_max = np.clip(gated_max, gated_min + epsilon, None)
    else:
        gated_min = min(min_val, 0.0)
        gated_max = max(max_val, 0.0)
        gated_max = max(gated_max, gated_min + epsilon)

    return gated_min, gated_max


def is_non_strict_symmetric(use_symmetric_encodings: bool,
                            use_strict_symmetric: bool,
                            is_unsigned_symmetric: bool) -> bool:
    """
    Check whether non-strict symmetric encoding or not
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :param is_unsigned_symmetric: is_unsigned_symmetric flag
    :return: True if it satisfies non-strict symmetric else False
    """
    return use_symmetric_encodings and \
           not use_strict_symmetric and \
           not is_unsigned_symmetric


def create_encoding_from_min_max(min_val: float, max_val: float, bitwidth: int, use_symmetric_encodings: bool,
                                 use_strict_symmetric: bool) -> libpymo.TfEncoding:
    """
    Returns a TfEncoding object with the provided min/max/bitwidth/symmetry

    :param min_val: Min value of the encoding
    :param max_val: Max value of the encoding
    :param bitwidth: Encoding bitwidth
    :param use_symmetric_encodings: If True, results in encoding with min = -max - delta
    :param use_strict_symmetric: If True, results in encoding with min = -max
    :return: libpymo.TfEncoding object
    """
    delta, offset = calculate_delta_offset(min_val, max_val, bitwidth, use_symmetric_encodings, use_strict_symmetric)

    encoding = libpymo.TfEncoding()
    encoding.bw = bitwidth
    encoding.min = min_val
    encoding.max = max_val
    encoding.delta = delta
    encoding.offset = offset
    # Note: need to recompute grid to account for offset rounding
    return recompute_grid_params(encoding, bitwidth, use_symmetric_encodings, use_strict_symmetric)


def calculate_delta_offset(min_val: float, max_val: float, bitwidth: int, use_symmetric_encodings: bool,
                           use_strict_symmetric: bool) -> Tuple[float, int]:
    """
    Calculates delta and offset given min and max.

    :param min_val: min encoding value
    :param max_val: max encoding value
    :param bitwidth: bitwidth used for quantization
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :return: delta and offset values computed
    """
    num_steps = 2 ** bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    min_val, max_val = gate_min_max(min_val, max_val)

    # Use only max val to compute delta in the case of signed symmetric
    if use_symmetric_encodings and min_val < 0:
        num_positive_steps = np.floor(num_steps / 2)
        delta = max_val / num_positive_steps
        offset = -num_positive_steps
        if not use_strict_symmetric:
            offset -= 1
    else:
        delta = (max_val - min_val) / num_steps
        offset = round(min_val / delta)

    return delta, offset

def compute_min_max_given_delta_offset(delta: float, offset: int, bitwidth: int, use_symmetric_encodings: bool,
                                       use_strict_symmetric: bool) -> Tuple[float, float]:
    """
    Compute min and max given delta and offset.

    :param delta: Delta to compute with
    :param offset: Offset to compute with
    :param bitwidth: Bitwidth for finding number of steps
    :param use_symmetric_encodings: True if symmetric, False otherwise
    :param use_strict_symmetric: True if using strict symmetric, False otherwise
    :return: Tuple of computed min and max values
    """
    num_steps = 2 ** bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    min_val = delta * offset
    max_val = (num_steps + offset) * delta
    return min_val, max_val

def recompute_grid_params(current_encoding: libpymo.TfEncoding, bitwidth: int,
                          use_symmetric_encoding: bool, use_strict_symmetric: bool = False) -> libpymo.TfEncoding:
    """
    Recomputes the encoding grid params - min/max/offset and delta.

    :param current_encoding: Encoding associated with the quantizer as TfEncoding
    :param bitwidth: bit width configured for the quantizer
    :param use_symmetric_encoding: symmetric or asymmetric mode
    :param use_strict_symmetric: True if using strict symmetric, False otherwise
    :return: updated encoding params as libpymo.TfEncoding type.
    """

    MIN_RANGE = 0.01
    min_val = min(0., current_encoding.min)
    max_val = max(0., current_encoding.max, (min_val + MIN_RANGE))
    updated_encoding = libpymo.TfEncoding()

    # check mode used to recompute delta and offset
    if use_symmetric_encoding:
        num_positive_steps = (2 ** (bitwidth - 1)) - 1
        abs_max_val = max(abs(max_val), abs(min_val))
        delta = abs_max_val / num_positive_steps
        offset = -(num_positive_steps + int(not use_strict_symmetric))
        # recompute min/max values
        min_val = delta * offset
        max_val = delta * num_positive_steps

    else:
        num_steps = (2 ** bitwidth) - 1
        delta = (max_val - min_val) / num_steps
        # @todo check zero point representation related code
        offset = round(min_val / delta)
        # recompute min/max values
        min_val = delta * offset
        max_val = min_val + delta * num_steps

    updated_encoding.bw = bitwidth
    updated_encoding.min = min_val
    updated_encoding.max = max_val
    updated_encoding.delta = delta
    updated_encoding.offset = offset

    return updated_encoding


def validate_quantsim_inputs(
        quant_scheme: Union[str, QuantScheme],
        rounding_mode: str,
        default_output_bw: int,
        default_param_bw: int,
        data_type: QuantizationDataType = QuantizationDataType.int
):
    """
    Perform sanity checks on inputs to QuantSim
    :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or 'percentile'
                         or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
                         or QuantScheme.post_training_percentile
    :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
    :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs
    :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
    :param data_type: Data type of the quantized values (int or float).
    """
    _validate_quant_scheme(quant_scheme)
    _validate_rounding_mode(rounding_mode)
    _validate_bitwidth(default_output_bw, default_param_bw, data_type)


def _validate_quant_scheme(quant_scheme: Union[str, QuantScheme]):
    if quant_scheme not in ('tf_enhanced', 'tf', 'percentile') and not isinstance(quant_scheme, QuantScheme):
        raise ValueError('Parameter quantization mode is not a valid selection. Valid selections are '
                         'tf, tf_enhanced, percentile, QuantScheme.post_training_tf, '
                         'QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_percentile')

def _validate_rounding_mode(rounding_mode: str):
    if rounding_mode not in ('nearest', 'stochastic'):
        raise ValueError('Parameter round mode is not a valid selection. Valid selections are nearest or '
                         'stochastic')

def _validate_bitwidth(default_output_bw: int,
                       default_param_bw: int,
                       data_type: QuantizationDataType = QuantizationDataType.int):
    if default_param_bw < 4 or default_param_bw > 32:
        raise ValueError('Default bitwidth for parameters must be between 4 and 32, not ' + str(default_param_bw))

    if default_output_bw < 4 or default_output_bw > 32:
        raise ValueError('Activation bitwidth must be between 4 and 32, not ' + str(default_output_bw))

    if ALLOW_EXPERIMENTAL:
        if data_type == QuantizationDataType.float and default_output_bw not in [8, 16, 32]:
            raise ValueError(
                'float data_type can only be used when default_output_bw set to 8, 16 or 32, not ' + str(default_output_bw))

        if data_type == QuantizationDataType.float and default_param_bw not in [8, 16, 32]:
            raise ValueError(
                'float data_type can only be used when default_param_bw set to 8, 16 or 32, not ' + str(default_param_bw))

    else:
        if data_type == QuantizationDataType.float and default_output_bw not in [16, 32]:
            raise ValueError(
                'float data_type can only be used when default_output_bw set to 16 or 32, not ' + str(default_output_bw))

        if data_type == QuantizationDataType.float and default_param_bw not in [16, 32]:
            raise ValueError(
                'float data_type can only be used when default_param_bw set to 16 or 32, not ' + str(default_param_bw))


def extract_global_quantizer_args(quant_scheme: Union[str, QuantScheme],
                                  quantsim_configurator: QuantSimConfigurator) -> Dict:
    """
    Extracts quantizer arguments used to configure QuantSim
    :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or 'percentile'
                         or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
                         or QuantScheme.post_training_percentile
    :param quantsim_configurator: An instance of QuantSimConfigurator which has been populated either by config file
                                  or via function arguments.
    :return: A dictionary of quantizer arguments
    """
    quant_args = {}
    default_dict = quantsim_configurator.quantsim_configs["defaults"]
    param_dict = default_dict["params"]
    is_per_channel_quant = default_dict[
        "per_channel_quantization"] if "per_channel_quantization" in default_dict else False

    if (isinstance(quant_scheme, str) and quant_scheme == QuantScheme.training_range_learning_with_tf_init or
            quant_scheme == QuantScheme.training_range_learning_with_tf_init):
        quant_scheme = QuantScheme.post_training_tf
    if (isinstance(quant_scheme, str) and quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init) or \
            quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init:
        quant_scheme = QuantScheme.post_training_tf_enhanced

    quant_args.update({"quant_scheme": quant_scheme.name if isinstance(quant_scheme, QuantScheme) else quant_scheme,
                       "param_bitwidth": quantsim_configurator.default_param_bw,
                       "activation_bitwidth": quantsim_configurator.default_output_bw,
                       "dtype": quantsim_configurator.default_data_type.name,
                       "is_symmetric": param_dict[
                           "is_symmetric"] if "is_symmetric" in param_dict else is_per_channel_quant,
                       "per_channel_quantization": is_per_channel_quant
                       })
    return quant_args


def _get_minimum_scale(num_steps: int) -> float:
    """
    Return the minimum scale given the number of steps in the quantization grid.

    We define the minimum scale as the largest s <= float32.eps such that
    -0.005 <= s * min(x_int) <  s * max(x_int) <= 0.005

    Following this rule, the minimum scale in practice will be:

      | dtype | minimum scale |
      |-------|---------------|
      |  int4 |    1.19e-07   | (note: float32.eps = 1.19e-07)
      |  int8 |    1.19e-07   |
      | int16 |    1.19e-07   |
      | int32 |    2.33e-12   | (note: float64.eps = 2.22e-16)

    """
    fp32_eps = np.finfo(np.float32).eps

    _MINIMUM_RANGE_TO_REPRESENT = (-0.005, 0.005)
    _min, _max = _MINIMUM_RANGE_TO_REPRESENT
    return min(fp32_eps, (_max - _min) / num_steps)


_INT4_MINIMUM_SCALE = _get_minimum_scale(2**4-1)
_INT8_MINIMUM_SCALE = _get_minimum_scale(2**8-1)
_INT16_MINIMUM_SCALE = _get_minimum_scale(2**16-1)
_INT32_MINIMUM_SCALE = _get_minimum_scale(2**32-1)
