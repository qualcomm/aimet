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
# pylint: disable=redefined-builtin
""" Float quantizers """

import contextlib
import functools
from typing import Optional, List, Dict
import math

import torch
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer, _flag_extreme_min_max
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.float import FloatEncoding
from aimet_torch.v2.utils import StatisticsNotFoundError, patch_attr
from aimet_torch.fp_quantization import fake_cast_to_ieee_float


__all__ = ['QuantizeDequantize', 'FloatQuantizeDequantize']


def _ieee_float_max_representable_value(exponent_bits, mantissa_bits):
    exponent_max = 2 ** exponent_bits - 1
    exponent_bias = exponent_max // 2
    return (2 - 2**-mantissa_bits) * 2 ** (exponent_max - exponent_bias - 1)


_IEEE_FLOAT16_EXPONENT_BITS = 5
_IEEE_FLOAT16_MANTISSA_BITS = 10
assert _ieee_float_max_representable_value(_IEEE_FLOAT16_EXPONENT_BITS, _IEEE_FLOAT16_MANTISSA_BITS) == \
        torch.finfo(torch.float16).max

_BFLOAT16_EXPONENT_BITS = 8
_BFLOAT16_MANTISSA_BITS = 7
assert _ieee_float_max_representable_value(_BFLOAT16_EXPONENT_BITS, _BFLOAT16_MANTISSA_BITS) == \
        torch.finfo(torch.bfloat16).max


class FloatQuantizeDequantize(QuantizerBase): # pylint: disable=abstract-method
    r"""
    Simulates quantization by fake-casting the input

    If dtype is provided, this is equivalent to

    .. math::
        out = x.to(dtype).to(x.dtype) \\


    If the exponent and mantissa bits are provided, this is equivalent to

    .. math::
        out = \left\lceil\frac{x_c}{scale}\right\rfloor * scale

    where

    .. math::
        x_c   &= clamp(x, -max, max) \\
        bias  &= 2^{exponent} - \log_2(max) + \log_2(2 - 2^{-mantissa}) - 1 \\
        scale &= 2 ^ {\left\lfloor \log_2 |x_c| + bias \right\rfloor - mantissa - bias} \\


    The IEEE standard computes the maximum representable value by

    .. math::
        max = (2 - 2^{-mantissa}) * 2^{(\left\lfloor 0.5 * exponent\_max \right\rfloor)} \\

    where

    .. math::
        exponent\_max = 2^{exponent} - 1 \\

    Args:
        exponent_bits (int): Number of exponent bits to simulate
        mantissa_bits (int):  Number of mantissa bits to simulate
        dtype (torch.dtype): torch.dtype to simulate. This argument is mutually exclusive with exponent_bits and mantissa_bits.
        encoding_analyzer (EncodingAnalyzer): If specified, the maximum value to represent will be determined dynamically based on the input statistics for finer precision.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.tensor([[ 1.8998, -0.0947],[-1.0891, -0.1727]])
        >>> qdq = Q.float.FloatQuantizeDequantize(mantissa_bits=7, exponent_bits=8)
        >>> # Unlike AffineQuantizer, FloatQuantizer is initialized without calling compute_encodings()
        >>> qdq.is_initialized()
        True
        >>> qdq.is_bfloat16()
        True
        >>> qdq.bitwidth
        16
        >>> qdq(input)
        tensor([[ 1.8984, -0.0947], [-1.0859, -0.1729]])

        >>> from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
        >>> encoding_analyzer = MinMaxEncodingAnalyzer(shape=[])
        >>> qdq = Q.float.FloatQuantizeDequantize(dtype=torch.float16, encoding_analyzer=encoding_analyzer)
        >>> qdq.is_float16()
        True
        >>> qdq.bitwidth
        16
        >>> qdq(input)
        tensor([[ 1.8994, -0.0947], [-1.0889, -0.1727]])
    """

    maxval: torch.Tensor

    def __init__(self,
                 exponent_bits: int = None,
                 mantissa_bits: int = None,
                 dtype: torch.dtype = None,
                 encoding_analyzer: EncodingAnalyzer = None):
        super().__init__()

        if dtype is None:
            if exponent_bits is None or mantissa_bits is None:
                raise ValueError('Neither "dtype" nor "exponent/mantissa_bits" was specified.')

        if dtype is not None:
            if exponent_bits is not None or mantissa_bits is not None:
                raise ValueError(
                    'Argument "dtype" is mutually exclusive with "exponent/mantissa_bits".')

            if dtype not in (torch.half, torch.float16, torch.bfloat16):
                raise ValueError(
                    f"Float quantizer only supports torch.float16 and torch.bfloat16. Got {dtype}.")

            if dtype in (torch.half, torch.float16):
                exponent_bits = _IEEE_FLOAT16_EXPONENT_BITS
                mantissa_bits = _IEEE_FLOAT16_MANTISSA_BITS
            else:
                exponent_bits = _BFLOAT16_EXPONENT_BITS
                mantissa_bits = _BFLOAT16_MANTISSA_BITS

        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.encoding_analyzer = encoding_analyzer

        if self.encoding_analyzer:
            shape = self.encoding_analyzer.observer.shape
            maxval = _ieee_float_max_representable_value(exponent_bits, mantissa_bits)
            self.register_buffer('maxval', torch.full(shape, maxval))
        else:
            self.register_buffer('maxval', None)

    def get_extra_state(self):
        extra_state_dict = {}
        extra_state_dict['exponent_bits'] = torch.tensor(self.exponent_bits)
        extra_state_dict['mantissa_bits'] = torch.tensor(self.mantissa_bits)
        super_extra_state = super().get_extra_state()
        extra_state_dict.update(super_extra_state)
        return extra_state_dict

    def set_extra_state(self, state):
        self.exponent_bits = state['exponent_bits'].item()
        self.mantissa_bits = state['mantissa_bits'].item()
        super().set_extra_state(state)

    def load_state_dict(self, state_dict, strict: bool = True):
        if 'maxval' in state_dict:
            if self.maxval is None:
                del self.maxval
                self.register_buffer('maxval', state_dict['maxval'])
        elif self.maxval is not None:
            del self.maxval
            self.register_buffer('maxval', None)

        ret = super().load_state_dict(state_dict, strict)
        return ret

    @property
    def bitwidth(self):
        """
        Returns bitwidth of the quantizer
        """
        return self.exponent_bits + self.mantissa_bits + 1

    def is_float16(self):
        """
        Returns true if current configuration simulates IEEE float16
        """
        return self.exponent_bits == _IEEE_FLOAT16_EXPONENT_BITS and \
               self.mantissa_bits == _IEEE_FLOAT16_MANTISSA_BITS

    def is_bfloat16(self):
        """
        Returns true if current configuration simulates bfloat16
        """
        return self.exponent_bits == _BFLOAT16_EXPONENT_BITS and \
               self.mantissa_bits == _BFLOAT16_MANTISSA_BITS

    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        :meta private:
        """
        return [{'bitwidth': self.bitwidth, 'dtype': 'float'}]

    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        :meta private:
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'bitwidth': int, 'dtype': str},
            ...
        ]
        """
        if encodings[0]['bitwidth'] != 16:
            raise RuntimeError(f"{self.__class__} can only import 16-bit legay encodings.")
        self.exponent_bits = 5
        self.mantissa_bits = 10

    def get_encodings(self) -> Optional[FloatEncoding]:
        if self.is_initialized():
            return FloatEncoding(self.mantissa_bits, self.exponent_bits, self.maxval)
        return None

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not self.encoding_analyzer or not self._allow_overwrite:
            yield
            return

        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            input = input.as_subclass(torch.Tensor)
            batch_statistics = self.encoding_analyzer.update_stats(input)
            num_steps = math.pow(2, self.bitwidth) - 1
            dynamic_min, dynamic_max =\
                    self.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                        num_steps,
                                                                        is_symmetric=False)
            dynamic_absmax = torch.maximum(dynamic_min.abs(), dynamic_max.abs())
            dynamic_absmax = dynamic_absmax.to(dtype=self.maxval.dtype,
                                               device=self.maxval.device).expand_as(self.maxval)

            with patch_attr(self, 'maxval', dynamic_absmax):
                return original_forward(input)

        self.encoding_analyzer.reset_stats()

        try:
            with patch_attr(self, 'forward', forward_wrapper):
                yield
        except: # pylint: disable=try-except-raise
            raise
        else:
            try:
                num_steps = math.pow(2, self.bitwidth) - 1
                min, max = self.encoding_analyzer.compute_encodings(num_steps,
                                                                    is_symmetric=False)
                _flag_extreme_min_max(min, max)
            except StatisticsNotFoundError:
                return

            if min is None or max is None:
                return

            absmax = torch.maximum(min.abs(), max.abs()).expand_as(self.maxval)
            with torch.no_grad():
                self.maxval.copy_(absmax)

    def forward(self, input: torch.Tensor):
        """
        :param input: Input to quantize and dequantize
        :return: Quantize-dequantized output
        """
        maxval = self.maxval
        exponent_bits = self.exponent_bits
        mantissa_bits = self.mantissa_bits

        if maxval is None:
            if self.is_float16() or self.is_bfloat16():
                # Fast forward using type casting
                orig_dtype = input.dtype
                dtype = torch.float16 if self.is_float16() else torch.bfloat16
                return input.to(dtype).to(orig_dtype)

            maxval = _ieee_float_max_representable_value(exponent_bits, mantissa_bits)

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        return fake_cast_to_ieee_float(input.as_subclass(torch.Tensor), maxval, exponent_bits, mantissa_bits)

    def extra_repr(self):
        """
        :meta private:
        """
        return f'exponent_bits={self.exponent_bits}, mantissa_bits={self.mantissa_bits}'

class QuantizeDequantize(FloatQuantizeDequantize):
    r"""
    Alias of FloatQuantizeDequantize
    """
