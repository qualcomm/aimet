# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Affine quantizers """

import abc
from itertools import chain, repeat
from typing import Optional, List, Dict, Tuple, overload
import contextlib
import functools

import torch
from torch import nn

from aimet_torch.v2.utils import patch_attr, _is_expandable, StatisticsNotFoundError, docstring
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer, MinMaxEncodingAnalyzer, _flag_extreme_min_max
from aimet_torch.v2.quantization.affine import AffineEncoding, GroupedBlockEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine.backends import quantize, quantize_dequantize, torch_builtins, _derive_qmin_qmax
from aimet_torch.v2.utils import ste_round
from aimet_torch.v2.deepspeed_utils import SafeGatheredParameters
from ._utils import _GridMixin, _register_signature # pylint: disable=import-error


__all__ = ['AffineQuantizerBase', 'MinMaxQuantizer', 'Quantize', 'QuantizeDequantize',
           'GroupedBlockQuantizeDequantize']



class AffineQuantizerBase(QuantizerBase, _GridMixin):
    """
    Base class for linear quantization modules.

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)

    """
    _init_signatures = []

    @overload
    @_register_signature(_init_signatures)
    def __init__(self, shape, qmin: int, qmax: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None,
                 block_size: Optional[Tuple[int, ...]] = None):
        ...

    @overload
    @_register_signature(_init_signatures)
    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None,
                 block_size: Optional[Tuple[int, ...]] = None):
        ...

    def __init__(self, shape, *args, **kwargs):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        full_args = (shape, *args)

        # Pad positional args with None's such that len(args) == 5
        args = tuple(chain(args, repeat(None, 5 - len(args))))
        arg0 = kwargs.pop('qmin', kwargs.pop('bitwidth', args[0]))
        arg1 = kwargs.pop('qmax', args[1])

        if arg1 is not None and not isinstance(arg1, bool):
            # (arg0, arg1, arg2) == (qmin, qmax, symmetric)
            qmin, qmax = arg0, arg1
            symmetric = kwargs.pop('symmetric', args[2])

            if (qmin is None) or (qmax is None) or (symmetric is None):
                raise self._arg_parsing_error(full_args, kwargs)

            encoding_analyzer = kwargs.pop('encoding_analyzer', args[3])
            block_size = kwargs.pop('block_size', args[4])
        else:
            # (arg0, arg1) == (bitwidth, symmetric)
            bitwidth = arg0
            symmetric = kwargs.pop('symmetric', args[1])

            if (bitwidth is None) or (symmetric is None):
                raise self._arg_parsing_error(full_args, kwargs)

            # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
            qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=symmetric)
            encoding_analyzer = kwargs.pop('encoding_analyzer', args[2])
            block_size = kwargs.pop('block_size', args[3])

        assert qmin is not None
        assert qmax is not None

        if kwargs:
            cls = type(self).__qualname__
            unexpected_keys = ', '.join(kwargs.keys())
            raise TypeError(f"{cls}.__init__ got unexpected keyword argument: {unexpected_keys}")

        if qmin >= qmax:
            raise ValueError(f"qmax should be strictly larger than qmin. Got qmax={qmax}, qmin={qmin}")

        self.qmin = qmin
        self.qmax = qmax
        self._symmetric = symmetric
        self.block_size = block_size

        self.encoding_analyzer = encoding_analyzer or \
                                 MinMaxEncodingAnalyzer(torch_builtins.get_encoding_shape_with_blocks(self.shape,
                                                                                                      self.block_size))

        if self.block_size is None and not _is_expandable(self.encoding_analyzer.observer.shape, self.shape):
            raise RuntimeError(f'Encoding analyzer of shape {self.encoding_analyzer.observer.shape} '
                               f'is incompatible with quantizer of shape {self.shape}.')

    @abc.abstractmethod
    def get_min(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization min to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed min

        Returns:
            Quantization min

        """

    @abc.abstractmethod
    def get_max(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization max to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed max

        Returns:
            Quantization max

        """

    @abc.abstractmethod
    def get_scale(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed scale

        Returns:
            Quantization scale

        """

    @abc.abstractmethod
    def get_offset(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization offset to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed offset

        Returns:
            Quantization offset

        """

    @abc.abstractmethod
    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """

    def get_encodings(self) -> Optional[AffineEncoding]:
        """
        Return the quantizer's encodings as an AffineEncoding object
        """
        if self.is_initialized():
            return AffineEncoding(self.get_scale(dtype=torch.float32),
                                  self.get_offset(dtype=torch.float32),
                                  self.qmin, self.qmax, self._symmetric, self.block_size)
        return None

    @torch.no_grad()
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """
        # pylint: disable=redefined-builtin, protected-access

        if not self.is_initialized():
            return None

        return self.get_encodings()._to_legacy_format()

    @torch.no_grad()
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            ...
        ]
        """
        def str_to_bool(s: str):
            s = s.lower()
            if s == "false":
                return False
            if s == "true":
                return True
            raise ValueError

        bitwidth = encodings[0]['bitwidth']
        symmetric = str_to_bool(encodings[0]['is_symmetric'])
        # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
        self.qmin, self.qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=symmetric)
        self.symmetric = symmetric
        # Note: We can only accurately infer signed-ness in the symmetric case, but AIMET uses unsigned for asymmetric
        min_ = torch.tensor([e['min'] for e in encodings]).view(self.shape)
        max_ = torch.tensor([e['max'] for e in encodings]).view(self.shape)
        self.set_range(min_, max_)

    def extra_repr(self) -> str:
        extra_repr = f'shape={self.shape}'

        if self.block_size is not None:
            extra_repr += f", block_size={self.block_size}"

        extra_repr += f', qmin={self.qmin}, qmax={self.qmax}, symmetric={self.symmetric}'
        return extra_repr

    @property
    def symmetric(self) -> bool:
        """
        Indicates whether this quantizer uses symmetric quantization
        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self, symmetric: bool):
        """
        Set the quantizer symmetry

        :param symmetric: If True, use symmetric encodings. Else, use asymmetric encodings
        """
        self._symmetric = symmetric

    @property
    @docstring(_GridMixin._get_bitwidth.__doc__)
    def bitwidth(self) -> int: # pylint: disable=missing-function-docstring
        return self._get_bitwidth()

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        self._set_bitwidth(bitwidth)

    @property
    @docstring(_GridMixin._get_signed.__doc__)
    def signed(self) -> bool: # pylint: disable=missing-function-docstring
        return self._get_signed()

    @signed.setter
    def signed(self, signed: bool):
        self._set_signed(signed)


class MinMaxQuantizer(AffineQuantizerBase): # pylint: disable=abstract-method
    """
    Affine quantizer with min-max as trainable parameters
    """

    min: torch.nn.Parameter
    max: torch.nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_quantization_parameter('min', nn.Parameter(-torch.ones(self.shape)))
        self.register_quantization_parameter('max', nn.Parameter(torch.ones(self.shape)))

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not self._allow_overwrite:
            yield
            return

        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            input = input.as_subclass(torch.Tensor)
            expanded_input = torch_builtins.reshape_tensor_for_blocks(input, self.shape, self.block_size)
            batch_statistics = self.encoding_analyzer.update_stats(expanded_input)
            num_steps = self.qmax - self.qmin
            dynamic_min, dynamic_max =\
                    self.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                        num_steps,
                                                                        self.symmetric)
            if self.block_size is not None:
                dynamic_min = dynamic_min.view(self.min.shape)
                dynamic_max = dynamic_max.view(self.max.shape)
            dynamic_min = dynamic_min.to(dtype=self.min.dtype,
                                         device=self.min.device).expand_as(self.min)
            dynamic_max = dynamic_max.to(dtype=self.max.dtype,
                                         device=self.max.device).expand_as(self.max)

            with patch_attr(self, 'min', dynamic_min),\
                    patch_attr(self, 'max', dynamic_max):
                return original_forward(input)

        self.encoding_analyzer.reset_stats()

        try:
            with patch_attr(self, 'forward', forward_wrapper):
                yield
        except: # pylint: disable=try-except-raise
            raise
        else:
            try:
                num_steps = self.qmax - self.qmin
                enc_min, enc_max = self.encoding_analyzer.compute_encodings(num_steps, self.symmetric)
                if self.block_size is not None:
                    enc_min = enc_min.view(self.min.shape)
                    enc_max = enc_max.view(self.max.shape)
                _flag_extreme_min_max(enc_min, enc_max)

            except StatisticsNotFoundError:
                return

            if enc_min is None or enc_max is None:
                return

            self.set_range(enc_min, enc_max)

    def get_min(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :param dtype: dtype of the computed min. Use of self.min.dtype by default.
        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        return self.get_scale(dtype) * (self.get_offset(dtype) + self.qmin)

    def get_max(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :param dtype: dtype of the computed max. Use of self.min.dtype by default.
        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        return self.get_scale(dtype) * (self.get_offset(dtype) + self.qmax)

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: Quantization scale
        """
        if not self.is_initialized():
            return None

        dtype = dtype or torch.float32
        num_steps = self.qmax - self.qmin

        scale = (self.max.to(dtype) - self.min.to(dtype)) / num_steps
        return scale.to(dtype)

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.

        :param dtype: dtype of the computed offset. Use of self.min.dtype by default.
        :return: Quantization offset
        """
        if not self.is_initialized():
            return None

        dtype = dtype or torch.float32

        if self.symmetric:
            offset = torch.full_like(self.min,
                                     fill_value=-round((self.qmin + self.qmax) / 2),
                                     requires_grad=False,
                                     dtype=dtype)
        else:
            offset = ste_round(self.min.to(dtype) / self.get_scale(dtype)) - self.qmin

        return offset.to(dtype)

    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """
        with torch.no_grad(), SafeGatheredParameters(self.parameters(recurse=False), modifier_rank=0):
            self.min.copy_(min)
            self.max.copy_(max)


class Quantize(MinMaxQuantizer):
    r"""Applies quantization to the input.

    Precisely,

    .. math::
        out = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    where :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)
        block_size (Tuple[int, ...], optional): Block size

    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`Quantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> q.is_initialized()
        False
        >>> with q.compute_encodings():
        ...     _ = q(input)
        ...
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[129.,  64., 255., 122.,   0., 192., 106.,  94., 255.,   0.],
                         [  0., 145., 181., 255., 144., 255., 194.,   0.,  74.,  86.],
                         [122.,   0., 255., 150.,  33., 103., 103.,   0.,  37., 255.],
                         [255., 111., 237., 218.,   0.,  49., 155., 255.,   0., 179.],
                         [  0.,  66., 255.,  89., 110.,  17.,  36.,  83., 255.,   0.]],
                        grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> q.is_initialized()
        False
        >>> q.min = torch.nn.Parameter(-torch.ones_like(q.min))
        >>> q.max = torch.nn.Parameter(torch.ones_like(q.max))
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[187., 186., 131.,   0., 203.,  64.,  80.,   0., 143., 152.],
                         [ 16.,   0., 255.,   0.,   0., 150.,   0., 255.,  32., 255.],
                         [255., 226.,   0., 255.,  55., 172.,   0., 255., 145., 255.],
                         [207., 146., 216., 238.,   0.,   0., 141., 178., 255., 188.],
                         [ 63.,  59.,  19., 162.,  30., 255., 109., 255.,   0., 255.]],
                        grad_fn=<AliasBackward0>)
    """
    # NOTE: Deepspeed has a bug where it will inadvertently patch __init__ method permanently
    #       unless each leaf class explicitly defines its own __init__ separately.
    #       As a temporary workaround, we define __init__ to avoid triggering this bug.
    # pylint: disable=useless-super-delegation
    def __init__(self, shape, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        """Quantizes the input tensor

        Args:
            input (torch.Tensor): Input to quantize

        Returns:
            Quantized output

        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run Quantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        encoding = self.get_encodings()

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        input = input.as_subclass(torch.Tensor)

        output = quantize(input,
                          encoding.scale,
                          encoding.offset,
                          encoding.qmin,
                          encoding.qmax,
                          block_size=self.block_size)
        output = output.as_subclass(QuantizedTensor)
        output.encoding = encoding
        return output


class QuantizeDequantize(MinMaxQuantizer):
    r"""Applies fake-quantization by quantizing and dequantizing the input.

    Precisely,

    .. math::
        out = (\overline{input} + offset) * scale

    where

    .. math::
        \overline{input} = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    and :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} &= (\overline{input}_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}\\
        \overline{input}_{j_0 \cdots j_{D-1}} &= clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)
        block_size (Tuple[int, ...], optional): Block size

    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`QuantizeDequantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 2), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> qdq.is_initialized()
        False
        >>> with qdq.compute_encodings():
        ...     _ = qdq(input)
        ...
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[-0.2771,  0.3038,  1.0819,  0.9700,  0.9487, -0.1307,
                            -1.7894, -0.1709, -0.2212,  0.7741],
                           [-1.0295, -1.2265, -1.0295,  1.0564,  0.6177, -1.0386,
                            -0.0176, -2.6054,  1.8836, -0.1232],
                           [-0.8229,  0.5540,  0.3992, -0.2363,  1.2546, -1.0036,
                             0.2355,  0.1741,  1.6079,  0.6247],
                           [-1.0115,  1.2458,  0.9157, -1.4694, -0.0639, -0.2568,
                             0.0680,  1.6695,  0.7932, -0.1889],
                           [ 0.0158,  0.5695,  0.5220,  0.1977, -1.4475, -0.0424,
                            -1.1128, -0.8796, -0.1060,  1.5897]],
                          grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 2), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> qdq.is_initialized()
        False
        >>> qdq.min = torch.nn.Parameter(-torch.ones_like(qdq.min))
        >>> qdq.max = torch.nn.Parameter(torch.ones_like(qdq.max))
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[-0.6196, -0.9961,  0.0549, -0.6431,  1.0039, -0.8706,
                             1.0039,  0.4706, -0.2353,  0.8078],
                           [ 0.3451, -0.1176, -0.9961, -0.4549, -0.0549, -0.0471,
                            -0.5255, -0.2353,  1.0039, -0.9961],
                           [-0.4157,  0.0784,  0.5333,  0.1647, -0.9961, -0.9961,
                            -0.2118, -0.2196,  0.9176,  0.9490],
                           [ 1.0039, -0.7765,  0.4784, -0.8706,  1.0039,  0.6039,
                            -0.4157, -0.2118, -0.9961,  0.3137],
                           [ 1.0039,  0.3216, -0.2353, -0.7765, -0.9961,  0.8000,
                             1.0039,  0.4157,  0.4392,  0.4863]],
                          grad_fn=<AliasBackward0>)
    """
    # NOTE: Deepspeed has a bug where it will inadvertently patch __init__ method permanently
    #       unless each leaf class explicitly defines its own __init__ separately.
    #       As a temporary workaround, we define __init__ to avoid triggering this bug.
    # pylint: disable=useless-super-delegation
    def __init__(self, shape, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """Quantizes and dequantizes the input tensor

        Args:
            input (torch.Tensor): Input to quantize and dequantize

        Returns:
            Quantize-dequantized output

        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run QuantizeDequantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        encoding = self.get_encodings()

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        input = input.as_subclass(torch.Tensor)

        output = quantize_dequantize(input,
                                     encoding.scale,
                                     encoding.offset,
                                     encoding.qmin,
                                     encoding.qmax,
                                     block_size=self.block_size)
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output


class GroupedBlockQuantizeDequantize(QuantizeDequantize): # pylint: disable=too-many-ancestors
    """ Class for performing Grouped Block Quantize Dequantize """
    def __init__(self, shape, bitwidth: int, symmetric: bool, decompressed_bw: int,
                 encoding_analyzer: EncodingAnalyzer = None, block_size: Optional[Tuple[int, ...]] = None,
                 block_grouping: Optional[Tuple[int, ...]] = None):
        """
        Grouped Block Quantize Dequantize constructor.

        :param shape: Shape of the quantization parameters
        :type shape: tuple
        :param bitwidth: Quantization bitwidth
        :type bitwidth: int
        :param symmetric: If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        :type symmetric: bool
        :param decompressed_bw: Bitwidth used for decompression
        :type decompressed_bw: int
        :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings
                                  (default: absolute min-max encoding analyzer)
        :type encoding_analyzer: EncodingAnalyzer, optional
        :param block_size: Block size per dimension.
        :type block_size: Tuple
        :param block_grouping: Block grouping per dimension. If provided, every set of block_group scales will be
                               grouped together, and the maximum scale for all blocks in the group will be used to find
                               the scale in the decompressed_grid to be shared by all blocks in the group.
                               If no block_grouping is provided, default behavior uses a block group of 1 for all dims,
                               equivalent to Blockwise Quantization.
                               A value of -1 for a block group for a dimension is equivalent to grouping all blocks in
                               the dimension in one group. This is also equivalent to a block group value equal to the
                               number of blocks for that dimension.
        :type block_grouping: Tuple
        """
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer, block_size)
        self.decompressed_bw = decompressed_bw
        self.block_grouping = block_grouping
        if self.block_grouping is None:
            # Default to BQ behavior with 1 for all block grouping dims if not provided
            self.block_grouping = tuple(1 for _ in enumerate(self.shape))

        if block_grouping is not None:
            if len(block_grouping) != len(shape):
                raise RuntimeError(f'Length of block grouping {block_grouping} must equal length of shape {shape}.')
            for idx, block_group in enumerate(block_grouping):
                if block_group != -1 and shape[idx] % block_group != 0:
                    raise RuntimeError(f'Quantizer shape dimensions must divide evenly with corresponding block '
                                       f'grouping values for shapes {shape} and block grouping {block_grouping}.')

        if self.decompressed_bw < self.bitwidth:
            raise RuntimeError(f'Decompressed bitwidth {decompressed_bw} cannot be smaller than self.bitwidth '
                               f'{bitwidth}')

        if not symmetric:
            raise RuntimeError('GroupedBlockQuantizeDequantize only supports symmetric quantization.')

    def get_scale(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Overrides QuantizeDequantize self.get_scale() to apply the grouped block algorithm for calculating modified
        scales.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: Updated scale
        """
        orig_scale = super().get_scale(dtype)
        orig_scale_shape = orig_scale.shape
        reshaped_scale = orig_scale.view(self.get_expanded_scale_shape())
        max_scale = torch.amax(reshaped_scale, list(range(1, len(orig_scale_shape) * 2, 2)), keepdim=True)
        per_channel_scale = max_scale / 2 ** (self.decompressed_bw - self.bitwidth)
        updated_scale = quantize_dequantize(reshaped_scale,
                                            scale=per_channel_scale,
                                            offset=torch.zeros_like(per_channel_scale),
                                            qmin=1,
                                            qmax=2 ** (self.decompressed_bw - self.bitwidth))
        return updated_scale.view(orig_scale_shape)

    def get_expanded_scale_shape(self) -> Tuple[int, ...]:
        """
        Get expanded scale shape which breaks each scale dimension into a pair of dimensions with sizes
        (original_shape / block_grouping, block_grouping).

        :return: Expanded scale shape
        """
        expanded_shape = []
        for idx, block_group in enumerate(self.block_grouping):
            # Block group of -1 is equivalent to grouping all blocks together
            if block_group == -1:
                expanded_shape.append(1)
                expanded_shape.append(self.shape[idx])
            else:
                expanded_shape.append(self.shape[idx] // block_group)
                expanded_shape.append(block_group)
        return expanded_shape

    def get_per_channel_scale(self, dtype=None) -> torch.Tensor:
        """
        Get per channel scale.

        :return: Per channel scale
        """
        orig_scale = super().get_scale(dtype)
        orig_scale_shape = orig_scale.shape
        reshaped_scale = orig_scale.view(self.get_expanded_scale_shape())
        max_scale = torch.amax(reshaped_scale, list(range(1, len(orig_scale_shape) * 2, 2)), keepdim=True)
        per_channel_scale = max_scale / 2 ** (self.decompressed_bw - self.bitwidth)
        return per_channel_scale

    def get_per_block_integer_scale(self) -> torch.Tensor:
        """
        Get per block integer scale.

        :return: Per block integer scale
        """
        per_channel_scale = self.get_per_channel_scale()
        expanded_scale = self.get_scale().view(self.get_expanded_scale_shape())
        integer_scale = torch.round(expanded_scale / per_channel_scale).int().view(self.get_scale().shape)
        return integer_scale

    def get_encodings(self) -> Optional[GroupedBlockEncoding]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """
        if self.is_initialized():
            return GroupedBlockEncoding(scale=self.get_scale(dtype=torch.float32),
                                        offset=self.get_offset(dtype=torch.float32),
                                        bitwidth=self.bitwidth,
                                        signed=self.signed,
                                        symmetry=self.symmetric,
                                        block_size=self.block_size,
                                        block_grouping=self.block_grouping,
                                        decompressed_bw=self.decompressed_bw,
                                        per_channel_scale=self.get_per_channel_scale(dtype=torch.float32),
                                        per_block_int_scale=self.get_per_block_integer_scale())
        return None
