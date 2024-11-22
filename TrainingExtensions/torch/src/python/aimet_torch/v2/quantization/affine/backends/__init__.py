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
# pylint: disable=all

import math
from itertools import chain, repeat
from typing import overload, Union, Tuple, Optional
import torch
from .utils import *


@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             bitwidth: Union[int, float], signed: bool = False,
             block_size: Optional[Tuple[int, ...]] = None):
    ...

@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
             num_steps: int, signed: bool = False, block_size: Optional[Tuple[int, ...]] = None):
    ...

@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             qmin: int, qmax: int, block_size: Optional[Tuple[int, ...]] = None):
    ...


def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             *args, **kwargs):
    r"""
    Applies quantization to the input.

    Precisely,

    .. math::
        out = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    This function is overloaded with the signatures listed below:


    .. function:: quantize(tensor, scale, offset, bitwidth, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin= 
           \begin{cases}
               -\left\lceil\frac{2^{bitwidth}-1}{2}\right\rceil,& \text{if } signed\\
               0,                                               & \text{otherwise   (default)}
           \end{cases}
           qmax= 
           \begin{cases}
               \left\lfloor\frac{2^{bitwidth}-1}{2}\right\rfloor,& \text{if } signed\\
               2^{bitwidth}-1,                                   & \text{otherwise   (default)}
           \end{cases}

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int bitwidth: Bitwidth of quantized tensor based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, the output will be mapped to positive integers only.
           Otherwise, it will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize(tensor, scale, offset, *, num_steps, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin= 
           \begin{cases}
               -\left\lceil\frac{num\_steps}{2}\right\rceil,& \text{if } signed\\
               0,                                           & \text{otherwise   (default)}
           \end{cases}
           qmax= 
           \begin{cases}
               \left\lfloor\frac{num\_steps}{2}\right\rfloor,& \text{if } signed\\
               num\_steps,                                   & \text{otherwise   (default)}
           \end{cases}


       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int num_steps: The number of steps in the quantization range based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, the output will be mapped to positive integers only.
           Otherwise, it will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize(tensor, scale, offset, *, qmin, qmax, block_size=None)
       :noindex:

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int qmin: Minimum value of the quantization range
       :param int qmax: Maximum value of the quantization range
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional


    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.arange(start=-0.3, end=1.3, step=0.05)
        >>> print(input)
        tensor([-3.0000e-01, -2.5000e-01, -2.0000e-01, -1.5000e-01, -1.0000e-01,
                -5.0000e-02, -1.1921e-08,  5.0000e-02,  1.0000e-01,  1.5000e-01,
                2.0000e-01,  2.5000e-01,  3.0000e-01,  3.5000e-01,  4.0000e-01,
                4.5000e-01,  5.0000e-01,  5.5000e-01,  6.0000e-01,  6.5000e-01,
                7.0000e-01,  7.5000e-01,  8.0000e-01,  8.5000e-01,  9.0000e-01,
                9.5000e-01,  1.0000e+00,  1.0500e+00,  1.1000e+00,  1.1500e+00,
                1.2000e+00,  1.2500e+00])
        >>> scale = torch.tensor(1/15)
        >>> offset = torch.tensor(0.0)
        >>> Q.affine.quantize(input, scale, offset, bitwidth=4)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
        >>> Q.affine.quantize(input, scale, offset, num_steps=15)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
        >>> Q.affine.quantize(input, scale, offset, qmin=0, qmax=15)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
    """
    qmin, qmax, block_size = _parse_args(args, kwargs)
    return get_backend().quantize(tensor, scale, offset, qmin, qmax, block_size)


@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        bitwidth: Union[int, float], signed: bool = False,
                        block_size: Optional[Tuple[int, ...]] = None):
    ...

@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
                        num_steps: int, signed: bool = False, block_size: Optional[Tuple[int, ...]] = None):
    ...

@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        qmin: int, qmax: int, block_size: Optional[Tuple[int, ...]] = None):
    ...


def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        *args, **kwargs):
    r"""
    Applies fake-quantization by quantizing and dequantizing the input.

    Precisely,

    .. math::
        out = (\overline{input} + offset) * scale

    where

    .. math::
        \overline{input} = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)


    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} &= (\overline{input}_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}\\
        \overline{input}_{j_0 \cdots j_{D-1}} &= clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where } \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor


    This function is overloaded with the signatures listed below:


    .. function:: quantize_dequantize(tensor, scale, offset, bitwidth, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin= 
           \begin{cases}
               -\left\lceil\frac{2^{bitwidth}-1}{2}\right\rceil,& \text{if } signed\\
               0,                                               & \text{otherwise   (default)}
           \end{cases}
           qmax= 
           \begin{cases}
               \left\lfloor\frac{2^{bitwidth}-1}{2}\right\rfloor,& \text{if } signed\\
               2^{bitwidth}-1,                                   & \text{otherwise   (default)}
           \end{cases}

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int bitwidth: Bitwidth of quantized tensor based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, :math:`\overline{input}` will be mapped to positive integers only.
           Otherwise, :math:`\overline{input}` will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize_dequantize(tensor, scale, offset, *, num_steps, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin= 
           \begin{cases}
               -\left\lceil\frac{num\_steps}{2}\right\rceil,& \text{if } signed\\
               0,                                           & \text{otherwise   (default)}
           \end{cases}
           qmax= 
           \begin{cases}
               \left\lfloor\frac{num\_steps}{2}\right\rfloor,& \text{if } signed\\
               num\_steps,                                   & \text{otherwise   (default)}
           \end{cases}


       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int num_steps: The number of steps in the quantization range based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, :math:`\overline{input}` will be mapped to positive integers only.
           Otherwise, :math:`\overline{input}` will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize_dequantize(tensor, scale, offset, *, qmin, qmax, block_size=None)
       :noindex:

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int qmin: Minimum value of the quantization range
       :param int qmax: Maximum value of the quantization range
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional


    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.arange(start=-0.3, end=1.3, step=0.05)
        >>> print(input)
        tensor([-3.0000e-01, -2.5000e-01, -2.0000e-01, -1.5000e-01, -1.0000e-01,
                -5.0000e-02, -1.1921e-08,  5.0000e-02,  1.0000e-01,  1.5000e-01,
                2.0000e-01,  2.5000e-01,  3.0000e-01,  3.5000e-01,  4.0000e-01,
                4.5000e-01,  5.0000e-01,  5.5000e-01,  6.0000e-01,  6.5000e-01,
                7.0000e-01,  7.5000e-01,  8.0000e-01,  8.5000e-01,  9.0000e-01,
                9.5000e-01,  1.0000e+00,  1.0500e+00,  1.1000e+00,  1.1500e+00,
                1.2000e+00,  1.2500e+00])
        >>> scale = torch.tensor(1/15)
        >>> offset = torch.tensor(0.0)
        >>> Q.affine.quantize_dequantize(input, scale, offset, bitwidth=4)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
        >>> Q.affine.quantize_dequantize(input, scale, offset, num_steps=15)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
        >>> Q.affine.quantize_dequantize(input, scale, offset, qmin=0, qmax=15)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    """
    qmin, qmax, block_size = _parse_args(args, kwargs)
    return get_backend().quantize_dequantize(tensor, scale, offset, qmin, qmax, block_size)


def dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
               block_size: Optional[Tuple[int, ...]] = None):
    r"""
    Applies dequantization to the input.

    Precisely,

    .. math::
        out = (input + offset) * scale

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = (input_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    :param Tensor tensor: Tensor to dequantize
    :param Tensor scale: Scale for dequantization
    :param Tensor offset: Offset for dequantization
    :param block_size: Block size
    :type block_size: Tuple[int, ...], optional
    """
    return get_backend().dequantize(tensor, scale, offset, block_size)


def _parse_args(args, kwargs) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
    bitwidth = num_steps = signed = qmin = qmax = None

    # Pad positional args with None's such that len(args) == 3
    args = tuple(chain(args, repeat(None, 3 - len(args))))
    arg0 = kwargs.get('qmin', kwargs.get('bitwidth', args[0]))
    arg1 = kwargs.get('qmax', kwargs.get('signed', args[1]))
    block_size = kwargs.get('block_size', None) or args[2]

    if arg0 is None:
        num_steps = kwargs['num_steps']
        signed = kwargs['signed']
        qmin, qmax = _derive_qmin_qmax(num_steps=num_steps, signed=signed)
    elif arg1 is None or isinstance(arg1, bool):
        bitwidth, signed = arg0, bool(arg1)
        qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=signed)
    else:
        qmin, qmax = arg0, arg1

    assert qmin is not None
    assert qmax is not None

    return qmin, qmax, block_size


def _derive_qmin_qmax(*, bitwidth: int = None, num_steps: int = None, signed: bool):
    if bitwidth is not None:
        num_steps = 2 ** bitwidth - 1

    if signed:
        qmin = -math.ceil(num_steps/2)
        qmax = math.floor(num_steps/2)
    else:
        qmin = 0
        qmax = num_steps

    return qmin, qmax
