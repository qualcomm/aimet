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
""" Implements straight through gradient computation for Quant op"""
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

from aimet_torch.v1.tensor_factory_utils import constant_like

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from aimet_torch.v1.tensor_quantizer import LearnedGridTensorQuantizer


@dataclass
class IntermediateResult:
    """
    Data carrier containing intermediate result for learned grid backward computation
    """
    x_quant: torch.Tensor
    encoding_min: torch.nn.Parameter
    encoding_max: torch.nn.Parameter
    delta: torch.Tensor
    offset: torch.Tensor
    mask_tensor: torch.Tensor
    num_steps: torch.Tensor
    is_symmetric: bool
    is_unsigned: bool


def broadcast_to_tensor(tensor, encoding, ch_axis):
    """
    This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
    match the n-dimensional tensor
    :param tensor: Tensor to use as target for the broadcasting operation
    :param encoding: Encoding 1-dimensional tensor to broadcast
    :param ch_axis: Channel axis along which broadcasting happens
    :return: Broad-casted tensor
    """
    if not isinstance(encoding, torch.Tensor):
        encoding = torch.tensor(encoding).to(tensor.device)  # convert encoding to a tensor

    assert len(encoding.shape) <= 1 # Should be 1-dimensional tensor

    if encoding.numel() == 1:
        return encoding

    # Shape of encoding should match the channel dimension of the input
    assert encoding.numel() == tensor.shape[ch_axis]

    shape = tuple(dim if axis == ch_axis else 1
                  for axis, dim in enumerate(tensor.shape))
    return encoding.view(shape)


def compute_dloss_by_dx(x, grad, encoding_min, encoding_max, ch_axis=0):
    """
    compute derivative w.r.t input using straight through estimator.
    :param x: input tensor
    :param grad: gradient flowing
    :param encoding_min: encoding min grid param used on forward pass
    :param encoding_max: encoding max grid param used on forward pass
    :param ch_axis: Channel axis to use for per-channel quant
    :return: gradient w.r.t input
    """
    encoding_max = broadcast_to_tensor(x, encoding_max, ch_axis)
    encoding_min = broadcast_to_tensor(x, encoding_min, ch_axis)

    # dL / dx = (dL / dx_q) * (dx_q / dx)
    #         =   `grad`    * (dx_q / dx)

    # x_q = quantize_dequantize(x)
    #     = +-- round(x / delta) * delta  if  encoding_min <= x <= encoding_max
    #       |-- encoding_min              if  x < encoding_min
    #       +-- encoding_max              if  x > encoding_max

    # dx_q / dx  =  +-- 1  if  encoding_min <= x <= encoding_max
    #  (`mask`)     |-- 0  if  x < encoding_min
    #               +-- 0  if  x > encoding_max
    mask = (encoding_min <= x).logical_and(x <= encoding_max)

    # Therefore, dL / dx = `grad` * `mask`
    return grad * mask


def get_computed_encodings(bitwidth: int,
                           encoding_min: torch.nn.Parameter,
                           encoding_max: torch.nn.Parameter,
                           use_symmetric_encodings: bool,
                           use_strict_symmetric: bool,
                           is_unsigned_symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute delta and offset, and number of steps given quantization parameters
    Followed the flow of C++ compute encoding function (quantization_utils::getComputedEncodings)
    :param bitwidth: Bitwidth
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :param use_symmetric_encodings: True if symmetric encoding is used. False otherwise
    :param use_strict_symmetric: True if strict symmetric encoding is used. False otherwise
    :param is_unsigned_symmetric: Whether to use signed/unsigned in symmetric case
    :return: Tuple of delta and offset and num_steps
    """
    num_steps = 2 ** bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1
    half_num_steps = num_steps / 2

    num_steps_tensor = constant_like(num_steps, encoding_min)
    if use_symmetric_encodings and not is_unsigned_symmetric:
        # signed symmetric
        delta = encoding_max / constant_like(math.floor(half_num_steps), encoding_min)
        offset = -constant_like(math.ceil(half_num_steps), encoding_min)
    else:
        delta = (encoding_max - encoding_min) / num_steps_tensor
        if use_symmetric_encodings:
            # unsigned symmetric
            offset = encoding_min / delta
        else:
            # asymmetric
            zero_tensor = constant_like(0., encoding_min)
            b_zero = torch.round(-encoding_min / delta)
            b_zero = torch.min(num_steps_tensor, torch.max(zero_tensor, b_zero))
            offset = -b_zero

    return delta, offset, num_steps_tensor


def _compute_variables_for_range_learning(tensor: torch.Tensor,
                                          bitwidth: int,
                                          encoding_min: torch.nn.Parameter,
                                          encoding_max: torch.nn.Parameter,
                                          channel_axis: int,
                                          use_symmetric_encodings: bool,
                                          use_strict_symmetric: bool,
                                          is_unsigned_symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate required variables for range learning
    :param tensor: torch Tensor
    :param bitwidth: Bitwidth for quantization
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :param channel_axis: Channel axis to use for per-channel quant
    :param use_symmetric_encodings: True if symmetric encoding is used. False otherwise
    :param use_strict_symmetric: True if strict symmetric encoding is used. False otherwise
    :param is_unsigned_symmetric: Whether to use signed/unsigned in symmetric case
    """
    delta, offset, num_steps = get_computed_encodings(bitwidth, encoding_min, encoding_max,
                                                      use_symmetric_encodings, use_strict_symmetric, is_unsigned_symmetric)
    delta = broadcast_to_tensor(tensor, delta, channel_axis)
    offset = broadcast_to_tensor(tensor, offset, channel_axis)

    return delta, offset, num_steps


# pylint:disable=too-many-locals
def calculate_forward_pass(tensor: torch.Tensor,
                           tensor_quantizer: "LearnedGridTensorQuantizer",
                           encoding_min: torch.nn.Parameter,
                           encoding_max: torch.nn.Parameter) -> Tuple[torch.Tensor, IntermediateResult]:
    """
    Calculate forward pass logic of range learning
    :param tensor: Target tensor to compute
    :param tensor_quantizer: LearnedGridTensorQuantizer corresponding to target tensor
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :return: QuantizeDequantize out and intermediate result tuple
    """
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError("Invalid input data type. Expected torch.float32 or torch.float16. "
                           f"Got {tensor.dtype}.")

    if not (tensor.dtype == encoding_min.dtype == encoding_max.dtype): # pylint: disable=superfluous-parens
        raise RuntimeError("Data type mismatch. Expected the input and encoding min & max to be of same dtype."
                           f"Got {tensor.dtype} input, {encoding_min.dtype} encoding_min, "
                           f"and {encoding_max.dtype} encoding_max")

    if tensor_quantizer.bitwidth >= 32:
        raise RuntimeError(f'Invalid bitwidth: {tensor_quantizer.bitwidth}')

    orig_dtype = tensor.dtype
    if tensor.dtype in (torch.float16, torch.bfloat16) and tensor_quantizer.bitwidth >= 16:
        tensor = tensor.float()
        encoding_min = encoding_min.float()
        encoding_max = encoding_max.float()

    use_symmetric_encodings = tensor_quantizer.use_symmetric_encodings
    is_unsigned_symmetric = tensor_quantizer.is_unsigned_symmetric
    delta, offset, num_steps = _compute_variables_for_range_learning(tensor,
                                                                     tensor_quantizer.bitwidth,
                                                                     encoding_min,
                                                                     encoding_max,
                                                                     tensor_quantizer.channel_axis,
                                                                     use_symmetric_encodings,
                                                                     tensor_quantizer.use_strict_symmetric,
                                                                     is_unsigned_symmetric)

    zero = torch.zeros_like(num_steps)

    x_round = torch.round(tensor / delta) - offset
    x_quant = x_round.clamp(zero, num_steps)
    x_dequant = (x_quant + offset) * delta

    mask_tensor = x_round.ge(zero) * x_round.le(num_steps)

    # Downcast x_quant if bitwidth is less than or equal to 8 to reduce memory consumption
    if tensor_quantizer.bitwidth <= 8:
        x_quant = x_quant.to(dtype=torch.uint8)

    intermediate_result = IntermediateResult(x_quant,
                                             encoding_min, encoding_max,
                                             delta, offset, mask_tensor, num_steps,
                                             use_symmetric_encodings, is_unsigned_symmetric)
    return x_dequant.to(orig_dtype), intermediate_result


# pylint:disable=too-many-locals
def asymmetric_gradients(tensor: torch.Tensor,
                         grad: torch.Tensor,
                         intermediate_result: IntermediateResult,
                         channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate asymmetric gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Input tensor of quant-dequant forward pass
    :param grad: Gradient w.r.t the output of quant-dequant
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to encoding min and max
    """
    mask_tensor = intermediate_result.mask_tensor
    encoding_min = intermediate_result.encoding_min
    encoding_max = intermediate_result.encoding_max
    delta = intermediate_result.delta
    offset = intermediate_result.offset
    x_quant = intermediate_result.x_quant
    num_steps = intermediate_result.num_steps

    grad_xq = delta * grad
    grad_scale = (x_quant + offset - tensor * mask_tensor / delta) * grad
    grad_offset = grad_xq * (~mask_tensor)

    if delta.numel() > 1 and len(tensor.shape) == 1:
        # NOTE: Handle when applying per-channel quant to 1-D Tensor case such as bias tensor in Conv or beta/gamma in BatchNorm
        intermediate_term1 = grad_scale / num_steps
        intermediate_term2 = num_steps / (encoding_max - encoding_min) ** 2 * grad_offset
    else:
        # Per-channel quant to k-D Tensor (k >= 2) or per-tensor case
        dim = list(range(len(tensor.shape)))
        if delta.numel() > 1 and len(tensor.shape) > 1:
            dim.pop(channel_axis)
        intermediate_term1 = grad_scale.sum(dim=dim) / num_steps
        intermediate_term2 = num_steps / (encoding_max - encoding_min) ** 2 * grad_offset.sum(dim=dim)

    grad_encoding_min = -intermediate_term1 + encoding_max * intermediate_term2
    grad_encoding_min = grad_encoding_min.view_as(encoding_min)
    grad_encoding_max = intermediate_term1 - encoding_min * intermediate_term2
    grad_encoding_max = grad_encoding_max.view_as(encoding_max)

    return grad_encoding_min, grad_encoding_max


# pylint:disable=too-many-locals
def symmetric_gradients(tensor: torch.Tensor,
                        grad: torch.Tensor,
                        intermediate_result: IntermediateResult,
                        channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate signed symmetric gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Input tensor of quant-dequant forward pass
    :param grad: Gradient w.r.t the output of quant-dequant
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to encoding min and max
    """
    mask_tensor = intermediate_result.mask_tensor
    delta = intermediate_result.delta
    offset = intermediate_result.offset
    x_quant = intermediate_result.x_quant
    num_steps = intermediate_result.num_steps

    if delta.numel() > 1 and len(tensor.shape) == 1:
        # NOTE: Handle when applying per-channel quant to 1-D Tensor case such as bias tensor in Conv or beta/gamma in BatchNorm
        grad_encoding_max = ((x_quant + offset) * grad) - (mask_tensor * (tensor / delta) * grad)
    else:
        # Per-channel quant to k-D Tensor (k >= 2) or per-tensor case
        dim = list(range(len(tensor.shape)))
        if delta.numel() > 1 and len(tensor.shape) > 1:
            dim.pop(channel_axis)
        grad_encoding_max = ((x_quant + offset) * grad).sum(dim=dim) - (mask_tensor * (tensor / delta) * grad).sum(dim=dim)

    grad_encoding_max = grad_encoding_max / torch.div(num_steps, 2, rounding_mode="floor")
    grad_encoding_max = grad_encoding_max.view_as(intermediate_result.encoding_max)

    return -grad_encoding_max, grad_encoding_max


def calculate_gradients(tensor: torch.Tensor,
                        grad: torch.Tensor,
                        intermediate_result: IntermediateResult,
                        channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Input tensor of quant-dequant forward pass
    :param grad: Gradient w.r.t the output of quant-dequant
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to encoding min and max
    """
    if intermediate_result.is_symmetric:
        return symmetric_gradients(tensor, grad, intermediate_result, channel_axis)

    return asymmetric_gradients(tensor, grad, intermediate_result, channel_axis)


# pylint: disable=abstract-method
class RoundStraightThrough(torch.autograd.Function):
    """
    Defining gradient of rounding function as pass-through since round is a non-linearity
    """

    @staticmethod
    # pylint: disable=arguments-differ
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad
