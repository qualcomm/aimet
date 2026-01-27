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
"""Default quantization backend for quantizing weights and activations"""

import functools
from packaging import version
from typing import Callable, Optional, List, Tuple
import torch

try:
    import torch.ao.quantization.fx._decomposed
except ImportError:
    pass
from aimet_torch.v2.utils import (
    _is_expandable,
    _ContextManager,
    _torch_compiler_is_exporting,
)
import aimet_torch.v2.experimental.onnx._export as _onnx


_torch_version: Tuple[int, int, int] = (
    version.parse(torch.__version__).major,
    version.parse(torch.__version__).minor,
    version.parse(torch.__version__).micro,
)

if _torch_version >= (2, 0, 0):
    _compile = torch.compile
else:
    _compile = lambda fn: fn


def _is_value_representable(dtype: torch.dtype, value: int):
    """
    Return whether an integer value can be represented with the given dtype
    """
    dtype_repr = torch.tensor(value, dtype=dtype)
    return dtype_repr.isfinite() and dtype_repr.long() == value


@functools.lru_cache(None)
def _is_grid_representable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range of integers can be represented with the given dtype
    """
    return (
        _is_value_representable(dtype, qmax)
        and _is_value_representable(dtype, qmax - 1)
        and _is_value_representable(dtype, qmin + 1)
        and _is_value_representable(dtype, qmin)
    )


def _is_numerically_stable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range can be **stably** represented with the given dtype
    """
    if not _is_grid_representable(dtype, qmin, qmax):
        return False

    # Degenerate case
    if qmin == qmax:
        return True

    # NOTE: This is a heuristic criteria. It doesn't perfectly guarantee numerical stability
    #       This criteria allows 8-bit quantization of float16, but it needs more discussion
    if torch.finfo(dtype).eps > 1e-1 / (qmax - qmin):
        return False

    return True


def _validate_arguments(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    qmin: int = None,
    qmax: int = None,
    block_size: Optional[List] = None,
):
    if block_size is not None:
        if len(scale.shape) != len(block_size):
            raise RuntimeError(
                f"Length of scale shape {scale.shape} must equal length of block size {block_size}"
            )
        for i in range(1, len(block_size) + 1):
            if block_size[-i] == -1:
                # Block size is calculated based on input and encoding parameter shape
                if tensor.shape[-i] % scale.shape[-i] != 0:
                    raise RuntimeError(
                        f"Each tensor dimension size for tensor shape {tensor.shape} must divide "
                        f"evenly with corresponding scale dimension value for scale shape {scale.shape}"
                    )
            else:
                if block_size[-i] * scale.shape[-i] != tensor.shape[-i]:
                    raise RuntimeError(
                        f"Each tensor dimension size for tensor shape {tensor.shape} must equal the "
                        f"corresponding scale dimension size * block size for scale shape {scale.shape} "
                        f"and block size {block_size}"
                    )

    elif not _is_expandable(scale.shape, tensor.shape):
        msg = f"Scale of shape {scale.shape} cannot be expanded like input tensor of shape {tensor.shape}. "
        # Additional message if the tensor is empty
        if tensor.numel() == 0:
            msg += (
                "Detected that the tensor is empty, which may be caused by the following reasons: "
                "1. The input tensor is incorrect. "
                "2. Improper use of model inference without initializing DeepSpeed after offloading parameters."
            )
        raise RuntimeError(msg)

    if qmin is not None and qmax is not None:
        if qmin > qmax:
            raise RuntimeError(
                f"qmin ({qmin}) must be smaller than or equal to qmax ({qmax})"
            )


@_onnx.register_symbolic(_onnx.quantize_symbolic)
def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[List] = None,
) -> torch.Tensor:
    """
    Performs differentiable quantization given scale, offset, and quantization range.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param qmin: Minimum value of the quantization range
    :param qmax: Maximum value of the quantization range
    :param block_size: Block sizes per dimension
    """
    _validate_arguments(tensor, scale, qmin, qmax, block_size)

    output_dtype = internal_dtype = tensor.dtype

    if not _is_grid_representable(tensor.dtype, qmin, qmax):
        msg = f"{tensor.dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    if not _is_numerically_stable(internal_dtype, qmin, qmax):
        internal_dtype = torch.float32
        if not _is_numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float64

    orig_tensor_shape = tensor.shape
    tensor = reshape_tensor_for_blocks(tensor, scale.shape, block_size)
    scale = scale.view(get_encoding_shape_with_blocks(scale.shape, block_size))
    offset = offset.view(get_encoding_shape_with_blocks(offset.shape, block_size))
    return (
        QuantizeFunc.apply(
            tensor, scale.to(internal_dtype), offset.to(internal_dtype), qmin, qmax
        )
        .to(output_dtype)
        .view(orig_tensor_shape)
    )


_ALLOW_FAST_FORWARD = True  # temporary flag for debugging


@_onnx.register_symbolic(_onnx.quantize_dequantize_symbolic)
def quantize_dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[List] = None,
    zero_point_shift: float = 0.0,
) -> torch.Tensor:
    """
    Performs differentiable quantize-dequantize given scale, offset, and quantization range.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param qmin: Minimum value of the quantization range
    :param qmax: Maximum value of the quantization range
    :param block_size: Block sizes per dimension
    :param zero_point_shift: Shift tensor by an amount proportional to scale during quantize dequantize
    """
    _validate_arguments(tensor, scale, qmin, qmax, block_size)

    _fast_forward = _ALLOW_FAST_FORWARD

    # torch.fake_quantize doesn't support blockwise quantization
    _fast_forward &= block_size is None

    # torch.fake_quantize doesn't support JIT tracing
    _fast_forward &= not torch.jit.is_tracing()

    # torch.fake_quantize doesn't compute gradients for scale/offset
    _fast_forward &= (not scale.requires_grad and not offset.requires_grad) or (
        not torch.is_grad_enabled()
    )

    # if user explicitly designated specific rounding function, honor it strictly
    _fast_forward &= _round_fn == torch.round and _round_fn_inplace == torch.round_

    # if user explicitly designated specific rounding function, honor it strictly
    _fast_forward &= zero_point_shift == 0.0

    if _fast_forward:
        ret = _torch_fake_quantize(tensor, scale, offset, qmin, qmax)

        if ret is not None:
            return ret

    if _torch_compiler_is_exporting():
        raise RuntimeError

    output_dtype = internal_dtype = tensor.dtype

    # Skip numerical stability check during torch.export.export
    # as if-else statements in these util functions lead to graph break
    # although those checks are irrelevant for the sake of export

    if not _is_numerically_stable(internal_dtype, qmin, qmax):
        internal_dtype = torch.float32
        if not _is_numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float64

    if not _is_grid_representable(internal_dtype, qmin, qmax):
        msg = f"{internal_dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    orig_tensor_shape = tensor.shape
    tensor = reshape_tensor_for_blocks(tensor, scale.shape, block_size)
    scale = scale.view(get_encoding_shape_with_blocks(scale.shape, block_size)).to(
        internal_dtype
    )
    offset = offset.view(get_encoding_shape_with_blocks(offset.shape, block_size))
    shifted_tensor = tensor
    if zero_point_shift != 0.0:
        shifted_tensor = torch.sub(tensor, scale, alpha=zero_point_shift)
    qdq_tensor = QuantDequantFunc.apply(
        shifted_tensor, scale, offset.to(internal_dtype), qmin, qmax
    )
    if zero_point_shift != 0.0:
        qdq_tensor = torch.add(qdq_tensor, scale, alpha=zero_point_shift)

    return qdq_tensor.to(output_dtype).view(orig_tensor_shape)


if _torch_version >= (2, 4, 0):
    torch.library.define(
        "aimet::quantize_dequantize",
        "("
        "  Tensor input,"
        "  Tensor scale,"
        "  Tensor offset,"
        "  int qmin,"
        "  int qmax,"
        "  int[]? block_size,"
        "  float zero_point_shift"
        ") -> Tensor",
    )

    @torch.library.register_fake("aimet::quantize_dequantize")
    def quantize_dequantize_meta(  # pylint: disable=unused-argument
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[List] = None,
        zero_point_shift: float = 0.0,
    ) -> torch.Tensor:
        return torch.empty_like(tensor)

    # Register quantize_dequantize as torch.ops.aimet.quantize_dequantize
    torch.library.impl("aimet::quantize_dequantize", "default")(quantize_dequantize)


def _torch_fake_quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
) -> Optional[torch.Tensor]:
    scale_internal_dtype = None if scale.dtype == torch.float32 else torch.float32
    tensor_internal_dtype = tensor.dtype
    output_dtype = tensor.dtype

    if _torch_version < (2, 6, 0) and tensor_internal_dtype == torch.bfloat16:
        # torch.fake_quantize only supports bfloat16 in >=2.6.0
        tensor_internal_dtype = torch.float32

    if output_dtype == tensor_internal_dtype:
        output_dtype = None

    if tensor_internal_dtype == tensor.dtype:
        tensor_internal_dtype = None

    is_per_tensor = scale.numel() == offset.numel() == 1

    if is_per_tensor:
        tensor = tensor.to(tensor_internal_dtype)
        scale = scale.to(scale_internal_dtype)
        zp = -offset.to(torch.int32)
        return _call_torch_fake_quantize_per_tensor(
            tensor,
            scale.view(()) if scale.dim() > 0 else scale,
            zp.view(()) if zp.dim() > 0 else zp,
            qmin,
            qmax,
        ).to(output_dtype)

    scale_shape = tuple((*(1 for _ in range(tensor.dim() - scale.dim())), *scale.shape))
    if scale_shape != scale.shape:
        scale = scale.view(*scale_shape)
    offset_shape = tuple(
        (*(1 for _ in range(tensor.dim() - offset.dim())), *offset.shape)
    )
    if offset_shape != offset.shape:
        offset = offset.view(*offset_shape)

    is_per_channel = scale.shape == offset.shape and all(
        scale_dim in (1, tensor_dim)
        for scale_dim, tensor_dim in zip(scale.shape, tensor.shape)
    )

    if is_per_channel:
        axes = [axis for axis, scale_dim in enumerate(scale.shape) if scale_dim != 1]
        assert axes

        if len(axes) == 1:
            (axis,) = axes
            try:
                tensor = tensor.to(tensor_internal_dtype)
                scale = scale.to(scale_internal_dtype)
                zp = -offset.to(torch.int32)
                return _call_torch_fake_quantize_per_channel(
                    tensor,
                    scale.flatten() if scale.dim() > 1 else scale,
                    zp.flatten() if zp.dim() > 1 else zp,
                    axis,
                    qmin,
                    qmax,
                ).to(output_dtype)
            except RuntimeError:
                # NOTE: torch.fake_quantize_per_channel_affine throws runtime error
                # if zero_point is not in [qmin, qmax]. In practice, this error will
                # almost never occur because per-channel quantization always uses zero_point=0
                return None

    return None


@functools.lru_cache
def _get_dtype(qmin: int, qmax: int) -> torch.dtype:
    # torch.export only supports int8, int16, int32, uint8, and uint16
    for bitwidth in (8, 16, 32):
        if bitwidth != 32 and qmin == 0 and qmax == 2**bitwidth - 1:
            try:
                return getattr(torch, f"uint{bitwidth}")
            except AttributeError:
                pass

        if -(2 ** (bitwidth - 1)) <= qmin < qmax < 2 ** (bitwidth - 1):
            try:
                return getattr(torch, f"int{bitwidth}")
            except AttributeError:
                pass

    raise RuntimeError(
        f"qmin={qmin}, qmax={qmax} isn't representable "
        "with any integer dtypes available in pytorch"
    )


def _call_torch_fake_quantize_per_tensor(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    if _torch_compiler_is_exporting():
        dtype = _get_dtype(qmin, qmax)
        input_q = torch.ops.quantized_decomposed.quantize_per_tensor(
            input,
            scale.item(),
            zero_point.item(),
            qmin,
            qmax,
            dtype,
        )
        return torch.ops.quantized_decomposed.dequantize_per_tensor(
            input_q,
            scale.item(),
            zero_point.item(),
            qmin,
            qmax,
            dtype,
        )

    return torch.fake_quantize_per_tensor_affine(
        input,
        scale,
        zero_point,
        qmin,
        qmax,
    )


def _call_torch_fake_quantize_per_channel(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    if _torch_compiler_is_exporting():
        dtype = _get_dtype(qmin, qmax)
        input_q = torch.ops.quantized_decomposed.quantize_per_channel(
            input,
            scale,
            zero_point,
            axis,
            qmin,
            qmax,
            dtype,
        )
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            input_q,
            scale,
            zero_point,
            axis,
            qmin,
            qmax,
            dtype,
        )

    return torch.fake_quantize_per_channel_affine(
        input,
        scale,
        zero_point,
        axis,
        qmin,
        qmax,
    )


@_onnx.register_symbolic(_onnx.dequantize_symbolic)
def dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    block_size: Optional[List] = None,
) -> torch.Tensor:
    """
    Performs differentiable dequantize operation given scale and offset.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param block_size: Block sizes per dimension
    :return: Resulting tensor
    """
    _validate_arguments(tensor, scale, block_size=block_size)

    output_dtype = internal_dtype = tensor.dtype

    orig_tensor_shape = tensor.shape
    tensor = reshape_tensor_for_blocks(tensor, scale.shape, block_size)
    scale = scale.view(get_encoding_shape_with_blocks(scale.shape, block_size))
    offset = offset.view(get_encoding_shape_with_blocks(offset.shape, block_size))
    return (
        DequantizeFunc.apply(
            tensor, scale.to(internal_dtype), offset.to(internal_dtype)
        )
        .to(output_dtype)
        .view(orig_tensor_shape)
    )


_round_fn = torch.round
_round_fn_inplace = torch.round_


def _set_round_fn(
    round_fn: Callable[[torch.Tensor], torch.Tensor],
    round_fn_inplace: Callable[[torch.Tensor], torch.Tensor],
):
    global _round_fn, _round_fn_inplace  # pylint: disable=global-statement
    _round_fn = round_fn
    _round_fn_inplace = round_fn_inplace


# pylint: disable=abstract-method
class QuantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for quantization
    """

    # pylint: disable=arguments-differ, protected-access
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        if _USE_COMPILED_IMPL:
            impl = __class__._compiled_forward_impl
        else:
            impl = __class__._forward_impl
        return impl(ctx, tensor, scale, offset, qmin, qmax)

    @staticmethod
    def _forward_impl(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        x_round = _round_fn_inplace(tensor.to(scale.dtype) / scale).sub_(offset)

        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (x_round >= qmin) * (x_round <= qmax)
        else:
            mask = None
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(
            tensor if scale.requires_grad else None,
            scale if tensor.requires_grad or scale.requires_grad else None,
            mask,
        )
        return x_round.clamp_(qmin, qmax)

    _compiled_forward_impl = staticmethod(_compile(_forward_impl.__func__))

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, mask = ctx.saved_tensors
        if (
            ctx.tensor_requires_grad
            or ctx.scale_requires_grad
            or ctx.offset_requires_grad
        ):
            masked_grad = grad * mask
        tensor_grad = masked_grad / scale if ctx.tensor_requires_grad else None
        scale_grad = (
            -(masked_grad / scale) * (tensor / scale)
            if ctx.scale_requires_grad
            else None
        )
        offset_grad = -masked_grad if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad, None, None


# pylint: disable=abstract-method
class DequantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for dequantization
    """

    # pylint: disable=arguments-differ, protected-access
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor):
        if _USE_COMPILED_IMPL:
            impl = __class__._compiled_forward_impl
        else:
            impl = __class__._forward_impl
        return impl(ctx, tensor, scale, offset)

    @staticmethod
    def _forward_impl(
        ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
    ):
        x_dequant = (tensor + offset).mul_(scale)
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(
            tensor if scale.requires_grad else None,
            scale if tensor.requires_grad or offset.requires_grad else None,
            offset if scale.requires_grad else None,
        )
        return x_dequant

    _compiled_forward_impl = staticmethod(_compile(_forward_impl.__func__))

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, offset = ctx.saved_tensors
        if ctx.tensor_requires_grad or ctx.offset_requires_grad:
            tensor_and_offset_grad = grad * scale
        tensor_grad = tensor_and_offset_grad if ctx.tensor_requires_grad else None
        scale_grad = grad * (tensor + offset) if ctx.scale_requires_grad else None
        offset_grad = tensor_and_offset_grad if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad


# pylint: disable=abstract-method
class QuantDequantFunc(torch.autograd.Function):
    """
    Custom gradient function for quant-dequant
    """

    # pylint: disable=arguments-differ, protected-access
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        if _USE_COMPILED_IMPL:
            impl = __class__._compiled_forward_impl
        else:
            impl = __class__._forward_impl
        return impl(ctx, tensor, scale, offset, qmin, qmax)

    @staticmethod
    def _forward_impl(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        x_round = _round_fn_inplace(tensor.to(scale.dtype) / scale).sub_(offset)

        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (qmin <= x_round) & (x_round <= qmax)
        else:
            mask = None

        x_quant = x_round.clamp_(qmin, qmax)
        x_dequant = x_quant.add_(offset).mul_(scale)

        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.save_for_backward(
            tensor if scale.requires_grad else None,
            scale if scale.requires_grad or offset.requires_grad else None,
            offset if scale.requires_grad else None,
            mask,
        )
        return x_dequant

    _compiled_forward_impl = staticmethod(_compile(_forward_impl.__func__))

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        qmax, qmin = ctx.qmax, ctx.qmin
        tensor, scale, offset, mask = ctx.saved_tensors

        if ctx.scale_requires_grad:
            tensor = tensor.to(scale.dtype) / scale
            scale_grad = grad * (
                _round_fn(tensor).clamp(offset + qmin, offset + qmax) - (tensor * mask)
            )
        else:
            scale_grad = None

        del tensor, offset

        tensor_grad = grad * mask if ctx.tensor_requires_grad else None
        offset_grad = grad * (~mask * scale) if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad, None, None


def get_encoding_shape_with_blocks(
    original_encoding_shape: torch.Size, block_size: List[int]
):
    """
    Get new encoding param shape to account for block sizes. If block_size is not None, the original shape is
    interleaved with '1' in between each dimension. Otherwise, the original shape is returned.

    :param original_encoding_shape: Original encoding shape
    :param block_size: Block sizes per dimension
    :return: Encoding shape accounting for blocks
    """
    if block_size is None:
        return original_encoding_shape

    new_encoding_shape = []

    for size in original_encoding_shape:
        new_encoding_shape.append(size)
        new_encoding_shape.append(1)

    return new_encoding_shape


def reshape_tensor_for_blocks(
    tensor: torch.Tensor, encoding_shape: torch.Size, block_size: Optional[List]
) -> torch.Tensor:
    """
    Reshape tensor to account for block sizes. The new shape separates each dimension into num blocks and block size.
    The resulting tensor shape has twice as many dimensions as the starting shape.

    For example, given the following:
    tensor shape: [dim_1_size, dim_2_size, dim_3_size]
    block_size: [block_1_size, block_2_size, block_3_size]

    The input is reshaped into the following expanded shape:
    expanded shape: [dim_1_size / block_1_size, block_1_size, dim_2_size / block_2_size, block_2_size,
                     dim_3_size / block_3_size, block_3_size]

    This assumes that dimension sizes are divisible by block sizes and that no padding is required.
    If block_size is None, the original shape is returned.

    :param tensor: Tensor to reshape
    :param encoding_shape: Encoding param shape (without taking blocks into consideration)
    :param block_size: Block sizes per dimension
    :return: Reshaped tensor
    """
    if block_size is None:
        return tensor

    input_reshape = []
    for i in range(1, len(block_size) + 1):
        if block_size[-i] == -1:
            input_reshape.insert(0, tensor.shape[-i] // encoding_shape[-i])
            input_reshape.insert(0, encoding_shape[-i])
        else:
            input_reshape.insert(0, block_size[-i])
            input_reshape.insert(0, encoding_shape[-i])

    input_reshape = list(tensor.shape[: -len(block_size)]) + input_reshape

    return tensor.view(input_reshape)


_USE_COMPILED_IMPL = False


def _use_compiled_impl(flag: bool = True):
    orig = _USE_COMPILED_IMPL

    def action():
        global _USE_COMPILED_IMPL  # pylint: disable=global-statement
        _USE_COMPILED_IMPL = flag

    def cleanup():
        global _USE_COMPILED_IMPL  # pylint: disable=global-statement
        _USE_COMPILED_IMPL = orig

    return _ContextManager(action, cleanup)
