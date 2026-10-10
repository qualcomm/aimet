# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Default quantization backend for quantizing weights and activations"""

import functools
from packaging import version
from typing import Callable, Optional, List, Tuple
import torch

try:
    import torch.ao.quantization.fx._decomposed
except ImportError:
    pass

from aimet_torch.utils import (
    _is_expandable,
    _ContextManager,
    _torch_compiler_is_exporting,
    _torch_compiler_is_compiling,
)
from ....experimental.onnx import _export as _onnx
from aimet_torch.experimental import pgs
from aimet_torch.quantization._utils import interleave, concretize_block_size


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
    if dtype == torch.float32:
        mantissa_bits = 23
    elif dtype == torch.float16:
        mantissa_bits = 10
    elif dtype == torch.bfloat16:
        mantissa_bits = 7
    elif dtype == torch.float64:
        mantissa_bits = 52
    else:
        raise ValueError(
            f"Expected dtype to be one of float32, float16, bfloat16, or float64; got {dtype}"
        )

    finfo = torch.finfo(dtype)
    within_range = finfo.min <= value <= finfo.max
    if not within_range:
        return False

    # Given floating point representation sign * 2^e * 1.m,
    # an integer i can be exactly represented with the target dtype
    # if the first non-zero bit (msb) and the last non-zero bit (lsb)
    # is at most mantissa_bits apart from each other.
    #
    #                msb       lsb
    #                 ↓         ↓
    #   i = ... 0 0 0 1 x ... x 1 0 0 ...
    #     =  2^msb * (1.x ... x 1)
    #     =  2^msb * 1.mantissa
    msb = max(0, value.bit_length() - 1)
    lsb = max(0, (value & -value).bit_length() - 1)
    return msb - lsb <= mantissa_bits


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
    #       This criteria allows up to 8-bit quantization with float16
    #       and 4-bit quantization with bfloat16
    if torch.finfo(dtype).eps > 0.25 / (qmax - qmin):
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
    if not _torch_compiler_is_compiling():
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

    if block_size:
        block_size = concretize_block_size(tensor.shape, scale.shape, block_size)
        tensor = tensor.reshape(-1, *interleave(scale.shape, block_size))
        scale = scale.view(interleave(scale.shape, 1))
        offset = offset.view(interleave(offset.shape, 1))

    return (
        QuantizeFunc.apply(
            tensor, scale.to(internal_dtype), offset.to(internal_dtype), qmin, qmax
        )
        .to(output_dtype)
        .view(orig_tensor_shape)
    )


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
    # Skip execution of actual Q/DQ logic during ONNX export to speed up export
    if torch.onnx.is_in_onnx_export():
        return tensor

    if not _torch_compiler_is_compiling():
        _validate_arguments(tensor, scale, qmin, qmax, block_size)

    _fast_forward = (
        # torch.fake_quantize doesn't support blockwise quantization
        block_size is None
        # torch.fake_quantize doesn't support JIT tracing
        and not torch.jit.is_tracing()
        # torch.fake_quantize doesn't compute gradients for scale/offset
        and (
            not scale.requires_grad
            and not offset.requires_grad
            or not torch.is_grad_enabled()
        )
        # if user explicitly designated specific rounding function, honor it strictly
        and _round_fn == torch.round
        and _round_fn_inplace == torch.round_
        and zero_point_shift == 0.0
        # PGS is not supported with torch.fake_quantize
        and not (tensor.requires_grad and pgs.is_pgs_enabled())
    )

    if _fast_forward:
        ret = _torch_fake_quantize(tensor, scale, offset, qmin, qmax)

        if ret is not None:
            return ret

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

    if block_size:
        block_size = concretize_block_size(tensor.shape, scale.shape, block_size)
        tensor = tensor.reshape(-1, *interleave(scale.shape, block_size))
        scale = scale.view(interleave(scale.shape, 1))
        offset = offset.view(interleave(offset.shape, 1))

    scale = scale.to(internal_dtype)
    shifted_tensor = tensor
    qdq_tensor = QuantDequantFunc.apply(
        shifted_tensor,
        scale,
        offset.to(internal_dtype),
        qmin,
        qmax,
        zero_point_shift,
    )

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

    @torch.library.impl("aimet::quantize_dequantize", "meta")
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

    if tensor_internal_dtype == tensor.dtype:
        tensor_internal_dtype = None

    is_per_tensor = scale.numel() == offset.numel() == 1

    if is_per_tensor:
        tensor = tensor.to(tensor_internal_dtype)
        scale = scale.to(scale_internal_dtype)
        if (
            hasattr(offset, "_precomputed_zero_point")
            and not _torch_compiler_is_compiling()
            and not _torch_compiler_is_exporting()
        ):
            zp = getattr(offset, "_precomputed_zero_point").to(offset.device)
            setattr(offset, "_precomputed_zero_point", zp)
        else:
            zp = -offset.to(torch.int32)

        output = _call_torch_fake_quantize_per_tensor(
            tensor,
            scale,
            zp,
            qmin,
            qmax,
        )
        if output is not None:
            output = output.to(output_dtype if output.dtype != output_dtype else None)
        return output

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
                if (
                    hasattr(offset, "_precomputed_zero_point")
                    and not _torch_compiler_is_compiling()
                    and not _torch_compiler_is_exporting()
                ):
                    zp = getattr(offset, "_precomputed_zero_point").to(offset.device)
                    setattr(offset, "_precomputed_zero_point", zp)
                else:
                    zp = -offset.to(torch.int32)

                output = _call_torch_fake_quantize_per_channel(
                    tensor,
                    scale.flatten() if scale.dim() > 1 else scale,
                    zp.flatten() if zp.dim() > 1 else zp,
                    axis,
                    qmin,
                    qmax,
                )
                if output is not None:
                    output = output.to(
                        output_dtype if output.dtype != output_dtype else None
                    )
                return output
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
) -> Optional[torch.Tensor]:
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

    if _torch_compiler_is_compiling():
        return None

    return torch.fake_quantize_per_tensor_affine(
        input,
        scale.view(()) if scale.dim() > 0 else scale,
        zero_point.view(()) if zero_point.dim() > 0 else zero_point,
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
) -> Optional[torch.Tensor]:
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

    if _torch_compiler_is_compiling():
        return None

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
    if not _torch_compiler_is_compiling():
        _validate_arguments(tensor, scale, block_size=block_size)

    output_dtype = internal_dtype = tensor.dtype

    orig_tensor_shape = tensor.shape

    if block_size:
        block_size = concretize_block_size(tensor.shape, scale.shape, block_size)
        tensor = tensor.reshape(-1, *interleave(scale.shape, block_size))
        scale = scale.view(interleave(scale.shape, 1))
        offset = offset.view(interleave(offset.shape, 1))

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
        # masked_grad is always set when any of the *_requires_grad below is True, since
        # they're all covered by the "or" in the assignment's guard above.
        tensor_grad = masked_grad / scale if ctx.tensor_requires_grad else None  # pylint: disable=possibly-used-before-assignment
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
        # tensor_and_offset_grad is always set when tensor_requires_grad or offset_requires_grad
        # is True, since both are covered by the "or" in the assignment's guard above.
        tensor_grad = tensor_and_offset_grad if ctx.tensor_requires_grad else None  # pylint: disable=possibly-used-before-assignment
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
        zero_point_shift: float,
    ):
        if _USE_COMPILED_IMPL:
            impl = __class__._compiled_forward_impl
        else:
            impl = __class__._forward_impl
        return impl(ctx, tensor, scale, offset, qmin, qmax, zero_point_shift)

    @staticmethod
    def _forward_impl(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        zero_point_shift: float,
    ):
        x_round = _round_fn_inplace(
            tensor.to(scale.dtype) / scale
            if zero_point_shift == 0
            else tensor.to(scale.dtype) / scale - zero_point_shift
        ).sub_(offset)

        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (qmin <= x_round) & (x_round <= qmax)
        else:
            mask = None

        x_quant = x_round.clamp_(qmin, qmax)
        x_dequant = x_quant.add_(
            offset if zero_point_shift == 0 else offset + zero_point_shift
        ).mul_(scale)

        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.zero_point_shift = zero_point_shift
        ctx.pgs_eps = pgs.get_pgs_eps()
        ctx.pgs_multiplier = pgs.get_pgs_multiplier()
        ctx.save_for_backward(
            tensor
            if scale.requires_grad or (tensor.requires_grad and pgs.is_pgs_enabled())
            else None,
            scale
            if scale.requires_grad
            or offset.requires_grad
            or (tensor.requires_grad and pgs.is_pgs_enabled())
            else None,
            offset if scale.requires_grad else None,
            mask,
        )
        return x_dequant

    _compiled_forward_impl = staticmethod(_compile(_forward_impl.__func__))

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        qmax, qmin = ctx.qmax, ctx.qmin
        zero_point_shift = ctx.zero_point_shift
        pgs_eps = ctx.pgs_eps
        pgs_multiplier = ctx.pgs_multiplier
        tensor, scale, offset, mask = ctx.saved_tensors

        tensor_grad = grad
        scale_grad = None
        offset_grad = None

        pgs_enabled = (
            pgs_eps > 0.0 and pgs_multiplier != 1.0 and ctx.tensor_requires_grad
        )

        if ctx.scale_requires_grad or pgs_enabled:
            x_scaled = tensor.to(scale.dtype) / scale - zero_point_shift
            x_rounded = _round_fn(x_scaled)
            rounding_err = x_rounded - x_scaled
            del x_scaled

            if ctx.scale_requires_grad:
                x_rounded.clamp_(offset + qmin, offset + qmax)
                if zero_point_shift != 0.0:
                    x_rounded.add_(zero_point_shift)
                scale_grad = grad * torch.where(
                    mask,
                    rounding_err,
                    x_rounded,
                )

            del x_rounded

            if pgs_enabled:
                is_near_rounding_boundary = rounding_err.abs_() > (1 - pgs_eps) / 2
                tensor_grad = torch.where(
                    is_near_rounding_boundary,
                    grad * pgs_multiplier,
                    grad,
                )
                del is_near_rounding_boundary

            del rounding_err

        if ctx.tensor_requires_grad:
            tensor_grad = torch.where(mask, tensor_grad, 0)
        else:
            tensor_grad = None

        if ctx.offset_requires_grad:
            offset_grad = torch.where(mask, 0, grad * scale)

        return tensor_grad, scale_grad, offset_grad, None, None, None


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
