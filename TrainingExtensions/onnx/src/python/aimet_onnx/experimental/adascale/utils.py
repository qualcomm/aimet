# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import math
from typing import Optional, Tuple, Callable, List
import functools

import torch


def derive_symmetric_qmin_qmax(bitwidth: int) -> tuple[int, int]:
    """
    Given the bitwidth generate qmin and qmax (symmetric mode)
    """
    num_steps = 2**bitwidth - 1
    qmin = -math.ceil(num_steps / 2)
    qmax = math.floor(num_steps / 2)

    return qmin, qmax


def get_symmetric_offset(qmin, qmax, shape, dtype, device):
    """
    Generate offset for symmetric qmin and qmax
    """
    return torch.full(
        shape,
        fill_value=-round((qmin + qmax) / 2),
        requires_grad=False,
        dtype=dtype,
        device=device,
    )


class _StraightThroughEstimator(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs):  # pylint:disable=arguments-differ, unused-argument
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)


def is_expandable(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def is_reducible(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return is_expandable(target_shape, src_shape)  # pylint: disable=arguments-out-of-order


def reduce(inp_tensor: torch.Tensor, shape: Tuple[int, ...], reduce_op: Callable):
    """
    Reduce input into given shape.

    :param inp_tensor: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    """
    if not is_reducible(inp_tensor.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(inp_tensor.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (*itertools.repeat(1, len(inp_tensor.shape) - len(shape)), *shape)
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(
        inp_tensor.permute(permute_dims).reshape(-1, *shape), dim=0, keepdim=False
    )


def validate_arguments(
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

    elif not is_expandable(scale.shape, tensor.shape):
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


def is_value_representable(dtype: torch.dtype, value: int):
    """
    Return whether an integer value can be represented with the given dtype
    """
    dtype_repr = torch.tensor(value, dtype=dtype)
    return dtype_repr.isfinite() and dtype_repr.long() == value


@functools.lru_cache(None)
def is_grid_representable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range of integers can be represented with the given dtype
    """
    return (
        is_value_representable(dtype, qmax)
        and is_value_representable(dtype, qmax - 1)
        and is_value_representable(dtype, qmin + 1)
        and is_value_representable(dtype, qmin)
    )


def is_numerically_stable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range can be **stably** represented with the given dtype
    """
    if not is_grid_representable(dtype, qmin, qmax):
        return False

    # Degenerate case
    if qmin == qmax:
        return True

    # NOTE: This is a heuristic criteria. It doesn't perfectly guarantee numerical stability
    #       This criteria allows 8-bit quantization of float16, but it needs more discussion
    if torch.finfo(dtype).eps > 1e-1 / (qmax - qmin):
        return False

    return True


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
