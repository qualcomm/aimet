# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=redefined-builtin, import-error, abstract-method, arguments-differ, no-member
from __future__ import annotations
from packaging.version import parse
from typing import Optional, Sequence
import numpy as np
import torch
from aimet_torch.experimental import pgs
from . import torch_builtins
from .torch_builtins import (
    _validate_arguments,
    _is_grid_representable,
    _is_numerically_stable,
)
from aimet_torch.common.utils import AimetLogger


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


try:
    import triton
    import triton.language as tl

    _is_triton_available: bool = bool(
        torch.cuda.is_available()
        and triton
        and parse(triton.__version__) >= parse("3.0.0")
    )

    @triton.jit
    def quantize_per_tensor(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        output_ptr: tl.pointer_type,
        n_elements: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start_index = pid * COMPUTE_BLOCK_SIZE
        idx = start_index + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < n_elements
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr)
        offset = tl.load(offset_ptr)
        input = input / scale - offset
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(output_ptr + idx, clamped, mask=mask)

    @triton.jit
    def quantize_per_channel(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        output_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Quantize input of shape (I, J, K) with per-channel scale and offset
        where J is the channel dimension.

        input:                   scale:
              ┌───────────────┐     ┌┐
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
            J ├───────────────┤   J ├┤
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
              └───────────────┘     └┘
                      K

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (1D tensor of size J).
            offset_ptr: Pointer to the quantization offset (1D tensor of size J).
            qmin: Minimum value in the quantization grid.
            qmax: Maximum value in the quantization grid.
            output_ptr: Pointer to the output buffer.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (channel dimension).
            K: Size of the third dimension of the input tensor.
            COMPUTE_BLOCK_SIZE: Number of elements to process per block.
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)
        j = (idx % (J * K)) // K
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + j)
        offset = tl.load(offset_ptr + j)
        input = input / scale - offset
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(output_ptr + idx, clamped, mask=mask)

    @triton.jit
    def quantize_per_block(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        output_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        BLK_SIZE_J: tl.int32,
        BLK_SIZE_K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Quantize input of shape (I, J, K) with block-wise scale and offset
        where J and K are block dimensions.

        input:                            scale:
            ┬  ┌──────┬──────┬───┬──────┐    ┬ ┌┬┬┬┬┬┐
            │  │B[1,1]│B[1,2]│...│B[1,M]│    │ ├┼┼┼┼┼┤
            │  ├──────┼──────┼───┼──────┤    N ├┼┼┼┼┼┤
            │  │B[2,1]│B[2,2]│...│B[2,M]│    │ ├┼┼┼┼┼┤
            J  ├──────┼──────┼───┼──────┤    ┴ └┴┴┴┴┴┘
            │  │ ...  │ ...  │...│ ...  │      ├─ M ─┤
            │  ├──────┼──────┼───┼──────┤
            │  │B[N,1]│B[N,2]│...│B[N,M]│
            ┴  └──────┴──────┴───┴──────┘
               ├─────────── K ──────────┤

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (2D tensor of size [N, M]).
            offset_ptr: Pointer to the quantization offset (2D tensor of size [N, M]).
            qmin: Minimum value in the quantization grid.
            qmax: Maximum value in the quantization grid.
            output_ptr: Pointer to the output buffer.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (block dimension 0).
            K: Size of the third dimension of the input tensor. (block dimension 1).
            BLK_SIZE_J: Block size along dimension J. (BLK_SIZE_J = J / N)
            BLK_SIZE_K: Block size along dimension K. (BLK_SIZE_K = K / M)
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)

        j = (idx % (J * K)) // K
        k = idx % K

        # B[n, m] where n and m are block indices along J and K dimensions
        # 0 <= n < N == J / BLK_SIZE_J
        # 0 <= m < M == K / BLK_SIZE_K
        n = j // BLK_SIZE_J
        m = k // BLK_SIZE_K

        scale_idx = n * (K // BLK_SIZE_K) + m

        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + scale_idx)
        offset = tl.load(offset_ptr + scale_idx)
        input = input / scale - offset
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(output_ptr + idx, clamped, mask=mask)

    @triton.jit
    def quantize_dequantize_per_tensor(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        zero_point_shift,
        output_ptr: tl.pointer_type,
        mask_ptr: tl.pointer_type,
        n_elements: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Args:
            input_ptr: Pointer to the input tensor.
            scale: Quantization scale (scalar).
            offset: Quantizaiton offset (scalar).
            qmin: Minimum value in the quantization grid.
            qmax: Maximum value in the quantization grid.
            output_ptr: Pointer to the output buffer.
            mask_ptr: If not None, store a boolean mask indicating whether each element was clamped.
            n_elements: Number of elements in the input tensor.
            COMPUTE_BLOCK_SIZE: Number of elements to process per block.
        """
        pid = tl.program_id(0)
        start_index = pid * COMPUTE_BLOCK_SIZE
        idx = start_index + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < n_elements
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr)
        offset = tl.load(offset_ptr)
        input = input / scale - (zero_point_shift + offset)
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(
            output_ptr + idx, (clamped + zero_point_shift + offset) * scale, mask=mask
        )

        if mask_ptr is not None:
            tl.store(mask_ptr + idx, (clamped == rounded), mask=mask)

    @triton.jit
    def quantize_dequantize_per_channel(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        zero_point_shift,
        output_ptr: tl.pointer_type,
        mask_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Quantize-dequantize input of shape (I, J, K) with per-channel scale and offset
        where J is the channel dimension.

        input:                   scale:
              ┌───────────────┐     ┌┐
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
            J ├───────────────┤   J ├┤
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
              └───────────────┘     └┘
                      K

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (1D tensor of size J).
            offset_ptr: Pointer to the quantization offset (1D tensor of size J).
            qmin: Minimum value in the quantization grid.
            qmax: Maximum value in the quantization grid.
            output_ptr: Pointer to the output buffer.
            mask_ptr: If not None, store a boolean mask indicating whether each element was clamped.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (channel dimension).
            K: Size of the third dimension of the input tensor.
            COMPUTE_BLOCK_SIZE: Number of elements to process per block.
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)
        j = (idx % (J * K)) // K
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + j)
        offset = tl.load(offset_ptr + j)
        input = input / scale - (zero_point_shift + offset)
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(
            output_ptr + idx, (clamped + zero_point_shift + offset) * scale, mask=mask
        )

        if mask_ptr is not None:
            tl.store(mask_ptr + idx, (clamped == rounded), mask=mask)

    @triton.jit
    def quantize_dequantize_per_block(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        zero_point_shift,
        output_ptr: tl.pointer_type,
        mask_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        BLK_SIZE_J: tl.int32,
        BLK_SIZE_K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Quantize input of shape (I, J, K) with block-wise scale and offset
        where J and K are block dimensions.

        input:                            scale:
            ┬  ┌──────┬──────┬───┬──────┐    ┬ ┌┬┬┬┬┬┐
            │  │B[1,1]│B[1,2]│...│B[1,M]│    │ ├┼┼┼┼┼┤
            │  ├──────┼──────┼───┼──────┤    N ├┼┼┼┼┼┤
            │  │B[2,1]│B[2,2]│...│B[2,M]│    │ ├┼┼┼┼┼┤
            J  ├──────┼──────┼───┼──────┤    ┴ └┴┴┴┴┴┘
            │  │ ...  │ ...  │...│ ...  │      ├─ M ─┤
            │  ├──────┼──────┼───┼──────┤
            │  │B[N,1]│B[N,2]│...│B[N,M]│
            ┴  └──────┴──────┴───┴──────┘
               ├─────────── K ──────────┤

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (2D tensor of size [N, M]).
            offset_ptr: Pointer to the quantization offset (2D tensor of size [N, M]).
            qmin: Minimum value in the quantization grid.
            qmax: Maximum value in the quantization grid.
            output_ptr: Pointer to the output buffer.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (block dimension 0).
            K: Size of the third dimension of the input tensor. (block dimension 1).
            BLK_SIZE_J: Block size along dimension J. (BLK_SIZE_J = J / N)
            BLK_SIZE_K: Block size along dimension K. (BLK_SIZE_K = K / M)
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)

        j = (idx % (J * K)) // K
        k = idx % K

        # B[n, m] where n and m are block indices along J and K dimensions
        # 0 <= n < M == J / BLK_SIZE_J
        # 0 <= m < N == K / BLK_SIZE_K
        n = j // BLK_SIZE_J
        m = k // BLK_SIZE_K

        scale_idx = n * (K // BLK_SIZE_K) + m

        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + scale_idx)
        offset = tl.load(offset_ptr + scale_idx)
        input = input / scale - (zero_point_shift + offset)
        rounded = tl.floor(input + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        tl.store(
            output_ptr + idx, (clamped + zero_point_shift + offset) * scale, mask=mask
        )

        if mask_ptr is not None:
            tl.store(mask_ptr + idx, (clamped == rounded), mask=mask)

    @triton.jit
    def quantize_dequantize_per_tensor_backward(
        output_grad_ptr: tl.pointer_type,
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        zero_point_shift,
        mask_ptr: tl.pointer_type,
        input_grad_ptr: tl.pointer_type,
        scale_grad_ptr: tl.pointer_type,
        offset_grad_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        n_elements: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
        pgs_eps: tl.constexpr = 0.0,
        pgs_multiplier: tl.constexpr = 1.0,
    ):
        pid = tl.program_id(0)
        start_index = pid * COMPUTE_BLOCK_SIZE
        idx = start_index + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < n_elements

        output_grad = tl.load(output_grad_ptr + idx, mask=mask)
        input_grad = output_grad
        is_within_clamping_boundary = tl.load(mask_ptr + idx, mask=mask)

        if scale_grad_ptr is not None or (
            pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None
        ):
            input = tl.load(input_ptr + idx, mask=mask)
            scale = tl.load(scale_ptr)
            offset = tl.load(offset_ptr)
            scaled = input / scale - (zero_point_shift + offset)
            rounded = tl.floor(scaled + 0.5)
            rounding_err = rounded - scaled

            if scale_grad_ptr is not None:
                clamped = tl.clamp(rounded, qmin, qmax) + offset
                scale_grad = output_grad * tl.where(
                    is_within_clamping_boundary,
                    rounding_err,
                    clamped,
                )
                scale_grad = scale_grad.sum()
                tl.atomic_add(scale_grad_ptr, scale_grad)

            if pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None:
                is_near_rounding_boundary = rounding_err.abs() > (1 - pgs_eps) / 2
                input_grad = tl.where(
                    is_near_rounding_boundary,
                    output_grad * pgs_multiplier,
                    output_grad,
                )

        if input_grad_ptr is not None:
            input_grad = input_grad * is_within_clamping_boundary
            tl.store(input_grad_ptr + idx, input_grad, mask=mask)

        if offset_grad_ptr is not None:
            scale = tl.load(scale_ptr)
            offset_grad = (output_grad * scale * ~is_within_clamping_boundary).sum()
            tl.atomic_add(offset_grad_ptr, offset_grad)

    @triton.jit
    def quantize_dequantize_per_channel_backward(
        output_grad_ptr: tl.pointer_type,
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        zero_point_shift,
        mask_ptr: tl.pointer_type,
        input_grad_ptr: tl.pointer_type,
        scale_grad_ptr: tl.pointer_type,
        offset_grad_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
        pgs_eps: tl.constexpr = 0.0,
        pgs_multiplier: tl.constexpr = 1.0,
    ):
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)
        j = (idx % (J * K)) // K

        output_grad = tl.load(output_grad_ptr + idx, mask=mask)
        input_grad = output_grad
        is_within_clamping_boundary = tl.load(mask_ptr + idx, mask=mask)

        if scale_grad_ptr is not None or (
            pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None
        ):
            input = tl.load(input_ptr + idx, mask=mask)
            scale = tl.load(scale_ptr + j)
            offset = tl.load(offset_ptr + j)
            scaled = input / scale - (zero_point_shift + offset)
            rounded = tl.floor(scaled + 0.5)
            rounding_err = rounded - scaled

            if scale_grad_ptr is not None:
                clamped = tl.clamp(rounded, qmin, qmax) + offset
                scale_grad = output_grad * tl.where(
                    is_within_clamping_boundary,
                    rounding_err,
                    clamped,
                )
                tl.store(scale_grad_ptr + idx, scale_grad, mask=mask)

            if pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None:
                is_near_rounding_boundary = rounding_err.abs() > (1 - pgs_eps) / 2
                input_grad = tl.where(
                    is_near_rounding_boundary,
                    output_grad * pgs_multiplier,
                    output_grad,
                )

        if input_grad_ptr is not None:
            input_grad = input_grad * is_within_clamping_boundary
            tl.store(input_grad_ptr + idx, input_grad, mask=mask)

        if offset_grad_ptr is not None:
            scale = tl.load(scale_ptr + j)
            offset_grad = output_grad * scale * ~is_within_clamping_boundary
            tl.store(offset_grad_ptr + idx, offset_grad, mask=mask)

    @triton.jit
    def quantize_dequantize_per_block_backward(
        output_grad_ptr: tl.pointer_type,
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        zero_point_shift,
        mask_ptr: tl.pointer_type,
        input_grad_ptr: tl.pointer_type,
        scale_grad_ptr: tl.pointer_type,
        offset_grad_ptr: tl.pointer_type,
        qmin: tl.int32,
        qmax: tl.int32,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        BLK_SIZE_J: tl.int32,
        BLK_SIZE_K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
        pgs_eps: tl.constexpr = 0.0,
        pgs_multiplier: tl.constexpr = 1.0,
    ):
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)

        output_grad = tl.load(output_grad_ptr + idx, mask=mask)
        input_grad = output_grad
        is_within_clamping_boundary = tl.load(mask_ptr + idx, mask=mask)

        # B[n, m] where n and m are block indices along J and K dimensions
        # 0 <= n < M == J / BLK_SIZE_J
        # 0 <= m < N == K / BLK_SIZE_K
        j = (idx % (J * K)) // K
        k = idx % K
        n = j // BLK_SIZE_J
        m = k // BLK_SIZE_K
        scale_idx = n * (K // BLK_SIZE_K) + m

        if scale_grad_ptr is not None or (
            pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None
        ):
            input = tl.load(input_ptr + idx, mask=mask)
            scale = tl.load(scale_ptr + scale_idx)
            offset = tl.load(offset_ptr + scale_idx)
            scaled = input / scale - (zero_point_shift + offset)
            rounded = tl.floor(scaled + 0.5)
            rounding_err = rounded - scaled

            if scale_grad_ptr is not None:
                clamped = tl.clamp(rounded, qmin, qmax) + offset
                scale_grad = output_grad * tl.where(
                    is_within_clamping_boundary,
                    rounding_err,
                    clamped,
                )
                tl.store(scale_grad_ptr + idx, scale_grad, mask=mask)

            if pgs_eps > 0.0 and pgs_multiplier != 1.0 and input_grad_ptr is not None:
                is_near_rounding_boundary = rounding_err.abs() > (1 - pgs_eps) / 2
                input_grad = tl.where(
                    is_near_rounding_boundary,
                    output_grad * pgs_multiplier,
                    output_grad,
                )

        if input_grad_ptr is not None:
            input_grad = input_grad * is_within_clamping_boundary
            tl.store(input_grad_ptr + idx, input_grad, mask=mask)

        if offset_grad_ptr is not None:
            scale = tl.load(scale_ptr + scale_idx)
            offset_grad = output_grad * scale * ~is_within_clamping_boundary
            tl.store(offset_grad_ptr + idx, offset_grad, mask=mask)

    @triton.jit
    def dequantize_per_tensor(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        output_ptr: tl.pointer_type,
        n_elements: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start_index = pid * COMPUTE_BLOCK_SIZE
        idx = start_index + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < n_elements
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr)
        offset = tl.load(offset_ptr)
        tl.store(output_ptr + idx, (input + offset) * scale, mask=mask)

    @triton.jit
    def dequantize_per_channel(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        output_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Dequantize input of shape (I, J, K) with per-channel scale and offset
        where J is the channel dimension.

        input:                   scale:
              ┌───────────────┐     ┌┐
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
            J ├───────────────┤   J ├┤
              ├───────────────┤     ├┤
              ├───────────────┤     ├┤
              └───────────────┘     └┘
                      K

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (1D tensor of size J).
            offset_ptr: Pointer to the quantization offset (1D tensor of size J).
            output_ptr: Pointer to the output buffer.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (channel dimension).
            K: Size of the third dimension of the input tensor.
            COMPUTE_BLOCK_SIZE: Number of elements to process per block.
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)
        j = (idx % (J * K)) // K
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + j)
        offset = tl.load(offset_ptr + j)
        tl.store(output_ptr + idx, (input + offset) * scale, mask=mask)

    @triton.jit
    def dequantize_per_block(
        input_ptr: tl.pointer_type,
        scale_ptr: tl.pointer_type,
        offset_ptr: tl.pointer_type,
        output_ptr: tl.pointer_type,
        I: tl.int32,
        J: tl.int32,
        K: tl.int32,
        BLK_SIZE_J: tl.int32,
        BLK_SIZE_K: tl.int32,
        COMPUTE_BLOCK_SIZE: tl.constexpr,
    ):
        """
        Dequantize input of shape (I, J, K) with block-wise scale and offset
        where J and K are block dimensions.

        input:                            scale:
            ┬  ┌──────┬──────┬───┬──────┐    ┬ ┌┬┬┬┬┬┐
            │  │B[1,1]│B[1,2]│...│B[1,M]│    │ ├┼┼┼┼┼┤
            │  ├──────┼──────┼───┼──────┤    N ├┼┼┼┼┼┤
            │  │B[2,1]│B[2,2]│...│B[2,M]│    │ ├┼┼┼┼┼┤
            J  ├──────┼──────┼───┼──────┤    ┴ └┴┴┴┴┴┘
            │  │ ...  │ ...  │...│ ...  │      ├─ M ─┤
            │  ├──────┼──────┼───┼──────┤
            │  │B[N,1]│B[N,2]│...│B[N,M]│
            ┴  └──────┴──────┴───┴──────┘
               ├─────────── K ──────────┤

        Args:
            input_ptr: Pointer to the input tensor.
            scale_ptr: Pointer to the quantization scale (2D tensor of size [N, M]).
            offset_ptr: Pointer to the quantization offset (2D tensor of size [N, M]).
            output_ptr: Pointer to the output buffer.
            I: Size of the first dimension of the input tensor.
            J: Size of the second dimension of the input tensor (block dimension 0).
            K: Size of the third dimension of the input tensor. (block dimension 1).
            BLK_SIZE_J: Block size along dimension J. (BLK_SIZE_J = J / N)
            BLK_SIZE_K: Block size along dimension K. (BLK_SIZE_K = K / M)
        """
        pid = tl.program_id(0)
        idx = pid * COMPUTE_BLOCK_SIZE + tl.arange(0, COMPUTE_BLOCK_SIZE)
        mask = idx < (I * J * K)

        j = (idx % (J * K)) // K
        k = idx % K

        # B[n, m] where n and m are block indices along J and K dimensions
        # 0 <= n < M == J / BLK_SIZE_J
        # 0 <= m < N == K / BLK_SIZE_K
        n = j // BLK_SIZE_J
        m = k // BLK_SIZE_K

        scale_idx = n * (K // BLK_SIZE_K) + m

        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + scale_idx)
        offset = tl.load(offset_ptr + scale_idx)
        tl.store(output_ptr + idx, (input + offset) * scale, mask=mask)
except ImportError:
    _is_triton_available = False


def is_available() -> bool:
    return _is_triton_available


class _NotSupportedError(RuntimeError): ...


def _get_axes(
    input: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    block_size: Optional[Sequence[int]],
):
    if scale.shape != offset.shape:
        raise RuntimeError(
            "Scale and offset must have the same shape. "
            f"Got scale shape {scale.shape} and offset shape {offset.shape}."
        )

    # Pad to match tensor dimensions
    scale_shape = [
        *(1 for _ in range(input.dim() - scale.dim())),
        *scale.shape,
    ]

    if block_size is None:
        block_size = [
            input_dim // scale_dim
            for input_dim, scale_dim in zip(input.shape, scale_shape)
        ]

    # Pad to match tensor dimensions
    block_size = [
        *(1 for _ in range(input.dim() - len(block_size))),
        *block_size,
    ]

    # Concretize wildcard block size (-1)
    block_size = [
        input_dim // scale_dim if B == -1 else B
        for input_dim, scale_dim, B in zip(input.shape, scale_shape, block_size)
    ]

    block_axes = [
        axis for axis, (dim, B) in enumerate(zip(input.shape, block_size)) if dim != B
    ]

    if len(block_axes) == 0:
        return None, None, block_size

    if len(block_axes) == 1:
        (axis,) = block_axes
        return axis, None, block_size

    if len(block_axes) == 2:
        axis_0, axis_1 = block_axes
        return axis_0, axis_1, block_size

    raise _NotSupportedError(
        "Triton quantization with more than 2 block dimensions is not supported"
    )


class TritonQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        _,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[Sequence[int]],
    ):
        axis_0, axis_1, block_size = _get_axes(tensor, scale, offset, block_size)

        output = torch.empty_like(
            tensor, memory_format=torch.contiguous_format, layout=torch.strided
        )
        COMPUTE_BLOCK_SIZE = 1024
        NUM_COMPUTE_BLOCKS = tensor.numel() // COMPUTE_BLOCK_SIZE + 1

        if axis_0 is None:
            quantize_per_tensor[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                output,
                tensor.numel(),
                COMPUTE_BLOCK_SIZE,
            )
        elif block_size[axis_0] == 1 and axis_1 is None:
            channel_axis = axis_0
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            quantize_per_channel[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                output,
                I,
                J,
                K,
                COMPUTE_BLOCK_SIZE,
            )
        elif axis_0 is not None and axis_1 is not None:
            blk_axis_0 = axis_0
            blk_axis_1 = axis_1
            I = int(np.prod(tensor.shape[:blk_axis_0]))
            J = int(np.prod(tensor.shape[blk_axis_0:blk_axis_1]))
            K = int(np.prod(tensor.shape[blk_axis_1:]))
            BLK_SIZE_J = int(np.prod(block_size[blk_axis_0:blk_axis_1]))
            BLK_SIZE_K = int(np.prod(block_size[blk_axis_1:]))

            quantize_per_block[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                output,
                I,
                J,
                K,
                BLK_SIZE_J,
                BLK_SIZE_K,
                COMPUTE_BLOCK_SIZE,
            )
        else:
            raise RuntimeError

        return output

    def backward(self, grad):
        raise NotImplementedError(
            "Triton quantize kernel does not support backward. "
            "Please fall back to torch builtin implementation with "
            "`aimet_torch.quantization.set_backend('torch_builtins')`."
        )


class TritonDequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        _,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        block_size: Optional[Sequence[int]],
    ):
        axis_0, axis_1, block_size = _get_axes(tensor, scale, offset, block_size)

        output = torch.empty_like(
            tensor, memory_format=torch.contiguous_format, layout=torch.strided
        )
        COMPUTE_BLOCK_SIZE = 1024
        NUM_COMPUTE_BLOCKS = tensor.numel() // COMPUTE_BLOCK_SIZE + 1

        if axis_0 is None:
            dequantize_per_tensor[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                output,
                tensor.numel(),
                COMPUTE_BLOCK_SIZE,
            )
        elif block_size[axis_0] == 1 and axis_1 is None:
            channel_axis = axis_0
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            dequantize_per_channel[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                output,
                I,
                J,
                K,
                COMPUTE_BLOCK_SIZE,
            )
        elif axis_0 is not None and axis_1 is not None:
            blk_axis_0 = axis_0
            blk_axis_1 = axis_1
            I = int(np.prod(tensor.shape[:blk_axis_0]))
            J = int(np.prod(tensor.shape[blk_axis_0:blk_axis_1]))
            K = int(np.prod(tensor.shape[blk_axis_1:]))
            BLK_SIZE_J = int(np.prod(block_size[blk_axis_0:blk_axis_1]))
            BLK_SIZE_K = int(np.prod(block_size[blk_axis_1:]))

            dequantize_per_block[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                output,
                I,
                J,
                K,
                BLK_SIZE_J,
                BLK_SIZE_K,
                COMPUTE_BLOCK_SIZE,
            )
        else:
            raise RuntimeError

        return output

    def backward(self, grad):
        raise NotImplementedError(
            "Triton dequantize kernel does not support backward. "
            "Please fall back to torch builtin implementation with "
            "`aimet_torch.quantization.set_backend('torch_builtins')`."
        )


class TritonQuantizeDequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[Sequence[int]],
        zero_point_shift: float,
    ):
        axis_0, axis_1, block_size = _get_axes(tensor, scale, offset, block_size)

        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = torch.empty_like(
                tensor,
                dtype=torch.bool,
                memory_format=torch.contiguous_format,
                layout=torch.strided,
            )
        else:
            mask = None

        output = torch.empty_like(
            tensor, memory_format=torch.contiguous_format, layout=torch.strided
        )

        COMPUTE_BLOCK_SIZE = 1024
        NUM_COMPUTE_BLOCKS = tensor.numel() // COMPUTE_BLOCK_SIZE + 1

        if axis_0 is None:
            quantize_dequantize_per_tensor[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                zero_point_shift,
                output,
                mask,
                tensor.numel(),
                COMPUTE_BLOCK_SIZE,
            )
        elif block_size[axis_0] == 1 and axis_1 is None:
            channel_axis = axis_0
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            quantize_dequantize_per_channel[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                zero_point_shift,
                output,
                mask,
                I,
                J,
                K,
                COMPUTE_BLOCK_SIZE,
            )
            ctx.I = I
            ctx.J = J
            ctx.K = K

        elif axis_0 is not None and axis_1 is not None:
            blk_axis_0 = axis_0
            blk_axis_1 = axis_1
            I = int(np.prod(tensor.shape[:blk_axis_0]))
            J = int(np.prod(tensor.shape[blk_axis_0:blk_axis_1]))
            K = int(np.prod(tensor.shape[blk_axis_1:]))
            BLK_SIZE_J = int(np.prod(block_size[blk_axis_0:blk_axis_1]))
            BLK_SIZE_K = int(np.prod(block_size[blk_axis_1:]))

            quantize_dequantize_per_block[(NUM_COMPUTE_BLOCKS,)](
                tensor.contiguous(),
                scale.contiguous(),
                offset.contiguous(),
                qmin,
                qmax,
                zero_point_shift,
                output,
                mask,
                I,
                J,
                K,
                BLK_SIZE_J,
                BLK_SIZE_K,
                COMPUTE_BLOCK_SIZE,
            )
            ctx.I = I
            ctx.J = J
            ctx.K = K
            ctx.BLK_SIZE_J = BLK_SIZE_J
            ctx.BLK_SIZE_K = BLK_SIZE_K

        else:
            raise RuntimeError

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
            offset
            if scale.requires_grad or (tensor.requires_grad and pgs.is_pgs_enabled())
            else None,
            mask,
        )
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.zero_point_shift = zero_point_shift
        ctx.axis_0 = axis_0
        ctx.axis_1 = axis_1
        ctx.block_size = block_size
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad):
        input, scale, offset, mask = ctx.saved_tensors
        zero_point_shift = ctx.zero_point_shift
        axis_0 = ctx.axis_0
        axis_1 = ctx.axis_1
        block_size = ctx.block_size

        grad = grad.contiguous()
        if input is not None:
            input = input.contiguous()
        if scale is not None:
            scale = scale.contiguous()
        if offset is not None:
            offset = offset.contiguous()
        if mask is not None:
            mask = mask.contiguous()

        input_grad = (
            torch.empty_like(
                grad, memory_format=torch.contiguous_format, layout=torch.strided
            )
            if ctx.tensor_requires_grad
            else None
        )

        COMPUTE_BLOCK_SIZE = 1024
        NUM_COMPUTE_BLOCKS = grad.numel() // COMPUTE_BLOCK_SIZE + 1

        pgs_eps = ctx.pgs_eps
        pgs_multiplier = ctx.pgs_multiplier

        if axis_0 is None:
            scale_grad = torch.zeros_like(scale) if ctx.scale_requires_grad else None
            offset_grad = torch.zeros_like(scale) if ctx.offset_requires_grad else None
            quantize_dequantize_per_tensor_backward[(NUM_COMPUTE_BLOCKS,)](
                grad,
                input,
                scale,
                offset,
                zero_point_shift,
                mask,
                input_grad,
                scale_grad,
                offset_grad,
                ctx.qmin,
                ctx.qmax,
                grad.numel(),
                COMPUTE_BLOCK_SIZE,
                pgs_eps,
                pgs_multiplier,
            )
        elif block_size[axis_0] == 1 and axis_1 is None:
            I = ctx.I
            J = ctx.J
            K = ctx.K
            scale_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.scale_requires_grad
                else None
            )
            offset_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.offset_requires_grad
                else None
            )
            quantize_dequantize_per_channel_backward[(NUM_COMPUTE_BLOCKS,)](
                grad,
                input,
                scale,
                offset,
                zero_point_shift,
                mask,
                input_grad,
                scale_grad,
                offset_grad,
                ctx.qmin,
                ctx.qmax,
                I,
                J,
                K,
                COMPUTE_BLOCK_SIZE,
                pgs_eps,
                pgs_multiplier,
            )
        elif axis_0 is not None and axis_1 is not None:
            I = ctx.I
            J = ctx.J
            K = ctx.K
            BLK_SIZE_J = ctx.BLK_SIZE_J
            BLK_SIZE_K = ctx.BLK_SIZE_K

            scale_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.scale_requires_grad
                else None
            )
            offset_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.offset_requires_grad
                else None
            )
            quantize_dequantize_per_block_backward[(NUM_COMPUTE_BLOCKS,)](
                grad,
                input,
                scale,
                offset,
                zero_point_shift,
                mask,
                input_grad,
                scale_grad,
                offset_grad,
                ctx.qmin,
                ctx.qmax,
                I,
                J,
                K,
                BLK_SIZE_J,
                BLK_SIZE_K,
                COMPUTE_BLOCK_SIZE,
                pgs_eps,
                pgs_multiplier,
            )

            if scale_grad is not None:
                scale_grad = scale_grad.view(
                    *(dim for dim in scale_grad.shape[:axis_0]),
                    scale_grad.shape[axis_0] // block_size[axis_0],
                    block_size[axis_0],
                    *(dim for dim in scale_grad.shape[axis_0 + 1 : axis_1]),
                    scale_grad.shape[axis_1] // block_size[axis_1],
                    block_size[axis_1],
                    *(dim for dim in scale_grad.shape[axis_1 + 1 :]),
                )
                scale_grad = scale_grad.sum(dim=(axis_0 + 1, axis_1 + 2), keepdim=False)

            if offset_grad is not None:
                offset_grad = offset_grad.view(
                    *(dim for dim in offset_grad.shape[:axis_0]),
                    offset_grad.shape[axis_0] // block_size[axis_0],
                    block_size[axis_0],
                    *(dim for dim in offset_grad.shape[axis_0 + 1 : axis_1]),
                    offset_grad.shape[axis_1] // block_size[axis_1],
                    block_size[axis_1],
                    *(dim for dim in offset_grad.shape[axis_1 + 1 :]),
                )
                offset_grad = offset_grad.sum(
                    dim=(axis_0 + 1, axis_1 + 2), keepdim=False
                )
        else:
            raise NotImplementedError

        return input_grad, scale_grad, offset_grad, None, None, None, None


def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[list] = None,
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
    if not is_available():
        raise RuntimeError(
            "Triton backend is not available. "
            "Please ensure triton>=3.0.0 is installed and CUDA is available."
        )

    if not _compile_success[quantize]:
        # Fall back to aten impl
        return torch_builtins.quantize(tensor, scale, offset, qmin, qmax, block_size)

    if not (tensor.is_cuda and scale.is_cuda and offset.is_cuda):
        # Fall back to aten impl if triton is not available or inputs are not on cuda
        return torch_builtins.quantize(tensor, scale, offset, qmin, qmax, block_size)

    _validate_arguments(tensor, scale, qmin, qmax, block_size)

    output_dtype = internal_dtype = tensor.dtype

    if not _is_grid_representable(tensor.dtype, qmin, qmax):
        msg = f"{tensor.dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    if not _is_numerically_stable(internal_dtype, qmin, qmax):
        internal_dtype = torch.float32
        if not _is_numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float64

    try:
        return TritonQuantize.apply(
            tensor.to(internal_dtype),
            scale.to(internal_dtype),
            offset.to(internal_dtype),
            qmin,
            qmax,
            block_size,
        ).to(output_dtype)
    except _NotSupportedError:
        return torch_builtins.quantize(tensor, scale, offset, qmin, qmax, block_size)
    except Exception:  # pylint: disable=broad-except
        _compile_success[quantize] = False
        logger.warning(
            "Triton quantize kernel failed to compile. "
            "Falling back to torch builtin implementation."
        )
        return torch_builtins.quantize(tensor, scale, offset, qmin, qmax, block_size)


def quantize_dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[list] = None,
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
    if not is_available():
        raise RuntimeError(
            "Triton backend is not available. "
            "Please ensure triton>=3.0.0 is installed and CUDA is available."
        )

    if not _compile_success[quantize_dequantize]:
        # Fall back to aten impl
        return torch_builtins.quantize_dequantize(
            tensor,
            scale,
            offset,
            qmin,
            qmax,
            block_size,
            zero_point_shift,
        )

    if not (tensor.is_cuda and scale.is_cuda and offset.is_cuda):
        # Fall back to aten impl if triton is not available or inputs are not on cuda
        return torch_builtins.quantize_dequantize(
            tensor,
            scale,
            offset,
            qmin,
            qmax,
            block_size,
            zero_point_shift,
        )

    _validate_arguments(tensor, scale, qmin, qmax, block_size)

    output_dtype = internal_dtype = tensor.dtype

    if not _is_numerically_stable(internal_dtype, qmin, qmax):
        internal_dtype = torch.float32
        if not _is_numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float64

    if not _is_grid_representable(internal_dtype, qmin, qmax):
        msg = f"{internal_dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    try:
        return TritonQuantizeDequantize.apply(
            tensor.to(internal_dtype),
            scale.to(internal_dtype),
            offset.to(internal_dtype),
            qmin,
            qmax,
            block_size,
            zero_point_shift,
        ).to(output_dtype)
    except _NotSupportedError:
        return torch_builtins.quantize_dequantize(
            tensor,
            scale,
            offset,
            qmin,
            qmax,
            block_size,
            zero_point_shift,
        )
    except Exception:  # pylint: disable=broad-except
        _compile_success[quantize_dequantize] = False
        logger.warning(
            "Triton quantize_dequantize kernel failed to compile. "
            "Falling back to torch builtin implementation."
        )
        return torch_builtins.quantize_dequantize(
            tensor,
            scale,
            offset,
            qmin,
            qmax,
            block_size,
            zero_point_shift,
        )
    else:
        raise RuntimeError


def dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    block_size: Optional[list] = None,
) -> torch.Tensor:
    """
    Performs differentiable dequantize operation given scale and offset.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param block_size: Block sizes per dimension
    :return: Resulting tensor
    """
    if not is_available():
        raise RuntimeError(
            "Triton backend is not available. "
            "Please ensure triton>=3.0.0 is installed and CUDA is available."
        )

    if not _compile_success[dequantize]:
        # Fall back to aten impl
        return torch_builtins.dequantize(tensor, scale, offset, block_size)

    if not (tensor.is_cuda and scale.is_cuda and offset.is_cuda):
        # Fall back to aten impl if triton is not available or inputs are not on cuda
        return torch_builtins.dequantize(tensor, scale, offset, block_size)

    _validate_arguments(tensor, scale, block_size=block_size)

    try:
        return TritonDequantize.apply(tensor, scale, offset, block_size)
    except _NotSupportedError:
        return torch_builtins.dequantize(tensor, scale, offset, block_size)
    except Exception:  # pylint: disable=broad-except
        _compile_success[dequantize] = False
        logger.warning(
            "Triton dequantize kernel failed to compile. "
            "Falling back to torch builtin implementation."
        )
        return torch_builtins.dequantize(tensor, scale, offset, block_size)


_compile_success = {
    quantize: True,
    quantize_dequantize: True,
    dequantize: True,
}
