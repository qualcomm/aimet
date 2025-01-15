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
"""Utility APIs for onnx export"""

from contextlib import contextmanager, ExitStack
import functools
from typing import Sequence

import onnxscript
from onnxscript import opset15 as ops
import torch
from torch.onnx import is_in_onnx_export, symbolic_helper

from aimet_torch.v2.utils import patch_attr


aimet_opset = onnxscript.values.Opset(domain="aimet", version=1)


@onnxscript.script(aimet_opset, default_opset=ops)
def quantize(tensor, scale, offset, qmin: int, qmax: int, block_size: Sequence[int]):
    """Onnxscript implementation of affine quantize"""
    # Upscale scale/offset by the factor of block_size
    upscaled_shape = ops.Shape(scale) * block_size
    scale = ops.Resize(scale, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    upscaled_shape = ops.Shape(offset) * block_size
    offset = ops.Resize(offset, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    x_round = ops.Round(tensor / scale) - offset
    x_int = ops.Clip(x_round, qmin, qmax)
    return ops.Reshape(x_int, ops.Shape(tensor))


@onnxscript.script(aimet_opset, default_opset=ops)
def dequantize(tensor, scale, offset, block_size: Sequence[int]):
    """Onnxscript implementation of affine dequantize"""
    # Upscale scale/offset by the factor of block_size
    upscaled_shape = ops.Shape(scale) * block_size
    scale = ops.Resize(scale, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    upscaled_shape = ops.Shape(offset) * block_size
    offset = ops.Resize(offset, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    x_dq = (tensor + offset) * scale
    return ops.Reshape(x_dq, ops.Shape(tensor))


@onnxscript.script(aimet_opset, default_opset=ops)
def quantize_dequantize(tensor, scale, offset, qmin: int, qmax: int, block_size: Sequence[int]):
    """Onnxscript implementation of affine quantize-dequantize"""
    # Upscale scale/offset by the factor of block_size
    upscaled_shape = ops.Shape(scale) * block_size
    scale = ops.Resize(scale, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    upscaled_shape = ops.Shape(offset) * block_size
    offset = ops.Resize(offset, roi=None, scales=None, sizes=upscaled_shape, mode='nearest')

    x_round = ops.Round(tensor / scale) - offset
    x_int = ops.Clip(x_round, qmin, qmax)
    x_dq = (x_int + offset) * scale
    return ops.Reshape(x_dq, ops.Shape(tensor))



def _unsqueeze_scalar(g, tensor):
    # pylint: disable=protected-access
    shape = symbolic_helper._get_tensor_sizes(tensor) or []
    if len(shape) == 0:
        tensor = symbolic_helper._unsqueeze_helper(g, tensor, [0])
    return tensor


def _shape(tensor):
    return symbolic_helper._get_tensor_sizes(tensor) # pylint: disable=protected-access


def quantize_symbolic(g, tensor, scale, offset, qmin, qmax, block_size=None):
    """Onnx symbolic function definition for affine quantize"""
    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors
    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(reversed([
            input_dim // num_blocks for input_dim, num_blocks in zip(_shape(tensor)[::-1], _shape(scale)[::-1])
        ]))
        assert all(old == new for old, new in zip(old_block_size, new_block_size) if old != -1)
        block_size = new_block_size

    return g.onnxscript_op(quantize, tensor, scale, offset,
                           qmin_i=qmin, qmax_i=qmax, block_size_i=block_size).setType(tensor.type())


def dequantize_symbolic(g, tensor, scale, offset, block_size=None):
    """Onnx symbolic function definition for affine dequantize"""
    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors
    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(reversed([
            input_dim // num_blocks for input_dim, num_blocks in zip(_shape(tensor)[::-1], _shape(scale)[::-1])
        ]))
        assert all(old == new for old, new in zip(old_block_size, new_block_size) if old != -1)
        block_size = new_block_size

    return g.onnxscript_op(dequantize, tensor, scale, offset, block_size_i=block_size).setType(tensor.type())


def quantize_dequantize_symbolic(g, tensor, scale, offset, qmin, qmax, block_size=None):
    """Onnx symbolic function definition for affine quantize-dequantize"""
    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors
    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(reversed([
            input_dim // num_blocks for input_dim, num_blocks in zip(_shape(tensor)[::-1], _shape(scale)[::-1])
        ]))
        assert all(old == new for old, new in zip(old_block_size, new_block_size) if old != -1)
        block_size = new_block_size

    return g.onnxscript_op(quantize_dequantize, tensor, scale, offset,
                           qmin_i=qmin, qmax_i=qmax, block_size_i=block_size).setType(tensor.type())



def register_symbolic(symbolic_fn):
    """
    Register ONNX symbolic function definition for a regular python function.
    """
    def decorator(python_fn):
        class SymbolicHelper(torch.autograd.Function): # pylint: disable=abstract-method
            """Helper class for coupling an arbitrary python function with a onnx symbolic function"""
            @staticmethod
            def forward(ctx, *args, **kwargs):
                return python_fn(*args, **kwargs)

            backward = NotImplemented
            symbolic = staticmethod(symbolic_fn)

        @functools.wraps(python_fn)
        def wrapper(*args, **kwargs):
            if is_in_onnx_export():
                return SymbolicHelper.apply(*args, **kwargs)
            return python_fn(*args, **kwargs)

        return wrapper

    return decorator


def export(model: torch.nn.Module, *args, **kwargs):
    """
    Export a torch model to ONNX with precomputed scale and offset.
    """
    if not isinstance(model, torch.nn.Module):
        raise NotImplementedError

    with _precompute_encodings(model):
        # Precompute scale/offset before entering torch.onnx.export so that
        # scale/offset are always represented as a leaf inputs in the onnx graphs
        return torch.onnx.export(model, *args, **kwargs)


@contextmanager
def _precompute_encodings(model: torch.nn.Module):
    # pylint: disable=import-outside-toplevel
    from aimet_torch.quantization.base import QuantizerBase
    with ExitStack() as stack:
        for q in model.modules():
            if isinstance(q, QuantizerBase):
                ctx = patch_attr(q, 'get_encodings', functools.lru_cache(q.get_encodings))
                stack.enter_context(ctx)
                with torch.no_grad():
                    q.get_encodings()
        yield
