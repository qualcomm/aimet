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
from collections import defaultdict
from typing import Sequence, Iterable

import onnx
import onnxscript
from onnxscript import opset15 as ops
import torch
from torch.onnx import is_in_onnx_export, symbolic_helper

from aimet_torch.v2.utils import patch_attr


ONNX_QUANTIZER_OP_TYPES = ("quantize", "quantize_dequantize")
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


def remove_quantization_nodes_from_onnx_graph(model: onnx.ModelProto):
    """
    Remove quantization nodes from ONNX graph with quantization nodes
    :param model: ONNX model with quantization nodes
    """
    tensor_to_encoding_map = {}
    name_to_producer, name_to_consumer = _get_producer_consumer_info_from_onnx_graph(model)
    node_list = list(model.graph.node)

    for node in node_list:
        if node.op_type not in ONNX_QUANTIZER_OP_TYPES:
            continue

        # Get quantizer name in torch model
        encoding = _get_encoding_from_onnx_node(model, node)

        # Remove qdq node from graph
        model.graph.node.remove(node)

        # Remove scale and offset from onnx graph
        _remove_constants(model, node.input[1:])

        # Connect next node to the prev node of quantizer node
        if node.output[0] in name_to_consumer:
            tensor_to_encoding_map[node.input[0]] = encoding
            next_nodes = name_to_consumer[node.output[0]]
            for next_node in next_nodes:
                for input_index, input_name in enumerate(next_node.input):
                    if input_name == node.output[0]:
                        next_node.input.remove(input_name)
                        next_node.input.insert(input_index, node.input[0])
                        break
                else:
                    raise ValueError(f"Could not find input name {node.output[0]} from node {next_node.name}")

        # Connect prev node to the next node of quantizer node if above is not possible
        elif node.input[0] in name_to_producer:
            tensor_to_encoding_map[node.output[0]] = encoding
            prev_node = name_to_producer[node.input[0]]
            for output_index, output_name in enumerate(prev_node.output):
                if output_name == node.input[0]:
                    prev_node.output.remove(output_name)
                    prev_node.output.insert(output_index, node.output[0])

        else:
            raise ValueError(f"Cannot find prev node and next node for quantization node {node.name}")

    return tensor_to_encoding_map


def _get_tensor_from_constant_name(onnx_model: onnx.ModelProto, constant_name: str):
    """
    Returns tensor from the constant name.
    """
    for node in onnx_model.graph.node:
        if constant_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return onnx.numpy_helper.to_array(attr.t)
            raise RuntimeError(f"Cannot find value attribute inside constant node {constant_name}")
    raise RuntimeError(f"Cannot find constant with name {constant_name} in onnx model")


def _get_encoding_from_onnx_node(onnx_model: onnx.ModelProto, quant_node: onnx.NodeProto):
    """
    Get encoding from quantization node.
    """
    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
    assert quant_node.op_type in ONNX_QUANTIZER_OP_TYPES

    qmin, qmax, block_size = None, None, None
    scale_name, offset_name = quant_node.input[1], quant_node.input[2]

    for attr in quant_node.attribute:
        if attr.name == "qmin":
            qmin = attr.i
        if attr.name == "qmax":
            qmax = attr.i
        if attr.name == "block_size":
            block_size = attr.ints
            if block_size == [1]:
                block_size = None

        scale = torch.tensor(_get_tensor_from_constant_name(onnx_model, scale_name))
        offset = torch.tensor(_get_tensor_from_constant_name(onnx_model, offset_name))

    return AffineEncoding(scale, offset, qmin, qmax, block_size=block_size)


def _remove_constants(onnx_model: onnx.ModelProto, constant_names: Iterable[str]):
    """
    Remove constants from onnx model.
    """
    constant_names = set(constant_names)
    for node in onnx_model.graph.node[::-1]:
        if node.op_type == "Constant" and node.output[0] in constant_names:
            onnx_model.graph.node.remove(node)


def _get_producer_consumer_info_from_onnx_graph(onnx_model: onnx.ModelProto):
    """
    Get producer and consumer information from ONNX graph for graph traversal.
    :param onnx_model: ONNX model
    :return: Tuple of name to producer mappings and name to consumer mappings
    """
    name_to_producer = {}
    name_to_consumer = defaultdict(list)

    for node in onnx_model.graph.node:
        for output_name in node.output:
            name_to_producer[output_name] = node

        for input_name in node.input:
            name_to_consumer[input_name].append(node)

    return name_to_producer, name_to_consumer


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
