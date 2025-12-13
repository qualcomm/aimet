# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Collection of onnx-related util functions that can be shared across aimet-onnx and aimet-torch"""

# pylint: disable=no-member, import-error

from collections import deque, defaultdict
import functools
import itertools
from typing import Iterable, Optional, Sequence, Dict, List, Union, Mapping

import os
import tempfile

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto
from onnx.numpy_helper import from_array, to_array
from onnx.external_data_helper import (
    load_external_data_for_tensor,
    uses_external_data,
    _get_all_tensors,
)

from ..onnx import opset10, opset13, opset21
from ..utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def _add_onnx_qdq_node(
    model: ModelProto,
    input_name: str,
    output_name: str,
    node_name_prefix: str,
    encodings: dict,
    float_type: int,
    onnx_opset: int,
    prequantize_constants: bool,
    base_dir: Optional[str] = None,
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encoding, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    _add_onnx_qdq_nodes(
        model,
        [input_name],
        [output_name],
        [node_name_prefix],
        [encodings],
        [float_type],
        onnx_opset,
        prequantize_constants,
        base_dir=base_dir,
    )


def _add_onnx_qdq_nodes(
    model: ModelProto,
    input_names: Iterable[str],
    output_names: Iterable[str],
    node_name_prefixes: Iterable[str],
    encodings: Iterable[dict],
    float_types: Iterable[np.dtype],
    onnx_opset: int,
    prequantize_constants: bool,
    base_dir: Optional[str] = None,
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encodings, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    if onnx_opset < 10:
        raise RuntimeError(
            "ONNX opset {} cannot represent QuantizeLinear and DequantizeLinear nodes."
            "So not able to export model as ONNX QDQ graph"
        )

    if onnx_opset < 13:
        opset = opset10
    elif onnx_opset < 21:
        opset = opset13
    else:
        opset = opset21

    constants = _get_all_constants(model)
    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding, float_type in zip(
        input_names, output_names, node_name_prefixes, encodings, float_types
    ):
        inputs_to_rename[input_name] = output_name
        output_dtype = encoding["output_dtype"]
        axis = encoding.get("axis", None)
        block_size = encoding.get("block_size", None)
        y_zero_point = encoding.get("y_zero_point", None)

        y_scale = np.array(
            encoding.get("y_scale") or encoding.get("per_channel_float_scale")
        ).astype(float_type)
        per_block_int_scale = (
            np.array(encoding["per_block_int_scale"])
            if "per_block_int_scale" in encoding
            else None
        )

        if y_zero_point is not None:
            y_zero_point = np.array(encoding["y_zero_point"], dtype=np.int64)
        elif per_block_int_scale is not None:
            y_zero_point = np.zeros(per_block_int_scale.shape, dtype=np.int64)
        else:
            y_zero_point = np.zeros(y_scale.shape, dtype=np.int64)

        tensors_to_add.append(
            opset.DequantizeLinear.make_zero_point(
                y_zero_point, dtype=output_dtype, name=f"{input_name}_zero_point"
            )
        )

        if per_block_int_scale is None:
            tensors_to_add.append(from_array(y_scale, name=f"{input_name}_scale"))
        else:
            # Export LPBQ.
            #
            # Strategy: Derive y_scale from per_channel_float_scale and per_block_uint_scale
            #
            #           (FLOAT)
            # per_channel_float_scale -----+
            #                              +--> DequantizeLinear -----+ (blockwise scale)
            #    per_block_uint_scale -----+                          |
            #           (UINT8)                               +-------+---------+
            #                                                 V                 V
            #                              weight ---> QuantizeLinear -> DequantizeLinear -> ...
            if output_dtype != "int4":
                raise RuntimeError(
                    f"LPBQ can be only exported with int4; got {output_dtype}"
                )

            try:
                weight_dims = next(
                    init for init in model.graph.initializer if init.name == input_name
                ).dims
            except StopIteration:
                weight_dims = None

            if weight_dims is not None and len(weight_dims) != 2:
                raise RuntimeError(
                    "LPBQ can be only applied to 2D matrices when exported to onnx QDQ. "
                    f'Got "{input_name}" with shape {weight_dims}'
                )

            if axis not in (-2, -1, 0, 1):
                raise RuntimeError(
                    "LPBQ can be only applied to 2D matrices when exported to onnx QDQ. "
                    f"Got axis {axis}"
                )

            tensors_to_add.extend(
                [
                    from_array(
                        y_scale.flatten(), name=f"{input_name}_per_channel_float_scale"
                    ),
                    from_array(
                        per_block_int_scale.astype(np.uint8),
                        name=f"{input_name}_per_block_uint_scale",
                    ),
                ]
            )
            nodes_to_add.extend(
                [
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_scale_dq",
                        inputs=[
                            f"{input_name}_per_block_uint_scale",
                            f"{input_name}_per_channel_float_scale",
                        ],
                        output=f"{input_name}_scale",
                        dtype="uint8",
                        axis=0 if axis in (-1, 1) else 1,  # == channel axis
                    )
                ]
            )

        input_q = None
        if prequantize_constants or output_dtype in ("int32", "uint32"):
            const = constants.get(input_name)
            if const:
                input_q = _quantize_const(
                    const,
                    f"{input_name}_q",
                    y_scale,
                    y_zero_point,
                    axis,
                    block_size,
                    output_dtype,
                    per_block_int_scale=per_block_int_scale,
                    base_dir=base_dir,
                )

        if input_q:
            nodes_to_add.append(
                opset.DequantizeLinear.make_node(
                    name=f"{node_name_prefix}_dq",
                    inputs=[
                        input_q.name,
                        f"{input_name}_scale",
                        f"{input_name}_zero_point",
                    ],
                    output=output_name,
                    dtype=output_dtype,
                    axis=axis,
                    block_size=block_size,
                )
            )
            tensors_to_remove[input_name] = True
            tensors_to_add.append(input_q)

        else:
            nodes_to_add.extend(
                [
                    opset.QuantizeLinear.make_node(
                        name=f"{node_name_prefix}_q",
                        inputs=[
                            input_name,
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=f"{input_name}_q",
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_dq",
                        inputs=[
                            f"{input_name}_q",
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=output_name,
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                ]
            )

    _finalize_graph_changes(
        model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove
    )


def _quantize_const(
    const: TensorProto,
    name: str,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    axis: Optional[int],
    block_size: Optional[int],
    output_dtype: str,
    per_block_int_scale: Optional[np.ndarray],
    base_dir: Optional[str] = None,
) -> TensorProto:
    const = to_array(const, base_dir=base_dir).astype(np.float32)
    unsigned, bitwidth = output_dtype.split("int")
    bitwidth = int(bitwidth)
    # Always quantize in float32
    y_scale = y_scale.astype(np.float32)

    if unsigned:
        clip_min = 0
        clip_max = 2**bitwidth - 1
    else:
        clip_min = -(2 ** (bitwidth - 1))
        clip_max = -clip_min - 1

    if per_block_int_scale is not None:
        block_axis = axis
        channel_axis = 0 if block_axis in (1, -1) else 1
        y_scale = y_scale.reshape(
            *(-1 if axis == channel_axis else 1 for axis in range(const.ndim))
        )
        y_scale = (y_scale * per_block_int_scale).astype(np.float32)

    y_scale = _broadcast(y_scale, const.ndim, axis=axis, block_size=block_size)
    y_zero_point = (
        _broadcast(y_zero_point, const.ndim, axis=axis, block_size=block_size)
        if y_zero_point is not None
        else np.zeros(y_scale.shape, dtype=np.int32)
    )

    y_scale = y_scale.astype(np.float32)
    const_q = (const / y_scale + y_zero_point).round()
    const_q = const_q.clip(clip_min, clip_max)
    return opset10.DequantizeLinear.make_int_arr(const_q, dtype=output_dtype, name=name)


def _dequantize_const(
    const_q: TensorProto,
    name: str,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    axis: Optional[int],
    block_size: Optional[int],
    output_dtype: str,
    per_block_int_scale: Optional[np.ndarray],
) -> TensorProto:
    if output_dtype == "bfloat16":
        raise RuntimeError("Unsupported data type: {}")

    const_q = to_array(const_q)
    # Always dequantize in float32
    y_scale = y_scale.astype(np.float32)

    if per_block_int_scale is not None:
        block_axis = axis
        channel_axis = 0 if block_axis in (1, -1) else 1
        y_scale = y_scale.reshape(
            *(-1 if axis == channel_axis else 1 for axis in range(const_q.ndim))
        )
        y_scale = (y_scale * per_block_int_scale).astype(np.float32)

    y_scale = _broadcast(y_scale, const_q.ndim, axis=axis, block_size=block_size)
    y_zero_point = (
        _broadcast(y_zero_point, const_q.ndim, axis=axis, block_size=block_size)
        if y_zero_point is not None
        else np.zeros(y_scale.shape, dtype=np.int32)
    )

    const_q = const_q.astype(np.int64)
    y_scale = y_scale.astype(np.float32)
    y_zero_point = y_zero_point.astype(np.int64)

    const_dq = (const_q - y_zero_point) * y_scale
    return from_array(const_dq.astype(output_dtype), name=name)


def _broadcast(
    x: np.ndarray, ndim: int, axis: Optional[int], block_size: Optional[int]
) -> np.ndarray:
    if axis is None:
        return x

    axis = (ndim + axis) % ndim  # Make positive
    if block_size is None:
        channel_axis = axis
        broadcast_shape = tuple(
            -1 if axis == channel_axis else 1 for axis in range(ndim)
        )
        x = x.reshape(broadcast_shape)
    else:
        block_axis = axis
        x = x.repeat(block_size, axis=block_axis)

    return x


def _finalize_graph_changes(
    model: ModelProto,
    nodes_to_add: Iterable,
    inputs_to_rename: Dict,
    tensors_to_add: List[TensorProto],
    tensors_to_remove: Dict,
):
    # Remove dangling tensors/nodes
    initializers = [
        init
        for init in model.graph.initializer
        if not tensors_to_remove.pop(init.name, None)
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)

    nodes = [
        node
        for node in model.graph.node
        if not (
            node.op_type == "Constant" and tensors_to_remove.pop(node.output[0], None)
        )
    ]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)

    # Redirect consumers that took the removed biases to take qdq bias instead
    # before:
    #     bias --------------------> consumer
    # after:
    #     bias_int32 --> DQ -------> consumer
    for node in model.graph.node:
        for i, old_name in enumerate(node.input):
            new_name = inputs_to_rename.get(old_name, None)
            if new_name is not None:
                node.input[i] = new_name

    # Add new tensors
    for t in tensors_to_add:
        model.graph.initializer.append(t)

    # Insert new nodes in a topologically order
    original_nodes = deque(list(model.graph.node))
    new_nodes = {node.input[0]: node for node in nodes_to_add if node.input}
    queue = deque([node for node in nodes_to_add if not node.input])

    queue.extend(
        [new_nodes.pop(inp.name) for inp in model.graph.input if inp.name in new_nodes]
    )
    queue.extend(
        [
            new_nodes.pop(init.name)
            for init in model.graph.initializer
            if init.name in new_nodes
        ]
    )

    if not queue and original_nodes:
        queue.append(original_nodes.popleft())

    model.graph.ClearField("node")

    while queue:
        node = queue.popleft()
        model.graph.node.append(node)

        qdq_nodes = [
            new_nodes.pop(output_name)
            for output_name in node.output
            if output_name in new_nodes
        ]
        if qdq_nodes:
            queue.extend(qdq_nodes)

        if not queue and original_nodes:
            queue.append(original_nodes.popleft())

    model.graph.node.extend(new_nodes.values())


class _ParamUtils:
    """Param utilities"""

    @staticmethod
    def get_shape(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[Sequence[int]]:
        """
        Returns a list of shape for the param specifies
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        param = _ParamUtils.get_param(model, node, param_index)
        if param:
            return param.dims
        return None

    @staticmethod
    def get_param(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[TensorProto]:
        """
        Returns the param tensor
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        if len(node.input) >= param_index + 1:
            param_name = node.input[param_index]
            return _ParamUtils.get_param_by_name(model, param_name)
        return None

    @staticmethod
    def get_param_by_name(model: ModelProto, param_name: str) -> Optional[TensorProto]:
        """
        Returns the param tensor

        :param model: ONNX model
        :param param_name: Name of parameter to retrieve
        """

        def find_param_in_model_initializers(param_name: str, model: ModelProto):
            for param in model.graph.initializer:
                if param.name == param_name:
                    return param
            return None

        def find_param_in_model_constants(param_name: str, model: ModelProto):
            for node in model.graph.node:
                if node.op_type == "Constant" and param_name in node.output:
                    for attribute in node.attribute:
                        if attribute.name == "value":
                            param = attribute.t
                            param.name = param_name
                            return param
                if node.op_type == "Identity" and param_name == node.output[0]:
                    return _ParamUtils.get_param(model, node, 0)
            return None

        param = find_param_in_model_initializers(param_name, model)
        if param is None:
            param = find_param_in_model_constants(param_name, model)
        return param


_all_op_schemas = {schema.name: schema for schema in onnx.defs.get_all_schemas()}


def _is_float_output(op_type: str) -> bool:
    """
    Returns True if op_type can return float output
    """
    schema = _all_op_schemas[op_type]
    output = schema.outputs[0]
    return _is_float(schema, output)


def _is_float(
    schema: onnx.defs.OpSchema, tensor_spec: onnx.defs.OpSchema.FormalParameter
) -> bool:
    type_str = tensor_spec.type_str
    try:
        type_constraint = next(
            type_constraint
            for type_constraint in schema.type_constraints
            if type_constraint.type_param_str == type_str
        )
    except StopIteration:
        type_constraint = None

    if type_constraint:
        allowed_type_strs = type_constraint.allowed_type_strs
    else:
        allowed_type_strs = [type_str]

    return any(
        t in ("tensor(float)", "tensor(double)", "tensor(float16)")
        for t in allowed_type_strs
    )


def _is_grid_preserving_op(op_type: str) -> bool:
    """
    Returns True if op_type can be considered a grid-preserving op.
    Data movement op is defined as a reshape or indexing operator
    whose output strictly preserves the quantization grid of the input

    Formally put,
    function `f` is a grid-preserving op if and only if y == y'
    where
        * x_q = quantize(x, scale_x, zp_x)
        * y  = f(x_q)
        * y' = requantize(y, scale_x, zp_x)
    """
    return op_type in (
        "BatchToSpace",
        "Col2Im",
        "Compress",
        "DepthToSpace",
        "Dropout",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "MaxPool",
        "MaxRoiPool",
        "NonZero",
        "Pad",
        "ReduceMax",
        "ReduceMin",
        "Reshape",
        "Slice",
        "SpaceToBatch",
        "SpaceToDepth",
        "Split",
        "SplitToSequence",
        "Squeeze",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
    )


def _is_htp_interpolation_op(op_type: str) -> bool:
    """
    Returns True if op_type can be considered an interpolation op in HTP.
    Although these operators aren't strictly data movement ops,
    HTP reuses the same quantization encoding for both input and output of
    the interpolation ops
    """
    # TODO: Absorb this function into redesigned config file
    return op_type in (
        "CropAndResize",
        "Resize",
        "ScatterElements",
        "Upsample",
    )


def _convert_version_with_external_weights(model, target_opset_version):
    """
    Upgrade opset version with weights flushed to disk temporarily
    """
    regular_tensors = {
        tensor.name
        for tensor in _get_all_tensors(model)
        if not uses_external_data(tensor)
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "model.onnx")

        # Temporarily switch regular (internal) tensors to external
        onnx.save_model(
            model,
            onnx_file,
            save_as_external_data=True,
            location="model.data",
        )
        external_tensors = {
            tensor.name: tensor.external_data[:]
            for tensor in _get_all_tensors(model)
            if uses_external_data(tensor)
        }

        model = onnx.version_converter.convert_version(model, target_opset_version)

        # Restore original state of the model
        for tensor in _get_all_tensors(model):
            if tensor.name in external_tensors:
                # Step 1. Restore external_data of all tensors
                # NOTE: This step is only necessary in onnx < 1.19,
                # where version converter strips away external_data information
                external_data = external_tensors[tensor.name]
                tensor.data_location = TensorProto.EXTERNAL
                del tensor.external_data[:]
                tensor.external_data.extend(external_data)

                # Step 2. Load raw_data of tensors that were original non-external
                if tensor.name in regular_tensors:
                    load_external_data_for_tensor(tensor, tmp_dir)
                    tensor.data_location = TensorProto.DEFAULT
                    del tensor.external_data[:]

    return model


def _convert_version(
    model: onnx.ModelProto, target_opset_version: int
) -> onnx.ModelProto:
    if any(uses_external_data(tensor) for tensor in _get_all_tensors(model)):
        model = _convert_version_with_external_weights(model, target_opset_version)
    else:
        try:
            model = onnx.version_converter.convert_version(model, target_opset_version)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # convert_version throws an exception on model > 2GB the observed exception was a
            # RuntimeError exception about ir_version, but possible other exceptions could be
            # triggered. leaving this very generic for now.
            logger.warning(
                "onnx.version_converter.convert_version failed with exception: %s. Retrying with external data",
                str(e),
            )
            model = _convert_version_with_external_weights(model, target_opset_version)

    logger.info("The opset of the onnx model is updated to %s.", target_opset_version)
    return model


def _remove_onnx_qdq_nodes(
    model: onnx.ModelProto,
) -> List[Dict[str, Union[str, int, np.ndarray]]]:
    initializers: Dict[str, TensorProto] = {
        init.name: init for init in model.graph.initializer
    }
    constants: Dict[str, TensorProto] = _get_all_constants(model)
    q_nodes: Dict[str, NodeProto] = {}
    dq_nodes: Dict[str, NodeProto] = {}
    producers: Dict[str, NodeProto] = {}
    consumers: Dict[str, Dict[str, NodeProto]] = defaultdict(dict)
    graph_outputs = set(output.name for output in model.graph.output)

    _validate_model(model, constants, consumers, producers)

    to_encoding = functools.partial(
        _to_encoding,
        model=model,
        constants=constants,
        consumers=consumers,
        producers=producers,
    )
    get_lpbq_nodes = functools.partial(
        _get_lpbq_nodes, producers=producers, consumers=consumers, constants=constants
    )
    get_qdq_nodes = functools.partial(
        _get_qdq_nodes, producers=producers, consumers=consumers, constants=constants
    )

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            q_nodes[node.name] = node
        elif node.op_type == "DequantizeLinear":
            dq_nodes[node.name] = node

        for inp in node.input:
            consumers[inp].update({node.name: node})
        for out in node.output:
            producers[out] = node

    to_be_removed = {}
    for dq in dq_nodes.values():
        lpbq_nodes = get_lpbq_nodes(dq)
        if lpbq_nodes:
            to_be_removed.update({node.name: node for node in lpbq_nodes})
            continue

        qdq_nodes = get_qdq_nodes(dq)
        if qdq_nodes:
            to_be_removed.update({node.name: node for node in qdq_nodes})

    encodings = {
        dq.name: to_encoding(dq)
        for dq in to_be_removed.values()
        if dq.op_type == "DequantizeLinear"
    }
    encodings = {
        node_name: encoding
        for node_name, encoding in encodings.items()
        if encoding is not None
    }

    # Reconnect nodes
    for dq in model.graph.node:
        if dq.op_type != "DequantizeLinear":
            continue

        producer = producers.get(dq.input[0])
        if producer and producer.op_type == "QuantizeLinear":
            q = producer
            producer = producers.get(q.input[0])
        else:
            q = None

        if dq.output[0] in graph_outputs and not producer:
            # Edge case: This means the model was in form of:
            #   (model_input) --> Q -> DQ -> (model_output)
            # or:
            #   (constant) -----> Q -> DQ -> (model_output)
            #
            # We can't preserve the I/O names in this case
            raise RuntimeError(
                f"Node {q.name} (op_type: {q.op_type}) can't be removed because "
                "it's the only connection between the model's input and output."
            )

        if not q:
            # Standalone DQs can be removed if it only takes static inputs.
            #
            # Before:
            #   (constant) -----> Q -> DQ ----> (consumers or model_output)
            #                               ↑
            #                           dq.output[0]
            #                           (=new_name)
            # After:
            #   (constant) -------------------> (consumers or model_output)
            #                               ↑
            #                           dq.output[0]
            #                           (=new_name)
            const = constants.get(dq.input[0])

            if const and const.name not in initializers:
                const_node = producers[const.name]
                to_be_removed[const_node.name] = const_node

            if const and dq.name in encodings:
                new_name = dq.output[0]
                e = encodings[dq.name]
                initializers[dq.output[0]] = _dequantize_const(
                    const,
                    name=new_name,
                    y_scale=e.get("y_scale", e.get("per_channel_float_scale")),
                    y_zero_point=e.get("y_zero_point"),
                    axis=e.get("axis"),
                    block_size=e.get("block_size"),
                    output_dtype="float32",
                    per_block_int_scale=e.get("per_block_int_scale"),
                )

            continue

        if dq.output[0] in graph_outputs:
            # DQ output is part of graph outputs.
            # We should preserve DQ's output name to preserve the graph output name
            #
            # Before:
            #                             +--> consumers
            #   producer -----> Q -> DQ --+--> (model_output)
            #                             ↑
            #                         dq.output[0]
            #                         (=new_name)
            # After:
            #                             +--> consumers
            #   producer -----------------+--> (model_output)
            #                             ↑
            #                         dq.output[0]
            #                         (=new_name)
            new_name = dq.output[0]
        else:
            # Before:
            #   producer -----> Q -> DQ -----> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            # After:
            #   producer --------------------> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            new_name = q.input[0]

        for consumer in consumers[dq.output[0]].values():
            for i, inp in enumerate(consumer.input):
                if inp == dq.output[0]:
                    consumer.input[i] = new_name
        if producer:
            for i, out in enumerate(producer.output):
                if out == q.input[0]:
                    producer.output[i] = new_name

    included_nodes = set()
    node = []
    # Nodes may appear in producer.values() multiple times, only use the first appearance
    for producer in producers.values():
        if producer.name not in to_be_removed and producer.name not in included_nodes:
            included_nodes.add(producer.name)
            node.append(producer)

    model.graph.ClearField("node")
    model.graph.node.extend(node)

    model.graph.ClearField("initializer")
    model.graph.initializer.extend(list(initializers.values()))
    from onnxruntime.quantization.onnx_quantizer import ONNXModel

    ONNXModel(model).remove_unused_constant()

    # Convert removed Q/DQ nodes to encoding
    return list(encodings.values())


def _validate_model(
    model: ModelProto,
    constants: Dict[str, TensorProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    producers: Dict[str, NodeProto],
) -> None:
    invalid_nodes = []

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            is_qdq = _is_q_dq_sequence(node, consumers)
        elif node.op_type == "DequantizeLinear":
            is_qdq = _get_qdq_nodes(
                node, producers, consumers, constants
            ) or _is_lpbq_subgraph(node, producers, consumers, constants)
        else:
            continue

        if not is_qdq:
            invalid_nodes.append(node)

    if not invalid_nodes:
        return

    invalid_node_names = ", ".join([node.name for node in invalid_nodes])
    raise RuntimeError(
        f"Invalid QuantizeLinear/DequantizeLinear detected: {invalid_node_names}.\n\n"
        "To import onnx QDQ model, please ensure the following requirements:"
        "  - All QuantizeLinear (if any) must be followed by DequantizeLinear.\n"
        "  - All DequantizeLinaer must be 1) preceded by QuantizeLinear or 2) take static constant as input"
    )


def _to_encoding(
    dq: NodeProto,
    model: ModelProto,
    constants: Dict[str, TensorProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    producers: Dict[str, NodeProto],
) -> Optional[Dict[str, Union[str, int, np.ndarray]]]:
    q = producers.get(dq.input[0])

    if q and q.op_type != "QuantizeLinear":
        raise RuntimeError(
            f"DequantizeLinear can be only preceded by QuantizeLinear. "
            f"Got {q.op_type} (name: {q.name})"
        )

    for consumer in consumers[dq.output[0]].values():
        if consumer.op_type == "DequantizeLinear":
            lpbq = _get_lpbq_nodes(consumer, producers, consumers, constants)
            if not lpbq:
                raise RuntimeError(
                    f"Back-to-back DequantizeLinear detected at {dq.name}. "
                    "Back-to-back DequantizeLinear is only supported in LPBQ"
                )
            return None

    if any(dq.output[0] == graph_out.name for graph_out in model.graph.output):
        input_name = dq.output[0]
    else:
        input_name = q.input[0] if q else dq.output[0]

    lpbq = _get_lpbq_nodes(dq, producers, consumers, constants)
    if lpbq:
        *_, scale_dq = lpbq
        scale = {
            "per_block_int_scale": to_array(constants[scale_dq.input[0]]),
            "per_channel_float_scale": to_array(constants[scale_dq.input[1]]),
        }
    else:
        scale = {"y_scale": to_array(constants[dq.input[1]])}

    if len(dq.input) > 2:
        zp_name = dq.input[2]
        zp_tensor_proto = constants[zp_name]

        if zp_tensor_proto.data_type not in (
            TensorProto.INT4,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.UINT4,
            TensorProto.UINT8,
            TensorProto.UINT16,
        ):
            raise RuntimeError(
                f'Found zero_point "{zp_name}" with unsupported dtype '
                f"{onnx.helper.tensor_dtype_to_string(zp_tensor_proto.data_type)}. "
                "Only [u]int4, [u]int8, [u]int16, and int32 are supported."
            )

        zp = to_array(zp_tensor_proto)
        output_dtype = zp_tensor_proto.data_type
    else:
        zp = None
        try:
            output_dtype = (
                next(attr.i for attr in q.attribute if attr.name == "output_dtype")
                if q
                else TensorProto.UINT8
            )
        except StopIteration:
            # ONNX assumes uint8 if neither zero_point nor output_dtype is specified
            output_dtype = TensorProto.UINT8

    *_, output_dtype = (
        onnx.helper.tensor_dtype_to_string(output_dtype).lower().split(".")
    )

    encoding = {
        "name": input_name,
        "output_dtype": output_dtype,
        **scale,
    }

    if zp is not None:
        encoding["y_zero_point"] = zp

    for attr in dq.attribute:
        if attr.name == "axis":
            encoding["axis"] = attr.i
        elif attr.name == "block_size":
            encoding["block_size"] = attr.i
        elif attr.name == "output_dtype":
            output_dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
            if output_dtype != encoding["output_dtype"]:
                raise RuntimeError(
                    f"Attribute output_dtype={output_dtype} of node {dq.name} "
                    "is inconsistent with "
                    f"the dtype of zero_point {encoding['output_dtype']} "
                )

    return encoding


def _get_all_constants(model: ModelProto) -> Dict[str, TensorProto]:
    constants = {
        **{init.name: init for init in model.graph.initializer},
        **{
            const.output[0]: attr.t
            for const in model.graph.node
            for attr in const.attribute
            if const.op_type == "Constant" and attr.name == "value"
        },
    }
    identities = [
        identity for identity in model.graph.node if identity.op_type == "Identity"
    ]

    while True:
        aliases = {
            identity.output[0]: constants[identity.input[0]]
            for identity in identities
            if identity.input[0] in constants
        }
        if aliases:
            constants.update(aliases)
            identities = [
                identity
                for identity in identities
                if identity.output[0] not in constants
            ]
        else:
            break

    return constants


def _get_qdq_nodes(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> List[NodeProto]:
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_get_qdq_nodes can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    qdq_nodes = []
    q = producers.get(dq.input[0])

    if not q:
        if set(dq.input) <= constants.keys():
            # Standalone DQ with static inputs
            qdq_nodes.append(dq)
    elif (
        _is_q_dq_sequence(q, consumers)
        # Scale and zp must be constants.
        and set(q.input[1:]) <= constants.keys()
    ):
        qdq_nodes.extend([q, *consumers[q.output[0]].values()])

    return qdq_nodes


def _get_lpbq_nodes(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> List[NodeProto]:
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_get_lpbq_nodes can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    scale_dq = producers.get(dq.input[1])

    if not scale_dq:
        return []

    is_lpbq = (
        set(scale_dq.input) <= constants.keys()
        and set(dq.input[2:]) <= constants.keys()
    )

    q = producers.get(dq.input[0])
    if q:
        is_lpbq &= (
            _is_q_dq_sequence(q, consumers)
            # Input of Q must be constant
            and q.input[0] in constants
        )

    if is_lpbq:
        return [q, dq, scale_dq] if q else [dq, scale_dq]

    return []


def _is_q_dq_sequence(q: NodeProto, consumers: Dict[str, Dict[str, NodeProto]]):
    return (
        q.op_type == "QuantizeLinear"
        and all(
            # All Q must be followed by DQ
            consumer.op_type == "DequantizeLinear"
            # Q-DQ must share same scale and zp
            and consumer.input[1:] == q.input[1:]
            # This rules out LPBQ which takes runtime-computed scale as input
            for consumer in consumers[q.output[0]].values()
        )
    )


def _is_lpbq_subgraph(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> bool:
    """
    per_channel_float_scale -----+
                                 +--> DequantizeLinear -----+ (blockwise scale)
       per_block_uint_scale -----+        (1st DQ)          |
                                                    +-------+---------+
                                                    V                 V
                                 weight ---> QuantizeLinear -> DequantizeLinear -> ...
                                                                   (2nd DQ)
    """
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_is_lpbq_subgraph can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    is_2nd_dq = bool(_get_lpbq_nodes(dq, producers, consumers, constants))
    is_1st_dq = all(
        consumer.op_type == "DequantizeLinear"
        and _get_lpbq_nodes(consumer, producers, consumers, constants)
        for consumer in consumers[dq.output[0]].values()
    )
    return is_1st_dq or is_2nd_dq


def _get_node_attribute(node: NodeProto, name: str):
    """
    Return the value of a node's attribute specified by its name

    :param node: NodeProto object to retrieve the attribute from
    :param name: string containing the name of the attribute to retrieve
    :return: value of the attribute
    """
    for item in node.attribute:
        if item.name == name:
            return onnx.helper.get_attribute_value(item)
    return None


def contains_tensor_type(model: ModelProto, tensor_type: int | List[int]):
    """
    Returns True if the model contains the specified tensor type(s).
    """
    if isinstance(tensor_type, int):
        tensor_type = [tensor_type]
    if any(
        tensor.type.tensor_type.elem_type in tensor_type
        for tensor in itertools.chain(model.graph.input, model.graph.output)
    ):
        return True

    if any(tensor.data_type in tensor_type for tensor in model.graph.initializer):
        return True

    for node in model.graph.node:
        if node.op_type == "Cast":
            cast_type = _get_node_attribute(node, "to")
            if cast_type in tensor_type:
                return True

        if node.op_type in ("Constant", "ConstantOfShape"):
            value = _get_node_attribute(node, "value")
            if value is None:
                continue

            if value.data_type in tensor_type:
                return True

    return False


def _derive_data_movement_op_encodings(
    model: onnx.ModelProto,
    encodings: Mapping[str, Mapping],
) -> Dict[str, Dict]:
    data_movement_ops = [
        node for node in model.graph.node if _is_grid_preserving_op(node.op_type)
    ]

    new_encodings = {}
    consumers: Mapping[str, List[onnx.NodeProto]] = defaultdict(list)

    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    def derive_encoding(node: onnx.NodeProto):
        derived_encodings = {}
        input_names = node.input[:] if node.op_type == "Concat" else [node.input[0]]
        output_names = node.output[:] if node.op_type == "Split" else [node.output[0]]

        can_propagate_forward = len(input_names) == 1
        can_propagate_backward = len(output_names) == 1

        for input_name, output_name in itertools.product(input_names, output_names):
            inp_encoding = encodings.get(input_name)
            out_encoding = encodings.get(output_name)

            if inp_encoding and out_encoding:
                # Both input and output encoding already exists; skip
                continue

            # Only per-tensor encodings can be safely propagated through data movement ops
            # because some data movement ops such as Reshape and Transpose can't reuse
            # the same channel/block axes across inputs and outputs
            if (
                out_encoding
                and out_encoding.get("axis") is None
                and can_propagate_backward
            ):
                if len(consumers[input_name]) > 1 or len(node.output) > 1:
                    # If input has more than one consumer or if there are more than one output,
                    # it is NOT safe to reuse output encoding for input quantization
                    continue
                else:
                    # Reuse output encoding for input quantization
                    derived_encodings.update({input_name: out_encoding.copy()})
                    continue

            # Only per-tensor encodings can be safely propagated through data movement ops
            # because some data movement ops such as Reshape and Transpose can't reuse
            # the same channel/block axes across inputs and outputs
            if (
                inp_encoding
                and inp_encoding.get("axis") is None
                and can_propagate_forward
            ):
                # Reuse input encoding for output quantization
                derived_encodings.update({output_name: inp_encoding.copy()})
                continue

        return derived_encodings

    for node in data_movement_ops:
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    # Repeat in reverse-DFS order
    for node in reversed(data_movement_ops):
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    return {key: enc for key, enc in new_encodings.items()}
