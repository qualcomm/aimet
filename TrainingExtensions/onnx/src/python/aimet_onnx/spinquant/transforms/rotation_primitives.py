# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Hadamard-rotation primitives shared across SpinQuant rotation passes.

These helpers operate on an arbitrary normalized rotation matrix ``R`` and an
op's storage metadata. They contain no knowledge of R1 / R2 / R3 specifics;
each rotation pass selects the matrix and decides which ops to call them on.

Storage / role conventions:

    Role          | Storage         | Formula
    --------------|-----------------|-------------------
    reading layer | [out, in]       | W @ R   (axis 1)
    reading layer | [in,  out]      | R^T @ W (axis 0)
    reading Conv  | [out, in, *k]   | W @ R   (axis 1)
    writing layer | [out, in]       | R^T @ W (axis 0)
    writing layer | [in,  out]      | W @ R   (axis -1)
    writing Conv  | [out, in, *k]   | R^T @ W (axis 0)
    Gather        | [vocab, hidden] | W @ R   (axis -1)
"""

from typing import List, Tuple
import re

import numpy as np
import onnx_ir

from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import set_static_tensor, static_tensor

from aimet_onnx.llm_topology.ir_analysis import (
    get_bias_value,
    get_weight_value,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def hadamard_rotation_matrix(hidden_size: int) -> np.ndarray:
    """Return ``H / sqrt(hidden_size)`` as a float64 normalized Hadamard rotation."""
    return (get_hadamard_matrix(hidden_size) / np.sqrt(hidden_size)).astype(np.float64)


def block_diag_repeat(R: np.ndarray, k: int) -> np.ndarray:
    """Return a ``[k*d, k*d]`` block-diagonal matrix with ``R`` repeated ``k`` times.

    Lets the existing whole-axis rotation helpers express per-head rotations:
    rotating an op's output channels per-head with ``R`` is identical to
    rotating the full axis with ``block_diag(R, R, ..., R)``.

    :param R: A ``[d, d]`` square matrix.
    :param k: Number of diagonal copies (e.g. number of attention heads).
    :return: A ``[k*d, k*d]`` block-diagonal matrix.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"block_diag_repeat: R must be square, got shape {R.shape}.")
    d = R.shape[0]
    out = np.zeros((k * d, k * d), dtype=R.dtype)
    for i in range(k):
        out[i * d : (i + 1) * d, i * d : (i + 1) * d] = R
    return out


#: Input index of a ``Gather``'s data input — the ``[vocab, hidden]`` embedding table.
_GATHER_TABLE_INDEX = 0


def rotate_gather_weight(node: onnx_ir.Node, R: np.ndarray):
    """Apply a right-side rotation ``W @ R`` to the table of a Gather (embed_tokens).

    :param node: The Gather node.
    :param R: Normalized rotation matrix [hidden, hidden].
    """
    table = (
        node.inputs[_GATHER_TABLE_INDEX]
        if len(node.inputs) > _GATHER_TABLE_INDEX
        else None
    )
    W = static_tensor(table)
    if W is None:
        raise RuntimeError(
            f"embed_tokens node '{node.name}': input {_GATHER_TABLE_INDEX} is not a "
            f"static embedding table."
        )

    W = W.numpy()
    set_static_tensor(table, right_multiply(W, R).astype(W.dtype))
    _logger.debug("Rotated embed_tokens '%s' shape %s.", table.name, W.shape)


def rotate_linear_weight(node: onnx_ir.Node, R: np.ndarray, is_writing: bool):
    """Apply a rotation to the weight (and bias if writing) of a MatMul/Gemm/Conv node.

    :param node: The MatMul, Gemm, or Conv node.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param is_writing: True for layers that write to the residual stream
        (e.g. o_proj, down_proj); False for layers that read from it
        (e.g. qkv, gate_up, lm_head).
    """
    weight_value, is_transposed = get_weight_value(node)
    weight = static_tensor(weight_value)
    if weight is None:
        raise RuntimeError(f"Node '{node.name}': no static weight found.")

    W = weight.numpy()
    W_new = apply_transform(W, R, node.op_type, is_transposed, is_writing)
    set_static_tensor(weight_value, W_new.astype(W.dtype))
    _logger.debug(
        "Rotated node '%s' (%s, transposed=%s, is_writing=%s) shape %s.",
        node.name,
        node.op_type,
        is_transposed,
        is_writing,
        W.shape,
    )

    if is_writing:
        bias_value = get_bias_value(node)
        bias = static_tensor(bias_value)
        if bias is not None:
            b = bias.numpy()
            set_static_tensor(bias_value, right_multiply(b, R, axis=-1).astype(b.dtype))
            _logger.debug(
                "Rotated bias for writing node '%s' shape %s.", node.name, b.shape
            )


def apply_transform(
    W: np.ndarray, R: np.ndarray, op_type: str, is_transposed: bool, is_writing: bool
) -> np.ndarray:
    """Dispatch to ``right_multiply`` or ``left_multiply`` based on storage and role.

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param op_type: ONNX op type.
    :param is_transposed: True if W is stored as [out, in]; False if [in, out].
    :param is_writing: True for writing layers; False for reading layers.
    :return: Rotated weight.
    """
    if op_type == "Conv":
        # [out, in, *k]
        return (
            left_multiply(W, R, axis=0) if is_writing else right_multiply(W, R, axis=1)
        )
    if is_transposed:
        # [out, in]
        return (
            left_multiply(W, R, axis=0) if is_writing else right_multiply(W, R, axis=1)
        )
    # [in, out]
    return right_multiply(W, R, axis=-1) if is_writing else left_multiply(W, R, axis=0)


def right_multiply(W: np.ndarray, R: np.ndarray, axis: int = -1) -> np.ndarray:
    """``W_new = W @ R`` along ``axis``.

    Corresponds to ``left_hand_transform(R^T)`` baked into reading layers
    ([out, in] storage), and ``right_hand_transform(R)`` baked into writing
    layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, -1)  # [*rest, hidden]
    moved_shape = W_moved.shape
    W_new = W_moved.reshape(-1, W_moved.shape[-1]) @ R  # [N, hidden]
    return np.moveaxis(W_new.reshape(moved_shape), -1, axis)


def left_multiply(W: np.ndarray, R: np.ndarray, axis: int = 0) -> np.ndarray:
    """``W_new = R^T @ W`` along ``axis``.

    Corresponds to ``right_hand_transform(R)`` baked into writing layers
    ([out, in] storage), and ``left_hand_transform(R^T)`` baked into reading
    layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, 0)  # [hidden, *rest]
    moved_shape = W_moved.shape
    W_new = R.T @ W_moved.reshape(W_moved.shape[0], -1)  # [hidden, N]
    return np.moveaxis(W_new.reshape(moved_shape), 0, axis)


def insert_online_hadamard_node(
    ir_model: onnx_ir.Model,
    target_value: onnx_ir.Value,
    consumer_nodes: List[onnx_ir.Node],
    H: np.ndarray,
    name_prefix: str,
) -> Tuple[onnx_ir.Value, onnx_ir.Node]:
    """Insert ``MatMul(target_value, H)`` between a producer and chosen consumers.

    Used by R1 and R3 to add an online Hadamard rotation immediately upstream of
    a chosen consumer (the QK^T MatMul, the past-key Concat, or the residual
    stream). The new MatMul reads ``target_value`` and writes a rotated tensor;
    only the listed consumers are rewired, so any other consumer of the original
    tensor keeps seeing the unrotated value.

    :param ir_model: IR model to mutate.
    :param target_value: The tensor to rotate.
    :param consumer_nodes: Nodes whose ``target_value`` input gets rewired.
    :param H: Hadamard rotation matrix to insert.
    :param name_prefix: Prefix used to name the inserted initializer / node /
        output tensor (e.g. ``"block0_q"``).
    :return: ``(rotated_value, new_node)`` — the new (rotated) tensor and the
        inserted ``MatMul`` node, so callers can wire a quantizer relative to it.
    """
    for consumer_node in consumer_nodes:
        if target_value not in consumer_node.inputs:
            raise ValueError(
                f"insert_online_hadamard_node: target tensor "
                f"'{target_value.name}' does not appear in the inputs of consumer "
                f"node '{consumer_node.name}': "
                f"{[inp.name if inp else None for inp in consumer_node.inputs]}"
            )

    dtype = _rotation_dtype(target_value)
    H = H.astype(dtype.numpy())

    initializer_name = f"{name_prefix}_hadamard"
    hadamard = onnx_ir.Value(
        name=initializer_name,
        type=onnx_ir.TensorType(dtype),
        shape=onnx_ir.Shape(H.shape),
        const_value=onnx_ir.tensor(H, name=initializer_name),
    )
    ir_model.graph.register_initializer(hadamard)

    new_node = onnx_ir.node(
        "MatMul",
        inputs=[target_value, hadamard],
        num_outputs=1,
        name=name_prefix,
    )
    rotated_value = new_node.outputs[0]
    rotated_value.name = f"{name_prefix}_out"
    rotated_value.type = onnx_ir.TensorType(dtype)
    rotated_value.shape = target_value.shape

    _insert_after_producer(ir_model.graph, target_value, new_node)

    for consumer_node in consumer_nodes:
        for index, inp in enumerate(consumer_node.inputs):
            if inp is target_value:
                consumer_node.replace_input_with(index, rotated_value)
                _logger.debug(
                    "Inserted online Hadamard MatMul '%s' on tensor '%s' "
                    "(dimension=%d, dtype=%s); rewired '%s'.input[%d].",
                    name_prefix,
                    target_value.name,
                    H.shape[0],
                    dtype,
                    consumer_node.name,
                    index,
                )

    return rotated_value, new_node


def _rotation_dtype(target_value: onnx_ir.Value) -> onnx_ir.DataType:
    """Return the dtype to cast the Hadamard to so it matches ``target_value``.

    Falls back to ``FLOAT`` when the graph carries no type for the tensor — ORT
    then raises a clear dtype error at session-build time if that guess is wrong,
    which is preferable to silently casting weights.
    """
    if target_value.dtype is not None:
        return target_value.dtype
    _logger.debug(
        "Tensor '%s' carries no dtype; assuming FLOAT for the online Hadamard.",
        target_value.name,
    )
    return onnx_ir.DataType.FLOAT


def _insert_after_producer(
    graph: onnx_ir.Graph, target_value: onnx_ir.Value, new_node: onnx_ir.Node
) -> None:
    """Insert ``new_node`` immediately after the node producing ``target_value``.

    ONNX requires nodes to appear in a topologically valid order. ORT tolerates
    out-of-order nodes for many graphs, but other tools (and serializers that
    re-validate) do not. When ``target_value`` has no producer — it is a graph
    input or an initializer — the new node belongs at the front of the graph.
    """
    producer = target_value.producer()
    if producer is not None:
        graph.insert_after(producer, new_node)
        return
    first_node = next(iter(graph), None)
    if first_node is None:
        graph.append(new_node)
    else:
        graph.insert_before(first_node, new_node)


# SpinQuant inserts online Hadamard rotations as ``MatMul`` nodes whose names
# are the ``spinquant_`` prefix plus a per-pass ``_R1`` / ``_R3`` suffix (see
# the ``name_prefix`` values passed to ``insert_online_hadamard_node`` in the R1
# and R3 passes). This regex is the single source of truth for recognizing those
# ops downstream; keep it in sync with the names produced there.
_ONLINE_ROTATION_OP_RE = re.compile(r"spinquant_.+_R[13]$")


def is_online_rotation_op(op) -> bool:
    """Return True if ``op`` is a SpinQuant online Hadamard rotation MatMul.

    R1 / R3 online rotations carry a fixed orthonormal Hadamard as their "weight"
    rather than a learnable parameter, so downstream optimizers (e.g.
    sequential MSE) should skip them.

    :param op: Anything that names an op: an ``onnx_ir.Node``, an ``onnx.NodeProto``
        (both spell the type ``op_type``) or a ConnectedGraph ``Op`` (which spells
        it ``type``). SpinQuant itself no longer builds a ConnectedGraph, but
        ``sequential_mse`` still calls this with one.
    """
    op_type = getattr(op, "op_type", None) or getattr(op, "type", None)
    return op_type == "MatMul" and bool(_ONLINE_ROTATION_OP_RE.match(op.name))
