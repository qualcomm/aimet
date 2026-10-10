# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Anchor detection for SpinQuant R3 rotation, pinned by ``past_key_*`` inputs.

Per decoder block, R3 needs two insertion anchors:

* the K-side: the current-K input of the ``Concat`` that fuses ``past_key_*``
  with the current-K tensor (so K entering the cache is rotated);
* the Q-side: the Q input of the QK^T MatMul (so the rotations algebraically
  cancel inside ``Q @ K^T``).

We pin the search on the ``past_key_*`` graph inputs (HF/optimum LLM exports
expose one per decoder block, in declaration order). For each one:

1. Find the ``Concat`` nodes consuming it. The OTHER input of that Concat
   is the current-K tensor (R3 K-side anchor).
2. Walk forward from the Concat output through pass-through ops
   (quantsim's grid-preserving data-movement ops plus ``Cast``; see
   :func:`_is_passthrough`) until reaching ``MatMul`` nodes — that's QK^T.
   Whichever MatMul input traces back to the Concat is K; the OTHER input is Q
   (R3 Q-side anchor).

R3 in this iteration requires KV-cache-style exports: the model must expose
one ``past_key_*`` graph input per decoder block. Prefill-only exports
without KV-cache inputs are not supported.

The whole search runs on ``onnx_ir``, where a ``Value`` already knows its
producer and its consumers — so there are no name-keyed producer/consumer index
tables to build, and none to keep in sync with the insertions R3 goes on to make.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import onnx_ir

from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import static_tensor

from aimet_onnx.llm_topology.ir_adapter import (
    IrLlmTopology,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def _is_passthrough(op_type: str, domain: str = "") -> bool:
    """Return True for ops the R3 anchor search treats as transparent.

    These are quantsim's grid-preserving (data-movement) ops — ``Transpose``,
    ``Reshape``, ``Identity``, ``Unsqueeze``, ``Expand``, etc. — plus ``Cast``.
    ``Unsqueeze`` / ``Expand`` are the ops GQA's ``repeat_kv`` inserts to
    broadcast each KV head across its query-head group; they form a clean
    linear chain (``Unsqueeze -> Expand -> Reshape``) between the past-key
    Concat and QK^T, so walking through them reaches the same QK^T MatMul as
    the MHA case. R3 rotates the per-head ``head_dim`` axis, which is identical
    for the KV heads before and after this broadcast, so the rotation is
    unaffected.

    ``Cast`` is not grid-preserving in a quantized graph (it changes dtype), so
    it is absent from :func:`_is_grid_preserving_op`. But R3 runs on the float
    graph before quantization, and HF/optimum exports insert dtype Casts on the
    Q/K paths that must be walked through — hence the explicit addition here.
    """
    return op_type == "Cast" or _is_grid_preserving_op(op_type, domain)


def _is_static_scalar(value: Optional[onnx_ir.Value]) -> bool:
    """Return True if ``value`` is a constant holding exactly one element."""
    tensor = static_tensor(value)
    return tensor is not None and tensor.size == 1


def _is_constant_rescale(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` rescales its input by a constant scalar."""
    if node.op_type == "Mul":
        return any(_is_static_scalar(operand) for operand in node.inputs)
    if node.op_type == "Div":
        return len(node.inputs) > 1 and _is_static_scalar(node.inputs[1])
    return False


@dataclass
class BlockR3Anchors:
    """Per-block anchors for inserting R3 online Hadamards.

    Members are ``onnx_ir`` objects, so the rotation pass can splice a node onto
    an edge and rewire ``node.inputs[idx]`` directly. Unlike the ConnectedGraph
    these were originally derived from, an IR graph stays valid — and these
    anchors stay live — across the insertions R3 makes.

    :param past_key_input_name: The ``past_key_*`` graph input that pinned this
        block's anchor search.
    :param k_input_value: The post-RoPE current-K tensor that R3 rotates. R3
        splices the Hadamard on this edge.
    :param k_consumers: Nodes consuming ``k_input_value``.
    :param qk_matmul_nodes: The QK^T attention MatMuls reached forward from the
        past-key Concat. R3 rewires the Q-side input of each.
    :param q_input_indices: For each MatMul in ``qk_matmul_nodes``, the index of
        its post-RoPE Q input (the input that does NOT trace back to the
        Concat).
    :param q_input_values: For each MatMul in ``qk_matmul_nodes``, the tensor at
        ``qk_matmul_node.inputs[q_input_indices[i]]``.
    """

    past_key_input_name: str
    k_input_value: onnx_ir.Value
    k_consumers: List[onnx_ir.Node]
    qk_matmul_nodes: List[onnx_ir.Node]
    q_input_indices: List[int]
    q_input_values: List[onnx_ir.Value]


def find_r3_anchors(
    role_map: IrLlmTopology, ir_model: onnx_ir.Model
) -> List[BlockR3Anchors]:
    """Return per-block R3 anchors, pinned by ``past_key_*`` graph inputs.

    The number of past_key inputs in the model must equal the number of
    decoder blocks in ``role_map``. The two are paired by graph-input
    declaration order, which matches HF/optimum export conventions.

    :param role_map: Backbone topology, resolved onto ``ir_model``.
    :param ir_model: The IR model R3 will mutate.
    :raises ValueError: If past_key input count does not match block count;
        if any past_key input is not consumed by exactly one Concat; if the
        forward walk from a Concat to its QK^T MatMul is ambiguous; or if
        the same MatMul ends up matched by two different blocks.
    """
    past_key_input_names = role_map.past_key_input_names
    n_blocks = len(role_map.blocks)
    if len(past_key_input_names) != n_blocks:
        raise ValueError(
            f"R3 rotation: expected {n_blocks} past_key_* graph inputs "
            f"(one per decoder block), found {len(past_key_input_names)}: "
            f"{past_key_input_names}. R3 requires a KV-cache-style export."
        )

    past_key_values = _resolve_graph_inputs(ir_model, past_key_input_names)

    seen_matmuls: Set[onnx_ir.Node] = set()
    result: List[BlockR3Anchors] = []
    for block_idx, past_key_name in enumerate(past_key_input_names):
        _logger.debug(
            "R3 anchors: processing block %d (past_key='%s').",
            block_idx,
            past_key_name,
        )
        past_key_value = past_key_values[past_key_name]
        # Find the Concats that combine past_key_in into the present key
        # (one per KV head in SHA).
        key_concats = _find_concat_consumers(past_key_value)
        for concat_node in key_concats:
            current_key_value = _find_current_k_input_of_concat(
                concat_node, past_key_value
            )

            # K transpose can occur before or after concat, R3 logic assumes rotation before Transpose
            producer = current_key_value.producer()
            if producer is not None and producer.op_type == "Transpose":
                current_key_value = producer.inputs[0]

            current_key_consumers = list(current_key_value.consumers())
            if not current_key_consumers:
                raise ValueError(
                    f"R3 rotation: current-K tensor '{current_key_value.name}' has "
                    f"no consumers."
                )

            qk_matmul_nodes = _walk_forward_to_matmuls(concat_node)
            q_input_indices = []
            q_input_values = []
            for node in qk_matmul_nodes:
                if node in seen_matmuls:
                    raise ValueError(
                        f"R3 rotation: past_key input '{past_key_name}': QK^T MatMul "
                        f"'{node.name}' was already matched by an earlier "
                        f"block. past_key_* graph inputs may be misordered."
                    )
                seen_matmuls.add(node)

                q_input_idx, q_input_value = _find_q_input_of_qk_matmul(
                    node, concat_node
                )
                q_input_indices.append(q_input_idx)
                q_input_values.append(q_input_value)

            result.append(
                BlockR3Anchors(
                    past_key_input_name=past_key_name,
                    k_input_value=current_key_value,
                    k_consumers=current_key_consumers,
                    qk_matmul_nodes=qk_matmul_nodes,
                    q_input_indices=q_input_indices,
                    q_input_values=q_input_values,
                )
            )

    _logger.debug("R3 anchors resolved for %d block(s).", len(result))
    return result


def _resolve_graph_inputs(
    ir_model: onnx_ir.Model, input_names: List[str]
) -> Dict[str, onnx_ir.Value]:
    """Return ``{name: graph input Value}`` for ``input_names``.

    :raises ValueError: If a name is not a graph input of ``ir_model`` — the
        topology and the IR model were built from different graphs.
    """
    by_name = {value.name: value for value in ir_model.graph.inputs if value.name}
    resolved = {}
    for name in input_names:
        value = by_name.get(name)
        if value is None:
            raise ValueError(
                f"R3 rotation: '{name}' is not a graph input of the supplied IR "
                f"model. The topology and the IR model were built from different "
                f"graphs."
            )
        resolved[name] = value
    return resolved


def _find_concat_consumers(past_key_value: onnx_ir.Value) -> List[onnx_ir.Node]:
    """Return the Concats downstream of ``past_key_value`` (one per KV head).

    Walk through pass-through ops (see :func:`_is_passthrough`) that may sit
    between the graph input and the downstream Concats.
    """
    concat_nodes: List[onnx_ir.Node] = []
    visited: Set[onnx_ir.Value] = set()
    queue: deque = deque([past_key_value])
    while queue:
        value = queue.popleft()
        if value in visited:
            continue
        visited.add(value)
        for consumer in value.consumers():
            if consumer.op_type == "Concat":
                concat_nodes.append(consumer)
            elif _is_passthrough(consumer.op_type, consumer.domain):
                queue.extend(consumer.outputs)
    if not concat_nodes:
        consumer_summary = [
            (node.name, node.op_type) for node in past_key_value.consumers()
        ]
        raise ValueError(
            f"R3 rotation: no downstream Concat reachable from past_key input "
            f"'{past_key_value.name}' through pass-through ops "
            f"(direct consumers: {consumer_summary})."
        )
    return concat_nodes


def _find_current_k_input_of_concat(
    concat_node: onnx_ir.Node,
    past_key_value: onnx_ir.Value,
) -> onnx_ir.Value:
    """Return the current-K input of ``concat_node``.

    The past-key input may not be ``past_key_value`` directly: data-movement
    ops (Cast / Identity) can sit between the graph input and the Concat. We
    identify the past-key-side input by tracing backward through pass-through
    ops to the graph input.
    """
    past_indices = []
    cur_indices = []
    for index, operand in enumerate(concat_node.inputs):
        if _input_traces_back_to(operand, {past_key_value}):
            past_indices.append(index)
        else:
            cur_indices.append(index)
    if len(cur_indices) != 1:
        raise ValueError(
            f"R3 rotation: past_key input '{past_key_value.name}': Concat "
            f"'{concat_node.name}' has {len(cur_indices)} non-past_key inputs; "
            f"expected 1 (inputs="
            f"{[inp.name if inp is not None else None for inp in concat_node.inputs]}, "
            f"past_indices={past_indices})."
        )
    return concat_node.inputs[cur_indices[0]]


def _walk_forward_to_matmuls(start_node: onnx_ir.Node) -> List[onnx_ir.Node]:
    """Walk forward from ``start_node`` through pass-through ops until MatMuls.

    Steps through passthrough and scalar Mul/Div ops, returning all
    MatMul consumers reached at the first level where any appear (one per query
    head in SHA, one per group member in GQA). If at any step before reaching a
    MatMul the consumers fan out ambiguously (multiple pass-through consumers,
    or zero consumers), raises.
    """
    visited: Set[onnx_ir.Node] = set()
    cur_values = list(start_node.outputs)
    while True:
        consumers: List[onnx_ir.Node] = []
        seen: Set[onnx_ir.Node] = set()
        for value in cur_values:
            for consumer in value.consumers():
                if consumer not in seen:
                    seen.add(consumer)
                    consumers.append(consumer)

        if not consumers:
            raise ValueError(
                f"R3 rotation: forward walk from '{start_node.name}' reached "
                f"a dead end with no MatMul."
            )

        matmuls = [node for node in consumers if node.op_type == "MatMul"]
        if matmuls:
            return matmuls

        passthrough = [
            node
            for node in consumers
            if _is_passthrough(node.op_type, node.domain) or _is_constant_rescale(node)
        ]
        if len(passthrough) != 1:
            raise ValueError(
                f"R3 rotation: forward walk from '{start_node.name}' is "
                f"ambiguous; expected exactly one pass-through (data-movement) "
                f"op, found {[(node.name, node.op_type) for node in consumers]}."
            )
        next_node = passthrough[0]
        if next_node in visited:
            raise ValueError(
                f"R3 rotation: cycle detected at '{next_node.name}' while walking "
                f"forward from '{start_node.name}'."
            )
        visited.add(next_node)
        cur_values = list(next_node.outputs)


def _find_q_input_of_qk_matmul(
    qk_matmul_node: onnx_ir.Node,
    concat_node: onnx_ir.Node,
) -> Tuple[int, onnx_ir.Value]:
    """Return ``(input_idx, value)`` of the Q-side input of ``qk_matmul_node``.

    The K-side input is whichever traces back (through pass-through ops) to
    ``concat_node``; the Q-side is the other one.
    """
    if len(qk_matmul_node.inputs) != 2:
        raise ValueError(
            f"R3 rotation: QK^T MatMul '{qk_matmul_node.name}' has "
            f"{len(qk_matmul_node.inputs)} inputs; expected 2."
        )

    concat_outputs = set(concat_node.outputs)

    k_idx = None
    for index, operand in enumerate(qk_matmul_node.inputs):
        if _input_traces_back_to(operand, concat_outputs):
            if k_idx is not None:
                raise ValueError(
                    f"R3 rotation: both inputs of QK^T MatMul "
                    f"'{qk_matmul_node.name}' trace back to Concat "
                    f"'{concat_node.name}'."
                )
            k_idx = index

    if k_idx is None:
        raise ValueError(
            f"R3 rotation: neither input of QK^T MatMul "
            f"'{qk_matmul_node.name}' traces back to Concat "
            f"'{concat_node.name}'."
        )

    q_idx = 1 - k_idx
    return q_idx, qk_matmul_node.inputs[q_idx]


def _input_traces_back_to(
    start_value: Optional[onnx_ir.Value],
    target_values: Set[onnx_ir.Value],
) -> bool:
    """BFS backward from ``start_value`` through pass-through producers."""
    if start_value is None:
        return False
    visited: Set[onnx_ir.Value] = set()
    queue = deque([start_value])
    while queue:
        value = queue.popleft()
        if value in visited:
            continue
        visited.add(value)
        if value in target_values:
            return True
        producer = value.producer()
        if producer is None:
            continue
        if not _is_passthrough(
            producer.op_type, producer.domain
        ) and not _is_constant_rescale(producer):
            continue
        for operand in producer.inputs:
            if operand is not None and operand not in visited:
                queue.append(operand)
    return False


__all__ = ["BlockR3Anchors", "find_r3_anchors"]
