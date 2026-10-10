# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Affine RMSNorm detection for decoder block topology.

An *active norm* is an affine RMSNorm whose scaled output has at least one
downstream weight MatMul/Gemm/Conv reachable through grid-preserving ops.
Internal norms (e.g. Qwen3 q_norm/k_norm) whose outputs feed into attention ops
before reaching any linear weight are excluded automatically.

Detection runs on the analysis IR (see
:mod:`aimet_onnx.llm_topology.ir_analysis`), where every decomposed
RMSNorm has already been fused into a single ``RMSNormalization`` supergroup
node. Finding norms is therefore a node-type lookup rather than a multi-op
pattern match, and needs no ConnectedGraph.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import is_static
from aimet_onnx.utils import ModelProto

from aimet_onnx.llm_topology import ir_analysis

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


@dataclass
class ActiveNorm:
    """An affine RMSNorm that has at least one downstream weight linear op.

    Described entirely by name: no ``onnx_ir.Node`` is retained, so this outlives
    the graph object it was derived from.
    :func:`~.ir_adapter.resolve_active_norms` turns it into the handle-bearing
    :class:`~.ir_adapter.IrActiveNorm` that SpinQuant consumes.

    :param norm: Name of the fused ``RMSNormalization`` node.
    :param input_tensor: Residual-stream tensor entering the norm (``inputs[0]``).
        This is the tensor decoder-block boundaries are expressed in.
    :param scale_name: Name of the gamma (scale) initializer in the model.
    :param downstream_linears: Names of the weighted linears reachable from the
        norm's scaled output.
    :param topo_index: Position of the norm node in the analysis IR's topological
        order. Used to assign norms to decoder blocks.
    """

    norm: str
    input_tensor: str
    scale_name: str
    downstream_linears: List[str] = field(default_factory=list)
    topo_index: int = -1


def find_active_norms(model: ModelProto) -> List[ActiveNorm]:
    """Return all affine RMSNorms in ``model`` with at least one downstream weight linear.

    :param model: ONNX ModelProto. Not mutated — analysis runs on a private
        onnx_ir copy (see :func:`~.ir_analysis.build_analysis_ir`).
    :return: ``ActiveNorm`` objects in topological order.
    """
    return find_active_norms_in_ir(ir_analysis.build_analysis_ir(model))


def find_active_norms_in_ir(
    ir_model: onnx_ir.Model,
    topo_index: Optional[Dict[onnx_ir.Node, int]] = None,
) -> List[ActiveNorm]:
    """Return the active norms of an already-built analysis IR model.

    Iterates the graph in topological order and collects every fused
    ``RMSNormalization`` node that is affine (carries a gamma input) and whose
    scaled output reaches at least one weighted MatMul/Gemm/Conv through
    reshape-only ops. Norms with no weight consumers are omitted.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param topo_index: Precomputed node → topological index map; computed here
        when omitted.
    :return: ``ActiveNorm`` objects in topological order.
    """
    if topo_index is None:
        topo_index = ir_analysis.topological_index(ir_model)

    result: List[ActiveNorm] = []
    for node in ir_model.graph:
        if not ir_analysis.is_rms_norm(node):
            continue

        scale = _gamma_input(node)
        if scale is None:
            _logger.debug(
                "RMSNorm '%s': non-affine (no gamma input), skipping.",
                ir_analysis.node_name(node),
            )
            continue

        input_tensor = _residual_input(node)
        if input_tensor is None:
            _logger.debug(
                "RMSNorm '%s': inputs[0] is a constant, not a residual activation; skipping.",
                ir_analysis.node_name(node),
            )
            continue

        downstream_linears = _find_downstream_linears(node, topo_index)
        if not downstream_linears:
            _logger.debug(
                "RMSNorm scale '%s' (node '%s'): no downstream weight linears, skipping.",
                scale.name,
                ir_analysis.node_name(node),
            )
            continue

        result.append(
            ActiveNorm(
                norm=ir_analysis.node_name(node),
                input_tensor=input_tensor,
                scale_name=scale.name or "",
                downstream_linears=downstream_linears,
                topo_index=topo_index[node],
            )
        )

    _logger.debug("Found %d active norm(s).", len(result))
    return result


def get_active_norm(
    node: onnx_ir.Node,
    topo_index: Dict[onnx_ir.Node, int],
) -> Optional[ActiveNorm]:
    """Return the :class:`ActiveNorm` for one fused RMSNorm node.

    Unlike :func:`find_active_norms_in_ir` the norm is not required to feed a
    weighted linear, so ``downstream_linears`` may be empty (e.g. the final norm
    of a headless backbone). The caller decides whether that is acceptable.

    :param node: Fused ``RMSNormalization`` node, e.g. one found by name.
    :param topo_index: Node → topological index map.
    :return: The norm, or None if ``node`` is not a fused affine RMSNorm applied
        to an activation.
    """
    if not ir_analysis.is_rms_norm(node):
        return None
    scale = _gamma_input(node)
    input_tensor = _residual_input(node)
    if scale is None or input_tensor is None:
        return None
    return ActiveNorm(
        norm=ir_analysis.node_name(node),
        input_tensor=input_tensor,
        scale_name=scale.name or "",
        downstream_linears=_find_downstream_linears(node, topo_index),
        topo_index=topo_index[node],
    )


def get_last_norm_input_tensor(model: ModelProto) -> str:
    """Return the residual tensor entering the last RMSNorm in topological order.

    Unlike :func:`find_active_norms` this ignores whether the norm has downstream
    weight linears, so it also finds the trailing final norm of a headless
    backbone (no lm_head) — which is what bounds the last decoder block there.

    :param model: ONNX ModelProto.
    :return: Name of the tensor entering the last RMSNorm.
    :raises RuntimeError: If the graph contains no RMSNorm.
    """
    return get_last_norm_input_tensor_in_ir(ir_analysis.build_analysis_ir(model))


def get_last_norm_input_tensor_in_ir(ir_model: onnx_ir.Model) -> str:
    """Analysis-IR form of :func:`get_last_norm_input_tensor`."""
    for node in reversed(tuple(ir_model.graph)):
        if not ir_analysis.is_rms_norm(node):
            continue
        input_tensor = _residual_input(node)
        if input_tensor is not None:
            return input_tensor
    raise RuntimeError("No RMSNorm ops found in graph")


def is_affine_rms_norm(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` is a fused RMSNorm that applies a gamma scale.

    Unlike :func:`find_active_norms_in_ir` this says nothing about whether the
    norm has downstream weight linears — it only asks whether there is a scale at
    all. Used by precondition checks that must reject *any* affine norm in a
    position (e.g. SpinQuant R1 rejects one between a writing layer and the
    residual add).
    """
    return ir_analysis.is_rms_norm(node) and _gamma_input(node) is not None


def _gamma_input(norm_node: onnx_ir.Node) -> Optional[onnx_ir.Value]:
    """Return the gamma (scale) input of an affine RMSNorm node, else None.

    Per the ONNX ``RMSNormalization`` spec gamma is the second input. A
    non-affine norm has only ``X``, and there is no scale to report.
    """
    if len(norm_node.inputs) < 2:
        return None
    scale = norm_node.inputs[1]
    if scale is None or not is_static(scale):
        return None
    return scale


def _residual_input(norm_node: onnx_ir.Node) -> Optional[str]:
    """Return the name of the residual activation entering ``norm_node``.

    ``inputs[0]`` of an ``RMSNormalization`` node is the normalized activation by
    spec. Returns None if it is static (a malformed norm) or unnamed, so the
    caller can skip it rather than emit a bogus boundary tensor.
    """
    if not norm_node.inputs:
        return None
    activation = norm_node.inputs[0]
    if activation is None or not activation.name:
        return None
    if is_static(activation):
        return None
    return activation.name


def _find_downstream_linears(
    norm_node: onnx_ir.Node,
    topo_index: Dict[onnx_ir.Node, int],
) -> List[str]:
    """Return the weighted linears reachable from ``norm_node``'s scaled output.

    MatMul/Gemm/Conv nodes can be direct consumers, or reached through a chain of
    reshape/reformat ops (Unsqueeze, Transpose, Cast, Reshape, ...) that adjust
    the activation layout into Conv-compatible format. The walk stops at the first
    linear on each path and does not cross an unweighted MatMul.

    :param norm_node: The fused ``RMSNormalization`` node (walk start).
    :param topo_index: Node → topological index map, used to order the result.
    :return: The name of each weighted linear consuming the scaled activations,
        in topological order.
    """
    linears: List[onnx_ir.Node] = []
    visited = set()
    queue = list(norm_node.outputs[0].consumers())
    while queue:
        consumer = queue.pop()
        if consumer in visited:
            continue
        visited.add(consumer)
        if ir_analysis.is_weighted_linear(consumer):
            linears.append(consumer)
        elif consumer.op_type in ir_analysis.GRID_PRESERVING_TYPES:
            queue.extend(consumer.successors())
    return [
        ir_analysis.node_name(node)
        for node in ir_analysis.sorted_by_topology(linears, topo_index)
    ]


__all__ = [
    "ActiveNorm",
    "get_active_norm",
    "find_active_norms",
    "find_active_norms_in_ir",
    "get_last_norm_input_tensor",
    "get_last_norm_input_tensor_in_ir",
    "is_affine_rms_norm",
]
