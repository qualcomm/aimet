# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Plan which tensors sequential MSE samples for each level of layers.

Sequential MSE optimizes the weights of a model level by level. A layer's level is the number of
optimized layers on the longest path before it, so the layers of one level never read each
other's outputs and are optimized together, on inputs computed with every earlier level frozen.

:func:`plan_frontier` walks the graph once and gives every level a block for
:class:`~aimet_onnx.blockwise_sampler.BlockwiseSampler`: the inputs of the level's layers, plus
the frontier, i.e. the tensors computed from earlier levels that a later level still reads (e.g.
the residual stream). Caching the frontier lets the sampler advance from one level to the next
without re-running earlier levels. Tensors that no layer with a weight is upstream of (attention
masks, RoPE tables, embeddings) are not part of the frontier; they are recomputed from the model
inputs when needed.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import get_weight_value, is_static, outer_scope_values
from aimet_onnx.spinquant import is_online_rotation_op

# The following modules with weights are supported
SUPPORTED_MODULES = ("Conv", "Gemm", "MatMul")

_QUANTIZER_OP_TYPE = "QcQuantizeOp"

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


@dataclass
class SeqMseLayer:
    """A layer whose weight sequential MSE optimizes.

    :ivar node: Node of the layer.
    :ivar op_input_names: Names of the layer's activation inputs.
    :ivar op_output_names: Names of the layer's outputs.
    """

    node: onnx_ir.Node
    op_input_names: List[str]
    op_output_names: List[str]


@dataclass
class FrontierPlan:
    """Levels of layers, and the block of tensors to sample for each level.

    :ivar levels: Per level, its layers in graph order.
    :ivar block_inputs: Per level, the tensors to sample: the activation inputs of the level's
        layers first, then the frontier tensors to cache for later levels.
    :ivar block_outputs: Per level, the outputs of the level's layers.
    """

    levels: List[List[SeqMseLayer]]
    block_inputs: List[List[str]]
    block_outputs: List[List[str]]


def _layer_weight(node: onnx_ir.Node) -> Optional[onnx_ir.Value]:
    """
    Get the static weight of a supported node. Only a MatMul may read it through a Transpose:
    that is the only such weight ConnectedGraph gives a weight quantizer.

    :param node: Node to check
    :return: The weight value, or None if the node has no weight sequential MSE can optimize
    """
    if node.op_type not in SUPPORTED_MODULES:
        return None
    weight, _ = get_weight_value(node)
    if weight is None or (node.op_type != "MatMul" and weight is not node.inputs[1]):
        return None
    return weight


def is_supported_op(node: onnx_ir.Node) -> bool:
    """
    Checks if the node is a layer sequential MSE can optimize: a Conv, Gemm or MatMul with a
    static weight, which a MatMul may read through a Transpose.

    :param node: Node to check
    :return: True, if the node is supported
    """
    return _layer_weight(node) is not None


def get_weight_name(node: onnx_ir.Node) -> str:
    """
    Get the name of the weight of a supported node

    :param node: Supported node
    :return: Weight name
    """
    weight = _layer_weight(node)
    if weight is None:
        raise ValueError(
            f"No static weight found for {node.name} (type {node.op_type})."
        )
    return weight.name


def resolve_nodes_to_exclude(
    nodes: Iterable[onnx_ir.Node], nodes_to_exclude: Optional[Iterable[str]] = None
) -> List[str]:
    """
    The user-supplied exclusions plus the ops sequential MSE always skips: SpinQuant R3 online
    rotation MatMuls, which carry fixed Hadamards rather than learnable weights and are unsafe
    to parallelize under GQA. The naming convention is owned by the R3 pass; see
    is_online_rotation_op.

    :param nodes: Nodes of the model
    :param nodes_to_exclude: Names of the supported ops the user excludes
    :return: Names of the ops to exclude
    """
    excluded = list(nodes_to_exclude or [])
    online_rotation_ops = [
        node.name
        for node in nodes
        if is_online_rotation_op(node) and node.name not in excluded
    ]
    if online_rotation_ops:
        _logger.info(
            "Excluding %d SpinQuant R3 online rotation op(s) from sequential MSE: %s",
            len(online_rotation_ops),
            ", ".join(online_rotation_ops),
        )
        excluded.extend(online_rotation_ops)
    return excluded


def find_layers(
    nodes: Iterable[onnx_ir.Node], nodes_to_exclude: Iterable[str]
) -> List[SeqMseLayer]:
    """
    The layers sequential MSE optimizes, in graph order: the supported nodes not excluded.

    :param nodes: Nodes of the model, in graph order
    :param nodes_to_exclude: Names of the nodes to exclude
    :return: Layers to optimize
    """
    excluded = set(nodes_to_exclude)
    layers = []
    for node in nodes:
        if node.name in excluded or not is_supported_op(node):
            continue
        # The weight slot holds the weight, or a Transpose of it; neither is an activation
        weight_input = node.inputs[1]
        op_input_names = [
            inp.name
            for inp in node.inputs
            if inp is not None and inp is not weight_input and not is_static(inp)
        ]
        if not op_input_names:
            _logger.debug("Skipping %s: no dynamic activation input found", node.name)
            continue
        layers.append(
            SeqMseLayer(
                node,
                op_input_names=op_input_names,
                op_output_names=[out.name for out in node.outputs],
            )
        )
    return layers


def plan_frontier(
    graph: onnx_ir.Graph,
    layers: List[SeqMseLayer],
    weighted_op_names: Iterable[str] = (),
) -> FrontierPlan:
    """
    Group ``layers`` into levels and plan the tensors to sample for each level.

    A tensor is in level ``t``'s frontier if it is computed from levels before ``t`` only (so
    it stays valid once they are frozen), some layer with a weight is upstream of it, and an op
    that reads it needs level ``t`` or later to run and leads to a later layer. Every tensor a
    later level needs is then computed from the frontier, from tensors no weighted layer is
    upstream of, and from model inputs. Tensors produced by quantizers are planned as their
    quantizer's input, so a tensor and its quantized copy (which sequential MSE disables) are
    never cached both.

    Only the top-level graph is planned: layers inside control-flow bodies are not optimized,
    since a body runs as a whole inside its op.

    :param graph: Topologically sorted graph the layers are in, e.g. the sim's.
    :param layers: Layers to optimize. Layers that are not top-level nodes of ``graph`` are
        left out of the plan.
    :param weighted_op_names: Names of the ops with a weight that are not optimized, e.g.
        excluded layers. Tensors downstream of them are cached rather than recomputed.
    :return: The plan.
    """
    nodes = list(graph)
    node_names = {node.name for node in nodes}
    skipped = [layer.node.name for layer in layers if layer.node.name not in node_names]
    if skipped:
        _logger.debug("Not optimizing layers outside the top-level graph: %s", skipped)
    layer_of = {
        layer.node.name: layer for layer in layers if layer.node.name in node_names
    }
    weighted = set(layer_of) | set(weighted_op_names)

    def _values_read(node: onnx_ir.Node) -> List[onnx_ir.Value]:
        return [inp for inp in node.inputs if inp is not None] + outer_scope_values(
            node, graph
        )

    # depth: number of layers on the longest path before a node. A value's level is the
    # number of layers on the longest path up to and including its producer.
    depth: Dict[onnx_ir.Node, int] = {}
    value_level: Dict[onnx_ir.Value, int] = {}
    # Values that a weighted op is upstream of
    downstream_of_weight: Set[onnx_ir.Value] = set()
    consumers: Dict[onnx_ir.Value, List[onnx_ir.Node]] = {}
    for node in nodes:
        reads = _values_read(node)
        for value in reads:
            consumers.setdefault(value, []).append(node)
        depth[node] = max((value_level.get(value, 0) for value in reads), default=0)
        out_level = depth[node] + (1 if node.name in layer_of else 0)
        is_downstream = node.name in weighted or any(
            value in downstream_of_weight for value in reads
        )
        for out in node.outputs:
            value_level[out] = out_level
            if is_downstream:
                downstream_of_weight.add(out)

    def _out_level(node: onnx_ir.Node) -> int:
        return depth[node] + (1 if node.name in layer_of else 0)

    # Whether a node leads to a layer (or is one), visiting consumers before producers
    leads_to_layer: Dict[onnx_ir.Node, bool] = {}
    for node in reversed(nodes):
        leads_to_layer[node] = node.name in layer_of or any(
            leads_to_layer.get(consumer, False)
            for out in node.outputs
            for consumer in consumers.get(out, [])
        )

    layer_nodes = [node for node in nodes if node.name in layer_of]
    num_levels = max((depth[node] for node in layer_nodes), default=-1) + 1
    levels: List[List[SeqMseLayer]] = [[] for _ in range(num_levels)]
    for node in layer_nodes:
        levels[depth[node]].append(layer_of[node.name])

    frontier: List[Dict[str, None]] = [{} for _ in range(num_levels)]
    for node in nodes:
        for value in node.outputs:
            if value not in downstream_of_weight:
                continue
            # The first level that can't compute an op that reads the value
            end = max(
                (
                    _out_level(consumer)
                    for consumer in consumers.get(value, [])
                    if leads_to_layer[consumer]
                ),
                default=0,
            )
            name = _unquantized(value).name
            for level in range(value_level[value], min(end, num_levels)):
                frontier[level][name] = None

    block_inputs = []
    block_outputs = []
    for level, level_layers in enumerate(levels):
        names = dict.fromkeys(
            name for layer in level_layers for name in layer.op_input_names
        )
        names.update(frontier[level])
        block_inputs.append(list(names))
        block_outputs.append([layer.op_output_names[0] for layer in level_layers])
    return FrontierPlan(levels, block_inputs, block_outputs)


def _unquantized(value: onnx_ir.Value) -> onnx_ir.Value:
    """The tensor ``value`` is a quantized copy of, or ``value`` itself."""
    while (producer := value.producer()) is not None and (
        producer.op_type == _QUANTIZER_OP_TYPE
    ):
        value = producer.inputs[0]
    return value
