# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Topology-driven decoder block boundary detection.

Decoder block detection relies on the premise that transformer decoder stacks
contain exactly ``k`` active norms per block plus one final active norm:

    active norms (topological order): [n0, n1, ..., n_{kN}]
    block i boundaries               : (n_{k*i}, n_{k*(i+1)})

Most architectures use k=2 (pre-attention norm + pre-FFN norm, e.g. Llama/Qwen).
"""

from typing import Dict, List, Optional, Tuple

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology import ir_analysis
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    find_active_norms_in_ir,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


def get_decoder_block_boundaries(
    model: ModelProto,
    *,
    expected_num_blocks: Optional[int] = None,
    active_norms_per_block: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Return the residual-stream boundary tensors for each decoder block.

    ``total_active_norms`` is either ``k * N`` or ``k * N + 1`` for ``k`` active
    norms per block and ``N`` decoder blocks. The last block ends at the residual
    input of the trailing final norm when that norm is active (lm_head present),
    or at the output of the final residual ``Add`` when it is not (headless
    backbone).

    :param model: ONNX ModelProto. May be a float export or a
        ``QuantizationSimModel`` graph — quantizer nodes are stripped from the
        private analysis copy, so the returned tensor names are the un-suffixed
        names of the underlying float graph either way. Not mutated.
    :param expected_num_blocks: If provided, raises ``ValueError`` when the
        detected block count does not match.
    :param active_norms_per_block: Number of **active** norms per decoder block
      (norms whose scaled output has at least one downstream weight linear).
      Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
      NOTE: Do NOT count internal norms (e.g. Qwen3 q_norm/k_norm) — these
      are filtered out automatically and must not be included in this count.
    :return: A list of ``(start_tensor, end_tensor)`` tuples, one per decoder
        block in topological order. Both are ONNX tensor (edge) names on the
        residual stream: ``start_tensor`` is the tensor entering the block's
        input-norm, ``end_tensor`` is the tensor entering the next block's
        input-norm (for the last block, the tensor entering the final norm).
    :raises ValueError: If active norm count is inconsistent with ``k``, or if
        ``expected_num_blocks`` is given and does not match the detected count.
    """
    return get_decoder_block_boundaries_in_ir(
        ir_analysis.build_analysis_ir(model),
        expected_num_blocks=expected_num_blocks,
        active_norms_per_block=active_norms_per_block,
    )


def get_decoder_block_boundaries_in_ir(
    ir_model: onnx_ir.Model,
    active_norms: Optional[List[ActiveNorm]] = None,
    expected_num_blocks: Optional[int] = None,
    active_norms_per_block: Optional[int] = None,
    topo_index: Optional[Dict[onnx_ir.Node, int]] = None,
) -> List[Tuple[str, str]]:
    """Analysis-IR form of :func:`get_decoder_block_boundaries`.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param active_norms: Active norms in topological order. Recomputed via
        :func:`~.norm_detection.find_active_norms_in_ir` when not supplied; pass a
        precomputed value to avoid a redundant graph scan.
    :param expected_num_blocks: See :func:`get_decoder_block_boundaries`.
    :param active_norms_per_block: See :func:`get_decoder_block_boundaries`.
    :param topo_index: Precomputed node → topological index map.
    """
    if topo_index is None:
        topo_index = ir_analysis.topological_index(ir_model)
    if active_norms is None:
        active_norms = find_active_norms_in_ir(ir_model, topo_index)
    num_active_norms = len(active_norms)

    if num_active_norms == 0:
        raise ValueError(
            "No active RMSNorms found. The model may use a normalization pattern "
            "the RMSNormalization supergroup fusion does not cover, or all norms "
            "lack downstream weight linear layers."
        )

    resolved_norms_per_block = _resolve_norms_per_block(
        num_active_norms, expected_num_blocks, active_norms_per_block
    )

    # If lm_head is present, exclude its active norm from calculations
    has_lm_head = (num_active_norms - 1) % resolved_norms_per_block == 0
    if has_lm_head:
        num_active_norms = num_active_norms - 1

    remainder = num_active_norms % resolved_norms_per_block
    if remainder:
        raise ValueError(
            f"Active norm count {num_active_norms} is inconsistent with active_norms_per_block={resolved_norms_per_block}: "
            f"expected num_active_norms to be divisible by {resolved_norms_per_block} "
            f"(i.e. resolved_norms_per_block*N active norms for N decoder blocks)."
        )

    num_blocks = num_active_norms // resolved_norms_per_block
    if expected_num_blocks is not None and num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} decoder blocks but detected {num_blocks}."
        )
    _logger.debug(
        "Detected %d decoder block(s) from %d active norm(s) (%d per block).",
        num_blocks,
        num_active_norms,
        resolved_norms_per_block,
    )
    block_boundaries = [
        (
            active_norms[resolved_norms_per_block * i].input_tensor,
            active_norms[resolved_norms_per_block * (i + 1)].input_tensor,
        )
        for i in range(num_blocks - 1)
    ]

    last_block_start = active_norms[
        resolved_norms_per_block * (num_blocks - 1)
    ].input_tensor

    if has_lm_head:
        block_boundaries.append(
            (
                last_block_start,
                active_norms[resolved_norms_per_block * num_blocks].input_tensor,
            )
        )
    else:
        # Headless backbone: bound the last block with the trailing final
        # (non-active) norm's residual input.
        block_boundaries.append(
            (
                last_block_start,
                headless_block_end(ir_model, last_block_start, topo_index),
            )
        )

    return block_boundaries


def resolve_residual_tensor_name(graph: onnx_ir.Graph, tensor_name: str) -> str:
    """Walk back through leading ``Cast`` producers to the true residual tensor.

    A boundary tensor reported by :func:`get_decoder_block_boundaries` is the
    value entering a norm. In fp16 exports that value is often the output of a
    ``Cast`` sitting between the residual ``Add`` and the norm, so the boundary
    name is one hop downstream of the tensor that actually carries the
    cross-block residual. Consumers that slice the graph at a block boundary
    (e.g. AdaScale) need the pre-``Cast`` name so the slice starts on the
    residual itself.

    :param graph: Graph the name belongs to.
    :param tensor_name: Boundary tensor name to resolve.
    :return: The deepest upstream value name reachable through ``Cast``
        producers only, or ``tensor_name`` unchanged when it is absent from
        ``graph`` or is not produced by a ``Cast`` (already resolved).
    """
    name_to_value = onnx_ir.convenience.create_value_mapping(graph)
    if tensor_name not in name_to_value:
        return tensor_name
    value = name_to_value[tensor_name]
    while True:
        producer = value.producer()
        if producer is None or producer.op_type != "Cast":
            break
        upstream = producer.inputs[0]
        if upstream is None or upstream.name is None:
            break
        value = upstream
    return value.name


def _resolve_norms_per_block(
    num_active_norms: int,
    expected_num_blocks: Optional[int],
    active_norms_per_block: Optional[int],
) -> int:
    """Resolve ``k`` from an explicit value, from the expected block count, or default to 2."""
    if active_norms_per_block is not None:
        return active_norms_per_block

    if expected_num_blocks is not None:
        # Remainder of 1 allows for a trailing final norm (lm_head present).
        if num_active_norms % expected_num_blocks not in (0, 1):
            raise ValueError(
                f"Cannot infer active_norms_per_block: {num_active_norms} active norm(s) and "
                f"expected_num_blocks={expected_num_blocks} are inconsistent "
                f"(require num_active_norms mod expected_num_blocks in {{0, 1}})."
            )
        return num_active_norms // expected_num_blocks

    _logger.debug(
        "Neither expected_num_blocks nor active_norms_per_block was provided. "
        "Defaulting to active_norms_per_block=2 (Llama/Qwen2/Mistral/Phi). "
        "Pass expected_num_blocks=<N> to validate the detected block count."
    )
    return 2  # default: Llama/Qwen2/Mistral/Phi family


def headless_block_end(
    ir_model: onnx_ir.Model,
    last_block_start: str,
    topo_index: Dict[onnx_ir.Node, int],
) -> str:
    """Return the residual tensor that ends the last block of a headless backbone.

    With no active final norm there is no norm input to bound the last block, so
    walk the residual stream forward from ``last_block_start`` through its chain of
    ``Add`` ops and end on the output of the last one.

    :raises RuntimeError: If no residual ``Add`` chain is found.
    """
    residual_adds = _downstream_residual_adds(ir_model, last_block_start, topo_index)
    if not residual_adds:
        raise RuntimeError(
            "Could not isolate lm_head layer or final residual add operation for graph"
        )
    return residual_adds[-1].outputs[0].name


def _downstream_residual_adds(
    ir_model: onnx_ir.Model,
    residual_start: str,
    topo_index: Dict[onnx_ir.Node, int],
) -> List[onnx_ir.Node]:
    """Collect the ``Add`` ops on the residual stream starting at ``residual_start``.

    Follows only ``Add`` (the residual writes) and ``Cast`` (dtype hops between
    them); anything else ends that path. Returned in topological order, so the
    last entry is the final residual write.
    """
    start = _find_value(ir_model, residual_start)
    if start is None:
        return []

    adds: List[onnx_ir.Node] = []
    visited = set()
    queue = list(start.consumers())
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node.op_type not in ("Add", "Cast"):
            continue
        if node.op_type == "Add":
            adds.append(node)
        queue.extend(node.successors())

    return ir_analysis.sorted_by_topology(adds, topo_index)


def _find_value(ir_model: onnx_ir.Model, tensor_name: str) -> Optional[onnx_ir.Value]:
    """Return the graph value named ``tensor_name``, or None.

    Searches every place a tensor can come from — an initializer, a graph input,
    or a node output — so the lookup is not silently blind to one of them.
    """
    initializer = ir_model.graph.initializers.get(tensor_name)
    if initializer is not None:
        return initializer
    for value in ir_model.graph.inputs:
        if value.name == tensor_name:
            return value
    for node in ir_model.graph:
        for output in node.outputs:
            if output.name == tensor_name:
                return output
    return None


__all__ = [
    "get_decoder_block_boundaries",
    "get_decoder_block_boundaries_in_ir",
    "headless_block_end",
    "resolve_residual_tensor_name",
]
