# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Topology-driven decoder block boundary detection.

Decoder block detection relies on the premise that transformer decoder stacks
contain exactly ``k`` active norms per block plus one final active norm:

    active norms (topological order): [n0, n1, ..., n_{kN}]
    block i boundaries               : (n_{k*i}, n_{k*(i+1)})

Most architectures use k=2 (pre-attention norm + pre-FFN norm, e.g. Llama/Qwen).
"""

from typing import List, Optional, Tuple

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op, Product
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology.norm_detection import (
    find_active_norms,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


def get_decoder_block_boundaries(
    model: ModelProto,
    connected_graph: ConnectedGraph,
    expected_num_blocks: Optional[int] = None,
    active_norms_per_block: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Return the residual-stream boundary tensors for each decoder block.

    ``total_active_norms`` is either ``k * N`` or ``k * N + 1`` for ``k`` active
    norms per block and ``N`` decoder blocks. The last block ends at the trailing
    final norm when it is active (lm_head present), or at the last RMSNorm in the
    graph when it is not (headless backbone).

    :param model: ONNX ModelProto.
    :param connected_graph: ConnectedGraph built from model.
    :param expected_num_blocks: If provided, raises ``ValueError`` when the
        detected block count does not match.
    :param active_norms_per_block: Number of **active** norms per decoder block
      (norms whose gamma-scale Mul has at least one downstream weight linear).
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
    active_norms = find_active_norms(model, connected_graph)
    num_active_norms = len(active_norms)

    if num_active_norms == 0:
        raise ValueError(
            "No active RMSNorms found. The model may use a normalization pattern "
            "not covered by match_rms_norm_pattern, or all norms lack downstream "
            "weight linear layers."
        )

    resolved_norms_per_block: int
    if active_norms_per_block is not None:
        resolved_norms_per_block = active_norms_per_block
    elif expected_num_blocks is not None:
        # Remainder of 1 allows for a trailing final norm (lm_head present).
        if num_active_norms <= 0 or num_active_norms % expected_num_blocks not in (
            0,
            1,
        ):
            raise ValueError(
                f"Cannot infer active_norms_per_block: {num_active_norms} active norm(s) and "
                f"expected_num_blocks={expected_num_blocks} are inconsistent "
                f"(require num_active_norms mod expected_num_blocks in {{0, 1}})."
            )
        resolved_norms_per_block = num_active_norms // expected_num_blocks
    else:
        resolved_norms_per_block = 2  # default: Llama/Qwen2/Mistral/Phi family
        _logger.debug(
            "Neither expected_num_blocks nor active_norms_per_block was provided. "
            "Defaulting to active_norms_per_block=2 (Llama/Qwen2/Mistral/Phi). "
            "Pass expected_num_blocks=<N> to validate the detected block count."
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
            _residual_input_tensor_name(
                active_norms[resolved_norms_per_block * i].norm_op
            ),
            _residual_input_tensor_name(
                active_norms[resolved_norms_per_block * (i + 1)].norm_op
            ),
        )
        for i in range(num_blocks - 1)
    ]

    last_block_start = _residual_input_tensor_name(
        active_norms[resolved_norms_per_block * (num_blocks - 1)].norm_op
    )

    # Headless backbone: bound the last block with the trailing final non-active norm
    if not has_lm_head:
        prev_residual_output = connected_graph.get_product(last_block_start)
        residual_stream = _get_downstream_residuals(prev_residual_output)
        if not residual_stream:
            raise RuntimeError(
                "Could not isolate lm_head layer or final residual add operation for graph"
            )
        block_boundaries.append((last_block_start, residual_stream[-1].outputs[0].name))
    else:
        block_boundaries.append(
            (
                last_block_start,
                _residual_input_tensor_name(
                    active_norms[resolved_norms_per_block * num_blocks].norm_op
                ),
            )
        )

    return block_boundaries


def tensor_to_first_consumer_index(connected_graph: ConnectedGraph) -> dict:
    """Inverse of :func:`get_decoder_block_boundaries`: boundary tensor name -> op topo index.

    NOTE: Assumes the block's norm is the first op consuming the residual edge.
      That edge also feeds a later residual ``Add``; ``setdefault`` keeps the norm
      because it precedes the Add in topological order (true for pre-norm decoders).

    :param connected_graph: ConnectedGraph built from the model.
    :return: ``{tensor_name: topological_index}`` over first ``inputs[0]`` consumers.
    """
    tensor_to_index = {}
    for index, op in enumerate(connected_graph.ordered_ops):
        if op.inputs:
            tensor_to_index.setdefault(op.inputs[0].name, index)
    return tensor_to_index


def _residual_input_tensor_name(norm_op: Op) -> str:
    """Return the residual-stream tensor name entering ``norm_op`` (its ``inputs[0]``).

    NOTE: Assumes ``inputs[0]`` is the normalized activation: true for the norm-start
      ops match_rms_norm_pattern accepts (Pow(x,2) / Mul(x,x) / RMSNormalization),
      where the other input is the constant exponent or gamma.
    """
    assert not (norm_op.inputs[0].is_const or norm_op.inputs[0].is_parm), (
        f"norm op '{norm_op.name}' inputs[0] is a constant/param, not the residual activation."
    )
    return norm_op.inputs[0].name


def _get_downstream_residuals(residual_start: Product):
    """Collect all directly connected Add ops to residual_start tensor"""
    consumers = residual_start.consumers
    residual_stream = []
    queue = list(consumers)
    while queue:
        curr_op = queue.pop(0)
        if curr_op.type not in ("Add", "Cast"):
            continue
        if curr_op.type == "Add":
            residual_stream.append(curr_op)
        queue.extend(curr_op.outputs[0].consumers)

    return residual_stream
