# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology: block boundaries + intra-block structure.

Top-level entry point for describing the structure of an ONNX decoder-stack
model — where the blocks are, and what the q/k/v/o, gate/up/down projections
and dynamic attention MatMuls are inside each block. Technique-agnostic.

Analysis runs on onnx_ir and reports results by ONNX name
(:class:`LlmTopology`, from :func:`analyze_llm_topology`). A name-based topology
outlives the graph object it was derived from, so it can be handed around and
applied to a ``ModelProto`` directly.

A consumer that goes on to *rewrite* the graph needs handles rather than names:
:func:`resolve_topology` re-attaches an :class:`onnx_ir.Model` and returns the
same structure carrying ``onnx_ir.Node`` / ``onnx_ir.Value`` objects, under the
``Ir``-prefixed types (:class:`IrLlmTopology` and friends). Resolve against the
model you intend to mutate, not the analysis IR — see :mod:`~.ir_adapter`.
"""

from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    resolve_residual_tensor_name,
)
from aimet_onnx.experimental.llm_topology.ir_adapter import (
    IrActiveNorm,
    IrBlockTopology,
    IrLinearGroup,
    IrLlmTopology,
    resolve_active_norms,
    resolve_topology,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    find_active_norms,
)
from aimet_onnx.experimental.llm_topology.topology import (
    ACTIVE_NORM_MODEL_TYPES,
    analyze_llm_topology,
    analyze_llm_topology_by_norm_count,
    get_llm_topology,
)
from aimet_onnx.experimental.llm_topology.topology_types import (
    BlockTopology,
    LinearGroup,
    LlmTopology,
)

__all__ = [
    "ACTIVE_NORM_MODEL_TYPES",
    "ActiveNorm",
    "BlockTopology",
    "IrActiveNorm",
    "IrBlockTopology",
    "IrLinearGroup",
    "IrLlmTopology",
    "LinearGroup",
    "LinearRole",
    "LlmTopology",
    "analyze_llm_topology",
    "analyze_llm_topology_by_norm_count",
    "classify_linear_role",
    "find_active_norms",
    "get_decoder_block_boundaries",
    "get_llm_topology",
    "resolve_active_norms",
    "resolve_residual_tensor_name",
    "resolve_topology",
]
