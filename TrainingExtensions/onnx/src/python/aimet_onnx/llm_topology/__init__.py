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

from aimet_onnx.llm_topology.block_boundaries import get_decoder_block_boundaries
from aimet_onnx.llm_topology.ir_adapter import resolve_topology
from aimet_onnx.llm_topology.topology import analyze_llm_topology
from aimet_onnx.llm_topology.topology_types import LlmTopology

__all__ = [
    "LlmTopology",
    "analyze_llm_topology",
    "get_decoder_block_boundaries",
    "resolve_topology",
]
