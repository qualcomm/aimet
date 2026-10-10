# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Base class and shared context for SpinQuant rotation passes."""

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import onnx_ir
import torch

from aimet_onnx.llm_topology.ir_adapter import (
    IrActiveNorm,
    IrLlmTopology,
)


@dataclass
class SpinquantContext:
    """Inputs and pre-computed analysis shared across rotation passes.

    Built once by :func:`apply_spinquant`, then handed to every pass. Passes must
    not mutate the analysis fields; they rewrite the graph through
    ``backbone_ir`` / ``visual_ir``, which :func:`apply_spinquant` serializes back
    onto the caller's ``ModelProto``\\ s once every pass has succeeded.

    Two IRs describe the backbone, and the distinction matters:

    * ``backbone_ir`` is a faithful copy of the caller's graph and is the *only*
      thing a pass may mutate. Every node and tensor the topology names is
      resolved against it.
    * ``backbone_analysis_ir`` is the detection view from
      :func:`~.ir_analysis.build_analysis_ir`: quantizer-stripped, with decomposed
      RMSNorms replaced by fused ``RMSNormalization`` nodes. It is read-only, and
      must never be serialized back to a caller.

    :param backbone_ir: Faithful IR of backbone.onnx. Mutated by the passes.
    :param backbone_analysis_ir: Read-only analysis IR of the same backbone, for
        checks that need the fused norm view (e.g. R1's post-writing-norm check).
    :param backbone_topology: LLM topology (blocks + intra-block roles) for the
        backbone, resolved onto ``backbone_ir``.
    :param backbone_active_norms: Active norms in topological order, resolved onto
        ``backbone_ir``.
    :param backbone_hidden_size: Hidden dimension of the language backbone residual stream.
    :param backbone_head_dim: Per-head dimension derived from a ``past_value`` graph
        input. ``None`` if the export has no KV-cache inputs; passes that need
        ``head_dim`` (e.g. R2) must error in that case.
    :param visual_ir: Faithful IR of visual.onnx (VLM only). Mutated by R1.
    :param visual_merger_linear2: PatchMerger linear_fc2 nodes (VLM only).
    :param embedding: Optional raw embedding tensor (VLM with use_inputs_embeds=True).
    """

    backbone_ir: onnx_ir.Model
    backbone_analysis_ir: onnx_ir.Model
    backbone_topology: IrLlmTopology
    backbone_active_norms: List[IrActiveNorm]
    backbone_hidden_size: int
    backbone_head_dim: Optional[int] = None
    visual_ir: Optional[onnx_ir.Model] = None
    visual_merger_linear2: Optional[List[onnx_ir.Node]] = field(default=None)
    embedding: Optional[torch.Tensor] = None


class RotationPass(abc.ABC):
    """A single SpinQuant rotation (e.g. R1, R2, R3).

    Sub-classes encapsulate:

    * which rotation matrix to construct, and at what dimension;
    * which ops in the role map to rotate, and on which axis;
    * any architectural pre-conditions to check before mutating the model;
    * any required setup (e.g. norm fusion) that must run before the rotation.

    The orchestrator validates every pass first, then applies them in order, so
    a bad configuration cannot leave the model half-rotated.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in log messages."""

    @abc.abstractmethod
    def validate(self, ctx: SpinquantContext) -> None:
        """Raise if the model or context is incompatible with this rotation.

        Called for every pass before any pass mutates the model.
        """

    @abc.abstractmethod
    def apply(self, ctx: SpinquantContext) -> None:
        """Mutate the IR model(s) and any auxiliary tensors in-place."""
