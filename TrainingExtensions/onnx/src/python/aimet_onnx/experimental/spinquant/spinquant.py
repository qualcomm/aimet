# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel.

This module is the entry-point orchestrator. It takes the decoder-stack topology
from its caller, builds a :class:`SpinquantContext`, then runs the rotation
passes selected by the caller via boolean flags (``enable_r1`` / ``enable_r2``).
"""

import warnings
from typing import List, Optional

import onnx
import onnx_ir
import torch

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.experimental.llm_topology.ir_adapter import resolve_topology
from aimet_onnx.experimental.llm_topology.ir_analysis import build_analysis_ir
from aimet_onnx.experimental.llm_topology.topology import (
    analyze_llm_topology_by_norm_count,
)
from aimet_onnx.experimental.llm_topology.topology_types import (
    BlockKind,
    LlmTopology,
)
from aimet_onnx.experimental.spinquant.model_analysis import (
    find_merger_linear2,
)
from aimet_onnx.experimental.spinquant.passes import (
    R1RotationPass,
    R2RotationPass,
    R3RotationPass,
    RotationPass,
    SpinquantContext,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_spinquant(
    model: onnx.ModelProto,
    visual_model: Optional[onnx.ModelProto] = None,
    embedding: Optional[torch.Tensor] = None,
    *,
    enable_r1: bool = True,
    enable_r2: bool = False,
    enable_r3: bool = False,
    topology: Optional[LlmTopology] = None,
) -> None:
    """Apply SpinQuant rotation transforms to an ONNX transformer model.

    SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
    quantization error. This function modifies the ONNX model(s) in-place by:

    1. Reading the backbone's decoder-stack structure (block boundaries, role map,
       hidden size) off ``topology``, and analyzing the optional visual encoder
       (PatchMerger output projection).
    2. Validating every selected rotation pass against the analysis.
    3. Applying every selected pass in order (R1 before R2 before R3).
    4. Serializing the result back onto the caller's ``ModelProto``\\ s.

    The rotations are performed on an ``onnx_ir`` copy of the graph and written
    back only once every pass has succeeded, so a failure part-way through leaves
    the caller's model untouched rather than half-rotated. (The ``embedding``
    tensor is the exception: it is a ``torch.Tensor`` rotated in place.)

    Must be called on the float ONNX model BEFORE creating a
    ``QuantizationSimModel``. The rotation modifies float weight initializers
    (R1 / R2) and may insert new nodes (R3); build the sim on the rotated graph
    and run ``compute_encodings`` afterward so quantizer scales are calibrated on
    the rotated weights.

    Supported architectures:
        - LLaMA, Qwen2, Qwen3, Phi3 (backbone only)
        - Qwen2.5-VL, Qwen3-VL (backbone + visual)

    :param model: backbone.onnx ModelProto. Mutated in-place.
    :param visual_model: Optional visual.onnx ModelProto (VLM only). Mutated in-place.
    :param embedding: Optional ``torch.Tensor`` of shape ``[vocab, hidden]`` loaded
        from ``embedding.pth`` (VLM only). Rotated in-place with R_L.
    :param enable_r1: If ``True`` (default), apply the R1 (residual-stream) rotation.
    :param enable_r2: If ``True``, apply the R2 (per-head) rotation. Defaults to ``False``.
        Not supported on architectures with fused QKV projections (e.g. Phi3).
    :param enable_r3: If ``True``, apply the R3 online Hadamard rotation on Q and K
        paths into each block's QK^T MatMul. Defaults to ``False``. Inserts new
        ``MatMul`` nodes into the ONNX graph (does not mutate existing weights).
        MHA only — not supported on fused QKV or per-head split exports. The K-side
        rotation is placed upstream of the past-key ``Concat`` so K values entering
        the KV cache are already rotated; the model's ``present_key`` output then
        carries rotated K (cache convention is self-consistent across steps).
    :param topology: Decoder-stack topology of ``model``, from
        :func:`~aimet_onnx.experimental.llm_topology.analyze_llm_topology`.
        Every rotation is placed by it — which linears read from and write to the
        residual stream (R1), which are the V/O projections (R2), which are the Q/K edges
        into QKᵀ (R3) — because discovering model structure belongs to ``llm_topology``,
        not to SpinQuant.
        Analyze the same ``model`` this call rotates.
        Passing it explicitly is also the only way to override the analysis: this function
        would otherwise analyze with ``active_norms_per_block=2`` and the default role
        patterns, which no argument here can change.
        Optional today: when omitted the topology is analyzed internally from ``model`` and
        a warning is raised. Passing it explicitly is recommended, and may become required
        in a future release.
    :raises ValueError: If no rotation is enabled, if block detection or role
        classification fails, if ``topology`` does not describe ``model``, or if any
        expected weight is missing / has the wrong shape.

    Example (LLM)::

        topology = analyze_llm_topology(model, model_type="llama")
        apply_spinquant(model, topology=topology)
        sim = QuantizationSimModel(model)   # built on the rotated graph
        sim.compute_encodings(calibration_data)

    Example (VLM)::

        embedding = torch.load("embedding.pth")   # torch.Tensor [vocab, hidden]
        topology = analyze_llm_topology(backbone_model, model_type="qwen2_5_vl")
        apply_spinquant(
            backbone_model,
            visual_model=visual_model,
            embedding=embedding,
            topology=topology,
        )
        torch.save(embedding, "embedding.pth")    # overwrite with rotated weights
        backbone_sim = QuantizationSimModel(backbone_model)
        visual_sim = QuantizationSimModel(visual_model)
        backbone_sim.compute_encodings(backbone_calibration_data)
        visual_sim.compute_encodings(visual_calibration_data)
    """
    rotations: List[RotationPass] = []
    if enable_r1:
        rotations.append(R1RotationPass())
    if enable_r2:
        rotations.append(R2RotationPass())
    if enable_r3:
        rotations.append(R3RotationPass())
    if not rotations:
        raise ValueError(
            "apply_spinquant requires at least one rotation enabled "
            "(set enable_r1=True and/or enable_r2=True and/or enable_r3=True)."
        )

    ctx = _build_context(model, visual_model, embedding, topology)

    # Validate every pass before mutating anything: a bad config must not
    # leave the model half-rotated.
    for rotation in rotations:
        rotation.validate(ctx)

    for rotation in rotations:
        _logger.info("Applying %s rotation pass.", rotation.name)
        rotation.apply(ctx)

    # The passes rewrote the IR; serialize it back onto the caller's proto(s).
    # Doing this only once every pass has succeeded is what makes a mid-flight
    # failure leave the caller's model untouched rather than half-rotated.
    _write_back(model, ctx.backbone_ir)
    if visual_model is not None:
        _write_back(visual_model, ctx.visual_ir)


def _build_context(
    model: onnx.ModelProto,
    visual_model: Optional[onnx.ModelProto],
    embedding: Optional[torch.Tensor],
    name_topology: Optional[LlmTopology],
) -> SpinquantContext:
    """Build the context shared across passes from the caller's topology.

    Builds two IRs for the backbone — the faithful one the passes mutate, and the
    quantizer-stripped / RMSNorm-fused analysis one the norm checks run on — and
    resolves the topology onto the faithful one. See :class:`SpinquantContext` on
    why the two cannot be the same object.
    """
    if name_topology is None:
        # TODO: if 'topology' is ever made required, delete this branch along with
        # the analyze_llm_topology_by_norm_count import.
        warnings.warn(
            "apply_spinquant() was called without 'topology', so the decoder-stack "
            "structure is being analyzed internally. Prefer building it with "
            "aimet_onnx.experimental.llm_topology.analyze_llm_topology(model, model_type) "
            "and passing topology=...; this argument may become required in a future release.",
            UserWarning,
            stacklevel=3,
        )
    else:
        _validate_topology(name_topology)

    # A faithful copy of the caller's graph: not sorted, not stripped, not fused,
    # so what we hand back differs from what we were given only where a rotation
    # actually changed something.
    backbone_ir = onnx_ir.from_proto(model)

    # Needed whether or not the caller supplied a topology: R1's post-writing-norm
    # check reads the fused view, which the faithful IR cannot provide.
    analysis_ir = build_analysis_ir(model)

    if name_topology is None:
        # Derives block boundaries, per-block roles, active norms, hidden_size and
        # head_dim in one pass. head_dim is only needed by R2/R3; it is left None
        # when the export has no KV-cache 'past_value' input, and those passes raise
        # a targeted error when they actually need it.
        name_topology = analyze_llm_topology_by_norm_count(model, ir_model=analysis_ir)

    # Raises if any name is absent from the graph — a topology built from another
    # model is caught here, before any pass has mutated anything.
    topology = resolve_topology(name_topology, backbone_ir)

    visual_ir = None
    visual_merger_linear2 = None
    if visual_model is not None:
        visual_ir = onnx_ir.from_proto(visual_model)
        visual_merger_linear2 = find_merger_linear2(visual_ir)

    _check_embedding_consistency(topology, embedding)

    return SpinquantContext(
        backbone_ir=backbone_ir,
        backbone_analysis_ir=analysis_ir,
        backbone_topology=topology,
        backbone_active_norms=topology.active_norms,
        backbone_hidden_size=topology.hidden_size,
        backbone_head_dim=topology.head_dim,
        visual_ir=visual_ir,
        visual_merger_linear2=visual_merger_linear2,
        embedding=embedding,
    )


def _validate_topology(topology: LlmTopology) -> None:
    """Reject a topology that cannot place SpinQuant's rotations.

    Only the fields every enabled pass needs before it can even start are checked
    here; per-pass pre-conditions stay in the passes' own ``validate``. Whether the
    names in the topology actually exist in the graph is answered separately, by
    :func:`~.ir_adapter.resolve_topology`.

    :param topology: Topology supplied by the caller.
    :raises ValueError: If the topology has no decoder blocks, no ``hidden_size``,
        or no ``active_norms``.
    """
    if not topology.blocks:
        raise ValueError(
            "topology contains no decoder blocks, so there is nothing for SpinQuant to "
            "rotate. Verify that analyze_llm_topology() was run on the model "
            "being rotated."
        )

    # The passes read only the attention/MLP fields, which a Mamba block leaves
    # empty: rotating around them would silently skip its mixer projections.
    mamba = [
        i for i, block in enumerate(topology.blocks) if block.kind is BlockKind.MAMBA
    ]
    if mamba:
        raise ValueError(
            f"topology has Mamba blocks {mamba}, which SpinQuant does not support yet."
        )

    if topology.hidden_size is None:
        raise ValueError(
            "topology.hidden_size is None, so R1 has no residual-stream dimension to build "
            "its Hadamard at. analyze_llm_topology() fills this in; a hand-built "
            "topology must set it to the model's hidden size."
        )

    # An empty list would not fail: norm fusion would quietly fuse nothing, and R1
    # would then rotate around un-absorbed RMSNorm scales — a numerically wrong model
    # rather than an error. Refuse it here.
    if not topology.active_norms:
        raise ValueError(
            "topology.active_norms is empty, so R1 has no RMSNorm scales to absorb into "
            "the linears it rotates. Rotating without that fusion silently changes the "
            "model's outputs, so it is refused. analyze_llm_topology() populates "
            "this field."
        )


def _write_back(model: onnx.ModelProto, ir_model: onnx_ir.Model) -> None:
    """Serialize ``ir_model`` onto ``model`` in place.

    Every model the caller handed us is written back, whether or not the enabled
    passes touched it. Tracking which IRs were mutated would save a serialization
    in the rare configuration that passes a visual encoder without enabling R1,
    at the cost of silently dropping rotations the day a pass forgets to report
    one — an unnecessary round trip is the cheaper mistake.
    """
    model.CopyFrom(onnx_ir.to_proto(ir_model))


def _check_embedding_consistency(topology, embedding: Optional[torch.Tensor]) -> None:
    """Reject inconsistent (embedding, embed_tokens) combinations.

    A backbone with embed_tokens must not receive an external embedding: R1 is
    already absorbed by the Gather weight, so the tensor would be rotated twice.
    """
    if embedding is not None and topology.embed_tokens:
        raise ValueError(
            "embedding was provided but backbone contains embed_tokens op(s). "
            "Pass embedding only for VLM backbones exported with use_inputs_embeds=True "
            "(i.e. backbone has no Gather op for token embeddings)."
        )
