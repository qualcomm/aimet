# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""onnx_ir-flavored view of an LLM topology.

The analyzers in this package describe a decoder stack by name (see
:mod:`~.topology_types`). A consumer that goes on to *rewrite* the graph needs
handles it can mutate, so :func:`resolve_topology` re-attaches an
:class:`onnx_ir.Model` to a name-based topology and hands back the same structure
carrying ``onnx_ir.Node`` and ``onnx_ir.Value`` objects — the ``Ir``-prefixed
dataclasses below.

An ``onnx_ir.Value`` already knows its producer and its consumers, so a consumer
needs no side tables, and the graph object stays valid across node insertions.

Which model to resolve against
------------------------------
Resolve against the model you intend to **mutate** — a plain
``onnx_ir.from_proto(model)`` — not against the analysis IR from
:func:`~.ir_analysis.build_analysis_ir`. The analysis IR is quantizer-stripped
and has its decomposed RMSNorms replaced by fused ``RMSNormalization``
supergroup nodes, so it must never be serialized back to a caller. Every name a
topology reports survives that fusion (linears, embeddings and dynamic MatMuls
are untouched by it; gammas and block-boundary tensors are region *boundaries*),
which is what lets a topology analyzed on the fused IR resolve cleanly onto the
faithful one.

Resolution keys on node names. :func:`~.ir_analysis.node_name` rejects unnamed
nodes during analysis, so a name is always present by the time anything reaches
here.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Union

import onnx_ir

from aimet_onnx.experimental.llm_topology.layer_roles import LinearRole
from aimet_onnx.experimental.llm_topology.norm_detection import ActiveNorm
from aimet_onnx.experimental.llm_topology.topology_types import (
    AttentionBlockTopology,
    BlockKind,
    LinearGroup,
    LlmTopology,
    MambaBlockTopology,
)


@dataclass
class IrActiveNorm:
    """An affine RMSNorm that has at least one downstream weighted linear.

    The IR-bearing counterpart of :class:`~.norm_detection.ActiveNorm`. A
    consumer that absorbs the norm's gamma into its downstream linears (e.g.
    SpinQuant's R1) needs to rewrite the gamma tensor itself, so ``scale`` is the
    gamma ``Value`` rather than just its name.

    :param scale: The gamma (scale) tensor of the norm — an initializer or a
        ``Constant`` node output.
    :param downstream_linears: MatMul/Gemm/Conv nodes reachable from the norm.
    :param input_tensor: Residual-stream tensor entering the norm.
    """

    scale: onnx_ir.Value
    downstream_linears: List[onnx_ir.Node] = field(default_factory=list)
    input_tensor: Optional[onnx_ir.Value] = None

    @property
    def scale_name(self) -> str:
        """Name of the gamma tensor."""
        return self.scale.name


@dataclass
class IrLinearGroup:
    """A norm's downstream weighted linears, together with their role split.

    ``nodes`` is the coarse read group (the single source of truth): every
    weighted linear reading through one active norm. ``by_role`` is a *partition*
    of ``nodes`` — each node appears under exactly one :class:`LinearRole`
    (unmatched nodes fall under :attr:`LinearRole.UNKNOWN`).

    Consumers that treat the whole group uniformly (e.g. an R1 residual-stream
    rotation) iterate ``nodes``; consumers that touch one projection (e.g. R2
    rotates only V) read :meth:`role`. A role maps to a *list* because per-head
    split exports (SHA) emit one node per head, and fused exports (e.g. Phi3
    ``qkv_proj``) collapse several logical roles into a single node reported
    under a ``FUSED_*`` role.

    :param nodes: All weighted linears reading through one active norm.
    :param by_role: Partition of ``nodes`` keyed by :class:`LinearRole`.
    """

    nodes: List[onnx_ir.Node] = field(default_factory=list)
    by_role: Dict[LinearRole, List[onnx_ir.Node]] = field(default_factory=dict)

    def role(self, role: LinearRole) -> List[onnx_ir.Node]:
        """Return the nodes classified as ``role`` (empty list if none)."""
        return self.by_role.get(role, [])


@dataclass
class IrBlockTopology:
    """Residual-stream boundaries shared by every IR decoder block kind.

    :param residual_input: Residual-stream tensor entering the block.
    :param residual_output: Residual-stream tensor leaving the block.
    """

    residual_input: Optional[onnx_ir.Value] = None
    residual_output: Optional[onnx_ir.Value] = None


@dataclass
class IrAttentionBlockTopology(IrBlockTopology):
    """Attention and MLP projections, plus dynamic attention MatMuls.

    The two weighted read groups are :class:`IrLinearGroup` values — each exposes
    both its coarse ``nodes`` list and the fine-grained role split (see
    :class:`IrLinearGroup` and the ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``gate_proj`` / ``up_proj`` convenience properties below). The two write
    projections and the dynamic attention MatMuls are plain node lists.

    :param qkv: Attention read group — the Q/K/V (or fused QKV) projections
        reading through the block's input norm.
    :param o_proj: Attention-output projection node(s) writing to the residual.
    :param gate_up: MLP read group — the gate/up (or fused gate-up) projections
        reading through the post-attention norm.
    :param down_proj: MLP-output projection node(s) writing to the residual.
    :param qk_matmul: The dynamic (non-weighted) Q·Kᵀ attention MatMul node(s) —
        one per query head in SHA exports.
    :param attn_v_matmul: The dynamic (non-weighted) softmax·V MatMul node(s).
    """

    kind: ClassVar[BlockKind] = BlockKind.ATTENTION

    qkv: IrLinearGroup = field(default_factory=IrLinearGroup)
    o_proj: List[onnx_ir.Node] = field(default_factory=list)
    gate_up: IrLinearGroup = field(default_factory=IrLinearGroup)
    down_proj: List[onnx_ir.Node] = field(default_factory=list)

    qk_matmul: List[onnx_ir.Node] = field(default_factory=list)
    attn_v_matmul: List[onnx_ir.Node] = field(default_factory=list)

    @property
    def q_proj(self) -> List[onnx_ir.Node]:
        """Query projection node(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.Q_PROJ)

    @property
    def k_proj(self) -> List[onnx_ir.Node]:
        """Key projection node(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.K_PROJ)

    @property
    def v_proj(self) -> List[onnx_ir.Node]:
        """Value projection node(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.V_PROJ)

    @property
    def gate_proj(self) -> List[onnx_ir.Node]:
        """Gate projection node(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.GATE_PROJ)

    @property
    def up_proj(self) -> List[onnx_ir.Node]:
        """Up projection node(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.UP_PROJ)


@dataclass
class IrMambaBlockTopology(IrBlockTopology):
    """Mamba mixer projections around the selective scan.

    :param mixer: Read group — the mixer's ``in_proj`` node(s).
    :param mixer_out_proj: Mixer ``out_proj`` node(s) writing to the residual.
    """

    kind: ClassVar[BlockKind] = BlockKind.MAMBA
    mixer: IrLinearGroup = field(default_factory=IrLinearGroup)
    mixer_out_proj: List[onnx_ir.Node] = field(default_factory=list)


@dataclass
class IrLlmTopology:
    """Topology of an ONNX decoder-stack model: blocks + backbone-level roles + dims.

    :param embed_tokens: Token-embedding ``Gather`` node(s) that produce the
        initial residual-stream activations.
    :param lm_head: Vocabulary-projection linear(s) downstream of the final norm.
    :param blocks: Per-decoder-block topology in topological order.
    :param past_key_input_names: Raw ``past_key_*`` graph inputs in declaration
        order, collected tolerantly (empty for prefill-only exports without a
        KV-cache). Pairing these to ``blocks`` and validating that their count
        matches the block count are the consumer's responsibility (e.g. R3) —
        R1-only and prefill-only flows do not require KV-cache inputs.
    :param past_key_output_names: Key-cache graph outputs in declaration order.
    :param past_value_input_names: Value-cache graph inputs in declaration order.
    :param past_value_output_names: Value-cache graph outputs in declaration order.
    :param active_norms: Active norms in topological order used to build the
        topology.
    :param hidden_size: Residual-stream hidden dimension (``None`` if not
        inferred; :func:`~.topology.analyze_llm_topology` fills it).
    :param head_dim: Per-head dimension (``None`` when it could not be derived,
        e.g. an export without KV-cache inputs).
    """

    embed_tokens: List[onnx_ir.Node] = field(default_factory=list)
    lm_head: List[onnx_ir.Node] = field(default_factory=list)
    blocks: List[Union[IrAttentionBlockTopology, IrMambaBlockTopology]] = field(
        default_factory=list
    )
    past_key_input_names: List[str] = field(default_factory=list)
    past_key_output_names: List[str] = field(default_factory=list)
    past_value_input_names: List[str] = field(default_factory=list)
    past_value_output_names: List[str] = field(default_factory=list)
    active_norms: Optional[List[IrActiveNorm]] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None


def resolve_topology(
    topology: LlmTopology,
    ir_model: onnx_ir.Model,
) -> IrLlmTopology:
    """Re-attach ``ir_model`` to a name-based topology.

    Every node name in the topology is replaced with the ``onnx_ir.Node`` of that
    name; every tensor name becomes the ``onnx_ir.Value`` of that name.

    :param topology: Name-based topology, as returned by
        :func:`~.topology.analyze_llm_topology`.
    :param ir_model: The IR model to resolve against — the one the caller intends
        to mutate. See the module docstring on why this is not the analysis IR.
    :return: The equivalent IR-bearing :class:`IrLlmTopology`.
    :raises ValueError: If any name cannot be resolved against ``ir_model`` — the
        two were built from different graphs.
    """
    nodes = _node_by_name(ir_model)
    values = onnx_ir.convenience.create_value_mapping(ir_model.graph)

    resolved = IrLlmTopology(
        embed_tokens=_resolve_nodes(topology.embed_tokens, nodes),
        lm_head=_resolve_nodes(topology.lm_head, nodes),
        past_key_input_names=list(topology.past_key_input_names),
        past_key_output_names=list(topology.past_key_output_names),
        past_value_input_names=list(topology.past_value_input_names),
        past_value_output_names=list(topology.past_value_output_names),
        active_norms=resolve_active_norms(topology.active_norms or [], ir_model),
        hidden_size=topology.hidden_size,
        head_dim=topology.head_dim,
    )
    for block in topology.blocks:
        residual_input = _resolve_value(block.residual_input, values)
        residual_output = _resolve_value(block.residual_output, values)
        if isinstance(block, AttentionBlockTopology):
            resolved_block = IrAttentionBlockTopology(
                qkv=_resolve_group(block.qkv, nodes),
                o_proj=_resolve_nodes(block.o_proj, nodes),
                gate_up=_resolve_group(block.gate_up, nodes),
                down_proj=_resolve_nodes(block.down_proj, nodes),
                qk_matmul=_resolve_nodes(block.qk_matmul, nodes),
                attn_v_matmul=_resolve_nodes(block.attn_v_matmul, nodes),
                residual_input=residual_input,
                residual_output=residual_output,
            )
        elif isinstance(block, MambaBlockTopology):
            resolved_block = IrMambaBlockTopology(
                mixer=_resolve_group(block.mixer, nodes),
                mixer_out_proj=_resolve_nodes(block.mixer_out_proj, nodes),
                residual_input=residual_input,
                residual_output=residual_output,
            )
        else:
            raise TypeError(f"Unsupported block topology: {type(block).__name__}")
        resolved.blocks.append(resolved_block)
    return resolved


def resolve_active_norms(
    active_norms: List[ActiveNorm],
    ir_model: onnx_ir.Model,
) -> List[IrActiveNorm]:
    """Re-attach ``ir_model`` to name-based active norms.

    :param active_norms: Name-based norms from
        :func:`~.norm_detection.find_active_norms`.
    :param ir_model: The IR model to resolve against.
    :return: The equivalent IR-bearing :class:`IrActiveNorm`\\ s, in the same order.
    :raises ValueError: If a gamma tensor or a downstream linear cannot be resolved.
    """
    nodes = _node_by_name(ir_model)
    values = onnx_ir.convenience.create_value_mapping(ir_model.graph)
    return [
        IrActiveNorm(
            scale=_resolve_value(active_norm.scale_name, values),
            downstream_linears=_resolve_nodes(active_norm.downstream_linears, nodes),
            input_tensor=_resolve_value(active_norm.input_tensor or None, values),
        )
        for active_norm in active_norms
    ]


def _node_by_name(ir_model: onnx_ir.Model) -> Dict[str, onnx_ir.Node]:
    """Return ``{node name: node}``, skipping unnamed nodes.

    Unlike :func:`~.ir_analysis.node_by_name` this tolerates unnamed nodes rather
    than rejecting them: a faithful (unfused) graph can carry unnamed ``Constant``
    nodes an exporter emitted, and none of them can be a topology target anyway
    — a name is what a topology reports.
    """
    return {node.name: node for node in ir_model.graph if node.name}


def _resolve_group(
    group: LinearGroup,
    nodes: Dict[str, onnx_ir.Node],
) -> IrLinearGroup:
    """Resolve a name-based read group, preserving its role split."""
    return IrLinearGroup(
        nodes=_resolve_nodes(group.linears, nodes),
        by_role={
            role: _resolve_nodes(names, nodes) for role, names in group.by_role.items()
        },
    )


def _resolve_nodes(
    node_names: List[str],
    nodes: Dict[str, onnx_ir.Node],
) -> List[onnx_ir.Node]:
    """Resolve every node name to its ``onnx_ir.Node``."""
    resolved = []
    for name in node_names:
        node = nodes.get(name)
        if node is None:
            raise ValueError(
                f"Cannot resolve node '{name}' against the supplied IR model. "
                "The topology and the IR model were built from different graphs."
            )
        resolved.append(node)
    return resolved


def _resolve_value(
    tensor_name: Optional[str],
    values: Dict[str, onnx_ir.Value],
) -> Optional[onnx_ir.Value]:
    """Resolve a tensor name to its ``onnx_ir.Value``.

    :raises ValueError: If the tensor is absent. Returning ``None`` here would
        push the failure downstream into a consumer, which can only report it as
        a missing edge.
    """
    if tensor_name is None:
        return None
    value = values.get(tensor_name)
    if value is None:
        raise ValueError(
            f"Tensor '{tensor_name}' is absent from the supplied IR model. The "
            "topology and the IR model were built from different graphs."
        )
    return value


__all__ = [
    "IrActiveNorm",
    "IrAttentionBlockTopology",
    "IrBlockTopology",
    "IrLinearGroup",
    "IrLlmTopology",
    "IrMambaBlockTopology",
    "resolve_active_norms",
    "resolve_topology",
]
