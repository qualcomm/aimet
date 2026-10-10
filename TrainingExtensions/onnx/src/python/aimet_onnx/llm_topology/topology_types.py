# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Name-based description of an LLM decoder stack.

The dataclasses here are what :func:`~.topology.get_llm_topology` produces,
built entirely from ONNX names. Nothing in here holds an ``onnx_ir.Node``, so a
topology stays valid after the graph object it was derived from is gone, and can
be applied directly to a ``ModelProto``.

A consumer that needs mutable graph handles rather than names resolves one of
these onto an :class:`onnx_ir.Model` with
:func:`~.ir_adapter.resolve_topology`, which returns the ``Ir``-prefixed
counterparts in :mod:`~.ir_adapter`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Pattern, Union

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
)
from aimet_onnx.llm_topology.norm_detection import ActiveNorm

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


@dataclass
class LinearGroup:
    """A norm's downstream weighted linears, together with their role split.

    ``linears`` is the coarse read group (the single source of truth): every
    weighted linear reading through one active norm. ``by_role`` is a name-based
    *partition* of ``linears`` produced by :func:`classify_linear_role` — each
    entry appears under exactly one :class:`LinearRole` (unmatched entries fall
    under :attr:`LinearRole.UNKNOWN`). Because the split is derived from
    ``linears`` at construction, the two can never disagree.

    Consumers that treat the whole group uniformly (e.g. an R1 residual-stream
    rotation) iterate ``linears``; consumers that touch one projection (e.g. R2
    rotates only V) read :meth:`role`. A role maps to a *list* because per-head
    split exports (SHA) emit one node per head, and fused exports (e.g. Phi3
    ``qkv_proj``) collapse several logical roles into a single node reported
    under a ``FUSED_*`` role.

    :param linears: Names of all weighted linears reading through one active norm.
    :param by_role: Partition of ``linears`` keyed by :class:`LinearRole`.
    """

    linears: List[str] = field(default_factory=list)
    by_role: Dict[LinearRole, List[str]] = field(default_factory=dict)

    @classmethod
    def classify(
        cls,
        linears: List[str],
        role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    ) -> "LinearGroup":
        """Build a group from ``linears``, splitting it into roles by module name."""
        return cls(linears=list(linears), by_role=split_by_role(linears, role_patterns))

    def role(self, role: LinearRole) -> List[str]:
        """Return the linears classified as ``role`` (empty list if none)."""
        return self.by_role.get(role, [])


class BlockKind(Enum):
    """What a decoder block computes between its residual-stream read and write."""

    #: Attention followed by an MLP (Llama, Qwen, Gemma, Phi, ...).
    ATTENTION = "attention"
    #: A selective state-space mixer (Mamba2): one norm, ``in_proj`` reads the
    #: residual stream, the scan, ``out_proj`` writes it back. No MLP.
    MAMBA = "mamba"


@dataclass
class BlockTopology:
    """Residual-stream boundaries shared by every decoder block kind.

    :param residual_input: Name of the residual-stream tensor entering the block.
    :param residual_output: Name of the residual-stream tensor leaving the block.
    """

    residual_input: Optional[str] = None
    residual_output: Optional[str] = None


@dataclass
class AttentionBlockTopology(BlockTopology):
    """Attention and MLP projections, plus dynamic attention MatMuls.

    The two weighted read groups are :class:`LinearGroup` values — each
    exposes both its coarse ``linears`` list and the fine-grained role split (see
    the ``q_proj`` / ``k_proj`` / ``v_proj`` / ``gate_proj`` / ``up_proj``
    convenience properties below). The two write projections and the dynamic
    attention MatMuls are plain lists.

    :param qkv: Attention read group — the Q/K/V (or fused QKV) projections
        reading through the block's input norm.
    :param o_proj: Name(s) of the attention-output projection writing to the residual.
    :param gate_up: MLP read group — the gate/up (or fused gate-up) projections
        reading through the post-attention norm.
    :param down_proj: Name(s) of the MLP-output projection writing to the residual.
    :param qk_matmul: Node names of the dynamic (non-weighted) Q·Kᵀ attention
        MatMul(s) — one per query head in SHA exports. Plain names, because
        nothing about these nodes beyond their identity is needed.
    :param attn_v_matmul: Node names of the dynamic (non-weighted) softmax·V
        MatMul(s).
    """

    kind: ClassVar[BlockKind] = BlockKind.ATTENTION

    qkv: LinearGroup = field(default_factory=LinearGroup)
    o_proj: List[str] = field(default_factory=list)
    gate_up: LinearGroup = field(default_factory=LinearGroup)
    down_proj: List[str] = field(default_factory=list)

    qk_matmul: List[str] = field(default_factory=list)
    attn_v_matmul: List[str] = field(default_factory=list)

    @property
    def q_proj(self) -> List[str]:
        """Query projection(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.Q_PROJ)

    @property
    def k_proj(self) -> List[str]:
        """Key projection(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.K_PROJ)

    @property
    def v_proj(self) -> List[str]:
        """Value projection(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.V_PROJ)

    @property
    def gate_proj(self) -> List[str]:
        """Gate projection(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.GATE_PROJ)

    @property
    def up_proj(self) -> List[str]:
        """Up projection(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.UP_PROJ)


@dataclass
class MambaBlockTopology(BlockTopology):
    """Mamba mixer projections around the selective scan.

    :param mixer: Read group — the mixer's ``in_proj`` through the block's norm.
    :param mixer_out_proj: Name(s) of the mixer's ``out_proj`` writing to the residual.
    """

    kind: ClassVar[BlockKind] = BlockKind.MAMBA
    mixer: LinearGroup = field(default_factory=LinearGroup)
    mixer_out_proj: List[str] = field(default_factory=list)


@dataclass
class LlmTopology:
    """Topology of an ONNX decoder-stack model: blocks + backbone-level roles + dims.

    :param embed_tokens: Names of the token-embedding ``Gather``(s) that produce
        the initial residual-stream activations.
    :param lm_head: Names of the vocabulary-projection linear(s) downstream of
        the final norm.
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

    embed_tokens: List[str] = field(default_factory=list)
    lm_head: List[str] = field(default_factory=list)
    blocks: List[Union[AttentionBlockTopology, MambaBlockTopology]] = field(
        default_factory=list
    )
    past_key_input_names: List[str] = field(default_factory=list)
    past_key_output_names: List[str] = field(default_factory=list)
    past_value_input_names: List[str] = field(default_factory=list)
    past_value_output_names: List[str] = field(default_factory=list)
    active_norms: Optional[List[ActiveNorm]] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None


def split_by_role(
    linears: List[str],
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
) -> Dict[LinearRole, List[str]]:
    """Classify each name in ``linears`` by module name into its fine-grained role.

    :param linears: A read group (the ``qkv`` or ``gate_up`` group), by name.
    :param role_patterns: Optional override passed to :func:`classify_linear_role`.
    :return: A mapping from every :class:`LinearRole` to the (possibly empty)
        list of linears classified as that role. Entries that do not match any
        role are logged under :attr:`LinearRole.UNKNOWN` (they remain in the
        coarse group either way).
    """
    out: Dict[LinearRole, List[str]] = {role: [] for role in LinearRole}
    for linear in linears:
        role = classify_linear_role(linear, role_patterns)
        out[role].append(linear)
        if role is LinearRole.UNKNOWN:
            _logger.debug(
                "Linear '%s' did not match any known role pattern; left "
                "un-split (still present in its coarse read group).",
                linear,
            )
    return out


__all__ = [
    "AttentionBlockTopology",
    "BlockKind",
    "BlockTopology",
    "LinearGroup",
    "LlmTopology",
    "MambaBlockTopology",
    "split_by_role",
]
