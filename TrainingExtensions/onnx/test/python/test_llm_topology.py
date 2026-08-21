# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ``aimet_onnx.experimental.llm_topology`` package.

Three groups:

* **Role classification** — direct, table-driven tests for
  :func:`classify_linear_role` / :func:`module_name_of` (no ONNX export needed).
* **Block / role-map detection** — :class:`TestBlockIdentifier` and
  :class:`TestDecoderRoleMap`, exercising :func:`find_active_norms`,
  :func:`get_decoder_block_boundaries`, and :func:`get_llm_topology` on tiny
  hand-built decoders (relocated here from ``test_spinquant.py``).
* **End-to-end facade** — :class:`TestAnalyzeLlmTopology`, covering
  :func:`analyze_llm_topology`.

Broader coverage across real HuggingFace architectures lives in
``test_llm_topology_integration.py``.
"""

import re

import pytest
import torch
import torch.nn as nn

from aimet_onnx.meta.connectedgraph import ConnectedGraph

from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    tensor_to_first_consumer_index,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
    module_name_of,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    find_active_norms,
    get_last_norm_op,
)
from aimet_onnx.experimental.llm_topology import topology as topology_module
from aimet_onnx.experimental.llm_topology.topology import (
    analyze_llm_topology,
    get_llm_topology,
)
from aimet_onnx.experimental.llm_topology.weight_utils import (
    infer_hidden_size as _infer_hidden_size,
)

from .models.test_models import RMSNorm
from .models.style_decoders import (
    _NORM_KW,
    _H,
    _HEAD_DIM,
    _VOCAB,
    _SEQ,
    _export_decoder,
    _export_decoder_with_ids,
    _fuse_rms_norms,
    _LlamaBlock,
    LlamaStyleDecoder,
    Qwen3StyleDecoder,
    Phi3StyleDecoder,
    Gemma3StyleDecoder,
)

# Decoder flavors that must all yield the same backbone shape: 2 blocks,
# 2 active norms per block + 1 final norm, hidden_size _H, head_dim _HEAD_DIM.
# Each contributes a distinct wrinkle — unfused q/k/v (llama), internal
# q_norm/k_norm (qwen3), fused qkv/gate_up (phi3), post-writing norms (gemma3).
_DECODERS = [
    pytest.param(LlamaStyleDecoder, id="llama"),
    pytest.param(Qwen3StyleDecoder, id="qwen3"),
    pytest.param(Phi3StyleDecoder, id="phi3"),
    pytest.param(Gemma3StyleDecoder, id="gemma3"),
]


# ===========================================================================
# Role classification — direct, table-driven (no ONNX export).
# ===========================================================================
class _FakeOp:
    """Minimal stand-in for a ConnectedGraph ``Op`` with just a ``name``."""

    def __init__(self, name):
        self.name = name


class TestClassifyLinearRole:
    """Direct tests for the name-based role classifier and its helper."""

    @pytest.mark.parametrize(
        "module_name, expected_role",
        [
            # canonical HF names
            ("q_proj", LinearRole.Q_PROJ),
            ("k_proj", LinearRole.K_PROJ),
            ("v_proj", LinearRole.V_PROJ),
            ("o_proj", LinearRole.O_PROJ),
            ("gate_proj", LinearRole.GATE_PROJ),
            ("up_proj", LinearRole.UP_PROJ),
            ("down_proj", LinearRole.DOWN_PROJ),
            # short / alternate aliases
            ("q", LinearRole.Q_PROJ),
            ("query", LinearRole.Q_PROJ),
            ("out_proj", LinearRole.O_PROJ),
            ("dense", LinearRole.O_PROJ),
            ("wo", LinearRole.O_PROJ),
            ("w1", LinearRole.GATE_PROJ),
            ("w3", LinearRole.UP_PROJ),
            ("w2", LinearRole.DOWN_PROJ),
            # fused variants
            ("qkv_proj", LinearRole.FUSED_QKV),
            ("c_attn", LinearRole.FUSED_QKV),
            ("Wqkv", LinearRole.FUSED_QKV),
            ("in_proj", LinearRole.FUSED_QKV),
            ("gate_up_proj", LinearRole.FUSED_GATE_UP),
            ("gateup_proj", LinearRole.FUSED_GATE_UP),
            # unrecognized
            ("mlp_router", LinearRole.UNKNOWN),
            ("in_proj_z", LinearRole.UNKNOWN),
        ],
    )
    def test_canonical_names(self, module_name, expected_role):
        op = _FakeOp(f"/model/layers.0/self_attn/{module_name}/MatMul")
        assert classify_linear_role(op) is expected_role

    @pytest.mark.parametrize(
        "role_name", ["q_proj", "v_proj", "gate_proj", "down_proj"]
    )
    def test_per_head_sha_suffix(self, role_name):
        """Per-head split (SHA) names carry a ``_sha`` suffix and optional index."""
        role = classify_linear_role(_FakeOp(f"/m/attn/{role_name}_sha/MatMul"))
        role_indexed = classify_linear_role(
            _FakeOp(f"/m/attn/{role_name}_sha.3/MatMul")
        )
        assert role is role_indexed
        assert role is not LinearRole.UNKNOWN

    @pytest.mark.parametrize("fused_name", ["qkv_proj", "gate_up_proj"])
    def test_fused_names_have_no_sha_suffix(self, fused_name):
        """A fused projection is the opposite of a per-head split, so ``_sha`` on a
        fused name must NOT classify as fused (SHA implies the projection was split)."""
        assert (
            classify_linear_role(_FakeOp(f"/m/attn/{fused_name}_sha/MatMul"))
            is LinearRole.UNKNOWN
        )

    def test_fused_beats_single_projection(self):
        """``qkv_proj`` must resolve to FUSED_QKV, not Q_PROJ (priority order)."""
        assert (
            classify_linear_role(_FakeOp("/m/attn/qkv_proj/MatMul"))
            is LinearRole.FUSED_QKV
        )

    def test_custom_role_patterns_override(self):
        """A supplied ``role_patterns`` mapping replaces the default table; only the
        roles present are tested (the hook for exotic exports)."""
        patterns = {LinearRole.V_PROJ: re.compile(r"^value_layer$")}
        # Custom name matches the override.
        assert (
            classify_linear_role(_FakeOp("/m/attn/value_layer/MatMul"), patterns)
            is LinearRole.V_PROJ
        )
        # Default names no longer match, since only V_PROJ is in the override.
        assert (
            classify_linear_role(_FakeOp("/m/attn/q_proj/MatMul"), patterns)
            is LinearRole.UNKNOWN
        )

    def test_module_name_of(self):
        assert (
            module_name_of(_FakeOp("/model/layers.0/self_attn/v_proj/MatMul"))
            == "v_proj"
        )
        # Too few segments to carry a module name.
        assert module_name_of(_FakeOp("MatMul")) is None

    def test_unnamed_op_is_unknown(self):
        assert classify_linear_role(_FakeOp("MatMul")) is LinearRole.UNKNOWN


# ===========================================================================
# Block / active-norm detection.
# ===========================================================================
class TestBlockIdentifier:
    """Tests for find_active_norms and decoder block detection boundaries."""

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_active_norm_count(self, decoder_cls):
        """2 blocks × 2 active norms/block + 1 final norm → 5 active norms.

        Holds across all flavors: qwen3's internal q_norm/k_norm are excluded
        (9 norms total, 5 active), phi3's fused qkv/gate_up do not affect norm
        detection, and gemma3's post-writing norms are not counted as active.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)
        active_norms = find_active_norms(model, cg)
        assert len(active_norms) == 5

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_all_have_downstream_linears(self, decoder_cls):
        """Every returned ActiveNorm must expose at least one downstream weight linear.

        This is what makes a norm "active", and is why qwen3's q_norm/k_norm are
        excluded — their output feeds attention, not a weighted MatMul/Gemm/Conv.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)
        for active_norm in find_active_norms(model, cg):
            assert active_norm.downstream_linears

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_block_count(self, decoder_cls):
        """A 2-block decoder → 2 boundaries, for every flavor."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        assert len(blocks) == 2

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_boundaries_are_active_norm_ops(self, decoder_cls):
        """Every boundary tensor must be the residual input of an active norm op."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        active_norms = find_active_norms(model, cg)
        # Include the final norm's input — it bounds the last block.
        norm_input_tensors = {an.norm_op.inputs[0].name for an in active_norms}
        norm_input_tensors.add(get_last_norm_op(cg).inputs[0].name)
        for start_tensor, end_tensor in blocks:
            assert start_tensor in norm_input_tensors
            assert end_tensor in norm_input_tensors

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_boundaries_non_overlapping(self, decoder_cls):
        """end tensor of block i must equal start tensor of block i+1."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        for i in range(len(blocks) - 1):
            assert blocks[i][1] == blocks[i + 1][0]

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_boundary_tensor_resolves_to_norm_op(self, fuse_rmsnorm):
        """A boundary tensor must resolve to its norm op, not the residual Add
        that shares the same edge. Covers decomposed and fused RMSNorm.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)

        tensor_to_index = tensor_to_first_consumer_index(cg)
        norm_ops = {an.norm_op for an in find_active_norms(model, cg)}
        norm_ops.add(get_last_norm_op(cg))

        for start_tensor, end_tensor in blocks:
            for tensor in (start_tensor, end_tensor):
                resolved_op = cg.ordered_ops[tensor_to_index[tensor]]
                assert resolved_op in norm_ops, (
                    f"boundary tensor '{tensor}' resolved to '{resolved_op.name}' "
                    f"({resolved_op.type}), not a norm op."
                )

    def test_even_active_norms(self):
        """With no trailing norm, blocks are bounded by their residual adds."""
        torch.manual_seed(0)

        class _NoFinalNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()

            def forward(self, x):
                return self.block1(self.block0(x))

        model = _export_decoder(_NoFinalNorm())
        cg = ConnectedGraph(model)
        boundaries = get_decoder_block_boundaries(model, cg)

        assert len(boundaries) == 2
        # Boundaries chain, and the last block ends on the graph output.
        assert boundaries[0][1] == boundaries[1][0]
        assert boundaries[1][1] == model.graph.output[0].name


# ===========================================================================
# Role map (get_llm_topology).
# ===========================================================================
class TestDecoderRoleMap:
    """Tests for get_llm_topology.

    Each test parametrizes ``fuse_rmsnorm``: False keeps the decomposed RMSNorm pattern
    (ReduceMean / Sqrt / Mul chain), True coalesces it into a single ``RMSNormalization``
    supergroup op, mirroring what QuantizationSimModel does before constructing its
    ConnectedGraph. Both paths must produce identical role maps.
    """

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_llama_role_map_structure(self, fuse_rmsnorm):
        """LlamaStyleDecoder: verify per-block and model-level role counts.

        2 blocks × (3 qkv, 1 o_proj, 2 gate_up, 1 down_proj) + 1 lm_head + 1 embed_tokens.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        role_map = get_llm_topology(cg, blocks)

        assert len(role_map.blocks) == 2
        assert len(role_map.lm_head) == 1
        assert len(role_map.embed_tokens) == 1
        for block in role_map.blocks:
            assert len(block.qkv.ops) == 3
            assert len(block.o_proj) == 1
            assert len(block.gate_up.ops) == 2
            assert len(block.down_proj) == 1

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_qwen3_qkv_count(self, fuse_rmsnorm):
        """Qwen3 q_norm/k_norm are internal; qkv group still has 3 (q_proj, k_proj, v)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        role_map = get_llm_topology(cg, blocks)
        for block in role_map.blocks:
            assert len(block.qkv.ops) == 3

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_missing_embed_tokens_warns(self, fuse_rmsnorm):
        """A model without a Gather embedding (e.g. VLM backbone) warns, not raises."""
        torch.manual_seed(0)

        class _NoEmbedDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()
                self.norm = RMSNorm(_H, **_NORM_KW)
                self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

            def forward(self, x):
                x = self.block0(x)
                x = self.block1(x)
                return self.lm_head(self.norm(x))

        model = _export_decoder(_NoEmbedDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        # Must not raise — VLM backbones exported with use_inputs_embeds=True have no Gather.
        role_map = get_llm_topology(cg, blocks)
        assert role_map.embed_tokens == []

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_extra_prologue_gather_with_scalar_constant_excluded(self, fuse_rmsnorm):
        """Non-embedding Gather ops (e.g. position-id / shape-derived lookups) with
        scalar or 1-D static constants must not be admitted into ``role_map.embed_tokens``.

        Real ONNX exports (e.g. Qwen3-0.6B with rotary preprocessing) produce extra
        ``Gather(constant_table, dynamic_index)`` ops in the prologue whose static
        input is a small 1-D or scalar tensor — not a [vocab, hidden] embedding
        table. Those must be filtered out so ``infer_hidden_size`` doesn't read
        ``shape[-1]`` of a 0-/1-D tensor.
        """
        torch.manual_seed(0)

        class _DecoderWithPrologueGather(nn.Module):
            """LLaMA decoder + a non-embedding Gather over a 1-D constant in the prologue.

            The auxiliary Gather output participates in the model output, so ONNX
            constant-folding can't strip it. This mimics what real exports produce
            in the rotary / position-id preprocessing — Gathers over scalar / 1-D
            static constants that must not be confused with the embedding table.
            """

            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(_VOCAB, _H)
                # 1-D table that gets indexed dynamically; exports as a Gather
                # whose static data input is a 1-D constant — NOT an embedding.
                self.register_buffer(
                    "aux_table", torch.arange(_SEQ, dtype=torch.float32)
                )
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()
                self.norm = RMSNorm(_H, **_NORM_KW)
                self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

            def forward(self, token_ids):
                x = self.embed_tokens(token_ids)
                # Gather(aux_table, token_ids[:, 0]): static data is 1-D, dynamic index.
                # ids[:, 0] also produces a Gather(input, scalar_const_index).
                aux = self.aux_table[token_ids[:, 0]]  # [B]
                x = self.block0(x)
                x = self.block1(x)
                # Use aux in the output so constant folding can't remove the Gathers.
                return self.lm_head(self.norm(x)) + aux.unsqueeze(-1).unsqueeze(-1)

        model = _export_decoder_with_ids(_DecoderWithPrologueGather())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        role_map = get_llm_topology(cg, blocks)

        # Exactly one embed_tokens — the [vocab, hidden] embedding, not the 1-D Gather.
        assert len(role_map.embed_tokens) == 1
        # And infer_hidden_size doesn't IndexError on a scalar/1-D shape.
        assert _infer_hidden_size(model, role_map) == _H

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_wrong_active_norms_per_block_raises(self, fuse_rmsnorm):
        """Passing active_norms_per_block inconsistent with detected norms raises ValueError."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        with pytest.raises(ValueError):
            get_llm_topology(cg, blocks, active_norms_per_block=3)

    def test_topology_splits_v_projection(self):
        """Topology must identify the V projection (not Q or K) per block."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        topology = get_llm_topology(cg, blocks)

        for block in topology.blocks:
            assert len(block.v_proj) == 1
            assert "/v/" in block.v_proj[0].name
            # V must be split out of the coarse qkv read group, not duplicated.
            assert block.v_proj[0] in block.qkv.ops

    def test_topology_detects_fused_qkv(self):
        """Phi3-style fused QKV must classify as FUSED_QKV with no V split.

        R2 relies on this: a block with no ``v_proj`` (fused QKV) is rejected by
        ``R2RotationPass.validate`` because there is no per-head V path to rotate.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        topology = get_llm_topology(cg, blocks)

        for block in topology.blocks:
            assert not block.v_proj
            assert block.qkv.role(LinearRole.FUSED_QKV)

    def test_topology_splits_all_qkv_roles(self):
        """Unfused attention must split into distinct q/k/v ops within the qkv group."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        topology = get_llm_topology(cg, blocks)

        for block in topology.blocks:
            assert len(block.q_proj) == 1
            assert len(block.k_proj) == 1
            assert len(block.v_proj) == 1
            assert not block.qkv.role(LinearRole.FUSED_QKV)
            assert "/q/" in block.q_proj[0].name
            assert "/k/" in block.k_proj[0].name
            assert "/v/" in block.v_proj[0].name
            # The three splits together are exactly the coarse qkv read group.
            split = block.q_proj + block.k_proj + block.v_proj
            assert {op for op in split} == {op for op in block.qkv.ops}

    def test_topology_identifies_dynamic_attention_matmuls(self):
        """Each block must expose the two dynamic (non-weighted) attention MatMuls."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks = get_decoder_block_boundaries(model, cg)
        topology = get_llm_topology(cg, blocks)

        for block in topology.blocks:
            assert block.qk_matmul
            assert block.attn_v_matmul
            # Q·Kᵀ is distinct from softmax·V.
            assert set(block.qk_matmul).isdisjoint(set(block.attn_v_matmul))
            for m in block.qk_matmul + block.attn_v_matmul:
                assert m.type == "MatMul"


# ===========================================================================
# End-to-end facade (analyze_llm_topology).
# ===========================================================================
class TestAnalyzeLlmTopology:
    """Tests for the analyze_llm_topology one-shot facade."""

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_populates_dims_and_roles(self, decoder_cls):
        """The facade builds the CG itself and fills hidden_size / head_dim / active_norms."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        topology = analyze_llm_topology(model)

        assert len(topology.blocks) == 2
        assert len(topology.embed_tokens) == 1
        assert len(topology.lm_head) == 1
        assert topology.hidden_size == _H
        assert topology.head_dim == _HEAD_DIM
        # active_norms are retained (2 per block + final norm).
        assert topology.active_norms is not None
        assert len(topology.active_norms) == 5

    def test_reuses_supplied_connected_graph(self, monkeypatch):
        """A caller-supplied ConnectedGraph must be used as-is, never rebuilt.

        Asserting on dims/block counts alone would not catch a rebuild — the
        rebuilt graph produces the same values. So we (a) make construction fail
        loudly if attempted, and (b) check the returned ops are the very objects
        owned by the supplied graph.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "analyze_llm_topology rebuilt the ConnectedGraph instead of "
                "reusing the supplied one."
            )

        monkeypatch.setattr(topology_module, "ConnectedGraph", _fail)
        topology = analyze_llm_topology(model, connected_graph=cg)

        supplied_ops = set(cg.ordered_ops)
        returned_ops = [
            *topology.embed_tokens,
            *topology.lm_head,
            *(op for block in topology.blocks for op in block.o_proj),
        ]
        assert returned_ops
        assert all(op in supplied_ops for op in returned_ops)

    def test_head_dim_none_without_past_value_input(self):
        """No ``past_value`` graph input → head_dim tolerated as None (R1-only / prefill)."""
        torch.manual_seed(0)
        # _export_vlm_backbone-style: build a decoder WITHOUT the past_value input.
        from .models.style_decoders import _export_to_onnx

        model = _export_to_onnx(
            LlamaStyleDecoder(), torch.randint(0, _VOCAB, (1, _SEQ))
        )

        topology = analyze_llm_topology(model)

        assert topology.head_dim is None
        # hidden_size is still derivable from the embedding table.
        assert topology.hidden_size == _H

    def test_expected_num_blocks_mismatch_raises(self):
        """A wrong expected_num_blocks must be validated and raise."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        with pytest.raises(ValueError):
            analyze_llm_topology(model, expected_num_blocks=3)
