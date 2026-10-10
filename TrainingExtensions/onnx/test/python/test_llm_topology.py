# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ``aimet_onnx.llm_topology`` package.

Three groups:

* **Role classification** — direct, table-driven tests for
  :func:`classify_linear_role` / :func:`module_name_of` (no ONNX export needed).
* **Block / role-map detection** — :class:`TestBlockIdentifier` and
  :class:`TestDecoderRoleMap`, exercising :func:`find_active_norms`,
  :func:`get_decoder_block_boundaries`, and :func:`get_llm_topology` on tiny
  hand-built decoders (relocated here from ``test_spinquant.py``).
* **End-to-end facade** — :class:`TestAnalyzeLlmTopology`, covering
  :func:`analyze_llm_topology_by_norm_count` and the ``onnx_ir`` re-attachment in
  :func:`~.ir_adapter.resolve_topology`.

Broader coverage across real HuggingFace architectures lives in
``test_llm_topology_integration.py``.
"""

import copy
import dataclasses
import re

import numpy as np
import onnx_ir
import pytest
import torch
import torch.nn as nn

from aimet_onnx.quantsim import QuantizationSimModel

from aimet_onnx.llm_topology import ir_analysis
from aimet_onnx.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    get_decoder_block_boundaries_in_ir,
    resolve_residual_tensor_name,
)
from aimet_onnx.llm_topology.ir_adapter import (
    IrAttentionBlockTopology,
    IrMambaBlockTopology,
    resolve_topology,
)
from aimet_onnx.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
    module_name_of,
)
from aimet_onnx.llm_topology.norm_detection import (
    find_active_norms,
    find_active_norms_in_ir,
    get_last_norm_input_tensor,
)
from aimet_onnx.llm_topology.topology import (
    _PAST_KEY_INPUT_NAME_PATTERN,
    _PAST_KEY_OUTPUT_NAME_PATTERN,
    _PAST_VALUE_INPUT_NAME_PATTERN,
    _PAST_VALUE_OUTPUT_NAME_PATTERN,
    _collect_matching_names_in_order,
    _infer_hidden_size,
    analyze_llm_topology_by_norm_count,
    get_llm_topology,
)
from aimet_onnx.llm_topology.topology_types import (
    AttentionBlockTopology,
    BlockKind,
    MambaBlockTopology,
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
from .models.transformer_blocks import sha_gqa_decoder

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


def test_block_topologies_separate_common_and_kind_specific_fields():
    for attention_type, mamba_type in (
        (AttentionBlockTopology, MambaBlockTopology),
        (IrAttentionBlockTopology, IrMambaBlockTopology),
    ):
        common = {"residual_input", "residual_output"}
        attention = {field.name for field in dataclasses.fields(attention_type)}
        mamba = {field.name for field in dataclasses.fields(mamba_type)}
        assert attention & mamba == common
        assert {"qkv", "gate_up", "o_proj", "down_proj"} <= attention
        assert {"mixer", "mixer_out_proj"} <= mamba
        assert attention_type().kind is BlockKind.ATTENTION
        assert mamba_type().kind is BlockKind.MAMBA
        with pytest.raises(TypeError, match="kind"):
            mamba_type(kind=BlockKind.ATTENTION)


def _name_topology(model, **kwargs):
    """Build the name-based topology for ``model``, the way the facade does.

    Mirrors :func:`analyze_llm_topology_by_norm_count` but stops before dimension
    inference and exposes ``get_llm_topology``'s knobs, so tests can drive that
    function directly.
    """
    ir_model = ir_analysis.build_analysis_ir(model)
    topo_index = ir_analysis.topological_index(ir_model)
    active_norms = find_active_norms_in_ir(ir_model, topo_index)
    boundaries = get_decoder_block_boundaries_in_ir(
        ir_model, active_norms=active_norms, topo_index=topo_index
    )
    return get_llm_topology(
        ir_model,
        boundaries,
        active_norms=active_norms,
        topo_index=topo_index,
        **kwargs,
    )


# ===========================================================================
# Role classification — direct, table-driven (no ONNX export).
# ===========================================================================
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
        node_name = f"/model/layers.0/self_attn/{module_name}/MatMul"
        assert classify_linear_role(node_name) is expected_role

    @pytest.mark.parametrize(
        "role_name", ["q_proj", "v_proj", "gate_proj", "down_proj"]
    )
    def test_per_head_sha_suffix(self, role_name):
        """Per-head split (SHA) names carry a ``_sha`` suffix and optional index."""
        role = classify_linear_role(f"/m/attn/{role_name}_sha/MatMul")
        role_indexed = classify_linear_role(f"/m/attn/{role_name}_sha.3/MatMul")
        assert role is role_indexed
        assert role is not LinearRole.UNKNOWN

    @pytest.mark.parametrize("fused_name", ["qkv_proj", "gate_up_proj"])
    def test_fused_names_have_no_sha_suffix(self, fused_name):
        """A fused projection is the opposite of a per-head split, so ``_sha`` on a
        fused name must NOT classify as fused (SHA implies the projection was split)."""
        assert (
            classify_linear_role(f"/m/attn/{fused_name}_sha/MatMul")
            is LinearRole.UNKNOWN
        )

    def test_fused_beats_single_projection(self):
        """``qkv_proj`` must resolve to FUSED_QKV, not Q_PROJ (priority order)."""
        assert classify_linear_role("/m/attn/qkv_proj/MatMul") is LinearRole.FUSED_QKV

    def test_custom_role_patterns_override(self):
        """A supplied ``role_patterns`` mapping replaces the default table; only the
        roles present are tested (the hook for exotic exports)."""
        patterns = {LinearRole.V_PROJ: re.compile(r"^value_layer$")}
        # Custom name matches the override.
        assert (
            classify_linear_role("/m/attn/value_layer/MatMul", patterns)
            is LinearRole.V_PROJ
        )
        # Default names no longer match, since only V_PROJ is in the override.
        assert (
            classify_linear_role("/m/attn/q_proj/MatMul", patterns)
            is LinearRole.UNKNOWN
        )

    def test_module_name_of(self):
        assert module_name_of("/model/layers.0/self_attn/v_proj/MatMul") == "v_proj"
        # Too few segments to carry a module name.
        assert module_name_of("MatMul") is None

    def test_unnamed_op_is_unknown(self):
        assert classify_linear_role("MatMul") is LinearRole.UNKNOWN


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
        active_norms = find_active_norms(model)
        assert len(active_norms) == 5

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_all_have_downstream_linears(self, decoder_cls):
        """Every returned ActiveNorm must expose at least one downstream weight linear.

        This is what makes a norm "active", and is why qwen3's q_norm/k_norm are
        excluded — their output feeds attention, not a weighted MatMul/Gemm/Conv.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        for active_norm in find_active_norms(model):
            assert active_norm.downstream_linears

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_block_count(self, decoder_cls):
        """A 2-block decoder → 2 boundaries, for every flavor."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        blocks = get_decoder_block_boundaries(model)
        assert len(blocks) == 2

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_boundaries_are_active_norm_ops(self, decoder_cls):
        """Every boundary tensor must be the residual input of an active norm op."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        blocks = get_decoder_block_boundaries(model)
        active_norms = find_active_norms(model)
        # Include the final norm's input — it bounds the last block.
        norm_input_tensors = {an.input_tensor for an in active_norms}
        norm_input_tensors.add(get_last_norm_input_tensor(model))
        for start_tensor, end_tensor in blocks:
            assert start_tensor in norm_input_tensors
            assert end_tensor in norm_input_tensors

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_boundaries_non_overlapping(self, decoder_cls):
        """end tensor of block i must equal start tensor of block i+1."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        blocks = get_decoder_block_boundaries(model)
        for i in range(len(blocks) - 1):
            assert blocks[i][1] == blocks[i + 1][0]

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_boundary_tensor_resolves_to_norm_op(self, fuse_rmsnorm):
        """A boundary tensor must resolve to its norm op, not the residual Add
        that shares the same edge. Covers decomposed and fused RMSNorm.

        This is the invariant ``tensor_to_first_consumer_index`` documents: the
        block's norm is the *first* consumer of the residual edge.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        blocks = get_decoder_block_boundaries(model)

        ir_model = ir_analysis.build_analysis_ir(model)
        topo_index = ir_analysis.topological_index(ir_model)
        tensor_to_index = ir_analysis.tensor_to_first_consumer_index(
            ir_model, topo_index
        )
        nodes = tuple(ir_model.graph)

        for start_tensor, end_tensor in blocks:
            for tensor in (start_tensor, end_tensor):
                resolved = nodes[tensor_to_index[tensor]]
                assert ir_analysis.is_rms_norm(resolved), (
                    f"boundary tensor '{tensor}' resolved to "
                    f"'{ir_analysis.node_name(resolved)}' ({resolved.op_type}), "
                    "not a norm op."
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
        boundaries = get_decoder_block_boundaries(model)

        assert len(boundaries) == 2
        # Boundaries chain, and the last block ends on the graph output.
        assert boundaries[0][1] == boundaries[1][0]
        assert boundaries[1][1] == model.graph.output[0].name


# ===========================================================================
# Residual-name resolution (resolve_residual_tensor_name).
# ===========================================================================
class TestResolveResidualTensorName:
    """Tests for resolve_residual_tensor_name.

    A boundary tensor names the value entering a norm; in fp16 exports a ``Cast``
    sits between the residual ``Add`` and that norm, so a consumer slicing the
    graph at a boundary must first step back to the pre-``Cast`` residual.
    (Relocated here from ``test_adascale.py`` — AdaScale is a caller, not the
    owner, of this structural detail.)
    """

    @staticmethod
    def _graph(text: str) -> onnx_ir.Graph:
        return onnx_ir.from_onnx_text(text).graph

    def test_resolve_walks_past_leading_cast(self):
        """fp16 pattern: RMSNorm anchor's input is post-Cast; the true residual is upstream."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float16[N, 4] residual) => (float[N, 4] anchor_in) {
                anchor_in = Cast<to=1>(residual)
            }
            """
        )
        assert resolve_residual_tensor_name(graph, "anchor_in") == "residual"

    def test_resolve_walks_past_chained_casts(self):
        """Multiple Casts in a row (e.g., fp16 -> fp32 -> fp16) all get stripped."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float16[N, 4] residual) => (float16[N, 4] anchor_in) {
                mid = Cast<to=1>(residual)
                anchor_in = Cast<to=10>(mid)
            }
            """
        )
        assert resolve_residual_tensor_name(graph, "anchor_in") == "residual"

    def test_resolve_stops_at_non_cast_producer(self):
        """Non-Cast producer (Add) is not stepped through — the anchor is the true residual."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] a, float[N, 4] b) => (float[N, 4] residual) {
                residual = Add(a, b)
            }
            """
        )
        assert resolve_residual_tensor_name(graph, "residual") == "residual"

    def test_resolve_unknown_name_returns_input_unchanged(self):
        """Names not in the graph pass through — this is the fallback path for
        callers that may pass an already-resolved name."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] x) => (float[N, 4] y) {
                y = Identity(x)
            }
            """
        )
        assert resolve_residual_tensor_name(graph, "not_in_graph") == "not_in_graph"


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
        role_map = _name_topology(model)

        assert len(role_map.blocks) == 2
        assert len(role_map.lm_head) == 1
        assert len(role_map.embed_tokens) == 1
        for block in role_map.blocks:
            assert len(block.qkv.linears) == 3
            assert len(block.o_proj) == 1
            assert len(block.gate_up.linears) == 2
            assert len(block.down_proj) == 1

    @pytest.mark.parametrize(
        "pattern,names,expected",
        [
            (
                _PAST_KEY_INPUT_NAME_PATTERN,
                [
                    "input_ids",
                    "past_key_0_in",
                    "past_value_0_in",
                    "past_k_1_in",
                    "past_key_values.2.key",
                    "past_key_values.2.value",
                    "past_key",
                    "past_k",
                    "past_key_0",
                    "_past_k_tensor",
                    "past_key_values.x.key",
                    "past_key_values.2.key_extra",
                ],
                [
                    "past_key_0_in",
                    "past_k_1_in",
                    "past_key_values.2.key",
                    "past_key_0",
                ],
            ),
            (
                _PAST_VALUE_INPUT_NAME_PATTERN,
                [
                    "input_ids",
                    "past_key_0_in",
                    "past_value_0_in",
                    "past_v_1_in",
                    "past_key_values.2.key",
                    "past_key_values.2.value",
                    "past_value",
                    "past_v",
                    "past_value_0",
                    "_past_v_tensor",
                    "past_key_values.x.value",
                    "past_key_values.2.value_extra",
                ],
                [
                    "past_value_0_in",
                    "past_v_1_in",
                    "past_key_values.2.value",
                    "past_value_0",
                ],
            ),
            (
                _PAST_KEY_OUTPUT_NAME_PATTERN,
                [
                    "logits",
                    "past_key_0_out",
                    "past_value_0_out",
                    "past_k_1_out",
                    "past_key_2",
                    "present_key_1",
                    "present_value_1",
                    "present.2.key",
                    "present.2.value",
                    "present_key_values.3.key",
                    "present_key_values.3.value",
                    "yyy_present_k_xyz",
                    "present.xx.yy.key",
                    "present_xx_yy_key",
                    "present123.key",
                    "present_keyvalues.4.key",
                ],
                [
                    "past_key_0_out",
                    "past_k_1_out",
                    "past_key_2",
                    "present_key_1",
                    "present.2.key",
                ],
            ),
            (
                _PAST_VALUE_OUTPUT_NAME_PATTERN,
                [
                    "logits",
                    "past_key_0_out",
                    "past_value_0_out",
                    "past_v_1_out",
                    "past_value_2",
                    "present_key_1",
                    "present_value_1",
                    "present.2.key",
                    "present.2.value",
                    "present_key_values.3.key",
                    "present_key_values.3.value",
                    "yyy_present_v_x",
                    "present.xx.yy.value",
                    "present_xx_yy_value",
                    "present123.value",
                    "present_keyvalues.4.value",
                ],
                [
                    "past_value_0_out",
                    "past_v_1_out",
                    "past_value_2",
                    "present_value_1",
                    "present.2.value",
                ],
            ),
        ],
    )
    def test_kv_cache_name_patterns(self, pattern, names, expected):
        """KV-cache I/O variants are classified without losing declaration order."""

        values = [onnx_ir.Value(name=name) for name in names]

        assert _collect_matching_names_in_order(values, pattern) == expected

    def test_collects_all_kv_cache_inputs_and_outputs(self):
        """The topology populates every KV-cache field in declaration order."""
        model = sha_gqa_decoder(num_layers=2)

        topology = _name_topology(model)
        resolved = resolve_topology(topology, onnx_ir.from_proto(model))

        assert topology.past_key_input_names == [
            "past_key_0_in",
            "past_key_1_in",
        ]
        assert topology.past_value_input_names == [
            "past_value_0_in",
            "past_value_1_in",
        ]
        assert topology.past_key_output_names == [
            "past_key_0_out",
            "past_key_1_out",
        ]
        assert topology.past_value_output_names == [
            "past_value_0_out",
            "past_value_1_out",
        ]
        assert resolved.past_key_input_names == topology.past_key_input_names
        assert resolved.past_value_input_names == topology.past_value_input_names
        assert resolved.past_key_output_names == topology.past_key_output_names
        assert resolved.past_value_output_names == topology.past_value_output_names

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_qwen3_qkv_count(self, fuse_rmsnorm):
        """Qwen3 q_norm/k_norm are internal; qkv group still has 3 (q_proj, k_proj, v)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        role_map = _name_topology(model)
        for block in role_map.blocks:
            assert len(block.qkv.linears) == 3

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
        # Must not raise — VLM backbones exported with use_inputs_embeds=True have no Gather.
        role_map = _name_topology(model)
        assert role_map.embed_tokens == []

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_extra_prologue_gather_with_scalar_constant_excluded(self, fuse_rmsnorm):
        """Non-embedding Gather ops (e.g. position-id / shape-derived lookups) with
        scalar or 1-D static constants must not be admitted into ``role_map.embed_tokens``.

        Real ONNX exports (e.g. Qwen3-0.6B with rotary preprocessing) produce extra
        ``Gather(constant_table, dynamic_index)`` ops in the prologue whose static
        input is a small 1-D or scalar tensor — not a [vocab, hidden] embedding
        table. Those must be filtered out so ``_infer_hidden_size`` doesn't read
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
        role_map = _name_topology(model)

        # Exactly one embed_tokens — the [vocab, hidden] embedding, not the 1-D Gather.
        assert len(role_map.embed_tokens) == 1
        # And _infer_hidden_size doesn't IndexError on a scalar/1-D shape.
        ir_model = ir_analysis.build_analysis_ir(model)
        assert _infer_hidden_size(ir_model, role_map) == _H

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_wrong_active_norms_per_block_raises(self, fuse_rmsnorm):
        """Passing active_norms_per_block inconsistent with detected norms raises ValueError."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        with pytest.raises(ValueError):
            _name_topology(model, active_norms_per_block=3)

    def test_topology_splits_v_projection(self):
        """Topology must identify the V projection (not Q or K) per block."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        topology = _name_topology(model)

        for block in topology.blocks:
            assert len(block.v_proj) == 1
            assert "/v/" in block.v_proj[0]
            # V must be split out of the coarse qkv read group, not duplicated.
            assert block.v_proj[0] in block.qkv.linears

    def test_topology_detects_fused_qkv(self):
        """Phi3-style fused QKV must classify as FUSED_QKV with no V split.

        R2 relies on this: a block with no ``v_proj`` (fused QKV) is rejected by
        ``R2RotationPass.validate`` because there is no per-head V path to rotate.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())
        topology = _name_topology(model)

        for block in topology.blocks:
            assert not block.v_proj
            assert block.qkv.role(LinearRole.FUSED_QKV)

    def test_topology_splits_all_qkv_roles(self):
        """Unfused attention must split into distinct q/k/v ops within the qkv group."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        topology = _name_topology(model)

        for block in topology.blocks:
            assert len(block.q_proj) == 1
            assert len(block.k_proj) == 1
            assert len(block.v_proj) == 1
            assert not block.qkv.role(LinearRole.FUSED_QKV)
            assert "/q/" in block.q_proj[0]
            assert "/k/" in block.k_proj[0]
            assert "/v/" in block.v_proj[0]
            # The three splits together are exactly the coarse qkv read group.
            split = block.q_proj + block.k_proj + block.v_proj
            assert {op for op in split} == {op for op in block.qkv.linears}

    def test_topology_identifies_dynamic_attention_matmuls(self):
        """Each block must expose the two dynamic (non-weighted) attention MatMuls."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        topology = _name_topology(model)

        for block in topology.blocks:
            assert block.qk_matmul
            assert block.attn_v_matmul
            # Q·Kᵀ is distinct from softmax·V.
            assert set(block.qk_matmul).isdisjoint(set(block.attn_v_matmul))
            # Reported as plain node names, which must resolve to real MatMuls.
            matmul_names = {
                node.name
                for node in ir_analysis.build_analysis_ir(model).graph
                if node.op_type == "MatMul"
            }
            for name in block.qk_matmul + block.attn_v_matmul:
                assert name in matmul_names


# ===========================================================================
# End-to-end facade (analyze_llm_topology_by_norm_count).
# ===========================================================================
class TestAnalyzeLlmTopology:
    """Tests for the analyze_llm_topology_by_norm_count one-shot facade."""

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_populates_dims_and_roles(self, decoder_cls):
        """The facade fills hidden_size / head_dim / active_norms."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        topology = analyze_llm_topology_by_norm_count(model)

        assert len(topology.blocks) == 2
        assert len(topology.embed_tokens) == 1
        assert len(topology.lm_head) == 1
        assert topology.hidden_size == _H
        assert topology.head_dim == _HEAD_DIM
        # active_norms are retained (2 per block + final norm).
        assert topology.active_norms is not None
        assert len(topology.active_norms) == 5

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_block_residuals_match_get_decoder_block_boundaries(self, decoder_cls):
        """A topology's per-block residual names ARE the block boundaries.

        Consumers that used to call ``get_decoder_block_boundaries`` directly (e.g.
        AdaScale) now read ``blocks[i].residual_input`` / ``.residual_output``
        instead. That substitution is only valid while the two agree exactly —
        same pairs, same order — so pin it here rather than in the consumer.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        topology = analyze_llm_topology_by_norm_count(model)
        from_topology = [
            (block.residual_input, block.residual_output) for block in topology.blocks
        ]

        assert from_topology == get_decoder_block_boundaries(model)

    def test_head_dim_none_without_past_value_input(self):
        """No ``past_value`` graph input → head_dim tolerated as None (R1-only / prefill)."""
        torch.manual_seed(0)
        # _export_vlm_backbone-style: build a decoder WITHOUT the past_value input.
        from .models.style_decoders import _export_to_onnx

        model = _export_to_onnx(
            LlamaStyleDecoder(), torch.randint(0, _VOCAB, (1, _SEQ))
        )

        topology = analyze_llm_topology_by_norm_count(model)

        assert topology.head_dim is None
        # hidden_size is still derivable from the embedding table.
        assert topology.hidden_size == _H

    def test_expected_num_blocks_mismatch_raises(self):
        """A wrong expected_num_blocks must be validated and raise."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        with pytest.raises(ValueError):
            analyze_llm_topology_by_norm_count(model, expected_num_blocks=3)

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_ir_resolution_matches_name_topology(self, decoder_cls):
        """Re-attaching an IR model must not change what the topology says.

        The name-based analysis is the source of truth; ``resolve_topology`` only
        swaps names for ``onnx_ir`` handles. Every role must therefore come back
        with the same members, in the same order, under both flavors.

        Note the analysis runs on the *fused* analysis IR while resolution targets
        the *faithful* one built here — that asymmetry is exactly what SpinQuant
        relies on, so it is what this test exercises.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        by_name = analyze_llm_topology_by_norm_count(model)
        resolved = resolve_topology(by_name, onnx_ir.from_proto(model))

        assert by_name.embed_tokens == [n.name for n in resolved.embed_tokens]
        assert by_name.lm_head == [n.name for n in resolved.lm_head]
        assert by_name.hidden_size == resolved.hidden_size
        assert by_name.head_dim == resolved.head_dim
        assert by_name.past_key_input_names == resolved.past_key_input_names
        assert by_name.past_key_output_names == resolved.past_key_output_names
        assert by_name.past_value_input_names == resolved.past_value_input_names
        assert by_name.past_value_output_names == resolved.past_value_output_names
        assert [an.scale_name for an in by_name.active_norms] == [
            an.scale_name for an in resolved.active_norms
        ]
        assert len(by_name.blocks) == len(resolved.blocks)
        for name_block, ir_block in zip(by_name.blocks, resolved.blocks):
            assert isinstance(name_block, AttentionBlockTopology)
            assert isinstance(ir_block, IrAttentionBlockTopology)
            assert not hasattr(ir_block, "mixer")
            assert name_block.qkv.linears == [n.name for n in ir_block.qkv.nodes]
            assert name_block.gate_up.linears == [
                n.name for n in ir_block.gate_up.nodes
            ]
            assert name_block.o_proj == [n.name for n in ir_block.o_proj]
            assert name_block.down_proj == [n.name for n in ir_block.down_proj]
            assert name_block.qk_matmul == [n.name for n in ir_block.qk_matmul]
            assert name_block.attn_v_matmul == [n.name for n in ir_block.attn_v_matmul]
            assert name_block.residual_input == ir_block.residual_input.name
            assert name_block.residual_output == ir_block.residual_output.name
            # The role split survives resolution as a partition of the coarse group.
            assert name_block.v_proj == [n.name for n in ir_block.v_proj]


# ===========================================================================
# Analysis IR: the graph the analyzers actually run on.
# ===========================================================================
class TestAnalysisIr:
    """Tests for the private onnx_ir view the analyzers are built on."""

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_input_proto_is_not_mutated(self, decoder_cls):
        """Analysis must leave the caller's ModelProto byte-for-byte untouched.

        ``apply_spinquant`` analyzes the float model and then rewrites *that*
        proto's weights, so quantizer stripping and RMSNorm fusion must happen
        only on the private copy.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        before = model.SerializeToString()

        analyze_llm_topology_by_norm_count(model)

        assert model.SerializeToString() == before

    def test_boundaries_match_between_float_and_quantsim(self):
        """A sim graph must yield the same boundary tensors as the float graph.

        ``sim.model.model`` interleaves ``QcQuantizeOp`` everywhere and renames
        every consumed tensor ``T`` -> ``T_updated``. Callers (adascale,
        model_converter) feed the returned names back to the *quantizer-free*
        float graph, so the analysis has to report the un-suffixed names — which
        is what stripping quantizers off the private copy buys.
        """
        torch.manual_seed(0)
        float_model = _export_decoder_with_ids(LlamaStyleDecoder())
        dummy_input = {"input": np.zeros((1, _SEQ), dtype=np.int64)}

        float_boundaries = get_decoder_block_boundaries(float_model)
        sim = QuantizationSimModel(copy.deepcopy(float_model), dummy_input=dummy_input)
        sim_boundaries = get_decoder_block_boundaries(sim.model.model)

        assert sim_boundaries == float_boundaries
        # And the names are the float graph's, not quantizer outputs.
        float_tensors = {out for node in float_model.graph.node for out in node.output}
        for start_tensor, end_tensor in sim_boundaries:
            assert start_tensor in float_tensors or start_tensor in {
                inp.name for inp in float_model.graph.input
            }
            assert end_tensor in float_tensors

    def test_quantsim_active_norms_expose_downstream_linears(self):
        """Stripping quantizers must restore the norm -> linear edges a sim hides.

        Without it every static weight sits behind a ``QcQuantizeOp``, so no
        linear looks weighted and every norm is discarded as inactive.
        """
        torch.manual_seed(0)
        float_model = _export_decoder_with_ids(LlamaStyleDecoder())
        sim = QuantizationSimModel(
            copy.deepcopy(float_model),
            dummy_input={"input": np.zeros((1, _SEQ), dtype=np.int64)},
        )

        float_norms = find_active_norms(float_model)
        sim_norms = find_active_norms(sim.model.model)

        assert len(sim_norms) == len(float_norms) == 5
        for float_norm, sim_norm in zip(float_norms, sim_norms):
            assert sim_norm.input_tensor == float_norm.input_tensor
            assert sim_norm.scale_name == float_norm.scale_name
            assert sim_norm.downstream_linears == float_norm.downstream_linears

    @pytest.mark.parametrize("decoder_cls", _DECODERS)
    def test_prefused_model_analyzes_identically(self, decoder_cls):
        """Analyzing an already-fused model must match analyzing the decomposed one.

        Re-fusing a fused graph is a no-op, so the two must agree — this is what
        lets a caller pass either a raw export or a QuantizationSimModel graph.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        decomposed = analyze_llm_topology_by_norm_count(model)
        prefused = analyze_llm_topology_by_norm_count(_fuse_rms_norms(model))

        assert [an.scale_name for an in prefused.active_norms] == [
            an.scale_name for an in decomposed.active_norms
        ]
        assert prefused.lm_head == decomposed.lm_head
        assert len(prefused.blocks) == len(decomposed.blocks)
        for fused_block, plain_block in zip(prefused.blocks, decomposed.blocks):
            assert fused_block.qkv.linears == plain_block.qkv.linears
            assert fused_block.residual_input == plain_block.residual_input
