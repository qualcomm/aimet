# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for ``aimet_onnx.llm_topology`` on real models.

Exports real HuggingFace decoder / VLM architectures through the actual
transformers -> ONNX pipeline (both torchscript and dynamo backends) and asserts
block-boundary detection and role-map construction across the architecture
matrix. Unit-level logic tests on tiny hand-built decoders live in
``test_llm_topology.py``.
"""

import contextlib
import io
import re
import shutil
import pytest
import onnx
import torch

from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64(
    "transformers and onnx_sim is not available on Windows ARM64"
)

import transformers
from transformers import AutoModelForCausalLM
import transformers.masking_utils as mu

from aimet_onnx.ir_utils import static_tensor
from aimet_onnx.llm_topology import ir_analysis
from aimet_onnx.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    get_decoder_block_boundaries_in_ir,
)
from aimet_onnx.llm_topology.norm_detection import (
    find_active_norms_in_ir,
)
from aimet_onnx.llm_topology.layer_roles import LinearRole
from aimet_onnx.llm_topology.topology import get_llm_topology
from .utils import add_genai_tests_path, force_random_weight_init

_NUM_LAYERS = 2
_BOTH = ("torchscript", "dynamo")
_TS_ONLY = ("torchscript",)
_DYNAMO_ONLY = ("dynamo",)

# (test id, transformers config-class, export backends, detect kwargs, homogeneous).
# - Backends differ because some architectures only export under one exporter.
# - homogeneous=True means every decoder block has the same number of weighted
#   linear (MatMul/Gemm/Conv) ops; False for architectures whose layers genuinely
#   differ (MoE dense-replace / shared experts, Mamba-hybrid, per-layer attention
#   variation) so the uniformity assertion is skipped.
_BLOCK_TOPOLOGY_MODELS = [
    # dense pre-norm
    ("llama", "LlamaConfig", _BOTH, {}, True),
    ("qwen2", "Qwen2Config", _BOTH, {}, True),
    ("mistral", "MistralConfig", _BOTH, {}, True),
    ("ministral", "MinistralConfig", _BOTH, {}, True),
    ("ministral3", "Ministral3Config", _BOTH, {}, True),
    ("phi3", "Phi3Config", _BOTH, {}, True),
    ("gemma", "GemmaConfig", _BOTH, {}, True),
    ("granite", "GraniteConfig", _BOTH, {}, True),
    ("smollm3", "SmolLM3Config", _BOTH, {}, True),
    ("arcee", "ArceeConfig", _BOTH, {}, True),
    ("seed_oss", "SeedOssConfig", _BOTH, {}, True),
    # hybrid internal-norm
    ("qwen3", "Qwen3Config", _BOTH, {}, True),
    ("gemma2", "Gemma2Config", _BOTH, {}, True),
    ("gemma3_text", "Gemma3TextConfig", _BOTH, {}, True),
    ("gemma4_text", "Gemma4TextConfig", _BOTH, {}, False),
    # hybrid linear-attention
    ("lfm2", "Lfm2Config", _BOTH, {}, True),
    # parallel attn+MLP: single norm per block -> active_norms_per_block=1
    ("cohere", "CohereConfig", _DYNAMO_ONLY, {"active_norms_per_block": 1}, True),
    ("cohere2", "Cohere2Config", _DYNAMO_ONLY, {"active_norms_per_block": 1}, True),
    # dynamo-only dense
    ("glm4", "Glm4Config", _DYNAMO_ONLY, {}, True),
    ("ernie4_5", "Ernie4_5Config", _DYNAMO_ONLY, {}, True),
    # MoE
    ("qwen2_moe", "Qwen2MoeConfig", _TS_ONLY, {}, False),
    ("qwen3_moe", "Qwen3MoeConfig", _TS_ONLY, {}, True),
    ("glm4_moe", "Glm4MoeConfig", _TS_ONLY, {}, False),
    ("gpt_oss", "GptOssConfig", _TS_ONLY, {}, False),
    ("olmoe", "OlmoeConfig", _TS_ONLY, {}, True),
    ("nemotron_h", "NemotronHConfig", _TS_ONLY, {}, False),
    ("qwen3_next", "Qwen3NextConfig", _TS_ONLY, {}, True),
]

# VLM language backbones (text decoder + lm_head extracted from the full VLM).
_VLM_BACKBONE_MODELS = ["llava", "internvl", "qwen2_5_vl", "qwen3_vl"]


@contextlib.contextmanager
def _patch_sdpa_mask():
    """Make HF mask construction traceable under torch.jit.trace."""
    orig = mu.sdpa_mask
    orig_registered = mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def patched(*args, **kwargs):
        if (
            "q_length" in kwargs
            and isinstance(kwargs["q_length"], torch.Tensor)
            and kwargs["q_length"].ndim == 0
        ):
            kwargs["q_length"] = kwargs["q_length"].item()
        elif len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].ndim == 0:
            args = (args[0], args[1].item(), *args[2:])
        return orig(*args, **kwargs)

    try:
        mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = patched
        mu.sdpa_mask = patched
        yield
    finally:
        mu.sdpa_mask = orig
        mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = orig_registered


def _make_config(config_attr, **extra):
    """Build a tiny decoder config, or skip if the architecture is unavailable."""
    cfg_cls = getattr(transformers, config_attr, None)
    if cfg_cls is None:
        pytest.skip(f"{config_attr} not available in this transformers version")

    cfg = cfg_cls(**extra)
    overrides = dict(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1024,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # MoE knobs (best-effort; ignored by dense configs)
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        decoder_sparse_step=1,
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except (AttributeError, TypeError):
                pass
    return cfg


class _PrefillWrapper(torch.nn.Module):
    """input_ids -> logits prefill wrapper for ONNX export.

    NOTE: ``use_cache=False`` keeps the graph to a prefill pass,
    block detection needs no KV-cache I/O.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def _build_block_topology_model(config_attr):
    """Instantiate ``AutoModelForCausalLM`` with random, distinct norm weights."""
    cfg = _make_config(config_attr)
    torch.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_config(
            cfg, experts_implementation="eager"
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) * 0.5 + 0.75)
    return _PrefillWrapper(model).eval(), cfg


def _sample_inputs(cfg, seq_len=8):
    """Build dummy (input_ids, position_ids, attention_mask) for a prefill pass."""
    input_ids = torch.randint(3, cfg.vocab_size, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return input_ids, position_ids, attention_mask


def _build_vlm(name):
    """Build a tiny full VLM (text + vision)."""
    if name == "llava":
        text = _make_config("LlamaConfig")
        vision = transformers.CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=_NUM_LAYERS,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        cfg = transformers.LlavaConfig(text_config=text, vision_config=vision)
        return transformers.LlavaForConditionalGeneration(cfg).eval()
    if name == "internvl":
        text = _make_config("Qwen2Config")
        vision = transformers.InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=_NUM_LAYERS,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        cfg = transformers.InternVLConfig(text_config=text, vision_config=vision)
        return transformers.InternVLForConditionalGeneration(cfg).eval()
    if name == "qwen2_5_vl":
        text = _make_config(
            "Qwen2_5_VLTextConfig",
            rope_scaling={"rope_type": "mrope", "mrope_section": [2, 2, 4]},
        )
        vision = transformers.Qwen2_5_VLVisionConfig(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4
        )
        cfg = transformers.Qwen2_5_VLConfig(text_config=text, vision_config=vision)
        return transformers.Qwen2_5_VLForConditionalGeneration(cfg).eval()
    if name == "qwen3_vl":
        text = _make_config(
            "Qwen3VLTextConfig",
            rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 4]},
        )
        vision = transformers.Qwen3VLVisionConfig(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4
        )
        cfg = transformers.Qwen3VLConfig(text_config=text, vision_config=vision)
        return transformers.Qwen3VLForConditionalGeneration(cfg).eval()
    raise ValueError(f"unknown VLM '{name}'")


class _BackboneEmbedsWrapper(torch.nn.Module):
    """inputs_embeds -> logits prefill wrapper for a VLM language backbone."""

    def __init__(self, language_model, lm_head):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.lm_head(hidden)


def _build_vlm_backbone(name):
    """Build a VLM, extract its language backbone + lm_head, randomize norms."""
    torch.manual_seed(0)
    vlm = _build_vlm(name)
    language_model = vlm.model.language_model
    with torch.no_grad():
        for pname, param in language_model.named_parameters():
            if pname.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) * 0.5 + 0.75)
    return _BackboneEmbedsWrapper(
        language_model, vlm.lm_head
    ).eval(), language_model.config


def _vlm_sample_inputs(cfg, seq_len=8):
    """VLM backbones consume inputs_embeds (no token-embedding Gather)."""
    inputs_embeds = torch.randn(1, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return inputs_embeds, position_ids, attention_mask


def _export_onnx(model, inputs, backend, input_names=None):
    """Export the wrapped model to ONNX via the torchscript or dynamo backend."""
    if input_names is None:
        input_names = ["input_ids", "position_ids", "attention_mask"]
    buf = io.BytesIO()
    with _patch_sdpa_mask(), torch.no_grad():
        if backend == "dynamo":
            program = torch.export.draft_export(model, inputs, strict=False)
            torch.onnx.export(program, (), buf, input_names=input_names, dynamo=True)
        else:
            torch.onnx.export(
                model,
                inputs,
                buf,
                input_names=input_names,
                opset_version=17,
                dynamo=False,
            )
    buf.seek(0)
    return onnx.load_model(buf)


def _detect_from_config(config_attr, backend, detect_kwargs):
    """Build, export, and run block detection for one LLM (model, backend) case."""
    model, cfg = _build_block_topology_model(config_attr)
    onnx_model = _export_onnx(model, _sample_inputs(cfg), backend)
    return _detect_from_onnx(onnx_model, detect_kwargs)


def _detect_vlm_backbone(vlm_key, backend, detect_kwargs):
    """Build, export, and run block detection for one VLM-backbone (model, backend) case."""
    model, cfg = _build_vlm_backbone(vlm_key)
    onnx_model = _export_onnx(
        model,
        _vlm_sample_inputs(cfg),
        backend,
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
    )
    return _detect_from_onnx(onnx_model, detect_kwargs)


def _detect_from_onnx(onnx_model, detect_kwargs):
    """Run block/norm detection on an exported model.

    Returns the analysis IR alongside the results so the assertions below can
    reason about topological position, which the name-based results do not carry.
    """
    ir_model = ir_analysis.build_analysis_ir(onnx_model)
    topo_index = ir_analysis.topological_index(ir_model)
    active_norms = find_active_norms_in_ir(ir_model, topo_index)
    blocks = get_decoder_block_boundaries_in_ir(
        ir_model, active_norms=active_norms, topo_index=topo_index, **detect_kwargs
    )
    return blocks, active_norms, ir_model, onnx_model


def _layer_index(nodes, tensor_to_index, boundary_tensor):
    """Decoder layer index of the norm op a boundary tensor feeds, or None.

    NOTE: For now, this is only meaningful under torchscript.
          For dynamo, we will need a pass to fix op names to preserve scope hierarchy.
    """
    idx = tensor_to_index.get(boundary_tensor)
    if idx is None:
        return None
    match = re.search(r"layers\.(\d+)\b", nodes[idx].name or "")
    return int(match.group(1)) if match else None


def _layer_weighted_linear_count(nodes, tensor_to_index, start_tensor, end_tensor):
    """Count per-layer weighted linears in the span [start_tensor, end_tensor)."""
    start, end = tensor_to_index[start_tensor], tensor_to_index[end_tensor]
    return sum(
        1
        for node in nodes[start:end]
        if node.op_type in ("MatMul", "Gemm", "Conv") and "layers." in (node.name or "")
    )


def _assert_block_detection(
    blocks,
    active_norms,
    ir_model,
    backend,
    *,
    active_norms_per_block=2,
    homogeneous=True,
):
    """Assert detected blocks describe a correct decoder stack."""
    # counts
    assert len(blocks) == _NUM_LAYERS
    assert len(active_norms) == active_norms_per_block * _NUM_LAYERS + 1

    # contiguity: block i end tensor is block i+1 start tensor (same edge)
    for i in range(len(blocks) - 1):
        assert blocks[i][1] == blocks[i + 1][0]

    # every active norm feeds a downstream weight linear
    assert active_norms
    for active_norm in active_norms:
        assert active_norm.downstream_linears

    # Name-based structural checks only for torchscript backend
    if backend == "torchscript":
        topo_index = ir_analysis.topological_index(ir_model)
        tensor_to_index = ir_analysis.tensor_to_first_consumer_index(
            ir_model, topo_index
        )
        nodes = tuple(ir_model.graph)
        start_indices = [
            _layer_index(nodes, tensor_to_index, start) for start, _ in blocks
        ]
        assert all(idx is not None for idx in start_indices)
        assert start_indices == sorted(set(start_indices))
        for i, (_, end) in enumerate(blocks):
            if i < len(blocks) - 1:
                assert _layer_index(nodes, tensor_to_index, end) is not None
            else:
                assert _layer_index(nodes, tensor_to_index, end) is None

        # homogeneous stacks: identical per-layer weighted-linear count per block.
        if homogeneous:
            counts = [
                _layer_weighted_linear_count(nodes, tensor_to_index, s, e)
                for s, e in blocks
            ]
            assert len(set(counts)) == 1


def _block_topology_params():
    """Flatten the model matrix into (config_attr, backend, kwargs, homogeneous) params."""
    for (
        test_id,
        config_attr,
        backends,
        detect_kwargs,
        homogeneous,
    ) in _BLOCK_TOPOLOGY_MODELS:
        backend = backends[0]
        yield pytest.param(
            config_attr, backend, detect_kwargs, homogeneous, id=f"{test_id}-{backend}"
        )


# TODO enable role_map support for MoE models.
_ROLE_MAP_XFAIL = {
    "qwen2_moe",
    "qwen3_moe",
    "glm4_moe",
    "gpt_oss",
    "olmoe",
    "nemotron_h",
    "qwen3_next",
}


def _role_map_params():
    """Role-map params: skip parallel-residual models (active_norms_per_block=1),
    which the 2-norm-per-block role map does not describe. MoE decoders are
    marked xfail (see _ROLE_MAP_XFAIL)."""
    for (
        test_id,
        config_attr,
        backends,
        detect_kwargs,
        homogeneous,
    ) in _BLOCK_TOPOLOGY_MODELS:
        if detect_kwargs.get("active_norms_per_block", 2) != 2:
            continue
        backend = backends[0]
        marks = (
            [
                pytest.mark.xfail(
                    reason="MoE MLP residual writers not yet resolved", strict=True
                )
            ]
            if test_id in _ROLE_MAP_XFAIL
            else []
        )
        yield pytest.param(
            config_attr,
            backend,
            detect_kwargs,
            homogeneous,
            id=f"{test_id}-{backend}",
            marks=marks,
        )


def _residual_axis_size(ir_model, node_name, *, writes):
    """Size of the linear's residual-facing axis, derived from the analysis IR."""
    node = ir_analysis.node_by_name(ir_model).get(node_name)
    if node is None:
        return None
    weight, is_transposed = ir_analysis.get_weight_value(node)
    if weight is None:
        return None
    shape = tuple(static_tensor(weight).shape)
    transposed = node.op_type == "Conv" or is_transposed
    if writes:
        axis = 0 if transposed else -1
    else:
        axis = 1 if transposed else 0
    return shape[axis]


# torchscript op names carry the originating module scope, so when adding new
# architecture, make sure to extend the role module names in this dict.
_ROLE_MODULE_NAMES = {
    "qkv": {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "in_proj",
    },
    "gate_up": {
        "gate_proj",
        "up_proj",
        "gate_up_proj",
        "w1",
        "w3",
    },
    "o_proj": {"o_proj", "out_proj"},
    "down_proj": {"down_proj", "w2", "per_layer_projection"},
}


def _module_name(node_name):
    """Originating nn.Module name from a node's scoped name (torchscript)."""

    parts = node_name.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-1] in ("MatMul", "Gemm", "Conv"):
        return parts[-2]
    return parts[-1]


def _assert_role_map(role_map, ir_model, backend, *, expect_embed_tokens=True):
    """Assert the decoder role map assigns residual reads/writes correctly."""
    assert len(role_map.blocks) == _NUM_LAYERS

    # torchscript op names carry the originating nn.Module scope; dynamo names
    # are flat, so the name check only runs under torchscript.
    check_names = backend == "torchscript"
    residual_widths = set()
    for block_idx, block in enumerate(role_map.blocks):
        assert block.qkv.linears
        assert block.gate_up.linears
        assert len(block.o_proj) == 1
        assert block.down_proj

        reads = block.qkv.linears + block.gate_up.linears
        writes = block.o_proj + block.down_proj
        assert set(reads).isdisjoint(set(writes))

        # Fine-grained role split must partition each coarse read group: every
        # classified linear comes from its parent group, and the attention roles
        # do not overlap. (Some exotic reads — e.g. mamba in_proj_z — may be left
        # unclassified, so this is a subset, not an exact-cover, check.)
        qkv_refs = set(block.qkv.linears)
        gate_up_refs = set(block.gate_up.linears)
        q_refs = set(block.q_proj)
        k_refs = set(block.k_proj)
        v_refs = set(block.v_proj)
        for split in (
            block.q_proj,
            block.k_proj,
            block.v_proj,
            block.qkv.role(LinearRole.FUSED_QKV),
        ):
            assert set(split) <= qkv_refs
        for split in (
            block.gate_proj,
            block.up_proj,
            block.gate_up.role(LinearRole.FUSED_GATE_UP),
        ):
            assert set(split) <= gate_up_refs
        assert q_refs.isdisjoint(k_refs)
        assert q_refs.isdisjoint(v_refs)
        assert k_refs.isdisjoint(v_refs)

        read_sizes = {_residual_axis_size(ir_model, r, writes=False) for r in reads}
        write_sizes = {_residual_axis_size(ir_model, r, writes=True) for r in writes}
        block_sizes = read_sizes | write_sizes
        assert None not in block_sizes

        # All reads and writes of a block operate on one residual width.
        assert len(block_sizes) == 1
        residual_widths |= block_sizes

        # Names must match the expected module for each role (torchscript only).
        if check_names:
            for role, linears in (
                ("qkv", block.qkv.linears),
                ("gate_up", block.gate_up.linears),
                ("o_proj", block.o_proj),
                ("down_proj", block.down_proj),
            ):
                for linear in linears:
                    assert _module_name(linear) in _ROLE_MODULE_NAMES[role]

    # Every block shares the same residual width (the stream is continuous).
    assert len(residual_widths) == 1

    if expect_embed_tokens:
        assert len(role_map.embed_tokens) >= 1
    else:
        assert len(role_map.embed_tokens) == 0
    assert len(role_map.lm_head) == 1


def _strip_matmul_suffix(name):
    """Normalize op names by stripping trailing /MatMul for projection layers.

    Different transformers versions (or import-order side effects) may export
    nn.Linear projections as either ``v_proj/MatMul`` or just ``v_proj``.
    Stripping the suffix lets us assert on the logical layer identity.
    """
    if name.endswith("/MatMul") and name.count("/") > 3:
        prefix = name[: -len("/MatMul")]
        # Only strip if it looks like a named projection (not bare self_attn/MatMul)
        last_part = prefix.rsplit("/", 1)[-1]
        if last_part.endswith("_proj"):
            return prefix
    return name


def verify_find_blocks(sim):
    end_points = get_decoder_block_boundaries(sim.model.model)
    # Boundaries are residual-stream tensor names (the input to each block's
    # norm op). Map each back to the norm op it feeds to assert on logical
    # block identity.
    ir_model = ir_analysis.build_analysis_ir(sim.model.model)
    topo_index = ir_analysis.topological_index(ir_model)
    consumer_index = ir_analysis.tensor_to_first_consumer_index(ir_model, topo_index)
    nodes = tuple(ir_model.graph)
    end_points_names = [
        (
            nodes[consumer_index[start]].name,
            nodes[consumer_index[end]].name,
        )
        for start, end in end_points
    ]
    assert end_points_names == [
        (
            "/model/model/layers.0/input_layernorm",
            "/model/model/layers.1/input_layernorm",
        ),
        ("/model/model/layers.1/input_layernorm", "/model/model/norm"),
    ]
    # Per-block weighted linears: slice the graph between boundaries and keep
    # Conv/MatMul/Gemm nodes.
    #
    # Compared as sorted sets per block: which linears belong to a block is the
    # property under test, and the order among independent nodes is an artifact
    # of the traversal, not of the topology.
    linear_types = ("Conv", "MatMul", "Gemm")
    conv_linear_blocks_names = []
    for start, end in end_points:
        block_nodes = nodes[consumer_index[start] : consumer_index[end]]
        conv_linear_blocks_names.append(
            sorted(
                _strip_matmul_suffix(node.name)
                for node in block_nodes
                if node.op_type in linear_types
            )
        )

    assert conv_linear_blocks_names == [
        sorted(
            [
                "/model/model/layers.0/self_attn/v_proj",
                "/model/model/layers.0/self_attn/k_proj",
                "/model/model/layers.0/self_attn/q_proj",
                "/model/model/layers.0/self_attn/MatMul",
                "/model/model/layers.0/self_attn/MatMul_1",
                "/model/model/layers.0/self_attn/o_proj",
                "/model/model/layers.0/mlp/up_proj",
                "/model/model/layers.0/mlp/gate_proj",
                "/model/model/layers.0/mlp/down_proj",
            ]
        ),
        sorted(
            [
                "/model/model/layers.1/self_attn/v_proj",
                "/model/model/layers.1/self_attn/k_proj",
                "/model/model/layers.1/self_attn/q_proj",
                "/model/model/layers.1/self_attn/MatMul",
                "/model/model/layers.1/self_attn/MatMul_1",
                "/model/model/layers.1/self_attn/o_proj",
                "/model/model/layers.1/mlp/up_proj",
                "/model/model/layers.1/mlp/gate_proj",
                "/model/model/layers.1/mlp/down_proj",
            ]
        ),
    ]


def test_get_decoder_blocks(add_genai_tests_path):
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen2-0.5B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        with force_random_weight_init(vocab_size=1024):
            entry = LLM_ONNX.instantiate_float_model(model_id, 32, 16, small_model=True)
        collection = LLM_ONNX.instantiate_quantsim(entry)
        verify_find_blocks(collection.backbone)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_get_decoder_blocks_qwen3(add_genai_tests_path):
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen3-0.6B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        with force_random_weight_init(vocab_size=1024):
            entry = LLM_ONNX.instantiate_float_model(model_id, 32, 16, small_model=True)
        collection = LLM_ONNX.instantiate_quantsim(entry)
        verify_find_blocks(collection.backbone)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skip_on_windows_amd64(
    "insufficient disk for large ONNX export on Windows AMD64 runner"
)
def test_get_decoder_blocks_qwen3_5(add_genai_tests_path):
    import GenAILab.qai_hub_lm.transforms.exportable_linear_attention  # noqa: F401
    from GenAILab.bench.yaml_config_parser import YAMLConfigParser
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen3.5-0.8B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        model_cls = YAMLConfigParser.get_model_class(
            "qwen3_5", ["ExportableLinearAttention"]
        )
        with force_random_weight_init(vocab_size=1024):
            entry = model_cls.instantiate_float_model(
                model_id, 32, 16, small_model=True
            )
        collection = model_cls.instantiate_quantsim(entry)

        blocks, active_norms, ir_model, _ = _detect_from_onnx(
            collection.backbone.model.model, {}
        )
        assert len(blocks) == 2
        assert len(active_norms) == 2 * len(blocks) + 1
        for i in range(len(blocks) - 1):
            assert blocks[i][1] == blocks[i + 1][0]

        role_map = get_llm_topology(ir_model, blocks, active_norms=active_norms)
        assert len(role_map.blocks) == len(blocks)
        for block_idx, block in enumerate(role_map.blocks):
            assert len(block.o_proj) == 1
            assert block.qkv.linears
            assert block.gate_up.linears
            assert block.down_proj
        assert len(role_map.embed_tokens) == 1
        assert len(role_map.lm_head) == 1
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.skip_on_windows_amd64(
    "transformers ONNX export unreliable on Windows AMD64"
)
class TestDecoderBlockBoundaries:
    """Block detection across architectures."""

    @pytest.mark.parametrize(
        "config_attr, backend, detect_kwargs, homogeneous",
        list(_block_topology_params()),
    )
    def test_block_detection(self, config_attr, backend, detect_kwargs, homogeneous):
        """Detect block boundaries on a LLM decoder."""
        blocks, active_norms, ir_model, _ = _detect_from_config(
            config_attr, backend, detect_kwargs
        )
        _assert_block_detection(
            blocks,
            active_norms,
            ir_model,
            backend,
            active_norms_per_block=detect_kwargs.get("active_norms_per_block", 2),
            homogeneous=homogeneous,
        )

    @pytest.mark.parametrize("vlm_key", _VLM_BACKBONE_MODELS)
    def test_vlm_backbone_block_detection(self, vlm_key):
        """Detect blocks on a VLM language backbone (torchscript export)."""
        blocks, active_norms, ir_model, _ = _detect_vlm_backbone(
            vlm_key, "torchscript", {}
        )
        _assert_block_detection(blocks, active_norms, ir_model, "torchscript")

    @pytest.mark.parametrize(
        "config_attr, backend, detect_kwargs, homogeneous",
        list(_role_map_params()),
    )
    def test_role_map(self, config_attr, backend, detect_kwargs, homogeneous):
        """Build and validate the decoder role map on a LLM decoder."""
        blocks, active_norms, ir_model, onnx_model = _detect_from_config(
            config_attr, backend, detect_kwargs
        )
        role_map = get_llm_topology(
            ir_model,
            blocks,
            active_norms=active_norms,
            active_norms_per_block=detect_kwargs.get("active_norms_per_block", 2),
        )
        _assert_role_map(role_map, ir_model, backend)

    @pytest.mark.parametrize("vlm_key", _VLM_BACKBONE_MODELS)
    def test_vlm_backbone_topology(self, vlm_key):
        """Build and validate the role map on a VLM language backbone."""
        blocks, active_norms, ir_model, onnx_model = _detect_vlm_backbone(
            vlm_key, "torchscript", {}
        )
        role_map = get_llm_topology(ir_model, blocks, active_norms=active_norms)
        _assert_role_map(role_map, ir_model, "torchscript", expect_embed_tokens=False)
