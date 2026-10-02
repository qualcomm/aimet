# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HF ``model_type``-driven LLM topology analysis.

Name matching has two independent halves, tested separately:

* **Table vs HF modeling code** — :class:`TestPatternsOnHfModuleTrees` runs the
  built-in table against ``named_modules()`` of real transformers models built
  on the ``meta`` device (no weights, no export), so it can cover many
  architectures cheaply: registered ones must match, the rest must be rejected.
* **ONNX node name -> module path** — :class:`TestExporterNaming` exports a tiny
  synthetic Llama-layout model under torchscript and dynamo +
  ``fix_node_names_pass``, with one variant per known exporter naming quirk,
  and asserts each either matches correctly or fails with a clear error.

:class:`TestMatchRealHfExports` then checks the whole path on tiny real HF
exports, with expected node names derived from the torch model.

:func:`analyze_llm_topology` analyzes some model types by active norms and the
rest by name. The by-name analysis on top of the matches:

* :class:`TestAnalyzeLlmTopology` — every topology field on real HF exports of
  every supported model_type (Gemma 2, Gemma 3, Llama, Phi-3, Qwen2, Qwen3, and
  the Qwen2.5-VL / Qwen3-VL text backbones) under both exporters, plus agreement
  with the norm-counting analysis and the dispatch between the two.
* :class:`TestStructuralCrossChecks` — names swapped between nodes still match
  one-per-field, so only the structural checks can catch them.
* :class:`TestExportVariants` — KV-cache, headless, ``inputs_embeds`` and
  QuantizationSimModel graphs, under both analyses.
"""

import copy
import dataclasses
import os
import tempfile

import numpy as np
import onnx
import onnx_ir
import pytest
import torch
from torch import nn

from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64("transformers is not available on Windows ARM64")

import transformers

from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    fix_node_names_pass,
)
from aimet_onnx.graph_passes.fusions import fuse_supergroups
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import make_dummy_input
from aimet_onnx.experimental.llm_topology import ir_analysis
from aimet_onnx.experimental.llm_topology import topology as topology_module
from aimet_onnx.experimental.llm_topology.topology import (
    ACTIVE_NORM_MODEL_TYPES,
    _analyze_llm_topology_by_name,
    analyze_llm_topology,
    analyze_llm_topology_by_norm_count,
)
from aimet_onnx.experimental.llm_topology import hf_patterns
from aimet_onnx.experimental.llm_topology.hf_patterns import (
    HfModelPatterns,
    ModuleKind,
    ModuleNode,
    NamedLayerMatchError,
    get_hf_model_patterns,
    match_module_nodes,
    match_named_layers,
    module_path_of,
)
from aimet_onnx.experimental.llm_topology.layer_roles import LinearRole

from .models.transformer_blocks import qwen3_causal_lm

_NUM_LAYERS = 2
_SEQ = 8
_BACKENDS = ("torchscript", "dynamo")

# Causal-LM model_type -> transformers config class name.
_HF_MODELS = {
    "gemma2": "Gemma2Config",
    "gemma3_text": "Gemma3TextConfig",
    "llama": "LlamaConfig",
    "phi3": "Phi3Config",
    "qwen2": "Qwen2Config",
    "qwen3": "Qwen3Config",
}

# VLM text model_type -> (VLM config class name, vision config kwargs). The text
# config is passed as a dict, so the VLM config builds its own text config class.
_VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2)
_HF_VLMS = {
    "qwen2_5_vl_text": ("Qwen2_5_VLConfig", dict(_VISION, out_hidden_size=64)),
    "qwen3_vl_text": (
        "Qwen3VLConfig",
        dict(_VISION, out_hidden_size=64, deepstack_visual_indexes=[0]),
    ),
}

# Module attribute path under ``layers.<N>`` for each projection role.
_ROLE_MODULES = {
    LinearRole.Q_PROJ: "self_attn.q_proj",
    LinearRole.K_PROJ: "self_attn.k_proj",
    LinearRole.V_PROJ: "self_attn.v_proj",
    LinearRole.O_PROJ: "self_attn.o_proj",
    LinearRole.GATE_PROJ: "mlp.gate_proj",
    LinearRole.UP_PROJ: "mlp.up_proj",
    LinearRole.DOWN_PROJ: "mlp.down_proj",
}

# Phi-3 fuses q/k/v and gate/up into one projection each.
_FUSED_ROLE_MODULES = {
    LinearRole.FUSED_QKV: "self_attn.qkv_proj",
    LinearRole.O_PROJ: "self_attn.o_proj",
    LinearRole.FUSED_GATE_UP: "mlp.gate_up_proj",
    LinearRole.DOWN_PROJ: "mlp.down_proj",
}


def _role_modules(model_type):
    """Module path under ``layers.<N>`` per projection role of ``model_type``."""
    return _FUSED_ROLE_MODULES if model_type == "phi3" else _ROLE_MODULES


def _block_norms(model_type):
    """HF names of the pre-attention and pre-MLP norms of a ``model_type`` layer."""
    patterns = get_hf_model_patterns(model_type)
    return patterns.input_norm, patterns.post_attention_norm


_TINY_TEXT_CONFIG = dict(
    num_hidden_layers=_NUM_LAYERS,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    # Phi-3 defaults to pad_token_id=32000, outside this vocab.
    pad_token_id=0,
    tie_word_embeddings=False,
)


def _config(config_attr, **kwargs):
    """Instantiate ``transformers.<config_attr>``, or skip if unavailable."""
    cfg_cls = getattr(transformers, config_attr, None)
    if cfg_cls is None:
        pytest.skip(f"{config_attr} is not available in this transformers version")
    return cfg_cls(**kwargs)


def _vlm_config(model_type):
    """Tiny config of the VLM whose text config is ``model_type``."""
    config_attr, vision_config = _HF_VLMS[model_type]
    text_config = dict(_TINY_TEXT_CONFIG)
    if model_type == "qwen2_5_vl_text":
        # Multimodal RoPE splits head_dim / 2 = 8 frequencies over (t, h, w).
        text_config["rope_parameters"] = dict(
            rope_type="default", rope_theta=1e6, mrope_section=[2, 3, 3]
        )
    cfg = _config(config_attr, text_config=text_config, vision_config=vision_config)
    assert cfg.text_config.model_type == model_type
    return cfg


# ---------------------------------------------------------------------------
# module_path_of
# ---------------------------------------------------------------------------
class TestModulePathOf:
    """Node name -> dotted module path."""

    @staticmethod
    def _node(name, op_type="MatMul", domain=""):
        return onnx_ir.Node(domain, op_type, inputs=[], name=name)

    def test_regular_node_drops_op_segment(self):
        node = self._node("/model/layers.0/self_attn/q_proj/MatMul")
        assert module_path_of(node) == "model.layers.0.self_attn.q_proj"

    def test_fused_node_is_named_after_its_module(self):
        node = self._node(
            "/model/layers.0/input_layernorm", "RMSNormalization", "aimet.supergroup"
        )
        assert module_path_of(node) == "model.layers.0.input_layernorm"

    def test_fused_node_with_op_type_suffix(self):
        """Several supergroups under one module get ``/<op type>`` appended."""
        node = self._node(
            "/model/layers.0/input_layernorm/RMSNormalization",
            "RMSNormalization",
            "aimet.supergroup",
        )
        assert module_path_of(node) == "model.layers.0.input_layernorm"


def test_unknown_model_type_lists_supported():
    with pytest.raises(ValueError, match="Unsupported model_type 'gpt2'") as exc:
        get_hf_model_patterns("gpt2")
    for model_type in (*_HF_MODELS, *_HF_VLMS):
        assert model_type in str(exc.value)


class TestHfModelPatternsConstruction:
    """Every field has a visible default; a set fused projection takes precedence."""

    def test_split_projections_default_to_llama_names(self):
        patterns = HfModelPatterns()
        assert (patterns.q_proj, patterns.k_proj, patterns.v_proj) == (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        )
        assert (patterns.gate_proj, patterns.up_proj) == (
            "mlp.gate_proj",
            "mlp.up_proj",
        )
        assert patterns.qkv_proj is None and patterns.gate_up_proj is None

    def test_fused_projections_take_precedence_in_linears(self):
        """The split fields keep their defaults but are not matched."""
        patterns = HfModelPatterns(
            qkv_proj="self_attn.qkv_proj", gate_up_proj="mlp.gate_up_proj"
        )
        assert patterns.q_proj == "self_attn.q_proj"
        assert patterns.linears == {
            LinearRole.FUSED_QKV: "self_attn.qkv_proj",
            LinearRole.O_PROJ: "self_attn.o_proj",
            LinearRole.FUSED_GATE_UP: "mlp.gate_up_proj",
            LinearRole.DOWN_PROJ: "mlp.down_proj",
        }

    def test_one_fused_projection_keeps_the_other_side_split(self):
        """Only the side that is fused is replaced."""
        linears = HfModelPatterns(qkv_proj="self_attn.qkv_proj").linears
        assert LinearRole.FUSED_QKV in linears and LinearRole.Q_PROJ not in linears
        assert linears[LinearRole.GATE_PROJ] == "mlp.gate_proj"
        assert linears[LinearRole.UP_PROJ] == "mlp.up_proj"

    @pytest.mark.parametrize(
        "kwargs, conflicting",
        [
            ({"qkv_proj": "self_attn.qkv_proj", "q_proj": "self_attn.q"}, "q_proj"),
            (
                {"qkv_proj": "a.qkv", "k_proj": "a.k", "v_proj": "a.v"},
                "k_proj, v_proj",
            ),
            ({"gate_up_proj": "mlp.gate_up_proj", "up_proj": "mlp.up"}, "up_proj"),
        ],
    )
    def test_fused_and_overridden_split_is_rejected(self, kwargs, conflicting):
        with pytest.raises(ValueError, match=f"replaces .*but {conflicting} "):
            HfModelPatterns(**kwargs)

    def test_replace_derives_a_fused_entry(self):
        """``dataclasses.replace`` works: construction no longer rewrites fields."""
        derived = dataclasses.replace(HfModelPatterns(), qkv_proj="self_attn.qkv_proj")
        assert LinearRole.FUSED_QKV in derived.linears

    def test_language_model_default(self):
        assert HfModelPatterns().language_model == "language_model"

    def test_patterns_stay_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            HfModelPatterns().q_proj = "x"


@pytest.mark.parametrize(
    "config_attr, model_type",
    [
        # Phi-3.5-mini and Phi-4 / Phi-4-mini ship as Phi3ForCausalLM.
        ("Phi3Config", "phi3"),
        # Qwen2.5 ships as Qwen2ForCausalLM.
        ("Qwen2Config", "qwen2"),
        ("Gemma2Config", "gemma2"),
        # Also the text_config of a Gemma 3 VLM (model_type "gemma3").
        ("Gemma3TextConfig", "gemma3_text"),
        ("Qwen2_5_VLTextConfig", "qwen2_5_vl_text"),
        ("Qwen3VLTextConfig", "qwen3_vl_text"),
    ],
)
def test_registered_model_type_is_the_hf_config_model_type(config_attr, model_type):
    """Table keys are what ``config.model_type`` (``text_config`` for a VLM) reports."""
    assert _config(config_attr).model_type == model_type


# ---------------------------------------------------------------------------
# Table vs HF modeling code
# ---------------------------------------------------------------------------
def _module_nodes(model, exclude=()):
    """``named_modules()`` of ``model`` as the ModuleNodes an export would yield.

    Mirrors what survives into the analysis IR: ``nn.Linear`` -> weighted
    linear, ``*RMSNorm`` -> fused RMSNorm, ``nn.Embedding`` -> table Gather.
    Other norms (LayerNorm) do not fuse into RMSNorm, so they are dropped.

    :param exclude: Module-path prefixes to leave out, as when only part of the
        model is exported.
    """
    nodes = []
    for path, module in model.named_modules():
        if any(path.startswith(prefix) for prefix in exclude):
            continue
        if isinstance(module, nn.Linear):
            kind = ModuleKind.LINEAR
        elif type(module).__name__.endswith("RMSNorm"):
            kind = ModuleKind.NORM
        elif isinstance(module, nn.Embedding):
            kind = ModuleKind.EMBEDDING
        else:
            continue
        nodes.append(ModuleNode(f"/{path}/Op", path, kind))
    return nodes


def _meta_model(auto_cls, cfg):
    with torch.device("meta"):
        return getattr(transformers, auto_cls).from_config(cfg)


class TestPatternsOnHfModuleTrees:
    """Built-in table against real transformers module trees."""

    @pytest.mark.parametrize("model_type", sorted(_HF_MODELS))
    def test_causal_lm_matches(self, model_type):
        cfg = _config(_HF_MODELS[model_type], **_TINY_TEXT_CONFIG)
        match = match_module_nodes(
            _module_nodes(_meta_model("AutoModelForCausalLM", cfg)),
            get_hf_model_patterns(model_type),
        )

        assert match.decoder_prefix == "model"
        assert [b.layer_id for b in match.blocks] == list(range(_NUM_LAYERS))
        for block in match.blocks:
            assert set(block.linears) == set(_role_modules(model_type))
            for role, module in _role_modules(model_type).items():
                assert block.linears[role] == [
                    f"/model.layers.{block.layer_id}.{module}/Op"
                ]
            assert (
                block.input_norm == f"/model.layers.{block.layer_id}.input_layernorm/Op"
            )
        assert match.embed_tokens == "/model.embed_tokens/Op"
        assert match.final_norm == "/model.norm/Op"
        assert match.lm_head == "/lm_head/Op"

    @pytest.mark.parametrize("model_type", sorted(_HF_MODELS))
    def test_bare_backbone_matches(self, model_type):
        """``AutoModel`` (no lm_head): decoder layers at the root."""
        cfg = _config(_HF_MODELS[model_type], **_TINY_TEXT_CONFIG)
        match = match_module_nodes(
            _module_nodes(_meta_model("AutoModel", cfg)),
            get_hf_model_patterns(model_type),
        )

        assert match.decoder_prefix == ""
        assert len(match.blocks) == _NUM_LAYERS
        assert match.embed_tokens == "/embed_tokens/Op"
        assert match.final_norm == "/norm/Op"
        assert match.lm_head is None

    def test_qwen3_with_llama_patterns_is_rejected(self):
        """q_norm / k_norm are RMSNorms Llama does not have."""
        cfg = _config("Qwen3Config", **_TINY_TEXT_CONFIG)
        with pytest.raises(NamedLayerMatchError, match=r"self_attn\.q_norm"):
            match_module_nodes(
                _module_nodes(_meta_model("AutoModelForCausalLM", cfg)),
                get_hf_model_patterns("llama"),
            )

    def test_phi3_with_llama_patterns_is_rejected(self):
        """Fused qkv_proj / gate_up_proj are not Llama modules, and vice versa."""
        cfg = _config("Phi3Config", **_TINY_TEXT_CONFIG)
        with pytest.raises(NamedLayerMatchError, match=r"self_attn\.qkv_proj"):
            match_module_nodes(
                _module_nodes(_meta_model("AutoModelForCausalLM", cfg)),
                get_hf_model_patterns("llama"),
            )
        cfg = _config("LlamaConfig", **_TINY_TEXT_CONFIG)
        with pytest.raises(NamedLayerMatchError, match=r"self_attn\.q_proj"):
            match_module_nodes(
                _module_nodes(_meta_model("AutoModelForCausalLM", cfg)),
                get_hf_model_patterns("phi3"),
            )

    @pytest.mark.parametrize(
        "config_attr, extra, offending",
        [
            ("Gemma2Config", {}, r"pre_feedforward_layernorm"),
            ("Gemma3TextConfig", {}, r"pre_feedforward_layernorm"),
            # Experts are not nn.Linear, so no gate/up/down projection is found.
            (
                "Qwen3MoeConfig",
                {"num_experts": 4, "num_experts_per_tok": 2},
                r"expected exactly one gate_proj, matched \[\]",
            ),
        ],
    )
    def test_unregistered_layouts_are_rejected(self, config_attr, extra, offending):
        """Other layouts fail loudly under the llama table instead of half-matching."""
        cfg = _config(config_attr, **_TINY_TEXT_CONFIG, **extra)
        with pytest.raises(NamedLayerMatchError, match=offending):
            match_module_nodes(
                _module_nodes(_meta_model("AutoModelForCausalLM", cfg)),
                get_hf_model_patterns("llama"),
            )

    @pytest.fixture(scope="class")
    def vlms(self):
        """``{model_type: meta VLM}`` for VLMs with a registered text model."""
        vision = dict(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
        )
        return {
            **{
                model_type: _meta_model(
                    "AutoModelForImageTextToText", _vlm_config(model_type)
                )
                for model_type in _HF_VLMS
            },
            "llama": _meta_model(
                "AutoModelForImageTextToText",
                _config(
                    "LlavaConfig",
                    text_config=_config("LlamaConfig", **_TINY_TEXT_CONFIG),
                    vision_config=_config("CLIPVisionConfig", **vision),
                ),
            ),
            "gemma3_text": _meta_model(
                "AutoModelForImageTextToText",
                _config(
                    "Gemma3Config",
                    text_config=dict(_TINY_TEXT_CONFIG),
                    vision_config=dict(vision, image_size=28, patch_size=14),
                    mm_tokens_per_image=4,
                ),
            ),
            "qwen3": _meta_model(
                "AutoModelForImageTextToText",
                _config(
                    "InternVLConfig",
                    text_config=_config("Qwen3Config", **_TINY_TEXT_CONFIG),
                    vision_config=_config("InternVLVisionConfig", **vision),
                ),
            ),
        }

    #: Text models whose VLM is built in ``vlms``.
    _VLM_TEXT_MODELS = sorted(("gemma3_text", "llama", "qwen3", *_HF_VLMS))

    @pytest.mark.parametrize("model_type", _VLM_TEXT_MODELS)
    def test_vlm_language_backbone_matches(self, vlms, model_type):
        """The text backbone of a VLM, exported on its own, sits under language_model."""
        nodes = _module_nodes(
            vlms[model_type],
            exclude=(
                "model.vision_tower",
                "model.multi_modal_projector",
                "model.visual",
            ),
        )
        match = match_module_nodes(nodes, get_hf_model_patterns(model_type))

        assert match.decoder_prefix == "model.language_model"
        assert len(match.blocks) == _NUM_LAYERS
        assert match.embed_tokens == "/model.language_model.embed_tokens/Op"
        assert match.final_norm == "/model.language_model.norm/Op"
        assert match.lm_head == "/lm_head/Op"

    @pytest.mark.parametrize("model_type", _VLM_TEXT_MODELS)
    def test_full_vlm_is_rejected(self, vlms, model_type):
        """A vision tower's q_proj etc. must never be mistaken for decoder layers.

        LLaVA's CLIP and Gemma 3's SigLIP towers are a second ``layers.<N>``
        stack; InternVL's is
        ``layer.<N>``, so it is caught as look-alike names outside the decoder.
        Qwen2.5-VL's and Qwen3-VL's are ``blocks.<N>`` with names (``attn.qkv``,
        ``linear_fc1``) mostly unknown to the table, caught as nodes outside the
        language backbone.
        """
        tower = "visual" if model_type in _HF_VLMS else "vision_tower"
        with pytest.raises(NamedLayerMatchError, match=tower):
            match_module_nodes(
                _module_nodes(vlms[model_type]), get_hf_model_patterns(model_type)
            )


# ---------------------------------------------------------------------------
# ONNX node name -> module path, per exporter
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(dim) + 0.5)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _Attention(nn.Module):
    """Single-head attention.

    ``qk_norm`` adds Qwen3's per-head ``q_norm`` / ``k_norm``; ``fused`` replaces
    q/k/v with Phi-3's single ``qkv_proj``.
    """

    def __init__(self, dim, call_o_proj_twice=False, qk_norm=False, fused=False):
        super().__init__()
        if fused:
            self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        else:
            self.q_proj, self.k_proj, self.v_proj = (
                nn.Linear(dim, dim, bias=False) for _ in range(3)
            )
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = _RMSNorm(dim) if qk_norm else None
        self.k_norm = _RMSNorm(dim) if qk_norm else None
        self.fused = fused
        self.call_o_proj_twice = call_o_proj_twice

    def forward(self, x):
        if self.fused:
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        scores = q @ k.transpose(-1, -2)
        out = self.o_proj(torch.softmax(scores, dim=-1) @ v)
        if self.call_o_proj_twice:
            out = self.o_proj(out)
        return out


class _Mlp(nn.Module):
    """Gated MLP; ``fused`` replaces gate/up with Phi-3's single ``gate_up_proj``."""

    def __init__(self, dim, fused=False):
        super().__init__()
        if fused:
            self.gate_up_proj = nn.Linear(dim, 4 * dim, bias=False)
        else:
            self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        return _gated_mlp(self._modules, x)


class _DictMlp(nn.Module):
    """``_Mlp`` whose projections live in a ``ModuleDict``: ``mlp.proj.<name>``."""

    def __init__(self, dim, fused=False):
        super().__init__()
        self.proj = nn.ModuleDict(dict(_Mlp(dim, fused).named_children()))

    def forward(self, x):
        return _gated_mlp(self.proj, x)


def _gated_mlp(projections, x):
    """Gated MLP over ``projections``, a name -> module mapping."""
    if "gate_up_proj" in projections:
        gate, up = projections["gate_up_proj"](x).chunk(2, dim=-1)
    else:
        gate, up = projections["gate_proj"](x), projections["up_proj"](x)
    return projections["down_proj"](torch.relu(gate) * up)


class _Layer(nn.Module):
    """Decoder layer; ``sandwich_norm`` uses Gemma's four norms per layer.

    Gemma norms both the input and the output of attention and of the MLP, and
    its ``post_attention_layernorm`` is the attention *output* norm.
    """

    def __init__(
        self,
        dim,
        mlp_cls=_Mlp,
        call_o_proj_twice=False,
        qk_norm=False,
        fused=False,
        sandwich_norm=False,
    ):
        super().__init__()
        self.input_layernorm = _RMSNorm(dim)
        self.self_attn = _Attention(dim, call_o_proj_twice, qk_norm, fused)
        self.post_attention_layernorm = _RMSNorm(dim)
        if sandwich_norm:
            self.pre_feedforward_layernorm = _RMSNorm(dim)
            self.post_feedforward_layernorm = _RMSNorm(dim)
        self.mlp = mlp_cls(dim, fused)
        self.sandwich_norm = sandwich_norm

    def forward(self, x):
        if self.sandwich_norm:
            attn = self.self_attn(self.input_layernorm(x))
            x = x + self.post_attention_layernorm(attn)
            mlp = self.mlp(self.pre_feedforward_layernorm(x))
            return x + self.post_feedforward_layernorm(mlp)
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _VisionLayer(nn.Module):
    """ViT-style layer: fused ``qkv`` and ``fc1`` / ``fc2``, no Llama module name."""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.mlp.fc1(torch.relu(self.attn.qkv(x)))


class _LayerStack(nn.Module):
    """A ``layers.<N>`` stack on its own, as a VLM vision encoder has.

    :param layer_cls: ``_Layer`` (Llama module names) or ``_VisionLayer`` (names the
        llama table does not know).
    """

    def __init__(self, dim, layer_cls=None):
        super().__init__()
        layer_cls = layer_cls or _Layer
        self.layers = nn.ModuleList(layer_cls(dim) for _ in range(_NUM_LAYERS))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Decoder(nn.Module):
    """Llama module layout: ``embed_tokens``, ``layers.<N>``, ``norm``."""

    def __init__(self, dim, vocab, **layer_kwargs):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            _Layer(dim, **layer_kwargs) for _ in range(_NUM_LAYERS)
        )
        self.norm = _RMSNorm(dim)

    def forward(self, input_ids, extra_stack=None):
        x = self.embed_tokens(input_ids)
        if extra_stack is not None:
            x = extra_stack(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _CausalLm(nn.Module):
    """``model`` + ``lm_head``, plus switches for each exporter naming quirk.

    Every module is reached through its parents' ``forward``, as in HF: the
    torchscript exporter names a node after the modules on the *call* stack,
    not after the module that owns it (see ``direct_call``).

    :param direct_call: Call ``model.layers.1.mlp.down_proj`` once more from
        here, bypassing its parent modules.
    :param vision_tower: Add a second ``layers.<N>`` stack (a stand-in for a VLM
        vision encoder) under ``vision_tower``: ``"llama"`` for one with Llama
        module names, ``"vit"`` for one with names the llama table does not know.
    """

    def __init__(self, direct_call=False, vision_tower=None, **decoder_kwargs):
        super().__init__()
        dim, vocab = 16, 32
        self.model = _Decoder(dim, vocab, **decoder_kwargs)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        layer_cls = {None: None, "llama": _Layer, "vit": _VisionLayer}[vision_tower]
        self.vision_tower = _LayerStack(dim, layer_cls) if vision_tower else None
        self.direct_call = direct_call

    def forward(self, input_ids):
        x = self.model(input_ids, extra_stack=self.vision_tower)
        if self.direct_call:
            x = self.model.layers[1].mlp.down_proj(torch.cat([x, x], dim=-1))
        return self.lm_head(x)


def _export_to_onnx(module, args, **kwargs):
    """Return ``module`` exported to ONNX, via a temporary file.

    The sink is a real path rather than an ``io.BytesIO``: on torch 2.7 the dynamo
    exporter hands the sink to ``onnxscript``'s ``save_model_with_external_data``,
    which assumes a path and raises ``TypeError`` on a file object. A path works on
    every torch version tested, so it is used unconditionally.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(module, args, path, **kwargs)
        return onnx.load_model(path)


def _export_synthetic(module, backend):
    """Export ``module`` under ``backend``; dynamo exports get their names fixed."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 32, (1, _SEQ))
    onnx_model = _export_to_onnx(
        module.eval(),
        (input_ids,),
        opset_version=18,
        dynamo=backend == "dynamo",
    )
    if backend == "dynamo":
        onnx_model = fix_node_names_pass(onnx_model)
    return onnx_model


#: Synthetic-model layout per distinct module layout of the supported model
#: types (qwen2, gemma3_text and the VLM text models share one of these).
_SYNTHETIC_LAYOUT = {
    "gemma2": {"sandwich_norm": True},
    "llama": {},
    "phi3": {"fused": True},
    "qwen3": {"qk_norm": True},
}


def _synthetic_causal_lm(model_type="llama", **kwargs):
    """``_CausalLm`` in ``model_type``'s module layout.

    Qwen3 adds q_norm / k_norm; Phi-3 fuses q/k/v and gate/up; Gemma 2 norms
    both sides of attention and the MLP.
    """
    return _CausalLm(**_SYNTHETIC_LAYOUT[model_type], **kwargs)


def _match_synthetic(backend, model_type="llama", **kwargs):
    onnx_model = _export_synthetic(_synthetic_causal_lm(model_type, **kwargs), backend)
    return match_named_layers(
        ir_analysis.build_analysis_ir(onnx_model), get_hf_model_patterns(model_type)
    )


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestExporterNaming:
    """One synthetic export per exporter naming quirk.

    Every case runs under both exporters and in the module layout of every
    supported model type.
    """

    @pytest.fixture(params=sorted(_SYNTHETIC_LAYOUT))
    def model_type(self, request):
        return request.param

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_standard_layout_matches(self, backend, model_type):
        match = _match_synthetic(backend, model_type)

        assert match.decoder_prefix == "model"
        assert [b.layer_id for b in match.blocks] == list(range(_NUM_LAYERS))
        for block in match.blocks:
            assert set(block.linears) == set(_role_modules(model_type))
            assert all(len(names) == 1 for names in block.linears.values())
        assert match.embed_tokens is not None
        assert match.final_norm is not None
        assert match.lm_head is not None

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_module_called_twice_is_rejected(self, backend, model_type):
        """torchscript names the 2nd call ``o_proj_1``; dynamo gives ``o_proj`` 2 nodes."""
        expected = r"o_proj_1" if backend == "torchscript" else r"exactly one o_proj"
        with pytest.raises(NamedLayerMatchError, match=expected):
            _match_synthetic(backend, model_type, call_o_proj_twice=True)

    def test_module_dict_level_is_rejected_under_dynamo(self, model_type):
        """``mlp.proj.down_proj`` (a ``ModuleDict`` child) is not a table path."""
        with pytest.raises(NamedLayerMatchError, match=r"mlp\.proj\.down_proj"):
            _match_synthetic("dynamo", model_type, mlp_cls=_DictMlp)

    def test_module_dict_level_is_invisible_under_torchscript(self, model_type):
        """torchscript drops the ``ModuleDict`` level from node names.

        ``mlp.proj.gate_proj`` is exported as ``/.../mlp/gate_proj/MatMul``,
        indistinguishable by name from a real ``mlp.gate_proj``. Name matching
        cannot detect this; it is pinned here so a change in exporter behavior
        is noticed. The structural cross-checks in ``analyze_llm_topology``
        are the safety net for a mis-assigned role.
        """
        match = _match_synthetic("torchscript", model_type, mlp_cls=_DictMlp)
        assert match.blocks[0].linears[LinearRole.DOWN_PROJ] == [
            "/model/layers.0/mlp/down_proj/MatMul"
        ]

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_direct_grandchild_call_is_rejected(self, backend, model_type):
        """Node names follow the *call* stack, not module ownership.

        torchscript loses the skipped parents (``down_proj`` at the root); dynamo +
        ``fix_node_names_pass`` keeps the full path, so the layer gets a second
        ``down_proj``.
        """
        expected = (
            r"outside the decoder stack.*'down_proj'"
            if backend == "torchscript"
            else r"layer 1: expected exactly one down_proj"
        )
        with pytest.raises(NamedLayerMatchError, match=expected):
            _match_synthetic(backend, model_type, direct_call=True)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("vision_tower", ["llama", "vit"])
    def test_second_layer_stack_is_rejected(self, backend, model_type, vision_tower):
        """A second ``layers.<N>`` stack is rejected whatever its modules are named.

        ``"vit"`` names no module the table knows (``attn.qkv``, ``mlp.fc1``), so
        nothing under it matches a field; it must still count as a stack.
        """
        with pytest.raises(NamedLayerMatchError, match=r"2 stacks .*vision_tower"):
            _match_synthetic(backend, model_type, vision_tower=vision_tower)

    def test_qwen3_layout_is_rejected_by_llama_patterns(self):
        """q_norm / k_norm are RMSNorms the llama table does not name."""
        onnx_model = _export_synthetic(_synthetic_causal_lm("qwen3"), "torchscript")
        with pytest.raises(NamedLayerMatchError, match=r"self_attn\.q_norm"):
            match_named_layers(
                ir_analysis.build_analysis_ir(onnx_model),
                get_hf_model_patterns("llama"),
            )

    def test_raw_dynamo_export_says_to_run_fix_node_names_pass(self):
        onnx_model = _export_to_onnx(
            _CausalLm().eval(), (torch.randint(0, 32, (1, _SEQ)),), dynamo=True
        )
        ir_model = ir_analysis.build_analysis_ir(onnx_model)

        with pytest.raises(NamedLayerMatchError, match="fix_node_names_pass") as exc:
            match_named_layers(ir_model, get_hf_model_patterns("llama"))
        assert (
            "from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx "
            "import fix_node_names_pass" in str(exc.value)
        )

    def test_unnamed_graph_reports_no_decoder_layer(self):
        """No HF module paths at all: the generic message, not the dynamo one."""
        ir_model = onnx_ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[1, 4] x) => (float[1, 4] y) {
                w = Constant<value = float[4, 4] {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}>()
                y = MatMul(x, w)
            }
            """
        )
        with pytest.raises(NamedLayerMatchError, match="no decoder layer") as exc:
            match_named_layers(ir_model, get_hf_model_patterns("llama"))
        assert "fix_node_names_pass" not in str(exc.value)


# ---------------------------------------------------------------------------
# Real HF exports
# ---------------------------------------------------------------------------
class _LogitsWrapper(nn.Module):
    """``(input_ids, attention_mask) -> logits`` prefill wrapper for export.

    The mask is passed pre-built as 4-D so HF returns it as-is instead of
    constructing one, which does not trace under torchscript.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


def _build_hf_model(model_type):
    """Tiny randomly initialized model for ``model_type``.

    An ``AutoModelForCausalLM``, or for a VLM text model_type the whole
    ``AutoModelForImageTextToText``: exported with only ``input_ids``, its vision
    tower is never called and so drops out of the graph.

    Norm gammas are randomized: HF initializes them all to 1.0, and the exporters
    then de-duplicate the identical tensors (torchscript behind ``Identity``
    ops, dynamo into one shared initializer), which no trained model has.
    """
    torch.manual_seed(0)
    if model_type in _HF_VLMS:
        model = transformers.AutoModelForImageTextToText.from_config(
            _vlm_config(model_type)
        ).eval()
    else:
        cfg = _config(_HF_MODELS[model_type], **_TINY_TEXT_CONFIG)
        model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) + 0.5)
    return model


def _export_hf(model, backend):
    """Export ``model`` under ``backend``; dynamo exports get their names fixed."""
    input_ids = torch.randint(0, model.config.get_text_config().vocab_size, (1, _SEQ))
    attention_mask = torch.zeros(1, 1, _SEQ, _SEQ)
    onnx_model = _export_to_onnx(
        _LogitsWrapper(model),
        (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        opset_version=18,
        dynamo=backend == "dynamo",
    )
    if backend == "dynamo":
        onnx_model = fix_node_names_pass(onnx_model)
    return onnx_model


def _backbone(model_type):
    """Module path of the language backbone in ``_build_hf_model(model_type)``."""
    return "model.language_model" if model_type in _HF_VLMS else "model"


@pytest.fixture(scope="module", params=sorted((*_HF_MODELS, *_HF_VLMS)))
def hf_model(request):
    return request.param, _build_hf_model(request.param)


@pytest.fixture(scope="module", params=_BACKENDS)
def hf_onnx(request, hf_model):
    """``(model_type, torch model, ONNX export)``."""
    model_type, model = hf_model
    return model_type, model, _export_hf(model, request.param)


@pytest.fixture(scope="module")
def hf_export(hf_onnx):
    """``(model_type, torch model, analysis IR of its export)``."""
    model_type, model, onnx_model = hf_onnx
    return model_type, model, ir_analysis.build_analysis_ir(onnx_model)


def _node_prefix(module_path):
    """Node-name prefix of ``module_path`` inside ``_LogitsWrapper``.

    A ``ModuleList`` child keeps its index on the container's segment
    (``layers.0``), every other level is its own ``/``-segment.
    """
    segments = []
    for part in f"model.{module_path}".split("."):
        if part.isdigit():
            segments[-1] += f".{part}"
        else:
            segments.append(part)
    return "/" + "/".join(segments)


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestMatchRealHfExports:
    """match_named_layers on tiny real HF exports."""

    def test_every_named_layer_is_its_torch_module(self, hf_export):
        model_type, model, ir_model = hf_export
        match = match_named_layers(ir_model, get_hf_model_patterns(model_type))
        modules = dict(model.named_modules())

        backbone = _backbone(model_type)
        assert match.decoder_prefix == f"model.{backbone}"
        assert [b.layer_id for b in match.blocks] == list(
            range(model.config.get_text_config().num_hidden_layers)
        )
        for block in match.blocks:
            layer = f"{backbone}.layers.{block.layer_id}"
            for role, module in _role_modules(model_type).items():
                assert isinstance(modules[f"{layer}.{module}"], nn.Linear)
                (name,) = block.linears[role]
                assert name.startswith(_node_prefix(f"{layer}.{module}") + "/")
            input_norm, post_attention_norm = _block_norms(model_type)
            assert block.input_norm == _node_prefix(f"{layer}.{input_norm}")
            assert block.post_attention_norm == _node_prefix(
                f"{layer}.{post_attention_norm}"
            )
        assert match.embed_tokens in _embed_tokens_nodes(model_type)
        assert match.final_norm == _node_prefix(f"{backbone}.norm")
        assert match.lm_head.startswith(_node_prefix("lm_head") + "/")

    def test_qwen3_export_is_rejected_by_llama_patterns(self, hf_export):
        """q_norm / k_norm fuse into RMSNorms that llama does not name."""
        model_type, _, ir_model = hf_export
        if "self_attn.q_norm" not in get_hf_model_patterns(model_type).ignored:
            pytest.skip("q_norm / k_norm are Qwen3 / Qwen3-VL / Gemma 3 only")
        with pytest.raises(NamedLayerMatchError, match=r"self_attn\.q_norm"):
            match_named_layers(ir_model, get_hf_model_patterns("llama"))


def _embed_tokens_nodes(model_type):
    """Names the embedding ``Gather`` of ``_build_hf_model(model_type)`` may have.

    A VLM's outer model calls ``language_model.embed_tokens`` itself, which
    torchscript names after the outer model and dynamo + ``fix_node_names_pass``
    as a single ``language_model.embed_tokens`` segment.
    """
    if model_type not in _HF_VLMS:
        return {_node_prefix("model.embed_tokens") + "/Gather"}
    return {
        _node_prefix("model.embed_tokens") + "/Gather",
        _node_prefix("model") + "/language_model.embed_tokens/Gather",
    }


# ---------------------------------------------------------------------------
# analyze_llm_topology
# ---------------------------------------------------------------------------
def _norm_node(module_path):
    """Name of the fused RMSNorm node for ``module_path`` inside ``_LogitsWrapper``."""
    return _node_prefix(module_path)


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestAnalyzeLlmTopology:
    """The by-name analysis on tiny real HF exports, under both exporters.

    Run directly rather than through :func:`analyze_llm_topology`, which takes
    :data:`ACTIVE_NORM_MODEL_TYPES` the other way, so that the name tables of
    those model types stay covered too.
    """

    @pytest.fixture(scope="class")
    def analyzed(self, hf_onnx):
        """``(model_type, torch model, ONNX export, topology)``."""
        model_type, model, onnx_model = hf_onnx
        return (
            model_type,
            model,
            onnx_model,
            _analyze_llm_topology_by_name(onnx_model, model_type),
        )

    def test_blocks_hold_the_named_projections(self, analyzed):
        model_type, model, _, topology = analyzed
        backbone = _backbone(model_type)

        assert len(topology.blocks) == model.config.get_text_config().num_hidden_layers
        for i, block in enumerate(topology.blocks):
            by_role = {
                **{role: names for role, names in block.qkv.by_role.items() if names},
                **{
                    role: names
                    for role, names in block.gate_up.by_role.items()
                    if names
                },
                LinearRole.O_PROJ: block.o_proj,
                LinearRole.DOWN_PROJ: block.down_proj,
            }
            assert set(by_role) == set(_role_modules(model_type))
            for role, module in _role_modules(model_type).items():
                expected = _node_prefix(f"{backbone}.layers.{i}.{module}") + "/MatMul"
                assert by_role[role] == [expected], role

    def test_fused_projections_have_no_split_roles(self, analyzed):
        """Phi-3's fused qkv_proj / gate_up_proj report no q/k/v, gate/up."""
        model_type, _, _, topology = analyzed
        fused = model_type == "phi3"
        for block in topology.blocks:
            assert bool(block.qkv.role(LinearRole.FUSED_QKV)) == fused
            assert bool(block.gate_up.role(LinearRole.FUSED_GATE_UP)) == fused
            for role in (LinearRole.Q_PROJ, LinearRole.K_PROJ, LinearRole.V_PROJ):
                assert bool(block.qkv.role(role)) != fused
            for role in (LinearRole.GATE_PROJ, LinearRole.UP_PROJ):
                assert bool(block.gate_up.role(role)) != fused

    def test_model_level_roles(self, analyzed):
        model_type, _, _, topology = analyzed

        (embed_tokens,) = topology.embed_tokens
        assert embed_tokens in _embed_tokens_nodes(model_type)
        assert topology.lm_head == [_node_prefix("lm_head") + "/MatMul"]

    def test_residual_stream_chains_through_the_norms(self, analyzed):
        model_type, _, onnx_model, topology = analyzed
        backbone = _backbone(model_type)
        norm_input = {norm.norm: norm.input_tensor for norm in topology.active_norms}
        num_blocks = len(topology.blocks)

        for i, block in enumerate(topology.blocks):
            assert (
                block.residual_input
                == norm_input[_norm_node(f"{backbone}.layers.{i}.input_layernorm")]
            )
            if i + 1 < num_blocks:
                assert block.residual_output == topology.blocks[i + 1].residual_input
        assert (
            topology.blocks[-1].residual_output
            == norm_input[_norm_node(f"{backbone}.norm")]
        )
        tensors = {out for node in onnx_model.graph.node for out in node.output}
        for block in topology.blocks:
            assert {block.residual_input, block.residual_output} <= tensors

    def test_attention_matmuls_found_by_structure(self, analyzed):
        model_type, _, onnx_model, topology = analyzed
        op_type = {node.name: node.op_type for node in onnx_model.graph.node}

        for i, block in enumerate(topology.blocks):
            attn = _node_prefix(f"{_backbone(model_type)}.layers.{i}.self_attn") + "/"
            for names in (block.qk_matmul, block.attn_v_matmul):
                assert len(names) == 1
                assert op_type[names[0]] == "MatMul"
                assert names[0].startswith(attn)
            assert block.qk_matmul != block.attn_v_matmul

    def test_active_norms_are_the_named_norms(self, analyzed):
        """Input + post-attention norm per block, then the final norm.

        Qwen3's q_norm / k_norm are RMSNorms too, but are never active norms;
        nor are Gemma's norms on the attention / MLP outputs.
        """
        model_type, model, onnx_model, topology = analyzed
        backbone = _backbone(model_type)
        expected = [
            f"{backbone}.layers.{i}.{norm}"
            for i in range(model.config.get_text_config().num_hidden_layers)
            for norm in _block_norms(model_type)
        ] + [f"{backbone}.norm"]

        assert [n.norm for n in topology.active_norms] == [
            _norm_node(path) for path in expected
        ]
        # Gemma scales by (1 + weight), which the exporter folds into one unnamed
        # initializer, so the gamma is checked by value rather than by name.
        offset = 1.0 if model_type.startswith("gemma") else 0.0
        modules = dict(model.named_modules())
        initializers = {
            init.name: onnx.numpy_helper.to_array(init)
            for init in onnx_model.graph.initializer
        }
        for norm, path in zip(topology.active_norms, expected):
            gamma = modules[path].weight.detach().numpy() + offset
            np.testing.assert_allclose(initializers[norm.scale_name], gamma, rtol=1e-6)
        blocks = topology.blocks
        for i, block in enumerate(blocks):
            input_norm, post_attention_norm = topology.active_norms[2 * i : 2 * i + 2]
            assert input_norm.downstream_linears == block.qkv.linears
            assert post_attention_norm.downstream_linears == block.gate_up.linears
        assert topology.active_norms[-1].downstream_linears == topology.lm_head

    def test_dims(self, analyzed):
        _, model, _, topology = analyzed

        assert topology.hidden_size == model.config.get_text_config().hidden_size
        assert topology.head_dim is None  # prefill export: no past_value input
        assert topology.past_key_input_names == []

    def test_matches_norm_counting_analysis(self, analyzed):
        """Same result as the norm-counting analysis, field for field."""
        _, _, onnx_model, topology = analyzed
        assert analyze_llm_topology_by_norm_count(onnx_model) == topology

    def test_dispatch(self, analyzed, monkeypatch):
        """analyze_llm_topology analyzes by active norms or by name, by model_type.

        A VLM is also analyzed under its top-level model_type (e.g. ``qwen3_vl``),
        which is what a caller holding the VLM's config passes.
        """
        model_type, model, onnx_model, topology = analyzed
        calls = []

        def spy(name, fn):
            def wrapped(*args, **kwargs):
                calls.append(name)
                return fn(*args, **kwargs)

            monkeypatch.setattr(topology_module, name, wrapped)

        spy("analyze_llm_topology_by_norm_count", analyze_llm_topology_by_norm_count)
        spy("_analyze_llm_topology_by_name", _analyze_llm_topology_by_name)

        for requested in dict.fromkeys((model_type, model.config.model_type)):
            calls.clear()
            assert analyze_llm_topology(onnx_model, requested) == topology
            assert calls == [
                "analyze_llm_topology_by_norm_count"
                if requested in ACTIVE_NORM_MODEL_TYPES
                else "_analyze_llm_topology_by_name"
            ], requested

    def test_input_proto_is_not_mutated(self, hf_onnx):
        model_type, _, onnx_model = hf_onnx
        before = onnx_model.SerializeToString()
        analyze_llm_topology(onnx_model, model_type)
        _analyze_llm_topology_by_name(onnx_model, model_type)
        assert onnx_model.SerializeToString() == before

    def test_prefused_model_analyzes_identically(self, analyzed):
        """A graph whose RMSNorms are already fused gives the same topology.

        An export arrives decomposed; a ``QuantizationSimModel`` graph arrives with
        its norms already fused into ``RMSNormalization`` supergroups. The analysis
        fuses as needed, so both must agree, down to the fused norms' node names
        that the name match relies on.
        """
        model_type, _, onnx_model, topology = analyzed
        prefused = onnx_ir.to_proto(
            fuse_supergroups(
                onnx_ir.from_proto(onnx_model), patterns=["RMSNormalization"]
            )
        )
        assert any(node.op_type == "RMSNormalization" for node in prefused.graph.node)
        assert not any(
            node.op_type == "RMSNormalization" for node in onnx_model.graph.node
        )

        assert _analyze_llm_topology_by_name(prefused, model_type) == topology

    def test_unknown_model_type(self, hf_onnx):
        _, _, onnx_model = hf_onnx
        with pytest.raises(ValueError, match="Unsupported model_type 'gpt2'"):
            analyze_llm_topology(onnx_model, "gpt2")


def _swap_node_names(onnx_model, first, second):
    """Return a copy of ``onnx_model`` with nodes ``first`` and ``second`` renamed."""
    swapped = copy.deepcopy(onnx_model)
    nodes = {node.name: node for node in swapped.graph.node}
    nodes[first].name, nodes[second].name = second, first
    return swapped


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestStructuralCrossChecks:
    """A name that points at the wrong node must fail, not yield a wrong topology.

    Swapping two nodes' names keeps every field matched exactly once, so name
    matching alone accepts the graph; only the structural checks catch it.
    """

    @pytest.fixture(scope="class")
    def llama_onnx(self):
        return _export_hf(_build_hf_model("llama"), "torchscript")

    @staticmethod
    def _proj(module):
        return _node_prefix(f"model.layers.0.{module}") + "/MatMul"

    def test_swapped_read_projections(self, llama_onnx):
        """q_proj and gate_proj read through different norms."""
        swapped = _swap_node_names(
            llama_onnx, self._proj("self_attn.q_proj"), self._proj("mlp.gate_proj")
        )
        with pytest.raises(
            ValueError, match=r"Block 0 \(layer 0\): q/k/v by name"
        ) as exc:
            _analyze_llm_topology_by_name(swapped, "llama")
        assert "gate/up by name" in str(exc.value)

    def test_swapped_write_projections(self, llama_onnx):
        """o_proj and down_proj write different residual Adds."""
        swapped = _swap_node_names(
            llama_onnx, self._proj("self_attn.o_proj"), self._proj("mlp.down_proj")
        )
        with pytest.raises(ValueError, match=r"o_proj by name .* attention residual"):
            _analyze_llm_topology_by_name(swapped, "llama")

    def test_gemma_attention_output_norm_is_not_the_pre_mlp_norm(self, monkeypatch):
        """Gemma's post_attention_layernorm normalizes the attention *output*.

        A table that took it for the pre-MLP norm, as in Llama, still names one
        node per field; the norm feeds no gate/up, so the cross-check rejects it.
        """
        monkeypatch.setitem(
            hf_patterns._HF_MODEL_PATTERNS,
            "gemma2",
            HfModelPatterns(
                ignored=("pre_feedforward_layernorm", "post_feedforward_layernorm")
            ),
        )
        onnx_model = _export_hf(_build_hf_model("gemma2"), "torchscript")
        with pytest.raises(ValueError, match=r"gate/up by name .* post-attention norm"):
            _analyze_llm_topology_by_name(onnx_model, "gemma2")

    def test_swapped_norms(self, llama_onnx):
        """The node named input_layernorm must be the norm feeding q/k/v.

        A fused norm is named after the common prefix of the nodes inside it, so
        the swap is of that prefix, on every node of both norms.
        """
        first = _norm_node("model.layers.0.input_layernorm") + "/"
        second = _norm_node("model.layers.0.post_attention_layernorm") + "/"
        swapped = copy.deepcopy(llama_onnx)
        for node in swapped.graph.node:
            if node.name.startswith(first):
                node.name = second + node.name[len(first) :]
            elif node.name.startswith(second):
                node.name = first + node.name[len(second) :]

        with pytest.raises(
            ValueError, match=r"q/k/v by name .* != input norm consumers in the graph"
        ):
            _analyze_llm_topology_by_name(swapped, "llama")


# ---------------------------------------------------------------------------
# analyze_llm_topology: export variants
# ---------------------------------------------------------------------------
_QWEN3_KV = dict(
    num_hidden_layers=_NUM_LAYERS,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    vocab_size=128,
    max_position_embeddings=128,
)


@pytest.mark.skip_on_windows_amd64("torch.onnx export is not supported on Windows")
class TestExportVariants:
    """KV-cache, headless and inputs_embeds exports of a real Qwen3.

    Qwen3 is analyzed by active norms, so each test also runs the by-name
    analysis on the same export.
    """

    @pytest.fixture(
        params=[analyze_llm_topology, _analyze_llm_topology_by_name],
        ids=["by_active_norms", "by_name"],
    )
    def analyze(self, request):
        return request.param

    def test_kv_cache_export(self, analyze):
        topology = analyze(qwen3_causal_lm(**_QWEN3_KV), "qwen3")

        assert topology.head_dim == _QWEN3_KV["head_dim"]
        assert topology.hidden_size == _QWEN3_KV["hidden_size"]
        layers = range(_NUM_LAYERS)
        assert topology.past_key_input_names == [f"past_key_{i}_in" for i in layers]
        assert topology.past_value_input_names == [f"past_value_{i}_in" for i in layers]
        assert topology.past_key_output_names == [f"past_key_{i}_out" for i in layers]
        assert topology.past_value_output_names == [
            f"past_value_{i}_out" for i in layers
        ]

    def test_headless_backbone(self, analyze):
        """No lm_head: the final norm is inactive and the last block ends on its Add."""
        onnx_model = qwen3_causal_lm(with_lm_head=False, **_QWEN3_KV)
        topology = analyze(onnx_model, "qwen3")

        assert topology.lm_head == []
        assert len(topology.active_norms) == 2 * _NUM_LAYERS
        producer = {out: node for node in onnx_model.graph.node for out in node.output}
        assert producer[topology.blocks[-1].residual_output].op_type == "Add"

    def test_inputs_embeds_backbone(self, analyze):
        onnx_model = qwen3_causal_lm(with_embedding=False, **_QWEN3_KV)
        topology = analyze(onnx_model, "qwen3")

        assert topology.embed_tokens == []
        assert len(topology.blocks) == _NUM_LAYERS
        assert topology.hidden_size == _QWEN3_KV["hidden_size"]

    def test_quantsim_graph_matches_float_graph(self, analyze):
        """A sim graph interleaves QcQuantizeOps and renames consumed tensors.

        The analysis strips quantizers first, so the topology must be the float
        graph's, down to the un-suffixed tensor names.
        """
        onnx_model = qwen3_causal_lm(**_QWEN3_KV)
        dummy_input = make_dummy_input(onnx_model)
        sim = QuantizationSimModel(copy.deepcopy(onnx_model), dummy_input=dummy_input)

        float_topology = analyze(onnx_model, "qwen3")
        sim_topology = analyze(sim.model.model, "qwen3")

        assert sim_topology.blocks == float_topology.blocks
        assert sim_topology.active_norms == float_topology.active_norms
        assert sim_topology.lm_head == float_topology.lm_head
        assert sim_topology.embed_tokens == float_topology.embed_tokens
