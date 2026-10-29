# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-layer cache descriptors."""

from dataclasses import dataclass

import pytest
import torch

from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    PAST_POSITION_IDS,
    AttentionType,
    LayerCacheDescriptor,
    StateKind,
    _resolve_text_config,
    all_states,
    build_layer_cache_descriptors,
    cache_state_names,
    kv_length_dynamic_axes,
)


class TestLayerCacheDescriptor:
    def test_full_attention_shapes(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.FULL,
            num_kv_heads=4,
            head_dim=16,
        )
        shape_a, shape_b = desc.dummy_state_shapes(
            batch_size=1, context_length=32, sequence_length=8
        )
        # Full attention: (batch, kv_heads, context - seq, head_dim)
        assert shape_a == (1, 4, 24, 16)
        assert shape_b == (1, 4, 24, 16)

    def test_linear_attention_shapes(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.LINEAR,
            num_kv_heads=4,
            head_dim=16,
            conv_dim=640,
            conv_kernel_size=4,
            linear_num_v_heads=8,
            linear_head_k_dim=64,
            linear_head_v_dim=128,
        )
        conv_shape, recurrent_shape = desc.dummy_state_shapes(
            batch_size=1, context_length=32, sequence_length=8
        )
        assert conv_shape == (1, 640, 4)
        assert recurrent_shape == (1, 8, 64, 128)

    def test_clip_length_full(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.FULL,
            num_kv_heads=4,
            head_dim=16,
        )
        assert desc.clip_length(100) == 100

    def test_clip_length_sliding_window(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.SLIDING_WINDOW,
            num_kv_heads=4,
            head_dim=16,
            sliding_window_size=32,
        )
        assert desc.clip_length(100) == 100
        assert desc.clip_length(16) == 16

    def test_clip_length_linear(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.LINEAR,
            num_kv_heads=4,
            head_dim=16,
        )
        assert desc.clip_length(100) is None


class TestBuildLayerCacheDescriptors:
    def test_simple_full_attention(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16

        descs = build_layer_cache_descriptors(Cfg())
        assert len(descs) == 4
        assert all(d.attention_type == AttentionType.FULL for d in descs)
        assert all(d.num_kv_heads == 4 for d in descs)
        assert all(d.head_dim == 16 for d in descs)

    def test_mixed_attention_from_layer_types(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            layer_types: list = None
            linear_num_key_heads: int = 8
            linear_num_value_heads: int = 8
            linear_key_head_dim: int = 64
            linear_value_head_dim: int = 128
            linear_conv_kernel_dim: int = 4

        cfg = Cfg(
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ]
        )
        descs = build_layer_cache_descriptors(cfg)
        assert descs[0].attention_type == AttentionType.LINEAR
        assert descs[1].attention_type == AttentionType.FULL
        assert descs[2].attention_type == AttentionType.LINEAR
        assert descs[3].attention_type == AttentionType.FULL

    def test_linear_dimensions_populated(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 2
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            layer_types: list = None
            linear_num_key_heads: int = 8
            linear_num_value_heads: int = 8
            linear_key_head_dim: int = 64
            linear_value_head_dim: int = 128
            linear_conv_kernel_dim: int = 4

        cfg = Cfg(layer_types=["linear_attention", "full_attention"])
        descs = build_layer_cache_descriptors(cfg)

        linear_desc = descs[0]
        # conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
        assert linear_desc.conv_dim == 64 * 8 * 2 + 128 * 8
        assert linear_desc.conv_kernel_size == 4
        assert linear_desc.linear_num_v_heads == 8
        assert linear_desc.linear_head_k_dim == 64
        assert linear_desc.linear_head_v_dim == 128

        full_desc = descs[1]
        assert full_desc.conv_dim is None
        assert full_desc.conv_kernel_size is None

    def test_head_dim_derived_from_hidden_size(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 1
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128

        descs = build_layer_cache_descriptors(Cfg())
        assert descs[0].head_dim == 16  # 128 // 8

    def test_sliding_window_pattern(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            sliding_window: int = 32
            sliding_window_pattern: int = 2

        descs = build_layer_cache_descriptors(Cfg())
        # Pattern=2: layers 0, 2 are sliding window
        assert descs[0].attention_type == AttentionType.SLIDING_WINDOW
        assert descs[1].attention_type == AttentionType.FULL
        assert descs[2].attention_type == AttentionType.SLIDING_WINDOW
        assert descs[3].attention_type == AttentionType.FULL
        assert descs[0].sliding_window_size == 32

    def test_vlm_composite_config_resolved(self):
        """build_layer_cache_descriptors should resolve a VLM composite config
        to its nested text_config automatically."""

        @dataclass
        class TextCfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16

        @dataclass
        class VLMCfg:
            text_config: TextCfg = None
            model_type: str = "gemma3"

        vlm_cfg = VLMCfg(text_config=TextCfg())
        descs = build_layer_cache_descriptors(vlm_cfg)
        assert len(descs) == 4
        assert all(d.attention_type == AttentionType.FULL for d in descs)
        assert all(d.head_dim == 16 for d in descs)

    def test_vlm_composite_config_with_sliding_window(self):
        """VLM composite configs should propagate sliding window settings
        from the nested text_config."""

        @dataclass
        class TextCfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            sliding_window: int = 64
            sliding_window_pattern: int = 2

        @dataclass
        class VLMCfg:
            text_config: TextCfg = None
            model_type: str = "gemma3"

        vlm_cfg = VLMCfg(text_config=TextCfg())
        descs = build_layer_cache_descriptors(vlm_cfg)
        assert descs[0].attention_type == AttentionType.SLIDING_WINDOW
        assert descs[1].attention_type == AttentionType.FULL
        assert descs[0].sliding_window_size == 64

    def test_global_head_dim_for_full_attention(self):
        """Before transformers 5.17, Gemma4 keeps the full-attention head_dim
        in ``global_head_dim`` on the top-level config."""

        @dataclass
        class Cfg:
            num_hidden_layers: int = 2
            num_attention_heads: int = 8
            num_key_value_heads: int = 2
            hidden_size: int = 128
            head_dim: int = 16
            global_head_dim: int = 32
            layer_types: list = None

        cfg = Cfg(layer_types=["sliding_attention", "full_attention"])
        descs = build_layer_cache_descriptors(cfg)
        assert [d.head_dim for d in descs] == [16, 32]

    def test_per_layer_config(self):
        """From transformers 5.17, settings that differ between layers are only
        readable through ``per_layer_config``."""

        @dataclass
        class LayerCfg:
            num_attention_heads: int = 8
            num_key_value_heads: int = 2
            hidden_size: int = 128
            head_dim: int = 16

        class Cfg:
            num_hidden_layers = 2
            layer_types = ["sliding_attention", "full_attention"]
            is_heterogeneous = True
            per_layer_config = [
                LayerCfg(),
                LayerCfg(head_dim=32, num_key_value_heads=4),
            ]

            @property
            def head_dim(self):
                raise RuntimeError("per-layer attribute")

        descs = build_layer_cache_descriptors(Cfg())
        assert [d.head_dim for d in descs] == [16, 32]
        assert [d.num_kv_heads for d in descs] == [2, 4]


class TestResolveTextConfig:
    def test_returns_text_config_when_present(self):
        @dataclass
        class TextCfg:
            num_hidden_layers: int = 4

        @dataclass
        class VLMCfg:
            text_config: TextCfg = None

        vlm_cfg = VLMCfg(text_config=TextCfg())
        assert _resolve_text_config(vlm_cfg) is vlm_cfg.text_config

    def test_returns_same_config_for_plain_llm(self):
        @dataclass
        class LLMCfg:
            num_hidden_layers: int = 4

        cfg = LLMCfg()
        assert _resolve_text_config(cfg) is cfg

    def test_ignores_non_config_text_config(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            text_config: dict = None

        cfg = Cfg(text_config={"num_hidden_layers": 8})
        assert _resolve_text_config(cfg) is cfg


class TestQwen4ExpStates:
    """QSA layers carry indexer keys beside K/V, and the model carries past positions."""

    @staticmethod
    def _cfg(**overrides):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 2
            num_attention_heads: int = 2
            num_key_value_heads: int = 1
            hidden_size: int = 32
            head_dim: int = 16
            layer_types: list = None
            linear_num_key_heads: int = 2
            linear_num_value_heads: int = 4
            linear_key_head_dim: int = 8
            linear_value_head_dim: int = 8
            linear_conv_kernel_dim: int = 4
            indexer_head_dim: int = 12
            ple_layer_ids: list = None
            hc_count: int = 2
            ple_conv_kernel_size: int = 4
            ngram_size: int = 3

        return Cfg(
            layer_types=["linear_attention", "qwen_sparse_attention"], **overrides
        )

    def test_qsa_layer_is_full_attention_with_indexer_keys(self):
        linear, qsa = build_layer_cache_descriptors(self._cfg())
        assert qsa.attention_type == AttentionType.FULL
        assert qsa.indexer_head_dim == 12
        assert linear.indexer_head_dim is None
        assert [s.name for s in qsa.states] == [
            "past_key_1",
            "past_value_1",
            "past_indexer_key_1",
        ]
        assert qsa.states[2].seq_dim == 1

    def test_state_names_and_dummy_shapes(self):
        descs = build_layer_cache_descriptors(self._cfg())
        assert cache_state_names(descs, "in")[-2:] == [
            "past_indexer_key_1_in",
            f"{PAST_POSITION_IDS}_in",
        ]
        assert descs[1].dummy_state_shapes(2, 32, 8) == (
            (2, 1, 24, 16),
            (2, 1, 24, 16),
            (2, 24, 12),
        )
        (positions,) = [s for s in all_states(descs) if s.name == PAST_POSITION_IDS]
        assert positions.kind == StateKind.APPEND
        assert positions.dtype is torch.int64
        assert positions.dummy_shape(2, 24) == (2, 3, 24)

    def test_every_growing_input_is_dynamic(self):
        axes = kv_length_dynamic_axes(build_layer_cache_descriptors(self._cfg()))
        assert axes == {
            "past_key_1_in": {2: "kv_cache_length"},
            "past_value_1_in": {2: "kv_cache_length"},
            "past_indexer_key_1_in": {1: "kv_cache_length"},
            f"{PAST_POSITION_IDS}_in": {2: "kv_cache_length"},
        }

    def test_models_without_qsa_are_unchanged(self):
        cfg = self._cfg()
        cfg.layer_types = ["linear_attention", "full_attention"]
        descs = build_layer_cache_descriptors(cfg)
        assert cache_state_names(descs, "in") == [
            "conv_state_0_in",
            "recurrent_state_0_in",
            "past_key_1_in",
            "past_value_1_in",
        ]

    def test_ple_layer_carries_its_conv_window(self):
        # ple_layer_ids are one-indexed: [1] is decoder layer 0, the linear layer
        linear, qsa = build_layer_cache_descriptors(self._cfg(ple_layer_ids=[1]))
        assert [s.name for s in linear.states] == [
            "conv_state_0",
            "recurrent_state_0",
            "ple_conv_state_0",
        ]
        ple = linear.states[2]
        assert ple.kind == StateKind.REPLACE
        # hidden 32 x 2 streams; (kernel 4 - 1) x dilation 3 past columns
        assert ple.dummy_shape(2, 24) == (2, 64, 9)
        assert qsa.ple_conv_shape is None
        assert "ple_conv_state_0_in" not in kv_length_dynamic_axes([linear, qsa])

    def test_models_without_ple_are_unchanged(self):
        (linear, _) = build_layer_cache_descriptors(self._cfg())
        assert [s.name for s in linear.states] == ["conv_state_0", "recurrent_state_0"]
