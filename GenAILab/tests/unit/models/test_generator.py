# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Generator class — static shapes, KV cache, slicing, forward."""

import contextlib
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from GenAILab.qai_hub_lm.models.generator import (
    Generator,
    _FlatListCache,
    get_past_keyval_with_shift,
)
from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    AttentionType,
    LayerCacheDescriptor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    hidden_size: int = 64
    head_dim: int = 16
    vocab_size: int = 256


class _DummyModel(torch.nn.Module):
    """Model stub that returns (logits, *kv_pairs) with correct shapes."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or _Cfg()
        self._param = torch.nn.Parameter(torch.zeros(1))

    @property
    def config(self):
        return self.cfg

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, *args):
        # args: (input_tokens, attention_mask, position_ids, *kv_pairs)
        input_tokens = args[0]
        batch = input_tokens.shape[0]
        seq_len = input_tokens.shape[1]
        logits = torch.randn(batch, seq_len, self.cfg.vocab_size)
        kv_shape = (batch, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim)
        kv_tensors = [
            torch.randn(kv_shape) for _ in range(self.cfg.num_hidden_layers * 2)
        ]
        return (logits, *kv_tensors)


class _DeterministicModel(torch.nn.Module):
    """Model stub with deterministic outputs for sequence-length parity tests.

    Uses a fixed linear projection so that identical unmasked inputs produce
    identical outputs regardless of padding.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or _Cfg()
        torch.manual_seed(42)
        self.proj = torch.nn.Linear(
            self.cfg.hidden_size, self.cfg.vocab_size, bias=False
        )
        self.kv_proj = torch.nn.Linear(
            self.cfg.hidden_size, self.cfg.head_dim, bias=False
        )

    @property
    def config(self):
        return self.cfg

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, *args):
        input_tokens = args[0]  # [batch, seq_len] int or [batch, seq_len, hidden] float
        attention_mask = args[1]  # [batch, 1, seq_len, context_len]
        batch = input_tokens.shape[0]
        seq_len = input_tokens.shape[1]

        if input_tokens.ndim == 2:
            hidden = torch.nn.functional.one_hot(
                input_tokens.long() % self.cfg.hidden_size, self.cfg.hidden_size
            ).float()
        else:
            hidden = input_tokens.float()

        logits = self.proj(hidden)
        kv_base = self.kv_proj(hidden)
        kv_shape = (batch, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim)
        kv_tensors = [
            kv_base.view(batch, 1, seq_len, self.cfg.head_dim)
            .expand(kv_shape)
            .contiguous()
            for _ in range(self.cfg.num_hidden_layers * 2)
        ]
        return (logits, *kv_tensors)


@pytest.fixture
def cfg():
    return _Cfg()


@pytest.fixture
def model(cfg):
    return _DummyModel(cfg)


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    return tok


@pytest.fixture
def gen(model, tokenizer):
    return Generator(
        model=model,
        tokenizer=tokenizer,
        sequence_length=8,
        context_length=32,
    )


# ---------------------------------------------------------------------------
# prepare_inputs
# ---------------------------------------------------------------------------


class TestPrepareInputs:
    def test_pads_short_input(self, model):
        input_ids = torch.tensor([[1, 2, 3]])  # length 3
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        padded_input = result["input_ids"]
        assert padded_input.shape == (1, 8)
        # Original tokens should be at the end
        assert padded_input[0, -1].item() == 3
        assert padded_input[0, -3].item() == 1

    def test_returns_ordered_dict(self, model):
        from collections import OrderedDict

        input_ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        assert isinstance(result, OrderedDict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "position_ids" in result

    def test_attention_mask_padding(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        mask_4d = result["attention_mask"]
        assert mask_4d.shape == (1, 1, 8, 32)

    def test_kv_cache_shape(self, model, cfg):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        kv_keys = [
            k
            for k in result
            if k.startswith("past_key_") or k.startswith("past_value_")
        ]
        assert len(kv_keys) == cfg.num_hidden_layers * 2
        for k in kv_keys:
            assert result[k].shape == (1, cfg.num_key_value_heads, 32 - 8, cfg.head_dim)

    def test_position_ids_from_mask(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        position_ids = result["position_ids"]
        assert position_ids.shape[-1] == 8

    def test_position_ids_passthrough(self, model):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        custom_pos = torch.arange(8).unsqueeze(0)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
            position_ids=custom_pos,
        )
        position_ids = result["position_ids"]
        assert position_ids.shape[-1] == 8

    def test_embeds_instead_of_ids(self, model, cfg):
        embeds = torch.randn(1, 4, cfg.hidden_size)
        attn = torch.ones(1, 4, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
            inputs_embeds=embeds,
        )
        padded = result["inputs_embeds"]
        assert padded.shape == (1, 8, cfg.hidden_size)

    def test_both_ids_and_embeds_raises(self, model, cfg):
        with pytest.raises(ValueError, match="exactly one"):
            Generator.prepare_inputs(
                model=model,
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.ones(1, 2, dtype=torch.int32),
                past_key_values=[],
                sequence_length=8,
                context_length=32,
                inputs_embeds=torch.randn(1, 2, cfg.hidden_size),
            )

    def test_neither_ids_nor_embeds_raises(self, model):
        with pytest.raises(ValueError, match="exactly one"):
            Generator.prepare_inputs(
                model=model,
                input_ids=None,
                attention_mask=torch.ones(1, 2, dtype=torch.int32),
                past_key_values=[],
                sequence_length=8,
                context_length=32,
            )

    def test_default_attention_mask(self, model):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        assert result["input_ids"].shape == (1, 8)


# ---------------------------------------------------------------------------
# get_past_keyval_with_shift
# ---------------------------------------------------------------------------


class TestGetPastKeyvalWithShift:
    def test_empty_past(self):
        new_kvs = [
            torch.randn(1, 4, 5, 16),  # key
            torch.randn(1, 4, 5, 16),  # value
        ]
        result = get_past_keyval_with_shift([], new_kvs, length=10)
        assert len(result) == 2
        assert result[0].shape[2] == 5  # 0 + 5 = 5, clipped to min(5, 10)

    def test_empty_new(self):
        past_kvs = [
            torch.randn(1, 4, 8, 16),
            torch.randn(1, 4, 8, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, [], length=10)
        assert len(result) == 2
        assert result[0].shape[2] == 8

    def test_concatenation(self):
        past_kvs = [
            torch.randn(1, 4, 3, 16),
            torch.randn(1, 4, 3, 16),
        ]
        new_kvs = [
            torch.randn(1, 4, 2, 16),
            torch.randn(1, 4, 2, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, new_kvs, length=10)
        assert result[0].shape[2] == 5  # 3 + 2

    def test_truncation(self):
        past_kvs = [
            torch.randn(1, 4, 8, 16),
            torch.randn(1, 4, 8, 16),
        ]
        new_kvs = [
            torch.randn(1, 4, 5, 16),
            torch.randn(1, 4, 5, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, new_kvs, length=10)
        assert result[0].shape[2] == 10  # 8 + 5 = 13, truncated to 10

    def test_device(self):
        past_kvs = [torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)]
        new_kvs = [torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16)]
        result = get_past_keyval_with_shift(
            past_kvs, new_kvs, length=10, device=torch.device("cpu")
        )
        assert result[0].device == torch.device("cpu")

    def test_dtype(self):
        past_kvs = [torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)]
        new_kvs = [torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16)]
        result = get_past_keyval_with_shift(
            past_kvs, new_kvs, length=10, dtype=torch.float16
        )
        assert result[0].dtype == torch.float16


# ---------------------------------------------------------------------------
# slice_inputs_for_inference
# ---------------------------------------------------------------------------


class TestSliceInputs:
    def test_single_slice(self):
        inputs = torch.randn(1, 4)
        attn = torch.ones(1, 4)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        assert len(slices) == 1
        assert slices[0][0].shape[1] == 4

    def test_multiple_slices(self):
        inputs = torch.randn(1, 24)
        attn = torch.ones(1, 24)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        assert len(slices) == 3

    def test_reverse_order(self):
        # First yielded slice should contain the earliest tokens
        inputs = torch.arange(16).float().unsqueeze(0)
        attn = torch.ones(1, 16)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        # First slice should have tokens 0-7
        assert slices[0][0][0, 0].item() == 0.0
        # Second slice should have tokens 8-15
        assert slices[1][0][0, 0].item() == 8.0

    def test_position_ids_sliced(self):
        inputs = torch.randn(1, 16)
        attn = torch.ones(1, 16)
        pos = torch.arange(16).unsqueeze(0)
        slices = list(
            Generator.slice_inputs_for_inference(
                inputs, attn, sequence_length=8, position_ids=pos
            )
        )
        assert slices[0][2].shape[-1] == 8
        assert slices[1][2].shape[-1] == 8


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_returns_causal_lm_output(self, gen):
        input_ids = torch.randint(0, 256, (1, 6))
        result = gen.forward(input_ids=input_ids)
        assert isinstance(result, CausalLMOutputWithPast)

    def test_logits_strip_padding(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result = gen.forward(input_ids=input_ids)
        # Logits should have seq_len matching input, not padded sequence_length
        assert result.logits.shape[1] == 4

    def test_kv_cache_populated(self, gen):
        input_ids = torch.randint(0, 256, (1, 6))
        result = gen.forward(input_ids=input_ids)
        assert isinstance(result.past_key_values, DynamicCache)
        assert result.past_key_values.get_seq_length() > 0

    def test_kv_cache_accumulates(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result1 = gen.forward(input_ids=input_ids)
        kv_len1 = result1.past_key_values.get_seq_length()

        input_ids2 = torch.randint(0, 256, (1, 2))
        attn = torch.ones(1, 2, dtype=torch.int32)
        result2 = gen.forward(
            input_ids=input_ids2,
            attention_mask=attn,
            past_key_values=result1.past_key_values,
        )
        kv_len2 = result2.past_key_values.get_seq_length()
        assert kv_len2 > kv_len1

    def test_long_input_multi_slice(self, gen):
        # Input longer than sequence_length (8) should be auto-sliced
        input_ids = torch.randint(0, 256, (1, 20))
        result = gen.forward(input_ids=input_ids)
        assert result.logits.shape[1] == 20

    def test_creates_default_attention_mask(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result = gen.forward(input_ids=input_ids, attention_mask=None)
        assert result.logits.shape[1] == 4

    def test_both_ids_and_embeds_raises(self, gen, cfg):
        with pytest.raises(ValueError, match="exactly one"):
            gen.forward(
                input_ids=torch.tensor([[1, 2]]),
                inputs_embeds=torch.randn(1, 2, cfg.hidden_size),
            )


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


class TestPrefill:
    def test_yields_prepared_inputs(self, gen):
        from collections import OrderedDict

        input_ids = torch.randint(0, 256, (1, 6))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) >= 1
        assert isinstance(slices[0], OrderedDict)
        assert "input_ids" in slices[0]

    def test_single_slice_input(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) == 1

    def test_multi_slice_yields_multiple(self, gen):
        input_ids = torch.randint(0, 256, (1, 20))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) >= 2


# ---------------------------------------------------------------------------
# prepare_inputs_for_generation (HF generate API)
# ---------------------------------------------------------------------------


class TestPrepareInputsForGeneration:
    def test_first_call_no_stripping(self, gen):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        cache = DynamicCache()
        result = gen.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=cache,
        )
        assert result["input_ids"].shape[1] == 4

    def test_strips_consumed_tokens(self, gen, cfg):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        cache = DynamicCache()
        # Simulate 3 tokens already consumed (need one entry per layer)
        for i in range(cfg.num_hidden_layers):
            cache.update(
                torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim),
                torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim),
                layer_idx=i,
            )
        result = gen.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=cache,
        )
        assert result["input_ids"].shape[1] == 2  # 5 - 3


# ---------------------------------------------------------------------------
# combine_local_and_global_outputs
# ---------------------------------------------------------------------------


class TestCombineOutputs:
    def _make_local_outputs(self, cfg):
        from collections import OrderedDict

        local_logits = torch.randn(1, 8, cfg.vocab_size)
        kv_shape = (1, cfg.num_key_value_heads, 8, cfg.head_dim)
        out = OrderedDict()
        out["logits"] = local_logits
        for i in range(cfg.num_hidden_layers):
            out[f"past_key_{i}_out"] = torch.randn(kv_shape)
            out[f"past_value_{i}_out"] = torch.randn(kv_shape)
        return out

    def test_logits_concatenated(self, gen, cfg):
        global_outputs = {
            "past_key_values": [],
            "logits": torch.randn(1, 4, cfg.vocab_size),
        }
        local_outputs = self._make_local_outputs(cfg)
        gen.combine_local_and_global_outputs(3, local_outputs, global_outputs)
        # Should be 4 existing + 3 new valid tokens
        assert global_outputs["logits"].shape[1] == 7

    def test_logits_strip_padding(self, gen, cfg):
        global_outputs = {"past_key_values": []}
        local_outputs = self._make_local_outputs(cfg)
        gen.combine_local_and_global_outputs(3, local_outputs, global_outputs)
        # Only 3 valid tokens from an 8-wide padded output
        assert global_outputs["logits"].shape[1] == 3


# ---------------------------------------------------------------------------
# _FlatListCache
# ---------------------------------------------------------------------------


def _make_descriptors(types):
    """Create minimal LayerCacheDescriptor list from a sequence of AttentionTypes."""
    return [
        LayerCacheDescriptor(
            layer_idx=i,
            attention_type=t,
            num_kv_heads=4,
            head_dim=16,
        )
        for i, t in enumerate(types)
    ]


class TestFlatListCache:
    def test_seq_length_from_full_attention_layer(self):
        descs = _make_descriptors(
            [AttentionType.LINEAR, AttentionType.FULL, AttentionType.FULL]
        )
        # Layer 0 (linear): 2D state, no seq dim
        # Layer 1 (full): key shape (1, 4, 7, 16) → seq_length = 7
        # Layer 2 (full): key shape (1, 4, 5, 16)
        flat = [
            torch.randn(1, 256, 4),  # layer 0 conv_state
            torch.randn(1, 8, 64, 128),  # layer 0 recurrent_state
            torch.randn(1, 4, 7, 16),  # layer 1 key
            torch.randn(1, 4, 7, 16),  # layer 1 value
            torch.randn(1, 4, 5, 16),  # layer 2 key
            torch.randn(1, 4, 5, 16),  # layer 2 value
        ]
        cache = _FlatListCache(flat, descs)
        assert cache.get_seq_length() == 7

    def test_seq_length_empty_cache(self):
        descs = _make_descriptors([AttentionType.FULL])
        cache = _FlatListCache([], descs)
        assert cache.get_seq_length() == 0

    def test_seq_length_all_linear(self):
        descs = _make_descriptors([AttentionType.LINEAR, AttentionType.LINEAR])
        flat = [
            torch.randn(1, 256, 4),
            torch.randn(1, 8, 64, 128),
            torch.randn(1, 256, 4),
            torch.randn(1, 8, 64, 128),
        ]
        cache = _FlatListCache(flat, descs)
        # Falls back to layer 0 — conv_state is 3D so shape[-2] = 256
        assert cache.get_seq_length() == 256

    def test_to_legacy_cache(self):
        descs = _make_descriptors([AttentionType.FULL, AttentionType.FULL])
        k0, v0 = torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)
        k1, v1 = torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)
        cache = _FlatListCache([k0, v0, k1, v1], descs)
        legacy = cache.to_legacy_cache()
        assert len(legacy) == 2
        assert torch.equal(legacy[0][0], k0)
        assert torch.equal(legacy[0][1], v0)
        assert torch.equal(legacy[1][0], k1)
        assert torch.equal(legacy[1][1], v1)


# ---------------------------------------------------------------------------
# get_past_keyval_with_shift — linear attention
# ---------------------------------------------------------------------------


class TestGetPastKeyvalLinearAttention:
    def test_linear_layer_state_replaced_not_concatenated(self):
        descs = _make_descriptors([AttentionType.LINEAR])
        past = [torch.ones(1, 8, 64, 128), torch.ones(1, 8, 64, 128)]
        new = [torch.full((1, 8, 64, 128), 2.0), torch.full((1, 8, 64, 128), 3.0)]
        result = get_past_keyval_with_shift(
            past, new, length=10, layer_cache_descriptors=descs
        )
        assert len(result) == 2
        # New state should replace old, not concatenate
        assert torch.equal(result[0], new[0])
        assert torch.equal(result[1], new[1])

    def test_mixed_layers_linear_replaced_full_concatenated(self):
        descs = _make_descriptors([AttentionType.LINEAR, AttentionType.FULL])
        past = [
            torch.ones(1, 8, 64, 128),  # layer 0 (linear) state A
            torch.ones(1, 8, 64, 128),  # layer 0 (linear) state B
            torch.randn(1, 4, 3, 16),  # layer 1 (full) key
            torch.randn(1, 4, 3, 16),  # layer 1 (full) value
        ]
        new = [
            torch.full((1, 8, 64, 128), 5.0),  # layer 0 new state A
            torch.full((1, 8, 64, 128), 6.0),  # layer 0 new state B
            torch.randn(1, 4, 2, 16),  # layer 1 new key
            torch.randn(1, 4, 2, 16),  # layer 1 new value
        ]
        result = get_past_keyval_with_shift(
            past, new, length=20, layer_cache_descriptors=descs
        )
        # Linear layer: replaced
        assert result[0].flatten()[0].item() == 5.0
        assert result[1].flatten()[0].item() == 6.0
        # Full layer: concatenated (3 + 2 = 5)
        assert result[2].shape[2] == 5
        assert result[3].shape[2] == 5

    def test_empty_past_linear_layer(self):
        descs = _make_descriptors([AttentionType.LINEAR])
        new = [torch.randn(1, 8, 64, 128), torch.randn(1, 8, 64, 128)]
        result = get_past_keyval_with_shift(
            [], new, length=10, layer_cache_descriptors=descs
        )
        # Should create zeros_like past, then replace with new
        assert torch.equal(result[0], new[0])
        assert torch.equal(result[1], new[1])


# ---------------------------------------------------------------------------
# get_past_keyval_with_shift — sliding window attention
# ---------------------------------------------------------------------------


def _make_sliding_descriptors(window_size=32):
    """5 sliding + 1 full, like Gemma 3 pattern."""
    types = [AttentionType.SLIDING_WINDOW] * 5 + [AttentionType.FULL]
    return [
        LayerCacheDescriptor(
            layer_idx=i,
            attention_type=t,
            num_kv_heads=4,
            head_dim=16,
            sliding_window_size=window_size
            if t == AttentionType.SLIDING_WINDOW
            else None,
        )
        for i, t in enumerate(types)
    ]


class TestGetPastKeyvalSlidingWindow:
    def test_sliding_window_same_length_as_full(self):
        descs = _make_sliding_descriptors(window_size=4)
        past = [torch.randn(1, 4, 8, 16) for _ in range(12)]
        new = [torch.randn(1, 4, 2, 16) for _ in range(12)]
        result = get_past_keyval_with_shift(
            past, new, length=10, layer_cache_descriptors=descs
        )
        for i in range(0, 12, 2):
            assert result[i].shape[2] == 10

    def test_sliding_window_not_clipped_below_length(self):
        descs = _make_sliding_descriptors(window_size=4)
        past = [torch.randn(1, 4, 3, 16) for _ in range(12)]
        new = [torch.randn(1, 4, 2, 16) for _ in range(12)]
        result = get_past_keyval_with_shift(
            past, new, length=10, layer_cache_descriptors=descs
        )
        for i in range(0, 12, 2):
            assert result[i].shape[2] == 5


# ---------------------------------------------------------------------------
# prepare_inputs — mixed sliding window + full attention
# ---------------------------------------------------------------------------


class _MixedAttnCfg:
    num_hidden_layers = 6
    num_attention_heads = 4
    num_key_value_heads = 4
    hidden_size = 64
    head_dim = 16
    vocab_size = 256
    sliding_window = 32
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]


class _MixedAttnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = _MixedAttnCfg()
        self._param = torch.nn.Parameter(torch.zeros(1))

    @property
    def config(self):
        return self.cfg

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, *args):
        input_tokens = args[0]
        batch = input_tokens.shape[0]
        seq_len = input_tokens.shape[1]
        logits = torch.randn(batch, seq_len, self.cfg.vocab_size)
        kv_shape = (batch, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim)
        kv_tensors = [
            torch.randn(kv_shape) for _ in range(self.cfg.num_hidden_layers * 2)
        ]
        return (logits, *kv_tensors)


class TestPrepareInputsMixedAttention:
    def test_returns_dict_attention_mask(self):
        model = _MixedAttnModel()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        assert "attention_mask_full" in result
        assert "attention_mask_sliding_window" in result

    def test_kv_cache_shapes_uniform(self):
        model = _MixedAttnModel()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        kv_keys = [
            k
            for k in result
            if k.startswith("past_key_") or k.startswith("past_value_")
        ]
        expected_kv_len = 32 - 8  # context - sequence
        for k in kv_keys:
            assert result[k].shape[2] == expected_kv_len

    def test_kv_shapes_uniform_with_existing_cache(self):
        model = _MixedAttnModel()
        cfg = model.cfg
        kv_len = 5
        past_kvs = [
            torch.randn(1, cfg.num_key_value_heads, kv_len, cfg.head_dim)
            for _ in range(cfg.num_hidden_layers * 2)
        ]
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=past_kvs,
            sequence_length=8,
            context_length=32,
        )
        kv_keys = [
            k
            for k in result
            if k.startswith("past_key_") or k.startswith("past_value_")
        ]
        expected_kv_len = 32 - 8
        for k in kv_keys:
            assert result[k].shape[2] == expected_kv_len


# ---------------------------------------------------------------------------
# layer_cache_descriptors property
# ---------------------------------------------------------------------------


class TestLayerCacheDescriptorsProperty:
    def test_builds_from_config(self, model, tokenizer):
        gen = Generator(
            model=model,
            tokenizer=tokenizer,
            sequence_length=8,
            context_length=32,
        )
        descs = gen.layer_cache_descriptors
        assert len(descs) == 2  # _Cfg has num_hidden_layers=2
        assert all(d.attention_type == AttentionType.FULL for d in descs)
        assert descs[0].num_kv_heads == 4
        assert descs[0].head_dim == 16


# ---------------------------------------------------------------------------
# Multi-sequence-length support
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_seq_gen(model, tokenizer):
    return Generator(
        model=model,
        tokenizer=tokenizer,
        sequence_length=[1, 8],
        context_length=32,
    )


class TestMultiSequenceLength:
    def test_sequence_lengths_sorted(self, multi_seq_gen):
        assert multi_seq_gen.sequence_lengths == [1, 8]

    def test_sequence_length_property_returns_max(self, multi_seq_gen):
        assert multi_seq_gen.sequence_length == 8

    def test_single_int_wrapped_in_list(self, gen):
        assert gen.sequence_lengths == [8]
        assert gen.sequence_length == 8

    def test_select_picks_smallest_fit(self, multi_seq_gen):
        assert multi_seq_gen._select_sequence_length(1) == 1
        assert multi_seq_gen._select_sequence_length(5) == 8
        assert multi_seq_gen._select_sequence_length(8) == 8

    def test_select_falls_back_to_largest(self, multi_seq_gen):
        assert multi_seq_gen._select_sequence_length(20) == 8

    def test_forward_short_input_uses_small_seq_len(self, multi_seq_gen, cfg):
        input_ids = torch.randint(0, 256, (1, 1))
        result = multi_seq_gen.forward(input_ids=input_ids)
        assert isinstance(result, CausalLMOutputWithPast)
        assert result.logits.shape[1] == 1

    def test_forward_long_input_uses_large_seq_len(self, multi_seq_gen, cfg):
        input_ids = torch.randint(0, 256, (1, 6))
        result = multi_seq_gen.forward(input_ids=input_ids)
        assert result.logits.shape[1] == 6

    def test_forward_multi_slice(self, multi_seq_gen, cfg):
        input_ids = torch.randint(0, 256, (1, 20))
        result = multi_seq_gen.forward(input_ids=input_ids)
        assert result.logits.shape[1] == 20

    def test_kv_cache_shape_matches_selected_seq_len(self, model, tokenizer, cfg):
        gen = Generator(
            model=model,
            tokenizer=tokenizer,
            sequence_length=[1, 8],
            context_length=32,
        )
        # Single token: selects seq_len=1, KV cache clipped to ctx-1=31
        input_ids = torch.randint(0, 256, (1, 1))
        result = gen.forward(input_ids=input_ids)
        kv_len = result.past_key_values.get_seq_length()
        assert kv_len == 1

    def test_prefill_uses_large_seq_len(self, multi_seq_gen):
        from collections import OrderedDict

        input_ids = torch.randint(0, 256, (1, 6))
        slices = list(multi_seq_gen.prefill(input_ids=input_ids))
        assert len(slices) == 1
        assert isinstance(slices[0], OrderedDict)
        assert slices[0]["input_ids"].shape[1] == 8

    def test_prefill_single_token_uses_max_seq_len(self, multi_seq_gen):
        input_ids = torch.randint(0, 256, (1, 1))
        slices = list(multi_seq_gen.prefill(input_ids=input_ids))
        assert len(slices) == 1
        assert slices[0]["input_ids"].shape[1] == max(multi_seq_gen.sequence_lengths)

    def test_three_sequence_lengths(self, model, tokenizer):
        gen = Generator(
            model=model,
            tokenizer=tokenizer,
            sequence_length=[1, 4, 16],
            context_length=64,
        )
        assert gen._select_sequence_length(1) == 1
        assert gen._select_sequence_length(3) == 4
        assert gen._select_sequence_length(4) == 4
        assert gen._select_sequence_length(10) == 16
        assert gen._select_sequence_length(50) == 16


# ---------------------------------------------------------------------------
# Sequence-length parity: different configs produce identical results
# ---------------------------------------------------------------------------


class TestSequenceLengthParity:
    """Verify that generators with different sequence_length configs produce
    identical logits for the same input.

    A single-graph generator (sequence_length=16) and a multi-graph generator
    (sequence_length=[1, 8, 16]) must agree on both prefill and decode, since
    padding is masked out and should not affect the result.
    """

    @pytest.fixture
    def det_model(self, cfg):
        return _DeterministicModel(cfg)

    def _make_generator(self, model, tokenizer, sequence_length):
        return Generator(
            model=model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=64,
        )

    def test_prefill_logits_identical(self, det_model, tokenizer):
        gen_single = self._make_generator(det_model, tokenizer, 16)
        gen_multi = self._make_generator(det_model, tokenizer, [1, 8, 16])

        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
        with torch.no_grad():
            out_single = gen_single(input_ids=input_ids)
            out_multi = gen_multi(input_ids=input_ids)

        assert out_single.logits.shape == out_multi.logits.shape
        torch.testing.assert_close(out_single.logits, out_multi.logits)

    def test_decode_step_logits_identical(self, det_model, tokenizer):
        gen_single = self._make_generator(det_model, tokenizer, 16)
        gen_multi = self._make_generator(det_model, tokenizer, [1, 8, 16])

        input_ids = torch.tensor([[10, 20, 30, 40]])
        with torch.no_grad():
            prefill_single = gen_single(input_ids=input_ids)
            prefill_multi = gen_multi(input_ids=input_ids)

        next_token = torch.tensor([[99]])
        decode_mask = torch.ones(1, 5, dtype=torch.int32)
        with torch.no_grad():
            decode_single = gen_single(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=prefill_single.past_key_values,
            )
            decode_multi = gen_multi(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=prefill_multi.past_key_values,
            )

        assert decode_single.logits.shape == decode_multi.logits.shape
        torch.testing.assert_close(decode_single.logits, decode_multi.logits)

    def test_different_subset_same_max(self, det_model, tokenizer):
        """[4, 16] vs [1, 4, 8, 16] — same max, different intermediate options."""
        gen_a = self._make_generator(det_model, tokenizer, [4, 16])
        gen_b = self._make_generator(det_model, tokenizer, [1, 4, 8, 16])

        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out_a = gen_a(input_ids=input_ids)
            out_b = gen_b(input_ids=input_ids)

        assert out_a.logits.shape == out_b.logits.shape
        torch.testing.assert_close(out_a.logits, out_b.logits)

    def test_single_vs_list_of_one(self, det_model, tokenizer):
        """sequence_length=16 vs sequence_length=[16] must be identical."""
        gen_int = self._make_generator(det_model, tokenizer, 16)
        gen_list = self._make_generator(det_model, tokenizer, [16])

        input_ids = torch.tensor([[5, 10, 15, 20, 25]])
        with torch.no_grad():
            out_int = gen_int(input_ids=input_ids)
            out_list = gen_list(input_ids=input_ids)

        torch.testing.assert_close(out_int.logits, out_list.logits)


# ---------------------------------------------------------------------------
# get_past_keyval_with_shift / _FlatListCache — variable states per layer
# ---------------------------------------------------------------------------


class TestVariableStatesPerLayer:
    """Qwen4-Exp: QSA layers carry (key, value, indexer key) and the model carries
    int64 past positions, so the flat list is no longer two states per layer."""

    @staticmethod
    def _descs():
        return [
            LayerCacheDescriptor(
                layer_idx=0,
                attention_type=AttentionType.LINEAR,
                num_kv_heads=1,
                head_dim=16,
                conv_dim=64,
                conv_kernel_size=4,
                linear_num_v_heads=4,
                linear_head_k_dim=8,
                linear_head_v_dim=8,
            ),
            LayerCacheDescriptor(
                layer_idx=1,
                attention_type=AttentionType.FULL,
                num_kv_heads=1,
                head_dim=16,
                indexer_head_dim=12,
            ),
        ]

    @staticmethod
    def _states(kv_len, fill=0.0):
        return [
            torch.full((1, 64, 4), fill),
            torch.full((1, 4, 8, 8), fill),
            torch.full((1, 1, kv_len, 16), fill),
            torch.full((1, 1, kv_len, 16), fill),
            torch.full((1, kv_len, 12), fill),
            torch.arange(kv_len).expand(1, 3, kv_len).clone(),
        ]

    def test_each_state_follows_its_kind(self):
        past, new = self._states(5, 1.0), self._states(3, 2.0)
        result = get_past_keyval_with_shift(
            past, new, length=6, layer_cache_descriptors=self._descs()
        )
        assert torch.equal(result[0], new[0])  # linear state replaced
        assert torch.equal(result[1], new[1])
        assert result[2].shape == (1, 1, 6, 16)  # 5 + 3 clipped to 6
        assert result[4].shape == (1, 6, 12)  # indexer keys grow along dim 1
        assert result[4][0, :, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        # Positions stay integer and grow along dim 2
        assert result[5].dtype == torch.int64
        assert result[5][0, 0].tolist() == [2, 3, 4, 0, 1, 2]

    def test_empty_past(self):
        new = self._states(3, 2.0)
        result = get_past_keyval_with_shift(
            [], new, length=8, layer_cache_descriptors=self._descs()
        )
        for got, expected in zip(result, new):
            assert torch.equal(got, expected.to(got.dtype))
        assert result[5].dtype == torch.int64

    def test_flat_list_cache(self):
        cache = _FlatListCache(self._states(7), self._descs())
        assert cache.get_seq_length() == 7
        legacy = cache.to_legacy_cache()
        assert [len(layer) for layer in legacy] == [2, 3, 1]
