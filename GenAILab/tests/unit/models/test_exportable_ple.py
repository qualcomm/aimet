# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Qwen4-Exp PLE: host-computed n-gram embeddings and the ExportablePLE layer."""

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
modeling_qwen4_exp = pytest.importorskip(
    "transformers.models.qwen4_exp.modeling_qwen4_exp"
)

from transformers import DynamicCache  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextConfig,
)

from GenAILab.bench.yaml_config_parser import YAMLConfigParser  # noqa: E402
from GenAILab.qai_hub_lm.models.qwen4_exp import Qwen4ExpGenerator  # noqa: E402
from GenAILab.qai_hub_lm.models.utils.layer_cache import (  # noqa: E402
    PLE_CONV_STATE_IDX,
    build_layer_cache_descriptors,
)
from GenAILab.qai_hub_lm.transforms.exportable_ple import (  # noqa: E402
    Qwen4ExpExportablePLEAdaptation,
    _patch_ple_instances,
    exportable_ple_layer_forward,
    register_adaptations,
)

VOCAB = 128


def _config(ple_layer_ids=(1,), num_layers=4):
    return Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        hc_count=2,
        hc_lowrank=8,
        max_position_embeddings=64,
        layer_types=["linear_attention", "qwen_sparse_attention"] * (num_layers // 2),
        ple_layer_ids=list(ple_layer_ids),
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=101,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        eos_token_id=1,
        pad_token_id=0,
    )


def _model(config):
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = modeling_qwen4_exp.Qwen4ExpForCausalLM(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.abs().sum() == 0:
                param.normal_(std=0.05)
    return model


def _adapted(config):
    from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
        _patch_gated_delta_net_instances,
    )
    from GenAILab.qai_hub_lm.transforms.exportable_moe import replace_fused_experts
    from GenAILab.qai_hub_lm.transforms.exportable_qsa import _patch_qsa_instances

    model = _model(config)
    _patch_gated_delta_net_instances(model)
    replace_fused_experts(model, execution="dense", export_execution="predicated")
    _patch_qsa_instances(model)
    _patch_ple_instances(model)
    return model


def _generator(config, sequence_length):
    from GenAILab.qai_hub_lm.backends.torch.qwen4_exp import Qwen4Exp
    from GenAILab.qai_hub_lm.models.utils.exportable import (
        ONNXExportableModuleWithCache,
    )

    model = _adapted(config)
    names = Qwen4Exp.get_backbone_input_names(
        build_layer_cache_descriptors(model.config)
    )
    return Qwen4ExpGenerator(
        model=ONNXExportableModuleWithCache(model, input_names=names).eval(),
        tokenizer=MagicMock(eos_token_id=1, pad_token_id=0),
        sequence_length=sequence_length,
        context_length=48,
        **Qwen4Exp.get_extras(model),
    )


class TestLayer:
    """One PLE layer: host embeddings, left-padded chunks and a carried conv
    window reproduce the stock layer run unpadded over the whole sequence."""

    @pytest.fixture
    def layer(self):
        torch.manual_seed(0)
        layer = modeling_qwen4_exp.Qwen4ExpTextPLELayer(
            _config(num_layers=2), layer_idx=0, ple_layer_index=0
        ).eval()
        with torch.no_grad():
            for param in layer.parameters():
                if param.abs().sum() == 0:
                    param.normal_(std=0.05)
        return layer

    @pytest.mark.parametrize("chunk", [8, 1])
    def test_chunked_matches_stock(self, layer, chunk):
        config = _config(num_layers=2)
        channels, window = (
            config.hidden_size * config.hc_count,
            layer.short_conv_state_len,
        )
        torch.manual_seed(1)
        tokens, hidden_states = 21, None
        input_ids = torch.randint(2, VOCAB, (1, tokens))
        hidden_states = torch.randn(1, tokens, channels)

        with torch.no_grad():
            reference = layer(
                hidden_states, input_ids, DynamicCache(config=config), conv_mask=None
            )
            embeds = layer.ple_embedding(input_ids, None)
            outputs, state = [], torch.zeros(1, channels, window)
            for start in range(0, tokens, chunk):
                n = min(chunk, tokens - start)
                pad = chunk - n

                def left_pad(x):
                    return torch.nn.functional.pad(
                        x[:, start : start + n], (0, 0, pad, 0)
                    )

                cache = DynamicCache(config=config)
                cache.layers[0].conv_states[PLE_CONV_STATE_IDX] = state
                mask = torch.cat([torch.zeros(1, pad), torch.ones(1, n)], 1)
                out = exportable_ple_layer_forward(
                    layer,
                    left_pad(hidden_states),
                    left_pad(embeds),
                    cache,
                    conv_mask=mask,
                )
                outputs.append(out[:, pad:])
                state = cache.layers[0].conv_states[PLE_CONV_STATE_IDX]
        torch.testing.assert_close(torch.cat(outputs, 1), reference, rtol=0, atol=1e-6)

    def test_rejects_token_ids(self, layer):
        cache = DynamicCache(config=_config(num_layers=2))
        with pytest.raises(ValueError, match="n-gram embeddings"):
            exportable_ple_layer_forward(
                layer, torch.randn(1, 4, 64), torch.randint(2, VOCAB, (1, 4)), cache
            )


class TestGenerator:
    @pytest.mark.parametrize("ple_layer_ids", [(1,), (1, 3)])
    def test_chunked_matches_one_call(self, ple_layer_ids):
        """Chunked, left-padded prefill and token-by-token decode give the same
        logits as one call over the whole prompt: the carried conv windows and
        the host n-gram history lose nothing between calls."""
        config = _config(ple_layer_ids)
        torch.manual_seed(7)
        input_ids = torch.randint(2, VOCAB, (1, 21))
        with torch.no_grad():
            one_call = _generator(config, 24)(input_ids=input_ids).logits
            gen = _generator(config, 8)
            out = gen(input_ids=input_ids[:, :15])
            logits, past = [out.logits], out.past_key_values
            for pos in range(15, input_ids.shape[1]):
                out = gen(input_ids=input_ids[:, pos : pos + 1], past_key_values=past)
                logits.append(out.logits)
                past = out.past_key_values
        torch.testing.assert_close(torch.cat(logits, 1), one_call, rtol=0, atol=1e-6)

    def test_matches_stock(self):
        config = _config((1, 3))
        torch.manual_seed(7)
        input_ids = torch.randint(2, VOCAB, (1, 21))
        with torch.no_grad():
            reference = _model(config)(input_ids=input_ids, use_cache=False).logits
            got = _generator(config, 8)(input_ids=input_ids).logits
        # ExportableLinearAttention's kernel is approximate; the PLE path adds
        # nothing beyond it (see test_chunked_matches_one_call)
        torch.testing.assert_close(got, reference, rtol=0, atol=2e-3)

    def test_ngram_history_carries_across_calls(self):
        gen = _generator(_config((1,)), 8)
        input_ids = torch.randint(2, VOCAB, (1, 10))
        with torch.no_grad():
            out = gen(input_ids=input_ids)
        assert torch.equal(out.past_key_values.ple_token_history, input_ids[:, -2:])


def test_dynamo_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        ONNX_OPSET_VERSION,
        _dynamo_export,
        dynamic_axes_to_dynamic_shapes,
    )
    from GenAILab.qai_hub_lm.backends.torch.qwen4_exp import Qwen4Exp
    from GenAILab.qai_hub_lm.models.utils.exportable import (
        ONNXExportableModuleWithCache,
    )
    from GenAILab.qai_hub_lm.transforms.exportable_moe import forced_expert_activation

    config = _config((1, 3))
    model = _adapted(config)
    descriptors = build_layer_cache_descriptors(model.config)
    input_names = Qwen4Exp.get_backbone_input_names(descriptors)
    wrapped = ONNXExportableModuleWithCache(model, input_names=input_names).eval()
    sample = Qwen4Exp.get_sample_backbone_inputs(wrapped, 48, 8, descriptors)
    torch.manual_seed(3)
    sample = tuple(
        torch.randn_like(t) if name.startswith("ple_embeds_") else t
        for name, t in zip(input_names, sample)
    )
    axes = Qwen4Exp.get_backbone_dynamic_axes(descriptors)

    path = str(tmp_path / "backbone.onnx")
    with torch.no_grad(), forced_expert_activation(wrapped.model, phase="export"):
        _dynamo_export(
            wrapped,
            sample,
            path,
            input_names=list(input_names),
            output_names=list(Qwen4Exp.get_backbone_output_names(descriptors)),
            opset_version=ONNX_OPSET_VERSION,
            dynamic_shapes=dynamic_axes_to_dynamic_shapes(input_names, axes),
        )
        expected = wrapped(*sample)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    graph_inputs = {i.name for i in session.get_inputs()}
    assert {"ple_embeds_0", "ple_embeds_2", "ple_conv_state_0_in"} <= graph_inputs
    feeds = {n: t.numpy() for n, t in zip(input_names, sample) if n in graph_inputs}
    for got, want in zip(session.run(None, feeds), expected):
        got = torch.from_numpy(got)
        torch.testing.assert_close(got, want.to(got.dtype), rtol=0, atol=1e-5)


def test_registered_for_qwen4_exp_model_types():
    register_adaptations()
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        info = YAMLConfigParser._get_adaptation_info(model_type, "ExportablePLE")
        assert info.mixin_cls is Qwen4ExpExportablePLEAdaptation
        assert info.required_for_export
