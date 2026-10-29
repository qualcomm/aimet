# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the Qwen4-Exp ExportableQSA indexer against HuggingFace."""

import types

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
from GenAILab.qai_hub_lm.transforms.exportable_qsa import (  # noqa: E402
    Qwen4ExpExportableQSAAdaptation,
    _gather_selected_keys,
    _patch_qsa_instances,
    exportable_qsa_attention_forward,
    exportable_qsa_indexer_forward,
    register_adaptations,
)

VOCAB = 128


def _config(**overrides):
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
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
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[],
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        eos_token_id=1,
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


def _model(config, attn_implementation="sdpa"):
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    model = modeling_qwen4_exp.Qwen4ExpForCausalLM(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.abs().sum() == 0:
                param.normal_(std=0.05)
    return model


def _indexer_masks(indexer, hidden_states, position_embeddings, attention_mask):
    stock = indexer(hidden_states, position_embeddings, attention_mask, None)
    patched = exportable_qsa_indexer_forward(
        indexer, hidden_states, position_embeddings, attention_mask, None
    )
    return stock, patched


def _causal_mask(batch, seq, *, pad=0, dtype=torch.bool):
    keep = torch.ones(seq, seq, dtype=torch.bool).tril()
    mask = keep.expand(batch, 1, seq, seq).clone()
    if pad:
        mask[0, :, :, :pad] = False
    if dtype == torch.bool:
        return mask
    return torch.where(mask, 0.0, torch.finfo(dtype).min).to(dtype)


class TestIndexerMask:
    @pytest.fixture
    def indexer(self):
        model = _model(_config())
        return model.model.layers[1].self_attn.indexer

    @pytest.mark.parametrize("seq", [1, 3, 4, 13, 32])
    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
    def test_matches_stock(self, indexer, seq, dtype):
        torch.manual_seed(seq)
        batch = 2
        hidden_states = torch.randn(batch, seq, 32)
        cos, sin = torch.randn(2, batch, seq, 16).unbind(0)
        mask = _causal_mask(batch, seq, pad=min(5, seq - 1), dtype=dtype)

        with torch.no_grad():
            stock, patched = _indexer_masks(indexer, hidden_states, (cos, sin), mask)
        assert patched.dtype == stock.dtype
        assert torch.equal(patched, stock)

    def test_rejects_non_prefix_mask(self, indexer):
        hidden_states = torch.randn(1, 8, 32)
        cos, sin = torch.randn(2, 1, 8, 16).unbind(0)
        mask = _causal_mask(1, 8)
        mask[0, 0, 7, 2] = False

        with pytest.raises(ValueError, match="prefix"):
            exportable_qsa_indexer_forward(
                indexer, hidden_states, (cos, sin), mask, None
            )


class TestModel:
    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
    def test_logits_match_stock_with_padding(self, attn_implementation):
        stock = _model(_config(), attn_implementation)
        patched = _model(_config(), attn_implementation)
        _patch_qsa_instances(patched)

        torch.manual_seed(1)
        input_ids = torch.randint(2, VOCAB, (2, 24))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :5] = 0

        with torch.no_grad():
            ref = stock(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            out = patched(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        # Fully masked padding rows are NaN under eager in the stock model too
        torch.testing.assert_close(
            out.logits, ref.logits, rtol=0, atol=1e-5, equal_nan=True
        )

    def test_decode_with_cache_matches_stock(self):
        stock = _model(_config())
        patched = _model(_config())
        _patch_qsa_instances(patched)

        torch.manual_seed(2)
        input_ids = torch.randint(2, VOCAB, (1, 30))
        with torch.no_grad():
            for model in (stock, patched):
                cache = DynamicCache(config=model.config)
                logits = [
                    model(input_ids=input_ids[:, :20], past_key_values=cache).logits
                ]
                for pos in range(20, 30):
                    step = model(
                        input_ids=input_ids[:, pos : pos + 1], past_key_values=cache
                    )
                    logits.append(step.logits)
                model.collected = torch.cat(logits, dim=1)
        torch.testing.assert_close(
            patched.collected, stock.collected, rtol=0, atol=1e-5
        )

    def test_jit_trace(self):
        model = _model(_config())
        _patch_qsa_instances(model)
        input_ids = torch.randint(2, VOCAB, (1, 16))

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model(input_ids=input_ids, use_cache=False).logits

        wrapper = Wrapper(model)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (input_ids,), check_trace=False)
            torch.testing.assert_close(traced(input_ids), wrapper(input_ids))


@pytest.mark.parametrize(
    "module_cls, forward",
    [
        ("Qwen4ExpTextQSAIndexer", exportable_qsa_indexer_forward),
        ("Qwen4ExpTextAttention", exportable_qsa_attention_forward),
    ],
)
def test_patch_binds_every_qsa_module(module_cls, forward):
    model = _model(_config(layer_types=["qwen_sparse_attention"] * 2))
    _patch_qsa_instances(model)
    modules = [
        m
        for m in model.modules()
        if isinstance(m, getattr(modeling_qwen4_exp, module_cls))
    ]
    assert len(modules) == 2
    for module in modules:
        assert isinstance(module.forward, types.MethodType)
        assert module.forward.__func__ is forward


class TestGatherSelectedKeys:
    """Attention over the gathered keys equals attention over all keys under the
    same mask: the gather keeps every attended key, and adds only masked ones."""

    @staticmethod
    def _sparse_mask(batch, seq, kv, per_query, *, dtype, seed=0):
        gen = torch.Generator().manual_seed(seed)
        keep = torch.zeros(batch, 1, seq, kv, dtype=torch.bool)
        for b in range(batch):
            for s in range(seq):
                keep[b, 0, s, torch.randperm(kv, generator=gen)[:per_query]] = True
        if dtype == torch.bool:
            return keep
        return torch.where(keep, 0.0, torch.finfo(dtype).min).to(dtype)

    @staticmethod
    def _attend(q, k, v, mask):
        if mask.dtype == torch.bool:
            mask = torch.where(mask, 0.0, torch.finfo(q.dtype).min)
        weights = torch.softmax(q @ k.transpose(-1, -2) + mask, dim=-1)
        return weights @ v

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
    @pytest.mark.parametrize("seq", [1, 2, 3])
    def test_attention_unchanged(self, seq, dtype):
        batch, heads, kv, dim, per_query = 2, 2, 40, 8, 5
        torch.manual_seed(seq)
        q = torch.randn(batch, heads, seq, dim)
        k, v = torch.randn(2, batch, heads, kv, dim).unbind(0)
        mask = self._sparse_mask(batch, seq, kv, per_query, dtype=dtype, seed=seq)

        k_g, v_g, mask_g = _gather_selected_keys(k, v, mask, per_query)

        assert k_g.shape[-2] == v_g.shape[-2] == mask_g.shape[-1] == seq * per_query
        assert mask_g.dtype == mask.dtype
        torch.testing.assert_close(
            self._attend(q, k_g, v_g, mask_g),
            self._attend(q, k, v, mask),
            rtol=0,
            atol=1e-6,
        )

    def test_full_width_skips_the_gather(self):
        k, v = torch.randn(2, 1, 1, 12, 4).unbind(0)
        mask = self._sparse_mask(1, 4, 12, 3, dtype=torch.bool)
        k_g, v_g, mask_g = _gather_selected_keys(k, v, mask, 3)
        assert k_g is k and v_g is v and mask_g is mask


class TestGatheredAttention:
    """The whole model with ExportableQSA (indexer + gathered attention) matches
    stock HF when the cache outgrows a decode step's selection."""

    # budget 8 + ratio 4 - 1 = 11 keys per query < the 40-token cache below
    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
    def test_long_decode_matches_stock(self, attn_implementation):
        config = _config(layer_types=["qwen_sparse_attention"] * 2)
        stock = _model(config, attn_implementation)
        patched = _model(config, attn_implementation)
        _patch_qsa_instances(patched)

        torch.manual_seed(3)
        input_ids = torch.randint(2, VOCAB, (1, 48))
        with torch.no_grad():
            for model in (stock, patched):
                cache = DynamicCache(config=model.config)
                logits = [
                    model(input_ids=input_ids[:, :40], past_key_values=cache).logits
                ]
                for pos in range(40, 48):
                    step = model(
                        input_ids=input_ids[:, pos : pos + 1], past_key_values=cache
                    )
                    logits.append(step.logits)
                model.collected = torch.cat(logits, dim=1)
        torch.testing.assert_close(
            patched.collected, stock.collected, rtol=0, atol=1e-5
        )

    def test_decode_attends_to_the_gathered_width(self):
        config = _config(layer_types=["qwen_sparse_attention"] * 2)
        model = _model(config, "eager")
        _patch_qsa_instances(model)
        widths = []

        def record(module, args, kwargs, output):
            widths.append(output[1].shape[-1])  # attention weights: (B, H, S, width)

        handles = [
            layer.self_attn.register_forward_hook(record, with_kwargs=True)
            for layer in model.model.layers
        ]
        input_ids = torch.randint(2, VOCAB, (1, 41))
        with torch.no_grad():
            cache = DynamicCache(config=model.config)
            model(input_ids=input_ids[:, :40], past_key_values=cache)
            widths.clear()
            model(input_ids=input_ids[:, 40:], past_key_values=cache)
        for handle in handles:
            handle.remove()
        assert widths == [11, 11]  # min(41 keys, 1 query * (8 + 4 - 1))


def test_registered_for_qwen4_exp_model_types():
    # tests/unit/conftest.py wipes the registry before each test
    register_adaptations()
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        info = YAMLConfigParser._get_adaptation_info(model_type, "ExportableQSA")
        assert info.mixin_cls is Qwen4ExpExportableQSAAdaptation
        assert info.required_for_export
    assert "ExportableQSA" not in YAMLConfigParser.get_required_export_adaptations(
        "llama"
    )


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_backend_exports_with_dynamo(backend):
    import importlib

    module = importlib.import_module(
        f"GenAILab.qai_hub_lm.backends.{backend}.qwen4_exp"
    )
    assert module.Qwen4Exp.use_dynamo_export() is True


def test_adapted_model_class_exports_with_dynamo():
    from GenAILab.qai_hub_lm.backends.torch.qwen4_exp import Qwen4Exp

    # tests/unit/conftest.py wipes the registry before each test
    register_adaptations()
    YAMLConfigParser.model_lookup["qwen4_exp"] = Qwen4Exp
    model_cls = YAMLConfigParser.get_model_class("qwen4_exp", ["ExportableQSA"])
    assert issubclass(model_cls, Qwen4ExpExportableQSAAdaptation)
    assert issubclass(model_cls, Qwen4Exp)
    assert model_cls.use_dynamo_export() is True


class TestWithExportableMoE:
    """Qwen4-Exp is also MoE: ExportableMoE must unfuse its experts and compose
    with ExportableQSA. All-QSA layers keep this off the linear-attention path."""

    QSA_LAYERS = dict(layer_types=["qwen_sparse_attention"] * 2)

    @staticmethod
    def _inputs():
        torch.manual_seed(5)
        input_ids = torch.randint(2, VOCAB, (2, 20))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :4] = 0
        return input_ids, attention_mask

    def _adapted(self, execution):
        from GenAILab.qai_hub_lm.transforms.exportable_moe import replace_fused_experts

        model = _model(_config(**self.QSA_LAYERS))
        replaced = replace_fused_experts(model, execution=execution)
        _patch_qsa_instances(model)
        return model, replaced

    def test_every_experts_module_unfused(self):
        _, replaced = self._adapted("sparse")
        assert replaced == ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]

    @pytest.mark.parametrize("execution", ["sparse", "dense", "predicated"])
    def test_logits_match_stock(self, execution):
        input_ids, attention_mask = self._inputs()
        stock = _model(_config(**self.QSA_LAYERS))
        model, _ = self._adapted(execution)
        with torch.no_grad():
            ref = stock(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        real = attention_mask.bool()
        torch.testing.assert_close(
            out.logits[real], ref.logits[real], rtol=0, atol=1e-6
        )

    def test_dynamo_onnx_export_matches_torch(self):
        ort = pytest.importorskip("onnxruntime")
        from GenAILab.qai_hub_lm.transforms.exportable_moe import (
            forced_expert_activation,
        )

        class NoCache(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return self.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).logits

        model, _ = self._adapted("predicated")
        wrapper = NoCache(model)
        input_ids, attention_mask = (t[:1] for t in self._inputs())
        with forced_expert_activation(model, phase="export"):
            program = torch.onnx.export(
                wrapper, (input_ids, attention_mask), dynamo=True
            )
        # One If per expert per layer: the predicated realizer survived export
        assert sum(n.op_type == "If" for n in program.model_proto.graph.node) == 8

        session = ort.InferenceSession(program.model_proto.SerializeToString())
        names = [i.name for i in session.get_inputs()]
        (got,) = session.run(
            None, dict(zip(names, (input_ids.numpy(), attention_mask.numpy())))
        )
        with torch.no_grad():
            expected = wrapper(input_ids, attention_mask)
        # Fully masked pad rows are numerically arbitrary; compare real tokens
        real = attention_mask[0].bool()
        torch.testing.assert_close(
            torch.from_numpy(got)[0, real], expected[0, real], rtol=0, atol=1e-6
        )


class TestThroughGenerator:
    """The adapted model behind ONNXExportableModuleWithCache, driven by the
    Generator through chunked prefill and decode, as in evaluation and export.

    The wrapper must carry each QSA layer's indexer keys and the model's past
    positions as graph states, or a cached call sees only the current keys.
    """

    VOCAB_IDS = (1, 21)  # 21 tokens = 2 full chunks of 8 + a partial one

    @staticmethod
    def _wrapped(config):
        from GenAILab.qai_hub_lm.models.base import LLM
        from GenAILab.qai_hub_lm.models.utils.exportable import (
            ONNXExportableModuleWithCache,
        )
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            build_layer_cache_descriptors,
        )
        from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
            _patch_gated_delta_net_instances,
        )
        from GenAILab.qai_hub_lm.transforms.exportable_moe import replace_fused_experts

        model = _model(config)
        if "linear_attention" in config.layer_types:
            _patch_gated_delta_net_instances(model)
        # Dense for eval; predicated is what export bakes into the graph
        replace_fused_experts(model, execution="dense", export_execution="predicated")
        _patch_qsa_instances(model)
        names = LLM.get_backbone_input_names(
            build_layer_cache_descriptors(model.config)
        )
        return ONNXExportableModuleWithCache(model, input_names=names).eval()

    def _generate(self, config, sequence_length):
        from unittest.mock import MagicMock

        from GenAILab.qai_hub_lm.models.generator import Generator

        gen = Generator(
            model=self._wrapped(config),
            tokenizer=MagicMock(eos_token_id=1, pad_token_id=0),
            sequence_length=sequence_length,
            context_length=48,
        )
        torch.manual_seed(7)
        input_ids = torch.randint(2, VOCAB, self.VOCAB_IDS)
        with torch.no_grad():
            reference = _model(config)(input_ids=input_ids, use_cache=False).logits
            out = gen(input_ids=input_ids[:, :15])
            logits, past = [out.logits], out.past_key_values
            for pos in range(15, input_ids.shape[1]):
                out = gen(input_ids=input_ids[:, pos : pos + 1], past_key_values=past)
                logits.append(out.logits)
                past = out.past_key_values
        return torch.cat(logits, dim=1), reference

    @pytest.mark.parametrize("sequence_length", [8, 1])
    def test_qsa_matches_stock(self, sequence_length):
        got, reference = self._generate(
            _config(layer_types=["qwen_sparse_attention"] * 2), sequence_length
        )
        torch.testing.assert_close(got, reference, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("sequence_length", [8, 1])
    def test_hybrid_matches_stock(self, sequence_length):
        config = _config(
            layer_types=["linear_attention", "qwen_sparse_attention"] * 2,
            num_hidden_layers=4,
        )
        got, reference = self._generate(config, sequence_length)
        # ExportableLinearAttention's kernel is approximate: Qwen 3.5 behind
        # the same wrapper differs from stock by the same order
        torch.testing.assert_close(got, reference, rtol=0, atol=2e-3)


def test_hybrid_backbone_dynamo_export_matches_torch(tmp_path):
    """GatedDeltaNet + QSA + predicated MoE, exported the way the backend does:
    behind the cache wrapper, through ``_dynamo_export`` with dynamic axes."""
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        ONNX_OPSET_VERSION,
        _dynamo_export,
        dynamic_axes_to_dynamic_shapes,
    )
    from GenAILab.qai_hub_lm.models.base import LLM
    from GenAILab.qai_hub_lm.models.generator import Generator
    from GenAILab.qai_hub_lm.models.utils.layer_cache import (
        build_layer_cache_descriptors,
    )
    from GenAILab.qai_hub_lm.transforms.exportable_moe import forced_expert_activation

    config = _config(
        layer_types=["linear_attention", "qwen_sparse_attention"] * 2,
        num_hidden_layers=4,
    )
    wrapped = TestThroughGenerator._wrapped(config)
    descriptors = build_layer_cache_descriptors(wrapped.config)
    input_names = LLM.get_backbone_input_names(descriptors)
    torch.manual_seed(7)
    sample = Generator.prepare_inputs(
        model=wrapped,
        input_ids=torch.randint(2, VOCAB, (1, 8)),
        attention_mask=torch.ones(1, 8, dtype=torch.int32),
        past_key_values=[],
        sequence_length=8,
        context_length=48,
        pad_token=1,
        attention_mask_min=-100,
        inputs_embeds=None,
        position_ids=None,
        layer_cache_descriptors=descriptors,
    )
    args = tuple(sample.values())
    axes = LLM.get_backbone_dynamic_axes(descriptors)

    path = str(tmp_path / "backbone.onnx")
    with torch.no_grad(), forced_expert_activation(wrapped.model, phase="export"):
        _dynamo_export(
            wrapped,
            args,
            path,
            input_names=list(input_names),
            output_names=list(LLM.get_backbone_output_names(descriptors)),
            opset_version=ONNX_OPSET_VERSION,
            dynamic_shapes=dynamic_axes_to_dynamic_shapes(input_names, axes),
        )
        expected = wrapped(*args)

    def count_if(graph):
        n = 0
        for node in graph.node:
            n += node.op_type == "If"
            for attr in node.attribute:
                if attr.g.ByteSize():
                    n += count_if(attr.g)
        return n

    # One If per expert per layer: the predicated experts survived export
    assert count_if(onnx.load(path).graph) == config.num_experts * 4

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {i.name: sample[i.name].numpy() for i in session.get_inputs()}
    for got, want in zip(session.run(None, feeds), expected):
        got = torch.from_numpy(got)
        torch.testing.assert_close(got, want.to(got.dtype), rtol=0, atol=1e-5)
