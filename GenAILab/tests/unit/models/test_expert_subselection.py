# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Functional tests for the ExpertSubselection adaptation.

Organised around what the design relies on:

1. Decode (T=1) is bit-identical to the stock router, for any S > k.
2. Prefill (T>1) matches a straightforward per-sequence reference.
3. Subselection really changes prefill routing, even at S >= T*k.
4. S = E is the stock router at every T.

plus batch independence, registration guards, quantsim and ONNX export.
"""

import copy
import pytest

torch = pytest.importorskip("torch")
modeling_qwen3_moe = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe"
)

import torch.nn.functional as F  # noqa: E402

from GenAILab.bench.yaml_config_parser import YAMLConfigParser  # noqa: E402
from GenAILab.qai_hub_lm.transforms import expert_subselection  # noqa: E402
from GenAILab.qai_hub_lm.transforms import exportable_moe  # noqa: E402
from GenAILab.qai_hub_lm.transforms.expert_subselection import (  # noqa: E402
    ExpertSubselectionAdaptation,
    SubselectedTopKRouter,
    apply_expert_subselection,
)
from GenAILab.qai_hub_lm.transforms.exportable_moe import (  # noqa: E402
    PredicatedExperts,
    forced_expert_activation,
    replace_fused_experts,
)
from GenAILab.tests.unit.models.test_exportable_moe import (  # noqa: E402
    E,
    HIDDEN,
    INTER,
    TOPK,
    _init_params,
    _max_err,
    _toy_config,
)


def _stock_router(seed=0, **config_overrides):
    torch.manual_seed(seed)
    router = modeling_qwen3_moe.Qwen3MoeTopKRouter(_toy_config(**config_overrides))
    with torch.no_grad():
        router.weight.normal_(0, 1.0)
    return router


def _hidden(batch, seq_len, seed=1):
    return torch.randn(
        batch, seq_len, HIDDEN, generator=torch.Generator().manual_seed(seed)
    )


def _reference(router, hidden_states, num_selected):
    """Per sequence: keep the top-S experts by max-over-tokens, then top-k."""
    indices, scores = [], []
    for seq in hidden_states:
        probs = F.softmax(F.linear(seq, router.weight), dtype=torch.float, dim=-1)
        kept = probs.amax(dim=0).topk(num_selected).indices
        for token_probs in probs:
            among = token_probs[kept]
            value, pos = among.topk(router.top_k)
            if router.norm_topk_prob:
                value = value / value.sum()
            indices.append(kept[pos])
            scores.append(value)
    return torch.stack(indices), torch.stack(scores)


class TestDecodeIsExact:
    @pytest.mark.parametrize("num_selected", [TOPK + 1, E // 2, E])
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("norm_topk_prob", [True, False])
    def test_t1_bit_identical_to_stock(self, num_selected, batch, norm_topk_prob):
        stock = _stock_router(norm_topk_prob=norm_topk_prob)
        adapted = SubselectedTopKRouter(stock, num_selected)
        hs = _hidden(batch, 1)

        ref_logits, ref_scores, ref_indices = stock(hs.reshape(-1, HIDDEN))
        logits, scores, indices = adapted(hs)

        assert torch.equal(indices, ref_indices)
        assert torch.equal(scores, ref_scores)
        assert torch.equal(logits, ref_logits)


class TestPrefill:
    @pytest.mark.parametrize("num_selected", [TOPK + 1, 5, E])
    def test_matches_reference(self, num_selected):
        stock = _stock_router()
        adapted = SubselectedTopKRouter(stock, num_selected)
        hs = _hidden(2, 9)

        _, scores, indices = adapted(hs)
        ref_indices, ref_scores = _reference(stock, hs, num_selected)

        assert torch.equal(indices, ref_indices)
        assert _max_err(scores, ref_scores) < 1e-6

    def test_s_equals_e_is_stock_at_every_t(self):
        stock = _stock_router()
        adapted = SubselectedTopKRouter(stock, E)
        hs = _hidden(2, 11)
        _, ref_scores, ref_indices = stock(hs.reshape(-1, HIDDEN))
        _, scores, indices = adapted(hs)
        assert torch.equal(indices, ref_indices)
        assert _max_err(scores, ref_scores) < 1e-6

    def test_changes_routing_even_when_s_covers_t_times_k(self):
        """k=1, two tokens, S=2 >= T*k: B's own pick still loses to A's runner-up."""
        stock = _stock_router(num_experts=4, num_experts_per_tok=1, hidden_size=4)
        with torch.no_grad():
            stock.weight.copy_(torch.eye(4))
        # Logits = log-probabilities, so softmax gives these rows back.
        a = [0.50, 0.49, 0.005, 0.005]
        b = [0.01, 0.30, 0.35, 0.34]
        hs = torch.tensor([[a, b]]).log()

        _, _, ref_indices = stock(hs.reshape(-1, 4))
        _, _, indices = SubselectedTopKRouter(stock, 2)(hs)

        assert ref_indices.flatten().tolist() == [0, 2]
        # Peak scores [0.50, 0.49, 0.35, 0.34] keep {0, 1}; B falls back to 1.
        assert indices.flatten().tolist() == [0, 1]

    def test_sequences_in_a_batch_are_independent(self):
        """The max runs over T only; batching must not pool sequences."""
        stock = _stock_router()
        adapted = SubselectedTopKRouter(stock, TOPK + 1)
        hs = _hidden(3, 6)
        _, batched_scores, batched = adapted(hs)
        alone = torch.cat([adapted(hs[i : i + 1])[2] for i in range(3)])
        alone_scores = torch.cat([adapted(hs[i : i + 1])[1] for i in range(3)])
        assert torch.equal(batched, alone)
        assert torch.equal(batched_scores, alone_scores)


class TestValidation:
    @pytest.mark.parametrize("num_selected", [TOPK - 1, TOPK, E + 1])
    def test_out_of_range_s_rejected(self, num_selected):
        with pytest.raises(ValueError, match="num_selected_experts must be in"):
            SubselectedTopKRouter(_stock_router(), num_selected)

    def test_projection_shares_the_stock_weight(self):
        stock = _stock_router()
        assert SubselectedTopKRouter(stock, E).proj.weight is stock.weight

    def test_each_sequence_routes_among_exactly_s_experts(self):
        """With S = k + 1 and many tokens, a sequence's tokens jointly reach S."""
        router = SubselectedTopKRouter(_stock_router(), TOPK + 1)
        _, _, indices = router(_hidden(2, 32))
        per_sequence = indices.reshape(2, -1)
        assert [len(set(seq.tolist())) for seq in per_sequence] == [TOPK + 1] * 2

    def test_no_moe_block_raises(self):
        with pytest.raises(RuntimeError, match="found no MoE block"):
            apply_expert_subselection(torch.nn.Linear(2, 2), E)


class TestFullModel:
    @staticmethod
    def _model(seed=0):
        torch.manual_seed(seed)
        return modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config()).eval()

    def _adapted(self, num_selected, execution="sparse"):
        model = self._model()
        replaced = apply_expert_subselection(model, num_selected)
        replace_fused_experts(model, execution=execution)
        return model, replaced

    def test_every_moe_block_adapted(self):
        _, replaced = self._adapted(E)
        assert len(replaced) == 2
        assert all(name.endswith("mlp") for name in replaced)

    @pytest.mark.parametrize("execution", ["sparse", "dense", "predicated"])
    def test_decode_logits_match_stock(self, execution):
        ids = torch.randint(0, 128, (3, 1))
        with torch.no_grad():
            reference = self._model()(input_ids=ids).logits
            got = self._adapted(TOPK + 1, execution)[0](input_ids=ids).logits
        assert _max_err(got, reference) < 1e-5

    def test_prefill_logits_match_stock_when_off(self):
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            reference = self._model()(input_ids=ids).logits
            got = self._adapted(E)[0](input_ids=ids).logits
        assert _max_err(got, reference) < 1e-5

    def test_prefill_logits_change_when_on(self):
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            reference = self._model()(input_ids=ids).logits
            got = self._adapted(TOPK + 1)[0](input_ids=ids).logits
        assert _max_err(got, reference) > 1e-4

    def test_forced_activation_does_not_change_logits(self):
        """Calibration's force-all switch composes with subselection."""
        ids = torch.randint(0, 128, (1, 16))
        model, _ = self._adapted(TOPK + 1, execution="dense")
        with torch.no_grad():
            faithful = model(input_ids=ids).logits
            with forced_expert_activation(model, force_all=True):
                forced = model(input_ids=ids).logits
        assert _max_err(faithful, forced) < 1e-5


class TestQwen3_5Moe:
    """Qwen3.5-MoE: no norm_topk_prob flag, plus a shared expert in the block."""

    @staticmethod
    def _block():
        modeling = pytest.importorskip(
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
        )
        cfg_mod = pytest.importorskip(
            "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
        )
        config = cfg_mod.Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=HIDDEN,
            num_hidden_layers=2,
            num_experts=E,
            num_experts_per_tok=TOPK,
            moe_intermediate_size=INTER,
            shared_expert_intermediate_size=INTER,
            max_position_embeddings=64,
        )
        torch.manual_seed(0)
        return _init_params(modeling.Qwen3_5MoeSparseMoeBlock(config).eval())

    @pytest.mark.parametrize("num_selected,seq_len", [(TOPK + 1, 1), (E, 7)])
    def test_block_matches_stock(self, num_selected, seq_len):
        block = self._block()
        adapted = copy.deepcopy(block)
        apply_expert_subselection(adapted, num_selected)
        hs = _hidden(2, seq_len)
        with torch.no_grad():
            reference = block(hs)
            got = adapted(hs)
        assert _max_err(got, reference) < 1e-6

    def test_router_always_renormalizes(self):
        block = self._block()
        apply_expert_subselection(block, E)
        assert block.gate.norm_topk_prob is True


class TestRegistration:
    """The autouse registry fixture wipes ``adaptation_lookup`` before each test."""

    @pytest.fixture(autouse=True)
    def _register(self):
        exportable_moe.register_adaptations()
        expert_subselection.register_adaptations()

    @staticmethod
    def _cls(adaptations, kwargs):
        class FakeLLM:
            @classmethod
            def instantiate_model(cls, *args, **kw):
                torch.manual_seed(0)
                return modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config()).eval()

        YAMLConfigParser._default_llm_cls = FakeLLM
        return YAMLConfigParser.get_model_class("qwen3_moe", adaptations, kwargs)

    def test_registered_but_not_required_for_export(self):
        for model_type in ("qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"):
            info = YAMLConfigParser._get_adaptation_info(
                model_type, "ExpertSubselection"
            )
            assert info.mixin_cls is ExpertSubselectionAdaptation
            assert not info.required_for_export
            assert not info.exclusive
        assert (
            "ExpertSubselection"
            not in YAMLConfigParser.get_required_export_adaptations("qwen3_moe")
        )

    @pytest.mark.parametrize(
        "order",
        [
            ["ExportableMoE", "ExpertSubselection"],
            ["ExpertSubselection", "ExportableMoE"],
        ],
    )
    def test_composes_with_moe_in_either_order(self, order):
        cls = self._cls(order, {"ExpertSubselection": {"num_selected_experts": 4}})
        model = cls.instantiate_model("fake")
        blocks = [m for m in model.modules() if isinstance(m, SubselectedTopKRouter)]
        experts = [m for m in model.modules() if isinstance(m, PredicatedExperts)]
        assert len(blocks) == 2 and len(experts) == 2
        assert all(b.num_selected_experts == 4 for b in blocks)

    def test_requires_num_selected_experts(self):
        cls = self._cls(["ExportableMoE", "ExpertSubselection"], {})
        with pytest.raises(ValueError, match="needs `num_selected_experts`"):
            cls.instantiate_model("fake")

    def test_requires_moe(self):
        cls = self._cls(
            ["ExpertSubselection"], {"ExpertSubselection": {"num_selected_experts": 4}}
        )
        with pytest.raises(ValueError, match="needs the ExportableMoE adaptation"):
            cls.instantiate_model("fake")


class TestTorchQuantsimIntegration:
    CONTEXT_LENGTH = 64
    SEQ_LEN = 8

    @pytest.fixture(scope="class")
    def backend(self):
        pytest.importorskip("aimet_torch")
        from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

        return LLM_Torch

    def _sim(self, backend):
        torch.manual_seed(0)
        model = modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config()).eval()
        apply_expert_subselection(model, TOPK + 1)
        replace_fused_experts(model, execution="dense")
        sims = backend.instantiate_quantsim(model, self.CONTEXT_LENGTH, self.SEQ_LEN)
        return sims.backbone if hasattr(sims, "backbone") else sims["backbone"]

    def test_router_projection_quantized(self, backend):
        from aimet_torch.v2.nn.true_quant import QuantizationMixin

        sim = self._sim(backend)
        routers = [
            m for m in sim.model.modules() if isinstance(m, SubselectedTopKRouter)
        ]
        assert len(routers) == 2
        for router in routers:
            assert isinstance(router.proj, QuantizationMixin)
            assert router.proj.param_quantizers["weight"] is not None

    def test_calibrates_and_runs(self, backend):
        from aimet_torch.v2.nn import compute_encodings

        sim = self._sim(backend)
        inputs = backend.get_sample_backbone_inputs(
            sim.model, self.CONTEXT_LENGTH, self.SEQ_LEN
        )
        with forced_expert_activation(sim), torch.no_grad():
            with compute_encodings(sim.model):
                sim.model(*inputs)
        with torch.no_grad():
            out = sim.model(*inputs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        assert torch.isfinite(logits).all()


class TestOnnxExport:
    """One export with symbolic T serves decode and prefill."""

    @pytest.fixture(scope="class")
    def exported(self, tmp_path_factory):
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        torch.manual_seed(0)
        block = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock(_toy_config()).eval()
        _init_params(block)
        apply_expert_subselection(block, TOPK + 1)
        replace_fused_experts(block, execution="predicated")

        path = str(tmp_path_factory.mktemp("subselect_onnx") / "block.onnx")
        with torch.no_grad():
            torch.onnx.export(
                block,
                (_hidden(1, 8),),
                path,
                input_names=["hidden_states"],
                output_names=["out"],
                opset_version=20,
                dynamo=True,
                dynamic_shapes=({1: torch.export.Dim.AUTO},),
            )
        return {"block": block, "path": path, "model": onnx.load(path)}

    @staticmethod
    def _run(path, hs):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return sess.run(None, {"hidden_states": hs.numpy()})[0]

    def test_sequence_axis_is_symbolic(self, exported):
        dim = exported["model"].graph.input[0].type.tensor_type.shape.dim[1]
        assert dim.dim_param and not dim.dim_value

    @pytest.mark.parametrize("seq_len", [1, 8, 5])
    def test_ort_matches_torch_at_every_length(self, exported, seq_len):
        hs = _hidden(1, seq_len, seed=seq_len)
        with torch.no_grad():
            expected = exported["block"](hs).numpy()
        got = self._run(exported["path"], hs)
        assert abs(got - expected).max() < 1e-5
