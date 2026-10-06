# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Functional tests for the predicated-experts ExportableMoE adaptation.

Organised around four equalities the adaptation must satisfy:

1. ``sparse`` == ``dense`` == ``predicated`` at ``force_all = 0``.
2. ``force_all = 1`` == ``force_all = 0`` in output.
3. ``dense`` == stock HuggingFace ``Experts.forward``.
4. An exported graph patched to ``force_all = 1`` == the same graph at 0.

plus structure (unfusing, layout guards, selector) and integration.
"""

import pytest

torch = pytest.importorskip("torch")
modeling_qwen3_moe = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe"
)

import torch.nn as nn  # noqa: E402
from transformers.models.qwen3_moe.configuration_qwen3_moe import (  # noqa: E402
    Qwen3MoeConfig,
)

from GenAILab.bench.yaml_config_parser import YAMLConfigParser  # noqa: E402
from GenAILab.qai_hub_lm.transforms.exportable_moe import (  # noqa: E402
    EXECUTION_MODES,
    assert_experts_quantized,
    unquantized_subgraph_ops,
    ExpertMLP,
    ExportableMoEAdaptation,
    PredicatedExperts,
    forced_expert_activation,
    is_fused_experts,
    predicated_expert_modules,
    register_adaptations,
    replace_fused_experts,
    set_onnx_force_all,
)

E, TOPK, HIDDEN, INTER = 8, 2, 32, 16
TOL = 1e-5


def _toy_config(**overrides):
    """Tiny qwen3_moe config: real module code, seconds to run."""
    kwargs = dict(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=E,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=INTER,
        decoder_sparse_step=1,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return Qwen3MoeConfig(**kwargs)


def _fused_experts(seed=0):
    """A stock (fused, 3D-parameter) Qwen3MoeExperts with random weights."""
    torch.manual_seed(seed)
    experts = modeling_qwen3_moe.Qwen3MoeExperts(_toy_config())
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.05)
        experts.down_proj.normal_(0, 0.05)
    return experts


def _routing(num_tokens, *, seed=0, top_k=TOPK, num_experts=E):
    """Random top-k routing in the shape the stock router emits."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=gen)
    probs = torch.softmax(logits, dim=-1)
    weights, index = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return index, weights


def _adapted(fused, **kwargs):
    return PredicatedExperts(fused, **kwargs)


def _max_err(a, b):
    return (a - b).abs().max().item()


def _init_params(module, *, seed=0):
    """Initialize parameters: direct construction skips HuggingFace's weight
    init and ``Experts`` allocates with ``torch.empty``, so outputs are NaN."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0, 0.05)
    return module


def _poisoned(*, expert, execution="predicated", selection="routed"):
    """One expert's weights set to NaN, so executing it is observable.

    torch.cond's captured branches never fire module hooks, hence the poison.
    """
    adapted = _adapted(_fused_experts(), execution=execution, selection=selection)
    with torch.no_grad():
        adapted.experts[expert].gate_proj.weight.fill_(float("nan"))
    return adapted


class TestUnfusing:
    """Structure: the per-expert leaves must hold exactly the fused slices."""

    def test_weights_match_fused_slices_exactly(self):
        fused = _fused_experts()
        adapted = _adapted(fused)
        for e in range(E):
            expert = adapted.experts[e]
            assert torch.equal(expert.gate_proj.weight, fused.gate_up_proj[e, :INTER])
            assert torch.equal(expert.up_proj.weight, fused.gate_up_proj[e, INTER:])
            assert torch.equal(expert.down_proj.weight, fused.down_proj[e])

    def test_leaf_count_and_names(self):
        adapted = _adapted(_fused_experts())
        linears = [
            name for name, mod in adapted.named_modules() if isinstance(mod, nn.Linear)
        ]
        assert len(linears) == 3 * E
        # Names must match Qwen3MoeMLP so downstream role classification works.
        assert "experts.0.gate_proj" in linears
        assert "experts.0.up_proj" in linears
        assert "experts.0.down_proj" in linears

    def test_weights_are_copies_not_views(self):
        """In-place weight rewrites (SpinQuant/AdaScale) must not alias."""
        fused = _fused_experts()
        adapted = _adapted(fused)
        before = fused.gate_up_proj[1, :INTER].clone()
        with torch.no_grad():
            adapted.experts[1].gate_proj.weight.mul_(3.0)
        assert torch.equal(fused.gate_up_proj[1, :INTER], before)
        assert not torch.equal(adapted.experts[1].gate_proj.weight, before)

    def test_per_expert_weights_are_independent(self):
        adapted = _adapted(_fused_experts())
        other = adapted.experts[2].gate_proj.weight.clone()
        with torch.no_grad():
            adapted.experts[0].gate_proj.weight.zero_()
        assert torch.equal(adapted.experts[2].gate_proj.weight, other)


class TestLayoutGuards:
    def test_duck_typing_accepts_fused_experts(self):
        assert is_fused_experts(_fused_experts())

    def test_duck_typing_rejects_plain_module(self):
        assert not is_fused_experts(nn.Linear(4, 4))
        assert not is_fused_experts(ExpertMLP(4, 8, nn.SiLU(), torch.float32, "cpu"))

    def test_transposed_layout_rejected(self):
        """A gpt_oss-shaped [E, H, 2I] layout must fail loudly, not mis-split."""
        fused = _fused_experts()
        with torch.no_grad():
            fused.gate_up_proj = nn.Parameter(torch.zeros(E, HIDDEN, 2 * INTER))
            fused.down_proj = nn.Parameter(torch.zeros(E, INTER, HIDDEN))
        with pytest.raises(
            NotImplementedError, match="Unsupported fused-expert layout"
        ):
            _adapted(fused)

    def test_biased_experts_rejected(self):
        fused = _fused_experts()
        fused.gate_up_proj_bias = nn.Parameter(torch.zeros(E, 2 * INTER))
        with pytest.raises(NotImplementedError, match="biased experts"):
            _adapted(fused)

    def test_replace_on_non_moe_model_raises(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        with pytest.raises(RuntimeError, match="found no fused-experts module"):
            replace_fused_experts(model)


class TestSelector:
    def test_scatter_puts_weights_at_routed_columns_and_zero_elsewhere(self):
        adapted = _adapted(_fused_experts())
        hs = torch.randn(6, HIDDEN)
        index, weights = _routing(6)
        selected = adapted._selection_weights(hs, index, weights)

        assert selected.shape == (6, E)
        for t in range(6):
            for k in range(TOPK):
                assert selected[t, index[t, k]] == pytest.approx(
                    weights[t, k].item(), abs=1e-6
                )
            off = [e for e in range(E) if e not in index[t].tolist()]
            assert torch.all(selected[t, off] == 0.0)

    def test_rows_sum_to_one(self):
        adapted = _adapted(_fused_experts())
        hs = torch.randn(5, HIDDEN)
        selected = adapted._selection_weights(hs, *_routing(5))
        assert torch.allclose(selected.sum(-1), torch.ones(5), atol=1e-6)

    def test_sentinel_index_is_inert(self):
        """qwen4_exp-style index == num_experts must not scatter out of bounds."""
        adapted = _adapted(_fused_experts())
        hs = torch.randn(3, HIDDEN)
        index, weights = _routing(3)
        index = index.clone()
        index[:, 0] = E  # the dropped-token sentinel
        selected = adapted._selection_weights(hs, index, weights)
        # Only the surviving (non-sentinel) slot contributes.
        assert torch.allclose(selected.sum(-1), weights[:, 1], atol=1e-6)

    def test_input_mask_is_routed_mask_when_not_forcing(self):
        adapted = _adapted(_fused_experts(), selection="routed")
        weight_e = torch.tensor([[0.0], [0.5], [0.0]])
        assert torch.equal(
            adapted._input_mask(weight_e), torch.tensor([[0.0], [1.0], [0.0]])
        )

    def test_input_mask_is_all_ones_when_forcing(self):
        adapted = _adapted(_fused_experts(), selection="all")
        weight_e = torch.tensor([[0.0], [0.5], [0.0]])
        assert torch.equal(adapted._input_mask(weight_e), torch.ones(3, 1))


class TestFaithfulnessContract:
    """The four equalities the whole design rests on."""

    @pytest.mark.parametrize("num_tokens", [1, 7, 64])
    def test_contract_1_all_three_realizers_agree(self, num_tokens):
        fused = _fused_experts()
        hs = torch.randn(num_tokens, HIDDEN)
        index, weights = _routing(num_tokens)

        outs = {}
        for execution in EXECUTION_MODES:
            adapted = _adapted(fused, execution=execution)
            outs[execution] = adapted(hs, index, weights)

        assert _max_err(outs["sparse"], outs["dense"]) < TOL
        assert _max_err(outs["sparse"], outs["predicated"]) < TOL

    def test_contract_2_force_all_does_not_change_output(self):
        fused = _fused_experts()
        hs = torch.randn(16, HIDDEN)
        index, weights = _routing(16)

        faithful = _adapted(fused, execution="predicated", selection="routed")
        forcing = _adapted(fused, execution="predicated", selection="all")
        assert _max_err(faithful(hs, index, weights), forcing(hs, index, weights)) < TOL

        dense_routed = _adapted(fused, execution="dense", selection="routed")
        dense_all = _adapted(fused, execution="dense", selection="all")
        assert (
            _max_err(dense_routed(hs, index, weights), dense_all(hs, index, weights))
            < TOL
        )

    def test_contract_3_dense_matches_stock_huggingface(self):
        fused = _fused_experts()
        hs = torch.randn(12, HIDDEN)
        index, weights = _routing(12)

        reference = fused(hs, index, weights)
        for execution in EXECUTION_MODES:
            adapted = _adapted(fused, execution=execution)
            assert _max_err(adapted(hs, index, weights), reference) < TOL

    def test_routed_mode_zeroes_unrouted_expert_inputs(self):
        """The mechanism behind contract 2: masked rows produce exactly zero."""
        fused = _fused_experts()
        adapted = _adapted(fused, execution="dense", selection="routed")
        hs = torch.randn(4, HIDDEN)
        # Route every token to expert 0 only.
        index = torch.zeros(4, 1, dtype=torch.long)
        weights = torch.ones(4, 1)
        captured = {}

        def hook(_module, inputs, _output):
            captured["x"] = inputs[0].detach().clone()

        adapted.experts[3].gate_proj.register_forward_hook(hook)
        adapted(hs, index, weights)
        assert torch.all(captured["x"] == 0.0), "unrouted expert saw nonzero input"

    def test_all_mode_feeds_every_expert_real_tokens(self):
        fused = _fused_experts()
        adapted = _adapted(fused, execution="dense", selection="all")
        hs = torch.randn(4, HIDDEN)
        index = torch.zeros(4, 1, dtype=torch.long)
        weights = torch.ones(4, 1)
        captured = {}

        def hook(_module, inputs, _output):
            captured["x"] = inputs[0].detach().clone()

        adapted.experts[3].gate_proj.register_forward_hook(hook)
        adapted(hs, index, weights)
        assert torch.equal(captured["x"], hs), "forced expert did not see real tokens"

    def test_every_expert_runs_when_forcing(self):
        """force_all must light up all E experts even with top_k = 1."""
        index = torch.zeros(4, 1, dtype=torch.long)  # everything to expert 0
        weights = torch.ones(4, 1)
        hs = torch.randn(4, HIDDEN)

        faithful = _poisoned(expert=5, selection="routed")
        assert torch.isfinite(faithful(hs, index, weights)).all(), (
            "expert 5 ran despite no token routing to it"
        )

        forcing = _poisoned(expert=5, selection="all")
        assert not torch.isfinite(forcing(hs, index, weights)).all(), (
            "force_all did not run expert 5"
        )


class TestExecutionPolicy:
    def test_sparse_with_force_all_raises(self):
        adapted = _adapted(_fused_experts(), execution="sparse", selection="all")
        with pytest.raises(RuntimeError, match="cannot honour force_all"):
            adapted(torch.randn(3, HIDDEN), *_routing(3))

    def test_invalid_modes_rejected(self):
        with pytest.raises(ValueError, match="execution must be"):
            _adapted(_fused_experts(), execution="nonsense")
        with pytest.raises(ValueError, match="selection must be"):
            _adapted(_fused_experts(), selection="nonsense")

    def test_only_hit_experts_run_in_predicated_mode(self):
        """The skip is real in torch too: a NaN-poisoned unrouted expert would
        produce NaN even on a zeroed input, so finite output means it ran not."""
        hs = torch.randn(1, HIDDEN)
        index = torch.tensor([[1, 3]])
        weights = torch.tensor([[0.6, 0.4]])

        skipping = _poisoned(expert=5, execution="predicated")
        assert torch.isfinite(skipping(hs, index, weights)).all()

        # Control: the same poison DOES surface under dense execution, which
        # proves the probe is sensitive rather than the NaN being masked.
        dense = _poisoned(expert=5, execution="dense")
        assert not torch.isfinite(dense(hs, index, weights)).all()


class TestForcedExpertActivation:
    def test_sets_and_restores(self):
        adapted = _adapted(_fused_experts(), execution="sparse", selection="routed")
        model = nn.Sequential(adapted)

        with forced_expert_activation(model) as touched:
            assert len(touched) == 1
            assert adapted.execution == "dense"  # calibration_execution default
            assert float(adapted.force_all) == 0.0  # selection == "routed"
        assert adapted.execution == "sparse"

    def test_honours_explicit_overrides(self):
        adapted = _adapted(_fused_experts())
        model = nn.Sequential(adapted)
        with forced_expert_activation(model, execution="predicated", force_all=True):
            assert adapted.execution == "predicated"
            assert float(adapted.force_all) == 1.0
        assert adapted.execution == "sparse"
        assert float(adapted.force_all) == 0.0

    def test_restores_on_exception(self):
        adapted = _adapted(_fused_experts())
        model = nn.Sequential(adapted)
        with pytest.raises(ValueError):
            with forced_expert_activation(model):
                raise ValueError("boom")
        assert adapted.execution == "sparse"

    def test_selection_all_forces_by_default(self):
        adapted = _adapted(_fused_experts(), selection="all")
        with forced_expert_activation(nn.Sequential(adapted)):
            assert float(adapted.force_all) == 1.0

    def test_no_op_on_non_moe_objects(self):
        with forced_expert_activation(nn.Linear(2, 2)) as touched:
            assert touched == []
        with forced_expert_activation(object()) as touched:
            assert touched == []

    def test_unwraps_quantsim_like_objects(self):
        adapted = _adapted(_fused_experts())

        class FakeSim:
            def __init__(self, model):
                self.model = model

        assert len(predicated_expert_modules(FakeSim(nn.Sequential(adapted)))) == 1


class TestFullModel:
    """End-to-end on a real (tiny) Qwen3MoE stack."""

    @staticmethod
    def _model(seed=0):
        torch.manual_seed(seed)
        model = modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config())
        return model.eval()

    def test_replacement_covers_every_moe_layer(self):
        model = self._model()
        replaced = replace_fused_experts(model)
        assert len(replaced) == 2  # both decoder layers are sparse (step=1)
        assert all(name.endswith("mlp.experts") for name in replaced)
        assert not any(is_fused_experts(m) for m in model.modules())

    @pytest.mark.parametrize("execution", EXECUTION_MODES)
    @pytest.mark.parametrize("seq_len", [1, 16])
    def test_logits_match_stock_model(self, execution, seq_len):
        ids = torch.randint(0, 128, (1, seq_len))
        reference = self._model()(input_ids=ids).logits

        adapted_model = self._model()
        replace_fused_experts(adapted_model, execution=execution)
        got = adapted_model(input_ids=ids).logits
        assert _max_err(got, reference) < 1e-4

    def test_forcing_does_not_change_logits(self):
        ids = torch.randint(0, 128, (1, 16))
        model = self._model()
        replace_fused_experts(model, execution="predicated", selection="routed")
        faithful = model(input_ids=ids).logits
        with forced_expert_activation(model, execution="predicated", force_all=True):
            forced = model(input_ids=ids).logits
        assert _max_err(faithful, forced) < 1e-4


class TestRegistration:
    """The autouse ``_isolate_registry`` fixture in tests/unit/conftest.py wipes
    ``adaptation_lookup`` before each test, so these re-register explicitly."""

    @pytest.fixture(autouse=True)
    def _register(self):
        register_adaptations()
        from GenAILab.qai_hub_lm.transforms import exportable_linear_attention

        exportable_linear_attention.register_adaptations()

    def test_registered_for_moe_model_types(self):
        for model_type in ("qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"):
            info = YAMLConfigParser._get_adaptation_info(model_type, "ExportableMoE")
            assert info.mixin_cls is ExportableMoEAdaptation
            assert info.required_for_export
            assert not info.exclusive

    def test_not_required_for_non_moe_models(self):
        """A "*" registration would wrongly demand MoE for every export."""
        assert "ExportableMoE" not in YAMLConfigParser.get_required_export_adaptations(
            "llama"
        )
        assert "ExportableMoE" in YAMLConfigParser.get_required_export_adaptations(
            "qwen3_moe"
        )

    def test_yaml_kwargs_become_class_attributes(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM
        cls = YAMLConfigParser.get_model_class(
            "qwen3_moe",
            adaptations=["ExportableMoE"],
            adaptation_kwargs={"ExportableMoE": {"selection": "all"}},
        )
        assert cls.selection == "all"

    def test_mixin_requires_dynamo_export(self):
        assert ExportableMoEAdaptation.use_dynamo_export() is True

    def test_linear_attention_registered_for_moe_types(self):
        """Qwen3.5-MoE is half GatedDeltaNet; it needs both adaptations."""
        pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        for model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"):
            info = YAMLConfigParser._get_adaptation_info(
                model_type, "ExportableLinearAttention"
            )
            assert info.required_for_export


class TestOnnxForceAllPatching:
    def test_raises_when_initializer_absent(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        path = str(tmp_path / "no_force_all.onnx")
        onnx.save(onnx.helper.make_model(graph), path)
        with pytest.raises(RuntimeError, match="No 'force_all' initializer"):
            set_onnx_force_all(path, True)

    @staticmethod
    def _graph_with_force_all(op_type, path):
        onnx = pytest.importorskip("onnx")
        import numpy as np
        from onnx import numpy_helper

        init = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "force_all")
        other = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "other")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(op_type, ["other", "force_all"], ["y"])],
            "g",
            [],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
            initializer=[init, other],
        )
        onnx.save(onnx.helper.make_model(graph), path)

    def test_patches_initializer_value(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from onnx import numpy_helper

        path = str(tmp_path / "force_all.onnx")
        self._graph_with_force_all("Max", path)

        assert set_onnx_force_all(path, True) == 1
        reloaded = onnx.load(path)
        init = next(i for i in reloaded.graph.initializer if i.name == "force_all")
        assert float(numpy_helper.to_array(init)[0]) == 1.0

        set_onnx_force_all(path, False)
        reloaded = onnx.load(path)
        init = next(i for i in reloaded.graph.initializer if i.name == "force_all")
        assert float(numpy_helper.to_array(init)[0]) == 0.0

    def test_guard_rejects_foreign_consumer(self, tmp_path):
        """If dedup merged force_all into another op, refuse rather than corrupt."""
        path = str(tmp_path / "shared.onnx")
        self._graph_with_force_all("Greater", path)
        with pytest.raises(RuntimeError, match="consumed by \\['Greater'\\]"):
            set_onnx_force_all(path, True)

    def test_rejects_unexpected_dims(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        import numpy as np
        from onnx import numpy_helper

        init = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), "force_all")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Max", ["force_all", "force_all"], ["y"])],
            "g",
            [],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])],
            initializer=[init],
        )
        path = str(tmp_path / "wrong_dims.onnx")
        onnx.save(onnx.helper.make_model(graph), path)
        with pytest.raises(RuntimeError, match="expected \\[1\\]"):
            set_onnx_force_all(path, True)


class TestOnnxExport:
    """Contract 4, plus the graph structure that makes it meaningful.

    Exported once per module: dynamo export dominates the runtime of this file.
    """

    @pytest.fixture(scope="class")
    def exported(self, tmp_path_factory):
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        fused = _fused_experts()
        mod = PredicatedExperts(
            fused, execution="predicated", selection="routed"
        ).eval()

        hs = torch.randn(6, HIDDEN)
        index, weights = _routing(6)
        with torch.no_grad():
            faithful = mod(hs, index, weights)
            mod.force_all.fill_(1.0)
            forced = mod(hs, index, weights)
            mod.force_all.fill_(0.0)

        path = str(tmp_path_factory.mktemp("moe_onnx") / "experts.onnx")
        with torch.no_grad():
            torch.onnx.export(
                mod,
                (hs, index, weights),
                path,
                input_names=["hidden_states", "top_k_index", "top_k_weights"],
                output_names=["out"],
                opset_version=20,
                dynamo=True,
            )
        feeds = {
            "hidden_states": hs.numpy(),
            "top_k_index": index.numpy().astype("int64"),
            "top_k_weights": weights.numpy(),
        }
        return {
            "path": path,
            "model": onnx.load(path),
            "feeds": feeds,
            "torch_faithful": faithful.numpy(),
            "torch_forced": forced.numpy(),
        }

    @staticmethod
    def _count(graph, op_type):
        n = 0
        for node in graph.node:
            if node.op_type == op_type:
                n += 1
            for attr in node.attribute:
                if attr.g.ByteSize():
                    n += TestOnnxExport._count(attr.g, op_type)
                for sub in attr.graphs:
                    n += TestOnnxExport._count(sub, op_type)
        return n

    @staticmethod
    def _run(path, feeds):
        import onnxruntime as ort

        so = ort.SessionOptions()
        # Basic only: do not let extended folding rewrite the If nodes away.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return sess.run(None, feeds)[0]

    def test_one_if_per_expert(self, exported):
        assert self._count(exported["model"].graph, "If") == E

    def test_three_matmuls_per_expert(self, exported):
        graph = exported["model"].graph
        matmuls = self._count(graph, "MatMul") + self._count(graph, "Gemm")
        assert matmuls == 3 * E

    def test_ort_matches_torch(self, exported):
        got = self._run(exported["path"], exported["feeds"])
        assert abs(got - exported["torch_faithful"]).max() < 1e-5

    def test_contract_4_force_all_flip_preserves_output(self, exported, tmp_path):
        """One graph, both policies, identical output."""
        import os
        import shutil

        # Whole directory: initializers live in a sidecar .onnx.data.
        dest = tmp_path / "copy"
        shutil.copytree(os.path.dirname(exported["path"]), dest)
        path = str(dest / os.path.basename(exported["path"]))

        before = self._run(path, exported["feeds"])
        assert set_onnx_force_all(path, True) >= 1
        after = self._run(path, exported["feeds"])

        assert abs(after - exported["torch_forced"]).max() < 1e-5
        assert abs(after - before).max() < 1e-5

    def test_force_all_is_only_consumed_by_max(self, exported):
        """Regression: deduplication merged force_all with a ``w > 0`` constant, so
        patching it rewrote the predicate to ``w > 1``. It must feed only Max.
        """
        from GenAILab.qai_hub_lm.transforms.exportable_moe import (
            _force_all_consumers,
        )

        assert _force_all_consumers(exported["model"].graph) == {"Max"}

    def test_force_all_buffer_is_not_scalar(self, exported):
        """Shape [1] cannot be merged with a 0-d comparison constant."""
        from onnx import numpy_helper

        inits = [
            numpy_helper.to_array(i)
            for i in exported["model"].graph.initializer
            if "force_all" in i.name
        ]
        assert inits, "force_all initializer was folded away"
        assert all(a.shape == (1,) for a in inits)

    def test_decode_shape_exports_and_matches(self, tmp_path):
        """T = 1: the shape where expert skipping matters most."""
        pytest.importorskip("onnxruntime")
        mod = PredicatedExperts(_fused_experts(), execution="predicated").eval()
        hs = torch.randn(1, HIDDEN)
        index, weights = _routing(1)
        with torch.no_grad():
            expected = mod(hs, index, weights)

        path = str(tmp_path / "decode.onnx")
        with torch.no_grad():
            torch.onnx.export(
                mod,
                (hs, index, weights),
                path,
                input_names=["hidden_states", "top_k_index", "top_k_weights"],
                output_names=["out"],
                opset_version=20,
                dynamo=True,
            )
        got = self._run(
            path,
            {
                "hidden_states": hs.numpy(),
                "top_k_index": index.numpy().astype("int64"),
                "top_k_weights": weights.numpy(),
            },
        )
        assert abs(got - expected.numpy()).max() < 1e-5


class TestQwen35Moe:
    """Qwen3.5-MoE: adds an always-active ``shared_expert`` (left alone) and
    needs ExportableLinearAttention alongside, from its own namespace."""

    @staticmethod
    def _modeling():
        return pytest.importorskip(
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
        )

    @staticmethod
    def _config(**overrides):
        cfg_mod = pytest.importorskip(
            "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
        )
        kwargs = dict(
            vocab_size=128,
            hidden_size=HIDDEN,
            num_hidden_layers=2,
            num_experts=E,
            num_experts_per_tok=TOPK,
            moe_intermediate_size=INTER,
            shared_expert_intermediate_size=INTER,
            max_position_embeddings=64,
        )
        kwargs.update(overrides)
        return cfg_mod.Qwen3_5MoeTextConfig(**kwargs)

    def _fused(self):
        modeling = self._modeling()
        torch.manual_seed(0)
        experts = modeling.Qwen3_5MoeExperts(self._config())
        with torch.no_grad():
            experts.gate_up_proj.normal_(0, 0.05)
            experts.down_proj.normal_(0, 0.05)
        return experts

    def test_same_fused_layout_as_qwen3_moe(self):
        """One duck-typed adaptation covers both families."""
        assert is_fused_experts(self._fused())

    @pytest.mark.parametrize("execution", EXECUTION_MODES)
    def test_realizers_match_stock(self, execution):
        fused = self._fused()
        hs = torch.randn(9, HIDDEN)
        index, weights = _routing(9)
        reference = fused(hs, index, weights)
        adapted = PredicatedExperts(fused, execution=execution)
        assert _max_err(adapted(hs, index, weights), reference) < TOL

    def test_shared_expert_is_left_alone(self):
        """Only the routed experts are replaced; the shared MLP is dense already."""
        modeling = self._modeling()
        torch.manual_seed(0)
        block = modeling.Qwen3_5MoeSparseMoeBlock(self._config())
        replaced = replace_fused_experts(block)

        assert replaced == ["experts"]
        assert isinstance(block.experts, PredicatedExperts)
        assert isinstance(block.shared_expert, modeling.Qwen3_5MoeMLP)
        assert isinstance(block.gate, modeling.Qwen3_5MoeTopKRouter)

    @pytest.mark.parametrize("execution", EXECUTION_MODES)
    def test_block_output_matches_stock(self, execution):
        """Shared expert + router + routed experts, end to end."""
        import copy

        modeling = self._modeling()
        torch.manual_seed(0)
        block = modeling.Qwen3_5MoeSparseMoeBlock(self._config()).eval()
        # Initialize explicitly, then deep-copy so both sides share weights.
        _init_params(block)
        adapted_block = copy.deepcopy(block)
        replace_fused_experts(adapted_block, execution=execution)

        hs = torch.randn(1, 7, HIDDEN)
        with torch.no_grad():
            reference = block(hs)
            got = adapted_block(hs)
        assert torch.isfinite(reference).all(), "reference block produced NaN"
        assert _max_err(got, reference) < 1e-4

    def test_linear_attention_patches_moe_gated_delta_net(self):
        """ExportableLinearAttention must reach Qwen3.5-MoE's own GDN class."""
        pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        modeling = self._modeling()
        from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
            _patch_gated_delta_net_instances,
            exportable_gated_delta_net_forward,
        )

        torch.manual_seed(0)
        model = modeling.Qwen3_5MoeForCausalLM(self._config()).eval()
        gdns = [
            m
            for m in model.modules()
            if isinstance(m, modeling.Qwen3_5MoeGatedDeltaNet)
        ]
        assert gdns, "toy config has no linear-attention layer"

        _patch_gated_delta_net_instances(model)
        assert all(
            m.forward.__func__ is exportable_gated_delta_net_forward for m in gdns
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestDevicePlacement:
    """Regression: the adaptation replaces modules inside an already-placed
    model, so nothing calls .to() on the new module afterwards. A CPU-resident
    force_all buffer therefore failed at the first forward of a CUDA model with
    "found at least two devices" -- invisible to CPU-only tests, immediate on a
    real one.
    """

    def test_buffer_follows_expert_device(self):
        fused = _fused_experts().to("cuda")
        adapted = PredicatedExperts(fused, execution="dense")
        assert adapted.force_all.device.type == "cuda"

    @pytest.mark.parametrize("execution", EXECUTION_MODES)
    def test_forward_on_cuda(self, execution):
        fused = _fused_experts().to("cuda")
        adapted = PredicatedExperts(fused, execution=execution)
        hs = torch.randn(5, HIDDEN, device="cuda")
        index, weights = _routing(5)
        out = adapted(hs, index.to("cuda"), weights.to("cuda"))
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()

    def test_matches_cpu_result(self):
        fused = _fused_experts()
        hs = torch.randn(5, HIDDEN)
        index, weights = _routing(5)
        cpu_out = PredicatedExperts(fused, execution="predicated")(hs, index, weights)

        cuda_out = PredicatedExperts(
            _fused_experts().to("cuda"), execution="predicated"
        )(hs.to("cuda"), index.to("cuda"), weights.to("cuda"))
        assert _max_err(cuda_out.cpu(), cpu_out) < 1e-5


class TestExportPolicy:
    """Export uses the predicated realizer by default.

    aimet-onnx cannot quantize inside If bodies, but that gap is handled by
    ``assert_experts_quantized`` failing, not by exporting a worse graph.
    """

    def test_export_phase_defaults_to_predicated(self):
        adapted = _adapted(_fused_experts())
        with forced_expert_activation(nn.Sequential(adapted), phase="export"):
            assert adapted.execution == "predicated"

    def test_dense_remains_available_as_an_escape_hatch(self):
        adapted = _adapted(_fused_experts(), export_execution="dense")
        with forced_expert_activation(nn.Sequential(adapted), phase="export"):
            assert adapted.execution == "dense"

    def test_calibration_phase_uses_calibration_execution(self):
        adapted = _adapted(_fused_experts(), calibration_execution="predicated")
        with forced_expert_activation(nn.Sequential(adapted), phase="calibration"):
            assert adapted.execution == "predicated"

    def test_export_is_silent(self):
        """No warning: the graph is correct, so nothing to warn about."""
        import warnings as _warnings

        adapted = _adapted(_fused_experts())
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            with forced_expert_activation(nn.Sequential(adapted), phase="export"):
                pass

    def test_invalid_phase_rejected(self):
        adapted = _adapted(_fused_experts())
        with pytest.raises(ValueError, match="phase must be"):
            with forced_expert_activation(nn.Sequential(adapted), phase="nonsense"):
                pass

    def test_invalid_export_execution_rejected(self):
        with pytest.raises(ValueError, match="export_execution must be"):
            _adapted(_fused_experts(), export_execution="sparse")

    def test_mixin_exposes_export_knob(self):
        assert ExportableMoEAdaptation.export_execution == "predicated"


class TestUnquantizedSubgraphGuard:
    """aimet-onnx inserts no quantizers inside If bodies, so expert GEMMs run in
    float while the sim reports a quantized model. Convert that into an error.
    """

    @staticmethod
    def _graph_with_if(body_nodes, quantized_body=False):
        onnx = pytest.importorskip("onnx")

        nodes = list(body_nodes)
        if quantized_body:
            nodes.append(
                onnx.helper.make_node("QcQuantizeOp", ["t"], ["tq"], domain="aimet")
            )
        out_name = "tq" if quantized_body else "t"
        then_g = onnx.helper.make_graph(
            nodes,
            "then",
            [],
            [onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, [1])],
        )
        else_g = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["e"])],
            "else",
            [],
            [onnx.helper.make_tensor_value_info("e", onnx.TensorProto.FLOAT, [1])],
        )
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "If", ["c"], ["y"], then_branch=then_g, else_branch=else_g
                )
            ],
            "g",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info("c", onnx.TensorProto.BOOL, []),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        return graph

    def test_counts_unquantized_ops_in_if_bodies(self):
        onnx = pytest.importorskip("onnx")
        body = [onnx.helper.make_node("MatMul", ["x", "x"], ["t"])]
        assert unquantized_subgraph_ops(self._graph_with_if(body)) == 1

    def test_body_with_a_quantizer_is_not_counted(self):
        onnx = pytest.importorskip("onnx")
        body = [onnx.helper.make_node("MatMul", ["x", "x"], ["t"])]
        graph = self._graph_with_if(body, quantized_body=True)
        assert unquantized_subgraph_ops(graph) == 0

    def test_graph_without_control_flow_is_clean(self):
        onnx = pytest.importorskip("onnx")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["x", "x"], ["y"])],
            "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1])],
        )
        assert unquantized_subgraph_ops(graph) == 0

    def test_assert_raises_with_actionable_message(self):
        onnx = pytest.importorskip("onnx")
        body = [onnx.helper.make_node("Gemm", ["x", "x"], ["t"])]

        class FakeSim:
            class FakeModel:
                pass

        sim = FakeSim()
        sim.model = FakeSim.FakeModel()
        sim.model.model = onnx.helper.make_model(self._graph_with_if(body))

        with pytest.raises(RuntimeError) as excinfo:
            assert_experts_quantized(sim)
        message = str(excinfo.value)
        assert "FLOAT" in message
        assert "If/Loop bodies" in message
        # It must name the workaround without recommending it as the design.
        assert "export_execution: dense" in message

    def test_assert_is_noop_for_clean_sim(self):
        onnx = pytest.importorskip("onnx")
        body = [onnx.helper.make_node("Gemm", ["x", "x"], ["t"])]

        class FakeSim:
            class FakeModel:
                pass

        sim = FakeSim()
        sim.model = FakeSim.FakeModel()
        sim.model.model = onnx.helper.make_model(
            self._graph_with_if(body, quantized_body=True)
        )
        assert_experts_quantized(sim)  # must not raise

    def test_assert_tolerates_non_onnx_objects(self):
        assert_experts_quantized(object())
        assert_experts_quantized(nn.Linear(2, 2))


class TestTorchQuantsimIntegration:
    """The adapted model under a real aimet-torch QuantizationSimModel.

    Needs the router's quantized definition to exist at all, checks every expert
    projection gets its own quantizer, and pins that ``predicated`` fails
    actionably under quantsim rather than with a dynamo traceback.
    """

    CONTEXT_LENGTH = 64
    SEQ_LEN = 8

    @pytest.fixture(scope="class")
    def backend(self):
        pytest.importorskip("aimet_torch")
        from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

        return LLM_Torch

    def _sim(self, backend, *, selection="routed", execution="dense"):
        torch.manual_seed(0)
        model = modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config()).eval()
        replace_fused_experts(model, execution=execution, selection=selection)
        sims = backend.instantiate_quantsim(model, self.CONTEXT_LENGTH, self.SEQ_LEN)
        return sims.backbone if hasattr(sims, "backbone") else sims["backbone"]

    def _run(self, backend, sim):
        inputs = backend.get_sample_backbone_inputs(
            sim.model, self.CONTEXT_LENGTH, self.SEQ_LEN
        )
        return sim.model(*inputs)

    @staticmethod
    def _expert_weight_quantizers(sim):
        found = {}
        for name, module in sim.model.named_modules():
            if ".experts." not in name:
                continue
            params = getattr(module, "param_quantizers", None)
            if (
                params is not None
                and "weight" in params
                and params["weight"] is not None
            ):
                found[name] = params["weight"]
        return found

    def test_sim_builds_and_every_expert_projection_is_quantized(self, backend):
        sim = self._sim(backend)
        qtzrs = self._expert_weight_quantizers(sim)
        # gate + up + down, per expert, per MoE layer.
        assert len(qtzrs) == 3 * E * 2

    def test_calibration_initializes_every_expert_quantizer(self, backend):
        from aimet_torch.v2.nn import compute_encodings

        sim = self._sim(backend)
        qtzrs = self._expert_weight_quantizers(sim)
        assert not any(q.is_initialized() for q in qtzrs.values())

        with forced_expert_activation(sim), torch.no_grad():
            with compute_encodings(sim.model):
                self._run(backend, sim)

        assert all(q.is_initialized() for q in qtzrs.values())

    def test_quantized_sparse_eval_matches_dense(self, backend):
        """Encodings observed densely, applied under fast sparse routing."""
        from aimet_torch.v2.nn import compute_encodings

        sim = self._sim(backend)
        with forced_expert_activation(sim), torch.no_grad():
            with compute_encodings(sim.model):
                self._run(backend, sim)

        with torch.no_grad():
            for module in sim.model.modules():
                if isinstance(module, PredicatedExperts):
                    module.execution = "sparse"
            sparse = self._run(backend, sim)
            for module in sim.model.modules():
                if isinstance(module, PredicatedExperts):
                    module.execution = "dense"
            dense = self._run(backend, sim)

        sparse_t = sparse[0] if isinstance(sparse, (tuple, list)) else sparse
        dense_t = dense[0] if isinstance(dense, (tuple, list)) else dense
        assert _max_err(sparse_t, dense_t) < TOL

    def test_predicated_under_quantsim_raises_actionably(self, backend):
        from aimet_torch.v2.nn import compute_encodings

        sim = self._sim(backend)
        # Calibrate first, so the failure under test is cond/quantsim.
        with forced_expert_activation(sim), torch.no_grad():
            with compute_encodings(sim.model):
                self._run(backend, sim)

        for module in sim.model.modules():
            if isinstance(module, PredicatedExperts):
                module.execution = "predicated"
        with pytest.raises(RuntimeError, match="cannot run inside an aimet-torch"):
            with torch.no_grad():
                self._run(backend, sim)

    def test_unadapted_fused_experts_are_silently_unquantized(self, backend):
        """An unadapted model does not fail to build a sim: the fused Experts
        module is not a quantizable leaf, so its weights never get quantizers and
        the whole MoE silently runs in float.
        """
        torch.manual_seed(0)
        model = modeling_qwen3_moe.Qwen3MoeForCausalLM(_toy_config()).eval()
        sims = backend.instantiate_quantsim(model, self.CONTEXT_LENGTH, self.SEQ_LEN)
        sim = sims.backbone if hasattr(sims, "backbone") else sims["backbone"]

        fused_params = 0
        quantized = 0
        for _, module in sim.model.named_modules():
            if not is_fused_experts(module):
                continue
            fused_params += 2  # gate_up_proj, down_proj
            params = getattr(module, "param_quantizers", None)
            if params is not None:
                quantized += sum(
                    1
                    for key in ("gate_up_proj", "down_proj")
                    if key in params and params[key] is not None
                )

        assert fused_params > 0, "expected fused experts in an unadapted model"
        assert quantized == 0, (
            "fused expert weights unexpectedly got quantizers; the silent-float "
            "hazard this test documents may have been fixed upstream"
        )
