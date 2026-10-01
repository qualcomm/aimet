# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single up-front decoder-stack analysis and its handoff.

The ONNX runner analyzes the float model's topology exactly ONCE, before the
pre-sim chain and before the sim is built, and threads that one object through
both chains as a plain argument. Nothing owns or caches it. These tests pin the
two halves of that contract:

* both chains hand the topology to every step they run, and
* the recipes consume what they are given rather than deriving their own.

All heavy dependencies are mocked; nothing here loads a real model.
"""

from unittest.mock import MagicMock, patch

# Imported before ``bench.onnx.quant_recipes`` on purpose: the backend package and
# quant_recipes import each other, so whichever is imported first must be the
# backend (which is also the order the runner establishes).
import GenAILab.qai_hub_lm.backends.onnx  # noqa: F401 — breaks an import cycle


def _fake_step(**recipe_kwargs):
    """A ResolvedStep stand-in carrying a mock technique class."""
    technique_cls = MagicMock()
    technique_cls.cacheable.return_value = False
    technique_cls.__name__ = "FakeTechnique"

    step = MagicMock()
    step.name = "FakeTechnique"
    step.technique_cls = technique_cls
    step.recipe_kwargs = recipe_kwargs
    step.dataset_cls = None
    step.dataset_kwargs = {}
    return step


class TestRecipesConsumeSuppliedTopology:
    """Neither recipe may re-derive a topology it was handed."""

    def test_spinquant_uses_supplied_topology(self):
        """SpinQuant rotates using the runner's topology, not one of its own."""
        from GenAILab.bench.onnx.quant_recipes import SpinQuant

        topology = MagicMock(name="topology")
        float_model = MagicMock()
        float_model.embedding = None

        with patch(
            "GenAILab.bench.onnx.quant_recipes.apply_spinquant"
        ) as mock_spinquant:
            SpinQuant.apply(float_model, enable_r1=True, topology=topology)

        assert mock_spinquant.call_args.kwargs["topology"] is topology

    def test_adascale_uses_supplied_topology(self):
        """AdaScale optimizes using the runner's topology, not the sim's graph."""
        from GenAILab.bench.onnx.quant_recipes import AdaScale

        topology = MagicMock(name="topology")
        generator = MagicMock()
        generator.config.model_type = "llama"

        with (
            patch(
                "GenAILab.bench.onnx.quant_recipes._prefill_inputs",
                return_value=[{"input_ids": None}],
            ),
            patch("GenAILab.bench.onnx.quant_recipes.apply_adascale") as mock_adascale,
            patch.dict(
                "GenAILab.bench.onnx.quant_recipes.adascale_model_config_dict",
                {"llama": MagicMock()},
            ),
        ):
            AdaScale.apply(MagicMock(), generator, MagicMock(), topology=topology)

        assert mock_adascale.call_args.kwargs["topology"] is topology

    def test_neither_recipe_analyzes_topology_itself(self):
        """Topology analysis belongs to the runner, not to the recipes.

        A recipe that analyzed on its own would reintroduce the per-call analysis
        this change removed — and, after a rotation, would analyze a different
        graph than the one the run started from. Asserted by patching the analyzer
        at its source and requiring it is never reached, so re-importing it into a
        recipe under any alias still fails this test.
        """
        import GenAILab.bench.onnx.quant_recipes as quant_recipes
        from GenAILab.bench.onnx.quant_recipes import AdaScale, SpinQuant

        assert not hasattr(quant_recipes, "analyze_llm_topology_by_norm_count")

        topology = MagicMock(name="topology")
        float_model = MagicMock()
        float_model.embedding = None
        generator = MagicMock()
        generator.config.model_type = "llama"

        with patch(
            "aimet_onnx.experimental.llm_topology.topology.analyze_llm_topology_by_norm_count"
        ) as mock_analyze:
            with patch("GenAILab.bench.onnx.quant_recipes.apply_spinquant"):
                SpinQuant.apply(float_model, enable_r1=True, topology=topology)
            with (
                patch(
                    "GenAILab.bench.onnx.quant_recipes._prefill_inputs",
                    return_value=[{"input_ids": None}],
                ),
                patch("GenAILab.bench.onnx.quant_recipes.apply_adascale"),
                patch.dict(
                    "GenAILab.bench.onnx.quant_recipes.adascale_model_config_dict",
                    {"llama": MagicMock()},
                ),
            ):
                AdaScale.apply(MagicMock(), generator, MagicMock(), topology=topology)

        mock_analyze.assert_not_called()


class TestChainsThreadTopology:
    """Both chains pass the topology to every step they run."""

    def test_pre_sim_chain_forwards_topology(self):
        """The pre-sim chain hands the topology to each step (e.g. SpinQuant)."""
        from GenAILab.bench.recipe_chain import apply_pre_quantization_chain

        topology = MagicMock(name="topology")
        step = _fake_step(enable_r1=True)

        apply_pre_quantization_chain((step,), MagicMock(), topology=topology)

        assert step.technique_cls.apply.call_args.kwargs["topology"] is topology

    def test_on_sim_chain_forwards_topology(self):
        """The on-sim chain hands the topology to each step (e.g. AdaScale)."""
        from GenAILab.bench.recipe_chain import apply_quantization_chain

        topology = MagicMock(name="topology")
        step = _fake_step()

        with patch("GenAILab.bench.recipe_chain.GPUMeter"):
            apply_quantization_chain(
                [step],
                MagicMock(),
                MagicMock(),
                MagicMock(),
                context_length=32,
                image_size=None,
                profiler_kwargs={},
                profiler_capture_intermediate_data=False,
                framework="onnx",
                model_id="fake",
                precision=MagicMock(),
                model_kwargs={},
                topology=topology,
            )

        assert step.technique_cls.apply.call_args.kwargs["topology"] is topology

    def test_torch_spinquant_absorbs_topology(self):
        """Torch's pre-sim SpinQuant must tolerate the ONNX-only fixture.

        Both frameworks share ``apply_pre_quantization_chain``, which passes
        ``topology`` unconditionally. Torch analyzes no topology, so its recipe has
        to absorb the kwarg rather than fail on it.

        Read from the source rather than imported: ``aimet_torch`` is absent from
        the ONNX test environment, so importing the torch recipes here would skip
        the check exactly where it is cheapest to run.
        """
        import ast
        import pathlib

        import GenAILab

        source = (
            pathlib.Path(GenAILab.__file__).parent / "bench/torch/quant_recipes.py"
        ).read_text()
        applies = [
            fn
            for cls in ast.walk(ast.parse(source))
            if isinstance(cls, ast.ClassDef) and cls.name == "SpinQuant"
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name == "apply"
        ]
        assert applies, "torch SpinQuant.apply not found"
        assert applies[0].args.kwarg is not None, (
            "torch SpinQuant.apply must accept **kwargs to absorb 'topology'"
        )

    def test_topology_is_a_fixture_not_a_recipe_knob(self):
        """``topology`` is runner-supplied, so no YAML spec may declare it.

        The recipe registry enforces that an ``apply()`` implements exactly its
        spec's kwargs; ``topology`` is exempt only because it is a fixture, like
        ``quantsim`` and ``component``.
        """
        from GenAILab.bench.yaml_config_parser import _RECIPE_APPLY_FIXTURES

        assert "topology" in _RECIPE_APPLY_FIXTURES
