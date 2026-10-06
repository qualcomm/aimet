# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ONNX analysis passes: ``TruncationSimulation``, including
that truncation survives a session rebuild during evaluation, and
``QuantizerSensitivity``.

GPU-free: fake sims stand in for ``QuantizationSimModel``. The aimet_onnx
functions the passes call (``create_truncation_aware_session``, the
``aimet_onnx.analysis`` sweep and save functions) are faked through
``sys.modules`` entries, since the real ones need the C++-backed build.
"""

import contextlib
import json
import sys
import types
from types import SimpleNamespace

import pytest

from GenAILab.bench.onnx.analysis_passes import (
    QuantizerSensitivity,
    TruncationSimulation,
)
from GenAILab.bench.profiler import MetricResult


class FakeSim:
    def __init__(self, name):
        self.name = name
        self.session = "clean"

    def _rebuild_session(self):
        self.session = "clean"

    def __repr__(self):
        return f"FakeSim({self.name!r})"


class FakeSimCollection:
    def __init__(self, **components):
        self.backbone = FakeSim("backbone")
        self._components = {name: FakeSim(name) for name in components}

    def present_components(self):
        return tuple(self._components)

    def component(self, name):
        return self._components[name]

    def all_sims(self):
        return [self.backbone, *self._components.values()]


@pytest.fixture
def build_calls(monkeypatch):
    """Fake ``create_truncation_aware_session``; returns the list of its calls."""
    calls = []

    def _build(sim, truncation_bits=8):
        # Instrumenting must never run with the previous session still alive.
        assert sim.session is None, "old session not released before rebuild"
        calls.append((sim.name, truncation_bits))
        return f"trunc{truncation_bits}-{sim.name}-{len(calls)}"

    module = types.ModuleType("aimet_onnx.experimental._truncation_aware")
    module.create_truncation_aware_session = _build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def _score(slug):
    return [MetricResult(metric_name="Grace", result=50.0, profiler=None)]


def test_every_present_component_truncated_during_evaluate(build_calls):
    sims = FakeSimCollection(visual=True, audio=True)
    seen = []

    def evaluate(slug):
        seen.append((slug, [s.session for s in sims.all_sims()]))
        return _score(slug)

    output = TruncationSimulation(truncation_bits=[12, 8]).run(sims, evaluate=evaluate)

    assert [slug for slug, _ in seen] == ["trunc_bits12", "trunc_bits8"]
    assert all(s.startswith("trunc12-") for s in seen[0][1])
    assert all(s.startswith("trunc8-") for s in seen[1][1])
    assert [s.session for s in sims.all_sims()] == ["clean"] * 3
    assert output.table == [
        {"Condition": "truncation_bits=12", "Grace": 50.0},
        {"Condition": "truncation_bits=8", "Grace": 50.0},
    ]


def test_rebuild_during_evaluate_reinstruments(build_calls):
    """Grace's on_device() rebuilds the session; truncation must survive it."""
    sims = FakeSimCollection()

    def evaluate(slug):
        before = sims.backbone.session
        sims.backbone._rebuild_session()
        assert sims.backbone.session.startswith("trunc8-")
        assert sims.backbone.session != before
        return _score(slug)

    TruncationSimulation(truncation_bits=8).run(sims, evaluate=evaluate)

    # The patch is gone afterwards: a rebuild no longer instruments.
    n_calls = len(build_calls)
    sims.backbone._rebuild_session()
    assert len(build_calls) == n_calls
    assert sims.backbone.session == "clean"


def test_restored_clean_when_evaluate_raises(build_calls):
    sims = FakeSimCollection(visual=True)

    def evaluate(slug):
        raise RuntimeError("metric blew up")

    with pytest.raises(RuntimeError, match="metric blew up"):
        TruncationSimulation(truncation_bits=8).run(sims, evaluate=evaluate)

    for sim in sims.all_sims():
        assert sim.session == "clean"
        assert sim._rebuild_session.__func__ is FakeSim._rebuild_session


def test_restored_clean_when_instrumenting_a_later_sim_fails(build_calls, monkeypatch):
    """A failure instrumenting visual must not leave the backbone truncated."""
    sims = FakeSimCollection(visual=True)
    module = sys.modules["aimet_onnx.experimental._truncation_aware"]
    real_build = module.create_truncation_aware_session

    def _build(sim, truncation_bits=8):
        if sim.name == "visual":
            raise RuntimeError("shape inference failed")
        return real_build(sim, truncation_bits)

    monkeypatch.setattr(module, "create_truncation_aware_session", _build)

    with pytest.raises(RuntimeError, match="shape inference failed"):
        TruncationSimulation(truncation_bits=8).run(sims, evaluate=_score)

    for sim in sims.all_sims():
        assert sim.session == "clean"
        assert sim._rebuild_session.__func__ is FakeSim._rebuild_session


def test_session_swapped_outside_the_patch_is_caught(build_calls):
    sims = FakeSimCollection()

    def evaluate(slug):
        sims.backbone.session = "tampered"
        return _score(slug)

    with pytest.raises(AssertionError, match="rebuilt outside the patched"):
        TruncationSimulation(truncation_bits=8).run(sims, evaluate=evaluate)
    assert sims.backbone.session == "clean"


@pytest.mark.parametrize("bad", [[], "8", [8, "12"], True, 8.0])
def test_bad_truncation_bits_rejected(bad):
    with pytest.raises(ValueError, match="truncation_bits"):
        TruncationSimulation(truncation_bits=bad)


# ---------------------------------------------------------------------------
# QuantizerSensitivity
# ---------------------------------------------------------------------------


class FakeQuantizer:
    def __init__(self, enabled=True):
        self.enabled = enabled


class FakeQuantSim:
    """Backbone sim: two weight quantizers, one activation, two KV inputs.

    ``session`` reports which quantizers are enabled, so a test can see what
    state each session call ran in.
    """

    def __init__(self):
        self.qc_quantize_op_dict = {
            "w_q": FakeQuantizer(),
            "act_q": FakeQuantizer(),
            "w_k": FakeQuantizer(),
            "past_key_0_in": FakeQuantizer(),
            "past_value_0_in": FakeQuantizer(),
        }
        self.param_names = ["w_q", "w_k"]
        # KV inputs are graph inputs, which aimet_onnx also lists here.
        self.activation_names = ["act_q", "past_key_0_in", "past_value_0_in"]

    @property
    def session(self):
        return frozenset(n for n, q in self.qc_quantize_op_dict.items() if q.enabled)


# Error each quantizer adds on its own; PSNR falls as error rises.
_ERROR = {
    "w_q": 3.0,
    "w_k": 1.0,
    "act_q": 50.0,
    "past_key_0_in": 2.0,
    "past_value_0_in": 4.0,
}


@pytest.fixture
def fake_aimet_analysis(monkeypatch, tmp_path):
    """Fake ``aimet_onnx.analysis`` and ``aimet_onnx.utils.disable_quantizers``.

    The fake sweep mirrors the real one: disable everything, enable one group
    at a time, restore on exit, return most-sensitive-first.
    """
    seen = SimpleNamespace(fp_session=None, feeds=None, k=None, saved={})

    @contextlib.contextmanager
    def disable_quantizers(sim, names):
        before = {n: sim.qc_quantize_op_dict[n].enabled for n in names}
        for n in names:
            sim.qc_quantize_op_dict[n].enabled = False
        try:
            yield
        finally:
            for n, enabled in before.items():
                sim.qc_quantize_op_dict[n].enabled = enabled

    def make_topk_logit_psnr_metric(fp_session, inputs, k=10):
        seen.fp_session, seen.feeds, seen.k = fp_session, list(inputs), k
        return SimpleNamespace(
            name=f"Top{k}LogitPSNR",
            # Higher PSNR is better: lower error, higher score.
            eval=lambda session: 100.0 - sum(_ERROR[n] for n in session),
        )

    def analyze_per_quantizer_sensitivity(sim, metric, group_fn):
        groups = {}
        for name, q in sim.qc_quantize_op_dict.items():
            key = group_fn(name) if q.enabled else None
            if key is not None:
                groups.setdefault(key, []).append(name)
        scores = {}
        with disable_quantizers(sim, sim.qc_quantize_op_dict.keys()):
            for key, names in groups.items():
                for n in names:
                    sim.qc_quantize_op_dict[n].enabled = True
                scores[key] = metric.eval(sim.session)
                for n in names:
                    sim.qc_quantize_op_dict[n].enabled = False
        return dict(sorted(scores.items(), key=lambda kv: kv[1]))

    def group_by_op_name(sim):
        return get_quantizer_op_names(sim).get

    def get_quantizer_op_names(sim):
        return {
            "w_q": "/layers.0/q_proj/MatMul",
            "w_k": "/layers.0/k_proj/MatMul",
            "act_q": "/layers.0/q_proj/MatMul",
        }

    def _save(kind):
        def save(scores, *args, save_path, details=None, **kwargs):
            seen.saved[kind] = (save_path, list(scores), details)
            with open(save_path, "w") as f:
                json.dump(scores, f)

        return save

    analysis = types.ModuleType("aimet_onnx.analysis")
    analysis.make_topk_logit_psnr_metric = make_topk_logit_psnr_metric
    analysis.analyze_per_quantizer_sensitivity = analyze_per_quantizer_sensitivity
    analysis.group_by_op_name = group_by_op_name
    analysis.save_sensitivity_plot = _save("plot")
    analysis.save_sensitivity_results = _save("results")
    utils = types.ModuleType("aimet_onnx.utils")
    utils.disable_quantizers = disable_quantizers
    monkeypatch.setitem(sys.modules, "aimet_onnx.analysis", analysis)
    monkeypatch.setitem(sys.modules, "aimet_onnx.utils", utils)
    # Feeds come from Wikitext in real runs; the test only checks they reach
    # the metric.
    monkeypatch.setattr(
        QuantizerSensitivity,
        "_build_feeds",
        lambda self, *args: [{"input_ids": i} for i in range(self.num_samples)],
    )
    return seen


def _run_sensitivity(analysis_pass, sim, tmp_path):
    # No evaluate: the pass doesn't declare it, so the driver never passes it.
    return analysis_pass.run(
        SimpleNamespace(backbone=sim),
        output_dir=str(tmp_path),
        generator="gen",
        tokenizer="tok",
        context_length=64,
    )


def test_sensitivity_weights_sweeps_only_weight_quantizers(
    fake_aimet_analysis, tmp_path
):
    sim = FakeQuantSim()
    # A disabled weight quantizer is never swept.
    sim.qc_quantize_op_dict["w_k"].enabled = False

    output = _run_sensitivity(
        QuantizerSensitivity(mode="weights", num_samples=3, top_k=5), sim, tmp_path
    )

    seen = fake_aimet_analysis
    # FP reference: recorded with every quantizer off, on the pass's feeds.
    assert seen.fp_session == frozenset()
    assert seen.feeds == [{"input_ids": 0}, {"input_ids": 1}, {"input_ids": 2}]
    assert seen.k == 5
    # q_proj's group is its weight alone: act_q (same node) was disabled for
    # the sweep, so the score is 100 - 3, not 100 - 3 - 50.
    assert output.table == [
        {"Rank": 1, "Name": "/layers.0/q_proj/MatMul", "Top5LogitPSNR": "97.00"}
    ]
    assert seen.saved["results"] == (
        str(tmp_path / "weights_sensitivity.json"),
        ["/layers.0/q_proj/MatMul"],
        {"/layers.0/q_proj/MatMul": "w_q"},
    )
    assert seen.saved["plot"][0] == str(tmp_path / "weights_sensitivity.html")
    assert output.artifacts == ["weights_sensitivity.html", "weights_sensitivity.json"]
    # Enabled state restored, including the disabled one.
    assert {n: q.enabled for n, q in sim.qc_quantize_op_dict.items()} == {
        "w_q": True,
        "act_q": True,
        "w_k": False,
        "past_key_0_in": True,
        "past_value_0_in": True,
    }


def test_sensitivity_kv_cache_ranks_most_sensitive_first(fake_aimet_analysis, tmp_path):
    sim = FakeQuantSim()

    output = _run_sensitivity(
        QuantizerSensitivity(mode="kv_cache", report_top_n=1), sim, tmp_path
    )

    # report_top_n limits only the table; the JSON below still has both.
    assert [row["Name"] for row in output.table] == ["past_value_0_in"]
    assert fake_aimet_analysis.saved["results"][2] is None  # no node details
    assert output.artifacts == [
        "kv_cache_sensitivity.html",
        "kv_cache_sensitivity.json",
    ]
    # Plot is in graph order, results in rank order.
    assert fake_aimet_analysis.saved["plot"][1] == ["past_key_0_in", "past_value_0_in"]
    assert fake_aimet_analysis.saved["results"][1] == [
        "past_value_0_in",
        "past_key_0_in",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "activations"},
        {"num_samples": 0},
        {"top_k": "10"},
        {"report_top_n": True},
    ],
)
def test_bad_sensitivity_kwargs_rejected(kwargs):
    with pytest.raises(ValueError):
        QuantizerSensitivity(**kwargs)
