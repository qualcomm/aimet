# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the analysis-pass driver (``run_analysis``) and its markdown report.

GPU-free: a fake ``AnalysisPass`` and fake metric classes stand in for the
real ONNX pass and real metrics. Uses a real ``EvaluationContext`` and
``DiskBackedFPCache``, since several of these tests pin their caching behavior.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from GenAILab.bench.analysis import (
    AnalysisOutput,
    AnalysisPass,
    build_analysis_report,
    metric_detail_sections,
    run_analysis,
)
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.metrics import DistanceMetric, TextEvaluationMetric, run_metrics
from GenAILab.bench.profiler import MetricResult
from GenAILab.bench.yaml_config_parser import (
    ModelConfig,
    ProfilerConfig,
    ResolvedAnalysis,
    ResolvedMetric,
)

_MODEL = ModelConfig(
    model_cls=object,
    model_id="org/model",
    model_type="llama",
    context_length=64,
    sequence_length=32,
    adaptations=[],
)


class FakeSweepPass(AnalysisPass):
    """Calls ``evaluate`` once per slug, like a truncation sweep."""

    def __init__(self, slugs=("trunc_bits8",)):
        self.slugs = slugs

    def run(self, sim_collection, *, evaluate) -> AnalysisOutput:
        return AnalysisOutput(
            table=[
                {
                    "Condition": slug,
                    **{r.metric_name: r.result for r in evaluate(slug)},
                }
                for slug in self.slugs
            ],
            artifacts=["plot.html"],
        )


class FakeFilePass(AnalysisPass):
    """Declares only ``output_dir``, like a pass that writes its own files."""

    uses_metrics = False

    def run(self, sim_collection, *, output_dir) -> AnalysisOutput:
        self.received = (sim_collection, output_dir)
        (Path(output_dir) / "scores.json").write_text("{}")
        return AnalysisOutput(artifacts=["scores.json"])


class RecordingMetric(TextEvaluationMetric):
    """Records the ``output_dir`` of every call and computes one quant result."""

    calls = []

    @classmethod
    def evaluate(
        cls, model, tokenizer, context_length, *, eval_ctx, output_dir=None, **kwargs
    ):
        cls.calls.append(output_dir)
        return float(
            eval_ctx.get_or_compute_quant("collection", lambda: len(cls.calls))
        )


class FPMetric(DistanceMetric, TextEvaluationMetric):
    """A distance metric: needs an FP reference as well as a quant result."""

    fp_computes = 0

    @classmethod
    def evaluate(cls, model, tokenizer, context_length, *, eval_ctx, **kwargs):
        def _fp():
            cls.fp_computes += 1
            return 0.0

        eval_ctx.get_or_compute_fp("fp_collection", _fp)
        return 1.0


@pytest.fixture(autouse=True)
def _reset_metrics():
    RecordingMetric.calls = []
    FPMetric.fp_computes = 0


@pytest.fixture
def fp_cache(tmp_path):
    return DiskBackedFPCache(tmp_path / "fp_cache")


def _metric(metric_cls):
    return ResolvedMetric(
        name=metric_cls.__name__, metric_cls=metric_cls, metric_kwargs={}
    )


def _run(tmp_path, fp_cache, analysis_pass, subset, top_level=()):
    config = SimpleNamespace(
        model=_MODEL,
        metrics=top_level,
        profiler=ProfilerConfig(),
        analysis=ResolvedAnalysis(
            name="Fake", analysis_pass=analysis_pass, kwargs={"k": 1}, metrics=subset
        ),
    )
    return run_analysis(
        config,
        sim_collection="sims",
        generator="gen",
        tokenizer=object(),
        fp_cache=fp_cache,
        results_dir=tmp_path,
        precision={},
        recipe_chain="Skip",
    )


def _baseline(fp_cache, metrics):
    """What the runner does before ``run_analysis``: the plain-sim metrics run."""
    run_metrics(
        metrics,
        "gen",
        object(),
        64,
        EvaluationContext(fp_cache=fp_cache, model_config=_MODEL),
    )


def test_each_condition_gets_fresh_quant_cache_and_own_output_dir(tmp_path, fp_cache):
    recording = _metric(RecordingMetric)

    result = _run(
        tmp_path,
        fp_cache,
        FakeSweepPass(slugs=("trunc_bits12", "trunc_bits8")),
        (recording,),
    )

    # The report sits inside its own run directory, next to the artifacts.
    [report] = (tmp_path / "analysis").glob("*/*.md")
    artifacts = report.parent
    assert artifacts.name == report.stem
    assert RecordingMetric.calls == [
        f"{artifacts}/trunc_bits12",
        f"{artifacts}/trunc_bits8",
    ]
    # The quant cache is keyed on collection name, so a shared one would hand
    # the second condition the first one's 1.0.
    text = report.read_text()
    assert "| trunc_bits12 | 1.0 |" in text
    assert "| trunc_bits8 | 2.0 |" in text
    assert "[plot.html](plot.html)" in text
    assert result.name == "Fake"
    assert result.kwargs == {"k": 1, "metrics": ["RecordingMetric"]}
    assert result.report == f"analysis/{artifacts.name}/{report.name}"


def test_pass_gets_only_the_inputs_its_run_declares(tmp_path, fp_cache):
    # A pass that doesn't declare evaluate/generator/tokenizer/context_length
    # would raise TypeError if the driver passed them anyway.
    file_pass = FakeFilePass()

    result = _run(tmp_path, fp_cache, file_pass, ())

    sim_collection, output_dir = file_pass.received
    assert sim_collection == "sims"
    [report] = (tmp_path / "analysis").glob("*/*.md")
    assert Path(output_dir) == report.parent
    assert (report.parent / "scores.json").exists()
    assert result.report == f"analysis/{report.parent.name}/{report.name}"


def test_runs_in_the_same_second_get_distinct_directories(tmp_path, fp_cache):
    for _ in range(2):
        _run(
            tmp_path, fp_cache, FakeSweepPass(slugs=("a",)), (_metric(RecordingMetric),)
        )

    assert len(list((tmp_path / "analysis").glob("*/*.md"))) == 2


def test_pass_reuses_fp_reference_collected_by_baseline(tmp_path, fp_cache):
    fp = _metric(FPMetric)
    _baseline(fp_cache, (fp,))

    _run(tmp_path, fp_cache, FakeSweepPass(slugs=("a", "b")), (fp,), top_level=(fp,))

    assert FPMetric.fp_computes == 1


def test_pass_never_computes_fp_without_baseline(tmp_path, fp_cache):
    """With no baseline to warm the cache, an instrumented condition
    must raise rather than record an instrumented graph as the FP reference."""
    fp = _metric(FPMetric)

    with pytest.raises(RuntimeError, match="not in the FP cache"):
        _run(tmp_path, fp_cache, FakeSweepPass(), (fp,))

    assert FPMetric.fp_computes == 0


_GRACE_DETAILS = {
    "grader_model": "Qwen/Qwen3.6-35B-A3B",
    "num_items": 1,
    "total_points": 4,
    "max_points": 10,
    "num_unparsed": 0,
    "category_scores": {"reasoning": {"score_pct": 40.0, "num_scored": 1}},
    "summary_items": ["Struggles with multi-step math."],
    "items": [
        {
            "idx": 0,
            "category": "reasoning",
            "prompt": "What is 2+2?",
            "output": "5",
            "points": 4,
            "rationale": "Wrong answer.",
        }
    ],
}


def _report(output):
    return build_analysis_report(
        model_id="org/model",
        model_type="llama",
        precision={},
        recipe_chain="Skip",
        timestamp="t",
        has_baseline=True,
        pass_name="TruncationSimulation",
        pass_kwargs={"truncation_bits": [8, 12]},
        metric_names=["Grace"],
        output=output,
    )


def test_table_columns_are_union_of_row_keys_in_order():
    report = _report(
        AnalysisOutput(
            table=[
                {
                    "Condition": "truncation_bits=8",
                    "Grace": 68.1,
                    "PPL": 9.12,
                },
                {
                    "Condition": "truncation_bits=12",
                    "Grace": 51.66,
                    "TinyMMLU": 41.0,
                },
            ]
        )
    )
    table = [line for line in report.splitlines() if line.startswith("|")]
    assert table == [
        "| Condition | Grace | PPL | TinyMMLU |",
        "|---|---|---|---|",
        "| truncation_bits=8 | 68.1 | 9.1 |  |",
        "| truncation_bits=12 | 51.7 |  | 41.0 |",
    ]


def test_sections_only_output_has_no_table():
    report = _report(AnalysisOutput(sections=[("Ranking", "layer 12 first")]))
    assert not any(line.startswith("|") for line in report.splitlines())
    assert "### Ranking\n\nlayer 12 first" in report


def test_grace_details_rendered_as_section():
    results = [
        MetricResult(
            metric_name="Grace", result=40.0, profiler=None, details=_GRACE_DETAILS
        ),
        MetricResult(metric_name="PPL", result=9.1, profiler=None),
    ]
    sections = metric_detail_sections("truncation_bits=8", results)

    # PPL has no details, so only Grace gets a section.
    [(heading, body)] = sections
    assert heading == "truncation_bits=8 — Grace responses and summary"
    assert "What is 2+2?" in body
    assert "Wrong answer." in body
    assert "Struggles with multi-step math." in body
