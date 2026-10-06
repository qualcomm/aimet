# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Execution driver for an analysis pass.

The runner evaluates the top-level ``metrics`` on the plain sim first, as it
does without an analysis section. That is the baseline: its scores become
``accuracy_results`` and it warms the FP cache. :func:`run_analysis` then
runs the pass and writes its report.

The pass owns what it evaluates. ``AnalysisPass.run`` gets the sim
collection and decides what to evaluate and how: a truncation sweep calls
``evaluate`` once per bit width, a quantizer sensitivity pass runs its own
sweep and writes a plot under ``output_dir``. Like a metric's ``evaluate``,
``run`` declares the inputs it needs as keyword arguments and gets only
those. It returns an :class:`AnalysisOutput` that the report renders. The driver owns the cache
isolation for anything the pass evaluates, the report and the ``analysis`` DB
column.

The report is plain markdown built from the ``AnalysisOutput`` by
:func:`build_analysis_report`, which knows nothing about how the pass ran.

Adding a pass means writing one ``AnalysisPass`` subclass and registering it
with ``YAMLConfigParser.register_analysis``; nothing here changes. The pass's
``__init__`` signature is its config contract: the YAML kwargs (minus
``name`` and ``metrics``) are passed straight to it, so an unknown or missing
kwarg fails as an ordinary ``TypeError`` at parse time.
"""

from __future__ import annotations

import inspect
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.metrics import format_grader_summary, run_metrics
from GenAILab.bench.profiler import AnalysisResult, MetricResult
from GenAILab.bench.summary import _format_precision
from GenAILab.bench.yaml_config_parser import ParsedConfig


@dataclass(frozen=True)
class AnalysisOutput:
    """What a pass returns for the report.

    ``table`` is one dict per row, rendered as a markdown table whose columns
    are the keys in first-seen order. ``sections`` are ``(heading, markdown)``
    blocks. ``artifacts`` are paths relative to the ``output_dir`` the pass
    received, linked from the report.
    """

    table: list[dict[str, Any]] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


class AnalysisPass(ABC):
    """Base class for a registered analysis pass.

    Subclasses take their YAML kwargs as ``__init__`` keyword arguments and
    validate/normalize them there. Anything ``run`` changes on the sim must be
    undone before it returns, including when it raises.

    A pass that never calls ``evaluate`` sets ``uses_metrics = False``.
    The parser then rejects a ``metrics`` list on it, and no longer requires
    one anywhere in the config.
    """

    uses_metrics: bool = True

    @abstractmethod
    def run(self, sim_collection, **inputs) -> AnalysisOutput:
        """Evaluate the pass's conditions and return what the report shows.

        ``sim_collection`` is the runner's sims. A subclass declares, as
        keyword arguments, whichever of these inputs it needs; the driver
        passes only those:

        * ``evaluate(slug)`` runs the pass's metric subset on the sim as it
          is right now (so the pass instruments first), writes metric
          artifacts to ``<output_dir>/<slug>/``, and returns one
          ``MetricResult`` per metric. Each call gets a fresh quant cache and
          a read-only FP cache, so a pass never has to think about either.
        * ``output_dir`` is the run directory, where a pass writes its own
          artifacts (plots, JSON). The report goes there too.
        * ``generator``, ``tokenizer`` and ``context_length`` are the
          runner's own, for a pass that builds model inputs itself instead
          of calling ``evaluate``.
        """


class ReadOnlyFPCache:
    """Read-only view over a :class:`DiskBackedFPCache`.

    Used for everything a pass evaluates, so an FP reference can never be
    collected on an instrumented graph. When a baseline runs, this never
    fires: the runner evaluates the full metric set before the pass, and a
    subset's FP needs are a subset of the baseline's. Without a baseline (no
    top-level ``metrics``), a distance metric only works if an earlier run
    already left its FP reference in the disk cache; otherwise this raises,
    since computing it here would record an instrumented graph as FP.
    """

    def __init__(self, fp_cache: DiskBackedFPCache):
        self._fp_cache = fp_cache

    def get_or_compute(self, key: tuple, compute_fn, metadata: dict | None = None):
        result = self._fp_cache.get(key)
        if result is None:
            raise RuntimeError(
                f"FP reference for {key} is not in the FP cache. An instrumented "
                "condition must never compute a fresh FP reference. If the config "
                "has no top-level 'metrics' (no baseline run), add this metric "
                "there so the baseline collects its FP reference first."
            )
        return result


def metric_detail_sections(
    label: str, results: list[MetricResult]
) -> list[tuple[str, str]]:
    """One ``(heading, markdown)`` section per metric in ``results`` that has details.

    Only Grace reports details (its per-response grades and summary).
    """
    return [
        (
            f"{label} — {r.metric_name} responses and summary",
            "```\n"
            + format_grader_summary(r.details, r.details.get("items", []))
            + "\n```",
        )
        for r in results
        if r.details
    ]


def _format_cell(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    header = list(dict.fromkeys(key for row in rows for key in row))
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_cell(row.get(k, "")) for k in header) + " |"
        )
    return lines


def build_analysis_report(
    *,
    model_id: str,
    model_type: str,
    precision: dict,
    recipe_chain: str,
    timestamp: str,
    has_baseline: bool,
    pass_name: str,
    pass_kwargs: dict,
    metric_names: list[str],
    output: AnalysisOutput,
    artifact_links: list[str] = (),
) -> str:
    """Render the markdown report for one analysis pass.

    The baseline is not in the report: it feeds ``accuracy_results``, and this
    report describes only what the pass measured.
    """
    kwargs_str = ", ".join(f"{k}={v}" for k, v in pass_kwargs.items())
    lines = [
        "# Analysis report",
        "",
        "## Run identity",
        "",
        f"- Model: {model_id} ({model_type})",
        f"- Recipe chain: {recipe_chain}",
        f"- Precision: {_format_precision(precision)}",
        f"- Timestamp: {timestamp}",
        "- Baseline (plain sim, all top-level metrics): "
        + ("accuracy_results in profiling_data.json" if has_baseline else "not run"),
        "",
        f"## {pass_name}",
        "",
    ]
    if metric_names:
        lines += [f"Metrics analyzed: {', '.join(metric_names)}.", ""]
    lines += [f"Kwargs: {kwargs_str or '(none)'}"]

    if output.table:
        lines += ["", *_markdown_table(output.table)]

    if artifact_links:
        lines += ["", "Artifacts:", ""]
        lines += [f"- [{link.split('/')[-1]}]({link})" for link in artifact_links]

    for heading, markdown in output.sections:
        lines += ["", f"### {heading}", "", markdown]

    return "\n".join(lines) + "\n"


def run_analysis(
    config: ParsedConfig,
    *,
    sim_collection,
    generator,
    tokenizer,
    fp_cache: DiskBackedFPCache,
    results_dir: Path,
    precision: dict,
    recipe_chain: str,
) -> AnalysisResult:
    """Run ``config.analysis`` and write its report under ``<results_dir>/analysis/``.

    Must run after the baseline (the top-level ``metrics`` on the plain sim),
    since the pass may only read the FP cache. Everything a run writes lives in
    one directory, ``analysis/<slug>/``: the report ``<slug>.md`` next to the
    pass's artifacts. Returns the ``analysis`` DB column.
    """
    analysis = config.analysis
    model = config.model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = model.model_id.rstrip("/").split("/")[-1]
    # The timestamp alone can collide: two recipes in one config can reach this
    # in the same second, and CI merges results from several runners into one
    # results dir. The random suffix keeps every run's directory distinct.
    run_id = uuid.uuid4().hex[:8]
    report_slug = (
        f"{model.model_type}_{model_slug}_{analysis.name}_{timestamp}_{run_id}"
    )
    output_dir = results_dir / "analysis" / report_slug
    # Created up front, and never reused, so a pass cannot write into another
    # run's directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    read_only_fp_cache = ReadOnlyFPCache(fp_cache)

    def evaluate(slug: str) -> list[MetricResult]:
        # A fresh EvaluationContext per call: its quant cache is keyed on
        # collection name alone, so sharing one would hand every condition the
        # first condition's outputs.
        eval_ctx = EvaluationContext(fp_cache=read_only_fp_cache, model_config=model)
        return run_metrics(
            analysis.metrics,
            generator,
            tokenizer,
            model.context_length,
            eval_ctx,
            image_size=model.image_size,
            audio_frames=model.audio_frames,
            gpu_meter_kwargs=config.profiler.gpu_meter_kwargs,
            capture_intermediate_data=config.profiler.capture_intermediate_data,
            extra_kwargs={"output_dir": str(output_dir / slug)},
        )

    # Passed only if the pass's run() declares them, as run_metrics does for
    # a metric's evaluate(); a pass never receives inputs it didn't ask for.
    inputs = {
        "evaluate": evaluate,
        "output_dir": str(output_dir),
        "generator": generator,
        "tokenizer": tokenizer,
        "context_length": model.context_length,
    }
    declared = inspect.signature(analysis.analysis_pass.run).parameters
    output = analysis.analysis_pass.run(
        sim_collection, **{k: v for k, v in inputs.items() if k in declared}
    )

    report_path = output_dir / f"{report_slug}.md"
    report_path.write_text(
        build_analysis_report(
            model_id=model.model_id,
            model_type=model.model_type,
            precision=precision,
            recipe_chain=recipe_chain,
            timestamp=timestamp,
            has_baseline=bool(config.metrics),
            pass_name=analysis.name,
            pass_kwargs=analysis.kwargs,
            metric_names=[m.name for m in analysis.metrics],
            output=output,
            # Artifact paths are relative to output_dir, which holds the report too.
            artifact_links=list(output.artifacts),
        )
    )

    return AnalysisResult(
        name=analysis.name,
        kwargs={**analysis.kwargs, "metrics": [m.name for m in analysis.metrics]},
        report=str(report_path.relative_to(results_dir)),
    )
