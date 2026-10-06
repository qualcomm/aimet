# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the human-readable profiling summary table."""

from GenAILab.bench.summary import _build_footnote, _print_group


def _entry(**overrides):
    entry = {
        "model_id": "org/model",
        "components": {
            "backbone": {"recipe": "Calibration", "recipe_kwargs": {"num_batches": 32}}
        },
        "PPL": {"result": 12.5, "scoring_version": 1},
    }
    entry.update(overrides)
    return entry


class TestNonMetricKeys:
    """accuracy_details and analysis must never be discovered as metrics."""

    def test_accuracy_details_not_rendered_as_a_metric_column(self, capsys):
        entry = _entry(accuracy_details={"some": "breakdown"})
        _print_group("llama", "org/model", [(0, entry)])
        out = capsys.readouterr().out
        assert "accuracy_details" not in out

    def test_analysis_key_not_rendered_as_a_metric_column(self, capsys):
        entry = _entry(
            analysis={
                "name": "TruncationSimulation",
                "kwargs": {"truncation_bits": [8, 12]},
                "report": "analysis/foo.md",
            }
        )
        _print_group("llama", "org/model", [(0, entry)])
        out = capsys.readouterr().out
        assert "analysis" not in out.split("Recipe details:")[0]


class TestAnalysisFootnote:
    def test_footnote_includes_analysis_line_when_present(self):
        line = _build_footnote(
            "1",
            {"backbone": {"recipe": "Skip", "recipe_kwargs": {}}},
            analysis={
                "name": "TruncationSimulation",
                "kwargs": {"truncation_bits": [8, 12]},
                "report": "analysis/foo.md",
            },
        )
        assert (
            "Analysis: TruncationSimulation(truncation_bits=[8, 12]) -> analysis/foo.md"
            in line
        )

    def test_footnote_omits_analysis_line_when_absent(self):
        line = _build_footnote(
            "1", {"backbone": {"recipe": "Skip", "recipe_kwargs": {}}}
        )
        assert "Analysis:" not in line
