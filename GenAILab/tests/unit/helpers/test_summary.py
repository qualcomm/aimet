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


def _steps_component(*steps):
    """A component as ``profiler._serialize_component_stats`` writes it."""
    return {
        "steps": [
            {
                "recipe_name": name,
                "recipe_kwargs": kwargs,
                "dataset_name": dataset,
                "dataset_kwargs": dataset_kwargs,
                "resource_utilization": {},
            }
            for name, kwargs, dataset, dataset_kwargs in steps
        ],
        "recipe_name": "+".join(s[0] for s in steps),
        "resource_utilization": {},
    }


class TestRecipeStepChain:
    """Results written by the profiler carry a ``steps`` chain, not a flat ``recipe``."""

    def test_table_and_footnote_show_every_step(self, capsys):
        entry = _entry(
            components={
                "backbone": _steps_component(
                    ("SeqMSE", {"num_candidates": 20}, "Wikitext", {"split": "train"}),
                    ("Calibration", {}, "Wikitext", {"split": "train"}),
                )
            }
        )
        _print_group("llama", "org/model", [(0, entry)])
        out = capsys.readouterr().out
        assert "SeqMSE+Calibration" in out
        assert (
            "1. SeqMSE(num_candidates=20, dataset=Wikitext,split=train) + "
            "Calibration(dataset=Wikitext,split=train)"
        ) in out

    def test_component_without_steps_is_float(self, capsys):
        entry = _entry(
            components={
                "text_encoder": _steps_component(),
                "unet": _steps_component(
                    ("Calibration", {"num_prompts": 32}, "PartiPrompts", {})
                ),
            },
            precision={
                "unet": {
                    "weight": {"qtype": "int8", "granularity": "PCQ"},
                    "activations": "int16",
                }
            },
        )
        _print_group("StableDiffusionPipeline", "sd", [(0, entry)])
        out = capsys.readouterr().out
        assert "text_encoder: —" in out
        assert "unet: Calibration" in out
        assert (
            "1. text_encoder: none, unet: Calibration(num_prompts=32, "
            "dataset=PartiPrompts)"
        ) in out
        assert "Precision: unet=W:int8 A:int16" in out

    def test_legacy_flat_recipe_still_reads(self):
        line = _build_footnote(
            "1",
            {
                "backbone": {
                    "recipe": "Calibration",
                    "recipe_kwargs": {"num_batches": 32},
                    "dataset": "Wikitext",
                    "dataset_kwargs": {"split": "train"},
                }
            },
        )
        assert line == "1. Calibration(num_batches=32, dataset=Wikitext,split=train)"

    def test_llm_precision_is_unchanged(self):
        from GenAILab.bench.summary import _format_precision

        assert (
            _format_precision(
                {
                    "blocks": {
                        "default": {
                            "qtype": "int4",
                            "granularity": "LPBQ",
                            "block_size": 64,
                        }
                    },
                    "activations": "int16",
                    "kv_cache": "int8",
                    "embedding": "int16",
                    "lm_head": {"qtype": "int8", "granularity": "PCQ"},
                }
            )
            == "W=int4/LPBQ(64), A=int16, KV=int8, lm_head=int8"
        )
