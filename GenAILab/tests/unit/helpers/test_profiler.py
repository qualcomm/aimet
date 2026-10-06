# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for profiler stats writing and merging."""

import csv
import json
import os

import pytest

from GenAILab.bench.profiler import (
    write_stats_to_disk,
    merge_json_results,
    merge_csv_results,
    convert_gpu_meter_to_dict,
    convert_metric_result_to_dict,
    _collect_environment,
    ComponentRecipeStats,
    RecipeStepStats,
    MetricResult,
    ScoredResult,
    AnalysisResult,
)


def _unwrap_postgres_json(cell: str):
    """Parse a cell written by ``dict_to_postgres_csv_json_field``."""
    # Those cells carry their own quotes, which csv.reader hands back verbatim.
    return json.loads(cell.strip('"').replace('""', '"'))


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


def _write_sample(results_dir, model_type="llama", model_id="org/model"):
    components = {
        "backbone": ComponentRecipeStats(
            steps=[
                RecipeStepStats(
                    recipe_name="Calibration",
                    recipe_kwargs={"num_batches": 32},
                    dataset_name="C4",
                    dataset_kwargs={"split": "en"},
                    profiler=None,
                )
            ]
        )
    }
    accuracy_results = [
        MetricResult(metric_name="PPL", result=12.5, profiler=None),
    ]
    write_stats_to_disk(
        output_folder=results_dir,
        filename="profiling_data",
        model_type=model_type,
        model_id=model_id,
        model_modifiers={"context_length": 64},
        components=components,
        accuracy_results=accuracy_results,
    )


class TestWriteStats:
    def test_json_creates_file(self, results_dir):
        _write_sample(results_dir)
        json_path = os.path.join(results_dir, "profiling_data.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "llama" in data
        assert len(data["llama"]) == 1
        assert data["llama"][0]["model_id"] == "org/model"

    def test_json_records_scoring_version(self, results_dir):
        components = {
            "backbone": ComponentRecipeStats(
                steps=[
                    RecipeStepStats(
                        recipe_name="Calibration",
                        recipe_kwargs={},
                        dataset_name="C4",
                        dataset_kwargs={},
                        profiler=None,
                    )
                ]
            )
        }
        accuracy_results = [
            MetricResult(
                metric_name="MMMU", result=48.2, profiler=None, scoring_version=2
            ),
            MetricResult(
                metric_name="PPL", result=12.5, profiler=None
            ),  # default version
        ]
        write_stats_to_disk(
            output_folder=results_dir,
            filename="profiling_data",
            model_type="llama",
            model_id="org/model",
            model_modifiers={"context_length": 64},
            components=components,
            accuracy_results=accuracy_results,
        )
        json_path = os.path.join(results_dir, "profiling_data.json")
        with open(json_path) as f:
            data = json.load(f)
        entry = data["llama"][0]
        assert entry["MMMU"]["scoring_version"] == 2
        assert entry["PPL"]["scoring_version"] == 1

    def test_json_appends(self, results_dir):
        _write_sample(results_dir, model_id="org/model1")
        _write_sample(results_dir, model_id="org/model2")
        json_path = os.path.join(results_dir, "profiling_data.json")
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["llama"]) == 2

    def test_csv_creates_with_header(self, results_dir):
        _write_sample(results_dir)
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == [
            "model_type",
            "model_id",
            "model_modifiers",
            "precision",
            "components",
            "accuracy_results",
            "export",
            "environment",
            "run_group",
            "analysis",
        ]
        assert len(rows) == 2  # header + 1 data row

    def test_csv_appends_without_extra_header(self, results_dir):
        _write_sample(results_dir, model_id="org/model1")
        _write_sample(results_dir, model_id="org/model2")
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        # First row is header
        assert rows[0][0] == "model_type"

    def test_csv_json_fields_escaped(self, results_dir):
        _write_sample(results_dir)
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # model_modifiers column uses postgres CSV JSON format (quoted + escaped)
        raw = rows[1][2]
        # Unwrap postgres format: strip outer quotes, unescape doubled quotes
        inner = raw.strip('"').replace('""', '"')
        parsed = json.loads(inner)
        assert parsed["context_length"] == 64


class TestAnalysisColumn:
    """The sparse ``analysis`` column: omitted when absent, populated when given."""

    def test_omitted_from_json_when_none(self, results_dir):
        _write_sample(results_dir)
        with open(os.path.join(results_dir, "profiling_data.json")) as f:
            entry = json.load(f)["llama"][0]
        assert "analysis" not in entry

    def test_empty_in_csv_when_none(self, results_dir):
        _write_sample(results_dir)
        with open(os.path.join(results_dir, "profiling_data.csv")) as f:
            header, row = list(csv.reader(f))
        cells = dict(zip(header, row, strict=True))
        assert cells["analysis"] == ""

    def test_recorded_in_json(self, results_dir):
        write_stats_to_disk(
            output_folder=results_dir,
            filename="profiling_data",
            model_type="llama",
            model_id="org/model",
            model_modifiers={"context_length": 64},
            components={},
            accuracy_results=[
                MetricResult(metric_name="Grace", result=68.1, profiler=None)
            ],
            analysis=AnalysisResult(
                name="TruncationSimulation",
                kwargs={"truncation_bits": [8, 12], "metrics": ["Grace"]},
                report="analysis/foo.md",
            ),
        )
        with open(os.path.join(results_dir, "profiling_data.json")) as f:
            entry = json.load(f)["llama"][0]
        assert entry["analysis"] == {
            "name": "TruncationSimulation",
            "kwargs": {"truncation_bits": [8, 12], "metrics": ["Grace"]},
            "report": "analysis/foo.md",
        }

    def test_recorded_in_csv(self, results_dir):
        write_stats_to_disk(
            output_folder=results_dir,
            filename="profiling_data",
            model_type="llama",
            model_id="org/model",
            model_modifiers={"context_length": 64},
            components={},
            accuracy_results=[
                MetricResult(metric_name="Grace", result=68.1, profiler=None)
            ],
            analysis=AnalysisResult(
                name="TruncationSimulation",
                kwargs={"truncation_bits": [8, 12]},
                report="analysis/foo.md",
            ),
        )
        with open(os.path.join(results_dir, "profiling_data.csv")) as f:
            header, row = list(csv.reader(f))
        cells = dict(zip(header, row, strict=True))
        parsed = _unwrap_postgres_json(cells["analysis"])
        assert parsed == {
            "name": "TruncationSimulation",
            "kwargs": {"truncation_bits": [8, 12]},
            "report": "analysis/foo.md",
        }


class TestCsvRotation:
    """A stale-header CSV is rotated aside rather than misaligning columns."""

    def _seed_stale_csv(self, csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_type", "model_id"])  # old, narrower header
            writer.writerow(["llama", "org/old-model"])

    def test_stale_header_is_rotated_not_misaligned(self, results_dir):
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        self._seed_stale_csv(csv_path)

        _write_sample(results_dir)

        rotated_path = csv_path + ".stale-1"
        assert os.path.exists(rotated_path)
        with open(rotated_path) as f:
            rotated_rows = list(csv.reader(f))
        assert rotated_rows == [
            ["model_type", "model_id"],
            ["llama", "org/old-model"],
        ]

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[0][-1] == "analysis"
        assert len(rows) == 2  # fresh header + the new row

    def test_repeated_rotation_preserves_every_stale_file(self, results_dir):
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        self._seed_stale_csv(csv_path)
        _write_sample(results_dir)  # rotates the old file to .stale-1

        # Simulate a second schema change: hand-write a header that again
        # doesn't match _CSV_HEADER, forcing a second rotation.
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_type", "model_id", "something_else"])
            writer.writerow(["llama", "org/mid-model", "x"])

        _write_sample(results_dir)  # rotates again, to .stale-2

        assert os.path.exists(csv_path + ".stale-1")
        assert os.path.exists(csv_path + ".stale-2")
        with open(csv_path + ".stale-1") as f:
            assert list(csv.reader(f))[1] == ["llama", "org/old-model"]
        with open(csv_path + ".stale-2") as f:
            assert list(csv.reader(f))[1] == ["llama", "org/mid-model", "x"]

    def test_no_rotation_when_header_matches(self, results_dir):
        _write_sample(results_dir, model_id="org/model1")
        _write_sample(results_dir, model_id="org/model2")
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        assert not os.path.exists(csv_path + ".stale-1")
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 data rows, no rotation


class TestMerge:
    def test_merge_json(self, tmp_path):
        src = tmp_path / "src.json"
        dst = tmp_path / "dst.json"
        src.write_text(json.dumps({"llama": [{"model_id": "a"}]}))
        dst.write_text(json.dumps({"llama": [{"model_id": "b"}]}))
        count = merge_json_results(str(src), str(dst))
        assert count == 1
        with open(dst) as f:
            data = json.load(f)
        assert len(data["llama"]) == 2

    def test_merge_json_empty_source(self, tmp_path):
        count = merge_json_results(
            str(tmp_path / "nonexistent.json"), str(tmp_path / "dst.json")
        )
        assert count == 0

    def test_merge_csv(self, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        src.write_text("a,b\n1,2\n3,4\n")
        dst.write_text("a,b\n5,6\n")
        count = merge_csv_results(str(src), str(dst))
        assert count == 2
        with open(dst) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # header + 1 existing + 2 new

    def test_merge_csv_creates_dest(self, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        src.write_text("a,b\n1,2\n")
        count = merge_csv_results(str(src), str(dst))
        assert count == 1
        assert dst.exists()

    def test_merge_csv_header_mismatch_raises(self, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        src.write_text("a,b,c\n1,2,3\n")
        dst.write_text("a,b\n5,6\n")
        with pytest.raises(ValueError, match="header mismatch"):
            merge_csv_results(str(src), str(dst))
        # Dest is untouched -- no partial merge on a rejected header.
        with open(dst) as f:
            assert list(csv.reader(f)) == [["a", "b"], ["5", "6"]]


class TestHelpers:
    def test_collect_environment(self):
        env = _collect_environment()
        assert "python_version" in env
        assert "platform" in env
        assert env["run_type"] == "local"

    def test_collect_environment_cached(self):
        env1 = _collect_environment()
        env2 = _collect_environment()
        assert env1 is env2

    def test_convert_gpu_meter_none(self):
        assert convert_gpu_meter_to_dict(None) == {}


class TestMetricDetails:
    """``details`` is reported under the metric that produced it."""

    def test_metric_row_carries_details(self):
        details = {"category_scores": {"math": {"score_pct": 70.0}}}
        row = convert_metric_result_to_dict(
            MetricResult(
                metric_name="Grace", result=87.5, profiler=None, details=details
            )
        )
        assert row == {"result": 87.5, "scoring_version": 1, "details": details}

    def test_key_omitted_for_a_metric_without_details(self):
        without = convert_metric_result_to_dict(
            MetricResult(metric_name="Grace", result=87.5, profiler=None)
        )
        empty = convert_metric_result_to_dict(
            MetricResult(metric_name="Grace", result=87.5, profiler=None, details={})
        )
        assert without == empty == {"result": 87.5, "scoring_version": 1}

    def test_written_to_json_under_the_metric(self, results_dir):
        details = {"summary_items": ["repeated words (3 items)"], "num_unparsed": 0}
        write_stats_to_disk(
            output_folder=results_dir,
            filename="profiling_data",
            model_type="llama",
            model_id="org/model",
            model_modifiers={"context_length": 64},
            components={},
            accuracy_results=[
                MetricResult(
                    metric_name="Grace", result=87.5, profiler=None, details=details
                ),
                MetricResult(metric_name="PPL", result=12.5, profiler=None),
            ],
        )
        with open(os.path.join(results_dir, "profiling_data.json")) as f:
            entry = json.load(f)["llama"][0]
        assert entry["Grace"]["result"] == 87.5
        assert entry["Grace"]["details"] == details
        assert "details" not in entry["PPL"]
        assert "accuracy_details" not in entry

    def test_written_to_csv_under_the_metric(self, results_dir):
        details = {"summary_items": ["repeated words (3 items)"]}
        write_stats_to_disk(
            output_folder=results_dir,
            filename="profiling_data",
            model_type="llama",
            model_id="org/model",
            model_modifiers={"context_length": 64},
            components={},
            accuracy_results=[
                MetricResult(
                    metric_name="Grace", result=87.5, profiler=None, details=details
                )
            ],
        )
        with open(os.path.join(results_dir, "profiling_data.csv")) as f:
            header, row = list(csv.reader(f))
        cells = dict(zip(header, row, strict=True))
        assert "accuracy_details" not in cells
        accuracy = _unwrap_postgres_json(cells["accuracy_results"])
        assert accuracy["Grace"]["details"] == details

    def test_scored_result_defaults_to_no_details(self):
        assert ScoredResult(result=1.0).details is None
