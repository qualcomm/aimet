# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""General utils for GenAI model testing"""

import os
import json
import csv
import collections
import datetime
import platform
import subprocess
import sys
import fcntl
from contextlib import contextmanager
from dataclasses import asdict, dataclass

# Import GPUMeter from AIMETRegression evaluation module
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "../../AIMETRegression/evaluation",
    )
)
from metrics_utils import GPUMeter


@contextmanager
def _file_lock(filepath: str):
    """Context manager for process-safe file locking.

    Uses a separate .lock file to coordinate access between processes.
    """
    lock_path = filepath + ".lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def convert_gpu_meter_to_dict(
    profiler: GPUMeter, remove_finegrained: bool = False
) -> dict[str, str]:
    if profiler is None:
        return {}
    results = {
        "elapsed_ms": profiler.elapsed_ms,
        "cuda_starting_mb": profiler.cuda_first_mb,
        "cuda_peak_mb": profiler.cuda_peak_mb,
        "cuda_running_mb": profiler.cuda_running_mb if not remove_finegrained else None,
        "cpu_starting_mb": profiler.cpu_first_mb,
        "cpu_peak_mb": profiler.cpu_peak_mb,
        "cpu_running_mb": profiler.cpu_running_mb if not remove_finegrained else None,
    }
    return {key: value for key, value in results.items() if value is not None}


@dataclass
class ScoredResult:
    """What ``evaluate`` returns when it has more to report than one number.

    ``result`` keeps the same meaning as a bare returned score.
    """

    result: float | list[str]
    details: dict | None = None


@dataclass
class MetricResult:
    """Dataclass to hold accuracy and profiling results from running a metric"""

    metric_name: str
    result: float | list[str]
    profiler: GPUMeter
    scoring_version: int = 1  # EvaluationMetric.SCORING_VERSION; absent == 1
    # Breakdown; reported as the metric's own ``details`` key, omitted if empty.
    details: dict | None = None


def convert_metric_result_to_dict(
    result: MetricResult, remove_finegrained: bool = False
) -> dict:
    """Flatten one metric's result, profiling numbers and breakdown into a row.

    ``details`` is nested under the metric it belongs to, so everything a metric
    reported lives under ``accuracy_results-><metric>``. It is omitted when
    empty, keeping the shape of metrics that never report one. It can run to
    hundreds of kilobytes, so queries that only want scores should still project
    ``->'result'`` rather than the whole object.
    """
    row = {
        "result": result.result,
        "scoring_version": result.scoring_version,
    } | convert_gpu_meter_to_dict(result.profiler, remove_finegrained)
    if result.details:
        row["details"] = result.details
    return row


@dataclass
class RecipeStepStats:
    """Dataclass to hold recipe, dataset, and profiling info for a single recipe step"""

    recipe_name: str
    recipe_kwargs: dict
    dataset_name: str
    dataset_kwargs: dict
    profiler: GPUMeter = None


@dataclass
class ComponentRecipeStats:
    """Dataclass to hold a chain of recipe steps for a model component"""

    steps: list[RecipeStepStats]


@dataclass(frozen=True)
class AnalysisResult:
    """The sparse ``analysis`` column: which pass ran, its kwargs, and its report.

    ``report`` is a path relative to the results dir, so it survives the
    artifact download/merge path.
    """

    name: str
    kwargs: dict
    report: str


def recursive_update(d, u):
    """Internal helper function to update nested dict"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


_cached_environment = None


def _collect_environment():
    """Collect environment info: Python version, GPU, platform, key packages.

    Results are cached after the first call since the environment doesn't change
    within a single process.
    """
    global _cached_environment
    if _cached_environment is not None:
        return _cached_environment

    env = {
        "run_type": "local",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            env["pip_freeze"] = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    _cached_environment = env
    return env


def _aggregate_resource_utilization(
    steps: list[RecipeStepStats], remove_finegrained: bool = False
) -> dict:
    """Aggregate resource utilization across recipe steps for backward compat."""
    profilers = [s.profiler for s in steps if s.profiler is not None]
    if not profilers:
        return {}
    result = {
        "elapsed_ms": sum(p.elapsed_ms for p in profilers),
        "cuda_peak_mb": max((p.cuda_peak_mb for p in profilers), default=0),
        "cpu_peak_mb": max((p.cpu_peak_mb for p in profilers), default=0),
    }
    if not remove_finegrained:
        # Concatenate fine-grained traces across steps
        cuda_traces = [
            t for p in profilers if p.cuda_running_mb for t in p.cuda_running_mb
        ]
        cpu_traces = [
            t for p in profilers if p.cpu_running_mb for t in p.cpu_running_mb
        ]
        if cuda_traces:
            result["cuda_running_mb"] = cuda_traces
        if cpu_traces:
            result["cpu_running_mb"] = cpu_traces
    return {k: v for k, v in result.items() if v is not None}


def _serialize_component_stats(
    stats: ComponentRecipeStats, remove_finegrained: bool = False
) -> dict:
    """Serialize a ComponentRecipeStats to a dict with recipe_chain + backward-compat fields."""
    chain = [
        {
            "recipe_name": step.recipe_name,
            "recipe_kwargs": step.recipe_kwargs,
            "dataset_name": step.dataset_name,
            "dataset_kwargs": step.dataset_kwargs,
            "resource_utilization": convert_gpu_meter_to_dict(
                step.profiler, remove_finegrained=remove_finegrained
            ),
        }
        for step in stats.steps
    ]
    # Backward-compat summary fields
    return {
        "steps": chain,
        "recipe_name": "+".join(step.recipe_name for step in stats.steps),
        "resource_utilization": _aggregate_resource_utilization(
            stats.steps, remove_finegrained=remove_finegrained
        ),
    }


def _serialize_dtype(value: "str | torch.dtype") -> str:
    """Render a torch.dtype as a JSON-safe short name (e.g. "float16").

    Idempotent: a string passed in (already serialized) is returned as-is
    after the same prefix strip, so repeat calls don't double-mangle.
    """
    return str(value).removeprefix("torch.")


_CSV_HEADER = [
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


def _rotate_stale_csv(filename: str, expected_header: list[str]):
    """Rename ``filename`` aside if its header doesn't match ``expected_header``.

    Must be called with ``_file_lock(filename)`` already held, so a width or
    schema change (e.g. a new column) never misaligns a row into a
    pre-existing file -- the stale file is preserved rather than overwritten
    or corrupted, and a fresh file starts with the new header. The suffix
    loop finds the first free name, so repeated schema changes never clobber
    an earlier rotation.
    """
    if not os.path.exists(filename):
        return
    with open(filename, "r", newline="") as f:
        existing_header = next(csv.reader(f), None)
    if existing_header == expected_header:
        return
    suffix = 1
    while os.path.exists(f"{filename}.stale-{suffix}"):
        suffix += 1
    rotated = f"{filename}.stale-{suffix}"
    os.rename(filename, rotated)
    print(
        f"NOTICE: {filename} had an incompatible header (schema change); "
        f"rotated the existing file to {rotated} and started a new one."
    )


def write_stats_to_disk(
    output_folder: str,
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
    precision: dict | None = None,
    run_group: str | None = None,
    analysis: AnalysisResult | None = None,
):
    if "dtype" in model_modifiers:
        model_modifiers["dtype"] = _serialize_dtype(model_modifiers["dtype"])
    _write_stats_to_json(
        str(os.path.join(output_folder, filename + ".json")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
        precision,
        run_group,
        analysis,
    )

    _write_stats_to_csv(
        str(os.path.join(output_folder, filename + ".csv")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
        precision,
        run_group,
        analysis,
    )


def _write_stats_to_csv(
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
    precision: dict | None = None,
    run_group: str | None = None,
    analysis: AnalysisResult | None = None,
):
    def dict_to_postgres_csv_json_field(d):
        json_str = json.dumps(d, separators=(",", ":"))  # Compact JSON
        escaped = json_str.replace('"', '""')
        return f'"{escaped}"'

    accuracy_table = {
        result.metric_name: convert_metric_result_to_dict(
            result, remove_finegrained=True
        )
        for result in accuracy_results
    }

    # Convert components to serializable dict with recipe_chain + backward-compat summary
    components_dict = {
        name: _serialize_component_stats(stats, remove_finegrained=True)
        for name, stats in components.items()
    }

    stats = [
        model_type,
        model_id,
        dict_to_postgres_csv_json_field(model_modifiers),
        dict_to_postgres_csv_json_field(precision or {}),
        dict_to_postgres_csv_json_field(components_dict),
        dict_to_postgres_csv_json_field(accuracy_table),
        export_location if export_location is not None else "",
        dict_to_postgres_csv_json_field(_collect_environment()),
        run_group or "",
        dict_to_postgres_csv_json_field(asdict(analysis))
        if analysis is not None
        else "",
    ]

    # Use file lock to ensure process-safe writes
    with _file_lock(filename):
        _rotate_stale_csv(filename, _CSV_HEADER)

        if not os.path.exists(filename):
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(_CSV_HEADER)

        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stats)


def _write_stats_to_json(
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
    precision: dict | None = None,
    run_group: str | None = None,
    analysis: AnalysisResult | None = None,
):
    """Helper function to write collected stats to disk, only overwriting newly collected fields.

    This function is process-safe and uses file locking to prevent race conditions
    when multiple processes write to the same file.
    """
    stats = {
        "model_id": model_id,
        "model_modifiers": model_modifiers,
        "precision": precision or {},
        "environment": _collect_environment(),
        "components": {
            comp_name: _serialize_component_stats(comp_stats, remove_finegrained=False)
            for comp_name, comp_stats in components.items()
        },
    } | {
        result.metric_name: convert_metric_result_to_dict(result)
        for result in accuracy_results
    }

    if export_location is not None:
        stats["export"] = export_location

    if run_group is not None:
        stats["run_group"] = run_group

    if analysis is not None:
        stats["analysis"] = asdict(analysis)

    # Use file lock to ensure process-safe read-modify-write
    with _file_lock(filename):
        # Check if the file exists
        if os.path.exists(filename):
            # Open the file and load the existing data
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            # If the file does not exist, create an empty dictionary
            data = {}

        # Append to list of results for this model_type
        if model_type not in data:
            data[model_type] = []
        data[model_type].append(stats)

        # Write the updated dictionary back to the file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


def _read_json_data(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


def _append_json_entry(data, model_type, entry):
    if model_type not in data:
        data[model_type] = []
    data[model_type].append(entry)


def merge_json_results(source_path, dest_path):
    """Merge entries from a source profiling_data.json into a destination file.

    Each model_type key in the source has its entries appended to the
    corresponding list in the destination.
    """
    if not os.path.exists(source_path):
        return 0

    with open(source_path, "r") as f:
        source_data = json.load(f)

    dest_data = _read_json_data(dest_path)

    count = 0
    for model_type, entries in source_data.items():
        for entry in entries:
            _append_json_entry(dest_data, model_type, entry)
            count += 1

    with open(dest_path, "w") as f:
        json.dump(dest_data, f, indent=4)

    return count


def merge_csv_results(source_path, dest_path):
    """Append rows from a source profiling_data.csv into a destination file.

    Creates the destination with headers if it doesn't exist.
    """
    if not os.path.exists(source_path):
        return 0

    with open(source_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        return 0

    header = rows[0]
    data_rows = rows[1:]

    if os.path.exists(dest_path):
        with open(dest_path, "r", newline="") as f:
            dest_header = next(csv.reader(f), None)
        if dest_header != header:
            raise ValueError(
                f"Cannot merge {source_path} into {dest_path}: header mismatch "
                f"(source={header}, dest={dest_header}). Merging would silently "
                f"misalign columns."
            )
    else:
        with open(dest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(dest_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)

    return len(data_rows)
