# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Print a human-readable summary table from profiling_data.json."""

import json
import os
from collections import defaultdict


# Keys in each JSON entry that are NOT metric results
_NON_METRIC_KEYS = {
    "model_id",
    "model_modifiers",
    "precision",
    "environment",
    "components",
    "export",
    "run_group",
    "accuracy_details",
    "analysis",
}


def print_summary(results_json_path, config_path=None):
    """Read profiling_data.json and print a formatted summary table."""
    if not os.path.exists(results_json_path):
        return

    with open(results_json_path, "r") as f:
        data = json.load(f)

    # Collect all entries with their global doc number
    entries = []
    for model_type, items in data.items():
        for item in items:
            entries.append((model_type, item))

    if not entries:
        return

    # Group by (model_type, model_id)
    groups = defaultdict(list)
    for idx, (model_type, entry) in enumerate(entries):
        model_id = entry.get("model_id", "unknown")
        groups[(model_type, model_id)].append((idx, entry))

    # Print header
    config_label = f"    Config: {config_path}" if config_path else ""
    print()
    print("=" * 74)
    print(f" GenAI Scorecard Summary{config_label}")
    print("=" * 74)

    for (model_type, model_id), group_entries in groups.items():
        _print_group(model_type, model_id, group_entries)

    print("=" * 74)
    print(f" {len(entries)} experiment(s) completed{config_label}")
    print("=" * 74)
    print()


def _print_group(model_type, model_id, group_entries):
    """Print a single model group as a table with footnotes."""
    print()
    print(f" Model: {model_id}    Type: {model_type}")
    print()

    # Discover metric columns present in this group
    metric_names = []
    for _, entry in group_entries:
        for key in entry:
            if key not in _NON_METRIC_KEYS and key not in metric_names:
                metric_names.append(key)

    # Flag metrics whose results mix scoring versions within this table
    metric_versions = {}
    for name in metric_names:
        versions = set()
        for _, entry in group_entries:
            if name in entry:
                versions.add(entry[name].get("scoring_version", 1))
        metric_versions[name] = sorted(versions)

    mixed_version_metrics = [
        name for name, versions in metric_versions.items() if len(versions) > 1
    ]
    header_metric_names = [
        f"{name} (MIXED VERSIONS!)"
        if len(metric_versions[name]) > 1
        else f"{name} (v{metric_versions[name][0]})"
        for name in metric_names
    ]

    # Whether any entry has an encoder alongside the backbone, in which case the
    # recipe column is broken out per component. True for audio models too, not
    # just vision -- hence not "is_vlm".
    is_multi_component = any(
        len(entry.get("components", {})) > 1 for _, entry in group_entries
    )

    # Build table rows
    rows = []
    footnotes = []
    for doc_num, entry in group_entries:
        doc_label = str(doc_num + 1)
        components = entry.get("components", {})

        # Metric values
        metric_vals = []
        for m in metric_names:
            if m in entry:
                val = entry[m].get("result", "—")
                metric_vals.append(_format_metric(val))
            else:
                metric_vals.append("—")

        # Aggregate utilization across components
        total_elapsed_ms = 0
        peak_cuda_mb = 0
        for comp_stats in components.values():
            util = comp_stats.get("resource_utilization", {})
            total_elapsed_ms += util.get("elapsed_ms", 0)
            peak_cuda_mb = max(peak_cuda_mb, util.get("cuda_peak_mb", 0))

        # Recipe label for the table
        if is_multi_component:
            recipe_parts = []
            for comp_name, comp_stats in components.items():
                recipe_parts.append(f"{comp_name}: {comp_stats.get('recipe', '—')}")
            recipe_label = "\n".join(recipe_parts)
        else:
            comp = next(iter(components.values()), {})
            recipe_label = comp.get("recipe", "—")

        rows.append(
            {
                "doc": doc_label,
                "recipe": recipe_label,
                "metrics": metric_vals,
                "cuda_peak": f"{peak_cuda_mb:,.0f} MB" if peak_cuda_mb else "—",
                "time": _format_duration(total_elapsed_ms) if total_elapsed_ms else "—",
            }
        )

        # Build footnote
        precision = entry.get("precision")
        analysis = entry.get("analysis")
        footnotes.append(_build_footnote(doc_label, components, precision, analysis))

    # Calculate column widths
    metric_col_widths = []
    for i, name in enumerate(header_metric_names):
        vals = [r["metrics"][i] for r in rows]
        metric_col_widths.append(max(len(name), max((len(v) for v in vals), default=0)))

    doc_w = max(3, max(len(r["doc"]) for r in rows))
    recipe_lines = [line for r in rows for line in r["recipe"].split("\n")]
    recipe_w = max(6, max(len(line) for line in recipe_lines))
    cuda_w = max(9, max(len(r["cuda_peak"]) for r in rows))
    time_w = max(4, max(len(r["time"]) for r in rows))

    # Build format string
    def fmt_row(doc, recipe, metrics, cuda, time_val):
        parts = [f" {doc:>{doc_w}} ", f" {recipe:<{recipe_w}} "]
        for i, m in enumerate(metrics):
            parts.append(f" {m:>{metric_col_widths[i]}} ")
        parts.append(f" {cuda:>{cuda_w}} ")
        parts.append(f" {time_val:>{time_w}} ")
        return "|".join(parts)

    # Header
    header = fmt_row("#", "Recipe", header_metric_names, "CUDA Peak", "Time")
    total_w = len(header) + 4
    border = "+" + "-" * (total_w - 2) + "+"

    if mixed_version_metrics:
        print(
            " WARNING: mixed scoring versions for "
            f"{', '.join(mixed_version_metrics)} in this table -- these "
            "results were computed under different scoring semantics and "
            "must not be compared to each other."
        )

    print(f" {border}")
    print(f" | {header} |")
    print(f" {border.replace('-', '=')}")

    # Data rows
    for row in rows:
        lines = row["recipe"].split("\n")
        # First line gets all the data
        first = fmt_row(
            row["doc"], lines[0], row["metrics"], row["cuda_peak"], row["time"]
        )
        print(f" | {first} |")
        # VLM sub-rows for additional components
        for extra_line in lines[1:]:
            sub = fmt_row("", extra_line, [""] * len(metric_names), "", "")
            print(f" | {sub} |")

    print(f" {border}")

    # Footnotes
    print()
    print(" Recipe details:")
    for fn in footnotes:
        print(f"   {fn}")


def _build_footnote(doc_label, components, precision=None, analysis=None):
    """Build a footnote string describing recipe kwargs, precision, and any active analysis pass."""
    parts = []
    for comp_name, comp_stats in components.items():
        recipe = comp_stats.get("recipe", "?")
        kwargs = comp_stats.get("recipe_kwargs", {})
        dataset = comp_stats.get("dataset", "")
        dataset_kwargs = comp_stats.get("dataset_kwargs", {})

        param_strs = [f"{k}={v}" for k, v in kwargs.items()]
        if dataset:
            ds_parts = [dataset] + [f"{k}={v}" for k, v in dataset_kwargs.items()]
            param_strs.append(f"dataset={','.join(ds_parts)}")

        param_str = ", ".join(param_strs) if param_strs else "defaults"

        if len(components) > 1:
            parts.append(f"{comp_name}: {recipe}({param_str})")
        else:
            parts.append(f"{recipe}({param_str})")

    line = f"{doc_label}. {', '.join(parts)}"

    if precision:
        line += f"\n      Precision: {_format_precision(precision)}"

    if analysis:
        kwargs_str = ", ".join(
            f"{k}={v}" for k, v in analysis.get("kwargs", {}).items()
        )
        line += f"\n      Analysis: {analysis['name']}({kwargs_str}) -> {analysis['report']}"

    return line


def _format_precision(precision):
    """Format a precision dict into a compact human-readable string."""
    parts = []

    # Summarize blocks weight precision
    blocks = precision.get("blocks", {})
    if blocks:
        default_block = blocks.get("default", {})
        weight_qtype = default_block.get("qtype", "?")
        granularity = default_block.get("granularity", "")
        block_str = weight_qtype
        if granularity and granularity != "PCQ":
            block_str += f"/{granularity}"
            block_size = default_block.get("block_size")
            if block_size:
                block_str += f"({block_size})"
        parts.append(f"W={block_str}")

    act = precision.get("activations")
    if act:
        parts.append(f"A={act}")

    kv = precision.get("kv_cache")
    if kv:
        parts.append(f"KV={kv}")

    lm_head = precision.get("lm_head", {})
    if lm_head:
        parts.append(f"lm_head={lm_head.get('qtype', '?')}")

    visual = precision.get("visual")
    if visual:
        vw = visual.get("weight", {}).get("qtype", "?")
        va = visual.get("activations", "?")
        parts.append(f"visual=W{vw}A{va}")

    return ", ".join(parts)


def _format_metric(value):
    """Format a metric result for display."""
    if isinstance(value, float):
        if value > 100:
            return f"{value:,.1f}"
        return f"{value:.3f}"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    return str(value)


def _format_duration(ms):
    """Format milliseconds into a human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.0f}s"
    mins = secs / 60
    if mins < 60:
        whole_mins = int(mins)
        remaining_secs = int(secs - whole_mins * 60)
        if remaining_secs:
            return f"{whole_mins}m {remaining_secs}s"
        return f"{whole_mins}m"
    hours = int(mins / 60)
    remaining_mins = int(mins - hours * 60)
    if remaining_mins:
        return f"{hours}h {remaining_mins}m"
    return f"{hours}h"
