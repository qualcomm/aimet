# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs every text-to-image diffusion experiment in a config file.

Shared by the per-framework entry points, ``GenAILab/bench/torch/run_diffusion.py``
and ``GenAILab/bench/onnx/run_diffusion.py``.
"""

import argparse
import pdb
import traceback
from pathlib import Path

import yaml

from GenAILab.bench.fp_cache import DiskBackedFPCache

# Same defaults as the LLM runner, so both write to GenAILab/artifacts/.
_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts"


def _experiment_label(index, total, doc):
    model_id = (doc.get("model") or {}).get("model_id", "<unknown model>")
    return f"[{index}/{total}] {model_id}"


def run(args, framework: str) -> int:
    """Run every document in ``args.config``. Returns the process exit code."""
    from GenAILab.bench.diffusion.experiment import run_experiment

    with open(args.config, "r") as f:
        docs = list(yaml.safe_load_all(f))

    fp_cache = DiskBackedFPCache(args.fp_cache_dir)
    if args.clear_fp_cache:
        fp_cache.clear()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    framework_kwargs = {"framework": framework}
    if framework == "onnx":
        from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache

        onnx_cache = DiffusionONNXCache(args.onnx_cache_dir)
        if args.clear_onnx_cache:
            onnx_cache.clear()
        framework_kwargs["onnx_cache"] = onnx_cache

    passed, failed = [], []
    for index, doc in enumerate(docs, start=1):
        if doc is None:
            print(f"[{index}/{len(docs)}] Skipping empty config document.")
            continue
        label = _experiment_label(index, len(docs), doc)
        print(f"\n===== diffusion ({framework}): {label} =====", flush=True)
        try:
            run_experiment(
                doc,
                fp_cache=fp_cache,
                results_dir=args.results_dir,
                **framework_kwargs,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            traceback.print_exc()
            if args.pdb:
                pdb.post_mortem(e.__traceback__)
            failed.append(label)
            if args.fail_fast:
                break
        else:
            passed.append(label)

    print(
        f"\n===== diffusion ({framework}): {len(passed)} passed, "
        f"{len(failed)} failed =====",
        flush=True,
    )
    for label in failed:
        print(f"FAILED {label}")

    results_json = Path(args.results_dir) / "profiling_data.json"
    if results_json.exists():
        from GenAILab.bench.summary import print_summary

        print_summary(str(results_json), args.config)
    return 1 if failed else 0


def build_parser(framework: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"GenAILab text-to-image diffusion runner ({framework})"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--results-dir",
        default=str(_OUTPUT_ROOT / "results"),
        help="Directory for profiling results and images (default: %(default)s)",
    )
    parser.add_argument(
        "--fp-cache-dir",
        default=str(_OUTPUT_ROOT / "cache" / "fp"),
        help="Directory for FP results cache (default: %(default)s)",
    )
    parser.add_argument(
        "--clear-fp-cache",
        action="store_true",
        default=False,
        help="Clear the FP cache before running (default: %(default)s)",
    )
    if framework == "onnx":
        parser.add_argument(
            "--onnx-cache-dir",
            default=str(_OUTPUT_ROOT / "cache" / "diffusion_onnx"),
            help="Directory for exported ONNX graphs (default: %(default)s)",
        )
        parser.add_argument(
            "--clear-onnx-cache",
            action="store_true",
            default=False,
            help="Clear the exported ONNX graphs before running (default: %(default)s)",
        )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop after the first failed experiment (default: %(default)s)",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Start the debugger when an experiment fails (default: %(default)s)",
    )
    return parser
