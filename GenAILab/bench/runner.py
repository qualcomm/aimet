# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs every experiment in a GenAILab config file for one framework.

Usually launched by ``python -m GenAILab``, which starts one runner process
per framework. It can also be run directly::

    python -m GenAILab.bench.runner --framework onnx --config cfg.yaml
"""

import argparse
import ctypes
import ctypes.util
import pdb
import sys
import traceback
import warnings
from pathlib import Path

import yaml

# All generated artifacts live under GenAILab/artifacts/ so they don't
# clutter the repo root and can be excluded from rsync with a single path.
_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def add_run_arguments(parser):
    """Add the runner flags to ``parser`` and return the created actions."""
    return [
        parser.add_argument(
            "--force-export",
            action="store_true",
            default=False,
            help="Force export of all artifacts (default: %(default)s)",
        ),
        parser.add_argument(
            "--export-dir",
            default=str(_OUTPUT_ROOT / "exports"),
            help="Base directory for exported model artifacts (default: %(default)s)",
        ),
        parser.add_argument(
            "--results-dir",
            default=str(_OUTPUT_ROOT / "results"),
            help="Directory for profiling results (default: %(default)s)",
        ),
        parser.add_argument(
            "--fp-cache-dir",
            default=str(_OUTPUT_ROOT / "cache" / "fp"),
            help="Directory for FP results cache (default: %(default)s)",
        ),
        parser.add_argument(
            "--clear-fp-cache",
            action="store_true",
            default=False,
            help="Clear the FP cache before running (default: %(default)s)",
        ),
        parser.add_argument(
            "--model-cache-dir",
            default=str(_OUTPUT_ROOT / "cache" / "model"),
            help="Directory for ONNX model cache (default: %(default)s)",
        ),
        parser.add_argument(
            "--clear-model-cache",
            action="store_true",
            default=False,
            help="Clear the model cache before running (default: %(default)s)",
        ),
        parser.add_argument(
            "--recipe-cache-dir",
            default=str(_OUTPUT_ROOT / "cache" / "recipe"),
            help="Directory for recipe chain cache (default: %(default)s)",
        ),
        parser.add_argument(
            "--no-recipe-cache",
            action="store_true",
            default=False,
            help="Disable recipe chain caching (default: %(default)s)",
        ),
        parser.add_argument(
            "--clear-recipe-cache",
            action="store_true",
            default=False,
            help="Clear the recipe cache before running (default: %(default)s)",
        ),
        parser.add_argument(
            "--fail-fast",
            action="store_true",
            default=False,
            help="Stop after the first failed experiment (default: %(default)s)",
        ),
        parser.add_argument(
            "--pdb",
            action="store_true",
            default=False,
            help="Start the debugger when an experiment fails (default: %(default)s)",
        ),
    ]


def _configure_process():
    """Process-wide setup that must happen before torch/onnxruntime are imported."""
    # glibc's malloc keeps freed large blocks in per-thread arenas instead of
    # returning them to the OS once the arena has already grown, so host RSS
    # climbs monotonically across model-load/export cycles even though live
    # heap usage doesn't (https://github.com/pytorch/pytorch/issues/41486).
    # Lowering the mmap/trim thresholds forces those large blocks through
    # mmap/munmap instead, which glibc does release immediately. mallopt() takes
    # effect for all allocations made after the call, so this must run before
    # torch/onnxruntime/transformers do any sizeable allocating.
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        m_trim_threshold = -1
        m_mmap_threshold = -3
        libc.mallopt(m_trim_threshold, 65536)
        libc.mallopt(m_mmap_threshold, 65536)
    except (OSError, AttributeError):
        pass  # non-glibc platform; no-op

    import onnxruntime as ort

    # Suppress verbose onnxruntime INFO/WARNING messages (e.g. EP selection,
    # graph optimisation notes) that clutter the output.  Level 3 = ERROR only.
    ort.set_default_logger_severity(3)


def load_documents(config_path, force_export=False):
    """Load every YAML document in ``config_path``, applying ``--force-export``."""
    with open(config_path, "r") as f:
        docs = list(yaml.safe_load_all(f))
    if force_export:
        for doc in docs:
            if doc is None:
                continue
            if doc.get("export") and doc["export"] != True:
                warnings.warn(
                    "force_export is True, all artifacts will be saved to default export path."
                )
            doc["export"] = True
    return docs


def _experiment_label(index, total, doc):
    model_id = (doc.get("model") or {}).get("model_id", "<unknown model>")
    return f"[{index}/{total}] {model_id}"


def run(args):
    """Run every experiment in ``args.config``. Returns the process exit code."""
    from GenAILab.bench.fp_cache import DiskBackedFPCache
    from GenAILab.bench.recipe_cache import RecipeCache

    if args.framework == "onnx":
        from GenAILab.bench.model_cache import DiskBackedModelCache
        from GenAILab.bench.onnx.run_genai import run_experiment
    else:
        from GenAILab.bench.torch.run_genai import run_experiment

    docs = load_documents(args.config, args.force_export)

    # Caches are built once and shared by every experiment, so FP outputs,
    # ONNX exports and recipe steps are computed at most once per run.
    fp_cache = DiskBackedFPCache(args.fp_cache_dir)
    if args.clear_fp_cache:
        fp_cache.clear()

    recipe_cache = None
    if not args.no_recipe_cache:
        recipe_cache = RecipeCache(args.recipe_cache_dir)
        if args.clear_recipe_cache:
            recipe_cache.clear()

    Path(args.export_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        fp_cache=fp_cache,
        recipe_cache=recipe_cache,
        export_dir=args.export_dir,
        results_dir=args.results_dir,
    )
    if args.framework == "onnx":
        model_cache = DiskBackedModelCache(args.model_cache_dir)
        if args.clear_model_cache:
            model_cache.clear()
        kwargs["model_cache"] = model_cache

    passed, failed = [], []
    for index, doc in enumerate(docs, start=1):
        if doc is None:
            print(f"[{index}/{len(docs)}] Skipping empty config document.")
            continue
        label = _experiment_label(index, len(docs), doc)
        print(f"\n===== {args.framework}: {label} =====", flush=True)
        try:
            run_experiment(doc, **kwargs)
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
        f"\n===== {args.framework}: {len(passed)} passed, {len(failed)} failed =====",
        flush=True,
    )
    for label in failed:
        print(f"FAILED {label}")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GenAILab experiment runner")
    parser.add_argument("--framework", choices=["torch", "onnx"], required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    add_run_arguments(parser)
    args = parser.parse_args(argv)

    _configure_process()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
