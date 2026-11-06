# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs every text-to-image diffusion experiment in a config file (torch, FP).

A standalone entry point while diffusion support is young; the LLM harness
(``python -m GenAILab``) does not run diffusion configs yet::

    python -m GenAILab.bench.torch.run_diffusion \\
        --config GenAILab/bench/diffusion/configs/diffusion_fp.yaml

Results are appended to ``profiling_data.json``/``.csv`` in ``--results-dir``,
and ``ImageDump`` images are written under ``<results-dir>/images/``.
"""

import argparse
import pdb
import sys
import traceback
from pathlib import Path

import yaml

from GenAILab.bench.fp_cache import DiskBackedFPCache

# Same defaults as the LLM runner, so both write to GenAILab/artifacts/.
_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts"


def _experiment_label(index, total, doc):
    model_id = (doc.get("model") or {}).get("model_id", "<unknown model>")
    return f"[{index}/{total}] {model_id}"


def run(args) -> int:
    """Run every document in ``args.config``. Returns the process exit code."""
    from GenAILab.bench.diffusion.experiment import run_experiment

    with open(args.config, "r") as f:
        docs = list(yaml.safe_load_all(f))

    fp_cache = DiskBackedFPCache(args.fp_cache_dir)
    if args.clear_fp_cache:
        fp_cache.clear()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    passed, failed = [], []
    for index, doc in enumerate(docs, start=1):
        if doc is None:
            print(f"[{index}/{len(docs)}] Skipping empty config document.")
            continue
        label = _experiment_label(index, len(docs), doc)
        print(f"\n===== diffusion: {label} =====", flush=True)
        try:
            run_experiment(doc, fp_cache=fp_cache, results_dir=args.results_dir)
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
        f"\n===== diffusion: {len(passed)} passed, {len(failed)} failed =====",
        flush=True,
    )
    for label in failed:
        print(f"FAILED {label}")

    results_json = Path(args.results_dir) / "profiling_data.json"
    if results_json.exists():
        from GenAILab.bench.summary import print_summary

        print_summary(str(results_json), args.config)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GenAILab text-to-image diffusion runner"
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


def main(argv=None):
    sys.exit(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
