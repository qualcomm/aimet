#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

r"""CLI for grading LLM responses.

Reads a JSON file containing a list of items in the form::

    [
      {"idx": 0, "category": "knowledge", "prompt": "What is gravity?",
       "output": "Gravity is ..."},
      ...
    ]

The ``prompt`` is passed through to the grader as-is, and ``output`` is the
generated response to grade. Grading is delegated to
:mod:`GenAILab.qai_hub_lm.scoring.grace.grader`.

A closing pass sends the rationales back to the grader model and condenses them
into at most five recurring failure modes, stored under ``summary_items`` in
``--output-json``. Pass ``--no-summary`` to skip it.

Example usage::

    python -m GenAILab.bench.grade_responses responses.json \
        --output-json grader_summary.json

This runs as its own process on purpose. The grader is a 35B MoE, and once torch
has loaded it the caching allocator keeps ~60 GiB of segments reserved for the
rest of the process even after every tensor is freed. ONNX Runtime allocates
from the driver rather than through torch, so it cannot reuse those segments and
fails to rebuild its session afterwards. Process exit is the only thing that
reliably hands the memory back.

Ported from ``qai_hub_models/scripts/llm/grade_responses.py`` in
ai-hub-models-internal, and kept structurally aligned with it so upstream
changes are easy to bring over. Deliberate divergences, all marked "AIMET-only" below:

* ``--metric-name`` and ``--device-map`` arguments, which upstream does not have.
* No human-readable report is printed here. The parent process renders a richer
  one from the summary JSON (see ``format_grader_summary`` in
  :mod:`GenAILab.bench.metrics`), so printing it here would only duplicate it.
* ``build_summary`` therefore runs unconditionally, and the overall score is
  printed when no ``--output-json`` was given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from GenAILab.bench.determinism import set_seed
from GenAILab.qai_hub_lm.scoring.grace.grader import (
    DEFAULT_PROMPT_TEMPLATE,
    MAX_POINTS,
    ResponseGrader,
    resolve_device,
)
from GenAILab.qai_hub_lm.scoring.grace.report import build_summary

GRADER_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade LLM responses from a JSON file on a 0-10 rubric.",
    )
    parser.add_argument(
        "responses_json",
        type=str,
        nargs="?",
        help="Path to a JSON file: list of {idx, prompt, output} objects.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to grading prompt template (must contain {response}). "
        "If omitted, the default template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model id to use as grader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the grader model (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit grading on CPU. Without this, an unusable CUDA install is a "
        "hard error rather than a ~13-minute-per-file CPU fallback.",
    )
    parser.add_argument(
        "--check-device-only",
        action="store_true",
        help="Resolve and print the grader device, then exit. Lets CI fail fast on "
        "a misconfigured venv without loading a model or reading responses.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Grader model dtype.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a machine-readable summary to this path.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the closing summary pass over the rationales.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the per-item score and the grader's rationale.",
    )
    # AIMET-only arguments.
    parser.add_argument(
        "--metric-name",
        type=str,
        default="Grace",
        help="Label the report is filed under.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Passed to from_pretrained; 'auto' spreads the model and offloads.",
    )
    args = parser.parse_args()

    # Keep this before the model loads: once CUDA has started,
    # CUBLAS_WORKSPACE_CONFIG is silently ignored.
    _make_deterministic()

    # Resolved up front so a misconfigured venv fails before a model download.
    try:
        device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
    if args.check_device_only:
        print(f"grader device: {device}")
        return
    if args.responses_json is None:
        parser.error("responses_json is required unless --check-device-only is set.")

    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    items = json.loads(Path(args.responses_json).read_text())
    if not items:
        raise ValueError(f"No items found in {args.responses_json}")
    print(f"Loaded {len(items)} items from {args.responses_json}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    grader = ResponseGrader(
        model_id=args.model,
        device=device,
        dtype=dtype_map[args.dtype],
        prompt_template=prompt_template,
        allow_cpu=args.allow_cpu,
        device_map=args.device_map,  # AIMET-only
    )

    summary = grader.grade(items, summary=not args.no_summary)

    if args.verbose:
        for item, result in zip(items, summary.results, strict=True):
            print(
                f"  idx={item['idx']}: {result.points:2d}/{MAX_POINTS} pts"
                + ("  [skipped: empty response]" if result.skipped else "")
                + ("  [rating forced after token limit]" if result.forced else "")
                + ("  [GRADER FAILURE: no rating]" if not result.parsed else "")
            )
            if result.rationale:
                print(f"      {result.rationale}")

    # AIMET-only: upstream prints the score, the per-category table, the items
    # that lost points and the summary items here. The parent process prints all
    # of that from the summary JSON instead, so nothing is reported here.
    out = build_summary(
        items,
        summary,
        metric_name=args.metric_name,
        grader_model=args.model,
        input_file=str(args.responses_json),
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote grading summary to {args.output_json}")
    else:
        # Nothing above printed the score, so it would be lost on exit.
        print(
            f"Overall score: {out['score_pct']:.1f}%  "
            f"({out['total_points']}/{out['max_points']} pts)"
        )


def _make_deterministic() -> None:
    """Make the grader's output repeat bit for bit.

    Decoding is greedy, so a tiny numeric difference can flip a near-tied token
    and change the rest of the rationale, and sometimes the rating.
    """
    # Read only when the CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # warn_only: an op with no deterministic kernel warns instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # cuDNN attention, which torch picks for fp16/bf16 on H100, is not
    # repeatable, and the deterministic flag above does not cover it.
    # Torch falls back to flash attention.
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Nothing random is expected in a greedy decode; seeded to be safe.
    set_seed(GRADER_SEED)
    print(
        f"Grader deterministic mode: seed={GRADER_SEED}, "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}, "
        f"cudnn_sdp={torch.backends.cuda.cudnn_sdp_enabled()}"
    )


if __name__ == "__main__":
    main()
