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

import sys

from GenAILab.bench.diffusion import runner

FRAMEWORK = "torch"


def run(args) -> int:
    """Run every document in ``args.config``. Returns the process exit code."""
    return runner.run(args, FRAMEWORK)


def build_parser():
    return runner.build_parser(FRAMEWORK)


def main(argv=None):
    sys.exit(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
