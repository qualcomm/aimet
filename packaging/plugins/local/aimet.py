# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from __future__ import annotations

import itertools
import os
import pathlib
import shlex

__all__ = ["dynamic_metadata"]


def __dir__() -> list[str]:
    return __all__


def is_cmake_option_enabled(option_name: str) -> bool:
    """Returns True if CMAKE_ARGS environment variable contains `-D{option_name}=ON/YES/1/TRUE` and False otherwise."""
    cmake_args = {k:v for k,v in (arg.split("=", 1) for arg in shlex.split(os.environ.get("CMAKE_ARGS", "")))}
    return not cmake_args.get(f"-D{option_name}", "").upper() in {"OFF", "NO", "FALSE", "0", "N" }


def get_aimet_variant() -> str:
    """Return a variant based on CMAKE_ARGS environment variable"""
    enable_cuda = is_cmake_option_enabled("ENABLE_CUDA")
    enable_torch = is_cmake_option_enabled("ENABLE_TORCH")
    enable_tensorflow = is_cmake_option_enabled("ENABLE_TENSORFLOW")
    enable_onnx = is_cmake_option_enabled("ENABLE_ONNX")

    variant = ""
    if enable_tensorflow:
        variant += "tf-"
    if enable_torch:
        variant += "torch-"
    if enable_onnx:
        variant = "onnx-"
    variant += "gpu" if enable_cuda else "cpu"
    return variant


def get_aimet_dependencies() -> list[str]:
    """Read dependencies form the corresponded files and return them as a list (!) of strings"""
    aimet_variant = get_aimet_variant()
    deps_path = pathlib.Path("packaging", "dependencies", "fast-release" if aimet_variant in ("torch-gpu",) else "", aimet_variant)
    deps_files = [*deps_path.glob("reqs_pip_*.txt")]
    print(f"CMAKE_ARGS='{os.environ.get('CMAKE_ARGS', '')}'")
    print(f"Read dependencies for variant '{get_aimet_variant()}' from the following files: {deps_files}")
    deps = {d for d in itertools.chain.from_iterable(line.replace(" -f ", "\n-f ").split("\n") for f in deps_files for line in f.read_text(encoding="utf8").splitlines()) if not d.startswith(("#", "-f"))}
    return list(sorted(deps))


def get_version() -> str:
    return pathlib.Path("packaging", "version.txt").read_text(encoding="utf8")


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str:
    if settings:
        raise ValueError("No inline configuration is supported")
    if field == "name":
        return f"aimet-{get_aimet_variant()}"
    if field == "dependencies":
        return get_aimet_dependencies()
    if field == "version":
        return get_version()
    raise ValueError(f"Unsupported field '{field}'")
