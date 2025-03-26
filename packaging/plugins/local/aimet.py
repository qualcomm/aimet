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
import subprocess

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

    if enable_torch and enable_tensorflow and enable_onnx:
        variant = "tf-torch-"
    elif enable_tensorflow:
        variant = "tf-"
    elif enable_torch:
        variant = "torch-"
    elif enable_onnx:
        variant = "onnx-"
    else:
        raise RuntimeError("\n".join([
            "Only one or all of ENABLE_{TORCH, TENSORFLOW, ONNX} should set to ON."
            "Your passed:"
            f"  * ENABLE_TORCH:      {'ON' if enable_torch else 'OFF'}",
            f"  * ENABLE_ONNX:       {'ON' if enable_onnx else 'OFF'}",
            f"  * ENABLE_TENSORFLOW: {'ON' if enable_onnx else 'OFF'}",
        ]))

    variant += "gpu" if enable_cuda else "cpu"
    return variant


def get_name() -> str:
    aimet_variant = get_aimet_variant()

    # List of suffixes to remove
    suffixes = ["-cpu", "-gpu"]

    # Remove suffix from the aimet_variant
    for suffix in suffixes:
        if aimet_variant.endswith(suffix):
            aimet_variant = aimet_variant.replace(suffix, "")

    return f"aimet-{aimet_variant}"


def get_aimet_dependencies() -> list[str]:
    """Read dependencies form the corresponded files and return them as a list (!) of strings"""
    aimet_variant = get_aimet_variant()

    if aimet_variant in ("torch-gpu", "onnx-cpu", "tf-torch-cpu"):
        deps_path = pathlib.Path("packaging", "dependencies", "fast-release", aimet_variant)
    else:
        deps_path = pathlib.Path("packaging", "dependencies", aimet_variant)

    deps_files = [*deps_path.glob("reqs_pip_*.txt")]
    print(f"CMAKE_ARGS='{os.environ.get('CMAKE_ARGS', '')}'")
    print(f"Read dependencies for variant '{get_aimet_variant()}' from the following files: {deps_files}")
    deps = {d for d in itertools.chain.from_iterable(line.replace(" -f ", "\n-f ").split("\n") for f in deps_files for line in f.read_text(encoding="utf8").splitlines()) if not d.startswith(("#", "-f"))}
    return list(sorted(deps))


def get_cuda_version():
    cuda_version = ""
    try:
        # Run the nvcc command to get CUDA version
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        # Extract the version number from the output
        for line in output.split('\n'):
            if 'release' in line:
                cuda_version = line.split('release')[-1].strip().split(',')[0]
                # Remove the decimal point
                cuda_version = cuda_version.replace(".", "")
    except Exception:
        pass

    return cuda_version


def get_version() -> str:
    version = pathlib.Path("packaging", "version.txt").read_text(encoding="utf8").splitlines()[0]
    cuda_version = get_cuda_version()

    # For PyPi releases, just return the version without appending the variant string
    cmake_args = {k:v for k,v in (arg.split("=", 1) for arg in shlex.split(os.environ.get("CMAKE_ARGS", "")))}
    if cmake_args.get("-DPIP_INDEX", "") == "pypi":
        return version

    # Append the variant string to the original software version that was passed in
    variant_string = f"cu{cuda_version}" if cuda_version else "cpu"
    version = version + "+" + variant_string

    return version


def optional_dependencies() -> dict[str, list[str]]:
    optional_dependencies = {
        "dev": [
            # duplicate build-system.requires for editable mode (non-isolated)
            "scikit-build-core[wheels]>=0.10",
            # and the rest
        ],
        "test": [
            "beautifulsoup4",
            "deepspeed",
            "matplotlib",
            "onnxruntime-extensions",
            "onnxsim",
            "peft",
            "pylint<3",
            "pytest",
            "pytest-github-report",
            "pytorch-ignite",
            "safetensors",
            "torchvision",
            "transformers",
        ],
        "docs": [
            "furo",
            "nbsphinx",
            "pandoc",
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "sphinx-design",
            "sphinx-jinja",
            "sphinx-rtd-theme",
            "sphinx-tabs",
        ],
        "v1-deps": [] # This is empty for aimet-onnx and aimet-tensorflow
    }

    aimet_variant = get_aimet_variant()

    if aimet_variant not in ("torch-gpu", "torch-cpu"):
        return optional_dependencies

    optional_dependencies["test"].append("onnxruntime")

    try:
        import torch
    except ImportError:
        return optional_dependencies

    from packaging import version
    v = version.parse(torch.__version__)

    optional_dependencies["test"].append("spconv")
    optional_dependencies["v1-deps"].append(f"torch=={v.major}.{v.minor}.*")

    return optional_dependencies


def get_description() -> str:
    aimet_variant = get_aimet_variant()

    variant_map = {
        "torch": "AIMET torch package",
        "onnx": "AIMET onnx package",
        "tf": "AIMET tensorflow package"
    }

    for key in variant_map:
        if key in aimet_variant:
            return variant_map[key]

    raise RuntimeError("No matching AIMET variant found.")


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str | list[str] | dict[str, list[str]]:
    if settings:
        raise ValueError("No inline configuration is supported")
    if field == "name":
        return get_name()
    if field == "dependencies":
        return get_aimet_dependencies()
    if field == "optional-dependencies":
        return optional_dependencies()
    if field == "version":
        return get_version()
    if field == "description":
        return get_description()
    raise ValueError(f"Unsupported field '{field}'")
