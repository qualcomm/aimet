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


def get_aimet_dependencies() -> list[str]:
    """Read dependencies form the corresponded files and return them as a list (!) of strings"""
    aimet_variant = get_aimet_variant()

    if aimet_variant in ("torch-gpu", "tf-torch-cpu"):
        deps_path = pathlib.Path("packaging", "dependencies", "fast-release", aimet_variant)
    else:
        deps_path = pathlib.Path("packaging", "dependencies", aimet_variant)

    deps_files = [*deps_path.glob("reqs_pip_*.txt")]
    print(f"CMAKE_ARGS='{os.environ.get('CMAKE_ARGS', '')}'")
    print(f"Read dependencies for variant '{get_aimet_variant()}' from the following files: {deps_files}")
    deps = {d for d in itertools.chain.from_iterable(line.replace(" -f ", "\n-f ").split("\n") for f in deps_files for line in f.read_text(encoding="utf8").splitlines()) if not d.startswith(("#", "-f"))}
    return list(sorted(deps))


def get_version() -> str:
    return pathlib.Path("packaging", "version.txt").read_text(encoding="utf8")


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
            "peft",
            "pylint<3",
            "pytest",
            "pytest-github-report",
            "pytorch-ignite",
            "safetensors",
            "spconv",
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

    import sys
    import sysconfig

    py_impl = sys.implementation
    assert py_impl.name == "cpython"
    assert sysconfig.get_platform() == "linux-x86_64"

    major, minor, *_ = py_impl.version
    platform = f"cp{major}{minor}-cp{major}{minor}-linux_x86_64"

    from packaging import version
    v = version.parse(torch.__version__)
    torch_pkg = f"torch-{v.major}.{v.minor}.{v.micro}"

    if aimet_variant == 'torch-gpu':
        major, minor = torch.version.cuda.split('.')
        cuda = f"cu{major}{minor}"
    else:
        cuda = "cpu"

    torch_pkg += f"%2B{cuda}"

    optional_dependencies.update({
        "v1-deps": [
            f"torch@https://download.pytorch.org/whl/{cuda}/{torch_pkg}-{platform}.whl",
        ]
    })

    return optional_dependencies


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str | list[str] | dict[str, list[str]]:
    if settings:
        raise ValueError("No inline configuration is supported")
    if field == "name":
        return f"aimet-{get_aimet_variant()}"
    if field == "dependencies":
        return get_aimet_dependencies()
    if field == "optional-dependencies":
        return optional_dependencies()
    if field == "version":
        return get_version()
    raise ValueError(f"Unsupported field '{field}'")
