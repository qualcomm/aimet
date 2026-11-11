# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import os
import pathlib
import re
import shlex
import subprocess
import sys

__all__ = ["dynamic_metadata"]
_PKG_ROOT = (
    pathlib.Path(os.path.dirname(__file__), "..", "..", "..").absolute().resolve()
)


def __dir__() -> list[str]:
    return __all__


def is_cmake_option_enabled(option_name: str) -> bool:
    """Returns True if CMAKE_ARGS environment variable contains `-D{option_name}=ON/YES/1/TRUE` and False otherwise."""
    cmake_args = {
        k: v
        for k, v in (
            arg.split("=", 1) for arg in shlex.split(os.environ.get("CMAKE_ARGS", ""))
        )
    }
    return not cmake_args.get(f"-D{option_name}", "").upper() in {
        "OFF",
        "NO",
        "FALSE",
        "0",
        "N",
    }


def is_target_windows_arm64() -> bool:
    """Returns True if running on Windows with ARM64 target platform.

    Checks CMAKE_ARGS for generator platform/system processor, or detects native ARM64.
    Matches CMake logic: WIN32 AND (CMAKE_GENERATOR_PLATFORM == "ARM64" OR CMAKE_SYSTEM_PROCESSOR matches "ARM64|aarch64")
    """
    import platform

    is_windows = sys.platform == "win32"

    # Check CMAKE_ARGS for -DCMAKE_GENERATOR_PLATFORM and -DCMAKE_SYSTEM_PROCESSOR
    cmake_args = {
        k: v
        for k, v in (
            arg.split("=", 1) for arg in shlex.split(os.environ.get("CMAKE_ARGS", ""))
        )
    }
    cmake_generator_platform = cmake_args.get("-DCMAKE_GENERATOR_PLATFORM", "").upper()
    cmake_system_processor = cmake_args.get("-DCMAKE_SYSTEM_PROCESSOR", "").upper()

    # Check if ARM64 is specified in CMAKE_ARGS or if running natively on ARM64
    native_machine = platform.machine().upper()
    is_arm64 = (
        cmake_generator_platform == "ARM64"
        or cmake_system_processor in ("ARM64", "AARCH64")
        or native_machine in ("ARM64", "AARCH64")
    )

    return is_windows and is_arm64


def is_pip_index_pypi() -> bool:
    """Returns True if CMAKE_ARGS environment variable contains `-DPIP_INDEX=pypi`"""
    cmake_args = {
        k: v
        for k, v in (
            arg.split("=", 1) for arg in shlex.split(os.environ.get("CMAKE_ARGS", ""))
        )
    }
    pip_index_pypi = False
    if cmake_args.get("-DPIP_INDEX", "") == "pypi":
        pip_index_pypi = True

    return pip_index_pypi


def get_aimet_variant() -> str:
    """Return a variant based on CMAKE_ARGS environment variable"""
    enable_cuda = is_cmake_option_enabled("ENABLE_CUDA")
    enable_torch = is_cmake_option_enabled("ENABLE_TORCH")
    enable_onnx = is_cmake_option_enabled("ENABLE_ONNX")

    enabled_variants = [enable_torch, enable_onnx]
    enabled_count = sum(enabled_variants)

    if enabled_count == 2:
        variant = "onnx-torch-"
    elif enabled_count == 1:
        if enable_torch:
            variant = "torch-"
        elif enable_onnx:
            variant = "onnx-"
    else:
        raise RuntimeError(
            "\n".join(
                [
                    "Only one or all of ENABLE_{TORCH, ONNX} should set to ON."
                    "Your passed:"
                    f"  * ENABLE_TORCH:      {'ON' if enable_torch else 'OFF'}",
                    f"  * ENABLE_ONNX:       {'ON' if enable_onnx else 'OFF'}",
                ]
            )
        )

    if enable_cuda:
        variant += "gpu"
    elif is_target_windows_arm64():
        variant += "qnn"
    else:
        variant += "cpu"
    return variant


def get_name() -> str:
    aimet_variant = get_aimet_variant()

    # List of suffixes to remove
    suffixes = ["-cpu", "-gpu", "-qnn"]

    # Remove suffix from the aimet_variant
    for suffix in suffixes:
        if aimet_variant.endswith(suffix):
            aimet_variant = aimet_variant.replace(suffix, "")

    return f"aimet-{aimet_variant}"


def get_aimet_dependencies() -> list[str]:
    """Read dependencies form the corresponded files and return them as a list (!) of strings"""
    aimet_variant = get_aimet_variant()
    base_path = pathlib.Path(_PKG_ROOT, "packaging", "dependencies")
    deps_path = pathlib.Path(base_path, "fast-release", aimet_variant)

    deps_files = [*deps_path.glob("reqs_pip_*.txt")]
    print(f"CMAKE_ARGS='{os.environ.get('CMAKE_ARGS', '')}'")
    print(
        f"Read dependencies for variant '{get_aimet_variant()}' from the following files: {deps_files}"
    )
    deps = {
        d
        for d in itertools.chain.from_iterable(
            line.replace(" -f ", "\n-f ").split("\n")
            for f in deps_files
            for line in f.read_text(encoding="utf8").splitlines()
        )
        if not d.startswith(("#", "-f"))
    }

    deps = list(sorted(deps))

    # aimet-onnx released Github depends on onnxruntime-gpu,
    # while the same package on PyPI temporarily depends on onnxruntime (cpu).
    # This conditional logic will be removed once onnxruntime-gpu
    # becomes a valid dependency for the aimet-onnx PyPI distribution.
    if aimet_variant == "onnx-gpu" and not is_pip_index_pypi():
        # Replace onnxruntime with onnxruntime-gpu
        deps = [
            re.sub(r"(onnxruntime)([<>=].+)?", r"onnxruntime-gpu\2", pkg)
            for pkg in deps
        ]

    return deps


def get_cuda_version():
    cuda_version = ""
    try:
        # Run the nvcc command to get CUDA version
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        # Extract the version number from the output
        for line in output.split("\n"):
            if "release" in line:
                cuda_version = line.split("release")[-1].strip().split(",")[0]
                # Remove the decimal point
                cuda_version = cuda_version.replace(".", "")
    except Exception:
        pass

    return cuda_version


def get_version() -> str:
    version = (
        pathlib.Path(_PKG_ROOT, "packaging", "version.txt")
        .read_text(encoding="utf8")
        .splitlines()[0]
    )

    # For PyPi releases, just return the version without appending the variant string
    if is_pip_index_pypi():
        return version

    # Only use CUDA version if ENABLE_CUDA is set in CMAKE_ARGS
    enable_cuda = is_cmake_option_enabled("ENABLE_CUDA")
    variant_string = "cpu"
    if enable_cuda:
        cuda_version = get_cuda_version()
        variant_string = f"cu{cuda_version}" if cuda_version else "cu"

    version = version + "+" + variant_string
    return version


def _read_requirements_file(filename: str) -> list[str]:
    """Read dependencies from a requirements file, filtering comments and empty lines."""
    filepath = pathlib.Path(_PKG_ROOT, "packaging", "dependencies", filename)
    if not filepath.exists():
        return []
    deps = []
    for line in filepath.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            deps.append(line)
    return deps


def optional_dependencies() -> dict[str, list[str]]:
    # Read test dependencies from requirements file
    test_deps = _read_requirements_file("reqs_pip_test.txt")

    optional_deps = {
        "dev": [
            # duplicate build-system.requires for editable mode (non-isolated)
            "scikit-build-core[wheels]==0.11.1",
            "build",
            "auditwheel; sys_platform == 'linux'",  # Linux only (ELF-based)
            # auditwheel>=6.8.1 requires patchelf>=0.14.5, but ubuntu jammy only
            # ships 0.14.3, so take patchelf from pypi instead of apt.
            "patchelf==0.19.1.0; sys_platform == 'linux'",
            "cython>=3.0",
            # and the rest
        ],
        "test": test_deps,
        "docs": [
            "sphinx-lumina-theme",
            "nbsphinx",
            "pandoc",
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "sphinx-design",
            "sphinx-jinja",
            "sphinx-rtd-theme",
            "sphinx-substitution-extensions",
            "sphinx-tabs",
        ],
    }

    aimet_variant = get_aimet_variant()

    if aimet_variant in ("onnx-qnn",):
        optional_deps["test"].extend(
            [
                "onnxruntime-qnn",
            ]
        )

    if aimet_variant in ("torch-gpu", "torch-cpu"):
        # Read torch-specific test deps (deepspeed, spconv - Linux only)
        torch_test_deps = _read_requirements_file("reqs_pip_test_torch.txt")
        optional_deps["test"].extend(torch_test_deps)

    return optional_deps


def get_description() -> str:
    aimet_variant = get_aimet_variant()

    variant_map = {
        "torch": "AIMET torch package",
        "onnx": "AIMET onnx package",
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
