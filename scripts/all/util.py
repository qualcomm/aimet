# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for AIMET build scripts."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def on_linux() -> bool:
    """Check if running on Linux."""
    return platform.uname().system == "Linux"


def on_macos() -> bool:
    """Check if running on macOS."""
    return platform.uname().system == "Darwin"


def on_windows() -> bool:
    """Check if running on Windows."""
    return platform.uname().system == "Windows"


def venv_python_path(venv_path: Path) -> Path:
    """Path to the interpreter inside a venv, independent of this process's own
    interpreter -- needed because tasks run outside any activated venv."""
    if on_windows():
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def uv_env(venv_path: Path) -> dict[str, str]:
    """Environment with VIRTUAL_ENV pointed at ``venv_path``, the same way
    ``setup-uv-env`` does in CI: uv's pip subcommands pick up VIRTUAL_ENV
    automatically, so every ``uv pip ...`` call targets this venv without
    needing an explicit ``--python`` on each invocation."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    bin_dir = venv_path / ("Scripts" if on_windows() else "bin")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    if on_windows() or on_macos():
        # setup-uv-env's own `uv venv`/`uv pip` calls on these runners pass
        # --native-tls to read certs from the platform store (Windows) or
        # Keychain (macOS); match that here so uv calls made by this script
        # don't hit TLS failures the composite action's calls wouldn't.
        env["UV_NATIVE_TLS"] = "1"
    return env


def get_repo_root() -> Path:
    """Repo root, from this file's known location -- not `git`, which may not
    be installed yet when this module is first imported."""
    return Path(__file__).resolve().parents[2]


def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package_name}"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_aimet_torch_installed() -> bool:
    """Check if AIMET Torch variant is installed."""
    return is_package_installed("aimet_torch")


def is_aimet_onnx_installed() -> bool:
    """Check if AIMET ONNX variant is installed."""
    return is_package_installed("aimet_onnx")


def get_cuda_version() -> str:
    """Get CUDA version from nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(",")[0]
                return version.replace(".", "")
    except Exception:
        pass
    return ""


def get_torch_index_url(enable_cuda: bool) -> str:
    """Get PyTorch extra-index-url based on CUDA setting."""
    if enable_cuda:
        cuda_version = get_cuda_version()
        if cuda_version:
            return f"https://download.pytorch.org/whl/cu{cuda_version}"
        return "https://download.pytorch.org/whl/cu124"  # Default CUDA version
    return "https://download.pytorch.org/whl/cpu"


def torch_index_url_for_cuda_version(ver_cuda: str) -> str:
    """PyTorch extra-index-url for a dotted CUDA toolkit version string
    (e.g. "13.2.0" -> cu132, "" -> cpu). Uses the CI-supplied version, not
    whatever `nvcc` reports on the runner."""
    parts = ver_cuda.split(".")
    if not parts or not parts[0]:
        return "https://download.pytorch.org/whl/cpu"
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else ""
    return f"https://download.pytorch.org/whl/cu{major}{minor}"


def are_ubuntu_deps_installed() -> bool:
    """Check if required Ubuntu dependencies are already installed."""
    if not on_linux():
        return True  # Skip check on non-Linux
    required_packages = ["cmake", "clang", "patchelf", "libeigen3-dev", "pandoc"]
    try:
        result = subprocess.run(
            ["dpkg", "-s"] + required_packages,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        # Also check for uv (installed via curl, not apt)
        if not shutil.which("uv"):
            return False
        return True
    except Exception:
        return False


def are_macos_deps_installed() -> bool:
    """Check if required macOS dependencies are already installed."""
    if not on_macos():
        return True  # Skip check on non-macOS
    required_packages = ["cmake", "eigen", "pandoc", "pkg-config", "uv"]
    try:
        result = subprocess.run(
            ["brew", "list"] + required_packages,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def pytest_cuda_marker_args(enable_cuda: bool) -> list[str]:
    """Pytest marker args that select all tests on GPU runners, or skip
    cuda-marked tests on CPU runners."""
    return [] if enable_cuda else ["-m", "not cuda"]


def ld_library_path_with_libpython() -> str:
    """LD_LIBRARY_PATH value with the active interpreter's libdir prepended,
    needed for dynamic linking against libpython3.so on Linux/macOS."""
    import sysconfig

    lib_dir = sysconfig.get_config_var("LIBDIR") or ""
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{lib_dir}:{existing}" if existing else lib_dir
