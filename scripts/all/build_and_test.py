#!/usr/bin/env python3

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Build and test script for AIMET (AI Model Efficiency Toolkit).

This script provides a task-based build system with dependency management.

Usage:
    python scripts/all/build_and_test.py --task <task_name> [options]

Examples:
    # Build AIMET Torch (CPU)
    python scripts/all/build_and_test.py --task build --package aimet_torch

    # Build AIMET Torch (GPU/CUDA)
    python scripts/all/build_and_test.py --task build --package aimet_torch --gpu

    # Build AIMET ONNX (CPU)
    python scripts/all/build_and_test.py --task build --package aimet_onnx

    # Build documentation
    python scripts/all/build_and_test.py --task build --package docs

    # Run tests (skips build if already installed)
    python scripts/all/build_and_test.py --task test --package aimet_torch

    # Run tests with CUDA
    python scripts/all/build_and_test.py --task test --package aimet_torch --gpu

    # Force rebuild even if already installed
    python scripts/all/build_and_test.py --task build --package aimet_torch --force-rebuild

    # List available tasks
    python scripts/all/build_and_test.py --task list
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from util import (
    are_macos_deps_installed,
    are_ubuntu_deps_installed,
    get_repo_root,
    get_torch_index_url,
    is_aimet_onnx_installed,
    is_aimet_torch_installed,
    ld_library_path_with_libpython,
    on_linux,
    on_macos,
    on_windows,
    pytest_cuda_marker_args,
    torch_index_url_for_cuda_version,
    uv_env,
    venv_python_path,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_VENV_PATH = REPO_ROOT / "aimet-dev"
DEFAULT_PYTHON = "python3"

# Valid package choices
VALID_PACKAGES = ["aimet_torch", "aimet_onnx", "docs"]

# Global registries for tasks and dependencies
ALL_TASKS: dict[str, str] = {}
PUBLIC_TASKS: dict[str, str] = {}
TASK_DEPENDENCIES: dict[str, list[str]] = {}


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""

    status: TaskStatus
    message: str = ""
    duration_seconds: float = 0.0


class Task(ABC):
    """Base class for all tasks."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    @abstractmethod
    def run(self) -> TaskResult:
        """Execute the task."""
        pass

    def does_work(self) -> bool:
        """Return True if this task does actual work (vs being a no-op)."""
        return True


class BashScriptTask(Task):
    """Task that runs a bash script."""

    def __init__(
        self,
        name: str,
        script_path: str | Path,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.script_path = Path(script_path)
        self.args = args or []
        self.env = env
        self.cwd = cwd or REPO_ROOT

    def run(self) -> TaskResult:
        cmd = [str(self.script_path)] + self.args
        logger.info(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Script failed with exit code {e.returncode}"
            )


class CommandTask(Task):
    """Task that runs a shell command."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.cmd = cmd
        self.env = env
        self.cwd = cwd or REPO_ROOT

    def run(self) -> TaskResult:
        logger.info(f"Running: {' '.join(self.cmd)}")

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                self.cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Command failed with exit code {e.returncode}"
            )
        except FileNotFoundError as e:
            return TaskResult(TaskStatus.FAILED, f"Command not found: {e}")


def _ensure_uv_installed() -> TaskResult | None:
    """Ensure uv is installed. Returns TaskResult on failure, None on success."""
    if shutil.which("uv"):
        return None
    return TaskResult(
        TaskStatus.FAILED,
        "uv not found. Run: python scripts/all/build_and_test.py --task install_deps",
    )


class WheelBuildTask(Task):
    """Base task that builds a wheel using `python -m build`: compiles and
    installs dev deps into --venv, then builds. Platform-specific pieces
    -- extra CMake args/env, and what to do with the wheel(s) `python -m
    build` drops into dist/ -- are implemented by the LinuxWheelBuildTask /
    WindowsWheelBuildTask / MacOSWheelBuildTask subclasses below.
    """

    def __init__(
        self,
        name: str,
        venv_path: Path,
        cwd: Path | None = None,
        skip_if_installed: Callable[[], bool] | None = None,
        force_rebuild: bool = False,
        enable_torch: bool = False,
        enable_onnx: bool = False,
        enable_cuda: bool = False,
        enable_docs: bool = False,
        enable_tests: bool = False,
        editable: bool = False,
        pip_index: str = "",
        ver_torch: str = "",
        ver_onnxruntime: str = "",
    ) -> None:
        super().__init__(name)
        self.venv_path = venv_path
        self.cwd = cwd or REPO_ROOT
        self.skip_if_installed = skip_if_installed
        self.force_rebuild = force_rebuild
        self.enable_torch = enable_torch
        self.enable_onnx = enable_onnx
        self.enable_cuda = enable_cuda
        self.enable_docs = enable_docs
        self.enable_tests = enable_tests
        self.editable = editable
        self.pip_index = pip_index
        self.ver_torch = ver_torch
        self.ver_onnxruntime = ver_onnxruntime

        # Build cmake_args from flags
        cuda = "ON" if enable_cuda else "OFF"
        torch = "ON" if enable_torch else "OFF"
        onnx = "ON" if enable_onnx else "OFF"
        tests = "ON" if enable_tests else "OFF"
        cmake_args = (
            f"-DENABLE_CUDA={cuda} -DENABLE_TORCH={torch} -DENABLE_ONNX={onnx} "
            f"-DENABLE_TESTS={tests}"
        )
        if self.pip_index:
            cmake_args += f" -DPIP_INDEX={self.pip_index}"
        self.cmake_args = cmake_args

        # Leave SKBUILD_BUILD_TARGETS unset (scikit-build-core's default
        # target) unless docs are requested: "all" isn't a real target name
        # under the Visual Studio generator Windows builds use (it's
        # ALL_BUILD there), so forcing "all" breaks those builds.
        self.skbuild_targets = "all;doc" if enable_docs else ""

    def _platform_cmake_args(self) -> TaskResult | str:
        """Extra -D... flags to append to CMAKE_ARGS for this platform.
        Returns a TaskResult on failure (e.g. toolchain setup), else a
        (possibly empty) string. Default: no extra flags."""
        return ""

    def _platform_build_env(self) -> dict[str, str]:
        """Extra environment variables for the `python -m build` subprocess.
        Default: none."""
        return {}

    def _package_built_wheels(self, dist_dir: Path, wheelhouse_dir: Path) -> TaskResult:
        """What to do with the wheel(s) `python -m build` dropped into
        dist_dir. Default: copy as-is (Windows/macOS have no manylinux-repair
        equivalent). Overridden by LinuxWheelBuildTask for manylinux repair."""
        for wheel_file in dist_dir.glob("aimet*.whl"):
            shutil.copy(wheel_file, wheelhouse_dir)
        return TaskResult(TaskStatus.SUCCESS)

    def _retag_windows_arm64(self) -> bool:
        """Whether this build needs the win_amd64 -> win_arm64 wheel retag.
        Only true for WindowsWheelBuildTask cross-compiling ARM64."""
        return False

    def run(self) -> TaskResult:
        # Check if we can skip the build
        if (
            not self.force_rebuild
            and self.skip_if_installed
            and self.skip_if_installed()
        ):
            logger.info(
                "Package already installed, skipping build (use --force-rebuild to override)"
            )
            return TaskResult(TaskStatus.SUCCESS, "Skipped - already installed")

        # Compile and install dev dependencies
        tmp_dir = Path(tempfile.gettempdir())
        dev_requirements_file = tmp_dir / "dev_requirements.txt"
        constraints_file = tmp_dir / "aimet_constraints.txt"
        torch_index_url = get_torch_index_url(self.enable_cuda)

        # Create constraints file (same approach as CI Dockerfile)
        constraints = [f"torch=={self.ver_torch or '2.10.*'}"]
        if self.enable_onnx:
            constraints.append(f"onnxruntime=={self.ver_onnxruntime or '1.23.*'}")
        constraints_file.write_text("\n".join(constraints) + "\n")
        logger.info(f"Created constraints file: {constraints_file}")

        # Ensure uv is installed
        if result := _ensure_uv_installed():
            return result

        # Every uv/python invocation below targets self.venv_path (not this
        # script's own interpreter), the same way setup-build-env/setup-uv-env
        # point uv at the build venv in CI via VIRTUAL_ENV.
        venv_python = str(venv_python_path(self.venv_path))

        # Compile dev dependencies
        logger.info("Compiling dev dependencies from pyproject.toml")
        compile_cmd = [
            "uv",
            "pip",
            "compile",
            "--output-file",
            str(dev_requirements_file),
            "--extra",
            "dev",
            "--constraint",
            str(constraints_file),
            "--extra-index-url",
            torch_index_url,
            "--index-strategy",
            "unsafe-best-match",
            "pyproject.toml",
        ]
        if self.enable_docs:
            compile_cmd.insert(-1, "--extra")
            compile_cmd.insert(-1, "docs")
        # Set CMAKE_ARGS so aimet's dynamic metadata plugin knows which variant to build
        compile_env = uv_env(self.venv_path)
        compile_env["CMAKE_ARGS"] = self.cmake_args
        logger.info(f"Running: CMAKE_ARGS='{self.cmake_args}' {' '.join(compile_cmd)}")
        try:
            subprocess.run(
                compile_cmd,
                cwd=self.cwd,
                check=True,
                capture_output=False,
                env=compile_env,
            )
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED,
                f"uv pip compile failed with exit code {e.returncode}",
            )

        # Install dev dependencies
        if dev_requirements_file.exists():
            logger.info("Installing dev dependencies")
            torch_index_url = get_torch_index_url(self.enable_cuda)
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--index-strategy",
                "unsafe-best-match",
                "--extra-index-url",
                torch_index_url,
                "-r",
                str(dev_requirements_file),
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(
                    install_cmd,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"dev deps install failed with exit code {e.returncode}",
                )

        # CMake's configure step imports onnx/onnxruntime/pybind11 directly
        # (set_onnx_version, set_onnxruntime_variables, add_library_pybind11)
        # to read their version/include paths; the dev/test extras above
        # don't carry these, so install them explicitly before `python -m build`.
        if self.enable_onnx:
            build_time_deps = ["onnx", "build", "pybind11"]
            if self.ver_onnxruntime:
                build_time_deps.append(f"onnxruntime=={self.ver_onnxruntime}")
            else:
                build_time_deps.append("onnxruntime")
            install_cmd = ["uv", "pip", "install", *build_time_deps]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(
                    install_cmd,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"build-time deps install failed with exit code {e.returncode}",
                )

        # Clean previous build artifacts
        for d in [self.cwd / "build", self.cwd / "dist", self.cwd / "wheelhouse"]:
            if d.exists():
                logger.info(f"Removing {d}")
                shutil.rmtree(d)
        (self.cwd / "wheelhouse").mkdir(exist_ok=True)

        cmake_args = self.cmake_args
        platform_cmake_args = self._platform_cmake_args()
        if isinstance(platform_cmake_args, TaskResult):
            return platform_cmake_args
        cmake_args += platform_cmake_args

        env = uv_env(self.venv_path)
        env["CMAKE_ARGS"] = cmake_args
        if self.skbuild_targets:
            env["SKBUILD_BUILD_TARGETS"] = self.skbuild_targets
        env.update(self._platform_build_env())

        # Editable install for local development
        # Note: uv pip doesn't handle editable installs with dynamic package names well,
        # so we use regular pip for editable installs.
        if self.editable:
            # Ensure pip is installed in the venv
            pip_install_cmd = ["uv", "pip", "install", "pip"]
            logger.info(f"Ensuring pip is installed: {' '.join(pip_install_cmd)}")
            subprocess.run(
                pip_install_cmd,
                cwd=self.cwd,
                capture_output=True,
                env=uv_env(self.venv_path),
            )

            # Uninstall any existing aimet packages to avoid conflicts with editable install
            for pkg in ["aimet-torch", "aimet-onnx", "aimet-onnx-torch"]:
                uninstall_cmd = [venv_python, "-m", "pip", "uninstall", "-y", pkg]
                logger.info(f"Uninstalling existing package: {' '.join(uninstall_cmd)}")
                subprocess.run(uninstall_cmd, cwd=self.cwd, capture_output=True)

            extras = ["dev", "test"]
            if self.enable_docs:
                extras.append("docs")
            extras_str = ",".join(extras)
            cmd = [
                venv_python,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-e",
                f".[{extras_str}]",
            ]
            logger.info(f"Running: CMAKE_ARGS='{cmake_args}' {' '.join(cmd)}")
            try:
                subprocess.run(
                    cmd,
                    env=env,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                )
                return TaskResult(TaskStatus.SUCCESS)
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"Editable install failed with exit code {e.returncode}",
                )

        # Build wheel
        cmd = [venv_python, "-m", "build", "--wheel", "--no-isolation", "."]

        logger.info(f"Running: CMAKE_ARGS='{cmake_args}' {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Build failed with exit code {e.returncode}"
            )

        # Package the wheel(s) `python -m build` dropped into dist/
        # (manylinux repair on Linux, plain copy on Windows/macOS).
        dist_dir = self.cwd / "dist"
        wheelhouse_dir = self.cwd / "wheelhouse"
        wheelhouse_dir.mkdir(exist_ok=True)

        if list(dist_dir.glob("aimet*.whl")):
            package_result = self._package_built_wheels(dist_dir, wheelhouse_dir)
            if package_result.status == TaskStatus.FAILED:
                return package_result

        if self.enable_torch or self._retag_windows_arm64():
            retag_result = WheelRetagTask(
                "Re-tagging built wheels",
                venv_path=self.venv_path,
                wheelhouse_dir=wheelhouse_dir,
                retag_torch=self.enable_torch,
                retag_windows_arm64=self._retag_windows_arm64(),
            ).run()
            if retag_result.status == TaskStatus.FAILED:
                return retag_result

        # Docs builds package the sphinx output (built via the "doc"
        # SKBUILD_BUILD_TARGETS above) inside the wheel; extract it into
        # wheelhouse/ alongside the wheel itself so it can be uploaded as a
        # directory of files (see Docs/CMakeLists.txt's `doc` target).
        if self.enable_docs:
            extract_cmd = [
                "uv",
                "pip",
                "install",
                "-t",
                str(wheelhouse_dir),
                "--no-deps",
                *(str(p) for p in wheelhouse_dir.glob("aimet*.whl")),
            ]
            logger.info(f"Running: {' '.join(extract_cmd)}")
            try:
                subprocess.run(
                    extract_cmd,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"docs package extraction failed with exit code {e.returncode}",
                )

        return TaskResult(TaskStatus.SUCCESS)


_SYSCONFIG_LIBDIR_EXPR = (
    "import sysconfig; from pathlib import Path; "
    "print(Path(sysconfig.get_config_var('prefix'), 'lib'))"
)


def _sonames_in(dir_path: Path, patterns: list[str]) -> list[str]:
    """auditwheel --exclude args for every soname matching patterns in dir_path."""
    excludes: list[str] = []
    for pattern in patterns:
        for so_file in dir_path.glob(pattern):
            try:
                result = subprocess.run(
                    ["patchelf", "--print-soname", str(so_file)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    excludes.extend(["--exclude", result.stdout.strip()])
            except Exception:
                pass
    return excludes


def _excludes_for_python_expr(
    venv_python: str, py_expr: str, patterns: list[str], label: str
) -> list[str]:
    """Evaluate py_expr in the venv to get a directory, then _sonames_in it."""
    try:
        result = subprocess.run(
            [venv_python, "-c", py_expr], capture_output=True, text=True, check=True
        )
        dir_path = Path(result.stdout.strip())
        if dir_path.exists():
            return _sonames_in(dir_path, patterns)
    except Exception as e:
        logger.warning(f"Could not get {label} excludes: {e}")
    return []


class LinuxWheelBuildTask(WheelBuildTask):
    """Linux wheel build: repairs aimet_onnx wheels for manylinux
    compatibility (it carries compiled shared libraries); aimet_torch
    alone has no compiled extension, so its wheel is just copied as-is."""

    def _get_manylinux_exclude_libs(self) -> list[str]:
        """Libraries already provided by the target system/venv that
        auditwheel shouldn't vendor into the manylinux-repaired wheel."""
        venv_python = str(venv_python_path(self.venv_path))
        exclude_libs: list[str] = []

        if self.enable_torch:
            exclude_libs += _excludes_for_python_expr(
                venv_python,
                "import torch; print(f'{torch.utils.cmake_prefix_path}/../../lib')",
                ["*.so*"],
                "Torch",
            )
        if self.enable_cuda:
            exclude_libs += _excludes_for_python_expr(
                venv_python,
                _SYSCONFIG_LIBDIR_EXPR,
                ["libcu*.so*", "libnv*.so*", "libnp*.so*"],
                "CUDA",
            )
        exclude_libs += _excludes_for_python_expr(
            venv_python, _SYSCONFIG_LIBDIR_EXPR, ["libpython*.so*"], "libpython"
        )
        return exclude_libs

    def _package_built_wheels(self, dist_dir: Path, wheelhouse_dir: Path) -> TaskResult:
        wheel_files = list(dist_dir.glob("aimet*.whl"))
        if self.enable_onnx:
            exclude_libs = self._get_manylinux_exclude_libs()
            for wheel_file in wheel_files:
                auditwheel_cmd = [
                    "auditwheel",
                    "repair",
                    "--plat",
                    "manylinux_2_34_x86_64",
                    *exclude_libs,
                    "-w",
                    str(wheelhouse_dir),
                    str(wheel_file),
                ]
                logger.info(f"Running: {' '.join(auditwheel_cmd)}")
                try:
                    subprocess.run(
                        auditwheel_cmd,
                        check=True,
                        capture_output=False,
                        env=uv_env(self.venv_path),
                    )
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"auditwheel repair failed with exit code {e.returncode}",
                    )
                except FileNotFoundError:
                    # auditwheel ships via the `dev` extra; missing means a
                    # venv problem, not something to silently skip.
                    return TaskResult(
                        TaskStatus.FAILED,
                        "auditwheel not found in venv; manylinux repair skipped",
                    )
        else:
            # aimet_torch alone has no compiled extension to repair.
            logger.info("Skipping auditwheel, copying wheels to wheelhouse")
            for wheel_file in wheel_files:
                shutil.copy(wheel_file, wheelhouse_dir)
        return TaskResult(TaskStatus.SUCCESS)


class WindowsWheelBuildTask(WheelBuildTask):
    """Windows wheel build: sets up the vcpkg/Eigen3 CMake toolchain and
    (optionally) cross-compiles for ARM64 using the pythonarm64 nuget
    package, since scikit-build-core tags the result win_amd64 under the
    MSVC ARM64 cross-compiler (host arch amd64) -- re-tag it to win_arm64."""

    def __init__(
        self,
        *args,
        windows_arch: str = "",
        windows_vcpkg_triplet: str = "",
        windows_python_version: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.windows_arch = windows_arch
        self.windows_vcpkg_triplet = windows_vcpkg_triplet
        self.windows_python_version = windows_python_version

    def _setup_windows_vcpkg_eigen(self) -> TaskResult | str:
        """Install vcpkg (if not already present) and Eigen3 for the given
        triplet, returning the vcpkg CMake toolchain file path."""
        vcpkg_root = self.cwd / "vcpkg"
        vcpkg_exe = vcpkg_root / "vcpkg.exe"
        if not vcpkg_exe.exists():
            clone_cmd = [
                "git",
                "clone",
                "https://github.com/microsoft/vcpkg",
                str(vcpkg_root),
            ]
            logger.info(f"Running: {' '.join(clone_cmd)}")
            try:
                subprocess.run(clone_cmd, check=True, capture_output=False)
                subprocess.run(
                    [str(vcpkg_root / "bootstrap-vcpkg.bat"), "-disableMetrics"],
                    check=True,
                    capture_output=False,
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"vcpkg bootstrap failed with exit code {e.returncode}",
                )

        install_cmd = [
            str(vcpkg_exe),
            "install",
            f"eigen3:{self.windows_vcpkg_triplet}",
        ]
        logger.info(f"Running: {' '.join(install_cmd)}")
        try:
            subprocess.run(install_cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED,
                f"vcpkg install eigen3 failed with exit code {e.returncode}",
            )

        toolchain = vcpkg_root / "scripts" / "buildsystems" / "vcpkg.cmake"
        return str(toolchain).replace("\\", "/")

    def _download_windows_arm64_python_libs(self) -> TaskResult | tuple[str, str]:
        """Download the pythonarm64 nuget package for Windows ARM64
        cross-compilation, returning (libs_dir, include_dir) with CMake-safe
        forward slashes."""
        import urllib.request
        import zipfile

        python_arm64_dir = self.cwd / "python-arm64"
        python_arm64_dir.mkdir(parents=True, exist_ok=True)
        nuget_zip = python_arm64_dir / "python.zip"

        url = f"https://www.nuget.org/api/v2/package/pythonarm64/{self.windows_python_version}"
        logger.info(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, nuget_zip)
            with zipfile.ZipFile(nuget_zip) as zf:
                zf.extractall(python_arm64_dir)
        except Exception as e:
            return TaskResult(
                TaskStatus.FAILED, f"ARM64 Python nuget download/extract failed: {e}"
            )

        libs_dir = str(python_arm64_dir / "tools" / "libs").replace("\\", "/")
        include_dir = str(python_arm64_dir / "tools" / "include").replace("\\", "/")
        return libs_dir, include_dir

    def _platform_cmake_args(self) -> TaskResult | str:
        if not self.windows_arch:
            return ""

        cmake_args = ""
        toolchain = self._setup_windows_vcpkg_eigen()
        if isinstance(toolchain, TaskResult):
            return toolchain
        cmake_args += f" -DCMAKE_TOOLCHAIN_FILE={toolchain}"

        if self.windows_arch == "arm64":
            arm64_libs = self._download_windows_arm64_python_libs()
            if isinstance(arm64_libs, TaskResult):
                return arm64_libs
            libs_dir, include_dir = arm64_libs
            py_lib_name = f"python{self.windows_python_version.replace('.', '')}.lib"
            cmake_args += (
                f" -DPython3_LIBRARY={libs_dir}/{py_lib_name}"
                f" -DPython3_INCLUDE_DIR={include_dir}"
            )

        cmake_args += (
            " -DCMAKE_GENERATOR='Visual Studio 17 2022'"
            f" -DCMAKE_GENERATOR_PLATFORM='{self.windows_arch}'"
        )
        return cmake_args

    def _retag_windows_arm64(self) -> bool:
        return self.windows_arch == "arm64"


class MacOSWheelBuildTask(WheelBuildTask):
    """macOS wheel build: pins MACOSX_DEPLOYMENT_TARGET (both as a CMake
    flag and an env var, since wheel platform-tagging reads it via
    sysconfig) lower than the runner's live macOS version, so the wheel
    isn't tagged with a macOS version too new for other machines to
    install."""

    def __init__(self, *args, macos_deployment_target: str = "15.0", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.macos_deployment_target = macos_deployment_target

    def _platform_cmake_args(self) -> TaskResult | str:
        return f" -DCMAKE_OSX_DEPLOYMENT_TARGET={self.macos_deployment_target}"

    def _platform_build_env(self) -> dict[str, str]:
        return {"MACOSX_DEPLOYMENT_TARGET": self.macos_deployment_target}


class WheelRetagTask(Task):
    """Task that re-tags wheels already built into --wheel-dir (default:
    wheelhouse/): with retag_torch, strips the wheel's python/abi tags to
    py3-none-any -- only valid for a pure-Python wheel with no compiled
    extensions (aimet-torch; also aimet-onnx when bundled into a docs
    build), so this makes one wheel install across Python versions. With
    retag_windows_arm64, re-tags a Windows ARM64 cross-compiled wheel from
    win_amd64 (what scikit-build-core tags it, since the MSVC ARM64
    cross-compiler host arch is amd64) to win_arm64."""

    def __init__(
        self,
        name: str,
        venv_path: Path,
        wheelhouse_dir: Path,
        retag_torch: bool = False,
        retag_windows_arm64: bool = False,
    ) -> None:
        super().__init__(name)
        self.venv_path = venv_path
        self.wheelhouse_dir = wheelhouse_dir
        self.retag_torch = retag_torch
        self.retag_windows_arm64 = retag_windows_arm64

    def run(self) -> TaskResult:
        venv_python = str(venv_python_path(self.venv_path))

        if self.retag_windows_arm64:
            # The ARM64 cross-compile venv may not carry dev extras (which
            # is where `wheel` normally comes from for the torch retag
            # below), so install it explicitly before re-tagging.
            pip_install_wheel_cmd = ["uv", "pip", "install", "wheel"]
            logger.info(f"Running: {' '.join(pip_install_wheel_cmd)}")
            try:
                subprocess.run(
                    pip_install_wheel_cmd,
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"wheel package install failed with exit code {e.returncode}",
                )

        if self.retag_torch:
            for wheel_file in self.wheelhouse_dir.glob("aimet*.whl"):
                retag_cmd = [
                    venv_python,
                    "-m",
                    "wheel",
                    "tags",
                    "--remove",
                    "--python-tag=py310",
                    "--abi-tag=none",
                    "--platform-tag=any",
                    str(wheel_file),
                ]
                logger.info(f"Running: {' '.join(retag_cmd)}")
                try:
                    subprocess.run(retag_cmd, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"wheel tags failed: {e}")
                except FileNotFoundError:
                    logger.warning("wheel module not found, skipping re-tagging")

        # Re-tag Windows ARM64 wheels: scikit-build-core tags them
        # win_amd64 (the host arch under the MSVC ARM64 cross-compiler),
        # so re-tag to win_arm64 and drop the original amd64-tagged wheel.
        if self.retag_windows_arm64:
            for wheel_file in self.wheelhouse_dir.glob("aimet*win_amd64.whl"):
                retag_cmd = [
                    venv_python,
                    "-m",
                    "wheel",
                    "tags",
                    "--remove",
                    "--platform-tag=win_arm64",
                    str(wheel_file),
                ]
                logger.info(f"Running: {' '.join(retag_cmd)}")
                try:
                    subprocess.run(retag_cmd, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED, f"ARM64 wheel retag failed: {e}"
                    )

        return TaskResult(TaskStatus.SUCCESS)


class PyTestTask(Task):
    """Task that installs wheel with [test] extras and runs pytest."""

    def __init__(
        self,
        name: str,
        venv_path: Path,
        test_paths: list[str | Path],
        markers: str | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        skip_install_if: Callable[[], bool] | None = None,
        parallel: int = 8,
        enable_cuda: bool = False,
    ) -> None:
        super().__init__(name)
        self.venv_path = venv_path
        self.test_paths = [str(p) for p in test_paths]
        self.markers = markers
        self.extra_args = extra_args or []
        self.env = env
        self.cwd = cwd or REPO_ROOT
        self.skip_install_if = skip_install_if
        self.parallel = parallel
        self.enable_cuda = enable_cuda

    def run(self) -> TaskResult:
        # Ensure uv is installed
        if result := _ensure_uv_installed():
            return result

        # Check if we can skip wheel installation
        if self.skip_install_if and self.skip_install_if():
            logger.info("Package already installed, skipping wheel installation")
        else:
            # Find the wheel in wheelhouse/
            wheelhouse_dir = self.cwd / "wheelhouse"
            wheel_files = list(wheelhouse_dir.glob("aimet*.whl"))

            if not wheel_files:
                # Fallback to dist/ if wheelhouse is empty
                dist_dir = self.cwd / "dist"
                wheel_files = list(dist_dir.glob("aimet*.whl"))

            if not wheel_files:
                return TaskResult(
                    TaskStatus.FAILED, "No wheel file found in wheelhouse/ or dist/"
                )

            wheel_path = wheel_files[0]
            logger.info(f"Found wheel: {wheel_path}")

            # Install wheel with [test] extras
            torch_index_url = get_torch_index_url(self.enable_cuda)
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--extra-index-url",
                torch_index_url,
                "--index-strategy",
                "unsafe-best-match",
                f"{wheel_path}[test]",
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(
                    install_cmd,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"uv pip install failed with exit code {e.returncode}",
                )

        # Run pytest
        cmd = [str(venv_python_path(self.venv_path)), "-m", "pytest"]

        if self.parallel > 0:
            cmd.extend(["-n", str(self.parallel)])

        if self.markers:
            cmd.extend(["-m", self.markers])

        cmd.extend(self.extra_args)
        cmd.extend(self.test_paths)

        logger.info(f"Running: {' '.join(cmd)}")

        env = uv_env(self.venv_path)
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"pytest failed with exit code {e.returncode}"
            )


class _CiTaskBase(Task):
    """Shared helpers for CI tasks that install a pre-built wheel
    (downloaded as a CI artifact, not built from source) and run pytest
    against it: wheel discovery/install, and a pytest runner with the right
    markers/LD_LIBRARY_PATH.
    """

    def __init__(
        self,
        name: str,
        venv_path: Path,
        wheel_dir: Path,
        framework: str,
        ver_cuda: str = "",
        parallel: int = 8,
    ) -> None:
        super().__init__(name)
        self.venv_path = venv_path
        self.wheel_dir = wheel_dir
        self.framework = framework
        self.ver_cuda = ver_cuda
        self.parallel = parallel

    def _install_test_deps(self, install_all_wheels: bool) -> TaskResult:
        if install_all_wheels:
            # Acceptance tests exercise both packages together, so install
            # whatever wheels were downloaded (torch and onnx).
            wheel_paths = sorted(self.wheel_dir.glob("aimet_*.whl"))
        else:
            wheel_paths = sorted(self.wheel_dir.glob(f"aimet_{self.framework}*.whl"))[
                :1
            ]

        if not wheel_paths:
            return TaskResult(
                TaskStatus.FAILED, f"No aimet wheel found in {self.wheel_dir}"
            )

        torch_index_url = torch_index_url_for_cuda_version(self.ver_cuda)
        env = uv_env(self.venv_path)
        for wheel_path in wheel_paths:
            install_cmd = [
                "uv",
                "pip",
                "install",
                f"{wheel_path}[test]",
                "--extra-index-url",
                torch_index_url,
                "--index-strategy",
                "unsafe-best-match",
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(install_cmd, check=True, capture_output=False, env=env)
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"uv pip install failed with exit code {e.returncode}",
                )

        # A PIP_INDEX=pypi aimet-onnx wheel depends on onnxruntime (CPU), not
        # onnxruntime-gpu (see packaging/plugins/local/aimet.py), so CUDA legs
        # would otherwise run every ORT session on the CPU EP. Reinstalling the GPU
        # build
        if self.framework == "onnx" and self.ver_cuda:
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--reinstall",
                "--no-deps",
                "onnxruntime-gpu",
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(install_cmd, check=True, capture_output=False, env=env)
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"onnxruntime-gpu install failed with exit code {e.returncode}",
                )

        return TaskResult(TaskStatus.SUCCESS)

    def _pytest(self, test_paths: list[Path], extra_args: list[str]) -> TaskResult:
        cmd = [str(venv_python_path(self.venv_path)), "-m", "pytest"]
        if self.parallel > 0:
            cmd.extend(["-n", str(self.parallel)])
        cmd.extend(pytest_cuda_marker_args(enable_cuda=bool(self.ver_cuda)))
        cmd.extend(extra_args)
        cmd.extend(str(p) for p in test_paths)

        logger.info(f"Running: {' '.join(cmd)}")
        env = uv_env(self.venv_path)
        env["LD_LIBRARY_PATH"] = ld_library_path_with_libpython()
        try:
            subprocess.run(
                cmd, check=True, capture_output=False, env=env, cwd=REPO_ROOT
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"pytest failed with exit code {e.returncode}"
            )


class InstallTestDepsTask(_CiTaskBase):
    """Task that installs a pre-built wheel with [test] extras, without
    running any tests. Used standalone before steps (e.g. pylint) that only
    need the package importable, and as the shared dependency of
    UnitTestTask/AcceptanceTestTask."""

    def run(self) -> TaskResult:
        if result := _ensure_uv_installed():
            return result
        return self._install_test_deps(install_all_wheels=False)


class UnitTestTask(_CiTaskBase):
    """Task that installs a pre-built wheel and runs the unit test suite
    (C++ test binaries + the framework's pytest tree)."""

    def _run_cpp_test_binaries(self) -> TaskResult:
        for target in self.wheel_dir.rglob("*Test"):
            if not target.is_file():
                continue
            target.chmod(target.stat().st_mode | 0o111)
            logger.info(f"Running: {target}")
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ld_library_path_with_libpython()
            try:
                subprocess.run([str(target)], check=True, capture_output=False, env=env)
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"{target} failed with exit code {e.returncode}",
                )
        return TaskResult(TaskStatus.SUCCESS)

    def run(self) -> TaskResult:
        if result := _ensure_uv_installed():
            return result

        if result := self._install_test_deps(install_all_wheels=False):
            if result.status == TaskStatus.FAILED:
                return result

        if result := self._run_cpp_test_binaries():
            if result.status == TaskStatus.FAILED:
                return result

        test_paths = [REPO_ROOT / "TrainingExtensions/common/test"]
        if self.framework == "onnx":
            test_paths = [
                REPO_ROOT / "ModelOptimizations/DlQuantization/test",
                *test_paths,
                REPO_ROOT / "TrainingExtensions/onnx/test",
            ]
        elif self.framework == "torch":
            test_paths = [*test_paths, REPO_ROOT / "TrainingExtensions/torch/test"]

        return self._pytest(test_paths, ["-c", "pyproject.toml"])


class AcceptanceTestTask(_CiTaskBase):
    """Task that installs whatever pre-built wheels were downloaded to
    --wheel-dir and runs the acceptance test suite. NightlyTests/common
    needs both torch and onnx installed; only Linux legs download both, so
    it's skipped when aimet_torch isn't present (Windows/macOS onnx-cpu).

    TODO: #6211 test_torch_to_onnx.py is explicitly ignored on macOS/Windows
    below since it isn't supported there yet. Move torch-to-onnx tests out
    of this combined-wheel acceptance path instead of relying on an
    --ignore, and re-enable it on macOS/Windows once supported."""

    def run(self) -> TaskResult:
        if result := _ensure_uv_installed():
            return result

        if result := self._install_test_deps(install_all_wheels=True):
            if result.status == TaskStatus.FAILED:
                return result

        pytest_args = ["-c", "pyproject.toml"]
        test_paths = []
        if list(self.wheel_dir.glob("aimet_torch*.whl")):
            test_paths.append(REPO_ROOT / "NightlyTests/common")
            if on_macos() or on_windows():
                pytest_args += [
                    "--ignore",
                    str(REPO_ROOT / "NightlyTests/common/test_torch_to_onnx.py"),
                ]
        if self.framework == "onnx":
            test_paths.append(REPO_ROOT / "NightlyTests/onnx")
        elif self.framework == "torch":
            test_paths.append(REPO_ROOT / "NightlyTests/torch")
            venv_python = str(venv_python_path(self.venv_path))
            try:
                subprocess.run(
                    ["uv", "pip", "install", "wget"],
                    check=True,
                    capture_output=False,
                    env=uv_env(self.venv_path),
                )
                subprocess.run(
                    [
                        venv_python,
                        str(REPO_ROOT / "NightlyTests/torch/dependencies.py"),
                        str(REPO_ROOT / "NightlyTests/torch/resnet18_eval_scores.csv"),
                    ],
                    check=True,
                    capture_output=False,
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"torch acceptance test setup failed with exit code {e.returncode}",
                )

        return self._pytest(test_paths, pytest_args)


class ListTasksTask(Task):
    """Task that prints available tasks."""

    def __init__(self, tasks: dict[str, str]) -> None:
        super().__init__("list_tasks")
        self.tasks = tasks

    def run(self) -> TaskResult:
        print("\n" + "=" * 80)
        print("AIMET Build and Test Script")
        print("=" * 80)

        print("\nTasks:")
        print("-" * 80)
        for task_name, description in sorted(self.tasks.items()):
            print(f"  {task_name:<15} {description}")
        print("-" * 80)

        print("\nCommon Options:")
        print(
            "  --package PKG     Package to build/test: aimet_torch, aimet_onnx, docs"
        )
        print("  --gpu             Enable GPU/CUDA support (default: CPU only)")
        print("  --skip TASK       Skip specified task(s)")
        print("  --only            Run only the specified task, skip dependencies")
        print("  --dry-run         Print execution plan without running")

        print("\nBuild Options:")
        print("  --editable, -e    Install in editable mode for local development")
        print("  --force-rebuild   Force rebuild even if package is already installed")

        print("\nTest Options:")
        print("  --parallel N, -n  Run tests with N parallel workers (default: 8)")
        print("  --force-rebuild   Force wheel reinstall before testing")

        print("\nExamples:")
        print("  # Build aimet_torch (CPU, wheel)")
        print(
            "  python scripts/all/build_and_test.py --task build --package aimet_torch"
        )
        print()
        print("  # Build aimet_torch (GPU, editable for local dev)")
        print(
            "  python scripts/all/build_and_test.py --task build --package aimet_torch --gpu -e"
        )
        print()
        print("  # Run tests (skips build if already installed)")
        print("  python scripts/all/build_and_test.py --task test --package aimet_onnx")
        print()
        print("  # Build and test with forced rebuild")
        print(
            "  python scripts/all/build_and_test.py --task test --package aimet_torch --force-rebuild"
        )
        print()
        print("  # Run tests only (skip build dependency)")
        print(
            "  python scripts/all/build_and_test.py --task test --package aimet_torch --only"
        )
        print()
        print("=" * 80)

        return TaskResult(TaskStatus.SUCCESS)


def task(func: Callable) -> Callable:
    """Decorator to register a task."""
    task_name = func.__name__
    ALL_TASKS[task_name] = func.__doc__ or ""
    return func


def public_task(description: str) -> Callable:
    """Decorator to register a public (user-facing) task."""

    def decorator(func: Callable) -> Callable:
        task_name = func.__name__
        ALL_TASKS[task_name] = description
        PUBLIC_TASKS[task_name] = description
        return func

    return decorator


def depends(dependencies: list[str]) -> Callable:
    """Decorator to specify task dependencies."""

    def decorator(func: Callable) -> Callable:
        task_name = func.__name__
        TASK_DEPENDENCIES[task_name] = dependencies
        return func

    return decorator


@dataclass
class Plan:
    """Execution plan for tasks."""

    steps: list[str] = field(default_factory=list)
    tasks: dict[str, Task] = field(default_factory=dict)
    results: dict[str, TaskResult] = field(default_factory=dict)

    def add_step(self, task: Task, step_id: str) -> str:
        """Add a step to the plan."""
        if step_id not in self.steps:
            self.steps.append(step_id)
            self.tasks[step_id] = task
        return step_id

    def has_step(self, step_id: str) -> bool:
        """Check if a step exists in the plan."""
        return step_id in self.steps

    def run(self) -> bool:
        """Execute all steps in the plan."""
        import time

        all_success = True
        for step_id in self.steps:
            task = self.tasks[step_id]
            if not task.does_work():
                self.results[step_id] = TaskResult(TaskStatus.SKIPPED, "No-op task")
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Starting task: {step_id}")
            logger.info(f"{'=' * 60}")

            start_time = time.time()
            result = task.run()
            result.duration_seconds = time.time() - start_time

            self.results[step_id] = result

            if result.status == TaskStatus.FAILED:
                logger.error(f"Task {step_id} FAILED: {result.message}")
                all_success = False
                break
            else:
                logger.info(
                    f"Task {step_id} completed in {result.duration_seconds:.2f}s"
                )

        return all_success

    def print(self) -> None:
        """Print the plan (dry-run mode)."""
        print("\nExecution plan:")
        print("-" * 60)
        for i, step_id in enumerate(self.steps, 1):
            task = self.tasks[step_id]
            work_indicator = "" if task.does_work() else " (no-op)"
            print(f"  {i}. {step_id}{work_indicator}")
        print("-" * 60)

    def print_report(self) -> None:
        """Print execution report."""
        print("\n" + "=" * 60)
        print("Execution Report")
        print("=" * 60)

        for step_id, result in self.results.items():
            status_str = result.status.value.upper()
            duration_str = (
                f"{result.duration_seconds:.2f}s" if result.duration_seconds > 0 else ""
            )
            print(f"  {step_id:<40} {status_str:<10} {duration_str}")

        print("=" * 60)


@dataclass
class BuildConfig:
    """Configuration for build tasks."""

    package: str | None = None
    gpu: bool = False
    ver_torch: str = ""
    force_rebuild: bool = False
    editable: bool = False
    parallel: int = 8
    # Release-wheel options, for building the onnx-cpu release wheel
    # (pinned onnxruntime, PIP_INDEX=pypi, optional vcpkg/cross-compile).
    pip_index: str = ""
    ver_onnxruntime: str = ""
    enable_tests: bool = False
    macos_deployment_target: str = "15.0"
    windows_arch: str = ""
    windows_vcpkg_triplet: str = ""
    windows_python_version: str = ""


@dataclass
class CiConfig:
    """Configuration for the install_test_deps/unit_test/acceptance_test
    tasks, which install a pre-built wheel (downloaded as a CI artifact,
    not built from source) and run pytest."""

    wheel_dir: Path = field(default_factory=lambda: REPO_ROOT / "downloads")
    ver_cuda: str = ""
    parallel: int = 8


class TaskLibrary:
    """Library of all available tasks for AIMET."""

    def __init__(
        self,
        python_executable: str = DEFAULT_PYTHON,
        venv_path: Path | None = None,
        build_dir: Path | None = None,
        build_config: BuildConfig | None = None,
        ci_config: CiConfig | None = None,
        ver_python: str = "3.10",
    ) -> None:
        self.python_executable = python_executable
        self.venv_path = venv_path or DEFAULT_VENV_PATH
        self.build_dir = build_dir or DEFAULT_BUILD_DIR
        self.build_config = build_config or BuildConfig()
        self.ci_config = ci_config or CiConfig()
        self.ver_python = ver_python

    @staticmethod
    def to_dot(highlight: list[str] | None = None) -> str:
        """Render the task registry and its dependencies as a Graphviz DOT digraph.

        Public tasks (those with a user-facing description) are drawn filled;
        tasks passed in ``highlight`` (e.g. the plan for the task(s) actually
        requested on the command line) are drawn with a thicker border.
        """
        highlight = highlight or []
        elements: list[str] = []
        for tsk in ALL_TASKS:
            task_attrs: list[str] = []
            if tsk in PUBLIC_TASKS:
                task_attrs.append("style=filled")
            if tsk in highlight:
                task_attrs.append("penwidth=4.0")
            if task_attrs:
                elements.append(f"{tsk} [{' '.join(task_attrs)}]")
            else:
                elements.append(tsk)
        for tsk, deps in TASK_DEPENDENCIES.items():
            for dep in deps:
                elements.append(f"{tsk} -> {dep}")
        elements_str = "\n".join(f"  {element};" for element in elements)
        return f"digraph {{\n{elements_str}\n}}"

    # -------------------------------------------------------------------------
    # Dependency Installation Tasks
    # -------------------------------------------------------------------------

    @public_task("Install system dependencies (Linux/macOS)")
    def install_deps(self, plan: Plan) -> str:
        class ConditionalInstallDepsTask(Task):
            def __init__(self) -> None:
                super().__init__("Installing system dependencies")

            def run(self) -> TaskResult:
                if on_linux():
                    if are_ubuntu_deps_installed():
                        logger.info("Ubuntu dependencies already installed, skipping")
                        return TaskResult(
                            TaskStatus.SUCCESS, "Skipped - already installed"
                        )

                    cmd = [str(REPO_ROOT / "scripts/posix/install_ubuntu_deps.sh")]
                    logger.info(f"Running: {' '.join(cmd)}")
                    try:
                        subprocess.run(
                            cmd, cwd=REPO_ROOT, check=True, capture_output=False
                        )
                        # script's `export PATH` only applies to its own subshell
                        local_bin = str(Path.home() / ".local" / "bin")
                        if local_bin not in os.environ.get("PATH", "").split(
                            os.pathsep
                        ):
                            os.environ["PATH"] = (
                                f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"
                            )
                        return TaskResult(TaskStatus.SUCCESS)
                    except subprocess.CalledProcessError as e:
                        return TaskResult(
                            TaskStatus.FAILED,
                            f"Script failed with exit code {e.returncode}",
                        )

                elif on_macos():
                    if are_macos_deps_installed():
                        logger.info("macOS dependencies already installed, skipping")
                        return TaskResult(
                            TaskStatus.SUCCESS, "Skipped - already installed"
                        )

                    cmd = [str(REPO_ROOT / "scripts/posix/install_macos_deps.sh")]
                    logger.info(f"Running: {' '.join(cmd)}")
                    try:
                        subprocess.run(
                            cmd, cwd=REPO_ROOT, check=True, capture_output=False
                        )
                        return TaskResult(TaskStatus.SUCCESS)
                    except subprocess.CalledProcessError as e:
                        return TaskResult(
                            TaskStatus.FAILED,
                            f"Script failed with exit code {e.returncode}",
                        )

                else:
                    logger.info("Skipping deps installation (unsupported platform)")
                    return TaskResult(
                        TaskStatus.SUCCESS, "Skipped - unsupported platform"
                    )

        return plan.add_step(ConditionalInstallDepsTask(), "install_deps")

    # -------------------------------------------------------------------------
    # Build Task (unified)
    # -------------------------------------------------------------------------

    @public_task(
        "Build AIMET (use --package and --gpu). --windows-arch, "
        "--windows-vcpkg-triplet, --pip-index, --ver-onnxruntime, "
        "--enable-tests and --macos-deployment-target additionally support "
        "building the onnx-cpu release wheel for Windows cross-compile or "
        "a pinned macOS/PyPI release."
    )
    @depends(["venv"])
    def build(self, plan: Plan) -> str:
        cfg = self.build_config
        package = cfg.package
        gpu = cfg.gpu
        force_rebuild = cfg.force_rebuild
        editable = cfg.editable

        if not package:
            logger.error("--package is required for build task")
            sys.exit(1)
        if cfg.windows_arch and not cfg.windows_vcpkg_triplet:
            logger.error(
                "--windows-vcpkg-triplet is required when --windows-arch is set"
            )
            sys.exit(1)

        # Determine build flags based on package
        if package == "docs":
            enable_torch = True
            enable_onnx = True
            enable_docs = True
            skip_check = None
        elif package == "aimet_torch":
            enable_torch = True
            enable_onnx = False
            enable_docs = False
            skip_check = is_aimet_torch_installed
        elif package == "aimet_onnx":
            enable_torch = False
            enable_onnx = True
            enable_docs = False
            skip_check = is_aimet_onnx_installed
        else:
            logger.error(f"Unknown package: {package}")
            sys.exit(1)

        variant = "GPU" if gpu else "CPU"
        mode = "editable" if editable else "wheel"
        common_kwargs = dict(
            venv_path=self.venv_path,
            skip_if_installed=skip_check,
            force_rebuild=force_rebuild,
            enable_torch=enable_torch,
            enable_onnx=enable_onnx,
            enable_cuda=gpu,
            enable_docs=enable_docs,
            editable=editable,
            enable_tests=cfg.enable_tests,
            pip_index=cfg.pip_index,
            ver_torch=cfg.ver_torch,
            ver_onnxruntime=cfg.ver_onnxruntime,
        )
        name = f"Building {package} ({variant}, {mode})"
        if on_windows():
            build_task = WindowsWheelBuildTask(
                name,
                windows_arch=cfg.windows_arch,
                windows_vcpkg_triplet=cfg.windows_vcpkg_triplet,
                windows_python_version=cfg.windows_python_version or self.ver_python,
                **common_kwargs,
            )
        elif on_macos():
            build_task = MacOSWheelBuildTask(
                name,
                macos_deployment_target=cfg.macos_deployment_target,
                **common_kwargs,
            )
        else:
            build_task = LinuxWheelBuildTask(name, **common_kwargs)
        return plan.add_step(build_task, "build")

    @public_task(
        "Re-tag already-built wheels in --wheel-dir: strip python/abi tags "
        "to py3-none-any for any pure-Python (torch/docs) build, and/or "
        "(--windows-arch arm64) re-tag a Windows ARM64 cross-compiled wheel "
        "from win_amd64."
    )
    @depends(["venv"])
    def retag_wheels(self, plan: Plan) -> str:
        cfg = self.build_config
        return plan.add_step(
            WheelRetagTask(
                "Re-tagging built wheels",
                venv_path=self.venv_path,
                wheelhouse_dir=self.ci_config.wheel_dir,
                retag_torch=cfg.package in ("aimet_torch", "docs"),
                retag_windows_arm64=on_windows() and cfg.windows_arch == "arm64",
            ),
            "retag_wheels",
        )

    # -------------------------------------------------------------------------
    # Test Task (unified)
    # -------------------------------------------------------------------------

    @public_task("Run tests (use --package and --gpu)")
    @depends(["build"])
    def test(self, plan: Plan) -> str:
        package = self.build_config.package
        gpu = self.build_config.gpu
        force_rebuild = self.build_config.force_rebuild
        editable = self.build_config.editable

        if not package:
            logger.error("--package is required for test task")
            sys.exit(1)

        if package == "docs":
            logger.error("--package docs does not support tests")
            sys.exit(1)

        if package == "aimet_torch":
            test_paths = [REPO_ROOT / "TrainingExtensions/torch/test/python"]
            markers = None if gpu else "not cuda"
            skip_check = is_aimet_torch_installed
        elif package == "aimet_onnx":
            test_paths = [REPO_ROOT / "TrainingExtensions/onnx/test/python"]
            markers = None if gpu else "not cuda"
            skip_check = is_aimet_onnx_installed
        else:
            logger.error(f"Unknown package: {package}")
            sys.exit(1)

        # Skip wheel installation for editable mode or if already installed (unless --force-rebuild)
        if editable:
            skip_install_if = lambda: True  # Always skip for editable
        elif force_rebuild:
            skip_install_if = None
        else:
            skip_install_if = skip_check

        parallel = self.build_config.parallel
        variant = "GPU" if gpu else "CPU"
        return plan.add_step(
            PyTestTask(
                f"Running {package} tests ({variant})",
                venv_path=self.venv_path,
                test_paths=test_paths,
                markers=markers,
                skip_install_if=skip_install_if,
                parallel=parallel,
                enable_cuda=gpu,
            ),
            "test",
        )

    # -------------------------------------------------------------------------
    # CI Tasks
    # -------------------------------------------------------------------------
    #
    # Unlike build/test above, these never build from source: the wheel is
    # built in a separate CI job and downloaded as an artifact, so these
    # tasks only need to install it and run pytest.

    def _ci_framework(self) -> str:
        package = self.build_config.package
        if not package or package == "docs":
            logger.error("--package (aimet_torch or aimet_onnx) is required")
            sys.exit(1)
        return "onnx" if package == "aimet_onnx" else "torch"

    @public_task(
        "Install a pre-built wheel (from --wheel-dir) with [test] extras, "
        "without running any tests (use --package)"
    )
    @depends(["venv"])
    def install_test_deps(self, plan: Plan) -> str:
        package = self.build_config.package
        framework = self._ci_framework()
        return plan.add_step(
            InstallTestDepsTask(
                f"Installing {package} wheel",
                venv_path=self.venv_path,
                wheel_dir=self.ci_config.wheel_dir,
                framework=framework,
                ver_cuda=self.ci_config.ver_cuda,
                parallel=self.ci_config.parallel,
            ),
            "install_test_deps",
        )

    @public_task(
        "Install a pre-built wheel (from --wheel-dir) and run the unit test "
        "suite (use --package, --gpu)"
    )
    @depends(["install_test_deps"])
    def unit_test(self, plan: Plan) -> str:
        package = self.build_config.package
        framework = self._ci_framework()
        return plan.add_step(
            UnitTestTask(
                f"Running {package} unit tests",
                venv_path=self.venv_path,
                wheel_dir=self.ci_config.wheel_dir,
                framework=framework,
                ver_cuda=self.ci_config.ver_cuda,
                parallel=self.ci_config.parallel,
            ),
            "unit_test",
        )

    @public_task(
        "Install the pre-built torch and onnx wheels (from --wheel-dir) and "
        "run the acceptance test suite (use --package, --gpu)"
    )
    @depends(["venv"])
    def acceptance_test(self, plan: Plan) -> str:
        package = self.build_config.package
        framework = self._ci_framework()
        return plan.add_step(
            AcceptanceTestTask(
                f"Running {package} acceptance tests",
                venv_path=self.venv_path,
                wheel_dir=self.ci_config.wheel_dir,
                framework=framework,
                ver_cuda=self.ci_config.ver_cuda,
                parallel=self.ci_config.parallel,
            ),
            "acceptance_test",
        )

    # -------------------------------------------------------------------------
    # Utility Tasks
    # -------------------------------------------------------------------------

    @public_task("Clean build artifacts")
    def clean(self, plan: Plan) -> str:
        class CleanTask(Task):
            def __init__(self) -> None:
                super().__init__("Cleaning build artifacts")

            def run(self) -> TaskResult:
                dirs_to_clean = [
                    REPO_ROOT / "build",
                    REPO_ROOT / "dist",
                    REPO_ROOT / "wheelhouse",
                ]
                for d in dirs_to_clean:
                    if d.exists():
                        logger.info(f"Removing {d}")
                        shutil.rmtree(d)

                # Clean egg-info directories
                for egg_info in REPO_ROOT.glob("*.egg-info"):
                    logger.info(f"Removing {egg_info}")
                    shutil.rmtree(egg_info)

                return TaskResult(TaskStatus.SUCCESS)

        return plan.add_step(CleanTask(), "clean")

    @public_task("Create virtual environment using uv")
    @depends(["install_deps"])
    def venv(self, plan: Plan) -> str:
        class CreateVenvTask(Task):
            def __init__(self, venv_path: Path, ver_python: str) -> None:
                super().__init__("Creating virtual environment")
                self.venv_path = venv_path
                self.ver_python = ver_python

            def run(self) -> TaskResult:
                if self.venv_path.exists():
                    logger.info(f"Venv already exists at {self.venv_path}")
                    return TaskResult(TaskStatus.SUCCESS, "Already exists")

                # Check for system uv (installed via install_deps)
                if not shutil.which("uv"):
                    return TaskResult(
                        TaskStatus.FAILED,
                        "uv not found. Run: python scripts/all/build_and_test.py --task install_deps",
                    )

                logger.info(f"Creating venv at {self.venv_path}")
                cmd = ["uv", "venv", "--python", self.ver_python, str(self.venv_path)]
                logger.info(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"uv venv failed with exit code {e.returncode}",
                    )

                # Install pip in the venv
                venv_python = venv_python_path(self.venv_path)
                logger.info("Installing pip in venv...")
                try:
                    subprocess.run(
                        [str(venv_python), "-m", "ensurepip", "--upgrade"],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install pip in venv: {e}")

                logger.info(
                    f"Venv created. Activate: source {self.venv_path}/bin/activate"
                )
                return TaskResult(TaskStatus.SUCCESS)

        return plan.add_step(CreateVenvTask(self.venv_path, self.ver_python), "venv")

    @public_task("Run pre-commit hooks on all files")
    @depends(["venv"])
    def precommit(self, plan: Plan) -> str:
        return plan.add_step(
            CommandTask(
                "Running pre-commit",
                ["pre-commit", "run", "--all-files"],
            ),
            "precommit",
        )

    # -------------------------------------------------------------------------
    # List Tasks
    # -------------------------------------------------------------------------

    @public_task("List available tasks")
    def list(self, plan: Plan) -> str:
        return plan.add_step(ListTasksTask(PUBLIC_TASKS), "list")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and test AIMET (AI Model Efficiency Toolkit).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to run: build, test, clean, list, precommit, install_deps",
    )

    parser.add_argument(
        "--package",
        type=str,
        choices=VALID_PACKAGES,
        help="Package to build/test: aimet_torch, aimet_onnx, docs",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU/CUDA support (default: CPU only)",
    )

    parser.add_argument(
        "--skip",
        metavar="TASK",
        type=str,
        nargs="+",
        help="List of tasks to skip.",
    )

    parser.add_argument(
        "--only",
        action="store_true",
        help="Run only the listed task(s), skipping any dependencies.",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if packages are already installed.",
    )

    parser.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="Install in editable mode for local development.",
    )

    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=8,
        metavar="N",
        help="Run tests in parallel with N workers (default: 8, requires pytest-xdist).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, rather than running it.",
    )

    parser.add_argument(
        "--print-task-graph",
        action="store_true",
        help="Print the full task library as a Graphviz DOT digraph and exit "
        "(highlighting --task's plan), instead of running anything. "
        "Pipe to `dot -Tsvg -o tasks.svg` to visualize.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help="Python executable path or name.",
    )

    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(DEFAULT_BUILD_DIR),
        help=f"Build directory (default: {DEFAULT_BUILD_DIR}).",
    )

    parser.add_argument(
        "--venv",
        type=str,
        default=str(DEFAULT_VENV_PATH),
        help=f"Virtualenv directory to create/use (default: {DEFAULT_VENV_PATH}).",
    )

    parser.add_argument(
        "--ver-python",
        type=str,
        default="3.10",
        help="Python version for a newly created venv (default: 3.10).",
    )

    parser.add_argument(
        "--wheel-dir",
        type=str,
        default="downloads",
        help="Directory containing pre-built wheel(s) to install for "
        "install_test_deps/unit_test/acceptance_test (default: downloads), "
        "or to re-tag in place for retag_wheels (default: downloads; pass "
        "'wheelhouse' to retag wheels just built by this script's own "
        "build task).",
    )

    parser.add_argument(
        "--ver-cuda",
        type=str,
        default="",
        help="CUDA toolkit version string (e.g. '13.2.0'), empty for CPU. "
        "Used by install_test_deps/unit_test/acceptance_test to select the "
        "torch index and pytest markers.",
    )

    parser.add_argument(
        "--pip-index",
        type=str,
        default="",
        help="(build) PIP_INDEX to pass to CMake, e.g. 'pypi' to resolve "
        "build-time deps from public PyPI for a release wheel.",
    )

    parser.add_argument(
        "--ver-torch",
        type=str,
        default="",
        help="(build) torch version to pin when installing dev/test deps, "
        "e.g. '2.*'. Empty pins the default dev constraint (2.10.*).",
    )

    parser.add_argument(
        "--ver-onnxruntime",
        type=str,
        default="",
        help="(build) onnxruntime version to pin, e.g. for a release wheel "
        "build. Empty leaves onnxruntime unpinned.",
    )

    parser.add_argument(
        "--enable-tests",
        action="store_true",
        help="(build) Also build the ModelOptimizations C++ unit test "
        "binaries (-DENABLE_TESTS=ON).",
    )

    parser.add_argument(
        "--macos-deployment-target",
        type=str,
        default="15.0",
        help="(build) MACOSX_DEPLOYMENT_TARGET / CMAKE_OSX_DEPLOYMENT_TARGET "
        "to pin, lower than the runner's live macOS version (default: 15.0).",
    )

    parser.add_argument(
        "--windows-arch",
        type=str,
        default="",
        choices=["", "x64", "arm64"],
        help="(build) Windows target architecture. 'arm64' cross-compiles "
        "using the pythonarm64 nuget package. Empty builds for the host "
        "platform (Linux/macOS) with no vcpkg/MSVC setup.",
    )

    parser.add_argument(
        "--windows-vcpkg-triplet",
        type=str,
        default="",
        help="(build) vcpkg triplet for Eigen3 (e.g. 'x64-windows', "
        "'arm64-windows'). Required when --windows-arch is set.",
    )

    return parser.parse_args()


def plan_from_dependencies(
    main_tasks: list[str],
    task_library: TaskLibrary,
) -> Plan:
    """Build a plan with all dependencies resolved."""
    plan = Plan()
    work_list = list(reversed(main_tasks))

    for task_name in work_list:
        if not hasattr(task_library, task_name):
            logger.fatal(f"Task '{task_name}' does not exist.")
            sys.exit(1)

    while work_list:
        task_name = work_list.pop()
        if plan.has_step(task_name):
            continue

        unfulfilled_deps: list[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                if not hasattr(task_library, dep):
                    logger.fatal(
                        f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'."
                    )
                    sys.exit(1)

        if not unfulfilled_deps:
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            added_step = task_adder(plan)
            if added_step != task_name:
                logger.warning(
                    f"Task function '{task_name}' added a task with incorrect id '{added_step}'."
                )
        else:
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: list[str],
    task_library: TaskLibrary,
) -> Plan:
    """Build a plan with only the specified tasks (no dependencies)."""
    plan = Plan()
    for task_name in tasks:
        if not hasattr(task_library, task_name):
            logger.fatal(f"Task '{task_name}' does not exist.")
            sys.exit(1)
        task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
        task_adder(plan)
    return plan


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Create build config from args
    build_config = BuildConfig(
        package=args.package,
        gpu=args.gpu,
        ver_torch=args.ver_torch,
        force_rebuild=args.force_rebuild,
        editable=args.editable,
        parallel=args.parallel,
        pip_index=args.pip_index,
        ver_onnxruntime=args.ver_onnxruntime,
        enable_tests=args.enable_tests,
        macos_deployment_target=args.macos_deployment_target,
        windows_arch=args.windows_arch,
        windows_vcpkg_triplet=args.windows_vcpkg_triplet,
        windows_python_version=args.ver_python,
    )

    ci_config = CiConfig(
        wheel_dir=Path(args.wheel_dir),
        ver_cuda=args.ver_cuda,
        parallel=args.parallel,
    )

    task_library = TaskLibrary(
        python_executable=args.python,
        venv_path=Path(args.venv).resolve(),
        build_dir=Path(args.build_dir),
        build_config=build_config,
        ci_config=ci_config,
        ver_python=args.ver_python,
    )

    # Build the plan
    tasks = [args.task]
    if args.only:
        plan = plan_from_task_list(tasks, task_library)
    else:
        plan = plan_from_dependencies(tasks, task_library)

    # Skip specified tasks
    if args.skip:
        plan.steps = [s for s in plan.steps if s not in args.skip]

    if args.print_task_graph:
        print(task_library.to_dot(plan.steps))
        return 0

    # Execute or print the plan
    if args.dry_run:
        plan.print()
        return 0

    try:
        success = plan.run()
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        success = False
    finally:
        print()
        plan.print_report()
        print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
