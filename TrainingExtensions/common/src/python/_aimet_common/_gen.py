# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all
import glob
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import sysconfig


_PKG_ROOT = (
    pathlib.Path(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    .absolute()
    .resolve()
)


def import_from_path(module_name, file_path):
    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


aimet_packaging_plugin = import_from_path(
    module_name="aimet", file_path=_PKG_ROOT / "./packaging/plugins/local/aimet.py"
)


_GLIBC_VERSION_RE = re.compile(r"GLIBC_(\d+(?:\.\d+)*)")


def _version_key(version):
    return tuple(map(int, version.split(".")))


def _get_min_glibc_version(build_dir):
    if sys.platform != "linux":
        return None

    so_file_list = glob.glob(os.path.join(build_dir, "**/*.so"), recursive=True)
    so_file_list = [file for file in so_file_list if "artifacts" in file]
    glibc_ver_list = []
    try:
        for so_file in so_file_list:
            symbols = subprocess.check_output(["objdump", "-T", so_file]).decode(
                "utf-8"
            )
            versions = _GLIBC_VERSION_RE.findall(symbols)
            if versions:
                glibc_ver_list.append(max(versions, key=_version_key))
    except (subprocess.CalledProcessError, OSError):
        return None

    if not glibc_ver_list:
        return None

    return max(glibc_ver_list, key=_version_key)


def main(output_dir, build_dir):
    try:
        import torch
    except ImportError:
        torch = None

    min_glibc_version = _get_min_glibc_version(build_dir)

    _template = os.path.join(os.path.dirname(__file__), "_version.pyi")

    with open(_template) as f:
        copyright_string = [
            line.strip() for line in f.readlines() if line.startswith("#")
        ]

    content = [
        f"__version__ = '{aimet_packaging_plugin.get_version()}'",
        f"python_abi = '{sysconfig.get_config_var('SOABI')}'",
        "torch = " + (f"'{torch.__version__}'" if torch else "None"),
        "min_glibc = " + (f"'{min_glibc_version}'" if min_glibc_version else "None"),
        # "onnx = "       + (f"'{onnx.__version__}'" if onnx else "None"),
        "",
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "_version.py"), "w") as f:
        f.write("\n".join(copyright_string + content))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--build-dir", type=str)
    args = parser.parse_args()
    main(args.output_dir, args.build_dir)
