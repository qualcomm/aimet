# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=all
import os
import sys
import glob
import subprocess
import sysconfig


def _get_min_glibc_version(build_dir):
    if sys.platform != "linux":
        return None

    so_file_list = glob.glob(os.path.join(build_dir, "**/*.so"), recursive=True)
    so_file_list = [file for file in so_file_list if "artifacts" in file]
    glibc_ver_list = []
    try:
        for so_file in so_file_list:
            command = f"objdump -T {so_file} | grep GLIBC | sed 's/.*GLIBC_\\([.0-9]*\\).*/\\1/g' | sort -Vu | tail -1"
            glibc_ver = subprocess.check_output(command, shell=True).decode('utf-8').strip()
            glibc_ver_list.append(glibc_ver)
    except subprocess.CalledProcessError:
        return None

    if not glibc_ver_list:
        return None

    return sorted(glibc_ver_list, key=lambda x: list(map(int, x.split('.'))), reverse=True)[0]

def main(output_dir, build_dir):
    try:
        import torch
    except ImportError:
        torch = None

    min_glibc_version = _get_min_glibc_version(build_dir)

    _template = os.path.join(os.path.dirname(__file__), "_deps.pyi")

    with open(_template) as f:
        copyright_string = [
            line.strip() for line in f.readlines()
            if line.startswith('#')
        ]

    content = [
        f"python_abi = '{sysconfig.get_config_var('SOABI')}'",
        "torch = "      + (f"'{torch.__version__}'" if torch else "None"),
        "min_glibc = "  + (f"'{min_glibc_version}'" if min_glibc_version else "None"),
        # "tensorflow = " + (f"'{tf.__version__}'" if tf else "None"),
        # "onnx = "       + (f"'{onnx.__version__}'" if onnx else "None"),
        ""
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "_deps.py"), "w") as f:
        f.write('\n'.join(copyright_string + content))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--build-dir", type=str)
    args = parser.parse_args()
    main(args.output_dir, args.build_dir)
