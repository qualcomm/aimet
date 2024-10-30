#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
#==============================================================================

""" Package generation file for aimet torch package """

import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


CURRENT_DIR = Path(__file__).parent.resolve()
PACKAGING_DIR = CURRENT_DIR / ".." / ".." / ".." / ".." / "packaging"

def str2bool(str_):
    TRUE_VALS = {"true", "yes", "y", "on", "1"}
    FALSE_VALS = {"false", "no", "n", "off", "0"}
    if str_.lower() in TRUE_VALS:
        return True
    elif str_.lower() in FALSE_VALS:
        return False
    else:
        raise RuntimeError(f"Unknown boolean value '{str_}' (known values are: {TRUE_VALS | FALSE_VALS}")

ENABLE_CUDA = str2bool(os.environ.get("ENABLE_CUDA", "False"))

REQUIRES_DIR = Path("dependencies") / f"torch-{'gpu' if ENABLE_CUDA else 'cpu'}{'-p36' if sys.version_info.minor == 6 else ''}"

AIMET_TORCH_VERSION = os.environ.get("SW_VERSION")
if AIMET_TORCH_VERSION is None:
    AIMET_TORCH_VERSION = (PACKAGING_DIR / "version.txt").read_text().strip()

AIMET_TORCH_URL = subprocess.run(
        shlex.split("git config --get remote.origin.url"), check=True,
        cwd=CURRENT_DIR, stdout=subprocess.PIPE, encoding="utf8",
    ).stdout + f"/releases/download/{AIMET_TORCH_VERSION}"


class BuildExtensionCommand(build_ext):
    def run(self):
        super().run()
        # Create dest directories
        dst_dir = Path(self.get_ext_fullpath("dummy")).parent / "aimet_torch"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "bin").mkdir(parents=True, exist_ok=True)

        subprocess.run(
            shlex.split(f"cp -Lrv {PACKAGING_DIR / REQUIRES_DIR}/. {dst_dir / 'bin'}"),
            check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
            )

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    author_email="aimet.os@quicinc.com",
    author="Qualcomm Innovation Center, Inc.",
    cmdclass={"build_ext": BuildExtensionCommand, },
    description="AIMET PyTorch Package",
    distclass = BinaryDistribution,
    install_requires=list(filter(lambda r: not r.startswith('-'),
        subprocess.run([sys.executable, str(PACKAGING_DIR / "dependencies.py"), "pip"],
        check=True, stdout=subprocess.PIPE, encoding="utf8").stdout.splitlines()
    )),
    license="NOTICE.txt",
    long_description=(PACKAGING_DIR / "README.txt").read_text(),
    name="AimetTorch",
    package_data={"": ["*.json", "*.html", "*.js", "*.xml"], },
    package_dir={"": ".", },
    packages=find_namespace_packages(where=".", exclude=["build"]),
    platforms="x86",
    python_requires=">=3.6",
    url=AIMET_TORCH_URL,
    version=AIMET_TORCH_VERSION,
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
)