.. # =============================================================================
   #  @@-COPYRIGHT-START-@@
   #
   #  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

.. _installation-host:

##############################
AIMET Installation and Setup
##############################

This page provides instructions to install AIMET package on Ubuntu 22.04 LTS with Nvidia GPU. Please follow the instructions in the order provided, unless specified otherwise.

**NOTE:**
    #. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate.
    #. These instructions assume that pip packages will be installed in the path: /usr/local/lib/python3.10/dist-packages. If that is not the case, please modify it accordingly.


Install prerequisite packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the basic pre-requisite packages as follows:

.. code-block:: bash

    apt-get update
    apt-get install python3.10 python3.10-dev python3-pip
    python3 -m pip install --upgrade pip
    apt-get install --assume-yes wget gnupg2

If you have multiple python versions installed, set the default python version as follows:

.. code-block:: bash

    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    update-alternatives --set python3 /usr/bin/python3.10

Install GPU packages
~~~~~~~~~~~~~~~~~~~~~

**NOTE:**

#. Do this section ONLY for the GPU variants.
#. The released AIMET GPU packages *were tested* with the following CUDA toolkit versions:
    #. PyTorch 2.1 GPU variant: `CUDA Toolkit 12.1.0 <https://developer.nvidia.com/cuda-12-1-0-download-archive>`_
    #. PyTorch 1.13 GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_
    #. TensorFlow GPU variant: `CUDA Toolkit 11.8.0 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_
    #. ONNX GPU variant: `CUDA Toolkit 11.8.0 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_
#. The instructions in the sub-sections below correspond to our tested versions above. Visit this page https://developer.nvidia.com/cuda-toolkit-archive to obtain the correct version of the CUDA toolkit for your environment.

Install GPU packages for PyTorch 2.1 or PyTorch 1.13 or ONNX or TensorFlow
==========================================================================

**NOTE:**

#. Visit this page https://developer.nvidia.com/cuda-12-1-0-download-archive or https://developer.nvidia.com/cuda-11-8-0-download-archive or https://developer.nvidia.com/cuda-11-7-1-download-archive to obtain the exact and up-to-date installation instructions for your environment.
#. Please do not execute the final command "sudo apt-get install cuda" provided in aforementioned NVIDIA documentation links.

.. code-block:: bash

    apt-get update && apt-get install -y gnupg2
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update

Install AIMET packages
~~~~~~~~~~~~~~~~~~~~~~~

From PyPI
=========

The default AIMET Torch GPU variant may be installed from PyPI as follows:
    - Go to https://pypi.org/project/aimet-torch
    - Browse the Requirements section of each Release to identify the version you wish to install. Following are some tips:
        - For Pytorch 2.2.2 GPU with CUDA 12.1, use aimet-torch>=1.32.2
        - For Pytorch 2.1.2 GPU with CUDA 12.1, use aimet-torch==1.32.1.post1
        - For PyTorch 1.13 GPU with CUDA 11.7, use aimet-torch==1.31.2

Run the following commands to install the package (prepend with "sudo" and/or package version as needed):

.. code-block:: bash

    apt-get install liblapacke -y
    python3 -m pip install aimet-torch

From Release Package
====================

We also host python wheel packages for different variants which may be installed as follows:
    - Go to https://github.com/quic/aimet/releases
    - Identify the release tag of the package that you wish to install
    - Identify the .whl file corresponding to the package variant that you wish to install
    - Follow the instructions below to install AIMET from the .whl file

Set the package details as follows:

.. code-block:: bash

    # Set the release tag ex. "1.34.0"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension
    # ex. "aimet_torch-1.33.0.cu121-cp310-cp310-manylinux_2_34_x86_64.whl"
    export wheel_file_name="<wheel file name>"

    # NOTE: Do the following ONLY for the PyTorch and ONNX variant packages!
    export find_pkg_url_str="-f https://download.pytorch.org/whl/torch_stable.html"

Install the selected AIMET package as specified below:

**NOTE:** Python dependencies will automatically get installed.

.. code-block:: bash

    # Install the wheel package
    python3 -m pip install ${download_url}/${wheel_file_name} ${find_pkg_url_str}

Install common debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the common debian packages as follows:

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the PyTorch variant packages.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the ONNX variant packages.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes install

Install tensorflow GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the TensorFlow GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install

Install torch GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install

Install ONNX GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the ONNX GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes install

Replace Pillow with Pillow-SIMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optional:** Replace the Pillow package with Pillow-SIMD as follows:

.. code-block:: bash

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

Replace onnxruntime with onnxruntime-gpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block:: bash

    export ONNXRUNTIME_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')
    python3 -m pip uninstall -y onnxruntime
    python3 -m pip install --no-cache-dir onnxruntime-gpu==$ONNXRUNTIME_VER

Post installation steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

**NOTE:** Do the following step ONLY for the PyTorch or Tensorflow GPU packages.

.. code-block:: bash

    # NOTE: Please chose between the below command depending on the version of your CUDA driver toolkit
    ln -s /usr/local/cuda-11.7 /usr/local/cuda
    ln -s /usr/local/cuda-11.8 /usr/local/cuda
    ln -s /usr/local/cuda-12.1 /usr/local/cuda

Environment setup
~~~~~~~~~~~~~~~~~

Set the common environment variables as follows:

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh
