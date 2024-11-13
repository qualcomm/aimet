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

###################################
AIMET manual installation and setup
###################################

This page describes how to manually install AIMET, including all prerequisites and dependencies, for all framework and variants.

.. note::

   You might need to preface the **apt-get install** and **pip3 install** commands with **sudo -H** depending on your user privileges.
   
.. note::

   These instructions assume that pip packages are installed in **/usr/local/lib/python3.10/dist-packages**. Modify the command if you use a different install directory for packages.

.. _installation-prereq:

Prerequisites
=============

Ensure that you have the following prerequisites installed:

1. Python and pip.
2. The CUDA toolkit, if using GPUs.

Instructions follow.

**1. Install Python and pip.**

1.1 Install the latest build of Python 3.10.

.. code-block:: bash

    apt-get update
    apt-get install python3.10 python3.10-dev python3-pip
    python3 -m pip install --upgrade pip
    apt-get install --assume-yes wget gnupg2

1.2 Install library dependencies.

.. code-block:: bash

    apt-get install liblapacke -y


1.3 If you have multiple Python versions installed, set the default version.

.. code-block:: bash

    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    update-alternatives --set python3 /usr/bin/python3.10


**2. Install the CUDA toolkit (optional).**

.. note::

    The GPU toolkit is required only for GPU variants of AIMET.
    
The released AIMET GPU packages have been tested with the following CUDA toolkit versions:

- PyTorch 2.1 GPU variant: `CUDA Toolkit 12.1.0 <https://developer.nvidia.com/cuda-12-1-0-download-archive>`_
- PyTorch 1.13 GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_
- TensorFlow GPU variant: `CUDA Toolkit 11.8.0 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_
- ONNX GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_

2.1 Visit the CUDA Toolkit link above for the verison corresponding to your AIMET GPU package and download the tested version of the CUDA toolkit for your environment.

All versions of the CUDA toolkit are also listed at https://developer.nvidia.com/cuda-toolkit-archive.

.. note::

    In the next step, do not execute the final command, **sudo apt-get install cuda**, in the install instructions.

2.2 Follow the command-line instructions on the developer.nvidia.com download page to install the CUDA toolkit, but do *not* execute the final command, **sudo apt-get install cuda**.

2.3 Execute the following to update the CUDA repository key.

.. code-block:: bash

    apt-get update && apt-get install -y gnupg2
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update


Installing AIMET
================

**Choose your AIMET variant.**

Based on your machine learning framework and GPU preference, choose one of the install procedures below.

:ref:`1. Installing AIMET for PyTorch <man-install-torch>`

:ref:`2. Installing AIMET for TensorFlow <man-install-tf>`

:ref:`3. Installing AIMET for ONNX <man-install-onnx>`

.. _man-install-torch:

1. Installing AIMET for PyTorch
-------------------------------

**1.1 Select the release tag for the version you want to install.**

For example, "1.34.0". Releases are listed at:

https://github.com/quic/aimet/releases

 - Identify the .whl file corresponding to the package variant that you want to install.
 - Continue with the instructions below to install AIMET from the .whl file.

**1.2 Set the package details.**

.. code-block:: bash

    # Set the release tag, for example "1.34.0"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension,
    # for example "aimet_torch-1.33.0.cu121-cp310-cp310-manylinux_2_34_x86_64.whl"
    export wheel_file_name="<wheel file name>"

    # NOTE: Do the following ONLY for the PyTorch and ONNX variant packages!
    export find_pkg_url_str="-f https://download.pytorch.org/whl/torch_stable.html"


**1.3 Install the selected AIMET package.**

.. note::
    
    Python dependencies are automatically installed.

.. code-block:: bash

    # Install an compatible version of pip (since the latest version is NOT compatible with our wheel packages)
    python3 -m pip install pip==24.0

    # Install the wheel package
    python3 -m pip install ${download_url}/${wheel_file_name} ${find_pkg_url_str}

**1.4 Install the common Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**1.5 Install the Torch Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes install

**1.6 Install the Torch GPU Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install

**1.7 Replace Pillow with Pillow-SIMD (optional).**

Pillow-SIMD is an optimized version of the Pillow Python Imaging Library. It can improve image processing performance on x86 architecture machines.

.. code-block:: bash

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

**1.8 Link to executable paths.**

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
    ln -s /usr/local/cuda-<cuda-version> /usr/local/cuda

where **<cuda-version>** is the version of CUDA that you installed in the :ref:`Prerequisite section <_installation-prereq>`, for example **12.1.0**.

**1.9 Run the environment setup script to set common environment variables.**

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh

**Installation is complete.** Proceed to :ref:`Next steps <man-install-next>`.


.. _man-install-tf:

2. Installing AIMET for TensorFlow
----------------------------------

**2.1 Select the release tag for the version you want to install, for example, "1.34.0". Releases are listed at:**

https://github.com/quic/aimet/releases

    - Identify the .whl file corresponding to the package variant that you want to install
    - Continue with the instructions below to install AIMET from the .whl file

**2.2 Set the package details.**

.. code-block:: bash

    # Set the release tag, for example "1.34.0"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension,
    # for example "aimet_tensorflow-1.34.0.cu118-cp310-cp310-manylinux_2_34_x86_64.whl"
    export wheel_file_name="<wheel file name>"

**2.3 Install the selected AIMET package.**

.. note::
    
    Python dependencies are automatically installed.

.. code-block:: bash

    python3 -m pip install ${download_url}/${wheel_file_name}


**2.4 Install the common Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**2.5 Install the tensorflow GPU debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install

**2.6 Replace Pillow with Pillow-SIMD (optional).**

Pillow-SIMD is an optimized version of the Pillow Python Imaging Library. It can improve image processing performance on x86 architecture machines.

.. code-block:: bash

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

**2.7 Link to executable paths.**

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
    ln -s /usr/local/cuda-<cuda-version> /usr/local/cuda

where **<cuda-version>** is the version of CUDA that you installed in the :ref:`Prerequisite section <_installation-prereq>`, for example **11.8.0**.

**2.8 Run the environment setup script to set common environment variables.**

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh

**Installation is complete.** Proceed to :ref:`Next steps <man-install-next>`from PyPI.


.. _man-install-onnx:

3. Installing AIMET for ONNX
----------------------------

**3.1 Select the release tag for the version you want to install, for example, "1.34.0". Releases are listed at:**

https://github.com/quic/aimet/releases

    - Identify the .whl file corresponding to the package variant that you want to install
    - Continue with the instructions below to install AIMET from the .whl file

**3.2 Set the package details.**

.. code-block:: bash

    # Set the release tag, for example "1.34.0"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension,
    # for example "aimet_onnx-1.34.0.cu117-cp310-cp310-manylinux_2_34_x86_64.whl"
    export wheel_file_name="<wheel file name>"

**3.3 Install the selected AIMET package.**

.. note::
    
    Python dependencies are automatically installed.

.. code-block:: bash

    python3 -m pip install ${download_url}/${wheel_file_name}

**3.4 Install the common Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**3.5 Install the ONNX Debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes install

**3.6 Install the ONNX GPU debian packages.**

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes install


**3.7 Replace Pillow with Pillow-SIMD (optional).**

Pillow-SIMD is an optimized version of the Pillow Python Imaging Library. It can improve image processing performance on x86 architecture machines.

.. code-block:: bash

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1


**3.8 Replace onnxruntime with onnxruntime-gpu.**

.. code-block:: bash

    export ONNXRUNTIME_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')
    python3 -m pip uninstall -y onnxruntime
    python3 -m pip install --no-cache-dir onnxruntime-gpu==$ONNXRUNTIME_VER


**3.9 Link to executable paths.**

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib


**3.10 Run the environment setup script to set common environment variables.**

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh

**Installation is complete.** Proceed to :ref:`Next steps <man-install-next>`.


.. _man-install-next:


Next steps
==========

See the :doc:`Quantization User Guide </user_guide/model_quantization>` for a discussion of how to use AIMET quantization.

See the :doc:`Examples Documentation </user_guide/examples>` to try AIMET on example quantization and compression problems.
