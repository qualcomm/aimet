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

.. _ug-installation:

###################
AIMET Installation
###################

Quick Install
~~~~~~~~~~~~~

The AIMET PyTorch GPU PyPI packages are available for environments that meet the following requirements:

* 64-bit Intel x86-compatible processor
* Linux Ubuntu 22.04 LTS [Python 3.10] or Ubuntu 20.04 LTS [Python 3.8]
* CUDA 12.0
* Torch 2.2.2

**Pip install**

.. code-block:: bash

    apt-get install liblapacke libpython3-dev
    python3 -m pip install aimet-torch


Release Packages
~~~~~~~~~~~~~~~~

For other AIMET variants, install the *latest* version from the .whl files hosted at https://github.com/quic/aimet/releases (compatible with python 3.10 only at this time).

**Prerequisite**

The following pre-requisites apply to all variants. The GPU variants may need additional packages - please see `Advanced Installation Instructions`_ for details.


.. code-block:: bash

    # Install pre-requisite packages
    apt-get install liblapacke libpython3-dev


**PyTorch**

.. parsed-literal::

    # Pytorch 2.1 with CUDA 12.x
    python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu121\ |whl_suffix| -f |torch_pkg_url|

    # Pytorch 2.1 CPU only
    python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|

    # Pytorch 1.13 with CUDA 11.x
    python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|


**TensorFlow**

.. parsed-literal::

    # Tensorflow 2.10 GPU with CUDA 11.x
    python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cu118\ |whl_suffix|

    # Tensorflow 2.10 CPU only
    python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cpu\ |whl_suffix|


**ONNX**

.. parsed-literal::

    # ONNX 1.16 GPU with CUDA 11.x
    python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|

    # ONNX 1.16 CPU
    python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|


For older versions, please browse the releases at https://github.com/quic/aimet/releases and follow the documentation corresponding to that release to select and install the appropriate package.

.. |whl_suffix| replace:: -cp310-cp310-manylinux_2_34_x86_64.whl
.. |download_url| replace:: \https://github.com/quic/aimet/releases/download/
.. |torch_pkg_url| replace:: \https://download.pytorch.org/whl/torch_stable.html

System Requirements
~~~~~~~~~~~~~~~~~~~

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* Linux Ubuntu: 22.04 LTS
* bash command shell
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * nvidia-docker - Installation instructions: https://github.com/NVIDIA/nvidia-docker

To use the GPU accelerated training modules an Nvidia CUDA enabled GPU with a minimum Nvidia driver version of 455+ is required. Using the latest driver is always recommended, especially if using a newer GPU. Both CUDA and cuDNN (the more advanced CUDA interface) enabled GPUs are supported.


Advanced Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to setup and install AIMET:
    * On your host machine
    * Using our pre-built development `Docker images <https://artifacts.codelinaro.org/ui/native/codelinaro-aimet/aimet-dev>`_

Please click on the appropriate link for installation instructions:

.. toctree::
   :titlesonly:
   :maxdepth: 3

   Install in Host Machine <install_host>
   Install in Docker Container <install_docker>
