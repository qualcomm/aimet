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

.. role:: hideitem
   :class: hideitem

.. _ug-installation:

##################
AIMET Installation
##################

This page contains instructions for the following installation scenarios:

- :ref:`Quick Install of the latest version for PyTorch <idx-install-quick>`
- :ref:`Installation of the latest version for all platforms <idx-install-latest>`
- :ref:`Installation of older versions <idx-install-older>`

You can also:

- :doc:`Install manually on a host machine <install_host>`
- :doc:`Install a pre-built or locally built image in a Docker container <install_docker>`

.. note::
    
    The following system requirements apply to all installations.

System requirements
===================

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* Linux Ubuntu 22.04 LTS (Python 3.10) or Ubuntu 20.04 LTS (Python 3.8)
* bash command shell
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)
    * nvidia-docker - Installation instructions: https://github.com/NVIDIA/nvidia-docker

.. _idx-install-quick:

Quick Install
=============

The fastest way to install AIMET is to use the AIMET PyTorch GPU PyPI packages.

Prerequisites
-------------

The following software versions are required for the quick install:

* CUDA 12.0
* Torch 2.2.2

Ensure that you have these prerequisite packages installed:

.. code-block:: bash

    apt-get install liblapacke libpython3-dev

Installation
------------

- **Type the following command to install AIMET using Pip:**

.. code-block::

    python3 -m pip install aimet-torch


Next steps
----------

See the :doc:`Quantization User Guide </user_guide/model_quantization>` for a discussion of how to use AIMET quantization.

See the :doc:`Examples Documentation </user_guide/examples>` to try AIMET on example quantization and compression problems.


.. _idx-install-latest:

Installing  the latest version with release packages
====================================================

Install the latest version of any AIMET variant from the **.whl** files hosted at https://github.com/quic/aimet/releases.


Prerequisites
-------------

These prerequisites apply to all frameworks and variants.

Ensure that you have these prerequisite packages installed:

.. code-block:: bash

    apt-get install liblapacke libpython3-dev

Install a compatible version of pip. The latest version is *not* compatible with our wheel packages.

.. code-block:: bash

    python3 -m pip install pip==24.0

Choose and install a package
----------------------------

Use one of the following commands to install AIMET based on your choice of framework and runtime environment.

.. note::
    
    Python dependencies are automatically installed.

**PyTorch 2.1**

With CUDA 12.x:

.. parsed-literal::

   python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu121\ |whl_suffix| -f |torch_pkg_url|

With CPU only:

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|

**Pytorch 1.13 with CUDA 11.x**

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|


**Tensorflow 2.10 GPU**

With CUDA 11.x:

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cu118\ |whl_suffix| -f |torch_pkg_url|

With CPU only:

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|


**ONNX 1.16 GPU**

With CUDA 11.x:

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|

With CPU only:

.. parsed-literal::

    python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|


Next steps
----------

See the :doc:`Quantization User Guide </user_guide/model_quantization>` for a discussion of how to use AIMET quantization.

See the :doc:`Examples Documentation </user_guide/examples>` to try AIMET on example quantization and compression problems.


.. _idx-install-older:

Installing an older version
===========================

View the release notes for older versions at https://github.com/quic/aimet/releases. Follow the documentation corresponding to that release to select and install the appropriate package.

.. |whl_suffix| replace:: -cp310-cp310-manylinux_2_34_x86_64.whl
.. |download_url| replace:: \https://github.com/quic/aimet/releases/download/
.. |torch_pkg_url| replace:: \https://download.pytorch.org/whl/torch_stable.html

:hideitem:`Other installation options`
======================================

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   Install on a host machine <install_host>
   Install in a Docker container <install_docker>
