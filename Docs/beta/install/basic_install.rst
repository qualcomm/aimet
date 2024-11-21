.. _install-basic-install:

##################
Basic Installation
##################

Install the latest version of AIMET pacakge for all framework variants and compute platform from
the **.whl** files hosted at https://github.com/quic/aimet/releases.

Prerequisites
=============

The AIMET package requires the following host platform setup. Following prerequisites apply
to all frameworks variants.

* 64-bit Intel x86-compatible processor
* OS: Ubuntu 22.04 LTS
* Python 3.10
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)


Ensure that you have following debian package(s) installed:

.. code-block:: bash

    apt-get install liblapacke

Install a compatible version of pip. The latest version is *not* compatible with our wheel packages.

.. code-block:: bash

    python3 -m pip install pip==24.0

Choose and install a package
============================

Use one of the following commands to install AIMET based on your choice of framework and compute platform.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **PyTorch 2.1**

        With CUDA 12.x:

        .. parsed-literal::

           python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu121\ |whl_suffix| -f |torch_pkg_url|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|

        **Pytorch 1.13**

        with CUDA 11.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_torch-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|

    .. tab-item:: TensorFlow
        :sync: tf

        **Tensorflow 2.10 GPU**

        With CUDA 11.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cu118\ |whl_suffix| -f |torch_pkg_url|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_tensorflow-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|

    .. tab-item:: ONNX
        :sync: onnx

        **ONNX 1.16 GPU**

        With CUDA 11.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cu117\ |whl_suffix| -f |torch_pkg_url|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-\ |version|.cpu\ |whl_suffix| -f |torch_pkg_url|

.. |whl_suffix| replace:: -cp310-cp310-manylinux_2_34_x86_64.whl
.. |download_url| replace:: \https://github.com/quic/aimet/releases/download/
.. |torch_pkg_url| replace:: \https://download.pytorch.org/whl/torch_stable.html

Next steps
==========

See `Simple example` to test your installation.

See the `Optimization guide` to read about the model optimization workflow.



