.. _install-index:

############
Installation
############

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

.. note::
    Starting with the AIMET 2 release, there is no longer a dependency on ``liblapacke``. Therefore,
    you should only install the following debian package if you are still using AIMET 1.x.

.. code-block:: bash

    apt-get install liblapacke

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

           python3 -m pip install |download_url|\ |version|/aimet_torch-|version|\+cu121\ |torch_whl_suffix| -f |torch_pkg_url|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_torch-|version|\+cpu\ |torch_whl_suffix| -f |torch_pkg_url|


    .. tab-item:: TensorFlow
        :sync: tf

        **Tensorflow 2.10 GPU**

        With CUDA 11.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_tensorflow-|version|\+cu118\ |whl_suffix|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_tensorflow-|version|\+cpu\ |whl_suffix|

    .. tab-item:: ONNX
        :sync: onnx

        **ONNX 1.16 GPU**

        With CUDA 11.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-|version|\+cu118\ |whl_suffix| -f |torch_pkg_url|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-|version|\+cpu\ |whl_suffix| -f |torch_pkg_url|

.. |torch_whl_suffix| replace:: \-cp310-none-any.whl
.. |whl_suffix| replace:: \-cp310-cp310-manylinux_2_34_x86_64.whl
.. |download_url| replace:: \https://github.com/quic/aimet/releases/download/
.. |torch_pkg_url| replace:: \https://download.pytorch.org/whl/torch_stable.html

Advanced installation instructions (optional)
=============================================

Following are two ways to setup, including prerequisites and dependencies.

* :ref:`On your host machine <install-host>`
* :ref:`Using our pre-built or locally built Docker images <install-docker>`

Installing an older version
===========================

View the release notes for older versions at https://github.com/quic/aimet/releases. Follow the
documentation corresponding to that release to select and install the appropriate AIMET package.

Building from source
====================

To build the latest AIMET code from the source, see `build AIMET from source <https://github.com/quic/aimet/blob/develop/packaging/docker_install.md>`_.
