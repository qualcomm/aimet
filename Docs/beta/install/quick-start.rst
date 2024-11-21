.. _install-quick-start:

###########
Quick Start
###########

This page describes how to quickly install the latest version of AIMET for PyTorch framework.

For all the framework variants and compute platform, see :ref:`Installation <install-index>`.

Prerequisites
=============

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* OS : Linux
    * Ubuntu 22.04 LTS (Python 3.10)
    * Ubuntu 20.04 LTS (Python 3.8)
* python : Supported python version(s):  3.10, 3.8
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

The following software versions are required for the quick install:

* CUDA Toolkit 12.0
* Torch 2.2.2

Ensure that you have following debian packages installed:

.. code-block::

    apt-get install liblapacke libpython3-dev

Installation
============

Type the following command to install AIMET using pip package manager.

.. code-block::

    python3 -m pip install aimet-torch

Next steps
==========

See `Simple example` to test your installation.

See the `Optimization guide` to read about the model optimization workflow.
