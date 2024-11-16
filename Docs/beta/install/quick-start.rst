.. _install-quick-start:

###########
Quick Start
###########

This page describes how to quickly install the AI Model Efficiency Toolkit (AIMET) for PyTorch on a local . For other installation options, see :ref:`Installation <install-index>`.

Prerequisites
=============

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* Linux: Ubuntu 22.04 LTS (Python 3.10) or Ubuntu 20.04 LTS (Python 3.8)
* bash command shell
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

The following software versions are required for the quick install:

* CUDA 12.0
* Torch 2.2.2

Ensure that you have these prerequisite packages installed:

.. code-block:: bash

    apt-get install liblapacke libpython3-dev

Installation Workflow
=====================

- **Type the following command to install AIMET using Pip:**

.. code-block::

    python3 -m pip install aimet-torch

Next steps
==========

See `Simple example` to test your installation.

See the `Optimization guide` to read about the model optimization workflow.
