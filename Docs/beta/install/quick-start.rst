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
* Python 3.8–3.11
* Ubuntu 22.04
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

The following software versions are required for the quick install:

* CUDA Toolkit 12.0
* PyTorch 2.2

Installation
============

Type the following command to install AIMET for PyTorch framework using pip package manager.

.. code-block:: bash

    python3 -m pip install "aimet-torch>=2"

Next steps
==========

See `Simple example` to test your installation.

See the :ref:`User guide <opt-guide-index>` to read about the model optimization workflow.

See the :ref:`Examples <examples-index>` to try AIMET quantization techniques on your pre-trained models.

