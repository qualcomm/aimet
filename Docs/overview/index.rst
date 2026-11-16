:icon: info

.. _intro-to-aimet:

#####################
What is AIMET?
#####################

AIMET (AI Model Efficiency Toolkit) is a quantization tool that works on deep learning models such as PyTorch and ONNX.

AIMET enables developer to:

#. Simulate Quantization
#. Quantize models with Post-Training Quantization (PTQ) techniques
#. Quantization Aware Training (QAT) on PyTorch models with aimet-torch
#. Visualize and experiment with model accuracy with various precision on activations and weights
#. Create mixed-precision models
#. Export Quantized model to deployable ONNX model format

With AIMET, developers can rapidly iterate on model to find best quantization profile to achieve state-of-the-art accuracy and latency.
Developers can compile and run quantized model exported from AIMET to QNN or run directly via ONNX-Runtime.

.. image:: ../images/aimet_overview.png

AIMET provides two python packages:

1. AIMET-ONNX: Quantize ONNX model with PTQ techniques
2. AIMET-Torch: Quantize PyTorch model with QAT

We recommend to start with AIMET-ONNX PTQ techniques, which has fastest turn-around time for quantization and experimentation.
Defer to QAT with AIMET-Torch only if you have tried AIMET-ONNX mixed precision and advanced techniques for optimizing weights.

Supported platforms
====================

AIMET supports Linux, Windows, and macOS, with Python 3.10+ (tested through 3.13).
Operating system and architecture support differs by package; see
:ref:`Supported platforms <install-index>` for the full per-package,
per-architecture matrix.

Get Started
================

Visit here to :ref:`quick start <install-quick-start>`.

.. toctree::
    :hidden:

    Quick Start <install/quick-start>
    Install <install/index>
