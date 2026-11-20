:icon: info

.. include:: ../abbreviation.txt

.. _intro-to-aimet:

##############
What is AIMET?
##############

AIMET (AI Model Efficiency Toolkit) is an open-source Python library that **quantizes trained
PyTorch and ONNX models** so they run faster and use less memory and power on edge devices such as
phones, laptops, and cars, while keeping accuracy as close as possible to the original model.

You give AIMET a trained floating-point model and a small amount of representative data. AIMET
gives you back a quantized model, ready to compile and run on target hardware.

Why quantize?
=============

Trained models usually store weights and compute activations in 32-bit floating point. Quantization
represents them with fewer bits, for example 8-bit or 16-bit integers. Compared to 32-bit floating point,
8-bit weights take a quarter of the memory, and integer math runs faster and uses less
power on hardware accelerators.

The cost is accuracy. Each value is rounded to a coarser grid, and that rounding error accumulates
through the network. AIMET lets you **measure that loss before you deploy** and provides techniques
to **recover it**, so you can choose the lowest precision your accuracy target allows.

How AIMET works
===============

.. raw:: html
   :file: ../images/aimet_workflow.svg

#. **Simulate.** AIMET wraps your model in a :ref:`quantization simulation <quantsim-index>`
   (``QuantizationSimModel``) that inserts quantizers on weights and activations. A short
   calibration pass over representative data, typically 500 to 1,000 samples, computes the
   quantization parameters (*encodings*).
#. **Evaluate.** The simulated model runs on your CPU or GPU and behaves like the quantized model will
   on device. Run your existing evaluation pipeline on it to see the accuracy at the chosen precision.
#. **Improve.** If accuracy falls short, apply AIMET's techniques, then evaluate again:

   - :ref:`Post-training quantization (PTQ) techniques <featureguide-index>` adjust weights and
     encodings using calibration data only, with no training. Examples include AdaRound,
     SeqMSE, and AdaScale.
   - :ref:`Mixed precision <featureguide-mp-index>` keeps the few layers that are sensitive to
     quantization at higher precision and quantizes the rest more aggressively.
   - :ref:`Quantization-aware training (QAT) <techniques-qat>` fine-tunes the model with
     quantization in the loop. Use it when PTQ and mixed precision are not enough.

#. **Export.** AIMET exports the result as an ONNX model plus an encodings file, or as a single
   :ref:`ONNX QDQ model <techniques-onnx-qdq>` with ``QuantizeLinear`` and ``DequantizeLinear``
   nodes in the graph.

You then :ref:`deploy the exported model <opt-guide-on-target-inference>` with |qai_hub|_,
the Qualcomm\ |reg| AI Runtime (QAIRT) SDK, or |ort|_.

For a step-by-step guide to picking precisions and techniques, see the
:ref:`quantization workflow <tutorials-quantization-workflow>`.

What you can do with AIMET
==========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Capability
     - What it gives you
   * - :ref:`Quantization simulation <quantsim-index>`
     - Accuracy of the quantized model at any weight/activation precision (for example W8A8,
       W8A16, W4A16), measured on your own hardware before deployment.
   * - :ref:`Post-training quantization <techniques-ptq>`
     - Calibration plus accuracy-recovery techniques such as :ref:`AdaRound <ptq-adaround>`,
       :ref:`SeqMSE <ptq-seq-mse>`, :ref:`AdaScale <ptq-adascale>`, and
       :ref:`SpinQuant <ptq-spinquant>`, using unlabeled data and no training.
   * - :ref:`Mixed precision <featureguide-mp-index>`
     - Per-layer precision chosen automatically (:ref:`Lite-MP <featureguide-litemp>`,
       :ref:`AMP <featureguide-amp>`) or by hand, to balance accuracy and latency.
   * - :ref:`Blockwise quantization <techniques-lpbq>`
     - Low-Power Blockwise Quantization (LPBQ) for 4-bit weights, used by large language models.
   * - :ref:`Analysis tools <techniques-analysis-tools>`
     - Per-layer sensitivity reports that show which layers lose the most accuracy.
   * - :ref:`Quantization-aware training <techniques-qat>`
     - Training with quantization in the loop, for models that need it (``aimet-torch``).
   * - :ref:`LLM quantization recipes <quantization-genai-recipe>`
     - Tested end-to-end recipes for popular large language models, with accuracy results.

Choose a package
================

AIMET ships as two Python packages. Each one takes a model in its own framework's format.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 20 40 40

   * -
     - ``aimet-onnx``
     - ``aimet-torch``
   * - Input model
     - ONNX model (``onnx.ModelProto``)
     - PyTorch model (``torch.nn.Module``)
   * - Best for
     - Post-training quantization and fast experiments
     - Quantization-aware training, or staying in PyTorch
   * - Unique features
     - Lite mixed precision, ONNX QDQ conversion of existing encodings
     - QAT, OmniQuant, batch norm re-estimation, model compression
   * - Install
     - ``pip install aimet-onnx``
     - ``pip install aimet-torch``

.. tip::

   **Start with** ``aimet-onnx``. It has the fastest turnaround for calibration, mixed precision,
   and PTQ techniques. For a PyTorch model, export it to ONNX with ``torch.onnx.export`` first.
   Move to ``aimet-torch`` for QAT only if PTQ and mixed precision don't meet your accuracy
   target.

Supported platforms
===================

AIMET runs on Linux, Windows, and macOS with Python 3.10 or later (tested through 3.13). NVIDIA
GPUs are optional and speed up calibration and optimization. Support varies by package and
architecture; see :ref:`Installation <install-index>` for the full matrix.

Next steps
==========

- :ref:`Quick start <install-quick-start>`: install AIMET and quantize your first model.
- :ref:`Quantization workflow <tutorials-quantization-workflow>`: a recommended path from first
  simulation to a deployable model.
- :ref:`Example notebooks <examples-notebooks-index>`: end-to-end examples you can run.

.. toctree::
    :hidden:

    Quick Start <install/quick-start>
    Install <install/index>
