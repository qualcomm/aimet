.. _top-index:

######################################
AI Model Efficiency Toolkit Documentation
######################################

.. toctree::
   :hidden:
   :includehidden:

   Quick Start <../install/quick-start>
   Installation <../install/index>
   AIMET Optimization Guide <../opt-guide/index>
   Quantization Simulation Guide <../quantsim/index>
   AIMET Feature Guide <../featureguide/index>
   Examples <../examples/index>
   API Reference <../apiref/index>
   Release Notes <../rn/index>

AI Model Efficiency Toolkit (AIMET) is a software toolkit for quantizing and compressing models.

The goal of optimizing a model is to enable its use on an edge device such as a mobile phone or laptop. 

AIMET uses post-training and fine tuning techniques to optimize trained models in ways that minimize accuracy loss incurred during quantization or compression.

AIMET supports PyTorch, TensorFlow, and Keras models, and ONNX models with limited functionality.

Quick Start
===========

To install and get started as quickly as possibly using AIMET with PyTorch, see the :doc:`Quick Start guide <../install/quick-start>`.

Installation
=============

For other install options, including for TensorFlow and ONNX platforms or to run AIMET in a Docker container, see :doc:`Installation <../install/index>`.

Optimization Guide
==================

For a high-level explanation of how to use AIMET to optimize a model, see the :doc:`Optimization User Guide <../opt-guide/index>`.

Feature Guide
=============

For instructions on applying individual AIMET features, see the :doc:`Features User Guide <../featureguide/index>`.

Quantization Simulation Guide
=============================

Quantization simulation (QuantSim) provides an approximation of a quantized model by inserting quantization operations in a trained model. QuantSim enables application of optimization techniques to a model and testing of the resulting model before the model is exported.

Examples
========

To view end-to-end examples of model quantization and compression, and to download the examples in Jupyter notebook format, see :doc:`Examples <../examples/index>`.

API Reference
=============

For a detailed look at the AIMET API, see the :doc:`API Reference <../apiref/index>`.

Release Notes
=============

For information specific to this release, see :doc:`Release Notes <../rn/index>`.


| |project| is a product of |author|
| Qualcomm\ |reg| Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
