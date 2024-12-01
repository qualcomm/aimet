.. include:: abbreviation.txt

.. _top-index:

#########################################
AI Model Efficiency Toolkit Documentation
#########################################

.. toctree::
   :hidden:
   :includehidden:

   Quick Start <install/quick-start>
   Installation <install/index>
   User Guide <opt-guide/index>
   Quantization Simulation Guide <quantsim/index>
   Feature Guide <featureguide/index>
   Examples <examples/index>
   API Reference <apiref/index>
   Release Notes <rn/index>

AI Model Efficiency Toolkit (AIMET) is a software toolkit for quantizing and compressing models.

The goal of optimizing a model is to enable its use on an edge device such as a mobile phone or laptop. 

AIMET uses post-training and fine tuning techniques to optimize trained models in ways that minimize accuracy loss incurred during quantization or compression.

AIMET supports PyTorch, TensorFlow, and Keras models, and ONNX models with limited functionality.

Quick Start
===========

To install and get started as quickly as possibly using AIMET with PyTorch, see the :doc:`Quick start guide <../install/quick-start>`.

Installation
=============

For other install options, including for TensorFlow and ONNX platforms or to run AIMET in a Docker container, see :doc:`Installation <../install/index>`.

User Guide
==========

For a high-level explanation of how to use AIMET to optimize a model, see the :doc:`Optimization user guide <../opt-guide/index>`.

Quantization Simulation Guide
=============================

Quantization simulation (QuantSim) provides an approximation of a quantized model by inserting
quantization operations in a trained model. QuantSim enables application of optimization
techniques to a model and testing of the resulting model before the model is exported.
See the doc :doc:`Quantization simulation guide <../quantsim/index>`

Feature Guide
=============

For instructions on applying individual AIMET features, see the :doc:`Feature guide <../featureguide/index>`.

Examples
========

To view end-to-end examples of model quantization and compression, and to download the examples in Jupyter notebook format, see :doc:`Examples <../examples/index>`.

API Reference
=============

For a detailed look at the AIMET API, see the :doc:`API reference <../apiref/index>`.

Release Notes
=============

For information specific to this release, see :doc:`Release notes <../rn/index>`.


| |project| is a product of |author|
| Qualcomm\ |reg| Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.

