.. role:: hideitem
   :class: hideitem
.. _ug-index:

######################################
AI Model Efficiency Toolkit User Guide
######################################

Overview
========

AI Model Efficiency Toolkit (AIMET) is a software toolkit that enables users to quantize and compress models.
Quantization is a must for efficient edge inference using fixed-point AI accelerators.

AIMET optimizes pre-trained models (for example, FP32 trained models) using post-training and fine-tuning techniques that minimize accuracy loss incurred during quantization or compression.

AIMET supports PyTorch, TensorFlow, and Keras models.

The following diagram shows a high-level view of the AIMET workflow. 

.. image:: ../images/AIMET_index_no_fine_tune.png

You train a model in the PyTorch, TensorFlow, or Keras training framework, then pass the model to AIMET, using its APIs for compression and quantization. AIMET returns a compressed and/or quantized version of the model that you can fine-tune (or train further for a small number of epochs) to recover lost accuracy. You can then export the model using ONNX, meta/checkpoint, or h5 to an on-target runtime like the Qualcomm\ |reg| Neural Processing SDK.

Features
========

AIMET supports two model optimization techniques:

Model Quantization
  AIMET can simulate the behavior of quantized hardware for a trained model. This model can be optimized using Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) fine-tuning techniques.

Model Compression
  AIMET supports multiple model compression techniques that remove redundancies from a trained model, resulting in a smaller model that runs faster on target.

More Information
================

For more information about AIMET, see the following documentation:

- :ref:`Installation <ug-installation>`
- :ref:`Quantization User Guide <ug-model-quantization>`
- :ref:`Compression User Guide <ug-model-compression>`
- :ref:`Examples Documentation <ug-examples>`
- :ref:`API Documentation <ug-apidocs>`

Release Information
===================

For information specific to this release, see :ref:`Release Notes <ug-release-notes>` and :ref:`Known Issues <ug-known-issues>`.

:hideitem:`toc tree`
------------------------------------
.. toctree::
  :hidden:

   Installation <../install/index>
   Quantization User Guide <model_quantization>
   Compression User Guide <model_compression>
   API Documentation<../api_docs/index>
   Examples Documentation <examples>


| |project| is a product of |author|
| Qualcomm\ |reg| Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
