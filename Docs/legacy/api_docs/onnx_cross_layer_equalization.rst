:orphan:

.. _api-onnx-cle:

========================================
AIMET ONNX Cross Layer Equalization APIs
========================================

User Guide Link
===============
To learn more about this technique, please see :ref:`Cross-Layer Equalization<ug-post-training-quantization>`

Introduction
============
AIMET functionality for Cross Layer Equalization has 3 features-
   - BatchNorm Folding
   - Cross Layer Scaling
   - High Bias Fold


Cross Layer Equalization API
============================

The following API performs BatchNorm fold followed by Cross Layer Scaling followed by High Bias Fold.

Note: High Bias fold will not happen when the below API is used, if the model does not have BatchNorm layers

**API for Cross Layer Equalization**

.. autofunction:: aimet_onnx.cross_layer_equalization.equalize_model

|

**Note:** It is recommended to use onnx-simplifier before applying cross layer equalization.


Code Example
============

**Required imports**

.. literalinclude:: ../onnx_code_examples/cross_layer_equalization.py
    :language: python
    :lines: 39-40


**Cross Layer Equalization in auto mode**

.. literalinclude:: ../onnx_code_examples/cross_layer_equalization.py
    :language: python
    :pyobject: cross_layer_equalization
