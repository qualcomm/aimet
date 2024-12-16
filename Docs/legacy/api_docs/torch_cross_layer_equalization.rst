:orphan:

.. _api-torch-cle:

===========================================
AIMET PyTorch Cross Layer Equalization APIs
===========================================

User Guide Link
===============
To learn more about this technique, please see :ref:`Cross-Layer Equalization<ug-post-training-quantization>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch Cross Layer Equalization, please see :doc:`here<../Examples/torch/quantization/cle_bc>`.

Introduction
============
AIMET functionality for PyTorch Cross Layer Equalization has 3 features-
   - BatchNorm Folding
   - Cross Layer Scaling
   - High Bias Fold


Cross Layer Equalization API
============================

The following API performs BatchNorm fold followed by Cross Layer Scaling followed by High Bias Fold.

Note: High Bias fold will not happen when the below API is used, if the model does not have BatchNorm layers

**API for Cross Layer Equalization**

.. autofunction:: aimet_torch.cross_layer_equalization.equalize_model

|

Code Example
============

**Required imports**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 41, 46


**Cross Layer Equalization in auto mode**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_auto


Primitive APIs
==============
If the user would like to call the APIs individually, then the following APIs can be used-

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Primitive APIs for Cross Layer Equalization<torch_primitive_apis_cle>


