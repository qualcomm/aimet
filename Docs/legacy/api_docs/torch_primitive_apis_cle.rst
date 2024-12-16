:orphan:

====================================================
AIMET PyTorch Cross Layer Equalization Primitive API
====================================================

Introduction
============

If a user wants to modify the order of Cross Layer equalization, not use some features or manually tweak the list of
layers that need to be equalized, the following APIs can be used.

Higher level APIs can be used for using one or more features one after the other. It automatically finds the layers to
be folded or scaled.

Lower level APIs can be used to manually tweak the list of layers to be folded. The user has to pass the list of
layers in the correct order that they appear in the model.

Note: Before using High Bias fold, Cross Layer Scaling (CLS) needs to be applied and scaling factors obtained from
CLS need to be plugged in to High Bias Fold. And, if there are batchnorm layers, they need to be folded and the info
saved to be plugged into high bias fold API.


ClsSetInfo Definition
=====================

.. autoclass:: aimet_torch.cross_layer_equalization.ClsSetInfo
   :members:

|


Higher Level APIs for Cross Layer Equalization
==============================================


**API for Batch Norm Folding**

.. autofunction:: aimet_torch.v1.batch_norm_fold.fold_all_batch_norms

|


**API for Cross Layer Scaling**

.. autofunction:: aimet_torch.cross_layer_equalization.CrossLayerScaling.scale_model

|


**API for High Bias Folding**

.. autofunction:: aimet_torch.cross_layer_equalization.HighBiasFold.bias_fold

|


Code Examples for Higher Level APIs
===================================

**Required imports**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 40-41, 44-45, 48


**Cross Layer Equalization in auto mode calling each API**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_auto_step_by_step


Lower Level APIs for Cross Layer Equalization
=============================================



**API for Batch Norm Folding**

.. autofunction:: aimet_torch.v1.batch_norm_fold.fold_given_batch_norms

|

**API for Cross Layer Scaling**

.. autofunction:: aimet_torch.cross_layer_equalization.CrossLayerScaling.scale_cls_sets

|

**API for High bias folding**

.. autofunction:: aimet_torch.cross_layer_equalization.HighBiasFold.bias_fold

|


Code Examples for Lower Level APIs
==================================

**Required imports**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 41, 44-45, 48

**Cross Layer Equalization in manual mode**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_manual

**Cross Layer Equalization in manual mode for Depthwise Separable layer**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_depthwise_layers

