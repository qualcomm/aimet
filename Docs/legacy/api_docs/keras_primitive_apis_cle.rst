:orphan:

=======================================================
AIMET TensorFlow Cross Layer Equalization Primitive API
=======================================================
Introduction
============

If a user wants to modify the order of Cross Layer Equalization, not use some features, or manually tweak the list of
layers that need to be equalized, the following APIs can be used.

Higher level API can be used for using one or more features one after the other. It automatically finds the layers to
be folded or scaled.

Lower level APIs can be used to manually tweak the list of layers to be folded. The user has to pass the list of
layers in the correct order that they appear in the model.

Note: Before using High Bias fold, Cross Layer Scaling (CLS) needs to be applied and scaling factors obtained from
CLS need to be plugged in to High Bias Fold. And, if there are batchnorm layers, they need to be folded and the info
saved to be plugged into high bias fold API.

Higher Level APIs for Cross Layer Equalization
==============================================

**API for Batch Norm Folding**

.. autofunction:: aimet_tensorflow.keras.batch_norm_fold.fold_all_batch_norms

**API for Cross Layer Scaling**

.. autofunction:: aimet_tensorflow.keras.cross_layer_equalization.CrossLayerScaling.scale_model

**API for High Bias Folding**

.. autofunction:: aimet_tensorflow.keras.cross_layer_equalization.HighBiasFold.bias_fold

Code Examples for Higher Level APIs
===================================

**Required imports**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 37, 39, 41, 43

**Perform Cross Layer Equalization in auto mode step by step**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_auto_stepwise

Lower Level APIs for Cross Layer Equalization
=============================================

**API for Batch Norm Folding on subsets of convolution-batchnorm layer pairs**

.. autofunction:: aimet_tensorflow.keras.batch_norm_fold.fold_given_batch_norms

|

**API for Cross Layer Scaling on subset of conv layer groups**

.. autofunction:: aimet_tensorflow.keras.cross_layer_equalization.CrossLayerScaling.scale_cls_sets

|

**API for High bias folding**

.. autofunction:: aimet_tensorflow.keras.cross_layer_equalization.HighBiasFold.bias_fold

|


Custom Datatype used
=====================

.. autoclass:: aimet_tensorflow.keras.cross_layer_equalization.ClsSetInfo
   :members:

|

Code Example for Lower level APIs
=================================

**Required imports**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 37, 40, 41, 43

**Perform Cross Layer Equalization in manual mode**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_manual

Example helper methods to perform CLE in manual mode
====================================================

**Helper to pick layers for batchnorm fold**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: get_example_layer_pairs_resnet50_for_folding

**Helper to pick layers for cross layer scaling**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: get_consecutive_layer_list_from_resnet50_for_scaling

**Helper to format data from batchnorm fold and cross layer scaling for usage by high bias fold**

.. literalinclude:: ../keras_code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: format_info_for_high_bias_fold