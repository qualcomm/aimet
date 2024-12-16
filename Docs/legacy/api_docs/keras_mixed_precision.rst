
.. _api-keras-mixed-precision:

====================================
AIMET TensorFlow Mixed Precision API
====================================

Top-level API for Regular AMP
=============================
.. autofunction:: aimet_tensorflow.keras.mixed_precision.choose_mixed_precision

|

Top-level API for Fast AMP (AMP 2.0)
====================================
.. autofunction:: aimet_tensorflow.keras.mixed_precision.choose_fast_mixed_precision

|

**Note:** To enable phase-3 set the attribute GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

Currently only two candidates are supported - ((8,int), (8,int)) & ((16,int), (8,int))

|

Quantizer Groups definition
===========================
.. autoclass:: aimet_tensorflow.amp.quantizer_groups.QuantizerGroup
   :members:

|

CallbackFunc Definition
=======================
.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

Code Examples
=============

**Required imports**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :lines: 31-53

**Load Resnet50 model**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :pyobject: get_model

**Eval function**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :pyobject: get_eval_func

**Data Loader Wrapper function**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :pyobject: get_data_loader_wrapper

**Quantization with regular mixed precision**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :pyobject: mixed_precision

**Quantization with fast mixed precision**

.. literalinclude:: ../keras_code_examples/mixed_precision.py
    :language: python
    :pyobject: fast_mixed_precision
