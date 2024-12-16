
.. _api-onnx-mixed-precision:

==============================
AIMET ONNX Mixed Precision API
==============================

Top-level API
=============
.. autofunction:: aimet_onnx.mixed_precision.choose_mixed_precision

|

**Note:** It is recommended to use onnx-simplifier before applying mixed-precision.


Quantizer Groups definition
===========================
.. autoclass:: aimet_onnx.amp.quantizer_groups.QuantizerGroup
   :members:

|

CallbackFunc Definition
=======================
.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

|

.. autoclass:: aimet_onnx.amp.mixed_precision_algo.EvalCallbackFactory
   :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../onnx_code_examples/mixed_precision.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Quantization with mixed precision**

.. literalinclude:: ../onnx_code_examples/mixed_precision.py
    :language: python
    :pyobject: quantize_with_mixed_precision

**Quantization with mixed precision start from existing cache**

.. literalinclude:: ../onnx_code_examples/mixed_precision.py
    :language: python
    :pyobject: quantize_with_mixed_precision_start_from_existing_cache

**Eval function**

.. literalinclude:: ../onnx_code_examples/mixed_precision.py
    :language: python
    :pyobject: eval_callback_func

**Forward Pass**

.. literalinclude:: ../onnx_code_examples/mixed_precision.py
    :language: python
    :pyobject: forward_pass_callback
