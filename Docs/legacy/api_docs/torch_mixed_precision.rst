
.. _api-torch-mixed-precision:

=================================
AIMET PyTorch Mixed Precision API
=================================

Top-level API
=============
.. autofunction:: aimet_torch.v1.mixed_precision.choose_mixed_precision

|

**Note:** To enable phase-3 set the attribute GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

Currently only two candidates are supported - ((8,int), (8,int)) & ((16,int), (8,int))

|

Quantizer Groups definition
===========================
.. autoclass:: aimet_torch.amp.quantizer_groups.QuantizerGroup
   :members:

|

CallbackFunc Definition
=======================
.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

|

.. autoclass:: aimet_torch.amp.mixed_precision_algo.EvalCallbackFactory
   :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/mixed_precision.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Quantization with mixed precision**

.. literalinclude:: ../torch_code_examples/mixed_precision.py
    :language: python
    :pyobject: quantize_with_mixed_precision

**Quantization with mixed precision start from existing cache**

.. literalinclude:: ../torch_code_examples/mixed_precision.py
    :language: python
    :pyobject: quantize_with_mixed_precision_start_from_existing_cache

**Eval function**

.. literalinclude:: ../torch_code_examples/mixed_precision.py
    :language: python
    :pyobject: eval_callback_func

**Forward Pass**

.. literalinclude:: ../torch_code_examples/mixed_precision.py
    :language: python
    :pyobject: forward_pass_callback
