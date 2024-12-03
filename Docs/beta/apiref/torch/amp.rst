.. _api-torch-amp:

===========================
aimet_torch.mixed_precision
===========================

..
  # start-after

**Top-level API**

.. autofunction:: aimet_torch.mixed_precision.choose_mixed_precision

.. note::

    To enable phase-3 set the attribute GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

Currently only two candidates are supported - ((8,int), (8,int)) & ((16,int), (8,int))

**Quantizer Groups definition**

.. autoclass:: aimet_torch.amp.quantizer_groups.QuantizerGroup
   :members:

**CallbackFunc Definition**

.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

.. autoclass:: aimet_torch.amp.mixed_precision_algo.EvalCallbackFactory
   :members:

..
  # end-before
