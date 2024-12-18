.. _apiref-torch-v1-amp:

==============================
aimet_torch.v1.mixed_precision
==============================

..
  # start-after

.. note::
    This module is also available in the default :mod:`aimet_torch` namespace with
    the same top-level API.

**Top-level API**

.. autofunction:: aimet_torch.v1.mixed_precision.choose_mixed_precision

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
