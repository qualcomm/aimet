.. _api-tensorflow-amp:

================================
aimet_tensorflow.mixed_precision
================================

..
  # start-after

**Top-level API for Regular AMP**

.. autofunction:: aimet_tensorflow.keras.mixed_precision.choose_mixed_precision


**Top-level API for Fast AMP (AMP 2.0)**

.. autofunction:: aimet_tensorflow.keras.mixed_precision.choose_fast_mixed_precision

.. note::

    To enable phase-3 set the attribute GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

Currently only two candidates are supported - ((8,int), (8,int)) & ((16,int), (8,int))

**Quantizer Groups definition**

.. autoclass:: aimet_tensorflow.keras.amp.quantizer_groups.QuantizerGroup
   :members:

**CallbackFunc Definition**

.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

..
  # end-before
