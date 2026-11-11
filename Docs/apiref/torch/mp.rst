.. _api-torch-mp:

===========================
aimet_torch.mixed_precision
===========================

.. rubric:: Top-level API for Manual mixed precision

..
  # start-after mmp

.. autoclass:: aimet_torch.v2.mixed_precision.MixedPrecisionConfigurator
    :members:
    :member-order: bysource
    :no-index:

..
  # end-before mmp

..
  # start-after full-amp

.. rubric:: Top-level API for Automatic mixed precision

..
  # start-after amp-toplevel

.. autofunction:: aimet_torch.mixed_precision.choose_mixed_precision
    :no-index:

.. note::
    To enable phase-3 set the attribute GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

Currently only two candidates are supported - ((8,int), (8,int)) & ((16,int), (8,int))

..
  # end-before amp-toplevel

.. rubric:: Quantizer Groups definition

..
  # start-after amp-quantizergroups

.. autoclass:: aimet_torch.amp.quantizer_groups.QuantizerGroup
   :members:
   :no-index:

..
  # end-before amp-quantizergroups

.. rubric:: CallbackFunc Definition

..
  # start-after amp-callback

.. autoclass:: aimet_torch.common.defs.CallbackFunc
   :members:
   :no-index:

.. autoclass:: aimet_torch.amp.mixed_precision_algo.EvalCallbackFactory
   :members:
   :no-index:

..
  # end-before amp-callback
  # end-before full-amp
