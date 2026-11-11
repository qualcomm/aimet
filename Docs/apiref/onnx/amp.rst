.. _api-onnx-amp:

==========================
aimet_onnx.mixed_precision
==========================

..
  # start-after

.. rubric:: Top-level API

..
  # start-after amp-toplevel

.. autofunction:: aimet_onnx.mixed_precision.choose_mixed_precision
    :no-index:


.. note::

    It is recommended to use onnx-simplifier before applying mixed-precision.

..
  # end-before amp-toplevel

.. rubric:: Quantizer Groups definition

..
  # start-after amp-quantizergroups

.. autoclass:: aimet_onnx.amp.quantizer_groups.QuantizerGroup
   :members:
   :no-index:

..
  # end-before amp-quantizergroups

.. rubric:: CallbackFunc Definition

..
  # start-after amp-callback

.. autoclass:: aimet_onnx.common.defs.CallbackFunc
   :members:
   :no-index:

.. autoclass:: aimet_onnx.amp.mixed_precision_algo.EvalCallbackFactory
   :members:
   :no-index:

..
  # end-before amp-callback
  # end-before
