.. _api-onnx-amp:

==========================
aimet_onnx.mixed_precision
==========================

..
  # start-after

**Top-level API**

.. autofunction:: aimet_onnx.mixed_precision.choose_mixed_precision


.. note::

    It is recommended to use onnx-simplifier before applying mixed-precision.


**Quantizer Groups definition**

.. autoclass:: aimet_onnx.amp.quantizer_groups.QuantizerGroup
   :members:

**CallbackFunc Definition**

.. autoclass:: aimet_common.defs.CallbackFunc
   :members:

.. autoclass:: aimet_onnx.amp.mixed_precision_algo.EvalCallbackFactory
   :members:

..
  # end-before
