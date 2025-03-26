.. _apiref-onnx-quantsim:

###################
aimet_onnx.quantsim
###################

..
  # start-after

.. note::
    It is recommended to use onnx-simplifier before creating quantsim model.

.. autoclass:: aimet_onnx.quantsim.QuantizationSimModel
   :no-index:

**The following API can be used to compute encodings for calibration.**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.compute_encodings
   :no-index:

**The following API can be used to export the quantized model to target.**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.export
   :no-index:

Enum Definition
===============

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
    :no-index:
