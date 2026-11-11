.. _apiref-onnx-quantsim:

###################
aimet_onnx.quantsim
###################

..
  # start-after

.. note::
    It is recommended to use onnx-simplifier before creating quantsim model.

.. autoclass:: aimet_onnx.QuantizationSimModel
   :members: compute_encodings, export, to_onnx_qdq, from_onnx_qdq, set_tensor_precision

.. autofunction:: aimet_onnx.compute_encodings

.. autofunction:: aimet_onnx.encodings_to_onnx_qdq

.. autofunction:: aimet_onnx.quantsim.set_param_type

.. rubric:: Quantization Spec

.. autoclass:: aimet_onnx.QSpec
    :members:

.. rubric:: Quant Scheme Enum

.. autoclass:: aimet_onnx.common.defs.QuantScheme
    :members:
