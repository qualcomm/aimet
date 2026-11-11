.. _apiref-torch-quantsim:

####################
aimet_torch.quantsim
####################

..
  # start-after

.. autoclass:: aimet_torch.QuantizationSimModel
   :members: compute_encodings, export, fold_param_quantizers, onnx, qmodules, named_qmodules, quantizers, named_quantizers, quantizer_parameters, named_quantizer_parameters, quantizer_state_dict

.. autoclass:: aimet_torch.quantsim.QuantizationSimModelOnnxExporter
   :members: export

.. rubric:: Quant Scheme Enum

.. autoclass:: aimet_torch.common.defs.QuantScheme
    :members:
