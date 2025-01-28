.. _apiref-torch-quantsim:

####################
aimet_torch.quantsim
####################

..
  # start-after

.. autoclass:: aimet_torch.QuantizationSimModel
   :members: compute_encodings, export

**The following APIs can be used to save and restore the quantized model**

.. automethod:: aimet_torch.quantsim.save_checkpoint

.. automethod:: aimet_torch.quantsim.load_checkpoint

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
