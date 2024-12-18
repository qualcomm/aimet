.. _apiref-torch-v1-quantsim:

#######################
aimet_torch.v1.quantsim
#######################

..
  # start-after

.. note::
    This module is also available in the default :mod:`aimet_torch` namespace with
    the same top-level API.

.. autoclass:: aimet_torch.v1.quantsim.QuantizationSimModel

**The following API can be used to Compute encodings for calibration:**

.. automethod:: aimet_torch.v1.quantsim.QuantizationSimModel.compute_encodings

**The following APIs can be used to save and restore the quantized model**

.. automethod:: aimet_torch.v1.quantsim.save_checkpoint

.. automethod:: aimet_torch.v1.quantsim.load_checkpoint

**The following API can be used to export the quantized model to target:**

.. automethod:: aimet_torch.v1.quantsim.QuantizationSimModel.export

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
