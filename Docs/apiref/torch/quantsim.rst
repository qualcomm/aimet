.. _apiref-torch-quantsim:

####################
aimet_torch.quantsim
####################

..
  # start-after

.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API.

.. autoclass:: aimet_torch.quantsim.QuantizationSimModel

**The following API can be used to Compute encodings for calibration:**

.. automethod:: aimet_torch.quantsim.QuantizationSimModel.compute_encodings

**The following APIs can be used to save and restore the quantized model**

.. automethod:: aimet_torch.quantsim.save_checkpoint

.. automethod:: aimet_torch.quantsim.load_checkpoint

**The following API can be used to export the quantized model to target:**

.. automethod:: aimet_torch.quantsim.QuantizationSimModel.export

Enum Definition
===============

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
