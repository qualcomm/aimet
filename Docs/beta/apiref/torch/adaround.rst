.. _apiref-torch-adaround:

##########################
AIMET PyTorch AdaRound API
##########################


Top-level API
=============

.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API.
    To learn more about the differences between :mod:`aimet_torch` and :mod:`aimet_torch.v2`, see the
    QuantSim v2 Overview.

.. autofunction:: aimet_torch.v1.adaround.adaround_weight.Adaround.apply_adaround


Adaround Parameters
===================

.. autoclass:: aimet_torch.v1.adaround.adaround_weight.AdaroundParameters
    :members:


Enum Definition
===============

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members: