.. _apiref-torch-quantization:

###########################
aimet_torch.v2.quantization
###########################

.. currentmodule:: aimet_torch.v2.quantization

Quantized tensors
================

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    QuantizedTensorBase
    QuantizedTensor
    DequantizedTensor

.. _api-beta-quantizers:

Quantizers
==========

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    affine.Quantize
    affine.QuantizeDequantize
    float.FloatQuantizeDequantize

Functional APIs
===============

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: function.rst

    affine.quantize
    affine.quantize_dequantize
    affine.dequantize
