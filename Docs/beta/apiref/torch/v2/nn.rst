.. _apiref-torch-nn:

#################
aimet_torch.v2.nn
#################

.. currentmodule:: aimet_torch.v2.nn

Quantized modules
=================

To simulate the effects of running networks at a reduced bitwidth, AIMET introduced `quantized modules`, the extension of
standard torch.nn.Modules with some extra capabilities for quantization.
These quantized modules serve as drop-in replacements for their PyTorch counterparts, but can
hold :ref:`input, output, and parameter quantizers<api-beta-quantizers>` to perform quantization operations during the
module's forward pass and compute quantization encodings.

More specifically, a quantized module inherits both from :class:`QuantizationMixin` and a native torch.nn.Module type,
typically with "Quantized-" prefix prepended to the original class name, such as :class:`QuantizedConv2d` for torch.nn.Conv2d or :class:`QuantizedSoftmax` for torch.nn.Softmax.

For the full list of all built-in quantized modules in AIMET, see :ref:`api-built-in-quantized-modules`


.. autoclass:: QuantizationMixin
   :noindex:
   :members: __quant_init__, forward, compute_encodings

Configuration
=============

The quantization behavior of a quantized module is controlled by the :ref:`quantizers<api-beta-quantizers>` contained within the input, output,
and parameter quantizer attributes listed below.

===================== ====================== ========================================
Attribute             Type                   Description
===================== ====================== ========================================
``input_quantizers``  torch.nn.ModuleList    List of quantizers for input tensors
``param_quantizers``  torch.nn.ModuleDict    Dict mapping parameter names to quantizers
``output_quantizers`` torch.nn.ModuleList    List of quantizers for output tensors
===================== ====================== ========================================

By assigning and configuring quantizers to these structures, we define the type of quantization applied to the corresponding
input index, output index, or parameter name. By default, all the quantizers are set to `None`, meaning that no quantization
will be applied to the respective tensor.

Example: Create a linear layer which performs only per-channel weight quantization
    >>> import aimet_torch.v2 as aimet
    >>> import aimet_torch.quantization as Q
    >>> qlinear = aimet.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> # Per-channel weight quantization is performed over the `out_features` dimension, so encodings are shape (10, 1)
    >>> per_channel_quantizer = Q.affine.QuantizeDequantize(shape=(10, 1), bitwidth=8, symmetric=True)
    >>> qlinear.param_quantizers["weight"] = per_channel_quantizer

Example: Create an elementwise multiply layer which quantizes only the output and the second input
    >>> qmul = aimet.nn.custom.QuantizedMultiply()
    >>> qmul.output_quantizers[0] = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    >>> qmul.input_quantizers[1] = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)

In some cases, it may make sense for multiple tensors to share the same quantizer. In this case, we can assign the same
quantizer to multiple indices.

Example: Create an elementwise add layer which shares the same quantizer between its inputs
    >>> qadd = aimet.nn.custom.QuantizedAdd()
    >>> quantizer = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    >>> qadd.input_quantizers[0] = quantizer
    >>> qadd.input_quantizers[1] = quantizer

Computing encodings
===================

Before a module can compute a quantized forward pass, all quantizers must first be calibrated inside a `compute_encodings`
context. When a quantized module enters the `compute_encodings` context, it first disables all input and output quantization
while the quantizers observe the statistics of the activation tensors passing through them. Upon exiting the context,
the quantizers calculate appropriate quantization encodings based on these statistics (exactly *how* the encodings are
computed is determined by each quantizer's :ref:`encoding analyzer<api-torch-encoding-analyzer>`).

Example:
    >>> qlinear = aimet.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> qlinear.output_quantizers[0] = Q.affine.QuantizeDequantize((1, ), bitwidth=8, symmetric=False)
    >>> qlinear.param_quantizers[0] = Q.affine.QuantizeDequantize((10, 1), bitwidth=8, symmetric=True)
    >>> with qlinear.compute_encodings():
    ...     # Pass several samples through the layer to ensure representative statistics
    ...     for x, _ in calibration_data_loader:
    ...         qlinear(x)
    >>> print(qlinear.output_quantizers[0].is_initialized())
    True
    >>> print(qlinear.param_quantizers["weight"].is_initialized())
    True


.. _api-built-in-quantized-modules:

Built-in quantized modules
==========================

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    QuantizationMixin
    QuantizedAdaptiveAvgPool1d
    QuantizedAdaptiveAvgPool2d
    QuantizedAdaptiveAvgPool3d
    QuantizedAdaptiveMaxPool1d
    QuantizedAdaptiveMaxPool2d
    QuantizedAdaptiveMaxPool3d
    QuantizedAlphaDropout
    QuantizedAvgPool1d
    QuantizedAvgPool2d
    QuantizedAvgPool3d
    QuantizedBCELoss
    QuantizedBCEWithLogitsLoss
    QuantizedBatchNorm1d
    QuantizedBatchNorm2d
    QuantizedBatchNorm3d
    QuantizedBilinear
    QuantizedCELU
    QuantizedCTCLoss
    QuantizedChannelShuffle
    QuantizedCircularPad1d
    QuantizedCircularPad2d
    QuantizedCircularPad3d
    QuantizedConstantPad1d
    QuantizedConstantPad2d
    QuantizedConstantPad3d
    QuantizedConv1d
    QuantizedConv2d
    QuantizedConv3d
    QuantizedConvTranspose1d
    QuantizedConvTranspose2d
    QuantizedConvTranspose3d
    QuantizedCosineEmbeddingLoss
    QuantizedCosineSimilarity
    QuantizedCrossEntropyLoss
    QuantizedDropout
    QuantizedDropout1d
    QuantizedDropout2d
    QuantizedDropout3d
    QuantizedELU
    QuantizedEmbedding
    QuantizedEmbeddingBag
    QuantizedFeatureAlphaDropout
    QuantizedFlatten
    QuantizedFold
    QuantizedFractionalMaxPool2d
    QuantizedFractionalMaxPool3d
    QuantizedGELU
    QuantizedGLU
    QuantizedGRU
    QuantizedGRUCell
    QuantizedGaussianNLLLoss
    QuantizedGroupNorm
    QuantizedHardshrink
    QuantizedHardsigmoid
    QuantizedHardswish
    QuantizedHardtanh
    QuantizedHingeEmbeddingLoss
    QuantizedHuberLoss
    QuantizedInstanceNorm1d
    QuantizedInstanceNorm2d
    QuantizedInstanceNorm3d
    QuantizedKLDivLoss
    QuantizedL1Loss
    QuantizedLPPool1d
    QuantizedLPPool2d
    QuantizedLSTM
    QuantizedLSTMCell
    QuantizedLayerNorm
    QuantizedLeakyReLU
    QuantizedLinear
    QuantizedLocalResponseNorm
    QuantizedLogSigmoid
    QuantizedLogSoftmax
    QuantizedMSELoss
    QuantizedMarginRankingLoss
    QuantizedMaxPool1d
    QuantizedMaxPool2d
    QuantizedMaxPool3d
    QuantizedMaxUnpool1d
    QuantizedMaxUnpool2d
    QuantizedMaxUnpool3d
    QuantizedMish
    QuantizedMultiLabelMarginLoss
    QuantizedMultiLabelSoftMarginLoss
    QuantizedMultiMarginLoss
    QuantizedNLLLoss
    QuantizedNLLLoss2d
    QuantizedPReLU
    QuantizedPairwiseDistance
    QuantizedPixelShuffle
    QuantizedPixelUnshuffle
    QuantizedPoissonNLLLoss
    QuantizedRNN
    QuantizedRNNCell
    QuantizedRReLU
    QuantizedReLU
    QuantizedReLU6
    QuantizedReflectionPad1d
    QuantizedReflectionPad2d
    QuantizedReflectionPad3d
    QuantizedReplicationPad1d
    QuantizedReplicationPad2d
    QuantizedReplicationPad3d
    QuantizedSELU
    QuantizedSiLU
    QuantizedSigmoid
    QuantizedSmoothL1Loss
    QuantizedSoftMarginLoss
    QuantizedSoftmax
    QuantizedSoftmax2d
    QuantizedSoftmin
    QuantizedSoftplus
    QuantizedSoftshrink
    QuantizedSoftsign
    QuantizedTanh
    QuantizedTanhshrink
    QuantizedThreshold
    QuantizedTripletMarginLoss
    QuantizedTripletMarginWithDistanceLoss
    QuantizedUnflatten
    QuantizedUnfold
    QuantizedUpsample
    QuantizedUpsamplingBilinear2d
    QuantizedUpsamplingNearest2d
    QuantizedZeroPad1d
    QuantizedZeroPad2d
    QuantizedZeroPad3d
