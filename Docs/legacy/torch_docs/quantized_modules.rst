.. _api-torch-quantized-modules:

.. currentmodule:: aimet_torch.v2.nn

=================
Quantized Modules
=================

To simulate the effects of running networks at a reduced bitwidth, AIMET introduced `quantized modules`, the extension of
standard torch.nn.Modules with some extra capabilities for quantization.
These quantized modules serve as drop-in replacements for their PyTorch counterparts, but can
hold input, output, and parameter :ref:`quantizers<api-torch-quantizers>` to perform quantization operations during the
module's forward pass and compute quantization encodings.

More specifically, a quantized module inherits both from :ref:`QuantizationMixin<api-torch-quantization-mixin-summary>` and a native torch.nn.Module type,
typically with "Quantized-" prefix prepended to the original class name, such as QuantizedConv2d for torch.nn.Conv2d or QuantizedSoftmax for torch.nn.Softmax.
For more detailed API reference of QuantizationMixin class, see :ref:`QuantizationMixin API reference<api-torch-quantization-mixin>`.
For the full list of all built-in quantized modules in AIMET, see :ref:`api-quantized-module-class-table`


Top-level API
=============

.. _api-torch-quantization-mixin-summary:
.. autoclass:: aimet_torch.v2.nn.QuantizationMixin
   :members: __quant_init__, forward, compute_encodings

Configuration
=============

The quantization behavior of a quantized module is controlled by the :ref:`quantizers<api-torch-quantizers>` contained within the input, output,
and parameter quantizer attributes listed below.

================= ====================== ========================================
Attribute         Type                   Description
================= ====================== ========================================
input_quantizers  torch.nn.ModuleList    List of quantizers for input tensors
param_quantizers  torch.nn.ModuleDict    Dict mapping parameter names to quantizers
output_quantizers torch.nn.ModuleList    List of quantizers for output tensors
================= ====================== ========================================

By assigning and configuring :ref:`quantizers<api-torch-quantizers>` to these structures, we define the type of quantization applied to the corresponding
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

Computing Encodings
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


.. _api-quantized-module-class-table:

Quantized Module Classes
========================

============================================= =========================================
nn.Module                                     QuantizationMixin
============================================= =========================================
torch.nn.AdaptiveAvgPool1d                    QuantizedAdaptiveAvgPool1d
torch.nn.AdaptiveAvgPool2d                    QuantizedAdaptiveAvgPool2d
torch.nn.AdaptiveAvgPool3d                    QuantizedAdaptiveAvgPool3d
torch.nn.AdaptiveMaxPool1d                    QuantizedAdaptiveMaxPool1d
torch.nn.AdaptiveMaxPool2d                    QuantizedAdaptiveMaxPool2d
torch.nn.AdaptiveMaxPool3d                    QuantizedAdaptiveMaxPool3d
torch.nn.AlphaDropout                         QuantizedAlphaDropout
torch.nn.AvgPool1d                            QuantizedAvgPool1d
torch.nn.AvgPool2d                            QuantizedAvgPool2d
torch.nn.AvgPool3d                            QuantizedAvgPool3d
torch.nn.BatchNorm1d                          QuantizedBatchNorm1d
torch.nn.BatchNorm2d                          QuantizedBatchNorm2d
torch.nn.BatchNorm3d                          QuantizedBatchNorm3d
torch.nn.CELU                                 QuantizedCELU
torch.nn.ChannelShuffle                       QuantizedChannelShuffle
torch.nn.ConstantPad1d                        QuantizedConstantPad1d
torch.nn.ConstantPad2d                        QuantizedConstantPad2d
torch.nn.ConstantPad3d                        QuantizedConstantPad3d
torch.nn.Conv1d                               QuantizedConv1d
torch.nn.Conv2d                               QuantizedConv2d
torch.nn.Conv3d                               QuantizedConv3d
torch.nn.ConvTranspose1d                      QuantizedConvTranspose1d
torch.nn.ConvTranspose2d                      QuantizedConvTranspose2d
torch.nn.ConvTranspose3d                      QuantizedConvTranspose3d
torch.nn.Dropout                              QuantizedDropout
torch.nn.Dropout2d                            QuantizedDropout2d
torch.nn.Dropout3d                            QuantizedDropout3d
torch.nn.ELU                                  QuantizedELU
torch.nn.FeatureAlphaDropout                  QuantizedFeatureAlphaDropout
torch.nn.Flatten                              QuantizedFlatten
torch.nn.Fold                                 QuantizedFold
torch.nn.FractionalMaxPool2d                  QuantizedFractionalMaxPool2d
torch.nn.FractionalMaxPool3d                  QuantizedFractionalMaxPool3d
torch.nn.GELU                                 QuantizedGELU
torch.nn.GLU                                  QuantizedGLU
torch.nn.GroupNorm                            QuantizedGroupNorm
torch.nn.Hardshrink                           QuantizedHardshrink
torch.nn.Hardsigmoid                          QuantizedHardsigmoid
torch.nn.Hardswish                            QuantizedHardswish
torch.nn.Hardtanh                             QuantizedHardtanh
torch.nn.InstanceNorm1d                       QuantizedInstanceNorm1d
torch.nn.InstanceNorm2d                       QuantizedInstanceNorm2d
torch.nn.InstanceNorm3d                       QuantizedInstanceNorm3d
torch.nn.LPPool1d                             QuantizedLPPool1d
torch.nn.LPPool2d                             QuantizedLPPool2d
torch.nn.LayerNorm                            QuantizedLayerNorm
torch.nn.LeakyReLU                            QuantizedLeakyReLU
torch.nn.Linear                               QuantizedLinear
torch.nn.LocalResponseNorm                    QuantizedLocalResponseNorm
torch.nn.LogSigmoid                           QuantizedLogSigmoid
torch.nn.LogSoftmax                           QuantizedLogSoftmax
torch.nn.MaxPool1d                            QuantizedMaxPool1d
torch.nn.MaxPool2d                            QuantizedMaxPool2d
torch.nn.MaxPool3d                            QuantizedMaxPool3d
torch.nn.MaxUnpool1d                          QuantizedMaxUnpool1d
torch.nn.MaxUnpool2d                          QuantizedMaxUnpool2d
torch.nn.MaxUnpool3d                          QuantizedMaxUnpool3d
torch.nn.Mish                                 QuantizedMish
torch.nn.PReLU                                QuantizedPReLU
torch.nn.PixelShuffle                         QuantizedPixelShuffle
torch.nn.PixelUnshuffle                       QuantizedPixelUnshuffle
torch.nn.RReLU                                QuantizedRReLU
torch.nn.ReLU                                 QuantizedReLU
torch.nn.ReLU6                                QuantizedReLU6
torch.nn.ReflectionPad1d                      QuantizedReflectionPad1d
torch.nn.ReflectionPad2d                      QuantizedReflectionPad2d
torch.nn.ReplicationPad1d                     QuantizedReplicationPad1d
torch.nn.ReplicationPad2d                     QuantizedReplicationPad2d
torch.nn.ReplicationPad3d                     QuantizedReplicationPad3d
torch.nn.SELU                                 QuantizedSELU
torch.nn.SiLU                                 QuantizedSiLU
torch.nn.Sigmoid                              QuantizedSigmoid
torch.nn.Softmax                              QuantizedSoftmax
torch.nn.Softmax2d                            QuantizedSoftmax2d
torch.nn.Softmin                              QuantizedSoftmin
torch.nn.Softplus                             QuantizedSoftplus
torch.nn.Softshrink                           QuantizedSoftshrink
torch.nn.Softsign                             QuantizedSoftsign
torch.nn.Tanh                                 QuantizedTanh
torch.nn.Tanhshrink                           QuantizedTanhshrink
torch.nn.Threshold                            QuantizedThreshold
torch.nn.Unflatten                            QuantizedUnflatten
torch.nn.Unfold                               QuantizedUnfold
torch.nn.Upsample                             QuantizedUpsample
torch.nn.UpsamplingBilinear2d                 QuantizedUpsamplingBilinear2d
torch.nn.UpsamplingNearest2d                  QuantizedUpsamplingNearest2d
torch.nn.ZeroPad2d                            QuantizedZeroPad2d
torch.nn.BCELoss                              QuantizedBCELoss
torch.nn.BCEWithLogitsLoss                    QuantizedBCEWithLogitsLoss
torch.nn.Bilinear                             QuantizedBilinear
torch.nn.CTCLoss                              QuantizedCTCLoss
torch.nn.CosineSimilarity                     QuantizedCosineSimilarity
torch.nn.CrossEntropyLoss                     QuantizedCrossEntropyLoss
torch.nn.HingeEmbeddingLoss                   QuantizedHingeEmbeddingLoss
torch.nn.HuberLoss                            QuantizedHuberLoss
torch.nn.KLDivLoss                            QuantizedKLDivLoss
torch.nn.L1Loss                               QuantizedL1Loss
torch.nn.MSELoss                              QuantizedMSELoss
torch.nn.MultiLabelMarginLoss                 QuantizedMultiLabelMarginLoss
torch.nn.MultiLabelSoftMarginLoss             QuantizedMultiLabelSoftMarginLoss
torch.nn.MultiMarginLoss                      QuantizedMultiMarginLoss
torch.nn.NLLLoss                              QuantizedNLLLoss
torch.nn.NLLLoss2d                            QuantizedNLLLoss2d
torch.nn.PairwiseDistance                     QuantizedPairwiseDistance
torch.nn.PoissonNLLLoss                       QuantizedPoissonNLLLoss
torch.nn.SmoothL1Loss                         QuantizedSmoothL1Loss
torch.nn.SoftMarginLoss                       QuantizedSoftMarginLoss
torch.nn.CosineEmbeddingLoss                  QuantizedCosineEmbeddingLoss
torch.nn.GaussianNLLLoss                      QuantizedGaussianNLLLoss
torch.nn.MarginRankingLoss                    QuantizedMarginRankingLoss
torch.nn.TripletMarginLoss                    QuantizedTripletMarginLoss
torch.nn.TripletMarginWithDistanceLoss        QuantizedTripletMarginWithDistanceLoss
torch.nn.Embedding                            QuantizedEmbedding
torch.nn.EmbeddingBag                         QuantizedEmbeddingBag
torch.nn.GRU                                  QuantizedGRU
torch.nn.RNN                                  QuantizedRNN
torch.nn.GRUCell                              QuantizedGRUCell
torch.nn.RNNCell                              QuantizedRNNCell
torch.nn.LSTM                                 QuantizedLSTM
torch.nn.LSTMCell                             QuantizedLSTMCell
aimet_torch.v2.nn.custom.AvgPool2d            QuantizedAvgPool2d
aimet_torch.v2.nn.custom.CumSum               QuantizedCumSum
aimet_torch.v2.nn.custom.Sin                  QuantizedSin
aimet_torch.v2.nn.custom.Cos                  QuantizedCos
aimet_torch.v2.nn.custom.RSqrt                QuantizedRSqrt
aimet_torch.v2.nn.custom.Reshape              QuantizedReshape
aimet_torch.v2.nn.custom.MatMul               QuantizedMatMul
aimet_torch.v2.nn.custom.Add                  QuantizedAdd
aimet_torch.v2.nn.custom.Multiply             QuantizedMultiply
aimet_torch.v2.nn.custom.Subtract             QuantizedSubtract
aimet_torch.v2.nn.custom.Divide               QuantizedDivide
aimet_torch.v2.nn.custom.Bmm                  QuantizedBmm
aimet_torch.v2.nn.custom.Baddbmm              QuantizedBaddbmm
aimet_torch.v2.nn.custom.Addmm                QuantizedAddmm
aimet_torch.v2.nn.custom.Concat               QuantizedConcat
============================================= =========================================

..
    aimet_torch.v2.nn.custom.ChannelShuffle       QuantizedChannelShuffle
    aimet_torch.v2.nn.custom.MaxPool2d            QuantizedMaxPool2d
    aimet_torch.v2.nn.custom.AdaptiveAvgPool2d    QuantizedAdaptiveAvgPool2d
    aimet_torch.v2.nn.custom.Cast                 QuantizedCast
    aimet_torch.v2.nn.custom.DepthToSpaceDCRMode  QuantizedDepthToSpaceDCRMode
    aimet_torch.v2.nn.custom.OneHot               QuantizedOneHot
    aimet_torch.v2.nn.custom.Exponential          QuantizedExponential
    aimet_torch.v2.nn.custom.Erf                  QuantizedErf
    aimet_torch.v2.nn.custom.Sqrt                 QuantizedSqrt
    aimet_torch.v2.nn.custom.Log                  QuantizedLog
    aimet_torch.v2.nn.custom.Abs                  QuantizedAbs
    aimet_torch.v2.nn.custom.Neg                  QuantizedNeg
    aimet_torch.v2.nn.custom.ElementwiseCeil      QuantizedElementwiseCeil
    aimet_torch.v2.nn.custom.ElementwiseFloor     QuantizedElementwiseFloor
    aimet_torch.v2.nn.custom.Asin                 QuantizedAsin
    aimet_torch.v2.nn.custom.Atan                 QuantizedAtan
    aimet_torch.v2.nn.custom.Round                QuantizedRound
    aimet_torch.v2.nn.custom.LogicalNot           QuantizedLogicalNot
    aimet_torch.v2.nn.custom.NonZero              QuantizedNonZero
    aimet_torch.v2.nn.custom.ElementwiseUnarySign QuantizedElementwiseUnarySign
    aimet_torch.v2.nn.custom.Square               QuantizedSquare
    aimet_torch.v2.nn.custom.Mean                 QuantizedMean
    aimet_torch.v2.nn.custom.Sum                  QuantizedSum
    aimet_torch.v2.nn.custom.Prod                 QuantizedProd
    aimet_torch.v2.nn.custom.Argmin               QuantizedArgmin
    aimet_torch.v2.nn.custom.Argmax               QuantizedArgmax
    aimet_torch.v2.nn.custom.Gather               QuantizedGather
    aimet_torch.v2.nn.custom.RoiAlign             QuantizedRoiAlign
    aimet_torch.v2.nn.custom.Permute              QuantizedPermute
    aimet_torch.v2.nn.custom.IndexSelect          QuantizedIndexSelect
    aimet_torch.v2.nn.custom.TopK                 QuantizedTopK
    aimet_torch.v2.nn.custom.Tile                 QuantizedTile
    aimet_torch.v2.nn.custom.Norm                 QuantizedNorm
    aimet_torch.v2.nn.custom.Interpolate          QuantizedInterpolate
    aimet_torch.v2.nn.custom.Normalize            QuantizedNormalize
    aimet_torch.v2.nn.custom.Pad                  QuantizedPad
    aimet_torch.v2.nn.custom.Shape                QuantizedShape
    aimet_torch.v2.nn.custom.Expand               QuantizedExpand
    aimet_torch.v2.nn.custom.StridedSlice         QuantizedStridedSlice
    aimet_torch.v2.nn.custom.FloorDivide          QuantizedFloorDivide
    aimet_torch.v2.nn.custom.Greater              QuantizedGreater
    aimet_torch.v2.nn.custom.Less                 QuantizedLess
    aimet_torch.v2.nn.custom.GreaterEqual         QuantizedGreaterEqual
    aimet_torch.v2.nn.custom.LessEqual            QuantizedLessEqual
    aimet_torch.v2.nn.custom.NotEqual             QuantizedNotEqual
    aimet_torch.v2.nn.custom.Equal                QuantizedEqual
    aimet_torch.v2.nn.custom.Remainder            QuantizedRemainder
    aimet_torch.v2.nn.custom.Fmod                 QuantizedFmod
    aimet_torch.v2.nn.custom.Pow                  QuantizedPow
    aimet_torch.v2.nn.custom.CustomSiLU           QuantizedCustomSiLU
    aimet_torch.v2.nn.custom.Maximum              QuantizedMaximum
    aimet_torch.v2.nn.custom.Max                  QuantizedMax
    aimet_torch.v2.nn.custom.Minimum              QuantizedMinimum
    aimet_torch.v2.nn.custom.Min                  QuantizedMin
    aimet_torch.v2.nn.custom.LogicalOr            QuantizedLogicalOr
    aimet_torch.v2.nn.custom.LogicalAnd           QuantizedLogicalAnd
    aimet_torch.v2.nn.custom.CustomGather         QuantizedCustomGather
    aimet_torch.v2.nn.custom.GatherNd             QuantizedGatherNd
    aimet_torch.v2.nn.custom.ScatterND            QuantizedScatterND
    aimet_torch.v2.nn.custom.DynamicConv2d        QuantizedDynamicConv2d
    aimet_torch.v2.nn.custom.ScatterElements      QuantizedScatterElements
    aimet_torch.v2.nn.custom.BatchNorm            QuantizedBatchNorm
    aimet_torch.v2.nn.custom.GroupNorm            QuantizedAimetGroupNorm
    aimet_torch.v2.nn.custom.NonMaxSuppression    QuantizedNonMaxSuppression
    aimet_torch.v2.nn.custom.Split                QuantizedSplit
    aimet_torch.v2.nn.custom.Where                QuantizedWhere
    aimet_torch.v2.nn.custom.MaskedFill           QuantizedMaskedFill
