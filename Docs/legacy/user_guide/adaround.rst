.. _ug-adaround:


##############
AIMET AdaRound
##############

By default, AIMET uses *nearest rounding* for quantization. A single weight value in a weight tensor is illustrated in the following figure. In nearest rounding, this weight value is quantized to the nearest integer value.

The Adaptive Rounding (AdaRound) feature uses a subset of the unlabeled training data to adaptively round weights. In the following figure, the weight value is quantized to the integer value far from it.

.. image:: ../images/adaround.png
    :width: 900px

AdaRound optimizes a loss function using the unlabelled training data to decide whether to quantize a weight to the closer or further integer value. AdaRound quantization achieves accuracy closer to the FP32 model, while using low bit-width integer quantization.

When creating a QuantizationSimModel using AdaRounded, use the QuantizationSimModel provided in the API to set and freeze parameter encodings before computing the encodings. Refer the code example in the AdaRound API.

AdaRound use cases
==================

**Terminology**

The following abbreviations are used in the following use case descriptions:

BC
 Bias Correction
BNF
 Batch Norm Folding
CLE
 Cross Layer Equalization
HBF
  High Bias Folding
QAT
 Quantization Aware Training
{ }
 An optional step in the use case

**Recommended**

The following sequences are recommended:

 #. {BNF} --> {CLE} --> AdaRound
       Applying BNF and CLE are optional steps before applying AdaRound. Some models benefit from applying CLE while some don't.

 #. AdaRound --> QAT
       AdaRound is a post-training quantization feature, but for some models applying BNF and CLE may not help. For these models, applying AdaRound before QAT might help. AdaRound is a better weights initialization step that speeds up QAT.

**Not recommended**

Applying bias correction (BC) either before or after AdaRound is *not* recommended.

 #. AdaRound --> BC

 #. BC --> AdaRound

AdaRound hyper parameters guidelines
=====================================

A number of hyper parameters used during AdaRound optimization are exposed to users. The default values of some of these parameters lead to stable, good results over many models; we recommend that you not change these.

Use the following guideline for adjusting hyper parameters with AdaRound.

* Hyper Parameters to be changed often
    * Number of batches (approximately 500-1000 images. If batch size of data loader is 64, then 16x the    number of batches leads to 1024 images)
    * Number of iterations(default 10000)

* Hyper Parameters to change with caution
    * Regularization parameter (default 0.01)

* Hyper Parameters to avoid changing
    * Beta range (default (20, 2))
    * Warm start period (default 20%)

AdaRound API
============

See the AdaRound API variant for your platform:

- :ref:`AdaRound for PyTorch<api-torch-adaround>`
- :ref:`AdaRound for Keras<api-keras-adaround>`
- :ref:`AdaRound for ONNX<api-onnx-adaround>`

