.. _featureguide-index:

#####################################
Post Training Quantization Techniques
#####################################

.. toctree::
    :hidden:

    Adaptive rounding <adaround>
    Sequential MSE <seq_mse>
    Batch norm folding <bnf>
    Cross-layer equalization <cle>
    AdaScale <adascale>
    Batch norm re-estimation <bn>
    Quantized LoRa <quantized LoRa/index>
    OmniQuant <omniquant>
    Automatic quantization <autoquant>
    SpinQuant <spinquant>

:ref:`Adaptive rounding <ptq-adaround>`
================================================

Uses training data to improve accuracy over naïve rounding.

:ref:`Sequential MSE <ptq-seq-mse>`
============================================

Sequential MSE (SeqMSE) is a method that searches for optimal quantization encodings per operation
(i.e. per layer) such that the difference between the original output activation and the
corresponding quantization-aware output activation is minimized.

:ref:`Batch norm folding <ptq-bnf>`
============================================

Folds BN layers into adjacent Convolution or Linear layers.

:ref:`Cross-layer equalization <ptq-cle>`
==================================================

Scales the parameter ranges across different channels to increase the range for layers with low range and reduce range for layers with high range, enabling the same quantization parameters to be used across all channels.

:ref:`AdaScale <ptq-adascale>`
==================================================
AdaScale is a PTQ technique to improve accuracy of the quantized model by introducing learnable parameters in the weight quantizers and by performing BKD(Blockwise Knowledge Distillation) with respect to the corresponding FP output.


:ref:`Batch norm re-estimation <ptq-bn-estimate>`
=================================================

Re-estimated statistics are used to adjust the quantization scale parameters of preceding convolution or linear layers, effectively folding the BN layers.


:ref:`Quantized LoRa <ptq-lora>`
===================================================

Workflows to perform LoRa (Low-Rank Adaptation) on quantized large models.

:ref:`OmniQuant <ptq-omniquant>`
==================================================
OmniQuant is a PTQ technique to improve accuracy of the quantized model by introducing learnable parameter (scale) in the weight quantizers and by performing BKD(Blockwise Knowledge Distillation) with respect to the corresponding FP output.

:ref:`Automatic quantization <ptq-autoquant>`
======================================================

Analyzes the model, determines the best sequence of AIMET post-training quantization (PTQ) techniques, and applies these techniques.

:ref:`SpinQuant <ptq-spinquant>`
======================================================

SpinQuant is a PTQ technique which improves the accuracy of the quantized model by inserting rotations at specific points in the model to help with outliers in activation quantization.
