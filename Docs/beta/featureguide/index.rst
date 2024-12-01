.. _featureguide-index:

#######################
Optimization Techniques
#######################

.. toctree::
    :hidden:

    Adaptive rounding (Adaround) <adaround>
    Sequential MSE <seq_mse>
    Low power blockwise quantization (LPBQ) <lpbq>
    Batch norm folding <bnf>
    Cross-layer equalization (CLE) <cle>
    Quantization aware training (QAT) <qat>
    Automatic quantization (AutoQuant) <autoquant>
    Batch norm re-estimation <bn>
    Analysis tools <analysis tools/index>
    Compression <compression/index>

:ref:`Adaptive rounding (Adaround) <featureguide-adaround>`
===========================================================

Uses training data to improve accuracy over naïve rounding.

:ref:`Sequential MSE <featureguide-seq-mse>`
============================================

Sequential MSE (SeqMSE) is a method that searches for optimal quantization encodings per operation
(i.e. per layer) such that the difference between the original output activation and the
corresponding quantization-aware output activation is minimized.

:ref:`Low power blockwise quantization <featureguide-lpbq>`
===========================================================

tbd

:ref:`Batch norm folding (BNF) <featureguide-bnf>`
==================================================

Folds BN layers into adjacent Convolution or Linear layers.

:ref:`Cross-layer equalization (CLE) <featureguide-cle>`
=========================================================

Scales the parameter ranges across different channels to increase the range for layers with low range and reduce range for layers with high range, enabling the same quantizaion parameters to be used across all channels.

:ref:`Quantization aware training (QAT) <featureguide-qat>`
===========================================================

Fine-tunes the model parameters in the presence of quantization noise.

:ref:`Automatic quantization (AutoQuant) <featureguide-autoquant>`
==================================================================

Analyzes the model, determines the best sequence of AIMET post-training quantization (PTQ) techniques, and applies these techniques.


:ref:`Batch norm re-estimation (BN) <featureguide-bn>`
======================================================

Re-estimated statistics are used to adjust the quantization scale parameters of preceding convolution or linear layers, effectively folding the BN layers.

:ref:`Analysis tools <featureguide-analysis-tools-index>`
=========================================================

Analysis tools to automatically identify sensitive areas and hotspots in your pre-trained model.

:ref:`Compression <featureguide-compression-index>`
===================================================

Reduces pre-trained model’s Multiply-accumulate(MAC) and memory costs with a minimal drop in accuracy.
AIMET supports various compression techniques like Weight SVD, Spatial SVD and Channel pruning.

