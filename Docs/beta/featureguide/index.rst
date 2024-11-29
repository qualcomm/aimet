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
    Quantization analyzer <quant_analyzer>
    Visualization <visualization>
    Weight SVD <weight_svd>
    Spatial SVD <spatial_svd>
    Channel pruning <channel_pruning>


:ref:`Adaptive rounding (Adaround) <featureguide-adaround>`
===========================================================

Uses training data to improve accuracy over naïve rounding.

:ref:`Sequential MSE <featureguide-seq-mse>`
============================================

tbd

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

:ref:`Quantization analyzer (QuantAnalzer) <featureguide-quant-analyzer>`
=========================================================================

Automatically identify sensitive areas and hotspots in the model.

:ref:`Visualization <featureguide-visualization>`
=================================================

Automatically identify sensitive areas and hotspots in the model.

:ref:`Weight singular value decomposition (Weight SVD) <featureguide-weight-svd>`
=================================================================================

Decomposes one large 2D convolution or fully connected layer into two smaller layers.

:ref:`Spatial singular value decomposition (Spatial SVD) <featureguide-spatial-svd>`
====================================================================================

Decomposes one large 2D convolution layer into two smaller layers.

:ref:`Channel pruning (CP) <featureguide-channel-pruning>`
==========================================================

Prunes redundant input channels from 2D convolution layers.
