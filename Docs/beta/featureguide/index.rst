.. _featureguide--index:

#######################
Optimization Techniques
#######################

.. toctree::
    :hidden:

    Quantization aware training <qat>
    Automatic quantization <autoquant>
    Adaptive rounding <adaround>
    Cross-layer equalization <cle>
    Batch norm re-estimation <bn>
    Quantization analyzer <quant_analyzer>
    Visualization <visualization>
    Weight SVD <weight_svd>
    Spatial SVD <spatial_svd>
    Channel pruning <cp>

:ref:`Quantization aware training (QAT) <featureguide-qat>`
===========================================================

Fine-tunes the model parameters in the presence of quantization noise.

:ref:`Automatic quantization (AutoQuant) <featureguide-autoquant>`
==================================================================

Analyzes the model, determines the best sequence of AIMET post-training quantization techniques, and applies these techniques.

:ref:`Adaptive rounding (Adaround) <featureguide-adaround>`
===========================================================

Uses training data to improve accuracy over naïve rounding.

:ref:`Cross-layer equalization (CLE) <featureguide-cle>`
===================================================

Scales the parameter ranges across different channels to increase the range for layers with low range and reduce range for layers with high range, enabling the same quantizaion parameters to be used across all channels.

:ref:`Batch norm re-estimation (BN) <featureguide-bn>`
=================================================

Re-estimated statistics are used to adjust the quantization scale parameters of preceeding Convolution or Linear layers, effectively folding the BN layers.

:ref:`Quantization analyzer (QuantAnalzer) <featureguide-quant-analyzer>`
====================================================================

Automatically identify sensitive areas and hotspots in the model.

:ref:`Visualization <featureguide-visualization>`
============================================

Automatically identify sensitive areas and hotspots in the model.

:ref:`Weight singular value decomposition (Weight SVD) <featureguide-weight-svd>`
============================================================================

Decomposes one large MAC or memory layer into two smaller layers.

:ref:`Spatial singular value decomposition (Spatial SVD) <featureguide-spatial-svd>`
===============================================================================

Decomposes one large convolution (Conv) MAC or memory layer into two smaller layers.

:ref:`Channel pruning (CP) <featureguide-cp>`
========================================

Removes less-important input channels from 2D convolution layers.
