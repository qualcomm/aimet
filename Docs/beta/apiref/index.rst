.. _apiref-index:

#########
AIMET API
#########

.. toctree::
    :hidden:

    Quantization aware training <torch/qat>
    Automatic quantization <torch/autoquant>
    Adaptive rounding <torch/adaround>
    Cross-layer equalization <torch/cle>
    Batch norm re-estimation <bn>
    Quantization analyzer <torch/quant_analyzer>
    Visualization <torch/visualization>
    Weight SVD <torch/weight_svd>
    Spatial SVD <torch/spatial_svd>
    Channel pruning <torch/cp>

Quantization aware training (QAT) 
=================================

Fine-tunes the model parameters in the presence of quantization noise.

:ref:`Quantization aware training (QAT) <apiref-torch-qat>`


Automatic quantization (AutoQuant) 
==================================

Analyzes the model, determines the best sequence of AIMET post-training quantization techniques, and applies these techniques.

:ref:`Automatic quantization (AutoQuant) <apiref-torch-autoquant>`


Adaptive rounding (Adaround) 
============================

Uses training data to improve accuracy over naïve rounding.

:ref:`PyTorch <apiref-torch-adaround>`

:ref:`TensorFlow <apiref-keras-adaround>`

:ref:`ONNX <apiref-onnx-adaround>`


Cross-layer equalization (CLE) 
==============================

Scales the parameter ranges across different channels to increase the range for layers with low range and reduce range for layers with high range, enabling the same quantizaion parameters to be used across all channels.

:ref:`Cross-layer equalization (CLE) <apiref-torch-cle>`


Batch norm re-estimation (BN) 
=============================

Re-estimated statistics are used to adjust the quantization scale parameters of preceeding Convolution or Linear layers, effectively folding the BN layers.

:ref:`Batch norm re-estimation (BN) <apiref-torch-bn>`


Quantization analyzer (QuantAnalzer) 
=====================================

Automatically identify sensitive areas and hotspots in the model.

:ref:`Quantization analyzer (QuantAnalzer) <apiref-torch-quant-analyzer>`


Visualization 
=============

Automatically identify sensitive areas and hotspots in the model.

:ref:`Visualization <apiref-torch-visualization>`


Weight singular value decomposition (Weight SVD) 
============================================================================

Decomposes one large MAC or memory layer into two smaller layers.

:ref:`Weight singular value decomposition (Weight SVD) <apiref-torch-weight-svd>`


Spatial singular value decomposition (Spatial SVD) 
===============================================================================

Decomposes one large convolution (Conv) MAC or memory layer into two smaller layers.

:ref:`Spatial singular value decomposition (Spatial SVD) <apiref-torch-spatial-svd>`


Channel pruning (CP) 
========================================

Removes less-important input channels from 2D convolution layers.

:ref:`Channel pruning (CP) <apiref-torch-cp>`

