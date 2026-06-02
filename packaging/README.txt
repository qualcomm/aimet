========
Overview
========
AI Model Efficiency Toolkit (AIMET) is a library that provides advanced model
quantization and model compression techniques for trained neural network
models. It provides features that have been proven to improve run-time
performance of deep learning neural network models with lower compute and
memory requirements and minimal impact to task accuracy.

Features
========
AIMET supports the following features

- Model Quantization
  - Quantization simulation: Simulates on-target quantized inference.
    Specifically simulates Qualcomm SnapDragon DSP accelerators.
  - Quantization-aware training: Fine-tune models to improve on-target
    quantized accuracy
  - Data Free quantization: Post-training technique to improve quantized
    accuracy by equalizing model weights (Cross-Layer Equalization) and
    correcting shifts in layer outputs due to quantization (Bias Correction)

- Model Compression
  - Spatial SVD: Tensor decomposition technique to split a large layer
    into two smaller ones
  - Channel Pruning: Removes redundant input channels of convolutional
    layers and modifies the model graph accordingly
  - Compression-ratio Selection: Automatically selects per-layer compression
    ratios

============
Dependencies
============
See the https://qualcomm.github.io/aimet-pages/releases/latest/install/index.html for details.

=============
Documentation
=============
Please refer to the Documentation at https://qualcomm.github.io/aimet-pages/index.html
for the user guide and API documentation.

=================
Using the Package
=================
Please see https://github.com/quic/aimet#getting-started for package requirements
and usage.
