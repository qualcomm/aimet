.. include:: abbreviation.txt

.. _rn-index:

#############
Release notes
#############

2.0.0
=====

* New Features
    * Common
        * Reorganized the documentation to more clearly explain AIMET procedures
        * Redesigned the documentation using the `Furo theme <https://sphinx-themes.readthedocs.io/en/latest/sample-sites/furo/>`_
        * Added post-AIMET procedures on how to take AIMET quantized model to |qnn| and |qai_hub|
    * PyTorch
        * BREAKING CHANGE: :mod:`aimet_torch.v2` has become the default API. All the legacy APIs are migrated to :mod:`aimet_torch.v1` subpackage, for example from :mod:`aimet_torch.qc_quantize_op` to :mod:`aimet_torch.v1.qc_quantize_op`
        * Added Manual Mixed Precision Configurator (Beta) to make it easy to configure a model in Mixed Precision.
    * ONNX
        * Optimized :func:`QuantizationSimModel.__init__` latency
        * Align :mod:`ConnectedGraph` representation with onnx graph

* Bug Fixes
    * ONNX
        * Bug fixes for Adaround
        * Bug fixes for BN fold

* Upgrading
    * PyTorch
        * aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x. If you are using low-level components of :class:`QuantizationSimModel`, please see :doc:`Migrate to aimet_torch 2 </apiref/torch/migration_guide>`.

1.35.1
======

* PyTorch
    * Fixed package versioning for compatibility with latest pip version

1.35.0
======

* PyTorch
    * Added support for W16A16 in Autoquant.
* Deprecation Notice
    * Support for Pytorch 1.13 is deprecated. It will be removed in next release.
* ONNX
    * Optimized Memory and Speed utilization (for CPU).

1.34.0
======

* PyTorch
    * Added support for WSL2
    * CUDA version upgraded for Pytorch 2.1
    * Extended QuantAnalyzer functionality for LLM range analysis
* Keras
    * Adds support for certain TFOpLambda layers created by tf functional calls.
* ONNX
    * Upgraded AIMET to support ONNX version 1.16.1 and ONNXRUNTIME version 1.18.1.


1.33.5
======

* PyTorch
    * Various bugfixes/QoL updates for LoRA
    * Updated minimum scale value and registered additional custom quantized ops with QuantSim 2.0

1.33.0
======

* PyTorch
    * Enhancements done in export pipeline for GPU memory optimization with LLMs.
    * [Experimental] Added support for handling of LoRA (via PEFT API) in AIMET. and enabled export of
      required artifacts for QNN.
    * Added examples for training pipeline with for distributed KD-QAT.
    * [Experimental] Added support for block wise quantization (BQ) to support w4fp16 format, and the
      low-power block quantization (LPBQ) to support w4a8 and w4a16 formats. This feature needs
      QuantSim V2.

1.32.0
======

* PyTorch
    * Added MultiGPU support for Adaround.
    * Upgraded AIMET to support PyTorch version 2.1 as a new variant. AIMET with PyTorch version 1.13
      remains the default.
* Keras
    * For models with SeparableConv2D layers, use model_preparer first before applying any quantization
      API.
* Common
    * Upgraded AIMET to support Ubuntu22 and Python3.10 for all AIMET variants.

1.31.0
======

* ONNX
    * Added support for custom ops in QuantSim, CLE, AdaRound and AMP.
    * Added support for Quant Analyzer.
* Keras
    * Added support for unrolled quantized LSTM with only Quantsim in PTQ mode.
    * Fix for ReLU Encoding min going past 0 for QAT.
    * Fixes Input Quantizers for TFOpLambda Layers (kwargs)
    * Fixes logic for placing input quantizers

1.30.0
======

* ONNX
    * Upgraded AIMET to support Onnx version 1.14 and ONNXRUNTIME version 1.15.
    * Added support for AutoQuant.

1.29.0
======

* Keras
    * Fixes issues with TF Op Lambda Layers in Qc Quantize Wrappers call.
* PyTorch
    * [experimental] Support for embedding AIMET encodings within the graph using ONNX quantize/dequantize
      operators. Currently this option is only supported when using 8bit per-tensor quantization.
* ONNX
    * Added support for Adaround.

1.28.0
======

* Keras
    * Added Support for Spatial SVD Compression feature.
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.
* PyTorch
    * Upgraded AIMET Pytorch default version to 1.13. AIMET remains compatible with Pytorch version 1.9.
* ONNX
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.

1.27.0
======

* Keras
    * Update support for TFOpLambda layers in Batch Norm Folding with extra call args/kwargs.
* PyTorch
    * Added AIMET to support PyTorch version 1.13.0. Only ONNX opset 14 is supported for export.
    * [experimental] Debugging APIs have been added for dumping intermediate tensor data. This data can
      be used with current QNN/SNPE tools for debugging accuracy problems. Layer Output Generation API
      gives incorrect tensor data for the layer just before Relu when used for original FP32 model.
    * [experimental] Support for embedding AIMET encodings within the graph using ONNX quantize/dequantize
      operators. Currently this is option is only supported when using 8bit per-tensor quantization.
    * Fixed a bug in AIMET QuantSim for PyTorch models to handle non-contiguous tensors.
* ONNX
    * AIMET support for ONNX 1.11.0 has been added. However there is currently limited op support
      in QNN/SNPE. If the model fails to load please continue to use opset 11 for export.
* TensorFlow
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.

1.26.0
======

* Keras
    * Added a feature called BN Re-estimation that can improve model accuracy after QAT for INT4
      quantization.
    * Updated the AutoQuant feature to automatically choose the optimal calibration scheme, create an
      HTML report on which optimizations were applied.
    * Update to Model Preparer to replace separable conventional with depth wise and point wise conv
      layers.
    * Fixes BN fold implementation to account for a subsequent multi-input layer
    * Fixed a bug where min/max encoding values were not aligned with scale/offset during QAT.
* PyTorch
    * Several bug fixes
* TensorFlow
    * Added a feature called BN Re-estimation that can improve model accuracy after QAT for INT4
      quantization
    * Updated the AutoQuant feature to automatically choose the optimal calibration scheme, create an
      HTML report on which optimizations were applied.
    * Fixed a bug where min/max encoding values were not aligned with scale/offset during QAT.
* Common
    * Documentation updates for taking AIMET models to target.
    * Standalone Batchnorm layers parameter’s conversion such that it will behave as linear/dense layer.
    * [Experimental] Added new Architecture Checker feature to identify and report model architecture
      constructs that are not ideal for quantized runtimes. Users can utilize this information to change
      their model architectures accordingly.

1.25.0
======

* Keras
    * Added QuantAnalyzer feature
    * Adds Batch Normalization folding for Functional Keras Models. This allows the default config files
      to work for super grouping.
    * Resolved an issue with quantizer placement in Sequential blocks in subclassed models
* PyTorch
    * Added AutoQuant V2 which includes advanced features such as out-of-the-box inference, model
      preparer, quant scheme search, improved summary report, etc.
    * Fixes to resolve minor accuracy diffs in the learnedGrid quantizer for per-channel quantization
    * Fixes to improve EfficientNetB4 accuracy w/respect to target
    * Fixed rare case where quantizer may calculate incorrect offset when generating QAT 2.0 learned
      encodings
* TensorFlow
    * Added QuantAnalyzer feature
    * Fixed an accuracy issue due to rare cases where the incorrect BN epsilon was being used
    * Fixed an accuracy issue due to Quantsim export incorrectly recomputing QAT2.0 encodings
* Common
    * Updated AIMET python package version format to support latest pip
    * Fixed an issue where not all inputs might be quantized properly

1.24.0
======

* PyTorch
    * Fixes to resolve minor accuracy diffs in the learnedGrid quantizer for per-channel quantization
    * Added support for AMP 2.0 which enables faster automatic mixed precision
    * Added support for QAT for INT4 quantized models – includes a feature for performing BN Re-estimation
      after QAT
* Keras
    * Added support for AMP 2.0 which enables faster automatic mixed precision
    * Support for basic transformer networks
    * Added support for subclassed models. The current subclassing feature includes support for only a
      single level of subclassing and does not support lambdas.
    * Added QAT per-channel gradient support
    * Minor updates to the quantization configuration
    * Fixed QuantSim bug where layers using dtypes other than float were incorrectly quantized
* TensorFlow
    * Added an additional prelu mapping pattern to ensure proper folding and quantsim node placement
    * Fixed per-channel encoding representation to align with Pytorch and Keras
* Common
    * Export quantsim configuration for configuring downstream target quantization

1.23.0
======

* PyTorch
    * Fixed backward pass of the fake-quantize (QcQuantizeWrapper) nodes to handle symmetric mode
      correctly
    * Per-channel quantization is now enabled on a per-op-type basis
    * Support for recursively excluding module from a root module in QuantSim
    * Support for excluding layers when running model validator and model preparer
    * Reduced memory usage in AdaRound
    * Fixed bugs in AdaRound for per-channel quantization
    * Made ConnectedGraph more robust when identifying custom layers
    * Added jupyter notebook-based examples for the following features
    * AutoQuant: Added support for sparse conv layers in QuantSim (experimental)
* Keras
    * Added support for Keras per-channel quantization
    * Changed interface to CLE to accept a pre-compiled model
    * Added jupyter notebook-based examples for the following features: Transformer quantization
* TensorFlow
    * Fix to avoid unnecessary indexing in AdaRound
* Common
    * TF-enhanced calibration scheme has been accelerated using a custom CUDA kernel. Runs significantly
      faster now.
    * Installation instructions are now combined with rest of the documentation (User-Guide and API docs)

1.22.2
======

* Tensorflow
    * Added support for supergroups : MatMul + Add
    * Added support for TF-Slim BN name with backslash
    * Added support for Depthwise + Conv in CLS

1.22.1
======

* PyTorch
    * Added support for QuantizableMultiHeadAttention for PyTorch nn.transformer layers
    * Support functional conv2d in model preparer
    * Enable qat with multi gpu
    * Optimize forward pass logic of PyTorch QAT 2.0
    * Fix functional depthwise conv support on model preparer
    * Fix bug in model validator to correctly identify functional ops in leaf module
    * Support dynamic functional conv2d in model preparer
    * Added updated default runtime config, also a per-channel one.
    * Include residing module info in model validator
* Keras
    * Support for Keras MultiHeadAttention Layer

1.22.0
======

* PyTorch
    * Support for simulation and QAT for PyTorch transformer models (including support for torch.nn mha and
      encoder layers)

1.21.0
======

* PyTorch
    * PyTorch QuantAnalyzer - Visualize per-layer sensitivity and per-quantizer PDF histograms
    * PyTorch QAT with Range Learning: Added support for Per Channel Quantization
    * PyTorch: Enabled exporting of encodings for multi-output leaf module
* TensorFlow
    * * New feature: TensorFlow AutoQuant - Automatically apply various AIMET post-training quantization techniques
    * Adaround: Added ability to use configuration file in API to adapt to a specific runtime target
    * Adaround: Added Per-Channel Quantization support
    * TensorFlow QuantSim: Added support for FP16 inference and QAT
    * TensorFlow Per Channel Quantization
        * Fixed speed and accuracy issues
        * Fixed zero accuracy for 16-bits per channel quantization
        * Added support for DepthWise Conv2d Op
    * Multiple other bug fixes

1.20.0
======

* PyTorch
    * Propagated encodings for ONNX Ops that were expanded from a single PyTorch Op
* TensorFlow
    * Upgraded AIMET to support TensorFlow version 2.4. AIMET remains compatible with TensorFlow
      version 1.15
* Common
    * Added Jupyter Notebooks for Examples
    * Multiple bug fixes
    * Removed version pinning of many dependent software packages

1.19.1
======

* PyTorch
    * Added CLE support for Conv1d, ConvTranspose1d and Depthwise Separable Conv1d layers
    * Added High-Bias Fold support for Conv1D layer
    * Modified Elementwise Concat Op to support any number of tensors
    * Minor dependency fixes

1.18.0
======

* Common
    * Multiple bug fixes
    * Additional feature examples for PyTorch and TensorFlow

1.17.0
======

* TensorFlow
    * Add Adaround TF feature
* PyTorch
    * Added Examples for Torch quantization, and Channel Pruning & Spatial SVD compression

1.16.2
======

* PyTorch
    * Added a new post-training quantization feature called AdaRound, which stands for AdaptiveRounding
    * Quantization simulation and QAT now also support recurrent layers (RNN, LSTM, GRU)

1.16.1
======

* Added separate packages for CPU and GPU models. This allows users with CPU-only hosts to run AIMET.
* Added separate packages for PyTorch and TensorFlow. Reduces the number of dependencies that users would need to install.

1.16.0
======

* Ported AIMET PyTorch to work with PyTorch ver 1.7.1 with CUDA 11.0
* AIMET PyTorch and AIMET TensorFlow are now available as separate packages
* Version of the AIMET PyTorch and AIMET TensorFlow packages for CPU-only machines are now available

1.13.0
======

* PyTorch
    * Added Adaptive Rounding feature (AdaRound) for PyTorch.
    * Various bug fixes.
