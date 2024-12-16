.. _ug-quant-analyzer:


###################
AIMET QuantAnalyzer
###################

Overview
========

The QuantAnalyzer performs several analyses to identify sensitive areas and hotspots in the model. These analyses are performed automatically. To use QuantAnalyzer, you pass in callbacks to perform forward passes and evaluations, and optionally a dataloader for MSE loss analysis.

For each analysis, QuantAnalyzer outputs JSON and/or HTML files containing data and plots for visualization.

Requirements
============

To call the QuantAnalyzer API, you must provide the following:
    - An FP32 pre-trained model for analysis
    - A dummy input for the model that can contain random values but which must match the shape of the model's expected input
    - A user-defined function for passing 500-1000 representative data samples through the model for quantization calibration
    - A user-defined function for passing labeled data through the model for evaluation, returning an accuracy metric
    - (Optional, for running MSE loss analysis) A dataloader providing unlabeled data to be passed through the model

Other quantization-related settings are also provided in the call to analyze a model.
See :doc:`PyTorch QuantAnalyzer API Docs<../api_docs/torch_quant_analyzer>` for more about how to call the QuantAnalyzer feature.

.. note::
   Typically on quantized runtimes, batch normalization (BN) layers are folded where possible. So that you don't have to call a separate API to do so, QuantAnalyzer automatically performs Batch Norm Folding before running its analyses.

Detailed analysis descriptions
==============================

QuantAnalyzer performs the following analyses:

Sensitivity analysis to weight and activation quantization
    QuantAnalyzer compares the accuracies of the original FP32 model, an activation-only quantized model, and a weight-only quantized model. This helps determine which AIMET quantization technique(s) will be more beneficial for the model.

    For example, in situations where the model is more sensitive to activation quantization, PTQ techniques like Adaptive Rounding or Cross Layer Equalization might not be very helpful.

    Accuracy values for each model are printed as part of AIMET logging.

Per-layer quantizer enablement analysis
    Sometimes the accuracy drop incurred from quantization can be attributed to only a subset of quantizers within the model. QuantAnalyzer finds such layers by enabling and disabling individual quantizers to observe how the model accuracy changes.

    The following two types of quantizer enablement analyses are performed:

    1. Disable all quantizers across the model and, for each layer, enable only that layer's output quantizer and perform evaluation with the provided callback. This results in accuracy values obtained for each layer in the model when only that layer's quantizer is enabled, exposing the effects of individual layer quantization and pinpointing culprit layer(s) and hotspots.

    2. Enable all quantizers across the model and, for each layer, disable only that layer's output quantizer and perform evaluation with the provided callback. Once again, accuracy values are produced for each layer in the model when only that layer's quantizer is disabled.

    As a result of these analyses, AIMET outputs `per_layer_quant_enabled.html` and `per_layer_quant_disabled.html` respectively, containing plots mapping layers on the x-axis to model accuracy on the y-axis.

    JSON files `per_layer_quant_enabled.json` and `per_layer_quant_disabled.json` are also produced, containing the data shown in the .html plots.

Per-layer encodings min-max range analysis
    As part of quantization, encoding parameters for each quantizer must be obtained.
    These parameters include scale, offset, min, and max, and are used to map floating point values to quantized integer values.

    QuantAnalyzer tracks the min and max encoding parameters computed by each quantizer in the model as a result of forward passes through the model with representative data (from which the scale and offset values can be directly obtained).

    As a result of this analysis, AIMET outputs html plots and json files for each activation quantizer and each parameter quantizer (contained in the min_max_ranges folder) containing the encoding min/max values for each.

    If Per Channel Quantization (PCQ) is enabled, encoding min and max values for all the channels of each weight are shown.

Per-layer statistics histogram
    Under the TF Enhanced quantization scheme, encoding min/max values for each quantizer are obtained by collecting a histogram of tensor values seen at that quantizer and deleting outliers.

    When this quantization scheme is selected, QuantAnalyzer outputs plots for each quantizer in the model, displaying the histogram of tensor values seen at that quantizer.
    These plots are available as part of the `activations_pdf` and `weights_pdf` folders, containing a separate .html plot for each quantizer.

Per layer mean-square-error (MSE) loss (optional)
    QuantAnalyzer can monitor each layer's output in the original FP32 model as well as the corresponding layer output in the quantized model and calculate the MSE loss between the two.
    This helps identify which layers may contribute more to quantization noise.

    To enable this optional analysis, you pass in a dataloader that QuantAnalyzer reads from.
    Approximately 256 samples/images are sufficient for the analysis.

    A `per_layer_mse_loss.html` file is generated containing a plot that maps layer quantizers on the x-axis to MSE loss on the y-axis. A corresponding `per_layer_mse_loss.json` file is generated containing data corresponding to the .html file.

QuantAnalyzer API
=================

See the links below to view the QuantAnalyzer API for each AIMET variant:

- :ref:`QuantAnalyzer for PyTorch<api-torch-quant-analyzer>`
- :ref:`QuantAnalyzer for Keras<api-keras-quant-analyzer>`
- :ref:`QuantAnalyzer for ONNX<api-onnx-quant-analyzer>`
