.. _featureguide-quant-analyzer:

#####################
Quantization analyzer
#####################

Context
=======

The Quantization analyzer (QuantAnalyzer) performs several analyses to identify sensitive areas and
hotspots in your model. These analyses are performed automatically. To use QuantAnalyzer, you pass
in callbacks to perform forward passes and evaluations, and optionally a dataloader for MSE loss
analysis.

For each analysis, QuantAnalyzer outputs JSON and/or HTML files containing data and plots for
visualization.

Detailed analysis descriptions
==============================

QuantAnalyzer performs the following analyses:

1. Sensitivity analysis to weight and activation quantization
-------------------------------------------------------------

QuantAnalyzer compares the accuracies of the original FP32 model, an activation-only quantized model,
and a weight-only quantized model. This helps determine which AIMET quantization technique(s) will
be more beneficial for the model.

For example, in situations where the model is more sensitive to activation quantization, Post-training
quantization (PTQ) techniques like Adaptive Rounding (Adaround) or Cross-layer equalization (CLE) might
not be very helpful.

Quantized accuracy metric for your model are printed as part of AIMET logging.

2. Per-layer quantizer enablement analysis
------------------------------------------

Sometimes the accuracy drop incurred from quantization can be attributed to only a subset of layers
within the model. QuantAnalyzer finds such layers by enabling and disabling individual quantizers to
observe how the quantized model accuracy metric changes.

The following two types of quantizer enablement analyses are performed:

1. Disable all quantizers across the model and, for each layer, enable only that layer's output quantizer
and perform evaluation with the provided callback. This results in accuracy values obtained for each
layer in the model when only that layer's quantizer is enabled, exposing the effects of individual
layer quantization and pinpointing culprit layer(s) and hotspots.

2. Enable all quantizers across the model and, for each layer, disable only that layer's output quantizer
and perform evaluation with the provided callback. Once again, accuracy values are produced for each
layer in the model when only that layer's quantizer is disabled.

As a result of these analyses, AIMET outputs `per_layer_quant_enabled.html` and
`per_layer_quant_disabled.html` respectively, containing plots mapping layers on the x-axis to quantized
model accuracy metrics on the y-axis.

JSON files `per_layer_quant_enabled.json` and `per_layer_quant_disabled.json` are also produced,
containing the data shown in the .html plots.

3. Per-layer encodings min-max range analysis
---------------------------------------------

As part of quantization, encoding parameters for each quantizer must be obtained.
These parameters include scale, offset, min, and max, and are used to map floating point values to
quantized integer values.

QuantAnalyzer tracks the min and max encoding parameters computed by each quantizer in the model
as a result of forward passes through the model with representative data (from which the scale and
offset values can be directly obtained).

As a result of this analysis, AIMET outputs html plots and json files for each activation quantizer
and each parameter quantizer (contained in the min_max_ranges folder) containing the encoding min/max
values for each.

If Per-channel quantization (PCQ) is enabled, encoding min and max values for all the channels
of each weight parameters are shown.

4. Per-layer statistics histogram
---------------------------------

Under the TF-enhanced quantization scheme, encoding min/max values for each quantizer are obtained
by collecting a histogram of tensor values seen at that quantizer and deleting outliers.

When this quantization scheme is selected, QuantAnalyzer outputs plots for each quantizer in the model,
displaying the histogram of tensor values seen at that quantizer.

These plots are available as part of the `activations_pdf` and `weights_pdf` folders, containing a
separate .html plot for each quantizer.

5. Per layer mean-square-error (MSE) loss
-----------------------------------------

QuantAnalyzer can monitor each layer's output in the original FP32 model as well as the corresponding
layer output in the quantized model and calculate the MSE loss between the two.

This helps identify which layers may contribute more to quantization noise.

To enable this optional analysis, you pass in a dataloader that QuantAnalyzer reads from.
Approximately **256 samples** are sufficient for the analysis.

A `per_layer_mse_loss.html` file is generated containing a plot that maps layer quantizers on the
x-axis to MSE loss on the y-axis. A corresponding `per_layer_mse_loss.json` file is generated
containing data corresponding to the .html file.

Prerequisites
=============

To call the QuantAnalyzer API, you must provide the following:

- An FP32 pre-trained model for analysis
- A dummy input for the model that can contain random values but which must match the shape of the model's expected input
- A user-defined function for passing 500-1000 representative data samples through the model for quantization calibration
- A user-defined function for passing labeled data through the model for evaluation, returning an accuracy metric
- (Optional, for running MSE loss analysis) A dataloader providing unlabeled data to be passed through the model

.. note::
   Typically on quantized runtimes, batch normalization (BN) layers are folded where possible. So
   that you don't have to call a separate API to do so, QuantAnalyzer automatically performs Batch
   Norm Folding before running its analysis.

Workflow
========

Code example
------------

Step 1 Prepare callback for calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Required imports**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Prepare forward pass callback**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

    .. tab-item:: TensorFlow
        :sync: tf

        **Required imports**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :lines: 39-47

        **Prepare toy dataset to run example code**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Prepare toy dataset to run example code
            :end-before: # End step 0

        **Prepare forward pass callback**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

    .. tab-item:: ONNX
        :sync: onnx

        **Required imports**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Prepare forward pass callback**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

Step 2 Prepare callback for quantized model evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Prepare eval callback**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2

    .. tab-item:: TensorFlow
        :sync: tf

        **Prepare eval callback**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2

    .. tab-item:: ONNX
        :sync: onnx

        **Prepare eval callback**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2

Step 3 Prepare model and callback functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Prepare model and callback functions**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model and callback functions
            :end-before: # End step 3

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model
            :end-before: # End step 3

    .. tab-item:: ONNX
        :sync: onnx

        **Prepare model, callback functions and dataloader**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model, callback functions and dataloader
            :end-before: # End step 3

Step 4 Create QuantAnalyzer and run analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Create QuantAnalyzer object**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4

        **Run QuantAnalyzer**

        .. literalinclude:: ../../../torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 5. Run QuantAnalyzer
            :end-before: # End step 5

    .. tab-item:: TensorFlow
        :sync: tf

        **Create QuantAnalyzer object**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4

        **Run QuantAnalyzer**

        .. literalinclude:: ../../../keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 5. Run QuantAnalyzer
            :end-before: # End step 5

    .. tab-item:: ONNX
        :sync: onnx

        **Create QuantAnalyzer object**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4

        **Run QuantAnalyzer**

        .. literalinclude:: ../../../onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 5. Run QuantAnalyzer
            :end-before: # End step 5

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/quant_analyzer.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../../apiref/tensorflow/quant_analyzer.rst
           :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../../apiref/onnx/quant_analyzer.rst
           :start-after: # start-after
