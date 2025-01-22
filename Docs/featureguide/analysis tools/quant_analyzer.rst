.. _featureguide-quant-analyzer:

#####################
Quantization analyzer
#####################

Context
=======

The Quantization analyzer (QuantAnalyzer) automatically performs several analyses to identify sensitive areas in your model. To use QuantAnalyzer, you pass in callbacks to perform forward passes and evaluations, and optionally a dataloader for mean square error (MSE) loss analysis.

For each analysis, QuantAnalyzer generates JSON and/or HTML files containing the data, and plots for
visualization.

Analysis descriptions
=====================

QuantAnalyzer performs the following analyses.

1: Sensitivity to weight and activation quantization
----------------------------------------------------

QuantAnalyzer compares the accuracies of the original FP32 model, an activation-only quantized model,
and a weight-only quantized model. This helps determine which AIMET quantization technique(s) will
be more effective in the model.

For example, in situations where the model is more sensitive to activation quantization, post-training
quantization (PTQ) techniques like Adaptive Rounding (Adaround) or Cross-layer Equalization (CLE) might
not be very helpful.

Quantized accuracy metrics for your model are printed as part of AIMET logging.

2: Per-layer quantizer enablement
---------------------------------

Sometimes the accuracy drop incurred from quantization can be attributed to only a subset of layers
within the model. QuantAnalyzer finds such layers by enabling and disabling individual quantizers to
observe how the quantized model accuracy metric changes.

Two types of quantizer enablement analyses are performed:

1. **One at a time**: Disable all quantizers across the model and, for each layer, enable only that layer's output quantizer. Perform evaluation with the provided callback, giving accuracy values for each
layer in the model when it's the sole quantized layer. This and pinpoints hotspots by exposing the effects of individual
layer quantization.

2. **Elimination**: Enable all quantizers across the model and, for each layer, disable only that layer's output quantizer. Perform evaluation with the provided callback, giving accuracy values for each
layer in the model when only that layer's quantizer is disabled.

AIMET outputs the results of these analyses as `per_layer_quant_enabled.html` and
`per_layer_quant_disabled.html` respectively. These files contain plots of the quantized
model accuracy metrics for each layer.

JSON files `per_layer_quant_enabled.json` and `per_layer_quant_disabled.json` are also produced,
containing the data shown in the .html plots.

3: Per-layer encodings min-max range
------------------------------------

As part of quantization, encoding parameters for each quantizer must be calculated.
These parameters are used to map floating point values to
quantized integer values and include scale, offset, min, and max.

QuantAnalyzer tracks the min and max encoding parameters computed by each quantizer in the model
as a result of forward passes through the model with representative data (from which the scale and
offset values can be directly obtained).

AIMET outputs HTML plots and JSON files to the min_max_ranges folder for each activation quantizer
and each parameter quantizer, containing the encoding min/max values for each.

If per-channel quantization (PCQ) is enabled, encoding min and max values are shown for all the channels
of each weight parameter.

4: Per-layer statistics histogram
---------------------------------

Under the TF-enhanced quantization scheme, min/max encoding values for each quantizer are obtained
by deleting outliers from the histogram of tensor values seen at the quantizer.

When this quantization scheme is selected, QuantAnalyzer outputs the histogram of tensor values seen at each quantizer in the model.

These plots are available as part of the `activations_pdf` and `weights_pdf` folders. There is a
separate .html plot for each quantizer.

5: Per-layer mean-square-error loss
-----------------------------------

QuantAnalyzer can monitor each layer's output in the original FP32 model as well as the corresponding
layer output in the quantized model and calculate the MSE loss between the two.

This helps identify which layers may contribute more to quantization noise.

To enable this optional analysis, you pass in a dataloader that QuantAnalyzer reads from.
Approximately **256 samples** are sufficient for the analysis.

A `per_layer_mse_loss.html` file is generated containing a plot that maps layer quantizers on the
x-axis to MSE loss on the y-axis. A corresponding `per_layer_mse_loss.json` file is generated
containing data used in the .html file.

Prerequisites
=============

To call the QuantAnalyzer API, provide the following:

- An FP32 pre-trained model for analysis
- A dummy input for the model. This can contain random values but it must match the shape of the model's expected input
- A user-defined function for passing 500-1000 representative data samples through the model for quantization calibration
- A user-defined function for passing labeled data through the model for evaluation, returning an accuracy metric
- (Optional, for running MSE loss analysis) A dataloader providing unlabeled data to be passed through the model

.. note::
   Typically on quantized runtimes, batch normalization (BN) layers are folded where possible. So
   that you don't have to call a separate API to do so, QuantAnalyzer automatically performs Batch
   Norm Folding before running its analysis.

Workflow
========

Step 1 Importing libraries
--------------------------

Import required libraries.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :lines: 39-47

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0


Step 2 Preparing the calibration callback
-----------------------------------------

Prepare the callback for calibration.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

    .. tab-item:: TensorFlow
        :sync: tf

        **2.1 Prepare toy dataset to run example code**

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 0. Prepare toy dataset to run example code
            :end-before: # End step 0

        **2.2 Prepare forward pass callback**

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 1. Prepare forward pass callback
            :end-before: # End step 1

Step 3 Preparing the evaluation callback 
----------------------------------------

Prepare the callback for quantized model evaluation.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 2. Prepare eval callback
            :end-before: # End step 2


Step 4 Preparing model
----------------------

Prepare the model, callback functions, and dataloader as required per platform.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Prepare model, callback functions, and data**

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model and callback functions
            :end-before: # End step 3

    .. tab-item:: TensorFlow
        :sync: tf

        **Prepare the model**

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model
            :end-before: # End step 3

    .. tab-item:: ONNX
        :sync: onnx

        **Prepare model, callback functions and dataloader**

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 3. Prepare model, callback functions and dataloader
            :end-before: # End step 3

Step 5 Creating the QuantAnalyzer 
---------------------------------

Create the QuantAnalyzer.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 4. Create QuantAnalyzer object
            :end-before: # End step 4


Step 6 Running the analysis
---------------------------

Finally, run the QuantAnalyzer to analyze the data.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 5. Run QuantAnalyzer
            :end-before: # End step 5

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/quant_analyzer_code_example.py
            :language: python
            :start-after: # Step 5. Run QuantAnalyzer
            :end-before: # End step 5

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/quant_analyzer_code_example.py
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
