:orphan:

.. _api-onnx-quant-analyzer:

================================
AIMET ONNX Quant Analyzer API
================================

AIMET ONNX Quant Analyzer analyzes the ONNX model and points out sensitive layers to quantization in the model.
It checks model sensitivity to weight and activation quantization, performs per layer sensitivity and MSE analysis.
It also exports per layer encodings min and max ranges and statistics histogram for every layer.

Top-level API
=============
.. autoclass:: aimet_onnx.quant_analyzer.QuantAnalyzer
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.enable_per_layer_mse_loss
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.analyze


**Note:** It is recommended to use onnx-simplifier before applying quant-analyzer.


Run specific utility
=============
We can avoid running all the utilities that Quant Analyzer offers and only run those of our interest. For this we
need to have the quantsim object which can be obtained from 'create_quantsim_and_encodings()'. Then we call the
desired Quant Analyzer utility of our interest and pass the quantsim object to it.

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.create_quantsim_and_encodings
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.check_model_sensitivity_to_quantization
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_enabling_quantizers
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_disabling_quantizers
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_encoding_min_max_range
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_stats_histogram
|
.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_mse_loss
|

Code Examples
=============

**Required imports**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Prepare forward pass callback**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 1. Prepare forward pass callback
    :end-before: # End step 1

**Prepare eval callback**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 2. Prepare eval callback
    :end-before: # End step 2

**Prepare model, callback functions and dataloader**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 3. Prepare model, callback functions and dataloader
    :end-before: # End step 3

**Create QuantAnalyzer object**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 4. Create QuantAnalyzer object
    :end-before: # End step 4

**Run QuantAnalyzer**

.. literalinclude:: ../onnx_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 5. Run QuantAnalyzer
    :end-before: # End step 5
