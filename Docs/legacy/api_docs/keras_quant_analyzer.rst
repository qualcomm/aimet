:orphan:

.. _api-keras-quant-analyzer:

===================================
AIMET TensorFlow Quant Analyzer API
===================================

AIMET Keras Quant Analyzer analyzes the Keras model and points out sensitive layers to quantization in the model.
It checks model sensitivity to weight and activation quantization, performs per layer sensitivity and MSE analysis.
It also exports per layer encodings min and max ranges and statistics histogram for every layer.

Top-level API
=============
.. autoclass:: aimet_tensorflow.keras.quant_analyzer.QuantAnalyzer
    :members: analyze

Code Examples
=============

**Required imports**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :lines: 39-47

**Prepare toy dataset to run example code**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 0. Prepare toy dataset to run example code
    :end-before: # End step 0

**Prepare forward pass callback**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 1. Prepare forward pass callback
    :end-before: # End step 1

**Prepare eval callback**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 2. Prepare eval callback
    :end-before: # End step 2

**Prepare model**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 3. Prepare model
    :end-before: # End step 3

**Create QuantAnalyzer object**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 4. Create QuantAnalyzer object
    :end-before: # End step 4

**Run QuantAnalyzer**

.. literalinclude:: ../keras_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 5. Run QuantAnalyzer
    :end-before: # End step 5
