:orphan:

.. _api-onnx-layer-output-generation:

================================
AIMET ONNX Layer Output Generation API
================================

This API captures and saves intermediate layer-outputs of a model. The model can be original (FP32) or quantsim.
The layer-outputs are named according to the exported ONNX model by the quantsim export API. This allows layer-output comparison
amongst FP32 model, quantization simulated model and actually quantized model on target-device to debug accuracy miss-match issues.

Top-level API
=============

.. autoclass:: aimet_onnx.layer_output_utils.LayerOutputUtil

|

**The following API can be used to Generate Layer Outputs**

.. automethod:: aimet_onnx.layer_output_utils.LayerOutputUtil.generate_layer_outputs


|

Code Example
=============

**Imports**

.. literalinclude:: ../onnx_code_examples/layer_output_generation_code_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Obtain Original or QuantSim model from AIMET Export Artifacts**

.. literalinclude:: ../onnx_code_examples/layer_output_generation_code_example.py
    :language: python
    :start-after: # Step 1. Obtain original or quantsim model
    :end-before: # End step 1

**Obtain inputs for which we want to generate intermediate layer-outputs**

.. literalinclude:: ../onnx_code_examples/layer_output_generation_code_example.py
    :language: python
    :start-after: # Step 2. Obtain pre-processed inputs
    :end-before: # End step 2

**Generate layer-outputs**

.. literalinclude:: ../onnx_code_examples/layer_output_generation_code_example.py
    :language: python
    :start-after: # Step 3. Generate outputs
    :end-before: # End step 3
