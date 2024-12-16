:orphan:

.. _api-onnx-quantsim:

===============================
AIMET ONNX Quantization SIM API
===============================

Top-level API
=============

.. autoclass:: aimet_onnx.quantsim.QuantizationSimModel

|

**Note** :
 - It is recommended to use onnx-simplifier before creating quantsim model.
 - Since ONNX Runtime will be used for optimized inference only, ONNX framework will support Post Training Quantization schemes i.e. TF or TF-enhanced to compute the encodings.

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.compute_encodings


|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.export

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../onnx_code_examples/quantization.py
    :language: python
    :lines: 39-42

**User should write this function to pass calibration data**


.. literalinclude:: ../onnx_code_examples/quantization.py
   :language: python
   :pyobject: pass_calibration_data


**Quantize the model and finetune (QAT)**

.. literalinclude:: ../onnx_code_examples/quantization.py
    :language: python
    :pyobject: quantize_model
