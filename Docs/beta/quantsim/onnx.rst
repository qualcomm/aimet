.. _quantsim-onnx:

#############
Quantsim ONNX
#############

Workflow
========

**Required imports**

.. literalinclude:: ../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # imports start
   :end-before: # imports end

**User should write this function to pass calibration data**

.. literalinclude:: ../../onnx_code_examples/quantization.py
   :language: python
   :pyobject: pass_calibration_data

**Quantize the model**

.. literalinclude:: ../../onnx_code_examples/quantization.py
    :language: python
    :pyobject: quantize_model


API
===

.. autoclass:: aimet_onnx.quantsim.QuantizationSimModel

**Note** :
 - It is recommended to use onnx-simplifier before creating quantsim model.
 - Since ONNX Runtime will be used for optimized inference only, ONNX framework will support Post Training Quantization schemes i.e. TF or TF-enhanced to compute the encodings.

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.compute_encodings

**The following API can be used to Export the Model to target**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.export
