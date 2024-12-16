:orphan:

.. _api-keras-quantsim:

=====================================
AIMET TensorFlow Quantization SIM API
=====================================

User Guide Link
===============
To learn more about Quantization Simulation, please see :ref:`Quantization Sim<ug-quantsim>`


Top-level API
=============

.. autoclass:: aimet_tensorflow.keras.quantsim.QuantizationSimModel

|

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings


|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_tensorflow.keras.quantsim.QuantizationSimModel.export

|

Encoding format is described in the :ref:`Quantization Encoding Specification<api-quantization-encoding-spec>`

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../keras_code_examples/quantization.py
    :language: python
    :lines: 37-40

**Quantize with Fine tuning**

.. literalinclude:: ../keras_code_examples/quantization.py
    :language: python
    :pyobject: quantize_model
