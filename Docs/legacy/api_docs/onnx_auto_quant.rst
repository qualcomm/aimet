:orphan:

.. _api-onnx-auto-quant:

===========================
AIMET ONNX AutoQuant API
===========================

User Guide Link
===============
To learn more about this technique, please see :ref:`AutoQuant<ug-auto-quant>`

Top-level API
=============
.. autoclass:: aimet_onnx.auto_quant_v2.AutoQuant
    :members:
    :member-order: bysource


**Note:** It is recommended to use onnx-simplifier before applying auto-quant.


Code Examples
===============
.. literalinclude:: ../onnx_code_examples/auto_quant_v2.py
    :language: python
    :lines: 40-
