===============================
AIMET ONNX Quantization APIs
===============================

.. toctree::
   :titlesonly:
   :hidden:

    Quantization Simulation API<onnx_quantsim>
    Cross-Layer Equalization API<onnx_cross_layer_equalization>
    Adaptive Rounding API<onnx_adaround>
    AutoQuant API<onnx_auto_quant>
    QuantAnalyzer API<onnx_quant_analyzer>

AIMET Quantization for ONNX Models provides the following functionality.
   - :ref:`Quantization Simulation API<api-onnx-quantsim>`: Allows ability to simulate inference on quantized hardware
   - :ref:`Cross-Layer Equalization API<api-onnx-cle>`: Post-training quantization technique to equalize layer parameters
   - :ref:`Adaround API<api-onnx-adaround>`: Post-training quantization technique to optimize rounding of weight tensors
   - :ref:`AutoQuant API<api-onnx-auto-quant>`: Unified API that integrates the post-training quantization techniques provided by AIMET
   - :ref:`QuantAnalyzer API<api-onnx-quant-analyzer>`: Analyzes the model and points out sensitive layers to quantization
