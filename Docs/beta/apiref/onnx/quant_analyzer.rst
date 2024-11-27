.. _apiref-onnx-quant-analyzer:

#########################
aimet_onnx.quant_analyzer
#########################

..
  # Top level APIs start

**Top level APIs**

.. autoclass:: aimet_onnx.quant_analyzer.QuantAnalyzer
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.enable_per_layer_mse_loss
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.analyze
   :noindex:

**Note:** It is recommended to use onnx-simplifier before applying quant-analyzer.


**Alternatively, you can run specific utility**

You can avoid running all the utilities that QuantAnalyzer offers and only run those of your interest.
For this you need to have the :class:`QuantizationSimModel` object, Then you call the desired
QuantAnalyzer utility of your interest and pass the same object to it.


.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.create_quantsim_and_encodings
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.check_model_sensitivity_to_quantization
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_enabling_quantizers
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_disabling_quantizers
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_encoding_min_max_range
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_stats_histogram
   :noindex:

.. automethod:: aimet_onnx.quant_analyzer.QuantAnalyzer.export_per_layer_mse_loss
   :noindex:

