.. _apiref-torch-quant-analyzer:

##########################
aimet_torch.quant_analyzer
##########################

..
  # start-after

**Top level APIs**

.. autoclass:: aimet_common.utils.CallbackFunc

.. autoclass:: aimet_torch.quant_analyzer.QuantAnalyzer

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.analyze

**Alternatively, you can run specific utility**

You can avoid running all the utilities that QuantAnalyzer offers and only run those of your interest.
For this you need to have the :class:`QuantizationSimModel` object, Then you call the desired
QuantAnalyzer utility of your interest and pass the same object to it.

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.check_model_sensitivity_to_quantization

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_enabling_quant_wrappers

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_disabling_quant_wrappers

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.export_per_layer_encoding_min_max_range

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.export_per_layer_stats_histogram

.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.export_per_layer_mse_loss

..
  # end-before
