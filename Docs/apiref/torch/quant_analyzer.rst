.. _apiref-torch-quant-analyzer:

##########################
aimet_torch.quant_analyzer
##########################

..
  # start-after

.. rubric:: Top level APIs

..
  # start-after toplevel

.. autoclass:: aimet_torch.common.utils.CallbackFunc
    :no-index:

.. autoclass:: aimet_torch.quant_analyzer.QuantAnalyzer
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.analyze
    :no-index:

..
  # end-before toplevel

.. rubric:: Alternatively, you can run specific utility

You can avoid running all the utilities that QuantAnalyzer offers and only run those of your interest.
For this you need to have the :class:`QuantizationSimModel` object, Then you call the desired
QuantAnalyzer utility of your interest and pass the same object to it.

..
  # start-after specific-utility

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.check_model_sensitivity_to_quantization
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_enabling_quant_wrappers
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_disabling_quant_wrappers
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.export_per_layer_encoding_min_max_range
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.export_per_layer_stats_histogram
    :no-index:

.. automethod:: aimet_torch.quant_analyzer.QuantAnalyzer.export_per_layer_mse_loss
    :no-index:

..
  # end-before specific-utility
  # end-before
