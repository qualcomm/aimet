.. _featureguide-analysis-tools-index:

##############
Analysis tools
##############

.. toctree::
    :hidden:

    Interactive visualization <interactive_visualization>
    Quantization analyzer <quant_analyzer>
    Layer output generation <layer_output_generation>


:ref:`Interactive visualization <featureguide-interactive-visualization>`
-------------------------------------------------------------------------

Produces an interactive HTML to view the statistics collected by each quantizer during calibration.

:ref:`Quantization analyzer <featureguide-quant-analyzer>`
----------------------------------------------------------

QuantAnalyzer analyzes your pre-trained model and points out sensitive layers to quantization
in the model. It checks model sensitivity to weight and activation quantization, performs per
layer sensitivity and MSE analysis. It also exports per layer encodings min and max ranges and
statistics histogram for every layer.

:ref:`Layer output generation <featureguide-layer-output-generation>`
---------------------------------------------------------------------

This API captures and saves intermediate layer-outputs of a model. This allows layer-output
comparison between quantization simulated model (QuantSim object) and actually
quantized model on target-device to debug accuracy miss-match issues at the layer level.
