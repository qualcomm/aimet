.. _featureguide-analysis-tools-index:

##############
Analysis tools
##############

.. toctree::
    :hidden:

    Interactive visualization <interactive_visualization>
    Quantization analyzer <quant_analyzer>
    Layer output generation <layer_output_generation>

AIMET offers these tools to view and analyze a model's interal quantization results.

Interactive visualization
-------------------------

:ref:`Interactive visualization <featureguide-interactive-visualization>` produces an interactive HTML console showing the statistics collected by each quantizer during calibration.

Quantization analyzer
---------------------

:ref:`Quantization analyzer <featureguide-quant-analyzer>` (QuantAnalyzer) analyzes your pre-trained model and identifies layers sensitive to quantization. It checks model sensitivity to weight and activation quantization, and performs per-layer sensitivity and mean square error analysis. It also exports per-layer encoding min and max ranges and
statistics histograms for every layer.

Layer output generation
-----------------------

:ref:`Layer output generation <featureguide-layer-output-generation>` is an API that captures and saves intermediate layer model outputs. This allows layer-output comparison between a quantization simulated model (QuantSim object) and an actual quantized model on a target device in order to debug accuracy mismatch issues at the layer level.
