.. _featureguide-interactive-visualization:

#########################
Interactive visualization
#########################

Context
=======

Interactive visualization displays the range (min and max values) of activations and weights for all quantized modules
in the quantization simulation :class:`QuantizationSimModel` object.

Interactive visualization functionality includes:

- Adjustable threshold values to flag layers for which min or max activations or weights exceed these values
- Tables containing names and ranges for layers exceeding threshold values


Workflow
========

tbd

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. warning::
            This feature is under heavy development and API changes may occur without notice in future versions.

        .. currentmodule:: aimet_torch.v2


        .. rubric:: Visualization Tools

        .. autofunction:: aimet_torch.v2.visualization_tools.visualize_stats
            :no-index:

    .. tab-item:: ONNX
        :sync: onnx

        .. warning::
            This feature is under heavy development and API changes may occur without notice in future versions.

        .. rubric:: Visualization Tools

        .. autofunction:: aimet_onnx.analysis.visualize_stats
            :no-index:
