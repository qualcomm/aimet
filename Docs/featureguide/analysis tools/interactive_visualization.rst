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

        .. include:: ../../apiref/torch/interactive_visualization.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        Interactive visualization does not support TensorFlow.

    .. tab-item:: ONNX
        :sync: onnx

        Interactive visualization does not support ONNX.
