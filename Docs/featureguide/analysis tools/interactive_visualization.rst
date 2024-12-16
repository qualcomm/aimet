.. _featureguide-interactive-visualization:

#########################
Interactive visualization
#########################

Context
=======

Creates an interactive visualization of min and max activations/weights of all quantized modules
in the Quantization simulation :class:`QuantizationSimModel` object.

The features include:

- Adjustable threshold values to flag layers whose min or max activations/weights exceed the set thresholds

- Tables containing names and ranges for layers exceeding threshold values.


Workflow
========

tbd

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/v2/interactive_visualization.rst
            :start-after: # start-after
