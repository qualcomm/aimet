.. _ptq-adaround:

#################
Adaptive rounding
#################

Context
=======
`Adaptive rounding <https://arxiv.org/pdf/2004.10568>`_ (AdaRound) is a rounding mechanism for model weights designed to adapt to the data to improve the accuracy of the quantized model.

By default, AIMET uses nearest rounding for quantization, in which weight values are quantized to the nearest integer value. AdaRound, however, uses training data to determine how to round quantized weights. This technique often improves the accuracy of the quantized model.

The following figure illustrates how AdaRound may alter the rounding of a quantized value.

.. image:: ../images/adaround.png
    :width: 600px

Workflow
========

Prerequisites
-------------

To use AdaRound, you must:

- Load a trained model
- Create a training or validation dataloader for the model

Workflow
--------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. image:: ../images/adaround-workflow.png
            :width: 900px

        .. note::
            We recommend simplifying the ONNX model with onnx-simplifier before applying AdaRound.
            Simplification folds constants and removes redundant nodes, giving AdaRound a cleaner
            graph in which to identify and optimize the weight tensors of its supported node
            types (``Conv``, ``ConvTranspose``, ``MatMul``, ``Gemm``).

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Set up model
            :end-before: # End of model

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Set up dataloader
            :end-before: # End of dataloader

    .. tab-item:: PyTorch
        :sync: torch

        .. image:: ../images/adaround-workflow-torch.png
            :width: 900px

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: [setup]
            :end-before: [step_1]

Step 1
~~~~~~

Apply AdaRound to the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: [step_1]
            :end-before: [step_2]

Step 2
~~~~~~

Use AIMET's QuantSim to simulate quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: [step_2]
            :end-before: [step_3]

Step 3
~~~~~~

Evaluate the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: [step_3]
            :end-before: [step_4]

Step 4
~~~~~~

If AdaRound resulted in satisfactory accuracy, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: [step_4]

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/adaround.rst
            :start-after: # start-after toplevel
            :end-before: # end-before toplevel

        .. note::
            AdaRound parameters are passed directly as keyword arguments to ``apply_adaround``;
            there is no separate parameters class for ``aimet-onnx``.

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/adaround.rst
            :start-after: # start-after toplevel
            :end-before: # end-before toplevel

        .. include:: ../apiref/torch/adaround.rst
            :start-after: # start-after parameters
            :end-before: # end-before parameters
