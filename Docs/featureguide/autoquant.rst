.. _featureguide-autoquant:

######################
Automatic quantization
######################

Context
=======

AIMET toolkit offers a suite of post-training quantization (PTQ) techniques. Often,
applying these techniques in a specific sequence results in better quantized accuracy and performance.

The automatic quantization (AutoQuant) feature analyzes your trained model, determines the best
sequence of AIMET PTQ quantization techniques, and applies these techniques. You can specify the
tolerable accuracy drop in the AutoQuant API. As soon as this threshold accuracy is
reached, AutoQuant stops applying PTQ quantization techniques.

Without the AutoQuant feature, you must manually try combinations of AIMET quantization techniques.
This manual process is error-prone and time-consuming.

Workflow
========

The AutoQuant workflow is showin in the following figure.

.. image:: ../images/auto_quant_1.png
   :height: 450

Before entering the optimization workflow, AutoQuant prepares by:

1. Checking the validity of the model and converting the model into an AIMET quantization-friendly format (`Prepare Model`).
2. Selecting the best-performing quantization scheme for the given model (`QuantScheme Selection`)

After the preparation steps, AutoQuant proceeds to try four PTQ techniques:

1. :ref:`BatchNorm folding <featureguide-bnf>`
2. :ref:`Cross-layer equalization (CLE) <featureguide-cle>`
3. :ref:`Adaptive rounding (Adaround) <featureguide-adaround>` (if enabled)
4. :ref:`Automatic Mixed Precision (AMP) <featureguide-amp>` (if enabled)

These techniques are applied in a best-effort manner until the model meets the allowed accuracy drop.
If applying AutoQuant fails to satisfy the evaluation goal, AutoQuant returns the model that gave
the best results.


Prerequisites
=============

There are no special prerequisites to using AutoQuant. It can be applied to most models.

Procedure
---------

Step 1
~~~~~~

Load the model for automatic quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. container:: tab-heading

            In the following code example, the model is MobileNetV2.

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

    .. tab-item:: TensorFlow
        :sync: tf

        .. container:: tab-heading

            In the following code example, the model is MobileNetV2.

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of step 1

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            The following code example converts the PyTorch MobileNetV2 model to ONNX and uses it in the subsequent code.

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

Step 2
~~~~~~

Prepare the dataset.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

Step 3
~~~~~~

Prepare the evaluation callback.

For your model, implement the evaluation callback to serve your own goals, maintaining the function signature.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

Step 4
~~~~~~

Create the AutoQuant object.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

Step 5
~~~~~~

Set AdaRound parameters.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

Step 6
~~~~~~

Set AMP parameters.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 6
            :end-before: # End of step 6

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 6
            :end-before: # End of step 6

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 6
            :end-before: # End of step 6

Step 7
~~~~~~

Run AutoQuant.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 7
            :end-before: # End of step 7

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 7
            :end-before: # End of step 7

        .. rst-class:: script-output

          .. code-block:: none

            - Quantized Accuracy (before optimization): 0.0235
            - Quantized Accuracy (after optimization):  0.7164

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 7
            :end-before: # End of step 7

        .. rst-class:: script-output

          .. code-block:: none

            - Quantized Accuracy (before optimization): 0.0235
            - Quantized Accuracy (after optimization):  0.7164


API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/autoquant.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/autoquant.rst
            :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/autoquant.rst
           :start-after: # start-after

