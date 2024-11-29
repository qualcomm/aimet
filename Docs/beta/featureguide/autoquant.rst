.. _featureguide-autoquant:

######################
Automatic quantization
######################

Context
=======

AIMET toolkit offers a suite of neural network post-training quantization (PTQ) techniques. Often,
applying these techniques in a specific sequence results in better quantized accuracy and performance.

The Automatic quantization (AutoQuant) feature analyzes your pre-trained model, determines the best
sequence of AIMET PTQ quantization techniques, and applies these techniques. You can specify the
accuracy drop that can be tolerated in the AutoQuant API. As soon as this threshold accuracy is
reached, AutoQuant stops applying PTQ quantization techniques.

Without the AutoQuant feature, you must manually try combinations of AIMET quantization techniques.
This manual process is error-prone and time-consuming.

Prerequisites
=============

Workflow
========

The workflow looks like this:

    .. image:: ../../images/auto_quant_v2_flowchart.png

Before entering the optimization workflow, AutoQuant prepares by:

1. Checking the validity of the model and converting the model into an AIMET quantization-friendly format (`Prepare Model`).
2. Selecting the best-performing quantization scheme for the given model (`QuantScheme Selection`)

After the preparation steps, AutoQuant proceeds to try three techniques:

1. :ref:`BatchNorm folding <featureguide-bnf>`
2. :ref:`Cross-layer equalization (CLE) <featureguide-cle>`
3. :ref:`Adaptive rounding (Adaround) <featureguide-adaround>` (if enabled)
4. Automatic Mixed Precision (AMP) (if enabled)

These techniques are applied in a best-effort manner until the model meets the allowed accuracy drop.
If applying AutoQuant fails to satisfy the evaluation goal, AutoQuant returns the model that returned
the best results.

Code example
------------

Step 1
~~~~~~

Load the model for automatic quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_autoquant.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

    .. tab-item:: TensorFlow
        :sync: tf

        .. container:: tab-heading

            Load the model for automatic quantization. In this code example, we will use MobileNetV2

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of step 1

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Load the model for automatic quantization. In this code example, we will convert PyTorch MobileNetV2 to ONNX and use it in the subsequent code

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

Step 2
~~~~~~

Prepare model and dataloader

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

        .. container:: tab-heading

            Prepare dataset

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Prepare model and dataloader

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

Step 3
~~~~~~

Prepare eval callback

In the actual use cases, the users should implement this part to serve their own goals,
maintaining the function signature.

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

        .. container:: tab-heading

            Prepare eval callback

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Prepare eval callback

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

Step 4
~~~~~~

Create AutoQuant object.

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

        .. container:: tab-heading

            Create AutoQuant object

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Create AutoQuant object

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

Step 5
~~~~~~

Set AdaRound params

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

        .. container:: tab-heading

            Set AdaRound params

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Set AdaRound params

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

Step 6
~~~~~~

Set AMP params

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

        .. container:: tab-heading

            Set AMP params

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 6
            :end-before: # End of step 6

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Set AMP params

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 6
            :end-before: # End of step 6

Step 7
~~~~~~

Run AutoQuant

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

        .. container:: tab-heading

            Run AutoQuant

        .. literalinclude:: ../snippets/tensorflow/apply_autoquant.py
            :language: python
            :start-after: # Step 7
            :end-before: # End of step 7

        **Output**
        ::

            - Quantized Accuracy (before optimization): 0.0235
            - Quantized Accuracy (after optimization):  0.7164

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Run AutoQuant

        .. literalinclude:: ../snippets/onnx/apply_autoquant.py
            :language: python
            :start-after: # Step 7
            :end-before: # End of step 7

        **Output**
        ::

            - Quantized Accuracy (before optimization): 0.0235
            - Quantized Accuracy (after optimization):  0.7164

Results
=======

Next steps
==========

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

