.. _featureguide-seq-mse:

##############
Sequential MSE
##############

Context
=======

Sequential MSE (SeqMSE) is a method that searches for optimal quantization encodings per operation
(i.e. per layer) such that the difference between the original output activation and the
corresponding quantization-aware output activation is minimized.

Since SeqMSE is search-based rather than learning-based, it has several advantages:

- It requires only a small amount of calibration data
- It approximates the global minimum without getting trapped in local minima
- It is robust to overfitting


Workflow
========

Prerequisites
-------------

To use Seq MSE, you must:

- Use PyTorch or ONNX. Sequential MSE does not support TensorFlow models
- Load a pre-trained model
- Create a training or validation dataloader for the model

Procedure
---------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # [setup]
            :end-before: # End of load the model

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Prepare the dataloader
            :end-before: # End of dataloader

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        tbd.

Step 1
~~~~~~

Use AIMET's :ref:`quantization simulation<quantsim-index>` to create a QuantSimModel object.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Create Quantization Simulation Model
            :end-before: # End of QuantizationSimModel

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        tbd.


Step 2
~~~~~~

Apply SeqMSE to decide optimal quantization encodings for parameters of supported layers and operations.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Apply Seq MSE
            :end-before: # End of Seq MSE

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        tbd.

Step 3
~~~~~~

Apply SeqMSE to compute encodings for remaining parameters of uninitialized layers and operations.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Calibration callback
            :end-before: # End of compute_encodings

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        tbd.

Step 4
~~~~~~

Evaluate the quantized model using :class:`ImageClassificationEvaluator`.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Evaluation
            :end-before: # End of evaluation

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        tbd.

Step 5
~~~~~~

If the resulting quantized accuracy is satisfactory, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Export
            :end-before: # End of export

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/seq_mse.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/seq_mse.rst
            :start-after: # start-after


