.. _featureguide-seq-mse:

##############
Sequential MSE
##############

Context
=======

Sequential MSE (SeqMSE) is a method that searches for optimal quantization encodings per operation
(i.e. per layer) such that the difference between the original output activation and the
corresponding quantization-aware output activation is minimized.

Since SeqMSE is search-based rather than learning-based, it possesses several advantages:

- It requires only a small amount of calibration data,
- It approximates the global minimum without getting trapped in local minima, and
- It is robust to overfitting.

Workflow
========

Prerequisites
-------------

To use Seq MSE, you must:

- Load a pre-trained model
- Create a training or validation dataloader for the model.

Code example
------------

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

Step 1
~~~~~~

Create QuantizationSimModel object (simulate quantization through AIMET's QuantSim).

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Create Quantization Simulation Model
            :end-before: # End of QuantizationSimModel
Step 2
~~~~~~

Apply Seq MSE to decide optimal quantization encodings for parameters of supported layer(s)/operation(s).

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Apply Seq MSE
            :end-before: # End of Seq MSE

Step 3
~~~~~~

Compute encodings for all activations and remaining parameters of uninitialized layer(s)/operations(s).

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Calibration callback
            :end-before: # End of compute_encodings

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

Step 4
~~~~~~

If resulted quantized accuracy is satisfactory, export the model.

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
