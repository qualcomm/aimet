.. _ptq-seq-mse:

##############
Sequential MSE
##############

Context
=======

Sequential MSE (SeqMSE) is a quantization technique that optimizes the parameter encodings of each layer 
of a model individually to minimize the difference between the layer's original and quantized outputs. 
Rather than relying on training, SeqMSE uses a search-based approach, offering several benefits:

- It requires only a small amount of unlabeled data
- It approximates the global minimum without getting trapped in local minima
- It is robust to overfitting


Workflow
========

Prerequisites
-------------

To use SeqMSE, you must have the following:

- A pre-trained PyTorch or ONNX model
- A set of representative input samples for the model

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
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # [setup]
            :end-before: # End of load the model

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Prepare the dataloader
            :end-before: # End of dataloader

Step 1
~~~~~~

Create a :ref:`QuantizationSimModel<quantsim-index>` object for the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Create Quantization Simulation Model
            :end-before: # End of QuantizationSimModel
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1


Step 2
~~~~~~

Apply SeqMSE to find optimized parameter encodings for supported layer types.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Apply Seq MSE
            :end-before: # End of Seq MSE
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

Step 3
~~~~~~

Compute encodings for remaining uninitialized quantizers.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Calibration callback
            :end-before: # End of compute_encodings
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

Step 4
~~~~~~

Evaluate the quantized model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_seqmse.py
            :language: python
            :start-after: # Evaluation
            :end-before: # End of evaluation
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

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
    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_seqmse.py
            :language: python
            :start-after: # Step 5
            :end-before: # End of step 5

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. autofunction:: aimet_torch.seq_mse.apply_seq_mse
            :noindex:

        .. autoclass:: aimet_torch.seq_mse.SeqMseParams
            :members:
            :noindex:

    .. tab-item:: ONNX
        :sync: onnx

        .. autofunction:: aimet_onnx.apply_seq_mse
            :noindex:

        .. note::
            SeqMSE parameters are passed directly as keyword arguments to ``apply_seq_mse``; there is no
            separate parameters class for ``aimet-onnx``.


