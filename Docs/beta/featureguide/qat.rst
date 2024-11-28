.. _featureguide-qat:

###########################
Quantization aware training
###########################

Context
=======

When post-training quantization (PTQ) doesn't sufficiently reduce quantization error, the next step is to use quantization-aware training (QAT). QAT finds better-optimized solutions than PTQ by fine-tuning the model parameters in the presence of quantization noise. This higher accuracy comes at the usual cost of neural network training, including longer training times and the need for labeled data and hyperparameter search.

QAT modes
---------

There are two versions of QAT: without range learning and with range learning.

Without range learning
  In QAT without Range Learning, encoding values for activation quantizers are found once during calibration and are not updated again.

With range learning
  In QAT with Range Learning, encoding values for activation quantizers are set during calibration and can be updated during training, resulting in better scale and offset quantization parameters.

In both versions, parameter quantizer encoding values continue to be updated with the parameters themselves during training.

Recommendations for quantization-aware training
-----------------------------------------------

Here are some guidelines that can improve performance and speed convergence with QAT:

Initialization
    - It often helps to first apply post training quantization techniques like :ref:`AutoQuant<ug-auto-quant>` before applying QAT, especially if there is large drop in INT8 performance from the FP32 baseline.

Hyper-parameters
    - Number of epochs: 15-20 epochs are usually sufficient for convergence
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs


Workflow
========

Code example
------------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        Setup the model, data loader, and training loops for training.

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # setup
            :end-before: # step_1

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of dataset

Step 1
~~~~~~

Compute the initial quantization parameters.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_1
            :end-before: # step_2

        **Output**
        ::

            PTQ model accuracy: 0.68016

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

        **Output**
        ::

            PTQ model accuracy: 0.6583

Step 2
~~~~~~

Run the training loop on the quantized model to fine-tune the quantized accuracy.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_2
            :end-before: # step_3

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

Step 3
~~~~~~

Evaluate the fine-tuned model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_3
            :end-before: # step_4

        **Output**
        ::

            Model accuracy after QAT: 0.70838

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

        **Output**
        ::

            Model accuracy after QAT: 0.6910

Next steps
==========

After fine-tuning the model's quantized accuracy with QAT, export the updated weights and encodings.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_4

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4
