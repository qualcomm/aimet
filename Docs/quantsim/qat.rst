.. _quantsim-qat:

###########################
Quantization-aware training
###########################

Quantization-aware training (QAT) finds better-optimized solutions than post-training quantization  (PTQ)
by fine-tuning the model parameters in the presence of quantization noise. This higher accuracy comes at
the usual cost of neural network training, including longer training times and the need for labeled data
and hyperparameter search.

QAT modes
=========

There are two versions of QAT: without range learning and with range learning.

#. Without range learning:
    * In QAT without range Learning, encoding values for activation quantizers are found once during
      calibration and are not updated again.

#. With range learning:
    * In QAT with range Learning, encoding values for activation quantizers are set during calibration and can
      be updated during training, resulting in better scale and offset quantization parameters.

In both versions, parameter quantizer encoding values continue to be updated with the parameters themselves
during training.

QAT recommendations
===================

Here are some guidelines that can improve performance and speed convergence with QAT:

Initialization
    - It often helps to first apply PTQ techniques before applying QAT, especially if there is large drop in INT8 performance from the FP32 baseline.

Hyper-parameters
    - Number of epochs: 15-20 epochs are usually sufficient for convergence
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs

Workflow
========

Setup
-----

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

Compute the initial quantization parameters
-------------------------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_1
            :end-before: # step_2

        .. rst-class:: script-output

            .. code-block:: none

                Quantized accuracy (W8A8): 0.68016

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

        .. rst-class:: script-output

            .. code-block:: none

                Quantized accuracy (W8A8): 0.6583

Fine-tune quantized model
-------------------------

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

Evaluation
----------

Next, we evaluate the :class:`QuantizationSimModel` to get quantized accuracy.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_qat.py
            :language: python
            :start-after: # step_3
            :end-before: # step_4

        .. rst-class:: script-output

            .. code-block:: none

                Model accuracy after QAT: 0.70838

    .. tab-item:: TensorFlow
        :sync: tf

         .. literalinclude:: ../snippets/tensorflow/apply_qat.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

        .. rst-class:: script-output

            .. code-block:: none

                Model accuracy after QAT: 0.6910

Export
------

After fine-tuning the model's quantized accuracy with QAT, export a version of the model with quantization
operations removed and an encodings JSON file with quantization scale and offset parameters for the
model's activation and weight tensors.

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

Multi-GPU support
=================

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        For using multi-GPU with QAT,

        1. Create a :class:`QuantizationSimModel` for your pre-trained PyTorch model (Not in DataParallel mode)
        2. Perform :func:`QuantizationSimModel.compute_encodings` (NOTE: Do not use a forward function that moves the model to multi-gpu and back)
        3. Move :class:`QuantizationSimModel` to DataParallel.

        .. code-block:: python

            # "sim" here refers to QuantizationSimModel object.
            sim.model = torch.nn.DataParallel(sim.model)

        4. Perform eval and/or training.
        5. Export for on-target inference.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Top level APIs**

        .. autoclass:: aimet_torch.quantsim.QuantizationSimModel
            :members: compute_encodings, export, load_encodings
            :member-order: bysource
            :no-index:

        **Quant Scheme Enum**

        .. autoclass:: aimet_common.defs.QuantScheme
            :members:
            :no-index:

    .. tab-item:: TensorFlow
        :sync: tf

        **Top level APIs**

        .. autoclass:: aimet_tensorflow.keras.quantsim.QuantizationSimModel
            :members: compute_encodings, export, load_encodings_to_sim
            :member-order: bysource
            :no-index:

        **Quant Scheme Enum**

        .. autoclass:: aimet_common.defs.QuantScheme
            :members:
            :no-index:
