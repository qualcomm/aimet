.. _quantsim-qat:

###########################
Quantization-aware training
###########################

Quantization-aware training (QAT) finds better optimized solutions than post-training quantization (PTQ)
by fine-tuning the model parameters in the presence of quantization noise. This higher accuracy comes with
the usual costs of neural network training, including longer training times and the need for labeled data
and hyperparameter search.

QAT modes
=========

There are two versions of QAT: without range learning and with range learning.

Without range learning
      In QAT without range Learning, encoding values for activation quantizers are found once during calibration and are not updated again.

With range learning
      In QAT with range Learning, encoding values for activation quantizers are set during calibration and are updated during training, yielding better scale and offset quantization parameters.

In both versions, parameter quantizer encoding values are updated in sync with the parameters during training.

QAT recommendations
===================

These guidelines can improve performance and speed convergence with QAT.

Initialization
    - Apply PTQ techniques before applying QAT, especially if there is large drop in INT8 performance from the FP32 baseline.

Hyper-parameters
    - Number of epochs: 15-20 epochs are usually sufficient for convergence.
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs.

Workflow
========

Prerequisites
-------------

You need a PyTorch or TensorFlow model. ONNX does not support QAT.

.. _quantsim-qat-setup:

Step 1: Setup
-------------

Set up the model, data loader, and training callback.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch


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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.

.. _quantsim-qat-encodings:

Step 2: Computing the initial quantization parameters
-----------------------------------------------------

Compute the quantization parameters and calculate quantized accuracy.

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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.


.. _quantsim-qat-calibrate:

Step 3: Calibrate the quantized model
-------------------------------------

Train the model to fine-tune the parameters.

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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.

Step 4: Evaluating the model
----------------------------

Evaluate the :class:`QuantizationSimModel` to determine the improvement in accuracy.

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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.

Step 5: Exporting the model
---------------------------

Export the calibrated model to remove quantization operations and create the JSON encodings file containing quantization scale and offset parameters for the model's activation and weight tensors.

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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.

Multi-GPU support
=================

To use QAT with multi-GPU support, do the following. The instructions are the same as above except:

- Multi-GPU is supported only in PyTorch.
- There is an additional step to parallelize the model.
- It is important not to parallelize the model until after computing encodings.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. important::

            Do not invoke DataParallel or multi-GPU mode until after you compute the encodings (quantization parameters).

        **Step 1: Setup**

        Create a :class:`QuantizationSimModel` for your pre-trained PyTorch model per :ref:`Step 1 <quantsim-qat-setup>`. Do not use DataParallel mode.

        **Step 2: Compute encodings**

        Compute quantization encodings for the model per :ref:`Step 2 <quantsim-qat-encodings>`. Do not use a forward function that moves the model to multi-gpu and back.

        **Step 2.5 (additional step)**

        Move :class:`QuantizationSimModel` to DataParallel as follows.

            .. code-block:: python

                # "sim" here refers to the QuantizationSimModel object.
                sim.model = torch.nn.DataParallel(sim.model)

        **Steps 3 - 5**

        Evaluate, train, and export the model per :ref:`steps 3 - 5 <quantsim-qat-calibrate>`.

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.


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

    .. tab-item:: ONNX
        :sync: ONNX

        Not supported.

