.. _quantsim-torch:

################
Quantsim PyTorch
################

.. toctree::
    :hidden:

    Migration guide <migration_guide>
    Model guidelines <model_guidelines>
    Multi-GPU support <multi_gpu>
    Per-block quantization <block_quantization>
    Quantizers <quantizers>

.. important::

    aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x. If you are
    using low-level components of :class:`QuantizationSimModel`, please see :doc:`Migrate to aimet_torch 2 <migration_guide>`.


* :ref:`Migrate from aimet_torch 1.x to aimet_torch 2 <torch-migration-guide>`
* :ref:`PyTorch model guidelines <torch-model-guidelines>`
* :ref:`Workflow <torch-workflow>`

* Advanced
    * :ref:`Quantized modules <torch-nn>`
    * :ref:`Quantizers <torch-quantizers>`
    * :ref:`Per-block quantization <torch-per-block-quantization>`
    * :ref:`Quantization-aware training <torch-qat>`

.. _torch-workflow:

Workflow
========

For this example, we are going to load a pretrained MobileNetV2 model from torchvision. Similarly,
you can load any pretrained PyTorch model instead.

QuantSim creation
-----------------

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # PyTorch imports
   :end-before: # End of PyTorch imports

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # Load the model
   :end-before:  # End of load the model

Model preparation
~~~~~~~~~~~~~~~~~

AIMET quantization simulation requires the user's model definition to follow certain guidelines. For
example, :func:`torch.nn.functional` defined in forward pass should be changed to equivalent
:class:`torch.nn.Module`.

For more details on model definition guidelines and how :func:`prepare_model` API automates model
definition changes required to comply with the guidelines, please refer: :ref:`PyTorch model guidelines <torch-model-guidelines>`.

.. note::

    :func:`prepare_model` function uses :mod:`torch.fx` under the hood, which means it inherits all the limitations of :mod:`torch.fx`. Therefore, if :func:`prepare_model` cannot automatically prepare the model, you will need to manually adjust the model definition to comply with the model definition guidelines.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # Prepare the model
   :end-before:  # End of prepare_model

BatchNorm fold
~~~~~~~~~~~~~~

When models are executed in a quantized runtime, BatchNorm layers are typically folded into the weight
and bias of an adjacent convolution layer whenever possible in order to remove unnecessary computations.
To accurately simulate inference in these runtimes, it is generally a good idea to perform this BatchNorm
folding on the floating-point (FP32) model before applying quantization. AIMET provides the
:mod:`batch_norm_fold` API to do this.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # Fold the batchnorm
   :end-before:  # End of fold_all_batch_norms

Now we use AIMET to create a :class:`QuantizationSimModel`. This basically means that AIMET will insert
fake quantization operations in the model graph and will configure them. A few of the parameters are
explained here.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # Create Quantization Simulation Model
   :end-before:  # End of QuantizationSimModel

Calibration
-----------

Even though AIMET has added 'quantizer' operations to the model graph, the QuantSim is not ready to be used
yet. Before we can use the QuantSim for inference or training, we need to find appropriate scale/offset
quantization parameters for each 'quantizer' node. For activation quantization nodes, we need to pass
unlabeled data samples through the model to collect range statistics which will then let AIMET calculate
appropriate scale/offset quantization parameters.

Calibration callback
~~~~~~~~~~~~~~~~~~~~

So we create a routine to pass unlabeled representative data samples through the model. This should be
fairly simple - use the existing train or validation data loader to extract some samples and pass them
to the model.

In practice, we need a very small percentage of the overall data samples for computing encodings.
For computing encodings we only need 500 or 1000 representative data samples.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :pyobject: pass_calibration_data

Compute encodings
~~~~~~~~~~~~~~~~~

Now we call :func:`QuantizationSimModel.compute_encodings` to use the above callback to pass data through
the model and then subsequently compute the quantization encodings. Encodings here refer to scale/offset
quantization parameters.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :pyobject: get_calibration_and_eval_data_loaders

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
   :language: python
   :start-after: # Compute the Quantization Encodings
   :end-before:  # End of compute_encodings

Export
------

Lastly, evaluate the :class:`QuantizationSimModel` to get quantized accuracy and export a version
of the model with quantization operations removed and create an encodings JSON file with quantization
scale and offset parameters for the model's activation and weight tensors.

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
    :language: python
    :start-after: # Export the model for on-target inference.
    :end-before: # End of export

.. _torch-qat:

Quantization-aware training
===========================

Quantization-aware training (QAT) finds better-optimized solutions than post-training quantization  (PTQ)
by fine-tuning the model parameters in the presence of quantization noise. This higher accuracy comes at
the usual cost of neural network training, including longer training times and the need for labeled data
and hyperparameter search.

QAT modes
---------

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
-------------------

Here are some guidelines that can improve performance and speed convergence with QAT:

Initialization
    - It often helps to first apply PTQ techniques before applying QAT, especially if there is large drop in INT8 performance from the FP32 baseline.

Hyper-parameters
    - Number of epochs: 15-20 epochs are usually sufficient for convergence
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs

.. literalinclude:: ../../snippets/torch/apply_quantsim.py
    :language: python
    :start-after: # Finetune the model
    :end-before: # End of example

Multi-GPU support
-----------------

For using multi-GPU with QAT,

#. Create a :class:`QuantizationSimModel` for your pre-trained PyTorch model (Not in DataParallel mode)
#. Perform :func:`QuantizationSimModel.compute_encodings` (NOTE: Do not use a forward function that moves the model to multi-gpu and back)
#. Move Quantsim model to DataParallel::

    # "quant_sim" here refers to QuantizationSimModel object.
    quant_sim.model = torch.nn.DataParallel(quant_sim.model)

#. Perform eval and/or training.

API
===

**Top level APIs**

.. autoclass:: aimet_torch.quantsim.QuantizationSimModel
    :members: compute_encodings, export, load_encodings
    :member-order: bysource

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
