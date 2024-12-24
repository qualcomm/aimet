.. _quantsim-tensorflow:

###################
Quantsim TensorFlow
###################

.. toctree::
    :hidden:

    Model guidelines <model_guidelines>

* :ref:`TensorFlow model guidelines <tensorflow-model-guidelines>`
* :ref:`Workflow <tensorflow-workflow>`

.. _tensorflow-workflow:

Workflow
========

For this example, we are going to load a pretrained MobileNetV2 from TensorFlow applications. Similarly, you can load any
pretrained TensorFlow model instead.

QuantSim creation
-----------------

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # pylint: skip-file
    :end-before: # End of imports

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Load the model
    :end-before: # End of loading model

Now we use AIMET to create a :class:`QuantizationSimModel`. This basically means that AIMET will insert
fake quantization operations in the model graph and will configure them. A few of the parameters are
explained here.

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Create QuantSim object
    :end-before: # End of creating QuantSim object

Calibration
-----------

Even though AIMET has added 'quantizer' operations to the model graph, the QuantSim is not ready to be used
yet. Before we can use the QuantSim for inference or training, we need to find appropriate scale/offset
quantization parameters for each 'quantizer' node. For activation quantization nodes, we need to pass
small, representative data samples through the model to collect range statistics which will then let AIMET calculate
appropriate scale/offset quantization parameters.

Calibration callback
~~~~~~~~~~~~~~~~~~~~

So we create a routine to pass small, representative data samples through the model. This should be
fairly simple - use the existing train or validation data loader to extract some samples and pass them
to the model.

In practice, for computing encodings we only need 500-1000 representative data samples.

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Set up dataset
    :end-before: # End of dataset

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Calibration callback
    :end-before: # End of calibration callback

Compute encodings
~~~~~~~~~~~~~~~~~

Now we call :func:`QuantizationSimModel.compute_encodings` to use the above callback to pass data through
the model and then subsequently compute the quantization encodings. Encodings here refer to scale/offset
quantization parameters.

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Compute quantization encodings
    :end-before: # End of computing quantization encodings

Evaluation
----------

Next, we evaluate the :class:`QuantizationSimModel` to get quantized accuracy.

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Evaluation
    :end-before: # End of evaluation

.. rst-class:: script-output

  .. code-block:: none

    Quantized accuracy (W8A16): 0.7013

Export
------

Lastly, export a version of the model with quantization operations removed and an encodings JSON
file with quantization scale and offset parameters for the model's activation and weight tensors.

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Export the model
    :end-before: # End of exporting the model

.. _tensorflow-qat:

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

.. literalinclude:: ../../snippets/tensorflow/apply_quantsim.py
    :language: python
    :start-after: # Perform QAT
    :end-before: # End of export

.. rst-class:: script-output

  .. code-block:: none

    Model accuracy after QAT: 0.7032

API
===

**Top level APIs**

.. autoclass:: aimet_tensorflow.keras.quantsim.QuantizationSimModel
    :members: compute_encodings, export, load_encodings_to_sim
    :member-order: bysource

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
