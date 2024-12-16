:orphan:

.. _api-torch-adaround:

==================================
AIMET PyTorch AdaRound API
==================================

User Guide Link
===============
To learn more about this technique, please see :ref:`AdaRound<ug-adaround>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch AdaRound, please see :doc:`here<../Examples/torch/quantization/adaround>`.

Top-level API
=============


.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API. To
    learn more about the differences between :mod:`aimet_torch` and :mod:`aimet_torch.v2`, please visit the
    :ref:`QuantSim v2 Overview<ug-aimet-torch-v2-overview>`.

.. autofunction:: aimet_torch.v1.adaround.adaround_weight.Adaround.apply_adaround


Adaround Parameters
===================
.. autoclass:: aimet_torch.v1.adaround.adaround_weight.AdaroundParameters
    :members:


Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:

|

Code Example - Adaptive Rounding (AdaRound)
===========================================

This example shows how to use AIMET to perform Adaptive Rounding (AdaRound).

**Load the model**

For this example, we are going to load a pretrained ResNet18 model from torchvision. Similarly, you can load any
pretrained PyTorch model instead.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Load the model
   :end-before:  # Prepare the model

**Prepare the model for Quantization simulation**

AIMET quantization simulation requires the user's model definition to follow certain guidelines. For example,
functionals defined in forward pass should be changed to equivalent torch.nn.Module. AIMET user guide lists all these
guidelines. The following ModelPreparer API uses new graph transformation feature available in PyTorch 1.9+ version and
automates model definition changes required to comply with the above guidelines.

For more details, please refer:  :ref:`Model Preparer API<api-torch-model-preparer>`:

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Prepare the model
   :end-before:  # Apply AdaRound

**Apply AdaRound**

We can now apply AdaRound to this model.

Some of the parameters for AdaRound are described below

* **dataloader**: AdaRound needs a dataloader to use data samples for the layer-by-layer optimization to learn the rounding vectors. Either a training or validation dataloader could be passed in.

* **num_batches**: The number of batches used to evaluate the model while calculating the quantization encodings. Typically we want AdaRound to use around 2000 samples. So with a batch size of 32, this may translate to 64 batches. To speed up the execution here we are using a batch size of 1.

* **default_num_iterations**: The number of iterations to adaround each layer. Default value is set to 10000 and we strongly recommend to not reduce this number. But in this example we are using 32 to speed up the execution runtime.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Apply AdaRound
   :end-before:  # Create Quantization Simulation using the adarounded_model


**Create the Quantization Simulation Model**

Now we use AdaRounded model and create a QuantizationSimModel. This basically means that AIMET will insert fake
quantization ops in the model graph and will configure them. A few of the parameters are explained here

* **default_param_bw**: The QuantizationSimModel must be created with the same parameter bitwidth precision that was used in the apply_adaround() created.

* **Freezing the parameter encodings**: After creating the QuantizationSimModel, the set_and_freeze_param_encodings() API must be called before calling the compute_encodings() API. While applying AdaRound, the parameter values have been rounded up or down based on these initial encodings internally created. Fo r Quantization Simulation accuracy, it is important to freeze these encodings. If the parameters encodings are NOT frozen, the call to compute_encodings() will alter the value of the parameters encoding and Quantization Simulation accuracy will not be correct.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Create Quantization Simulation using the adarounded_model
   :end-before:  # Compute encodings

**An example User created function that is called back from compute_encodings()**

Even though AIMET has added 'quantizer' nodes to the model graph, the model is not ready to be used yet. Before we can
use the sim model for inference or training, we need to find appropriate scale/offset quantization parameters for each
'quantizer' node. For activation quantization nodes, we need to pass unlabeled data samples through the model to collect
range statistics which will then let AIMET calculate appropriate scale/offset quantization parameters. This process is
sometimes referred to as calibration. AIMET simply refers to it as 'computing encodings'.

So we create a routine to pass unlabeled data samples through the model. This should be fairly simple - use the existing
train or validation data loader to extract some samples and pass them to the model. We don't need to compute any
loss metric etc. So we can just ignore the model output for this purpose. A few pointers regarding the data samples

In practice, we need a very small percentage of the overall data samples for computing encodings. For example,
the training dataset for ImageNet has 1M samples. For computing encodings we only need 500 or 1000 samples.

It may be beneficial if the samples used for computing encoding are well distributed. It's not necessary that all
classes need to be covered etc. since we are only looking at the range of values at every layer activation. However,
we definitely want to avoid an extreme scenario like all 'dark' or 'light' samples are used - e.g. only using pictures
captured at night might not give ideal results.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :pyobject: pass_calibration_data

**Compute the Quantization Encodings**

Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization
encodings. Encodings here refer to scale/offset quantization parameters.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Compute encodings
   :end-before:  # Determine simulated accuracy

**Determine Simulated Accuracy**

Now the QuantizationSim model is ready to be used for inference. First we can pass this model to an evaluation routine.
The evaluation routine will now give us a simulated quantized accuracy score for INT8 quantization.

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # Determine simulated accuracy
   :end-before:  # Export the model

**Export the model**

So we have an improved model after AdaRound. Now the next step would be to actually take this model to target. For this
purpose, we need to export the model with the updated weights without the fake quant ops. We also to export the
encodings (scale/offset quantization parameters) that were updated during training since we employed QAT.
AIMET QuantizationSimModel provides an export API for this purpose.

.. literalinclude:: ../torch_code_examples/adaround.py
    :language: python
    :start-after: # Export the model
    :end-before: # End of example
