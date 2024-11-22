.. _quantsim-torch:

################
Quantsim PyTorch
################

Workflow
========

**Required imports**

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # PyTorch imports
   :end-before: # End of PyTorch imports

**Load the PyTorch Model**

For this example, we are going to load a pretrained ResNet18 model from torchvision. Similarly, you can load any
pretrained PyTorch model instead.

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Load the model
   :end-before:  # Prepare the model

**Prepare the model for Quantization simulation**

AIMET quantization simulation requires the user's model definition to follow certain guidelines. For example,
functionals defined in forward pass should be changed to equivalent torch.nn.Module. AIMET user guide lists all these
guidelines. The following ModelPreparer API uses new graph transformation feature available in PyTorch 1.9+ version and
automates model definition changes required to comply with the above guidelines.

For more details, please refer:  :ref:`Model Preparer API<api-torch-model-preparer>`:

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Prepare the model
   :end-before:  # Create Quantization Simulation Model

**Create the Quantization Simulation Model**

Now we use AIMET to create a QuantizationSimModel. This basically means that AIMET will insert fake quantization ops in
the model graph and will configure them. A few of the parameters are explained here

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Create Quantization Simulation Model
   :end-before:  # Compute the Quantization Encodings

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

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: pass_calibration_data

**Compute the Quantization Encodings**

Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization
encodings. Encodings here refer to scale/offset quantization parameters.

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Compute the Quantization Encodings
   :end-before:  # Finetune the model

**Finetune the Quantization Simulation Model**

To perform quantization aware training (QAT), we simply train the model for a few more epochs (typically 15-20). As with
any training job, hyper-parameters need to be searched for optimal results. Good starting points are to use a learning
rate on the same order as the ending learning rate when training the original model, and to drop the learning rate by a
factor of 10 every 5 epochs or so.

For the purpose of this example, we are going to train only for 1 epoch. But feel free to change these parameters as you
see fit.

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Finetune the model
   :end-before:   # Export the model

**Export the model**

So we have an improved model after QAT. Now the next step would be to actually take this model to target. For this
purpose, we need to export the model with the updated weights without the fake quant ops. We also to export the
encodings (scale/offset quantization parameters) that were updated during training since we employed QAT.
AIMET QuantizationSimModel provides an export API for this purpose.

.. literalinclude:: ../../torch_code_examples/quantsim_code_example.py
    :language: python
    :start-after: # Export the model
    :end-before: # End of example

API
===

.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API. To
    learn more about the differences between :mod:`aimet_torch` and :mod:`aimet_torch.v2`, please visit the
    :ref:`QuantSim v2 Overview<ug-aimet-torch-v2-overview>`.

.. autoclass:: aimet_torch.v1.quantsim.QuantizationSimModel


**The following API can be used to Compute Encodings for Model:**

.. automethod:: aimet_torch.v1.quantsim.QuantizationSimModel.compute_encodings


**The following APIs can be used to save and restore the quantized model**

.. automethod:: aimet_torch.v1.quantsim.save_checkpoint

.. automethod:: aimet_torch.v1.quantsim.load_checkpoint


**The following API can be used to Export the Model to target:**

.. automethod:: aimet_torch.v1.quantsim.QuantizationSimModel.export

Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
