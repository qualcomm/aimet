.. _quantsim-onnx:

#############
Quantsim ONNX
#############

Workflow
========

**Required imports**

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # imports start
   :end-before: # imports end

**Load MobileNetV2**

For this example, we are going to load a pretrained MobileNetV2 model from torchvision and convert it to ONNX
Similarly, you can use any ONNX model instead.

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Load the model
   :end-before:  # End of loading the model

**Prepare model with onnx-simplifier**

It's recommended to apply ONNX simplification before invoking AIMET functionalities

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Prepare model with onnx-simplifier
   :end-before:  # End of prepare model

**Set up dataloader**

We will use the ImageNet validation data in the subsequent code example

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Set up dataloader
   :end-before:  # End of setting up dataloader

**Create the Quantization Simulation Model**

Now we use AIMET to create a QuantizationSimModel. This basically means that AIMET will insert fake quantization ops in
the model graph and will configure them. A few of the parameters are explained here

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Create QuantSim object
   :end-before:  # End of creating QuantSim object

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

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :pyobject: pass_calibration_data

**Compute the Quantization Encodings**

Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization
encodings. Encodings here refer to scale/offset quantization parameters.

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Compute quantization encodings
   :end-before:  # End of computing quantization encodings

**Evaluate quantized accuracy**

Since we have calculated the quantization encodings, we can now evaluate the quantized accuracy

.. literalinclude:: ../../../onnx_code_examples/quantization.py
   :language: python
   :start-after: # Evaluate quantized accuracy
   :end-before:  # Enc of quantized accuracy

**Output**
        ::

        Quantized accuracy (W8A16): 0.7173

**Export the model**

So we have checked quantized accuracy. Now the next step would be to actually take this model to target. For this
purpose, we need to export the model with the updated weights without the fake quant ops. We also to export the
encodings (scale/offset quantization parameters) that were achieved from compute_encodings.
AIMET QuantizationSimModel provides an export API for this purpose.

.. literalinclude:: ../../../onnx_code_examples/quantization.py
    :language: python
    :start-after: # Export the model
    :end-before: # End of exporting the model


API
===

.. autoclass:: aimet_onnx.quantsim.QuantizationSimModel

**Note** :
 - It is recommended to use onnx-simplifier before creating quantsim model.
 - Since ONNX Runtime will be used for optimized inference only, ONNX framework will support Post Training Quantization schemes i.e. TF or TF-enhanced to compute the encodings.

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.compute_encodings

**The following API can be used to Export the Model to target**

.. automethod:: aimet_onnx.quantsim.QuantizationSimModel.export
