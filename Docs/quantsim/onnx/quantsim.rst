.. _quantsim-onnx:

#############
Quantsim ONNX
#############

Workflow
========

For this example, we are going to load a pretrained MobileNetV2 model from torchvision and convert it to ONNX.
Similarly, you can use any ONNX model instead.

QuantSim creation
-----------------

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # imports start
   :end-before: # imports end

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Load the model
   :end-before:  # End of loading the model

It's recommended to apply ONNX simplification before invoking AIMET functionalities

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Prepare model with onnx-simplifier
   :end-before:  # End of prepare model

Now we use AIMET to create a :class:`QuantizationSimModel`. This basically means that AIMET will insert
fake quantization operations in the model graph and will configure them. A few of the parameters are
explained here.

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Create QuantSim object
   :end-before:  # End of creating QuantSim object

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

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Set up dataloader
   :end-before:  # End of setting up dataloader

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Calibration callback
   :end-before:  # End of calibration callback

Compute encodings
~~~~~~~~~~~~~~~~~

Now we call :func:`QuantizationSimModel.compute_encodings` to use the above callback to pass small, representative
data through the quantized model. By doing so, the quantizers in the quantized model will observe the inputs
and initialize their quantization encodings according to the observed input statistics. Encodings here
refer to scale/offset quantization parameters.

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Compute quantization encodings
   :end-before:  # End of computing quantization encodings

Evaluation
----------

Next, we evaluate the :class:`QuantizationSimModel` to get quantized accuracy.

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
   :language: python
   :start-after: # Evaluate quantized accuracy
   :end-before:  # Enc of quantized accuracy

.. rst-class:: script-output

  .. code-block:: none

        Quantized accuracy (W8A16): 0.7173

Export
------

Lastly, export a version of the model with quantization operations removed and an encodings JSON
file with quantization scale and offset parameters for the model's activation and weight tensors.

.. literalinclude:: ../../snippets/onnx/apply_quantsim.py
    :language: python
    :start-after: # Export the model
    :end-before: # End of exporting the model

API
===

**Top level APIs**

.. autoclass:: aimet_onnx.quantsim.QuantizationSimModel
    :members: compute_encodings, export
    :member-order: bysource

**Note** :
 - It is recommended to use onnx-simplifier before creating quantsim model.
 - Since ONNX Runtime will be used for optimized inference only, ONNX framework will support Post Training Quantization schemes i.e. TF or TF-enhanced to compute the encodings.

.. autofunction:: aimet_onnx.quantsim.load_encodings_to_sim

**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:
