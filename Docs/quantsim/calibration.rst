.. _quantsim-calibration:

###########
Calibration
###########

Calibration involves determining the appropriate scale and offset parameters for the quantizers added
to your model graph. While quantization parameters for weights can be precomputed, activation quantization
requires passing small, representative data samples through the model to gather range statistics and
identify the appropriate scale and offset parameters.

Workflow
========

In this example, we will load a pretrained MobileNetV2 model. Similarly, you can use any pretrained model
you prefer.

QuantSim creation
-----------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. important::

            aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x. If you are
            using low-level components of :class:`QuantizationSimModel`, please see :doc:`Migrate to aimet_torch 2 <../apiref/torch/migration_guide>`.

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # PyTorch imports
           :end-before: # End of PyTorch imports

        To perform quantization simulation with :mod:`aimet_torch`, your model definition should adhere to specific guidelines. For
        example, :func:`torch.nn.functional` defined in forward pass should be changed to equivalent
        :class:`torch.nn.Module`. For more details on model definition guidelines, please refer: :ref:`PyTorch model guidelines <torch-model-guidelines>`.

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Load the model
           :end-before:  # End of load the model

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # pylint: skip-file
            :end-before: # End of imports

        To perform quantization simulation with :mod:`aimet_tensorflow`, your model definition must follow specific guidelines.
        For instance, models defined using subclassing APIs should be converted to functional APIs. For more
        details on model definition guidelines, please refer: :ref:`TensorFlow model guidelines <tensorflow-model-guidelines>`.

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Load the model
            :end-before: # End of loading model

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # imports start
            :end-before: # imports end

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Load the model
            :end-before:  # End of loading the model

        .. note::

            It's recommended to apply ONNX simplification before invoking AIMET functionalities.

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Prepare model with onnx-simplifier
            :end-before:  # End of prepare model

Now we use AIMET to create a :class:`QuantizationSimModel`. This basically means that AIMET will insert
fake quantization operations in the model graph and will configure them.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Create Quantization Simulation Model
           :end-before:  # End of QuantizationSimModel

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Create QuantSim object
            :end-before: # End of creating QuantSim object

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Create QuantSim object
            :end-before:  # End of creating QuantSim object


Calibration callback
--------------------

Even though AIMET has added 'quantizer' operations to the model graph, the :class:`QuantizationSimModel` object is not ready to be used
yet. Before we can use the :class:`QuantizationSimModel` for inference or training, we need to find appropriate scale/offset
quantization parameters for each 'quantizer' node.

So we create a routine to pass small, representative data samples through the model. This should be
fairly simple - use the existing train or validation data loader to extract some samples and pass them
to the model.

In practice, for computing encodings we only need 500-1000 representative data samples.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Dataloaders
           :end-before:  # End of dataloaders

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Calibration callback
           :end-before:  # End of calibration callback

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Set up dataset
            :end-before: # End of dataset

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Calibration callback
            :end-before: # End of calibration callback

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Set up dataloader
            :end-before:  # End of setting up dataloader

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Calibration callback
            :end-before:  # End of calibration callback

Compute encodings
~~~~~~~~~~~~~~~~~

Now we call :func:`QuantizationSimModel.compute_encodings` to use the above callback to pass small, representative
data through the quantized model. By doing so, the quantizers in the quantized model will observe the inputs
and initialize their quantization encodings according to the observed input statistics. Encodings here
refer to scale/offset quantization parameters.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Compute the Quantization Encodings
           :end-before:  # End of compute_encodings

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Compute quantization encodings
            :end-before: # End of computing quantization encodings

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Compute quantization encodings
            :end-before:  # End of computing quantization encodings

Evaluation
----------

Next, we evaluate the :class:`QuantizationSimModel` to get quantized accuracy.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # Evaluation
           :end-before:  # End of evaluation

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Evaluation
            :end-before: # End of evaluation

        .. rst-class:: script-output

            .. code-block:: none

                Quantized accuracy (W8A16): 0.7013

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
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

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
            :language: python
            :start-after: # Export
            :end-before: # End of export

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_quantsim.py
            :language: python
            :start-after: # Export the model
            :end-before: # End of exporting the model

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Export the model
            :end-before: # End of exporting the model

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
            :noindex:

    .. tab-item:: TensorFlow
        :sync: tf

        **Top level APIs**

        .. autoclass:: aimet_tensorflow.keras.quantsim.QuantizationSimModel
            :members: compute_encodings, export, load_encodings_to_sim
            :member-order: bysource
            :noindex:

        **Quant Scheme Enum**

        .. autoclass:: aimet_common.defs.QuantScheme
            :members:
            :noindex:

    .. tab-item:: ONNX
        :sync: onnx


        **Top level APIs**

        .. autoclass:: aimet_onnx.quantsim.QuantizationSimModel
            :members: compute_encodings, export
            :member-order: bysource
            :noindex:

        .. note::

            - It is recommended to use onnx-simplifier before creating quantsim model.
            - Since ONNX Runtime will be used for optimized inference only, ONNX framework will support Post Training Quantization schemes i.e. TF or TF-enhanced to compute the encodings.

        .. autofunction:: aimet_onnx.quantsim.load_encodings_to_sim
            :noindex:

        **Quant Scheme Enum**

        .. autoclass:: aimet_common.defs.QuantScheme
            :members:
            :noindex:
