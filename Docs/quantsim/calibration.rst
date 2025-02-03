.. _quantsim-calibration:

###########
Calibration
###########

Calibration is the process of determining the appropriate scale and offset parameters for the quantizers added
to your model graph. Quantization parameters for weights can be precomputed. Computing quantization parameters for activation
require passing small, representative data samples through the model to gather range statistics.

Workflow
========

Use the following procedure to calibrate your model.

Prerequisites
-------------

Load your trained model.

.. note::

    The examples below use a pretrained MobileNetV2 model. Substitute your model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. important::

            aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x. If you are
            using low-level components of :class:`QuantizationSimModel`, see the :doc:`aimet_torch 1 to aimet_torch 2 Migration Guide<../apiref/torch/migration_guide>`.

        .. literalinclude:: ../snippets/torch/apply_quantsim.py
           :language: python
           :start-after: # PyTorch imports
           :end-before: # End of PyTorch imports

        To perform quantization simulation with :mod:`aimet_torch`, your model definition must conform to
        the guidelines at :ref:`PyTorch model guidelines <torch-model-guidelines>`.
        For example, :func:`torch.nn.functional` defined in the forward pass should be changed to the equivalent
        :class:`torch.nn.Module`.

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

        To perform quantization simulation with :mod:`aimet_torch`, your model definition must conform to
        the guidelines at :ref:`TensorFlow model guidelines <tensorflow-model-guidelines>`.
        For example, models defined using subclassing APIs should be converted to functional APIs.

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

            We recommend that you apply ONNX simplification before invoking AIMET API functions.

        .. literalinclude:: ../snippets/onnx/apply_quantsim.py
            :language: python
            :start-after: # Prepare model with onnx-simplifier
            :end-before:  # End of prepare model



Step 1: Creating a QuantSim model
---------------------------------

Use AIMET to create a :class:`QuantizationSimModel`. AIMET inserts
fake quantization operations in the model graph and configures them.

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


Step 2: Creating a calibration callback
---------------------------------------

Before you can use the :class:`QuantizationSimModel` for inference or training, you must compute
scale and offset quantization parameters for each 'quantizer' node.

Create a routine to pass small, representative data samples through the model. A quick way to do this
is to use the existing train or validation data loader to extract samples and pass them
to the model.

500 to 1000 representative data samples are sufficient to compute the quantization parameters.

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

Step 3: Computing encodings
---------------------------

Next, call :func:`QuantizationSimModel.compute_encodings` to use the callback to pass representative
data through the quantized model. The quantizers in the quantized model use the observed inputs
to initialize their quantization encodings. "Encodings" refers to the scale and offset quantization parameters.

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

Step 4: Evaluation
------------------

Next, evaluate the :class:`QuantizationSimModel` to compute quantized accuracy.

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

Step 5: Exporting the model
---------------------------

Lastly, export a version of the model with quantization operations removed and an encodings JSON
file containing quantization scale and offset parameters.

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

            - We recommend you use onnx-simplifier before creating the QuantSim model.
            - Since ONNX Runtime is used for optimized inference only, ONNX framework supports Post Training Quantization schemes (such as TF or TF-enhanced) to compute the encodings.

        .. autofunction:: aimet_onnx.quantsim.load_encodings_to_sim
            :noindex:

        **Quant Scheme Enum**

        .. autoclass:: aimet_common.defs.QuantScheme
            :members:
            :noindex:
