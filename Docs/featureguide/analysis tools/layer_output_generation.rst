.. include:: ../../abbreviation.txt

.. _featureguide-layer-output-generation:

#######################
Layer output generation
#######################

Context
=======

Layer output generation is an API that captures and saves intermediate layer-outputs of your pre-trained model. The model
can be original (FP32) or a :class:`QuantizationSimModel`.

The layer outputs are named according to the exported model (PyTorch, ONNX, or TensorFlow) by the
QuantSim export API :func:`QuantizationSimModel.export`.

This enables layer output comparison between quantization simulated (QuantSim) models
and quantized models on target runtimes like |qnn|_ to debug accuracy mismatch
issues at the layer level (per operation).

Workflow
========

The layer output generation framework follows the same workflow for all model frameworks:

1. Imports
2. Load a model from AIMET
3. Obtain inputs
4. Generate layer outputs
   

Choose your framework below for code examples.

Step 1: Importing the API
-------------------------

Import the API.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0


Step 2: Loading a model
-----------------------

Export the original or QuantSim model from AIMET.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1


Step 3: Obtaining inputs
------------------------

Obtain inputs from which to generate intermediate layer outputs.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2


Step 4: Generating layer outputs
--------------------------------

Generate the specified layer outputs.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../legacy/torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 3. Generate outputs
            :end-before: # End step 3

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../legacy/keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 3. Generate outputs
            :end-before: # End step 3

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../legacy/onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 3. Generate outputs
            :end-before: # End step 3


API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/layer_output_generation.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../../apiref/tensorflow/layer_output_generation.rst
           :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../../apiref/onnx/layer_output_generation.rst
           :start-after: # start-after
