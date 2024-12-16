.. include:: ../../abbreviation.txt

.. _featureguide-layer-output-generation:

#######################
Layer output generation
#######################

Context
=======

This API captures and saves intermediate layer-outputs of your pre-trained model. The model
can be original (FP32) or :class:`QuantizationSimModel`.

The layer-outputs are named according to the exported PyTorch/ONNX/TensorFlow model by the
QuantSim export API :func:`QuantizationSimModel.export`.

This allows layer-output comparison amongst quantization simulated model (QuantSim)
and actually quantized model on target-runtimes like |qnn|_ to debug accuracy miss-match
issues at the layer level (per operation).

Workflow
========

Code example
------------

Step 1 Obtain Original or QuantSim model from AIMET Export Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../../torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Obtain Original or QuantSim model from AIMET Export Artifacts**

        .. literalinclude:: ../../../torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../../keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Obtain Original or QuantSim model from AIMET Export Artifacts**

        .. literalinclude:: ../../../keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../../onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Obtain Original or QuantSim model from AIMET Export Artifacts**

        .. literalinclude:: ../../../onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 1. Obtain original or quantsim model
            :end-before: # End step 1

Step 2 Generate layer-outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Obtain inputs for which we want to generate intermediate layer-outputs**

        .. literalinclude:: ../../../torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2

        **Generate layer-outputs**

        .. literalinclude:: ../../../torch_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 3. Generate outputs
            :end-before: # End step 3

    .. tab-item:: TensorFlow
        :sync: tf

        **Obtain inputs for which we want to generate intermediate layer-outputs**

        .. literalinclude:: ../../../keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2

        **Generate layer-outputs**

        .. literalinclude:: ../../../keras_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 3. Generate outputs
            :end-before: # End step 3

    .. tab-item:: ONNX
        :sync: onnx

        **Obtain inputs for which we want to generate intermediate layer-outputs**

        .. literalinclude:: ../../../onnx_code_examples/layer_output_generation_code_example.py
            :language: python
            :start-after: # Step 2. Obtain pre-processed inputs
            :end-before: # End step 2

        **Generate layer-outputs**

        .. literalinclude:: ../../../onnx_code_examples/layer_output_generation_code_example.py
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
