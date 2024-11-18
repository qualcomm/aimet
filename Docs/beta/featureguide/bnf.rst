.. _featureguide-bnf:

##################
Batch norm folding
##################

Context
=======

Batch norm folding is a technique widely used in deep learning inference runtimes, including the Qualcomm Neural Processing SDK.
Batch normalization layers are typically folded into the weights and biases of adjacent convolution layers whenever possible to eliminate unnecessary computations.
To accurately simulate inference in these runtimes, it is generally advisable to perform batch norm folding on the floating-point model before applying quantization.
Doing so not only results in a speedup in inferences per second by avoiding unnecessary computations but also often improves the accuracy of the quantized model by removing redundant computations and requantization.
We aim to simulate this on-target behavior by performing batch norm folding here.

Workflow
========

Code example
------------

Step 1
~~~~~~

Load the model for batch norm folding.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Load the model for batch norm folding. In this code example, we will convert PyTorch MobileNetV2 to ONNX and use it in the subsequent code

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # Load exported ONNX model

        .. container:: tab-heading

            We can still find there are consecutive convolution and batch normalization nodes in ONNX graph

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Load exported ONNX model
            :end-before: # End of step 1

Step 2
~~~~~~

Apply preparation step if necessary

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            It's recommended to simplify the ONNX model before applying AIMET functionalities

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

Step 3
~~~~~~

Execute AIMET batch norm folding API

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Execute AIMET batch norm folding API

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

Step 4
~~~~~~

Result after batch norm folding

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            We can find that the weight have changed because the batch normalization has been folded into the convolution.
            Also, we can confirm that the batch normalization has been removed (the second node in the example appears as Clip)

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4


API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/bnf.rst
            :start-after: _apiref-torch-bnf:

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/bnf.rst
           :start-after: _apiref-keras-bnf:

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/bnf.rst
           :start-after: _apiref-onnx-bnf:
