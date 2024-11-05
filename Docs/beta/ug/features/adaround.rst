.. _feature-adaround:

#################
Adaptive rounding
#################

Context
=======

.. include:: ../user_guide/adaround.rst
    :start-after: adaround-context
    :end-before: adaround-api

Prerequisites
-------------

Model, GPU, CUDA, dataloaders, dependencies.

Workflow
--------

Step 1
~~~~~~

.. tabs::

    .. tab:: PyTorch

        PyTorch code example.

        .. literalinclude:: ../torch_code_examples/adaround.py
            :language: python
            :pyobject: apply_adaround_example

    .. tab:: TensorFlow

        Keras code example.

        .. literalinclude:: ../keras_code_examples/adaround.py
            :language: python
            :pyobject: apply_adaround_example

    .. tab:: ONNX

        ONNX code example.

        .. literalinclude:: ../onnx_code_examples/adaround.py
            :language: python
            :pyobject: apply_adaround_example

Step 2
~~~~~~

... and so on.


Results
-------

Optional.

AdaRound should result in improved accuracy, but does not guaranteed sufficient improvement.


Next steps
----------

If AdaRound resulted in satisfactory accuracy, export the model.

.. tabs::

    .. tab:: PyTorch

        Link to PyTorch export procedure.

    .. tab:: TensorFlow

        Link to TensorFlow export procedure.

    .. tab:: ONNX

        Link to ONNX export procedure.

If the model is still not accurate enough, the next step is typically to try :doc:`quantization-aware training <qat>`.


API
===

.. tabs::

    .. tab:: PyTorch

        :ref:`PyTorch API <api-torch-adaround>`

    .. tab:: TensorFlow

        :ref:`Keras API <api-keras-adaround>`

    .. tab:: ONNX

        :ref:`ONNX API <api-onnx-adaround>`
