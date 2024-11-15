.. _featureguide-adaround:

#################
Adaptive rounding
#################

Context
=======

By default, AIMET uses *nearest rounding* for quantization, in which weight values are quantized to the nearest integer value.

AIMET adaptive rounding (AdaRound) uses training data to choose how to round quantized weights, improving the quantized model's accuracy in many cases.

The following figures illustrates how AdaRound might change the rounding of a quantized value.

.. image:: ../images/adaround.png
    :width: 600px

See the :doc:`Optimization User Guide <../optimization/index>` for a discussion of the recommended sequence of all quantization techniques.

Complementary techniques
------------------------

We recommend using AdaRound in combination with these other techniques:

- After batch norm folding (BNF) and cross layer equalization (CLE). Applying these techniques first might improve the accuracy gained using AdaRound.
- Before quantization aware training (QAT). For some models applying BNF and CLE may not help. For these models, applying AdaRound before QAT might help. AdaRound is a better weights initialization step that speeds up QAT

Conversely, we recommend that you *do not* apply bias correction (BC) before or after using AdaRound. 

Hyper parameters
----------------

A number of hyper parameters used during AdaRound optimization are exposed in the API. The default values of some of these parameters tend to lead to stable results and we recommend that you not change them.

Use the following guideline for adjusting hyper parameters with AdaRound.

Hyper Parameters to be changed at will
    - Number of batches. AdaRound should see 500-1000 images. Loader batch size times number of batches gives the number of images. For example if the data loader batch size is 64, set 16  batches to yield 1024 images.
    - Number of iterations. Default is 10,000.

Hyper Parameters to change with caution
    Regularization parameter. Default is 0.01.

Hyper Parameters to avoid changing
    - Beta range. Leave the value at the default of (20, 2).
    - Warm start period. Leave at the default value, 20%.

Workflow
========

Prerequisites
-------------

To use AdaRound, you must:

- Load a trained model
- Create a training or validation dataloader for the model

Workflow
--------

Step 1
~~~~~~

Prepare the model for quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. container:: tab-heading
    
            Prepare the model for quantization

        AIMET quantization simulation (QuantSim) for PyTorch requires the user's model definition to follow certain guidelines. For example, functionals defined in forward pass should be changed to an equivalent 
        **torch.nn.Module**. For a list of these guidelines, see the :ref:`Optimization Guide <opt-guide-quantization>`. 

        Use the :ref:`AIMET ModelPreparer API <apiref-torch-model-preparer>` graph transformation feature to automate the model definition changes required to comply with the QuantSim guidelines.

        .. literalinclude:: ../snippets/torch/prepare_model.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

        For details of the model preparer API see the 
        :ref:`Model Preparer API <apiref-torch-model-preparer>`.

    .. tab-item:: TensorFlow
        :sync: tf

        Tensorflow has no preparation requirements.

    .. tab-item:: ONNX
        :sync: onnx

        ONNX has no preparation requirements.


Step 2
~~~~~~

Apply AdaRound to the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adaround.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_adaround.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adaround.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

Step 3
~~~~~~

Evaluate the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/evaluate.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/evaluate.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/evaluate.py
            :language: python
            :start-after: # pylint: disable=missing-docstring


Results
-------

AdaRound should result in improved accuracy, but does not guaranteed sufficient improvement.


Next steps
----------

If AdaRound resulted in satisfactory accuracy, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/export.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/export.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/export.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

If the model is still not accurate enough, the next step is typically to try :ref:`quantization-aware training <featureguide-qat>`.


API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/adaround.rst
            :start-after: _apiref-torch-adaround:

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/adaround.rst
           :start-after: _apiref-keras-adaround:

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/adaround.rst
           :start-after: _apiref-onnx-adaround:
