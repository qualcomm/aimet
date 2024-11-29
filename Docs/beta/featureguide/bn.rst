.. _featureguide-bn:

########################
Batch norm re-estimation
########################

Context
=======
If applying batch norm folding to your model negatively impacts performance, the batch norm re-estimation feature may be of use. This feature uses a small subset of training data to re-estimate the statistics of the batch norm (BN) layers in a model. Using the re-estimated statistics, the BN layers are folded into the preceding convolution or linear layers. 

BN re-estimation is also recommended in the following cases:

- Models where the main issue is weight quantization
- Quantization of depth-wise separable layers as their batch norm statistics are sensitive to oscillations

Workflow
========

Prerequisites
-------------
To use BN re-estimation, you must:

- Load a trained model
- Create a training dataloader for the model
- Hold off on folding the batch norm layers until after quantization aware training (QAT)

Setup
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bn.py
            :language: python
            :start-after: [setup]
            :end-before: [step_1]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_bn.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of set up

Step 1
------

Create the QuantizationSimModel 

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        When creating the QuantizationSimModel model, ensure that per channel quantization is enabled. Please update the config file if needed. 

        .. literalinclude:: ../snippets/torch/apply_bn.py
            :language: python
            :start-after: [step_1]
            :end-before: [step_2]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_bn.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

Step 2
------

Perform QAT 

This involves training your model for a few additional epochs (usually around 15-20). When training, be aware of the hyper-parameters being used. 

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bn.py
            :language: python
            :start-after: [step_2]
            :end-before: [step_3]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_bn.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        **Output**
        ::

            Model accuracy before BN re-estimation: 0.0428

Step 3
------

Re-estimate the BN statistics and fold the BN layers. 

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bn.py
            :language: python
            :start-after: [step_3]
            :end-before: [step_4]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_bn.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

        **Output**
        ::

            Model accuracy after BN re-estimation: 0.5876

Step 4
------

If BN re-estimation resulted in satisfactory accuracy, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bn.py
            :language: python
            :start-after: [step_4]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_bn.py
            :language: python
            :start-after: # Step 4
            :end-before: # End of step 4

API
===
.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/bn.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/bn.rst
            :start-after: # start-after
