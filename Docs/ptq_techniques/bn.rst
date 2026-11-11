.. _ptq-bn-estimate:

########################
Batch norm re-estimation
########################

.. note::
    Not supported in ``aimet-onnx``.

Context
=======

Batch norm re-estimation (BN re-estimation) uses a small subset of training data to re-estimate the statistics of the batch norm (BN) layers in a model. AIMET then folds the BN layers into the preceding convolution or linear layers.

BN re-estimation is recommended under the following conditions:

- When :ref:`batch norm folding <ptq-bnf>` (BNF) reduces performance
- In models where the main issue is weight quantization
- In quantization of depth-wise separable layers, as their batch norm statistics are sensitive to oscillations

Workflow
========

Prerequisites
-------------

To use BN re-estimation, you must:

- Load a trained model
- Create a training dataloader for the model
- Hold off on folding the batch norm layers until after quantization aware training (QAT)

Execution
---------

Setup
~~~~~

.. literalinclude:: ../snippets/torch/apply_bn.py
    :language: python
    :start-after: [setup]
    :end-before: [step_1]

Step 1
~~~~~~

Create the quantization simulation model (QuantizationSimModel).

.. literalinclude:: ../snippets/torch/apply_bn.py
    :language: python
    :start-after: [step_1]
    :end-before: [step_2]

Step 2
~~~~~~

Perform Quantization-aware training (QAT).

QAT involves training your model for a few additional epochs (usually 15-20). When training, be aware of the hyper-parameters being used.

.. literalinclude:: ../snippets/torch/apply_bn.py
    :language: python
    :start-after: [step_2]
    :end-before: [step_3]

.. rst-class:: script-output

  .. code-block:: none

    Model accuracy before BN re-estimation: 0.0428

Step 3
~~~~~~

Re-estimate the BN statistics and fold the BN layers.

.. literalinclude:: ../snippets/torch/apply_bn.py
    :language: python
    :start-after: [step_3]
    :end-before: [step_4]

.. rst-class:: script-output

  .. code-block:: none

    Model accuracy after BN re-estimation: 0.5876

Step 4
~~~~~~

If BN re-estimation resulted in satisfactory accuracy, export the model.

.. literalinclude:: ../snippets/torch/apply_bn.py
    :language: python
    :start-after: [step_4]

API
===

.. include:: ../apiref/torch/bn.rst
    :start-after: # start-after
