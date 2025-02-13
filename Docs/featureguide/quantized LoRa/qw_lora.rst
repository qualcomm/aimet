.. _featureguide-qw-lora:

######################
QW-LoRa
######################

Context
=======

The QW-LoRa workflow involves determining the appropriate weight encodings for the base model before
performing some epochs of LoRa training. Finally, the activation encodings for the base model and weight and
activation encodings for the updated LoRa layers are calibrated. This is expressed in the block diagram below.

.. image:: ../../images/qw_lora_block_diagram.png
    :width: 900px

This workflow is especially useful if you have precomputed encodings for the weights of your model (using any technique)
and applied those encodings to your model (so that the model parameters have already been updated).

Workflow
========

Setup
-----

In this section, we instantiate the base model, LoRa adapters, and dataset using Huggingface APIs.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [setup]
            :end-before: [freeze_base_model_weights]

Quantize and Update Base Model Weights
-----
Step #1 of QW-LoRa

In this section, we calculate quantization parameters for the base model weights, and use those parameters to update
the weights. A helper function to do this is provided below if you do not already have a method of doing this.

.. note::
    The provided helper function applies 4-bit symmetric integer quantization to all model parameters. This function can
    be updated to suit your quantization requirements. In fact, you can use any method of calculating weight encodings,
    as long as these encodings are applied directly to the weights.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [freeze_base_model_weights]
            :end-before: [lora_training]

LoRa Training
-----
Step #2 of QW-LoRa


.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [lora_training]
            :end-before: [ptq]

PTQ
-----
Step #3 of QW-LoRa


.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [ptq]
