.. _featureguide-qw-lora:

######################
QW-LoRa
######################

Context
=======

The QWA-LoRa workflow involves determining the appropriate weight encodings for the base model before
performing some epochs of LoRa training. Finally, the activation encodings for the base model; and weight and
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
            :end-before: [create_quantsim]

Create QuantizationSimModel
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [create_quantsim]
            :end-before: [calibration_callback]

Calibration Callback
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [calibration_callback]
            :end-before: [lora_training_callback]

Training Callback
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [lora_training_callback]
            :end-before: [qwa_lora]

Run QW-LoRa
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwlora.py
            :language: python
            :start-after: [qwa_lora]
