.. _featureguide-qwa-lora:

######################
QWA-LoRa
######################

Context
=======

The QWA-LoRa workflow involves determining the appropriate weight and activation encodings for the base model before
performing some epochs of LoRa training. Finally, the weight and activations for the updated LoRa layers are calibrated.
This is expressed in the block diagram below.

.. image:: ../../images/qwa_lora_block_diagram.png
    :width: 900px

Workflow
========

Setup
-----

In this section, we instantiate the base model, LoRa adapters, and dataset using Huggingface APIs.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwalora.py
            :language: python
            :start-after: [setup]
            :end-before: [create_quantsim]

Create QuantizationSimModel
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwalora.py
            :language: python
            :start-after: [create_quantsim]
            :end-before: [calibration_callback]

Calibration Callback
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwalora.py
            :language: python
            :start-after: [calibration_callback]
            :end-before: [lora_training_callback]

Training Callback
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwalora.py
            :language: python
            :start-after: [lora_training_callback]
            :end-before: [qwa_lora]

Run QWA-LoRa
-----

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_qwalora.py
            :language: python
            :start-after: [qwa_lora]
