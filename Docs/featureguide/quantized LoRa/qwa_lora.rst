.. _featureguide-qwa-lora:

######################
QWA-LoRa
######################

Context
=======

The QWA-LoRa workflow involves determining the appropriate weight and activation encodings for the base model before
performing some epochs of LoRa training. Finally, the weight and activations for the updated LoRa layers are calibrated.

Workflow
========

Setup
-----

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
