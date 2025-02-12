.. _featureguide-quantized-lora-index:

###############
Quantized LoRa
###############

LoRa (Low-Rank Adaptation) is a popular and lightweight training technique for adapting large machine learning models
to specific use cases. Specifically, LoRa works by adding a small number of new trainable weights into the model. Only
these new weights are trained, while the base model weights are left unmodified. The workflows outlined in this
guide outline how to perform LoRa training with a *quantized* base model.

.. toctree::
    :hidden:

    QW-LoRa <qw_lora>
    QWA-LoRa <qwa_lora>

:ref:`QW-LoRa <featureguide-qw-lora>`
------------------------------------------------

QW-LoRa is a workflow that performs LoRa training on a base model that has quantized weights.

:ref:`QWA-LoRa <featureguide-qwa-lora>`
---------------------------------------------------

QWA-LoRa is a workflow that performs LoRa training on a base model that has quantized weights and quantized activations.