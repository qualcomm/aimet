.. include:: ../abbreviation.txt

.. _opt-guide-overview:

#####################
Quantization features
#####################

AIMET offers the following three classes of quantization features.

Quantization simulation
=======================

Quantization simulation (QuantSim) uses quantization and dequantization (QDQ) operations to mimic quantized behavior on floating-point hardware. QuantSim helps you estimate the accuracy of a quantized model without deploying it to quantized hardware.

A QuantSim workflow is illustrated here:

.. image:: ../images/quant_use_case_1.PNG

Quantization simulation is described in the :ref:`Quantization simulation guide <quantsim-index>`.


Post-training quantization
==========================

Post-training quantization (PTQ) techniques make a model more quantization-friendly without requiring model retraining
or fine-tuning. PTQ is a preferred tool in the quantization workflow because it is efficient and easy to use and does not require model training.

The PTQ workflow is illustrated here:

.. image:: ../images/quant_use_case_3.PNG

Post-training quantization techniques are described in :ref:`Optimization techniques <featureguide-index>`.


Quantization-aware training
===========================

Quatization-aware training (QAT) enables you to fine-tune a model with QDQ operations inserted in the
model graph. In effect, QAT makes the model parameters robust to quantization noise.

Compared to PTQ:

- QAT requires a training pipeline and dataset
- QAT takes longer because it needs some training to fine-tune the quantized model
- QAT requires hyper parameters search

However, QAT can provide better accuracy than PTQ, especially at lower bit-widths.

A typical QAT workflow is illustrated here:

.. image:: ../images/quant_use_case_2.PNG

Quantization-aware training is described in :ref:`Quantization aware training <quantsim-qat>`.


