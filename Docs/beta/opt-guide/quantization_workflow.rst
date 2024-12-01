.. include:: ../abbreviation.txt

.. _opt-guide-quantization-workflow:

#####################
Quantization workflow
#####################

This document outlines a clear approach and methodology to onboard, quantize and deploy any
machine-learning models on Qualcomm\ |reg| devices using AIMET toolkit.

Quantization features
=====================

AIMET toolkit offers following quantization features.

1. Quantization simulation (QuantSim):
--------------------------------------

It simulates quantized behavior using floating-point hardware. QuantSim efficiently enables
various quantization options and helps you estimate the off-target quantized accuracy metric
using quantization simulation (sequence of quantize and dequantize operations, known as QDQ)
without requiring actual quantized hardware.

A quantization simulation workflow is illustrated here:

    .. image:: ../../images/quant_use_case_1.PNG

2. Post-training quantization (PTQ):
------------------------------------

PTQ techniques make a model more quantization-friendly without requiring model retraining
or fine-tuning. PTQ is recommended as a go-to tool in a quantization workflow because:

- PTQ does not require the training pipeline
- PTQ is efficient and easy to use

The PTQ workflow is illustrated here:

    .. image:: ../../images/quant_use_case_3.PNG

3. Quantization-aware training (QAT):
-------------------------------------

QAT enables you to fine-tune a model with quantization operations (QDQ) inserted in the
model graph. In effect, it makes the model parameters robust to quantization noise.

Compared to PTQ:

- QAT requires a training pipeline and dataset,
- QAT takes longer because it needs some fine-tuning,
- QAT requires hyper parameters search

but it can provide better accuracy, especially at lower bit-widths.

A typical QAT workflow is illustrated here:

    .. image:: ../../images/quant_use_case_2.PNG

Determine supported precisions for on-target inference
======================================================

Before applying quantization techniques, you need to identify the supported precisions
to run inference on desired target runtimes. For weights and activations, supported
precisions can be FP32, FP16, INT16, INT8 and INT4.

Some of the recent runtimes also support heterogeneous bit-width or mixed-precision, enabling
sensitive operations to run at a higher precision within your model.

Supported precisions to run inference on target runtimes like |qnn|_ are:

.. list-table::
   :widths: 12 8 8
   :header-rows: 1

   * - Precision format
     - Weights
     - Activations
   * - Floating-point (No quantization)
     - FP16
     - FP16
   * - Integer (quantized W8A16)
     - INT8
     - INT16
   * - Integer (quantized W8A8)
     - INT8
     - INT8
   * - Integer (quantized W4A8)
     - INT4
     - INT8

Workflow
========

To decide which precision to run inference on target runtimes, you can follow the top-down
approach where you begin with the highest precision (For example FP16) and transition to
lower precision if necessary, which may require additional engineering effort.

Given that the off-target quantized accuracy using QuantSim is acceptable, following
on-target metrics should be considered depending on your application.

- Latency reduction and/or
- Memory size reduction

If any of the above on-target metrics are not met for your use case, you should consider
lowering the precision.

The figure below illustrates the recommended quantization workflow and the steps required
to deploy the quantized model on the target device.

.. figure:: ../images/quantization_workflow.png

   Recommended quantization workflow

FP16 precision (No quantization)
--------------------------------

Converting an FP32 model to FP16 precision without quantization is a recommended starting
point. For more details on how to compile FP16 models for target runtimes, please refer to
|qnn_docs|_ or |qai_hub_docs|_.

W16A16 sanity check
-------------------

Before using quantized integer format, it's important to ensure that the FP32 model
and the quantized model (QuantSim object) perform similarly during the forward pass, especially
when custom quantizers are included in the model.

Set the bit-width to 16 bits for both weights and activations when creating the QuantSim.
Then, obtain the off-target quantized accuracy metric for the quantized model and verify if
it aligns with the FP32 model. If it doesn't, please report an issue to |aimet|_.

Apply PTQ or QAT at specified precision
---------------------------------------

If any of the metrics are not acceptable with higher precision, begin with weights at
INT8 precision and activations at INT16 precision. In this step, before creating the QuantSim,
ensure that the FP32 model adheres to model specific guidelines. For instance, in PyTorch,
QuantSim can only quantize math operations performed by :class:`torch.nn.Module` objects, while
:class:`torch.nn.functional` calls will be incorrectly ignored. Please refer to framework specific
pages to know more about such model guidelines.

If the off-target quantized accuracy metric is not meeting expectations, you can use PTQ or QAT
techniques to improve the quantized accuracy for the desired precision. The decision between
PTQ and QAT should be based on the quantized accuracy and runtime needs.

Once the off-target quantized accuracy metric is satisfactory, proceed to :ref:`evaluate the
on-target metrics<opt-guide-on-target-inference>` at this precision. If the on-target metrics
still do not meet the your requirements, consider further reducing the precision
(for example W8A8, W4A8) and repeat the current step.

Deploy
------

Once the quantized accuracy and runtime requirements are achieved at the desired precision,
the optimized model is ready for deployment on the target runtimes.
