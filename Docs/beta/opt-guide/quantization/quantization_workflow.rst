.. include:: ../../abbreviation.txt

.. _opt-guide-quantization-workflow:

#####################
Quantization workflow
#####################

This document outlines the guided quantization workflow using AIMET. It presents a clear
approach and methodology for onboarding any non-GenAI model with AIMET.


Quantization features
=====================

AIMET toolkit offers following quantization features.

1. Quantization simulation (QuantSim):

It simulates quantized behavior using floating-point
hardware. QuantSim efficiently enables various quantization options and helps you estimate
the off-target quantized metric through fake quantization (quantize and dequantize operations,
known as QDQ) without requiring actual quantized hardware.

2. Post training quantization (PTQ):

PTQ techniques make a model more quantization-friendly
without requiring model retraining or fine-tuning. PTQ is recommended as a go-to tool in a
quantization workflow because:

- PTQ does not require the original training pipeline
- PTQ is fast and easy to use

3. Quantization aware training (QAT):

QAT enable you to fine-tune a model with quantization
operations (QDQ) inserted in the model graph. In effect, it makes the model parameters robust
to quantization noise. Compared to PTQ:

- QAT requires a training pipeline and dataset and
- QAT takes longer because it needs some fine-tuning,
- QAT requires hyper parameter search

but it can provide better accuracy, especially at lower bit-widths.


Workflow
========

To decide which precision to run inference on target runtime, you can follow the top-down
approach where you begin with the highest precision (For example FP16) and transition to
lower precision if necessary, which may require additional engineering effort.

Given that the off-target quantized accuracy using QuantSim is acceptable, following
on-target metrics should be considered depending on your application.

- Latency reduction and/or
- Memory size reduction

If any of the above on-target metrics are not met for your use case, you should consider
lowering the precision.

Determine precision for on-target inference
-------------------------------------------

Before applying quantization techniques using the AIMET toolkit, you need to identify the
supported precisions to run inference on desired target runtimes. For weights and activations,
supported precisions can be FP32, FP16, INT16, INT8 and INT4.

Some recent runtimes also support heterogeneous bit-width or mixed-precision, enabling
sensitive operations to run at higher precision within your model.

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

FP16 precision (No quantization)
--------------------------------

Converting an FP32 model to FP16 precision without quantization is a recommended starting
point. For more details on how to compile FP16 models for target runtimes, please refer to
|qnn_docs|_ or |qai_hub_docs|_.

Quick W16A16 sanity check
-------------------------

Before using quantized integer format, it's important to ensure that the FP32 model
and the quantized model (QuantSim object) perform similarly during the forward pass, especially
when custom quantizers are included in the model.

Set the bit-width to 16 bits for both weights and activations when creating the QuantSim.
Then, obtain the off-target quantized accuracy metric for the quantized model and verify if
it aligns with the FP32 model. If it doesn't, please report an issue to |aimet|_.

Try lower precision(s)
----------------------

If any of the metrics are not acceptable with higher precision, begin with weights at
INT8 precision and activations at INT16 precision. In this step, before creating the QuantSim,
ensure that the FP32 model adheres to model specific guidelines. For instance, in PyTorch,
QuantSim can only quantize math operations performed by :class:`torch.nn.Module` objects, while
:class:`torch.nn.functional` calls will be incorrectly ignored. Please refer to framework specific
pages to know more about such model guidelines.

If off-target quantized accuracy metric is unsatisfactory, you can apply PTQ/QAT techniques to
enhance the quantized metric for the specified precision. The choice between PTQ and QAT techniques
depends on the quantized accuracy and runtime requirements.

Once the off-target quantized accuracy metric is satisfactory, proceed to :ref:`evaluate the
on-target metrics<opt-guide-on-target-inference>` at this precision. If the on-target metrics still do not meet the
requirements, consider further reducing the precision (for example W8A8) and repeat the
previous step.
