.. include:: ../abbreviation.txt

.. _opt-guide-quantization-workflow:

#####################
Quantization workflow
#####################

This page outlines a methodology to onboard, quantize, and deploy machine-learning models on Qualcomm\ |reg| devices using the AI Model Efficiency Toolkit (AIMET).

Supported precisions for on-target inference
============================================

Before applying quantization techniques, identify the computational precisions supported by the target runtimes
on which you plan to run inference. For weights and activations, the precisions supported by AIMET are: FP32, FP16, INT16, INT8, and INT4.

Some recent runtimes also support *heterogeneous bit-width* (also called *mixed precision*), enabling you to run
sensitive operations at a higher precision within your model.

Precisions supported by AIMET for inference on target runtimes like |qnn|_ are:

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


Quantization workflow
=====================

We recommend the following workflow for quantizing a model to improve its efficiency. Unless you have a reason to do otherwise, we recommend you try these actions, listed in ascending order of effort:

1. Convert the model to 16-bit floating point (FP16) precision.
2. Quantize the model to W16A16.
3. Apply quantization techniques, including post-training quantization (PTQ) and quantization-aware training (QAT).

The following sections break these broad actions down into discrete procedures for taking these actions.

Step 1: Trying FP16 precision (no quantization)
-----------------------------------------------

We recommend that you start by converting the FP32 model to FP16 precision without quantization. For instructions on how to compile FP16 models for target runtimes, see |qnn_docs|_ or |qai_hub_docs|_.

If performance is unacceptable at FP16, the next step is to quantize the model.

Step 2: Trying W16A16 quantization
----------------------------------

Quantize the model weights and activations to 16-bit integer (W16A16). Do the following:

1. Ensure that the FP32 model adheres to model-specific guidelines. For instance, in PyTorch QuantSim can only quantize math operations performed by :class:`torch.nn.Module` objects, while :class:`torch.nn.functional` calls will be incorrectly ignored. See framework-specific pages to learn more about such model guidelines.
2. Once the model conforms to guidelines, create a quantization simulation (QuantSim) version of your model with the bit-width set to 16 bits for both weights and activations (W16A16). See :ref:`<quantsim-workflow>`.
3. Ensure that the original FP32 model and the quantized model (QuantSim object) perform similarly during the forward pass, especially when custom quantizers are included in the model. 
4. Compute the off-target quantized accuracy metric for the quantized model and verify that it agrees with the FP32 model. If it does not, you can help improve AIMET by reporting an issue to |aimet|_.

Step 3. Applying PTQ or QAT
---------------------------

If the off-target (simulated) quantized accuracy metric does not meet expectations, use PTQ or QAT techniques to improve the accuracy for the implemented precision. We suggest starting with with weights at INT8 precision and activations at INT16 precision (W8A16). 

The decision to use PTQ or QAT should balance your requirements for runtime accuracy vs performance. We usually recommend starting with PTQ. See :ref:`featureguide-index` (PTQ) and :ref:`quantsim-qat` (QAT).

.. image:: ../images/quantization_workflow_5.png

Next: deploying the model
-------------------------

Once the off-target quantized accuracy is satisfactory, proceed to :ref:`evaluate the
on-target metrics<opt-guide-on-target-inference>` at this precision. If the on-target metrics still do not meet your requirements, consider further reducing the precision (for example to W8A8 or W4A8) and repeat the application of PTQ or QAT to optimize the model.

Once the quantized accuracy and runtime requirements are achieved at the desired precision, deploy the optimized model on the target runtimes.
