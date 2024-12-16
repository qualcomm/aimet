.. _ug-quantsim:

#############################
AIMET quantization simulation
#############################

Overview
========

AIMET’s Quantization Simulation feature simulates the effects of quantized hardware. This enables you to apply post-training and/or fine-tuning techniques in AIMET to recover the accuracy lost in quantization before deploying the model to the target device.

When QuantSim is applied by itself, AIMET finds optimal quantization scale/offset parameters for each quantizer but does not apply techniques to mitigate accuracy loss. You can apply QuantSim directly to the original model or to a model updated using Post-Training Quantization.

Once a QuantSim object has been created, you can fine-tune the model using its existing pipeline. This technique is described in :ref:`Quantization Aware Training<ug-quantization-aware-training>`.

The quantization nodes used in QuantSim are custom quantizers defined in AIMET, and are not recognized by targets.
QuantSim provides an export functionality that saves a copy of the model with quantization nodes removed and generates an encodings file containing quantization scale and offset parameters for activation and weight tensors in the model.

A hardware runtime can ingest the encodings file and match it with the exported model to apply scale and offset values in the model.

QuantSim workflow
=================

Following is a typical workflow for using AIMET QuantSim to simulate on-target quantized accuracy.

1. Start with a pretrained floating-point (FP32) model.

2. Use AIMET to create a simulation model. AIMET inserts quantization simulation operations into the model graph (explained in the sub-section below).

3. AIMET configures the inserted simulation operations. The configuration of these operations can be controlled via a configuration file as discussed below.

4. Provide a callback method that feeds representative data samples through the model. AIMET uses this method to find optimal quantization parameters, such as scales and offsets, for the inserted quantization simulation operations. These samples can be from the training or calibration datasets. 1,000-2,000 samples are usually sufficient to optimize quantization parameters.

5. AIMET returns a quantization simulation model that can be used as a drop-in replacement for the original model in
   your evaluation pipeline. Running this simulation model through the evaluation pipeline yields a quantized accuracy
   metric that closely simulates on-target accuracy.

6. Call `.export()` on the sim object to save a copy of the model with quantization nodes removed, along with
   an encodings file containing quantization scale and offset parameters for each activation and weight tensor in the model.

Simulating quantization noise
=============================

The diagram below illustrates how quantization noise is introduced to a model when its inputs, outputs, or parameters are quantized and de-quantized.

    .. image:: ../images/quant_3.png

A de-quantizated value is not exactly equal to its corresponding quantized value. The difference between the two values is the quantization noise.

To simulate quantization noise, AIMET QuantSim adds quantizer operations to the PyTorch, TensorFlow, or Keras model graph. The resulting model graph can be used as-is in your evaluation or training pipeline.

Determining quantization parameters (encodings)
===============================================

Using a QuantSim model, AIMET determines the optimal quantization encodings (scale and offset parameters) for each quantizer operation.

To do this, AIMET passes calibration samples through the model and, using hooks, intercepts tensor data flowing through the model. AIMET creates a histogram to model the distribution of the floating point values in the output tensor for each layer.

.. image:: ../images/quant_2.png

An encoding for a layer consists of four numbers:

Min (q\ :sub:`min`\ )
   Numbers below these are clamped
Max (q\ :sub:`max`\ )
   Numbers above these are clamped
Delta 
   Granularity of the fixed point numbers (a function of the selected bit-width)
Offset
   Offset from zero

The Delta and Offset are calculated using Min and Max and vice versa using the equations:
    :math:`\textrm{Delta} = \dfrac{\textrm{Max} - \textrm{Min}}{{2}^{\textrm{bitwidth}} - 1} \quad \textrm{Offset} = \dfrac{-\textrm{Min}}{\textrm{Delta}}`

Using the floating point distribution in the output tensor for each layer, AIMET calculates quantization encodings using the specified quantization calibration technique described in the next section.

Quantization schemes
====================

AIMET supports various techniques, also called quantization schemes, for calculating min and max values for encodings:

**Min-Max (also referred to as "TF" in AIMET)**

   (The name "TF" derives from the origin of the technique and has no relation to which framework is using it.)

   To cover the whole dynamic range of the tensor, the quantization parameters Min and Max are defined as the observed Min and Max during the calibration process. This approach eliminates clipping error but is sensitive to outliers since extreme values induce rounding errors.

**Signal-to-Quantization-Noise (SQNR; also called “TF Enhanced” in AIMET)**

   (The name "TF Enhanced" derives from the origin of the technique and has no relation to which framework is using it.)

   The SQNR approach is similar to the mean square error (MSE) minimization approach. The qmin and qmax are found that minimize the total MSE between the original and the quantized tensor. 
   
   Quantization noise and saturation noise are different types of erros which are weighted differently.

For each quantization scheme, there are "post training" and "training range learning" variants. The "post training" variants are used during regular QuantSim inference and QAT without Range Learning to compute initial encoding values for each quantization node. In QAT without Range Learning, encoding values for activation quantizers remain static (encoding values for parameter quantizers change with changing parameter values during training).

The "training range learning" variants are used during QAT with Range Learning. The schemes define how to compute initial encoding values for each quantization node, but also allow encoding values for activations to be learned alongside parameter quantizer encodings during training.

For more details on QAT, see :ref:`Quantization Aware Training<ug-quantization-aware-training>`.

Configuring quantization simulation operations
==============================================

Different hardware and on-device runtimes support different quantization choices for neural network inference. For example, some runtimes support asymmetric quantization for both activations and weights, whereas others support asymmetric quantization just for weights.

As a result, quantization choices during simulation need to best reflect the target runtime and hardware. AIMET provides a default configuration file that can be modified. By default, the following configuration is used for quantization simulation:

Weight quantization
   Per-channel, symmetric quantization, INT8

Activation or layer output quantization
   Per-tensor, asymmetric quantization, INT8

Quantization options settable in the configuration file include:

- Enabling or disabling input and output quantizer ops
- Enabling or disabling parameter quantizer ops
- Enabling or disabling model input quantizer
- Enabling or disabling model output quantizer
- Symmetric or asymmetric quantization
- Unsigned or signed symmetric quantization
- Strict or non-strict symmetric quantization
- Per-channel or per-tensor quantization
- Defining groups of layers to be fused (no quantization is done on intermediate tensors within fused layers)

See the :ref:`Quantization Simulation Configuration <ug-quantsim-config>` page, which describes the configuration options in detail.

Quantization Simulation APIs
============================

See the AIMET Quantization Simulation API for your platform: 

- :ref:`Quantization Simulation for PyTorch<api-torch-quantsim>`
- :ref:`Quantization Simulation for Keras<api-keras-quantsim>`
- :ref:`Quantization Simulation for ONNX<api-onnx-quantsim>`
