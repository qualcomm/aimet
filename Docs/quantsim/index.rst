.. include:: ../abbreviation.txt

.. _quantsim-index:

#############################
Quantization simulation guide
#############################

.. toctree::
    :hidden:

    Calibration <calibration>
    QAT <qat>
    Blockwise quantization <blockwise>

Overview
========

AIMET’s Quantization Simulation (QuantSim) feature emulates the behavior of quantized hardware using floating
point hardware. QuantSim enables you to use post-training quantization (PTQ) and Quantization-aware
training (QAT) methods to restore accuracy lost during quantization before deploying the model to the
target device.

When used alone, QuantSim identifies the optimal quantization scale and offset parameters for each
quantizer but does not apply techniques to reduce accuracy loss. You can apply QuantSim directly to the
original model or to a model that has been updated with PTQ techniques.

The quantization operations in QuantSim are custom quantizers defined within AIMET and are not recognized
by target runtimes like |qnn|. QuantSim provides an export feature that saves a version of the model with the
quantization operations removed and creates an encodings file with quantization scale and offset parameters
for the model's activation and weight tensors. A hardware runtime can use this encodings file
to apply the appropriate scale and offset values to the exported model.

How quantization simulation works
=================================

To create a quantized model, QuantSim simulates quantization noise and determines quantization parameters as describe in the following sections.

Simulate quantization noise
---------------------------

The following figure illustrates how quantization noise is introduced to a model when its inputs, outputs,
or parameters are quantized and de-quantized.

    .. image:: ../images/quant_3.png

A de-quantizated value is not exactly equal to its corresponding original value. The difference between
the two is known as quantization noise.

To simulate quantization noise, AIMET QuantSim adds quantizer operations to the PyTorch, TensorFlow, or
ONNX model graph. The resulting model graph can be used as-is in your evaluation or training pipeline.

Determine quantization parameters (encodings)
---------------------------------------------

Using a QuantSim model, AIMET determines the optimal quantization encodings (scale and offset parameters)
for each quantizer operation.

To do this, AIMET passes calibration samples through the model and, using hooks, intercepts tensor data
flowing through the model. AIMET creates a histogram of the floating point values in the output tensor of each layer.

.. image:: ../images/quant_2.png

Following is a general definition for the quantization function, where a floating-point
number `x` is mapped to its fixed-point representation (quantization) x\ :sub:`int`\  
and then x\ :sub:`int`\  is approximated back to :math:`\hat{x}` on the floating point axis (de-quantization). 


A *quantization* step is defined as:

.. math::
    x_{int} = clamp\left(\left\lceil\frac{x}{scale}\right\rfloor - offset, q_{min}, q_{max}\right)

To approximate the floating-point number `x`, A *de-quantization* step is performed:

.. math::
    x \approx \hat{x} = (x_{int} + offset) * scale

An encoding for a layer consists of four values:

Min (q\ :sub:`min`\ )
    Values below these are clamped
Max (q\ :sub:`max`\ )
    Values above these are clamped
Delta (Scale)
    Granularity of the fixed point values (a function of the selected bit-width)
Offset (Zero-point)
    Offset from zero

The delta and offset are calculated using q\ :sub:`min`\  and q\ :sub:`max`\  and vice versa using the
equations. Following are the most important ones.

.. math::
    Delta = \frac{q_{max} - q_{min}}{2^{bitwidth} - 1} 

.. math::    
    Offset = \frac{-q_{min}}{Delta}

Using the floating point distribution in the output tensor for each layer, AIMET calculates quantization
encodings using the calibration technique described in the next section.

Quantization schemes
====================

AIMET supports various range estimation techniques, also called quantization schemes, for
calculating min and max values for encodings:

Min-Max (also called "TF" in AIMET)
-----------------------------------

To cover the whole dynamic range of the tensor, in Min-Max (also called "TF" in AIMET) the quantization parameters Min and Max are defined as the
observed minimum and maximum during the calibration process. This approach eliminates clipping error but is
sensitive to outliers since extreme values induce rounding errors.

.. note::

   The name "TF" derives from the origin of the technique and has no relation to which framework is using
   it.

Signal-to-Quantization-Noise
----------------------------

The Signal-to-Quantization-Noise (SQNR; also called “TF Enhanced” in AIMET) approach is similar to the mean square error (MSE) minimization approach. The q\ :sub:`min`\  and q\ :sub:`max`\  are found that minimize the total MSE between the original and the quantized tensor.

.. note::

   The name "TF Enhanced" derives from the origin of the technique and has no relation to which framework
   is using it.

Quantization granularity
========================

Different hardware and on-device runtimes support various levels of quantization granularity, such as per-tensor,
per-channel, and per-block. However, not all hardware can handle every level of granularity, as higher
granularity requires more overhead.

Per-tensor quantization
    All values in the entire tensor are grouped, and a single set of encodings are determined. Benefits include less computation and storage space needed to produce a single set of encodings. Drawbacks are that outlier values in the tensor distort the encodings of all other  values in the tensor.

Per-channel quantization
    The tensor is split into channels (typically in the output channels dimension).  One encoding is computed for each channel. The benefit over Per Tensor quantization is that  outlier values influence encodings only for the channel they are in.

Per-block quantization (Blockwise quantization)
    The tensor is split into chunks across multiple dimensions. This increases the granularity of encoding parameters, further isolating outliers. This optimizes the quantization grid for each block at the cost of more storage for an increased number of encodings.

Runtime configuration
=====================

Different hardware and on-device runtimes support different quantization choices for neural network
inference. For example, some runtimes support asymmetric quantization for both activations and weights,
while others support asymmetric quantization just for weights.

Quantization choices during simulation need to best reflect the target runtime and hardware.
AIMET provides a default configuration JSON file that can be modified. By default, the following configuration
is used for quantization simulation:

.. list-table::
   :widths: 5 12
   :header-rows: 1

   * - Quantization
     - Configuration
   * - Weight
     - Per-channel, symmetric quantization, INT8
   * - Activation
     - Per-tensor, asymmetric quantization, INT16

Quantization options settable in the runtime configuration file include:

* Enabling or disabling input/output/parameter quantizer ops
* Symmetric vs asymmetric quantization
    * Unsigned vs signed symmetric quantization
    * Strict vs non-strict symmetric quantization
* Per-channel vs per-tensor quantization
* Defining supergroups of operations to be fused

See the :ref:`Runtime configuration <quantsim-runtime-config>` page, which describes various configuration
options in detail.

.. _quantsim-workflow:

QuantSim workflow
=================

Following is a typical workflow for using AIMET QuantSim to simulate on-target quantized accuracy.

#. Start with a pretrained floating-point (FP32) model.

#. Use AIMET to create a :class:`QuantizationSimModel` model. AIMET inserts quantization simulation
   operations into the model graph.

#. AIMET configures the inserted quantization operations. The configuration of these operations can be
   controlled via a configuration file.

#. Provide a callback method that feeds representative data samples through the :class:`QuantizationSimModel` model.
   AIMET uses the callback to find optimal quantization parameters, such as scales and offsets, for the
   inserted quantization operations. These samples can be from the training or calibration datasets.
   500-1,000 samples are usually sufficient to compute optimal quantization parameters.

#. AIMET returns a :class:`QuantizationSimModel` model that can be used as a drop-in replacement for the
   original model in your evaluation pipeline. Running this simulation model through the evaluation
   pipeline yields a quantized accuracy metric that closely simulates on-target accuracy.

#. Call :func:`QuantizationSimModel.export` on the QuantSim object to save a copy of the model with
   quantization operations removed, along with an encodings file containing quantization scale and offset
   parameters for each activation and weight tensor in the model.

Exported Encodings
==================
An encodings file containing quantization parameters for tensors in the model is produced when calling :func:`QuantizationSimModel.export`.
The file, in conjunction with the exported model, can be used by a target runtime like |qnn| to run the quantized model on target.

See the :ref:`Encoding Format Specification <quantsim-encoding-spec>` page for a description on the specification and file format.
