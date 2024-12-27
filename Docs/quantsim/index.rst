.. include:: ../abbreviation.txt

.. _quantsim-index:

#############################
Quantization simulation guide
#############################

.. toctree::
    :hidden:

    Calibration <calibration>
    QAT <qat>
    Advanced <advanced>

Overview
========

AIMET’s Quantization Simulation (QuantSim) feature emulates the behavior of quantized hardware using floating
point hardware. QuantSim also allows you to use post-training quantization (PTQ) and/or Quantization-aware
training (QAT) methods to restore any accuracy lost during quantization before deploying the model to the
target device.

When used alone, QuantSim in AIMET identifies the optimal quantization scale and offset parameters for each
quantizer but does not apply techniques to reduce accuracy loss. You can apply QuantSim directly to the
original model or to a model that has been updated with PTQ technique(s).

The quantization operations in QuantSim are custom quantizers defined within AIMET and are not recognized
by target runtimes like |qnn|. QuantSim provides an export feature that saves a version of the model with
quantization operations removed and creates an encodings file with quantization scale and offset parameters
for the model's activation and weight tensors. Hardware runtime can then use this encodings file and the
exported model to apply the appropriate scale and offset values.

Simulate quantization noise
===========================

The diagram below illustrates how quantization noise is introduced to a model when its inputs, outputs,
or parameters are quantized and de-quantized.

    .. image:: ../images/quant_3.png

A de-quantizated value is not exactly equal to its corresponding original value. The discrepancy between
the two is known as quantization noise.

To simulate quantization noise, AIMET QuantSim adds quantizer operations to the PyTorch, TensorFlow, or
ONNX model graph. The resulting model graph can be used as-is in your evaluation or training pipeline.

Determine quantization parameters (encodings)
=============================================

Using a QuantSim model, AIMET determines the optimal quantization encodings (scale and offset parameters)
for each quantizer operation.

To do this, AIMET passes calibration samples through the model and, using hooks, intercepts tensor data
flowing through the model. AIMET creates a histogram to model the distribution of the floating point values
in the output tensor for each layer.

.. image:: ../images/quant_2.png

Following is a general definition for the quantization function where floating-point
number `x` is mapped to it's fixed-point representation (quantization) `xint`
and then `xint` is approximated back to floating point axis (de-quantization) `xhat`.

A `quantization` step is defined as:

.. math::
    xint = clamp\left(\left\lceil\frac{x}{scale}\right\rfloor - offset, qmin, qmax\right)

To approximate the floating-point number `x`, we perform `de-quantization` step:

.. math::
    x \approx \hat{x} = (xint + offset) * scale

An encoding for a layer consists of four numbers:

* Min (q\ :sub:`min`\ )
    * Numbers below these are clamped
* Max (q\ :sub:`max`\ )
    * Numbers above these are clamped
* Delta (Scale)
    * Granularity of the fixed point numbers (a function of the selected bit-width)
* Offset (Zero-point)
    * Offset from zero

The delta and offset are calculated using qmin and qmax and vice versa using the
equations:

:math:`\textrm{Delta} = \dfrac{\textrm{qmax} - \textrm{qmin}}{{2}^{\textrm{bitwidth}} - 1} \quad \textrm{Offset} = \dfrac{-\textrm{qmin}}{\textrm{Delta}}`

Using the floating point distribution in the output tensor for each layer, AIMET calculates quantization
encodings using the specified quantization calibration technique described in the next section.

Quantization schemes
====================

AIMET supports various range estimation techniques, also called quantization schemes, for
calculating min and max values for encodings:

**Min-Max (also referred to as "TF" in AIMET)**

.. note::

   The name "TF" derives from the origin of the technique and has no relation to which framework is using
   it.

To cover the whole dynamic range of the tensor, the quantization parameters Min and Max are defined as the
observed Min and Max during the calibration process. This approach eliminates clipping error but is
sensitive to outliers since extreme values induce rounding errors.

**Signal-to-Quantization-Noise (SQNR; also called “TF Enhanced” in AIMET)**

.. note::

   The name "TF Enhanced" derives from the origin of the technique and has no relation to which framework
   is using it.

The SQNR approach is similar to the mean square error (MSE) minimization approach. The qmin and qmax are
found that minimize the total MSE between the original and the quantized tensor.

Quantization granularity
========================

Different hardware and on-device runtimes support various levels of quantization granularity, such as per-tensor,
per-channel, and per-block. However, not all hardware can handle every level of granularity, as higher
granularity requires more overhead.

* Per-tensor quantization
    * All values in the entire tensor are grouped collectively, and a single set of encodings are
      determined. Benefits include less computation and storage space needed to produce a single set of
      encodings. Drawbacks are that outlier values in the tensor negatively affect the encodings which
      are used to quantize all other values in the tensor.

* Per-channel quantization
    * Values in the tensor are split into individual channels (typically in the output channels dimension). The
      number of encodings computed for the tensor is equal to the number of channels. The benefit as
      compared to Per Tensor quantization are that outlier values would only influence encodings for the channel
      the outlier resides in, and would not affect encodings for values in other channels.

* Per-block quantization (Blockwise quantization)
    * Values in the tensor are split into chunks across multiple dimensions. This further improves the granularity
      at which encoding parameters are found, isolating outliers and producing a more optimized quantization grid
      for each block, at the cost of more storage used to hold an increased number of encodings.

Runtime configuration
=====================

Different hardware and on-device runtimes support different quantization choices for neural network
inference. For example, some runtimes support asymmetric quantization for both activations and weights,
whereas others support asymmetric quantization just for weights.

As a result, quantization choices during simulation need to best reflect the target runtime and hardware.
AIMET provides a default configuration file that can be modified. By default, the following configuration
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

Quantization options settable in the runtime configuration JSON file include:

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

#. Provide a callback method that feeds representative data samples through :class:`QuantizationSimModel` model.
   AIMET uses this method to find optimal quantization parameters, such as scales and offsets, for the
   inserted quantization operations. These samples can be from the training or calibration datasets.
   500-1,000 samples are usually sufficient to compute optimal quantization parameters.

#. AIMET returns a :class:`QuantizationSimModel` model that can be used as a drop-in replacement for the
   original model in your evaluation pipeline. Running this simulation model through the evaluation
   pipeline yields a quantized accuracy metric that closely simulates on-target accuracy.

#. Call :func:`QuantizationSimModel.export` on the QuantSim object to save a copy of the model with
   quantization operations removed, along with an encodings file containing quantization scale and offset
   parameters for each activation and weight tensor in the model.
