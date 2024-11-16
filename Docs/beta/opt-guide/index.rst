.. _opt-guide-index:

##################
Model Optimization
##################

.. toctree::
    :hidden:

    Overview <overview/index>
    Quantization <quantization/index>
    Compression <compression/index>
    

Models are trained on floating-point hardware like CPUs and GPUs. However, when you run these models on quantized hardware with fixed-precision operations, the model parameters must be fixed-precision. For example, when running on hardware that supports 8-bit integer operations, the floating point parameters in the trained model need to be converted to 8-bit integers.

For some models, reduction to 8-bit fixed-precision introduces noise that causes a loss of accuracy. The AI Model Efficiency Toolkit (AIMET) provides techniques and tools to create quantized models that minimize this loss of accuracy.

The techniques used by AIMET fall into the following categories:

:term:`Quantization simulation` mimics a quantized model by introducing quantization operations on the  parameter outputs of a floating-point model, enabling you to estimate accuracy of a quantized model without exporting it to quantized hardware. 

Quantization simulation supports an array of powerful optimization techiques described in the :ref:`Quantization Simulation Guide <quantsim-index>`.

:term:`Post-training quantization` (PTQ) techniques make a model more quantization-friendly without requiring model retraining or fine-tuning. 

Post-training quantization:

- Does not require the original training pipeline; an evaluation pipeline is sufficient
- Requires only a small, unlabeled dataset for calibration
- Is fast and easy to use

PTQ is therefore recommended as a first step in a quantization workflow.

:term:`Quantization-aware training` (QAT) enables you to fine-tune a model with quantization operations inserted in the network graph. In effect, QAT makes the model parameters robust to quantization noise.

Unlike PTQ, QAT requires a training pipeline and dataset. QAT takes longer because it needs training epochs for fine-tuning, but it can provide better accuracy, especially at low bitwidths.

:term:`Compression` techniques improve model performance by removing inactive layers from a model, reducing runtime computation requirements. Compression is generally recommended as a final step,  after all quantization options have been exhausted.


This user guide is organized into the following sections:

:ref:`Overview <opt-guide-overview-index>` is a general discussion of how AIMET optimizes models.

:ref:`Quantization <opt-guide-quantization-index>` describes how AIMET applies quantization techniques.

:ref:`Compression <opt-guide-compression-index>` describes how AIMET applies compression techniques.