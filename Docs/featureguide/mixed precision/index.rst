.. _featureguide-mp-index:

###############
Mixed precision
###############

Quantization improves latency, uses less memory, and consumes less power to run a model, but it comes at the cost of reduced accuracy compared to full precision. The loss in accuracy becomes more pronounced the lower the bit width. Mixed precision helps bridge this accuracy gap. In mixed precision, sensitive layers in the model are run at higher precisions, achieving higher accuracy with a smaller model.

Using mixed precision in AIMET follows these steps:

1. Create a quantization simulation (QuantSim) object with a base precision.
2. Run the model in mixed precision by changing the bit width of selected activation and parameter quantizers.
3. Calibrate and simulate the accuracy of the mixed precision model.
4. Export configuration artifacts to create the mixed-precision model.

.. toctree::
    :hidden:

    Manual mixed precision <mmp>
    Automatic mixed precision <amp>

AIMET offers two methods for creating a mixed-precision model: a manual mixed-precision configurator and automatic mixed precision.

Manual mixed precision
----------------------

:ref:`Manual mixed precision <featureguide-mmp>` (MMP) enables different precision levels (bit width) in layers
that are sensitive to quantization.

Automatic mixed precision
-------------------------

:ref:`Automatic mixed precision <featureguide-amp>` (AMP) automatically finds a minimal set of layers that require higher precision to achieve a desired quantized accuracy.
