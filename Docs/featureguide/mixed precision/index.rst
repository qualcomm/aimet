.. _featureguide-mp-index:

###############
Mixed precision
###############

Quantization is a technique to improve the latency by running Deep Learning models in lower precision when
compared to full-precision floating point. Even though quantization helps achieve improved latency, store the model with
less memory and consume less power to run the models, it comes at a cost of reduced accuracy when compared to running
the model in Full Precision. The loss in accuracy is more pronounced as we run the model in lower bitwidths.
Mixed-Precision helps bridge the accuracy gap of quantized model when compared to floating point accuracy. In mixed
precision, different layers in the model are run in different precisions based on their sensitivity thereby getting the
benefit of higher accuracy but keeping the model size to be lower compared to full-precision floating point.

Mixed precision in AIMET currently follows the following steps,

* Create the QuantSim object with a base precision
* Set the model to run in mixed precision by changing the bitwidth of relevant activation and param quantizers
* Calibrate and simulate the accuracy of the mixed precision model
* Export the artifacts which can be used by backend tools like QNN to run the model in mixed precision

.. toctree::
    :hidden:

    Manual mixed precision <mmp>
    Automatic mixed precision <amp>

:ref:`Manual mixed precision <featureguide-mmp>`
------------------------------------------------

Manual mixed precision (MMP) allows to set different precision levels (bit-width) to layers
that are sensitive to quantization.

:ref:`Automatic mixed precision <featureguide-amp>`
---------------------------------------------------

Auto mixed precision (AMP) will automatically find a minimal set of layers that need to
run on higher precision, to get to the desired quantized accuracy.
