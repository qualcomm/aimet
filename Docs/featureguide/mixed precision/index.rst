.. _featureguide-mp-index:

###############
Mixed precision
###############

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
