.. _techniques-index:

##########
Techniques
##########

This section gives brief overview of quantization techniques and how to apply them with AIMET.  

.. toctree::
  :maxdepth: 1
  :hidden:

  Post Training Quantization <ptq>
  Quantization Aware Training <qat>
  Blockwise Quantization <blockwise>
  Mixed precision <mixed_precision/index>
  Analysis tools <analysis_tools/index>
  Compression <compression/index>


:ref:`Post Training Quantization <techniques-ptq>`
==============================================

Quantize model with given calibration data in user specified parameter and activation precision (bit-width).

:ref:`Quantization Aware Training <techniques-qat>`
==============================================

Train model with quantization awareness to minimize quantization noise.

:ref:`Blockwise Quantization <techniques-blockwise>`
==============================================

Quantize individual tensor with block size to balance accuracy and speed.

:ref:`Mixed Precision <featureguide-mp-index>`
==============================================

Configure per-layer bit-widths to optimize accuracy and performance.

:ref:`Analysis tools <techniques-analysis-tools>`
=========================================================

Analysis tools to automatically identify sensitive areas and hotspots in your pre-trained model.

:ref:`Compression <techniques-compression>`
===================================================

Reduces pre-trained model’s Multiply-accumulate(MAC) and memory costs with a minimal drop in accuracy.
AIMET supports various compression techniques like Weight SVD, Spatial SVD and Channel pruning.
