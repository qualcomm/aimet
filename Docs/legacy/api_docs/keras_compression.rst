
================================
AIMET TensorFlow Compression API
================================

Introduction
============
AIMET supports the following model compression techniques for keras models
   - Spatial SVD

To learn more about these model compression techniques, please see :ref:`Model Compression User Guide<ug-model-compression>`

For the Spatial SVD compression techniques, there are two modes in which you can invoke the AIMET API
   - Auto Mode: In **Auto** mode, AIMET will determine the optimal way to compress each layer of
                the model given an overall target compression ratio. Greedy Compression Ratio Selection Algorithm is used to pick appropriate compression ratios for each layer.

   - Manual Mode: In **Manual** mode, the user can pass in the desired compression-ratio per layer
                  to AIMET. AIMET will apply the specified compression technique for each of the
                  layers to achieve the desired compression-ratio per layer. It is recommended that
                  the user start with Auto mode, and then tweak per-layer compression-ratios using
                  Manual mode if desired.


|

Top-level API for Compression
=============================

.. autoclass:: aimet_tensorflow.keras.compress.ModelCompressor

|

.. automethod:: aimet_tensorflow.keras.compress.ModelCompressor.compress_model

|

Greedy Selection Parameters
===========================

.. autoclass:: aimet_common.defs.GreedySelectionParameters
   :members:
   :noindex:

|


Spatial SVD Configuration
=========================

.. autoclass:: aimet_tensorflow.keras.defs.SpatialSvdParameters
   :members:

|


Configuration Definitions
=========================

.. autoclass:: aimet_common.defs.CostMetric
   :members:

|


.. autoclass:: aimet_common.defs.CompressionScheme
   :members:

   .. note::
      Only Spatial SVD is supported for now.

|

.. autoclass:: aimet_tensorflow.keras.defs.ModuleCompRatioPair
   :members:

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../keras_code_examples/compression_code_examples.py
    :language: python
    :lines: 39-49

**Evaluation function**

.. literalinclude:: ../keras_code_examples/compression_code_examples.py
    :language: python
    :pyobject: get_eval_func

**Compressing using Spatial SVD in auto mode**

.. literalinclude:: ../keras_code_examples/compression_code_examples.py
    :language: python
    :pyobject: aimet_spatial_svd

**Sample Driver Code for Spatial SVD using Resnet50**

.. literalinclude:: ../keras_code_examples/compression_code_examples.py
    :language: python
    :pyobject: compress

