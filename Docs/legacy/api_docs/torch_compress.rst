
=============================
AIMET PyTorch Compression API
=============================

Introduction
============
AIMET supports the following model compression techniques for PyTorch models
   - Weight SVD
   - Spatial SVD
   - Channel Pruning

To learn more about these model compression techniques, please see :ref:`Model Compression User Guide<ug-model-compression>`

For all of these compression techniques there are two modes in which you can invoke the AIMET API
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

.. autoclass:: aimet_torch.compress.ModelCompressor

|

.. automethod:: aimet_torch.compress.ModelCompressor.compress_model

|

Greedy Selection Parameters
===========================

.. autoclass:: aimet_common.defs.GreedySelectionParameters
   :members:

|


Spatial SVD Configuration
=========================

.. autoclass:: aimet_torch.defs.SpatialSvdParameters
   :members:

|


Weight SVD Configuration
========================

.. autoclass:: aimet_torch.defs.WeightSvdParameters
   :members:

|

Channel Pruning Configuration
=============================

.. autoclass:: aimet_torch.defs.ChannelPruningParameters
   :members:

|

Configuration Definitions
=========================

.. autoclass:: aimet_common.defs.CostMetric
   :members:
   :noindex:

|

.. autoclass:: aimet_common.defs.CompressionScheme
   :members:
   :noindex:

|

.. autoclass:: aimet_torch.defs.ModuleCompRatioPair
   :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :lines: 40-49

**Evaluation function**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: evaluate_model

**Compressing using Spatial SVD in auto mode with multiplicity = 8 for rank rounding**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: spatial_svd_auto_mode

**Compressing using Spatial SVD in manual mode**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: spatial_svd_manual_mode

**Compressing using Weight SVD in auto mode**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: weight_svd_auto_mode

**Compressing using Weight SVD in manual mode with multiplicity = 8 for rank rounding**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: weight_svd_manual_mode

**Compressing using Channel Pruning in auto mode**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: channel_pruning_auto_mode

**Compressing using Channel Pruning in manual mode**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: channel_pruning_manual_mode

**Example Training Object**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: Trainer

**Compressing using Spatial SVD in auto mode with layer-wise fine tuning**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: spatial_svd_auto_mode_with_layerwise_finetuning

