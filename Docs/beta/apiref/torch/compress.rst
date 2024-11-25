.. _apiref-torch-compress:

####################
aimet_torch.compress
####################

..
  # common APIs start

**Top-level API for Compression**

.. autoclass:: aimet_torch.compress.ModelCompressor

.. automethod:: aimet_torch.compress.ModelCompressor.compress_model

**Greedy Selection Parameters**

.. autoclass:: aimet_common.defs.GreedySelectionParameters
   :members:

**Configuration Definitions**

.. autoclass:: aimet_common.defs.CostMetric
   :members:
   :noindex:

.. autoclass:: aimet_common.defs.CompressionScheme
   :members:
   :noindex:

..
  # common APIs end

..
  # Spatial SVD config starts

**Spatial SVD Configuration**

.. autoclass:: aimet_torch.defs.SpatialSvdParameters
   :members:

..
  # Spatial SVD config ends

..
  # Channel pruning config starts

**Channel Pruning Configuration**

.. autoclass:: aimet_torch.defs.ChannelPruningParameters
   :members:

..
  # Channel pruning config ends

..
  # Weight SVD config starts

**Weight SVD Configuration**

.. autoclass:: aimet_torch.defs.WeightSvdParameters
   :members:

..
  # Weight SVD config ends
