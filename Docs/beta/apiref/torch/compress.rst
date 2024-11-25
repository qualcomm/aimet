.. _apiref-torch-compress:

####################
aimet_torch.compress
####################

..
  # common APIs start

**Top-level API for Compression**

.. autoclass:: aimet_torch.compress.ModelCompressor
   :noindex:

.. automethod:: aimet_torch.compress.ModelCompressor.compress_model
   :noindex:

**Greedy Selection Parameters**

.. autoclass:: aimet_common.defs.GreedySelectionParameters
   :members:
   :noindex:

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
   :noindex:

..
  # Spatial SVD config ends

..
  # Channel pruning config starts

**Channel Pruning Configuration**

.. autoclass:: aimet_torch.defs.ChannelPruningParameters
   :members:
   :noindex:

..
  # Channel pruning config ends

..
  # Weight SVD config starts

**Weight SVD Configuration**

.. autoclass:: aimet_torch.defs.WeightSvdParameters
   :members:
   :noindex:

..
  # Weight SVD config ends
