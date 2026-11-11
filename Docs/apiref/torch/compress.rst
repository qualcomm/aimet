.. _apiref-torch-compress:

####################
aimet_torch.compress
####################

..
  # common APIs start

.. rubric:: Top-level API for Compression

.. autoclass:: aimet_torch.compress.ModelCompressor
   :noindex:

.. automethod:: aimet_torch.compress.ModelCompressor.compress_model
   :noindex:

.. rubric:: Greedy Selection Parameters

.. autoclass:: aimet_torch.common.defs.GreedySelectionParameters
   :members:
   :noindex:

.. rubric:: Configuration Definitions

.. autoclass:: aimet_torch.common.defs.CostMetric
   :members:
   :noindex:

.. autoclass:: aimet_torch.common.defs.CompressionScheme
   :members:
   :noindex:

..
  # common APIs end

..
  # Spatial SVD config starts

.. rubric:: Spatial SVD Configuration

.. autoclass:: aimet_torch.defs.SpatialSvdParameters
   :members:
   :noindex:

..
  # Spatial SVD config ends

..
  # Channel pruning config starts

.. rubric:: Channel Pruning Configuration

.. autoclass:: aimet_torch.defs.ChannelPruningParameters
   :members:
   :noindex:

..
  # Channel pruning config ends

..
  # Weight SVD config starts

.. rubric:: Weight SVD Configuration

.. autoclass:: aimet_torch.defs.WeightSvdParameters
   :members:
   :noindex:

..
  # Weight SVD config ends
