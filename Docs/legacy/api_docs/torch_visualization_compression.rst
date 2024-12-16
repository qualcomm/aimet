
===================================
AIMET Visualization Compression API
===================================

Top-level API Compression
==========================



.. autoclass:: aimet_torch.visualize_serialized_data.VisualizeCompression

|

.. automethod:: aimet_torch.visualize_serialized_data.VisualizeCompression.display_eval_scores

|

.. automethod:: aimet_torch.visualize_serialized_data.VisualizeCompression.display_comp_ratio_plot



Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/visualization_compression.py
    :language: python
    :start-after: # Visualization imports
    :end-before: # End of import statements

**Model Compression with Visualization Parameter**

.. literalinclude:: ../torch_code_examples/visualization_compression.py
    :language: python
    :pyobject: model_compression_with_visualization
