
========================================
AIMET Visualization for Quantization API
========================================

Top-level API Quantization
==========================

.. autofunction:: aimet_torch.visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers

|

.. autofunction:: aimet_torch.visualize_model.visualize_weight_ranges


|

.. autofunction:: aimet_torch.visualize_model.visualize_changes_after_optimization

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/visualization_quantization.py
   :language: python
   :start-after: # Quantization visualization imports
   :end-before: # End of import statements

**Comparing Model After Optimization**

.. literalinclude:: ../torch_code_examples/visualization_quantization.py
    :language: python
    :pyobject: visualize_changes_in_model_after_and_before_cle

**Visualizing weight ranges in Model**

.. literalinclude:: ../torch_code_examples/visualization_quantization.py
    :language: python
    :pyobject: visualize_weight_ranges_model

**Visualizing Relative weight ranges in Model**

.. literalinclude:: ../torch_code_examples/visualization_quantization.py
    :language: python
    :pyobject: visualize_relative_weight_ranges_model



