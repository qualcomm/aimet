:orphan:

.. _api-torch-quant-analyzer:

================================
AIMET PyTorch Quant Analyzer API
================================

AIMET PyTorch Quant Analyzer analyzes the PyTorch model and points out sensitive layers to quantization in the model.
It checks model sensitivity to weight and activation quantization, performs per layer sensitivity and MSE analysis.
It also exports per layer encodings min and max ranges and statistics histogram for every layer.

User Guide Link
===============
To learn more about this technique, please see :ref:`QuantAnalyzer<ug-quant-analyzer>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch QuantAnalyzer, please see :doc:`here<../Examples/torch/quantization/quant_analyzer>`.

Top-level API
=============


.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API. To
    learn more about the differences between :mod:`aimet_torch` and :mod:`aimet_torch.v2`, please visit the
    :ref:`QuantSim v2 Overview<ug-aimet-torch-v2-overview>`.

.. autoclass:: aimet_torch.v1.quant_analyzer.QuantAnalyzer
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.enable_per_layer_mse_loss
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.analyze
|
.. autoclass:: aimet_common.utils.CallbackFunc
|

Run specific utility
====================
We can avoid running all the utilities that QuantAnalyzer offers and only run those of our interest. For this we
need to have the QuantizationSimModel object, Then we call the desired QuantAnalyzer utility of our interest and pass
the same object to it.

.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.check_model_sensitivity_to_quantization
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_enabling_quant_wrappers
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.perform_per_layer_analysis_by_disabling_quant_wrappers
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.export_per_layer_encoding_min_max_range
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.export_per_layer_stats_histogram
|
.. automethod:: aimet_torch.v1.quant_analyzer.QuantAnalyzer.export_per_layer_mse_loss
|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Prepare forward pass callback**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 1. Prepare forward pass callback
    :end-before: # End step 1

**Prepare eval callback**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 2. Prepare eval callback
    :end-before: # End step 2

**Prepare model and callback functions**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 3. Prepare model and callback functions
    :end-before: # End step 3

**Create QuantAnalyzer object**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 4. Create QuantAnalyzer object
    :end-before: # End step 4

**Run QuantAnalyzer**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 5. Run QuantAnalyzer
    :end-before: # End step 5
