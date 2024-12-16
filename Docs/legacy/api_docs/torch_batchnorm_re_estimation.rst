:orphan:

.. _api-torch-bn-reestimation:

===============================================
AIMET PyTorch BatchNorm Re-estimation APIs
===============================================

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch Quantization-Aware Training followed by BatchNorm Re-estimation, please see :doc:`here<../Examples/torch/quantization/bn_reestimation>`

Introduction
============
Batch Norm (BN) Re-estimation re-estimates the statistics of BN layers after performing QAT. Using the re-estimated statistics, the BN layers are folded in to preceding Conv and Linear layers


Top-level APIs
============================

**API for BatchNorm Re-estimation**

.. autofunction:: aimet_torch.bn_reestimation.reestimate_bn_stats

**API for BatchNorm fold to scale**

.. autofunction:: aimet_torch.v1.batch_norm_fold.fold_all_batch_norms_to_scale

Code Example - BN-Reestimation
==============================

** Step 1. Load the model**

For this example, we are going to load a pretrained ResNet18 model from torchvision.

.. literalinclude:: ../torch_code_examples/bn_reestimation_example.py
   :language: python
   :start-after:  # Load FP32 model and prepare it
   :end-before:   # End of Load FP32 model and prepare it


**Step 2. Create QuantSim with Range Learning and Per Channel Quantization Enabled**

1. For an example of creating QuantSim with Range Learning QuantScheme, please see :doc:`here<../Examples/torch/quantization/qat_range_learning>`
2. For how to enable Per Channel Quantization, please see :doc:`here<../user_guide/quantization_configuration>`


**Step 3. Perform QAT**

.. literalinclude:: ../torch_code_examples/bn_reestimation_example.py
   :language: python
   :start-after:  # Perform QAT
   :end-before:   # End of Perform QAT

**Step 4 a. Perform BatchNorm Re-estimation**

.. literalinclude:: ../torch_code_examples/bn_reestimation_example.py
    :language: python
    :start-after: # Call reestimate_bn_stats
    :end-before:  # End of Call reestimate_bn_stats

**Step 4 b. Perform BatchNorm Fold to scale**

.. literalinclude:: ../torch_code_examples/bn_reestimation_example.py
    :language: python
    :start-after:  # Call fold_all_batch_norms_to_scale
    :end-before:  # End of Call fold_all_batch_norms_to_scale


**Step 5. Export the model and encodings and test on target**

For how to export the model and encodings, please see :doc:`here<../Examples/torch/quantization/bn_reestimation>`





