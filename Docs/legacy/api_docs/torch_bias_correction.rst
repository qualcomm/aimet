:orphan:

.. _api-torch-bias-correction:

=================================
AIMET PyTorch Bias Correction API
=================================

User Guide Link
===============
To learn more about this technique, please see :ref:`Cross-Layer Equalization<ug-post-training-quantization>`

Bias Correction API
===================

.. autofunction:: aimet_torch.bias_correction.correct_bias

|

ConvBnInfoType
==============
.. autoclass:: aimet_common.bias_correction.ConvBnInfoType

|

ActivationType
==============
.. autoclass:: aimet_common.defs.ActivationType
   :members:

Quantization Params
===================

.. autoclass:: aimet_torch.v1.quantsim.QuantParams
    :members:

|

Code Example #1 Empirical Bias Correction
=========================================

**Load the model**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
   :language: python
   :start-after:  # Load the model empirical
   :end-before:   # Apply Empirical Bias Correction

**Apply Empirical Bias Correction**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
   :language: python
   :start-after:  # Apply Empirical Bias Correction
   :end-before:   # End of example empirical

|

Code Example #2 Analytical + Empirical Bias correction
======================================================

**Load the model**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
   :language: python
   :start-after:  # Load the model analytical_empirical
   :end-before:   # Find BNs

**Find BN and Conv Modules**

Find BN + Conv module pairs for analytical Bias Correction and remaining Conv modules for Empirical Bias Correction.

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
   :language: python
   :start-after:  # Find BNs
   :end-before:   # Apply Analytical and Empirical Bias Correction

**Apply Analytical + Empirical Bias Correction**

.. literalinclude:: ../torch_code_examples/post_training_techniques_examples.py
   :language: python
   :start-after:  # Apply Analytical and Empirical Bias Correction
   :end-before:   # End of example analytical_empirical