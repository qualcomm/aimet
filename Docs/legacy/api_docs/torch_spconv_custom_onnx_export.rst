:orphan:

.. _api-torch-spconv-custom-onnx-export:

=======================================
AIMET Torch SparseConvolution custom onnx export
=======================================

This page gives an idea on how SparseConvolution based models can be used in AIMET. The SparseConvolution library
that is used/supported here is `Traveller's SparseConvolution Library`_.
Please note that,

- Only `SparseConvolution3D` is supported as of now.

- SpConv library (for `cpu`) is `not very stable` because it is found that the inference from the spconv module gives
  different outputs for the same inputs in the same runtime.

- If there's `bias` in the SpConv layer, please use `GPU`, as `bias` in SpConv is only supported in GPU.


.. _Traveller's SparseConvolution Library: https://github.com/traveller59/spconv

Custom API for the spconv modules
=================================

**The following api can be used to create a sparse tensor given indices and features in dense form**

.. autoclass:: aimet_torch.nn.modules.custom.SparseTensorWrapper

**The following api can be used to create a dense tensor given a sparse tensor**

.. autoclass:: aimet_torch.nn.modules.custom.ScatterDense

Code Example
=============

**Imports**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Create or load model with SpConv3D module(s)**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 1. Create or load model with SpConv3D module(s)
    :end-before: # End Step 1

**Obtain model inputs**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 2. Obtain model inputs
    :end-before: # End Step 2

**Apply model preparer pro**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 3. Apply model preparer pro
    :end-before: # End Step 3

**Apply QuantSim (or any other AIMET features)**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 4. Apply QuantSim
    :end-before: # End Step 4

**Compute Encodings**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 5. Compute encodings
    :end-before: # End Step 5

**QuantSim Exports**

.. literalinclude:: ../torch_code_examples/spconv3d_example.py
    :language: python
    :start-after: # Step 6. QuantSim export
    :end-before: # End Step 6


