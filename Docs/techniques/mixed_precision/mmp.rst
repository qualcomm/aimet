.. _featureguide-mmp:

######################
Manual mixed precision
######################

.. note::
    Not supported in ``aimet-onnx``.

Context
=======

To effectively use mixed precision, you must find the correct quantizers to run at higher precision settings. This requires complex, error-prone graph traversals. The AIMET manual mixed precision (MMP) configurator hides this issue by providing easy-to-use APIs to configure the model in mixed precision. You can change the precision of a layer by directly specifying the layer and the intended precision. MMP configurator also analyzes and reports how the mixed precision was achieved.

MMP configurator enables you to change the precision of the following within a model:

* A leaf layer
* A non-leaf layer (a layer composed of multiple leaf layers)
* All layers of a certain type
* Model input tensors or a subset of input tensors
* Model output tensors or a subset of output tensors


Workflow
========

Prerequisites
-------------

Manual mixed precision is supported only on PyTorch models.

Setup
-----

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [setup]
    :end-before: [set_precision_leaf]

Step 1: Applying MMP API options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    All requests are processed using the leaf layers in the model.

MMP provides the following APIs to change layers' precision. The APIs can be called in any order. In case of conflicts, the latest request overrides an older request. For example:

* If one of the following APIs is called multiple times but with a different precision for the same layer, only the latest call is serviced.
* The last request takes precedence even if the requests are from two different APIs. For example, say you call a non-leaf layer L1 with precision P1 and then a leaf layer L2, inside L1, with precision P2. This sets all the layers in L1 to precision P1, except layer L2 which is set to P2.

Set precision of a leaf layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [set_precision_leaf]
    :end-before: [set_precision_non_leaf]

Set precision of a non-leaf layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [set_precision_non_leaf]
    :end-before: [set_precision_type]

Set precision based on layer type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [set_precision_type]
    :end-before: [set_precision_model_input]

Set model input precision
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [set_precision_model_input]
    :end-before: [set_precision_model_output]

If a model has more than one input tensor (for example, the structure is [In1, In2]), you can set just one of them (say In2) to a new precision (say P1) by setting ``activation=[None, P1]`` in the above API.

Set model output precision
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [set_precision_model_output]
    :end-before: [apply]

If a model has more than one output tensor (for example, the structure is [Out1, Out2, Out3]), you can set just one of them (say Out2) to a new precision (say P1) by setting ``activation=[None, P1, None]`` in the above API.


Step 2: Applying the profile
----------------------------

All of the `set precision` family of calls from step 1 are processed at once when the following ``apply(...)`` API is called.

.. literalinclude:: ../../snippets/torch/apply_mmp.py
    :language: python
    :start-after: [apply]

The ``apply`` call generates a report detailing how the request was inferred, propagated to other layers, and eventually realized.

API
===

.. include:: ../../apiref/torch/mp.rst
    :start-after: # start-after mmp
    :end-before: # end-before mmp
