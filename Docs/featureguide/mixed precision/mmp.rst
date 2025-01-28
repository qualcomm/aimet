.. _featureguide-mmp:

######################
Manual mixed precision
######################

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

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [setup]
            :end-before: [set_precision_leaf]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Step 1: Applying MMP API options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    All requests are processed using the leaf layers in the model.

MMP provides the following APIs to change layers' precision. The APIs can be called in any order. In case of conflicts, the latest request overrides an older request. For example:

* If one of the following APIs is called multiple times but with a different precision for the same layer, only the latest call is serviced.
* The last request takes precedence even if the requests are from two different APIs. For example, say you call a non-leaf layer L1 with precision P1 and then a leaf layer L2, inside L1, with precision P2. This sets all the layers in L1 to precision P1, except layer L2 which is set to P2.

Set precision of a leaf layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_leaf]
            :end-before: [set_precision_non_leaf]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Set precision of a non-leaf layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_non_leaf]
            :end-before: [set_precision_type]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Set precision based on layer type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_type]
            :end-before: [set_precision_model_input]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Set model input precision
~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_model_input]
            :end-before: [set_precision_model_output]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

If a model has more than one input tensor (for example, the structure is [In1, In2]), you can set just one of them (say In2) to a new precision (say P1) by setting ``activation=[None, P1]`` in the above API.

Set model output precision
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_model_output]
            :end-before: [apply]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

If a model has more than one output tensor (for example, the structure is [Out1, Out2, Out3]), you can set just one of them (say Out2) to a new precision (say P1) by setting ``activation=[None, P1, None]`` in the above API.


Step 2: Applying the profile
----------------------------

All of the `set precision` family of calls from step 1 are processed at once when the following ``apply(...)`` API is called.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [apply]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


The ``apply`` call generates a report detailing how the request was inferred, propagated to other layers, and eventually realized.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/mp.rst
            :start-after: # start-after mmp
            :end-before: # end-before mmp

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

