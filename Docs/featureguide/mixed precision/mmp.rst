.. _featureguide-mmp:

######################
Manual mixed precision
######################

Context
=======

To set the model in mixed precision, AIMET user would have to find the correct quantizer(s) and change to the new
settings. This requires complex graph traversals which are error prone. Manual Mixed Precision (MMP) Configurator hides
this issue by providing easy to use APIs to configure the model in mixed precision. User can change the precision of a
layer by directly specifying the layer and the intended precision. User would also get a report to analyze how it was achieved.

MMP configurator provides the following mechanisms to change the precision in a model

* Change the precision of a leaf layer
* Change the precision of a non-leaf layer (layer composed of multiple leaf layers)
* Change the precision of all the layers in the model of a certain type
* Change the precision of model input tensors (or only a subset of input tensors)
* Change the precision of model output tensors (or only a subset of output tensors)


Workflow
========

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

MMP API options
---------------

MMP provides the following APIs to change the precision. The APIs can be called in any order. But, in case of conflicts, latest request will triumph the older request.

.. note::
    The requests are processed using the leaf layers in the model

* If one of the below APIs is called multiple times for the same layer but with a different precision in each of those calls, only the latest one would be serviced
* This rule holds good even if the requests are from two different APIs ie if user calls a non-leaf layer (L1) with precision (P1) and a leaf layer inside L1 (L2) with precision (P2). This would be serviced by setting all the layers in L1 at P1 precision, except layer L2 which would be set at P2 precision.

Set precision of a leaf layer
-----------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_leaf]
            :end-before: [set_precision_non_leaf]


Set precision of a non-leaf layer
---------------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_non_leaf]
            :end-before: [set_precision_type]


Set precision based on layer type
---------------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_type]
            :end-before: [set_precision_model_input]

Set model input precision
-------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_model_input]
            :end-before: [set_precision_model_output]

* Do note that if a model has more than one input tensor (say the structure is [In1, In2]), but only one of them (say In2) needs to be configured to a new precision (say P1), user can achieve it by setting ``activation=[None, P1]`` in the above API

Set model output precision
--------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [set_precision_model_output]
            :end-before: [apply]

* Do note that if a model has more than one output tensor (say the structure is [Out1, Out2, Out3]), but only one of them (say Out2) needs to be configured to a new precision (say P1), user can achieve it by setting ``activation=[None, P1, None]`` in the above API

Apply the profile
-----------------

All the above `set precision` family of calls would be processed at once when the below ``apply(...)`` API is called

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../snippets/torch/apply_mmp.py
            :language: python
            :start-after: [apply]

.. note::
    The above call would generate a report detailing how a user's request was inferred, propagated to other layers and realized eventually

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/mp.rst
            :start-after: # start-after mmp
            :end-before: # end-before mmp
