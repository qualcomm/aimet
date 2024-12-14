.. include:: ../../abbreviation.txt

.. _featureguide-amp:

#########################
Automatic mixed precision
#########################

This technique helps choose per-layer integer bit-widths to retain model accuracy when run on
fixed-point runtimes like |qnn|_.

As an example, say a particular model is not meeting a desired accuracy target when run in INT8.
The Auto Mixed Precision (AMP) feature will find a minimal set of layers that need to run on higher
precision, INT16 for example, to get to the desired quantized accuracy.

Choosing a higher precision for some layers necessarily involves a trade-off: lower inferences/sec
for higher accuracy and vice-versa. The AMP feature will generate a pareto curve that can guide
the user to decide the right operating point for this tradeoff.

Context
=======

For performing AMP, a user needs to start with a PyTorch, TensorFlow or ONNX model and create a
Quantization Simulation model :class:`QuantizationSimModel`. This QuantSim model, along with an
allowable accuracy drop, is passed to the API.

The function changes the QuantSim Sim model in place with different quantizers having different
bit-widths. This QuantSim model can be either exported or evaluated to get a quantization accuracy.

.. image:: ../../images/automatic_mixed_precision_1.png
    :width: 900px

Mixed Precision Algorithm
=========================

The algorithm involves 4 phases:

.. image:: ../../images/automatic_mixed_precision_2.png
    :width: 700px

1) Find layer groups
--------------------

    Layer Groups are defined as a group of layers grouped together based on certain rules.
    This helps in reducing search space over which the mixed precision algorithm operates.
    It also ensures that we search only over the valid bit-width settings for parameters and activations.

.. image:: ../../images/automatic_mixed_precision_3.png
    :width: 900px

2) Perform sensitivity analysis (Phase 1)
-----------------------------------------

    In this phase the algorithm performs a per-layer group sensitivity analysis.
    This will identify how sensitive is the model if we choose a lower quantization bit-width for a particular layer group.
    The sensitivity analysis yields an accuracy list which is cached and can be re-used again by the algorithm.

    Below is an example of a list generated using sensitivity analysis:

    .. image:: ../../images/accuracy_list.png
        :width: 900px

3) Create a Pareto-front list (Phase 2)
---------------------------------------

    A Pareto curve is a trade-off curve that describes how accuracy varies given a bit-ops target and vice versa.
    The AMP algorithm yields a Pareto front curve which consists of layer groups changed up to that point, relative bit-ops (relative to starting bit-ops),
    accuracy of the model, and the bit-width to which the layer group was changed to.

    An example of a Pareto list:

    .. image:: ../../images/pareto.png
        :width: 900px

    Bit-ops are computed as

    :math:`Bit-ops = Mac(op) * Bitwidth(parameter) * Bitwidth(Activation)`

    The Pareto list can be used for plotting a Pareto curve. A Bokeh plot for Pareto curve is generated and saved in the results directory.

    .. image:: ../../images/pareto_curve.png
        :width: 900px

.. note::

    A user can pass two different evaluation callbacks for phase 1 and phase 2. Since phase 1 is measuring sensitivity
    of each quantizer group, we can pass a smaller representative dataset for phase 1 for evaluation, or even use an indirect measure
    such as SQNR which can be computed faster than but correlates well with the real evaluation metric.

It is recommended to use the complete dataset for evaluation in phase 2.

4) Reduce Bit-width Convert Op Overhead (Phase 3)
-------------------------------------------------

Convert Ops are introduced in the mixed-precision model for transition between Ops that are assigned different activation
bit-widths or data types (float vs int). These Convert Ops contribute to the inference time along with bit-operations of Ops.
In this phase the algorithm derives a mixed-precision solution having less Convert Op overhead w.r.t. to original solution
keeping the mixed-precision accuracy intact. The algorithm produces mixed-precision solutions for a range of alpha values
(0.0, 0.2, 0.4, 0.6, 0.8, 1.0) where the alpha represents fraction of original Convert Op overhead allowed for respective solution.

Use Cases
=========

1) Choosing a very high accuracy drop (equivalent to setting allowed_accuracy_drop as None):

AIMET allows a user to save intermediate states for computation of the Pareto list. Therefore, if a user computes a Pareto
list corresponding to an accuracy drop of None, they can view the complete profile of how model accuracy will vary as bit-ops vary.

Thereafter, a user can visualize the Pareto curve plot and choose an optimal point for accuracy. The algorithm can be re-run with
the new accuracy drop to get a sim model with the required accuracy.

.. note::

    The Pareto list is not modified during the second run.

2) Choosing a lower accuracy drop and then continuing to compute pareto list from this point if more accuracy drop is acceptable:

To enable this a user can use the clean_start parameter in the API. If clean_start is set to False then the Pareto list will
start computation from the last point where it left off.

.. note::

    - It is recommended to set the clean_start parameter to False to use cached results for both use cases.
    - If the model or candidate bit-widths change, the user needs to do a clean start.

Workflow
========

Code example
------------

Step 1
~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Required imports**

        .. literalinclude:: ../../legacy/torch_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Load the model, define forward_pass and evaluation callbacks**

        .. literalinclude:: ../../legacy/torch_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 1
            :end-before: # End step 1

    .. tab-item:: TensorFlow
        :sync: tf

        **Required imports**

        .. literalinclude:: ../../legacy/keras_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Load the model, define forward_pass and evaluation callbacks**

        .. literalinclude:: ../../legacy/keras_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 1
            :end-before: # End step 1

    .. tab-item:: ONNX
        :sync: onnx

        **Required imports**

        .. literalinclude:: ../../legacy/onnx_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Instantiate a PyTorch model, convert to ONNX graph, define forward_pass and evaluation callbacks**

        .. literalinclude:: ../../legacy/onnx_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 1
            :end-before: # End step 1

Step 2
~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Quantization with mixed precision**

        .. literalinclude:: ../../legacy/torch_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 2
            :end-before: # End step 2

    .. tab-item:: TensorFlow
        :sync: tf

        **Quantization with regular mixed precision**

        .. literalinclude:: ../../legacy/keras_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 2
            :end-before: # End step 2

    .. tab-item:: ONNX
        :sync: onnx

        **Quantization with mixed precision**

        .. literalinclude:: ../../legacy/onnx_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 2
            :end-before: # End step 2

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/mp.rst
            :start-after: # start-after amp

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../../apiref/tensorflow/amp.rst
           :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../../apiref/onnx/amp.rst
           :start-after: # start-after
