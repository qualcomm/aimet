.. include:: ../../abbreviation.txt

.. _featureguide-amp:

#########################
Automatic mixed precision
#########################

Automatic mixed precision (AMP) helps choose per-layer integer bit widths to retain model accuracy on
fixed-point runtimes like |qnn|_.

For example, consider a model that is not meeting an accuracy target when run in INT8.
AMP finds a minimal set of layers that need to run on higher precision, INT16 for example, to achieve the target accuracy.

Choosing a higher precision for some layers involves a trade-off between performance (inferences per second)
and better accuracy. The AMP feature generates a Pareto curve you can use to help decide the right operating point for this tradeoff.

Context
=======

To perform AMP, you need a PyTorch, TensorFlow, or ONNX model. You use the model to create a
Quantization Simulation (QuantSim) model :class:`QuantizationSimModel`. This QuantSim model, along with an
allowable accuracy drop, is passed to the API.

The API function changes the QuantSim model in-place with different bit-width quantizers. You can export or evaluate this QuantSim model to calculate a quantization accuracy.

.. image:: ../../images/automatic_mixed_precision_1.png
    :width: 900px

Mixed Precision Algorithm
-------------------------

The algorithm involves four phases as shown in the following image.

.. image:: ../../images/automatic_mixed_precision_2.png
    :width: 700px

Phase 1: Find layer groups
~~~~~~~~~~~~~~~~~~~~~~~~~~

Layer Groups are defined based on certain rules.
Grouping layers helps reduce the search space over which the mixed precision algorithm operates.
It also ensures that the search occurs only over the valid bit-width settings for parameters and activations.

.. image:: ../../images/automatic_mixed_precision_3.png
    :width: 900px

Phase 2: Perform sensitivity analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm performs a per-layer group sensitivity analysis.
This identifies how sensitive the model is to lower quantization bit width for particular layer groups.
The sensitivity analysis creates and caches an accuracy list that is used in following phases by the algorithm.

Following is an an accuracy list generated using sensitivity analysis:

    .. image:: ../../images/accuracy_list.png
        :width: 900px

Phase 3: Create a Pareto-front list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Pareto curve or Pareto front describes the tradeoff between accuracy and bit-ops targets.
The AMP algorithm generates a Pareto curve showing, for each layer group changed:

- Bitwidth: The bit width to which the layer group was changed
- Accuracy: The accuracy of the model
- Relative bit-ops: The bit-ops relative to starting

An example of a Pareto list:

.. image:: ../../images/pareto.png
    :width: 900px

Bit-ops are computed as:

:math:`Bitops = Mac(op) * Bitwidth(parameter) * Bitwidth(Activation)`

The Pareto list can be used for plotting a Pareto curve. A plot of the Pareto curve is generated using Bokeh and saved in the results directory.

.. image:: ../../images/pareto_curve.png
    :width: 900px

You can pass two different evaluation callbacks for phase 1 and phase 2.

Since phase 1 measures sensitivity of each quantizer group, it can use a smaller representative dataset for evaluation, or even use an indirect measure such as SQNR that correlates with the direct evaluation metric but can be computed faster.

We recommend that you use the complete dataset for evaluation in phase 2.

Phase 4: Reduce bit-width convert op overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conversion operations (convert ops) are introduced in the mixed-precision model for transition between ops with different activation bit widths or data types (float vs int). Convert ops contribute to the inference time along with bit-operations of ops.

In this phase the algorithm derives a mixed-precision solution having less convert op overhead compared to the original solution, keeping the mixed-precision accuracy intact. The algorithm produces mixed-precision solutions for a range of alpha values (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) where the alpha represents the fraction of original convert op overhead allowed for a respective solution.

Use Cases
---------

1: Choosing a very high accuracy drop (equivalent to setting allowed_accuracy_drop to None)
    AIMET enables a user to save intermediate states for computation of the Pareto list. Computing a Pareto list corresponding to an accuracy drop of None generates the complete profile of model accuracy vs. bit-ops. You can thus visualize the Pareto curve plot and choose an optimal point for accuracy. The algorithm can be re-run with the new accuracy drop to get a sim model with the required accuracy.

    .. note::

        The Pareto list is not modified during the second run.

2: Choosing a lower accuracy drop and then continuing to compute a Pareto list
    Use this option if more accuracy drop is acceptable. Setting the clean_start parameter in the API to False causes the Pareto list to start computation from the point where it left off.

    .. note::

        - We recommend that you set the clean_start parameter to False to use cached results for both use cases.
        - If the model or candidate bit widths change, you must do a clean start.

Workflow
========

Procedure
---------

Step 1
~~~~~~

Setting up the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Import packages**

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

        **Import packages**

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

        **Import packages**

        .. literalinclude:: ../../legacy/onnx_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 0. Import statements
            :end-before: # End step 0

        **Instantiate a PyTorch model, convert to an ONNX graph, define forward_pass and evaluation callbacks**

        .. literalinclude:: ../../legacy/onnx_code_examples/mixed_precision.py
            :language: python
            :start-after: # Step 1
            :end-before: # End step 1

Step 2
~~~~~~

Quantizing the model.

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
