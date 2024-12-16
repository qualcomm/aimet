.. _featureguide-channel-pruning:

###############
Channel pruning
###############

.. toctree::
    :hidden:

    Winnowing<winnowing>

Context
=======

Channel pruning (CP) is a model compression technique that removes less-important input
channels from layers. AIMET supports channel pruning of 2D convolution (Conv2D) layers.

Procedure
---------

The following figure illustrates the different steps in channel pruning a layer. These
steps are repeated for all layers selected for compression in order of occurrence from the
top of the model.

.. image:: ../../../images/channel_pruning_1.png

The steps are explained below.

Channel selection
-----------------

For a layer and a specified compression ratio, channel selection analyzes the magnitude of
each input channel based its kernel weights. It chooses the lowest-magnitude channels to be
pruned.

Winnowing
---------

Winnowing is the process of removing the input channels identified in channel selection,
resulting in compressed tensors:

.. image:: ../../../images/cp_2.png

Once one or more input channels of a layer are removed, corresponding output channels of
an upstream layer are also be removed to gain further compression. Skip-connections or
residuals sometimes prevent upstream layers from being output-pruned.

.. image:: ../../../images/cp_3.jpg

For more details on winnowing, see :doc:`Winnowing<winnowing>`.

Weight reconstruction
---------------------

The final step in CP is to adjust the weight and bias parameters of a pruned layer to try
and match pre-pruning output values. AIMET does this by performing linear regression on
random samples of the layer's input from the pruned model against corresponding output from
the original model.

.. image:: ../../../images/cp_4.jpg

Workflow
========

Code example
------------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :lines: 40-49

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :pyobject: evaluate_model

Compression using Channel Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Compressing using Channel Pruning in auto mode.**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :pyobject: channel_pruning_auto_mode

        **Compressing using Channel Pruning in manual mode.**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :pyobject: channel_pruning_manual_mode

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../../apiref/torch/compress.rst
           :start-after: # common APIs start
           :end-before: # common APIs end

        .. include:: ../../apiref/torch/compress.rst
           :start-after: # Channel pruning config starts
           :end-before: # Channel pruning config ends
