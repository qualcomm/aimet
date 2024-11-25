#########
Winnowing
#########

Overview
========

The channel pruning (CP) model compression algorithm identifies modules in a model for which
a subset of input channels can be pruned without losing much accuracy. Unless explicitly
removed, these input channels take up memory and add unnecessary computation. The winnow tool
removes the input channels that were selected for pruning.

Only 2D convolution (Conv2D) layers can be winnowed.

Winnowing
---------

The following figure illustrates winnowing. In this example, a module in a model has an
input volume of **HxWx8**, where:

- H is Height
- W is Width
- The number of input channels is 8

The CP algorithm has identified input channels 1, 4 and 7 for pruning. Winnowing removes
these input channels, and the module's input volume is reduced to **HxWx5**.

.. image:: ../../images/winnow_1.png

How winnowing works
-------------------

When the number of input channels of a convolution (Conv) module is reduced, the output
channels of its preceding module must also be modified. If the preceding module is a another
Conv layer, that Conv layer's output channels are reduced to the number of input channels of
the winnowed module. If the preceding module is not a Conv layer (as in BatchNorm and ReLU
modules), that module instead propagates the changes upstream. That is, both the output and
the input channels of either a BatchNorm or ReLU module are winnowed to match the reduced
channels of the downstream Conv layer.

The following figure illustrates a scenario in which a Conv module has been identified for
winnowing of some of its input channels. The Conv module inputs are denoted in green on the
left side of the figure. The right side of the figure indicates the actions taken by winnowing:

1. The identified subset of input channels are removed from the Conv module (indicated in pink).
2. The module just above the winnowed Conv module (colored orange) is `not` a Conv module. It
could be a ReLU or a BatchNorm module. Its corresponding output and input channels are winnowed.
3. The module above the ReLU/BatchNorm is another Conv module. This Conv module's output
channels (shown in pink) are winnowed.

.. image:: ../../images/winnow_2.png
