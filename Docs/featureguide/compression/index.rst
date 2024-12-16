.. _featureguide-compression-index:

###########
Compression
###########

.. toctree::
    :hidden:

    Compression guidebook <feature_guidebook>
    Greedy compression ratio selection <greedy_compression_ratio_selection>
    Visualization <visualization_compression>
    Weight SVD <weight_svd>
    Spatial SVD <spatial_svd>
    Channel pruning <channel_pruning>

Overview
========

AIMET provides model compression techniques that can reduce a model's multiply-and-accumulate (MAC)
and memory costs with little loss of accuracy. AIMET supports various compression techniques like weight
singular value decomposition (SVD), Spatial SVD, and Channel pruning.

See the :ref:`Compression features guidebook<featureguide-compression-guidebook>` for a summary of how
to use the compression features, and how to combine them.

Use Case
========

AIMET can compress your pre-trained model to a specified compression ratio. The model can then be
further fine-tuned and exported to a target.

All of the compression schemes in AIMET use a two-phase process:

1. Compression ratio selection
2. Model compression

.. image:: ../../../images/compression_use_case.PNG

Both of these phases are explained below.

Compression ratio selection
===========================

In this phase, you select compression ratios automatically and/or manually. You can use AIMET Visualization to inspect these choices.

Automatic compression ratio selection
   AIMET computes optimal compression ratios for each layer, using the :ref:`Greedy compression ratio selection<greedy-comp-ratio-selection>` method for automatic compression.

Manual compression ratio selection
   You can manually specify compression ratios by layer. We suggest that you first use automatic compression ratio selection to get a baseline set of compression ratios, then manually change compression ratios for one or more layers.

Visualization
   Visualize compression as you apply these steps using :doc:`AIMET Visualization<visualization_compression>`.

Model compression
=================

In this phase, AIMET applies the compression ratios to each layer to create a compressed model.
AIMET supports the following model compression algorithms:

:ref:`Weight SVD <featureguide-weight-svd>`
-------------------------------------------

Decomposes one large 2D convolution or fully connected layer (FC) into two smaller layers.

:ref:`Spatial SVD <featureguide-spatial-svd>`
---------------------------------------------

Decomposes one large 2D convolution layer into two smaller layers.

:ref:`Channel pruning (CP) <featureguide-channel-pruning>`
----------------------------------------------------------

Prunes redundant input channels from 2D convolution layers.

Optional techniques
===================

AIMET supports the following optional techniques that can improve compression results.

- Rank-rounding
- Per-layer fine-tuning

Rank Rounding
-------------

AIMET techniques like weight SVD, spatial SVD, and channel pruning decompose or reduce input and output channel layers.

Certain types of layers (such as 2D convolution (Conv2D) or fully connected (FC)) in embedded ML accelerators are often optimized for certain multiplicities. Matching the expected multiplicity gives optimal runtime performance for that layer.

The rank-rounding feature in AIMET tries to reduce layers to match a user-provided multiplicity. AIMET only allows you to specify a multiplicity factor for the entire model, not per layer. Use this feature to optimize models to run on embedded targets. By default the feature is disabled.

.. _per-layer-fine-tuning:

Per-layer fine-tuning
---------------------

For a given user model and compression ratio, compression sometimes causes a sharp drop in accuracy before fine-tuning. Per-layer fine-tuning is a technique that can help maintain model accuracy for desired compression ratios.

In this feature, AIMET invokes a user-provided fine-tuning function after compressing every layer that was selected for compression. This fine tuning is done during the model compression phase described above.

.. admonition:: NOTE

   This feature may require careful selection of learning rates and learning-rate-decay parameters to be used during fine-tuning. You are responsible for choosing these training parameters.

FAQs
====

1. Which is the best technique to use for compression?

   *We see best results when Spatial SVD is performed followed by Channel pruning.*

2. Can I combine the different techniques?

   *Yes, different techniques can be combined to get better accuracy. Compression can be also combined with post-training quantization (PTQ) techniques to get a better model for target.*

3. How do I take a model to target after compression?

   *First, compress the model using the techniques described above. Then, quantize the model and export it to target.*

4. Greedy rank selection is very slow. Can something be done to speed it up?

   *The time-consuming part is creating the eval-score dictionary, not greedy rank selection itself. A single eval-score dictionary can be generated once and then loaded into the searcher for different experiments. Or, reduce the number of candidates over which the eval-score dictionary is created, but be aware that the fewer candidates, the worse the granularity. The default value of 10 candidates usually strikes a good balance.*

5. Is per-layer fine tuning helpful?

   *Per-layer fine tuning is an experimental technique. We have not observed major gains by using it. But, you can try it to see if it works for your model. In practice, the best results seem to come from doing one epoch of fine-tuning per layer, and then doing 10-15 epochs of fine-tuning for the entire compressed model at the end.*

References
==========

1.	Xiangyu Zhang, Jianhua Zou, Kaiming He, and Jian Sun. "Accelerating Very Deep Convolutional Networks for Classification and Detection." IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 10, pp. 1943-1955, 1 Oct. 2016.

2.	Yihui He, Xiangyu Zhang, and Jian Sun. "Channel Pruning for Accelerating Very Deep Neural Networks." IEEE International Conference on Computer Vision (ICCV), Venice, 2017, pp. 1398-1406.

3.	Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. "Speeding up Convolutional Neural Networks with Low Rank Expansions." British Machine Vision Conference, Jan. 2014.

4.	Andrey Kuzmin, Markus Nagel, Saurabh Pitre, Sandeep Pendyam, Tijmen Blankevoort, Max Welling. "Taxonomy and Evaluation of Structured Compression of Convolutional Neural Networks."
