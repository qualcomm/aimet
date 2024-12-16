.. _featureguide-compression-guidebook:

##############################
Compression features Guidebook
##############################

This page outlines a typical workflow for compressing a neural network using AIMET.

AIMET supports model compression using the following techniques:

- :ref:`Weight SVD <featureguide-weight-svd>`
- :ref:`Spatial SVD <featureguide-spatial-svd>`
- :ref:`Channel pruning (CP) <featureguide-channel-pruning>`

These are techniques for Multiply-and-Accumulate (MAC) reduction of convolution layers in a neural
network. You specify a MAC reduction ratio (MACs in the compressed model divided by MACs in the
uncompressed model). AIMET's compression algorithms automatically compress each convolution layer
in the network to (approximately) achieve the overall desired MAC reduction.

The on-target inference latency performance of a model depends on several factors: MACs, memory,
memory bandwidth, quantization, etc. Therefore, the improvement in runtime latency gained from
MAC-reduction compression can vary.

For best performance, we recommend spatial SVD followed by Channel pruning.

At a high level, you use the following steps to compress a network using a combination of spatial SVD
and Channel pruning:

.. image:: ../../../images/compression_flow.png
   :height: 500
   :width: 600

1. Choose a target compression ratio (C), which is the ratio of MACs in the final compressed model to the MACs in the uncompressed model. For example, a target compression ratio of 0.5 indicates that there are half as many MACs in the final model as in the original model.

2. Perform compression using spatial SVD as follows:

    a. Since the target compression ratio C is for the final spatial SVD-with-CP (SVD+CP) compressed model, the compression that should be targeted or can be achieved via spatial SVD is unknown at the start. As a result, you must try a few target compression ratios  (C\ :sub:`ssvd`). Choose a few C\ :sub:`ssvd` values greater than or equal to your target C and perform spatial SVD. For example, if C = 0.5, try C\ :sub:`ssvd` = {0.5, f0.65, 0.75}. This yields three spatial SVD compressed models.

    b. For each of the spatial SVD compressed models obtained in the previous step, perform :ref:`fine-tuning<per-layer-fine-tuning>` to improve model accuracy.

3. Pick a model (or a few models) that provide high accuracy from step 2b. For example, if the tolerable accuracy drop SVD+CP\ :sub:`compression` relative to the original uncompressed model is X%  (X = accuracy of the uncompressed model (%) divided by accuracy of the compressed model (%)) , then a model that has accuracy near (say within 5%) of the original uncompressed model accuracy should be selected to avoid a very large drop in accuracy after the CP step.

    If step 2b results in a very large accuracy drop or a drop not within tolerable accuracy, then step 2 should be revisited with adjusted compression ratios.

4. Perform compression using the CP technique as follows:

    a. Perform compression with a few target compression ratios (C\ :sub:`cp`). You can set the compression ratio(s) based on the C\ :sub:`ssvd` of the model obtained from spatial SVD step 3 such that C\ :sub:`ssvd` * C\ :sub:`cp` is approximately equal to C.

    b. Perform fine-tuning to improve model accuracy.

5. In the final step, select a model with a MAC ratio relative to the original uncompressed model close to C that meets your accuracy requirements. For example, for ResNet-50 results provided on `The AIMET product website <https://quic.github.io/aimet-pages/index.html>`_, C\ :sub:`ssvd` = 0.75 and C\ :sub:`cp` = 0.66 were used to achieve an overall compression of C = 0.5.
