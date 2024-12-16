################
AIMET weight SVD
################

Weight singular value decomposition (SVD) is a technique that decomposes one large layer (in terms of MAC or memory) into two smaller layers.

Consider a neural network layer with the kernel (𝑚,𝑛,ℎ,𝑤) where:

- 𝑚 is the input channels
-  𝑛 the output channels
-  ℎ is the height of the kernel
-  𝑤 is the width of the kernel 

Weight SVD decomposes the kernel into one of size (𝑚,𝑘,1,1) and another of size (𝑘,𝑛,h,𝑤), where 𝑘 is called the `rank`. The smaller the value of 𝑘, larger the degree of compression.

The following figure illustrates how weight SVD decomposes the output channel dimension. Weight SVD is currently supported for convolution (`Conv`) and fully connected (FC) layers in AIMET.

.. image:: ../images/weight_svd.png
    :width: 900px
