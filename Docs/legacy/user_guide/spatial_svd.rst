:orphan:

.. _ug-spatial-svd:

#################
AIMET spatial SVD
#################

Spatial singular value decomposition (spatial SVD) is a technique that decomposes one large convolution (Conv) MAC or memory layer into two smaller layers.

Consider a Conv layer with kernel (𝑚,𝑛,ℎ,𝑤), where:

- 𝑚 is the input channels
- 𝑛 the output channels
- ℎ is the height of the kernel
- 𝑤 is the width of the kernel 
  
Spatial SVD decomposes the kernel into two kernels, one of size (𝑚,𝑘,ℎ,1) and one of size (𝑘,𝑛,1,𝑤), where 𝑘 is called the `rank`. The smaller the value of 𝑘, the larger the degree of compression.

The following figure illustrates how spatial SVD decomposes both the output channel dimension and the size of the Conv kernel itself. 

.. image:: ../images/spatial_svd.png
    :width: 900px
