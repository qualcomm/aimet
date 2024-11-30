.. _featureguide-weight-svd:

##########
Weight SVD
##########

Context
=======

Weight singular value decomposition (Weight SVD) is a technique that decomposes one large
layer (in terms of multiply-accumulate (MAC) or memory) into two smaller layers.

Consider a convolution (Conv) layer with the kernel (m, n, h, w) where:

-  m is the input channels
-  n the output channels
-  h is the height of the kernel
-  w is the width of the kernel

Weight SVD decomposes the kernel into one of size (m, k, 1, 1) and another of size (k, n, h, w),
where 𝑘 is called the `rank`. The smaller the value of k, larger the degree of compression.

The following figure illustrates how weight SVD decomposes the output channel dimension. Weight SVD
is currently supported for convolution (`Conv`) and fully connected (`FC`) layers in AIMET.

.. image:: ../../../images/weight_svd.png
    :width: 900px

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

Compression using Weight SVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Compressing using Weight SVD in auto mode**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
            :language: python
            :pyobject: weight_svd_auto_mode

        **Compressing using Weight SVD in manual mode with multiplicity = 8 for rank rounding**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
            :language: python
            :pyobject: weight_svd_manual_mode

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
           :start-after: # Weight SVD config starts
           :end-before: # Weight SVD config ends

