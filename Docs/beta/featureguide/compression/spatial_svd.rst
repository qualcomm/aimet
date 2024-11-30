.. _featureguide-spatial-svd:

###########
Spatial SVD
###########

Context
=======

Spatial singular value decomposition (spatial SVD) is a tensor decomposition technique which decomposes
one large layer (in terms of Multiply-accumulate(MAC) or memory) into two smaller layers.

Consider a convolution (Conv) layer with kernel (𝑚, 𝑛, ℎ, 𝑤), where:

- 𝑚 is the input channels
- 𝑛 the output channels
- ℎ is the height of the kernel
- 𝑤 is the width of the kernel

Spatial SVD decomposes the kernel into two kernels, one of size (𝑚, 𝑘, ℎ, 1) and one of size (𝑘, 𝑛, 1, 𝑤),
where 𝑘 is called the `rank`. The smaller the value of 𝑘, the larger the degree of compression.

The following figure illustrates how spatial SVD decomposes both the output channel dimension and the size
of the Conv kernel itself.

.. image:: ../../../images/spatial_svd.png
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

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../../../keras_code_examples/compression_code_examples.py
           :language: python
           :lines: 39-49

        .. literalinclude:: ../../../keras_code_examples/compression_code_examples.py
           :language: python
           :pyobject: get_eval_func

Compressing using Spatial SVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        **Compressing using Spatial SVD in auto mode with multiplicity = 8 for rank rounding**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :pyobject: spatial_svd_auto_mode

        **Compressing using Spatial SVD in manual mode**

        .. literalinclude:: ../../../torch_code_examples/code_examples.py
           :language: python
           :pyobject: spatial_svd_manual_mode

    .. tab-item:: TensorFlow
        :sync: tf

        **Compressing using Spatial SVD in auto mode**

        .. literalinclude:: ../../../keras_code_examples/compression_code_examples.py
            :language: python
            :pyobject: aimet_spatial_svd

        **Sample Driver Code for Spatial SVD using Resnet50**

        .. literalinclude:: ../../../keras_code_examples/compression_code_examples.py
            :language: python
            :pyobject: compress

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
           :start-after: # Spatial SVD config starts
           :end-before: # Spatial SVD config ends

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../../apiref/tensorflow/compress.rst
           :start-after: _apiref-tensorflow-compress:
