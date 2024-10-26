

.. _api-torch-quantizers:


.. currentmodule:: aimet_torch.v2.quantization.affine.quantizer


==========
Quantizers
==========

AIMET quantizers are the low-level components of :ref:`quantized modules<api-torch-quantized-modules>` that
implement the quantization mechanism for PyTorch tensors.

AIMET quantizers are PyTorch modules that take a torch.Tensor as input and return 
a :class:`QuantizedTensor<aimet_torch.v2.quantization.tensor.QuantizedTensor>`
or :class:`DequantizedTensors<aimet_torch.v2.quantization.tensor.DequantizedTensor>`,
a subclass of regular torch.Tensor with some additional attributes and helper functions for quantization.
All quantizers are derived from the base class :class:`QuantizerBase` defined as below.

.. autoclass:: aimet_torch.v2.quantization.base.quantizer.QuantizerBase
    :members: forward, compute_encodings, is_initialized




Affine Quantizers
=================

Even though it is **strongly recommended** for most users to delegate the instantiation and configuration of quantizers to :ref:`QuantizationSimModel<api-torch-quantsim-v2>`,
it is worth understanding the underlying mechanism of quantizers for finer control over the quantized model.

The most commonly used quantizers are the affine quantizers such as :class:`QuantizeDequantize`.
Here is a quick example of how to create an 8-bit asymmetric affine quantizer.

.. code-block:: Python

    import aimet_torch.v2.quantization as Q
    qtzr = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    print(qtzr)

.. rst-class:: script-output

  .. code-block:: none

    QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)


Once you have created a quantizer object, you are first required to initialize the range of the input tensors
from which the quantization scale and offset will be derived. The most common way and recommended way to achieve
this is by using :meth:`QuantizerBase.compute_encodings`.

.. code-block:: Python

    print(f"Before compute_encodings:")
    print(f"  * is_initialized: {qtzr.is_initialized()}")
    print(f"  * scale: {qtzr.get_scale()}")
    print(f"  * offset: {qtzr.get_offset()}")
    print()

    input = torch.arange(256) / 256 # [0, 1/256, 2/256, ..., 255/256]

    with qtzr.compute_encodings():
        _ = qtzr(input)

    print(f"After compute_encodings:")
    print(f"  * is_initialized: {qtzr.is_initialized()}")
    print(f"  * scale: {qtzr.get_scale()}")
    print(f"  * offset: {qtzr.get_offset()}")
    print()

    # Quantizer encoding initialized. Now we're ready to run forward
    input_dq = qtzr(input)

.. rst-class:: script-output

  .. code-block:: none

    Before compute_encodings:
      * is_initialized: False
      * scale: None
      * offset: None

    After compute_encodings:
      * is_initialized: True
      * scale: tensor(0.0039, grad_fn=<DivBackward0>)
      * offset: tensor(0., grad_fn=<SubBackward0>)


Note that the output of the quantizer is either a :class:`QuantizedTensor<aimet_torch.v2.quantization.tensor.QuantizedTensor>` or :class:`DequantizedTensors<aimet_torch.v2.quantization.tensor.DequantizedTensor>`.

.. code-block:: Python

    print("Output (dequantized representation):")
    print(input_dq)
    print(f"  * scale: {input_dq.encoding.scale}")
    print(f"  * offset: {input_dq.encoding.offset}")
    print(f"  * bitwidth: {input_dq.encoding.bitwidth}")
    print(f"  * signed: {input_dq.encoding.signed}")
    print()

    input_q = input_dq.quantize() # Integer representation of input_dq
    print("Output (quantized representation):")
    print(input_q)
    print(f"  * scale: {input_q.encoding.scale}")
    print(f"  * offset: {input_q.encoding.offset}")
    print(f"  * bitwidth: {input_q.encoding.bitwidth}")
    print(f"  * signed: {input_q.encoding.signed}")

    # Sanity checks
    # 1. Quantizing and dequantizing input_dq shouldn't change the result
    assert torch.equal(input_dq, input_q.dequantize())
    # 2. (De-)Quantizing an already (de-)quantized tensor shouldn't change the result
    assert torch.equal(input_dq, input_dq.dequantize())
    assert torch.equal(input_q, input_q.quantize())


.. rst-class:: script-output

  .. code-block:: none

    Output (dequantized representation):
    DequantizedTensor([0.0000, 0.0039, 0.0078, 0.0117, 0.0156, 0.0195, 0.0234,
                       0.0273, 0.0312, 0.0352, 0.0391, 0.0430, 0.0469, 0.0508,
                       ...,
                       0.9570, 0.9609, 0.9648, 0.9688, 0.9727, 0.9766, 0.9805,
                       0.9844, 0.9883, 0.9922, 0.9961], grad_fn=<AliasBackward0>)
      * scale: tensor(0.0039, grad_fn=<DivBackward0>)
      * offset: tensor(0., grad_fn=<SubBackward0>)
      * bitwidth: 8
      * signed: False

    Output (quantized representation):
    QuantizedTensor([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
                      10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
                     ...,
                     240., 241., 242., 243., 244., 245., 246., 247., 248., 249.,
                     250., 251., 252., 253., 254., 255.], grad_fn=<AliasBackward0>)
      * scale: tensor(0.0039, grad_fn=<DivBackward0>)
      * offset: tensor(0., grad_fn=<SubBackward0>)
      * bitwidth: 8
      * signed: False



Channelwise Quantization
========================

Channelwise quantization is one of the advanced usages of affine quantizers where
one scale and offset will be associated with only one channel of the input tensor,
whereas one scale and offset was associated with the entire tensor in the previous example.

..
    TODO (kyunggeu): We need some visual diagram here

Channelwise quantization can be easily done by creating the quantizer with the desired shape of scale and offset.

.. code-block:: Python

    import aimet_torch.v2.quantization as Q
    N, C, H, W = 1, 3, 16, 16

    input = torch.empty(N, C, H, W)
    input[:, 0, :, :] = (torch.arange(256) / 256).view(16, 16)
    input[:, 1, :, :] = input[:, 0, :, :] * 2
    input[:, 2, :, :] = input[:, 1, :, :] * 4

    # Channelwise quantization along the channel axis (C) of the input
    qtzr = Q.affine.QuantizeDequantize(shape=(1, C, 1, 1), bitwidth=8, symmetric=False)
    print(qtzr)

    with qtzr.compute_encodings():
        _ = qtzr(input)

    scale = qtzr.get_scale()
    offset = qtzr.get_offset()
    print(f"  * scale: {scale} (shape: {tuple(scale.shape)})")
    print(f"  * offset: {offset} (shape: {tuple(offset.shape)})")

.. rst-class:: script-output

  .. code-block:: none

    QuantizeDequantize(shape=(1, 3, 1, 1), qmin=0, qmax=255, symmetric=False)
      * scale: tensor([[[[0.0039]],
                        [[0.0078]],
                        [[0.0312]]]], grad_fn=<DivBackward0>) (shape: (1, 3, 1, 1))
      * offset: tensor([[[[0.]],
                         [[0.]],
                         [[0.]]]], grad_fn=<SubBackward0>) (shape: (1, 3, 1, 1))


Note that:

* The shape :math:`(1, C, 1, 1)` of scale and offset is equal to that of the quantizer
* Every channel :math:`c \in [0, C)` of the quantized tensor is in the quantization grid of :math:`[0, 255]`, associated with :math:`scale_{:, c, :, :}` respectively

..
    (kyunggeu) Can't use this cool example yet because it's not implemented ;(

    .. code-block:: Python
        input_dq = qtzr(input)
        input_q = input_dq.quantize() # Integer representation of input_dq

        ch_0 = input_q[:, 0, :, :]
        ch_1 = input_q[:, 1, :, :]
        ch_2 = input_q[:, 2, :, :]

        print("input_q[:, 0, :, :]")
        print(f"  * scale: {ch_0.encoding.scale}")
        print(f"  * offset: {ch_0.encoding.offset}")

        print("input_q[:, 1, :, :]")
        print(f"  * scale: {ch_1.encoding.scale}")
        print(f"  * offset: {ch_1.encoding.offset}")

        print("input_q[:, 2, :, :]")
        print(f"  * scale: {   ch_2.encoding.scale}")
        print(f"  * offset: {  ch_2.encoding.offset}")


.. code-block:: Python

    input_dq = qtzr(input)
    input_q = input_dq.quantize() # Integer representation of input_dq

    print("Output (quantized representation):")
    print(input_q)
    print(f"  * scale: {input_q.encoding.scale}")
    print(f"  * offset: {input_q.encoding.offset}")

.. rst-class:: script-output

  .. code-block:: none

    Output (quantized representation):
    QuantizedTensor([[[[  0.,   1., ...], ..., [..., 254., 255.]],
                      [[  0.,   1., ...], ..., [..., 254., 255.]],
                      [[  0.,   1., ...], ..., [..., 254., 255.]]]],
                    grad_fn=<AliasBackward0>)
      * scale: tensor([[[[0.0039]],
                        [[0.0078]],
                        [[0.0312]]]], grad_fn=<DivBackward0>)
      * offset: tensor([[[[0.]],
                         [[0.]],
                         [[0.]]]], grad_fn=<SubBackward0>)


Blockwise Quantization
======================

Similar to how channelwise quantization was a mathematical generalization of tensor-wise quantization,
blockwise quantization is a even further generalization of channelwise quantization.

..
    TODO (kyunggeu): We need some visual diagram here

Blockwise quantization can be also easily done by creating the quantizer with the desired shape and block size.

.. code-block:: Python

    import torch
    import aimet_torch.v2.quantization as Q
    N, C, H, W = 1, 3, 32, 32
    input = torch.empty(N, C, H, W)
    
    B = 8 # block size
    block = (torch.arange(256) / 256).view(B, 32)
    input = torch.stack([
        block * 1,  block * 2,  block * 3,  block * 4,
        block * 5,  block * 6,  block * 7,  block * 8,
        block * 9,  block * 10, block * 11, block * 12,
    ]).view(N, C, H, W)
    
    # Blockwise quantization with block size B
    qtzr = Q.affine.QuantizeDequantize(shape=(1, C, 4, 1),
                                       block_size=(-1, 1, B, 32), # NOTE: -1 indicates wildcard block size
                                       bitwidth=8, symmetric=False)
    print(qtzr)
    
    with qtzr.compute_encodings():
        _ = qtzr(input)
    
    scale = qtzr.get_scale()
    offset = qtzr.get_offset()
    print(f"  * scale: {scale} (shape: {tuple(scale.shape)})")
    print(f"  * offset: {offset} (shape: {tuple(offset.shape)})")

.. rst-class:: script-output

  .. code-block:: none


    QuantizeDequantize(shape=(1, 3, 4, 1), block_size=(-1, 1, 8, 32), qmin=0, qmax=255, symmetric=False)
      * scale: tensor([[[[0.0039],
                         [0.0078],
                         [0.0117],
                         [0.0156]],
                        [[0.0195],
                         [0.0234],
                         [0.0273],
                         [0.0312]],
                        [[0.0352],
                         [0.0391],
                         [0.0430],
                         [0.0469]]]], grad_fn=<DivBackward0>) (shape: (1, 3, 4, 1))
      * offset: tensor([[[[0.],
                          [0.],
                          [0.],
                          [0.]],
                         [[0.],
                          [0.],
                          [0.],
                          [0.]],
                         [[0.],
                          [0.],
                          [0.],
                          [0.]]]], grad_fn=<SubBackward0>) (shape: (1, 3, 4, 1))

Note that:

* The shape :math:`(1, C, 4, 1)` of scale and offset is equal to that of the quantizer
* In runtime, scale and offset will be (theoretically but not literally) tiled by the factor of block size :math:`(-1, 1, B, 32)`
  to construct the tiled scale and offset of shape :math:`(-1, C \times 1, 4 \times B, 1 \times 32)`,
  which is equal to the shape of the input :math:`(-1, 3, 32, 32)`
* As a result, for every channel :math:`c \in [0, C)`, each block :math:`b \in [0, B)` is in the quantization grid of :math:`[0, 255]`, associated with :math:`scale_{:, c, b, :}` respectively

.. code-block:: Python

    input_dq = qtzr(input)
    input_q = input_dq.quantize() # Integer representation of input_dq
    print("Output (quantized representation)")
    print(input_q)
    print(f"  * scale: {input_q.encoding.scale}")
    print(f"  * offset: {offset} (shape: {tuple(offset.shape)})")

.. rst-class:: script-output

  .. code-block:: none

    QuantizedTensor([[[[  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.]],
                      [[  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.]],
                      [[  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.]],
                      [[  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.],
                       [  0.,   1., ...], ..., [..., 254., 255.]]]],
                    grad_fn=<AliasBackward0>)
      * scale: tensor([[[[0.0039],
                         [0.0078],
                         [0.0117],
                         [0.0156]],
                        [[0.0195],
                         [0.0234],
                         [0.0273],
                         [0.0312]],
                        [[0.0352],
                         [0.0391],
                         [0.0430],
                         [0.0469]]]], grad_fn=<DivBackward0>)
      * offset: tensor([[[[0.],
                          [0.],
                          [0.],
                          [0.]],
                         [[0.],
                          [0.],
                          [0.],
                          [0.]],
                         [[0.],
                          [0.],
                          [0.],
                          [0.]]]], grad_fn=<SubBackward0>)


API References
==============
* :ref:`api-torch-quantization-affine`
* :ref:`api-torch-quantized-tensor`
