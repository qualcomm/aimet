.. _apiref-torch-quantization:

########################
aimet_torch.quantization
########################

.. currentmodule:: aimet_torch.quantization

Quantizers
==========

AIMET quantizers are the low-level components of :ref:`quantized modules<api-torch-quantized-modules>` that
implement the quantization mechanism for PyTorch tensors.

AIMET quantizers are PyTorch modules that take a torch.Tensor as input and return
a :class:`QuantizedTensor<aimet_torch.v2.quantization.tensor.QuantizedTensor>`
or :class:`DequantizedTensors<aimet_torch.v2.quantization.tensor.DequantizedTensor>`,
a subclass of regular torch.Tensor with some additional attributes and helper functions for quantization.
All quantizers are derived from the base class :class:`QuantizerBase` defined as below.

.. autoclass:: aimet_torch.quantization.base.quantizer.QuantizerBase
    :members: forward, compute_encodings, is_initialized

Affine quantizers
-----------------

Even though it is **strongly recommended** for most users to delegate the instantiation and configuration of quantizers to :class:`QuantizationSimModel`,
it is worth understanding the underlying mechanism of quantizers for finer control over the quantized model.

The most commonly used quantizers are the affine quantizers such as :class:`QuantizeDequantize`.
Here is a quick example of how to create an 8-bit asymmetric affine quantizer.

.. code-block:: Python

    import aimet_torch.quantization as Q
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
    input_qdq = qtzr(input)

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
    print(input_qdq)
    print(f"  * scale: {input_qdq.encoding.scale}")
    print(f"  * offset: {input_qdq.encoding.offset}")
    print(f"  * bitwidth: {input_qdq.encoding.bitwidth}")
    print(f"  * signed: {input_qdq.encoding.signed}")
    print()

    input_q = input_qdq.quantize() # Integer representation of input_qdq
    print("Output (quantized representation):")
    print(input_q)
    print(f"  * scale: {input_q.encoding.scale}")
    print(f"  * offset: {input_q.encoding.offset}")
    print(f"  * bitwidth: {input_q.encoding.bitwidth}")
    print(f"  * signed: {input_q.encoding.signed}")

    # Sanity checks
    # 1. Quantizing and dequantizing input_qdq shouldn't change the result
    assert torch.equal(input_qdq, input_q.dequantize())
    # 2. (De-)Quantizing an already (de-)quantized tensor shouldn't change the result
    assert torch.equal(input_qdq, input_qdq.dequantize())
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

Per-channel quantization
------------------------

*Per-channel quantization* is one of the advanced usages of affine quantizers where
one scale and offset will be associated with only one channel of the input tensor,
whereas one scale and offset was associated with the entire tensor in the previous example.

..
    TODO (kyunggeu): We need some visual diagram here

Per-channel quantization can be easily done by creating the quantizer with the desired shape of scale and offset.

.. code-block:: Python

    import torch
    import aimet_torch.quantization as Q
    Cout, Cin = 8, 8

    weight = (torch.arange(-32, 32) / 64).view(Cin, Cout).transpose(0, 1)

    # Per-channel quantization along the output channel axis (Cout) of the weight
    qtzr = Q.affine.QuantizeDequantize(shape=(Cout, 1), bitwidth=8, symmetric=True)
    print(f"Quantizer:\n{qtzr}")

    with qtzr.compute_encodings():
        _ = qtzr(weight)

    scale = qtzr.get_scale()
    offset = qtzr.get_offset()
    print(f"\nScale:\n{scale} (shape: {tuple(scale.shape)})")
    print(f"\nOffset:\n{offset} (shape: {tuple(offset.shape)})")

.. rst-class:: script-output

  .. code-block:: none

    Quantizer:
    QuantizeDequantize(shape=(8, 1), qmin=-128, qmax=127, symmetric=True)

    Scale:
    tensor([[0.0039],
            [0.0038],
            [0.0037],
            [0.0035],
            [0.0034],
            [0.0036],
            [0.0037],
            [0.0038]], grad_fn=<DivBackward0>) (shape: (8, 1))

    Offset:
    tensor([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]]) (shape: (8, 1))


Note that:

* The shape :math:`(C_{out}, 1)` of scale and offset is equal to that of the quantizer
* Every channel :math:`c \in [0, C_{out})` of the quantized tensor is in the quantization grid of :math:`[-128, 127]`, associated with :math:`scale_{c, :}` respectively

..
    (kyunggeu) Can't use this cool example yet because it's not implemented ;(

    .. code-block:: Python
        input_qdq = qtzr(input)
        input_q = input_qdq.quantize() # Integer representation of input_qdq

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

    weight_qdq = qtzr(weight)
    weight_q = weight_qdq.quantize() # Integer representation of weight_qdq

    print("Output (quantized representation):\n{weight_q}")
    print(f"\nScale:\n{weight_q.encoding.scale}")
    print(f"\nOffset:\n{weight_q.encoding.offset}")

.. rst-class:: script-output

  .. code-block:: none

    Output (quantized representation):
    QuantizedTensor([[-128.,  -96.,  -64.,  -32.,    0.,   32.,   64.,   96.],
                     [-128.,  -95.,  -62.,  -29.,    4.,   37.,   70.,  103.],
                     [-128.,  -94.,  -60.,  -26.,    9.,   43.,   77.,  111.],
                     [-128.,  -93.,  -57.,  -22.,   13.,   49.,   84.,  119.],
                     [-127.,  -91.,  -54.,  -18.,   18.,   54.,   91.,  127.],
                     [-118.,  -83.,  -48.,  -13.,   22.,   57.,   92.,  127.],
                     [-110.,  -76.,  -42.,   -8.,   25.,   59.,   93.,  127.],
                     [-102.,  -70.,  -37.,   -4.,   29.,   61.,   94.,  127.]],
                    grad_fn=<AliasBackward0>)

    Scale:
    tensor([[0.0039],
            [0.0038],
            [0.0037],
            [0.0035],
            [0.0034],
            [0.0036],
            [0.0037],
            [0.0038]], grad_fn=<DivBackward0>)

    Offset:
    tensor([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]])


Per-block quantization
----------------------

*Per-block quantization* (also called *blockwise quantization*) is a further mathematical generalization of per-channel quantization,
similar to how per-channel quantization is a mathematical generalization of per-tensor quantization.

..
    TODO (kyunggeu): We need some visual diagram here

Blockwise quantization can be also easily done by creating a quantizer with the desired shape and block size.

.. code-block:: Python

    import torch
    import aimet_torch.quantization as Q
    Cout, Cin = 8, 8
    B = 4 # block size

    weight = torch.cat([
        (torch.arange(-16, 16) / 32).view(B, Cout).transpose(0, 1),
        (torch.arange(-16, 16) / 16).view(B, Cout).transpose(0, 1),
    ], dim=1)

    # Blockwise quantization with block size B
    qtzr = Q.affine.QuantizeDequantize(shape=(Cout, Cin // B),
                                       block_size=(-1, B), # NOTE: -1 indicates wildcard block size
                                       bitwidth=8, symmetric=True)
    print(f"Quantizer:\n{qtzr}")

    with qtzr.compute_encodings():
        _ = qtzr(weight)

    scale = qtzr.get_scale()
    offset = qtzr.get_offset()
    print(f"\nScale:\n{scale} (shape: {tuple(scale.shape)})")
    print(f"\nOffset:\n{offset} (shape: {tuple(offset.shape)})")

.. rst-class:: script-output

  .. code-block:: none

    Quantizer:
    QuantizeDequantize(shape=(8, 2), block_size=(-1, 4), qmin=-128, qmax=127, symmetric=True)

    Scale:
    tensor([[0.0039, 0.0078],
            [0.0037, 0.0073],
            [0.0034, 0.0068],
            [0.0032, 0.0063],
            [0.0030, 0.0059],
            [0.0032, 0.0064],
            [0.0034, 0.0069],
            [0.0037, 0.0074]], grad_fn=<DivBackward0>) (shape: (8, 2))

    Offset:
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]) (shape: (8, 2))

Note that:

* The shape :math:`\left(C_{out}, \frac{C_{in}}{B}\right) = (8, 2)` of scale and offset is equal to that of the quantizer
* For every channel :math:`c \in [0, C_{out})`, each block :math:`k \in \left[0, \frac{C_{in}}{B}\right)` is in the quantization grid of :math:`[-128, 127]`, associated with :math:`scale_{c, k:k+B}` respectively
* If :math:`C_{in}` is not divisible by block size :math:`B`, the quantizer will throw shape mismatch error in runtime.

.. code-block:: Python

    weight_qdq = qtzr(weight)
    weight_q = weight_qdq.quantize() # Integer representation of weight_qdq

    print("Output (quantized representation):\n{weight_q}")
    print(f"\nScale:\n{weight_q.encoding.scale}")
    print(f"\nOffset:\n{weight_q.encoding.offset}")

.. rst-class:: script-output

  .. code-block:: none

    Output (quantized representation):
    QuantizedTensor([[-128.,  -64.,    0.,   64., -128.,  -64.,    0.,   64.],
                     [-128.,  -60.,    9.,   77., -128.,  -60.,    9.,   77.],
                     [-128.,  -55.,   18.,   91., -128.,  -55.,   18.,   91.],
                     [-128.,  -49.,   30.,  108., -128.,  -49.,   30.,  108.],
                     [-127.,  -42.,   42.,  127., -127.,  -42.,   42.,  127.],
                     [-107.,  -29.,   49.,  127., -107.,  -29.,   49.,  127.],
                     [ -91.,  -18.,   54.,  127.,  -91.,  -18.,   54.,  127.],
                     [ -76.,   -8.,   59.,  127.,  -76.,   -8.,   59.,  127.]],
                    grad_fn=<AliasBackward0>)

    Scale:
    tensor([[0.0039, 0.0078],
            [0.0037, 0.0073],
            [0.0034, 0.0068],
            [0.0032, 0.0063],
            [0.0030, 0.0059],
            [0.0032, 0.0064],
            [0.0034, 0.0069],
            [0.0037, 0.0074]], grad_fn=<DivBackward0>)

    Offset:
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]])

API reference
=============

**Quantized tensors**

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    QuantizedTensorBase
    QuantizedTensor
    DequantizedTensor

.. _api-beta-quantizers:

**Quantizers**

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    affine.Quantize
    affine.QuantizeDequantize
    float.FloatQuantizeDequantize

**Functional APIs**

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: function.rst

    affine.quantize
    affine.quantize_dequantize
    affine.dequantize
