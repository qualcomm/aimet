.. include:: ../abbreviation.txt

.. _techniques-blockwise:

######################
Blockwise Quantization
######################

.. note::
   This section explains the general concept and usage of blockwise quantization.
   However, generic blockwise quantization is not yet supported by Qualcomm hardware.
   For a practical alternative of blockwise quantization supported by Qualcomm hardware, see :ref:`Low-Power Blockwise Quantization (LPBQ) <techniques-lpbq>`

*Blockwise quantization*, also known as *per-block quantization*, is a quantization scheme that divides the input into multiple small *blocks*, with each block sharing the same quantization scale.
Due to its finer granularity, blockwise quantization offers improved model accuracy compared to per-tensor or per-channel quantization.

.. image:: ../images/BQ.png


Executing blockwise quantization
================================

To enable blockwise quantization, instantiate a `QuantizeDequantize` object with blockwise settings and replace an existing quantizer with the new quantizer.

Specify the block sizes for each dimension of the tensor in the  `block_size` parameter. Note the relationship between   `block_size` arguments and the QuantizeDequantize object's shape, and with the shape of the tensor being quantized.

The following rules apply:

* If  `block_size` is provided, the length of  `block_size` must match the length of the `QuantizeDequantize` object's shape.

* If  `block_size` is provided, it must be no longer than the number of dimensions of the tensor.

* Block sizes must evenly divide each of the tensor's dimensions. For example, if a tensor's shape is (2, 2, 6, 10), then (2, 1, 3, 5) is a valid  `block_size`, since each tensor dimension is divisible by the corresponding block size. 

  In formal terms, for  `block_size` [b\ :sub:`1`\, b\ :sub:`2`\,, ..., b\ :sub:`n`\,] and `QuantizeDequantize` shape [s\ :sub:`1`\, s\ :sub:`2`\,, ..., s\ :sub:`n`\,], the tensor's shape must satisfy this relationship:

  .. math::
    tensor.shape\left[:-n\right] == \left[b_1 * s_1, b_2 * s_2, ..., b_n * s_n\right]
  
* For any dimension, you can use a block size value of -1 to instruct the quantizer to automatically determine the block size based on shape of the `QuantizeDequantize` object and the tensor in that dimension.

Following are examples of valid and invalid combinations of tensor shape, `QuantizeDequantize` shape, and  `block_size`.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            # Invalid combination: block_size is not the same length as QuantizeDequantize shape
            tensor shape: (1, 4, 10)
            QuantizeDequantize shape: (1,)
            block_size: (1, 4, 10)

            # Invalid combination: block_size * QuantizeDequantize shape != tensor shape:
            tensor shape: (1, 4, 10)
            QuantizeDequantize shape: (1, 2, 10)
            block_size: (1, 2, 5)

            # Valid combination:
            tensor shape: (16, 64, 3, 3)
            QuantizeDequantize shape: (16, 4, 1, 1)
            block_size: (1, 16, 3, 3)

            # Valid combination (note that though tensor shape is 3d, only the final 2 dimensions correspond to block_size
            # and QuantizeDequantize shape):
            tensor shape: (2, 4, 10)
            QuantizeDequantize shape: (2, 2)
            block_size: (2, 5)

            # Valid combination:
            tensor shape: (2, 4, 10)
            QuantizeDequantize shape: (2, 2)
            block_size: (-1, -1)    # block_size will be inferred to be (2, 5)
    .. tab-item:: ONNX
        :sync: onnx

        Not supported


To allow for experimentation, the `QuantizeDequantize` object supports arbitrary block sizes. However, the Qualcomm runtime imposes the following restrictions:

* Blockwise quantization runs on weight (not activation) quantizers only.
* Block size must be set to one for the output channel dimension.
* Block size may take an arbitrary value for the input channel dimension (it must still divide evenly into the input channel tensor shape).
*  Block size must be equal to the tensor size for all other dimensions.
* Layers running with blockwise-quantized weights must be running with quantized floating-point activations.

The following code examples show how to configure convolution and linear layers to  blockwise quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            from aimet_torch.quantization.affine import QuantizeDequantize

            # Assume sim.model.conv_1 refers to a QuantizedConv2d layer with weight param shape of (16, 64, 2, 2)
            # Below settings equate to a block size of 16 in the input channels dimension.
            sim.model.conv_1.param_quantizers['weight'] = QuantizeDequantize(shape=(16, 4, 1, 1),
                                                                             bitwidth=4,
                                                                             symmetric=True,
                                                                             block_size=(1, 16, 2, 2))  # (-1, -1, -1, -1) works too

            # Assume sim.model.linear_1 refers to a QuantizedLinear layer with weight param shape of (12, 16)
            # Below settings equate to a block size of 4 in the input channels dimension.
            sim.model.conv_1.param_quantizers['weight'] = QuantizeDequantize(shape=(12, 4),
                                                                             bitwidth=4,
                                                                             symmetric=True,
                                                                             block_size=(1, 4))  # (-1, -1) works too
    .. tab-item:: ONNX
        :sync: onnx

        .. note::
            Not supported in ``aimet-onnx``.

API
===

As described above, the Qualcomm runtime is constrained to running floating point activations for layers that use  blockwise quantization. We provide the following utility function to help transform multiple layers' quantizers to float quantization:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_activation_quantizers_to_float
    :noindex:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_blockwise_quantization_for_weights
    :noindex:

Note the second argument in the function, which specifies a subset of layers to switch to blockwise quantization. Refer to the function docstring for the valid input types for this argument.

The  `block_size` argument can be a single integer value instead of an array. In this case, the output channels dimension is set to a block size of one, the input channels dimension to the supplied value, and all other dimensions to the dimension's size. 

This enables you to handle layers with differing weight shapes (such as convolution layers with 4d weights vs. linear layers with 2d weights) with a single API call. If an array for  `block_size` is passed instead, the API has to be called multiple times for each set of layers with different weight dimensions (because the length of the  `block_size` array must match the number of dimensions for its layer's weight).
