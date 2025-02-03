.. include:: ../abbreviation.txt

.. _quantsim-blockwise:

######################
Per-block quantization
######################

This page describes blockwise techniques and tools for *calibration*, the process of determining quantization parameters. See :ref:`Calibration <quantsim-calibration>` for basic information about performing calibration.

.. note::
    The terms "per-block quantization" and "blockwise quantization" are used interchangeably.

When performing calibration for a tensor, you can compute encodings for the whole tensor, or split the tensor into parts (channels or blocks) and compute encodings for each part.

We recommended that you quantize as granularly as possible. Finer granularity typically results in better quantized accuracy. 

In order of preference, use:

1. Blockwise quantization (BQ)
2. Per-channel quantization
3. Per-tensor quantization

.. note::
  Blockwise and per-channel quantization are supported only for weights, not activations, on Qualcomm runtimes.


Executing per-block quantization
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

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

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

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Low power blockwise quantization
================================

|qnn| supports an alternative to blockwise quantization called Low Power Blockwise Quantization (LPBQ).

In LPBQ, blockwise encodings at a lower bit width are adjusted such that they lie on a common higher-bit-width per-channel grid. This enables models to run on existing per channel kernels and still benefit from blockwise quantization. 

LPBQ encodings require less storage than blockwise quantization encodings because only the low bit-width integer scale expansion factors need to be stored per-block (the floating point encoding scales are stored per-channel).

LPBQ quantization is part of the :class:`aimet_torch.quantization.affine.GroupedBlockQuantizeDequantize` class.

Besides the  `block_size` argument described in the  blockwise Quantization section, LPBQ requires two additional arguments:

`decompressed_bw`
  The higher bit-width value for the per channel grid that the lower bit-width blockwise encodings are expanded to. The `decompressed_bw` value must be greater than or equal to the  bit width of the quantizer.

`block_grouping`
  The number of blocks for each dimension that are grouped when expanding the lower bit-width blockwise encodings. The block grouping for a particular dimension must be divisible by the number of blocks in that dimension.

  As with block size, a block grouping value of -1 is automatically interpreted as the number of blocks in that dimension.

To allow for experimentation, the `GroupedBlockQuantizeDequantize` object supports arbitrary block sizes. However, the Qualcomm runtime imposes the following restrictions on LPBQ:

* Blockwise quantization runs on weight (not activation) quantizers only.
* Block size must be set to one for the output channel dimension.
* Block size may take an arbitrary value for the input channel dimension (it must still divide evenly into the input channel tensor shape).
*  Block size must be equal to the tensor size for all other dimensions.
* Block groupings must be set to one for all dimensions, except for the input channels dimension which should be
  set to the number of blocks in that dimension.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            from aimet_torch.quantization.affine import GroupedBlockQuantizeDequantize

            # Assume sim.model.conv_1 refers to a QuantizedConv2d layer with weight param shape of (16, 64, 2, 2)
            # Below settings equate to a block size of 16 in the input channels dimension.
            sim.model.conv_1.param_quantizers['weight'] = GroupedBlockQuantizeDequantize(shape=(16, 4, 1, 1),
                                                                                         bitwidth=4,
                                                                                         symmetric=True,
                                                                                         block_size=(1, 16, 2, 2),
                                                                                         decompressed_bw: 8,
                                                                                         block_grouping(1, 4, 1, 1))   # (1, -1, 1, 1) works too

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Exporting blockwise-quantized models
====================================

Blockwise quantization generates a larger number of encodings than per tensor or per channel quantization. To reduce the size of the exported encodings JSON file and the time needed to write the file, blockwise quantization uses an improved file format, designated 1.0.0, for the export.

The 1.0.0 encoding format is supported by the Qualcomm runtime and can be used to export per tensor, per channel, blockwise, and LPBQ quantizer encodings.

.. important::
    If  blockwise and/or LPBQ quantizers are present in the model, the 1.0.0 format *must* be used when exporting encodings for the Qualcomm runtime.

The following code snippet shows how to export encodings in the 1.0.0 format:

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            from aimet_common import quantsim

            # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.quantsim

            # Set encoding_version to 1.0.0
            quantsim.encoding_version = '1.0.0'
            sim.export('./data', 'exported_model', dummy_input)

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

See the :ref:`Encoding specifications <quantsim-encoding-spec>` page, which describes encodings specifications in detail.

API
===

**Top-level API to configure BQ quantization**

As described above, the Qualcomm runtime is constrained to running floating point activations for layers that use  blockwise quantization. We provide the following utility function to help transform multiple layers' quantizers to float quantization:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_activation_quantizers_to_float
    :noindex:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_blockwise_quantization_for_weights
    :noindex:

Note the second argument in the function, which specifies a subset of layers to switch to blockwise quantization. Refer to the function docstring for the valid input types for this argument.

The  `block_size` argument can be a single integer value instead of an array. In this case, the output channels dimension is set to a block size of one, the input channels dimension to the supplied value, and all other dimensions to the dimension's size. 

This enables you to handle layers with differing weight shapes (such as convolution layers with 4d weights vs. linear layers with 2d weights) with a single API call. If an array for  `block_size` is passed instead, the API has to be called multiple times for each set of layers with different weight dimensions (because the length of the  `block_size` array must match the number of dimensions for its layer's weight).

**Top-level API to configure LPBQ quantization**

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights
    :noindex:

This utility enables you to configure quantized layers to use grouped blockwise quantization by supplying a `decompressed_bw`,  `block_size`, and `block_grouping`. Similar to :func:`set_blockwise_quantization_for_weights`, `block_grouping` can be a single value. In this case the input_channel's dimension is assigned the value, and all other dimensions are assigned a value of one.

Different layers can have different numbers of blocks for the input channels dimension for the same block size. If you assign -1 as the single `block_grouping` value, the input channels dimension automatically uses a `block_grouping` value equal to the number of blocks in any affected layer. This enbles you to configure all affected layers to LPBQ quantization with a single API call.
