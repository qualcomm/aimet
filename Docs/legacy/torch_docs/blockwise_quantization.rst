.. _api-torch-blockwise-quantization:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future versions.

======================
Blockwise Quantization
======================

When performing integer quantization, it is necessary to determine quantization parameters (also known as encodings)
like scale and offset in order to define a quantization grid for mapping floating point values to their quantized integer
counterparts. This process of determining appropriate quantization parameters is known as calibration or computing encodings.

When performing calibration for a particular tensor, one can choose to come up with encodings to cover the whole tensor,
or to split the tensor into sections and compute encodings for each section. Below we describe several ways in which
tensors can be split, along with pros and cons of each:

    - **Per-Tensor quantization**: All values in the entire tensor are grouped collectively, and a single set of encodings
      are determined. Benefits include less computation and storage space needed to produce a single set of encodings.
      Drawbacks are that outlier values in the tensor negatively affect the encodings which are used to quantize all other
      values in the tensor.

    - **Per-Channel quantization**: Values in the tensor are split into individual channels (typically in the output channels
      dimension). The number of encodings computed for the tensor is equal to the number of channels. The benefit as
      compared to Per Tensor quantization are that outlier values would only influence encodings for the channel the
      outlier resides in, and would not affect encodings for values in other channels.

    - **Blockwise quantization**: Values in the tensor are split into chunks across multiple dimensions. This further
      improves the granularity at which encoding parameters are found, isolating outliers and producing a more optimized
      quantization grid for each block, at the cost of more storage used to hold an increased number of encodings.

In general, it is recommended to use Blockwise quantization in favor of Per-Channel quantization when possible, and similarly
Per-Channel quantization in favor of Per-Tensor quantization. The finer granularity provided by Blockwise quantization typically
leads to better quantized accuracy as a result. Note that Blockwise and Per-Channel quantization are supported only for
weights and not activations on Qualcomm runtime.

Blockwise quantization is supported as part of the :class:`aimet_torch.v2.quantization.affine.QuantizeDequantize` class.

Blockwise quantization can be enabled on an individual quantizer basis by instantiating a new QuantizeDequantize object with
the desired settings and replacing an existing quantizer with the new quantizer.
The block_size argument can be used to specify particular block sizes for each dimension of the tensor.
Note that there exists a relationship between the QuantizeDequantize's shape and block_size arguments, along with the shape
of the actual tensor being quantized.

The following rules must apply:

    - If block_size is provided, the length of block_size must match the length of the QuantizeDequantize's shape.

    - If block_size is provided, the length of block_size must be at most as long as the tensor to quantize's number of
      dimensions.

    - For block_size [b_1, b_2, ..., b_n] and QuantizeDequantize shape [s_1, s_2, ..., s_n], the tensor to quantize's shape
      must satisfy tensor.shape[:-n] == [b_1 * s_1, b_2 * s_2, ..., b_n * s_n]. In other words, block sizes for each
      dimension must evenly divide the size of the tensor in the corresponding dimension. For example, if a tensor's
      shape is (2, 2, 6, 10), a valid block_size would be (2, 1, 3, 5), since each block size is divisible by the tensor's
      corresponding dimension size.

    - For each dimension, a block size value of '-1' is permitted. In such cases, the block size is automatically determined
      based on the tensor's shape in that dimension and the QuantizeDequantize object's shape. This is essentially determining
      the block size for a dimension given the tensor's size along with the number of blocks for that dimension.

Below are examples of valid and invalid combinations of tensor shape, QuantizeDequantize shape, and block_size:

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

Note: While the QuantizeDequantize object supports arbitrary block sizes for experimental purposes, Qualcomm runtime restricts
Blockwise quantization to take place with the following constraints:

    - Blockwise quantization must run on weight quantizers only.

    - Block sizes must be set to 1 for the output channel dimension, may take arbitrary values for the input channel
      dimension (it must still be divisible by the input channel tensor shape), and must have block sizes equal to the
      tensor sizes for all other dimensions.

    - Layers with weights running with Blockwise quantization must themselves be running with floating-point quantized
      activations.

The below code examples show how to configure Convolution and Linear layers to Blockwise quantization:

.. code-block:: Python

    from aimet_torch.v2.quantization.affine import QuantizeDequantize

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

=======================================
Low Power Blockwise Quantization (LPBQ)
=======================================

Qualcomm runtime supports an alternative to Blockwise Quantization referred to as Low Power Blockwise Quantization (LPBQ).

In this scheme, blockwise encodings at a lower bitwidth are determined and then adjusted such that they lie on a common
higher bitwidth per channel grid. This allows models to achieve benefits of Blockwise quantization while allowing runtimes
to leverage existing per channel kernels in order to run the model. An additional benefit is that LPBQ encodings take less
storage space than Blockwise quantization encodings, due to the fact that floating point encoding scales are stored per
channel, and only low bitwidth integer scale expansion factors need to be stored in a per block fashion.

LPBQ quantization is supported as part of the :class:`aimet_torch.v2.quantization.affine.GroupedBlockQuantizeDequantize` class.

In addition to the block_size argument described in the Blockwise Quantization section, LPBQ introduces two new arguments:

    - **decompressed_bw**: The higher bitwidth value for the per channel grid which the lower bitwidth blockwise encodings will
      expand to. Decompressed_bw must be greater than or equal to the bitwidth of the quantizer.

    - **block_grouping**: The block_grouping argument defines the number of blocks for each dimension which will be grouped
      together when expanding the lower bitwidth blockwise encodings to the higher bitwidth per channel encodings.
      The block grouping for a particular dimension must be divisible by the number of blocks for that dimension.

As with block size, a block grouping value of '-1' is valid, and will correspond automatically to the number of blocks for
that dimension.

Note: While the GroupedBlockQuantizeDequantize quantizer supports arbitrary block groupings for experimental purposes,
Qualcomm runtime restricts LPBQ to take place with the following constraints:

    - Blockwise quantization must run on weight quantizers only.

    - Block sizes must be set to 1 for the output channel dimension, may take arbitrary values for the input channel
      dimension (it must still be divisible by the input channel tensor shape), and must have block sizes equal to the
      tensor sizes for all other dimensions.

    - Block groupings must be set to '1' for all dimensions, except for the input channels dimension, which should be
      set to the number of blocks for that dimension.

.. code-block:: Python

    from aimet_torch.v2.quantization.affine import GroupedBlockQuantizeDequantize

    # Assume sim.model.conv_1 refers to a QuantizedConv2d layer with weight param shape of (16, 64, 2, 2)
    # Below settings equate to a block size of 16 in the input channels dimension.
    sim.model.conv_1.param_quantizers['weight'] = GroupedBlockQuantizeDequantize(shape=(16, 4, 1, 1),
                                                                                 bitwidth=4,
                                                                                 symmetric=True,
                                                                                 block_size=(1, 16, 2, 2),
                                                                                 decompressed_bw: 8,
                                                                                 block_grouping(1, 4, 1, 1))   # (1, -1, 1, 1) works too

Top Level API
=============

Several top level API functions exist to make it easier to configure blockwise quantization and LPBQ quantization for
a model:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_blockwise_quantization_for_weights

This utility allows users to configure certain quantized layers in a model to use blockwise quantization with a specified
block_size.

Of significance is the second argument in the function, which allows users to specify a subset of layers to switch to
Blockwise quantization. Refer to the function docstring for valid types of inputs this argument supports.

For this API, the block_size argument can be a single integer value instead of an array. In this case, all
affected layers would be set to a block size of 1 for the output channels dimension, the specified value for the input
channels dimension, and block size equal to dimension size for all other dimensions.

Note that this allows layers with differing weight shapes (ex. Conv layers with 4d weights vs. Linear layers with 2d weights)
to be handled with a single API call. If an array for block_size is passed in instead, due to the requirement for the
length of the block_size array to match the number of dimensions for a particular layer's weight, the API would need to be
called multiple times for each set of layers with different weight dimensions.

As mentioned above, Qualcomm runtime is constrainted to running floating point activations for layers which use Blockwise
quantization. As a result, the following utility function is provided to assist in transforming multiple layers' quantizers
to float quantization:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_activation_quantizers_to_float

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights

This utility allows users to configure certain quantized layers in a model to use grouped blockwise quantization with a
specified decompressed_bw, block_size, and block_grouping. Similar to :func:`set_blockwise_quantization_for_weights`,
block_grouping can be a single value, in which case it will automatically be applied to the input_channel's dimension,
with all other dimensions using a block_grouping value of 1.

Additionally, as different layers may have a different number of blocks for the input channels dimension given the same
block size, a single block_grouping value of '-1' may be used, in which case the input channels dimension will automatically
use a block_grouping value equal to the number of blocks for any affected layer. This effectively allows users to configure
all affected layers to LPBQ quantization with a single API call.

Export
======

Using Blockwise quantization results in a larger number of encodings produced as compared to Per-Tensor or Per-Channel
quantization. As a result, a new method of exporting encodings to json has been developed to both reduce the exported
encodings file size as well as reduce the time needed to write exported encodings to the json file.

The following code snippet shows how to export encodings in the new 1.0.0 format:

.. code-block:: Python

    from aimet_common import quantsim

    # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim

    # Set encoding_version to 1.0.0
    quantsim.encoding_version = '1.0.0'
    sim.export('./data', 'exported_model', dummy_input)

The 1.0.0 encodings format is supported by Qualcomm runtime and can be used to export Per-Tensor, Per-Channel, Blockwise,
and LPBQ quantizer encodings. If Blockwise and/or LPBQ quantizers are present in the model, the 1.0.0 format must be
used when exporting encodings for Qualcomm runtime.
