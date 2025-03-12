# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Commonly used utilities for configuring quantsim """

from typing import overload, Callable, List, Optional, Tuple, Type, Union
import torch
from aimet_common.utils import AimetLogger
from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine import QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.utils import get_device

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

modules_with_in_out_channels = (torch.nn.Linear,
                                torch.nn.Conv1d,
                                torch.nn.Conv2d,
                                torch.nn.Conv3d,
                                torch.nn.ConvTranspose1d,
                                torch.nn.ConvTranspose2d,
                                torch.nn.ConvTranspose3d)

def _get_quantizer_device_or_default(quantizer: torch.nn.Module, default_device: Union[str, torch.device]):
    """ Get device from quantizer if possible, else return default_device """
    try:
        device = get_device(quantizer)
    except StopIteration:
        device = default_device
    return device

def _get_in_channels_dim(module: torch.nn.Module):
    """ Get input channels dimension for the given module """
    if not isinstance(module, modules_with_in_out_channels):
        raise AssertionError(f'In channels not defined for module of type {type(module)}')
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
        return 0
    return 1

def _get_out_channels_dim(module: torch.nn.Module):
    """ Get input channels dimension for the given module """
    if not isinstance(module, modules_with_in_out_channels):
        raise AssertionError(f'Out channels not defined for module of type {type(module)}')
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
        return 1
    return 0

def _get_block_size_array_for_module(module: torch.nn.Module, block_size: Union[int, Tuple[int, ...]]):
    """
    Return block size array for a module given a single block size value, assuming that the block size value is to be
    applied to the in_channels dimension, and the out_channels dimension is per channel.
    """
    if isinstance(block_size, (tuple, list)):
        return block_size

    assert isinstance(block_size, int)
    if not isinstance(module, modules_with_in_out_channels):
        error_msg = f'Single value block size is only supported for modules of types in ' \
                    f'config_utils.modules_with_in_out_channels, but got module of type {type(module)}. Update ' \
                    f'the argument for identifying modules to set to exclude unsupported types.'
        raise RuntimeError(error_msg)
    assert hasattr(module, 'weight')

    # Initialize block sizes with -1, meaning blocks will span the entire dimension for each axis
    block_size_array = [-1] * len(module.weight.shape)

    # Set in channels dimension block size to block_size, and set out channels dimension block size to 1 (per channel)
    block_size_array[_get_in_channels_dim(module)] = block_size
    block_size_array[_get_out_channels_dim(module)] = 1
    return block_size_array


def _get_block_grouping_array_for_module(module: torch.nn.Module, block_grouping: Union[int, Tuple[int, ...]]):
    """
    Return block grouping array for a module given a single block grouping value, assuming that the block grouping value
    is to be applied to the in_channels dimension.
    """
    if isinstance(block_grouping, (tuple, list)):
        return block_grouping

    assert isinstance(block_grouping, int)
    if not isinstance(module, modules_with_in_out_channels):
        error_msg = f'Single value block grouping is only supported for modules of types in ' \
                    f'config_utils.modules_with_in_out_channels, but got module of type {type(module)}. Update ' \
                    f'the argument for identifying modules to set to exclude unsupported types.'
        raise RuntimeError(error_msg)
    assert hasattr(module, 'weight')

    # Initialize block grouping with 1, meaning no blocks will be grouped together in each dimension
    block_grouping_array = [1] * len(module.weight.shape)

    # Set in channels dimension block size to block_grouping
    block_grouping_array[_get_in_channels_dim(module)] = block_grouping
    return block_grouping_array


def _parse_arg_for_condition(arg):
    """ Transform the given arg into a corresponding condition expression """
    if isinstance(arg, List) and isinstance(arg[0], type) and issubclass(arg[0], torch.nn.Module):
        module_type = arg
        condition = lambda module: isinstance(module, tuple(module_type))
    elif isinstance(arg, List):
        if not isinstance(arg[0], torch.nn.Module):
            raise RuntimeError('List given as arg must contain either torch.nn.Module types or specific '
                               'torch.nn.Module objects.')
        qmodules = arg
        condition = lambda module: module in qmodules
    else:
        condition = arg
    return condition


def _get_weight_quantizer_shape_from_block_size(quant_layer: torch.nn.Module, block_size: Tuple[int, ...]) \
        -> Optional[List[int]]:
    """ Given a block size, get the corresponding weight quantizer shape """
    weight_shape = quant_layer.weight.shape
    block_size = _get_block_size_array_for_module(quant_layer, block_size)
    assert len(block_size) == len(weight_shape)
    quantizer_shape = []
    for idx, shape in enumerate(weight_shape):
        if block_size[idx] != -1:
            if not shape % block_size[idx] == 0:
                return None

            quantizer_shape.append(shape // block_size[idx])
        else:
            quantizer_shape.append(1)
    return quantizer_shape


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                       exponent_bits: int = None, mantissa_bits: int = None,
                                       dtype: torch.dtype = None):
    """ Set activation quantizers of the given module type to float """


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, qmodules: List[torch.nn.Module],
                                       exponent_bits: int = None, mantissa_bits: int = None, dtype: torch.dtype = None):
    """ Set activation quantizers of the given qmodules to float """


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                       exponent_bits: int = None, mantissa_bits: int = None, dtype: torch.dtype = None):
    """ Set activation quantizers of all the modules that satisfy the given condition to float. """


def set_activation_quantizers_to_float(sim: QuantizationSimModel, arg, exponent_bits: int = None,
                                       mantissa_bits: int = None, dtype: torch.dtype = None):
    """
    Set activation quantizers of modules to float.

    :param sim: Quantsim to set activation quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:

        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set

        2. A list of torch.nn.Modules, in which case all modules in the list will be set

        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param exponent_bits: Number of exponent bits to simulate
    :param mantissa_bits: Number of mantissa bits to simulate
    :param dtype: torch.dtype to simulate. This argument is mutually exclusive with exponent_bits and mantissa_bits.

    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim
        >>> # Allows setting of all Linear and Conv output quantizers to floating point activation quantization:
        >>> set_activation_quantizers_to_float(sim=sim,
        ...                                    arg=[torch.nn.Linear, torch.nn.Conv2d],
        ...                                    dtype=torch.float16)
        >>> # Allows setting of specific model layers' output quantizers to floating point activation quantization:
        >>> set_activation_quantizers_to_float(sim=sim,
        ...                                    arg=[sim.model.conv2, sim.model.linear1],
        ...                                    dtype=torch.float16)
        >>> # Allows setting of only Convolution layers with input channels dim == 128 to floating point activation quantization:
        >>> set_activation_quantizers_to_float(sim=sim,
        ...                                    arg=lambda module: isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] == 128,
        ...                                    dtype=torch.float16)
    """

    condition = _parse_arg_for_condition(arg)
    _set_activation_quantizers_to_float(sim, condition, exponent_bits, mantissa_bits, dtype)


def _set_activation_quantizers_to_float(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                        exponent_bits: int = None, mantissa_bits: int = None,
                                        dtype: torch.dtype = None):
    """ Set activation quantizers of all the modules that satisfy the given condition to float. """
    model_device = get_device(sim.model)
    for _, quant_layer in sim.named_qmodules():
        if condition(quant_layer):
            for idx, quantizer in enumerate(quant_layer.input_quantizers):
                if quantizer is not None:
                    device = _get_quantizer_device_or_default(quantizer, model_device)
                    quant_layer.input_quantizers[idx] = FloatQuantizeDequantize(exponent_bits, mantissa_bits, dtype).to(device)

            for idx, quantizer in enumerate(quant_layer.output_quantizers):
                if quantizer is not None:
                    device = _get_quantizer_device_or_default(quantizer, model_device)
                    quant_layer.output_quantizers[idx] = FloatQuantizeDequantize(exponent_bits, mantissa_bits, dtype).to(device)


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                           bitwidth: int, symmetric: bool, block_size: Union[int, Tuple[int, ...]]):
    """ Set weight parameter quantizers of the given module type to blockwise """


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, qmodules: List[torch.nn.Module], bitwidth: int,
                                           symmetric: bool, block_size: Union[int, Tuple[int, ...]]):
    """ Set weight parameter quantizers of the given modules to blockwise """


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                           bitwidth: int, symmetric: bool, block_size: Union[int, Tuple[int, ...]]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to blockwise """


def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, arg, bitwidth: int, symmetric: bool,
                                           block_size: Union[int, Tuple[int, ...]]):
    """
    Set weight parameter quantizers of modules to blockwise.

    :param sim: Quantsim to set weight quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:

        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set

        2. A list of torch.nn.Modules, in which case all modules in the list will be set

        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param bitwidth: Bitwidth for affine quantization
    :param symmetric: True if affine quantization is symmetric, False otherwise
    :param block_size: Block size for affine quantization. This can be an array in which case all layers identified
        by arg must have weight shapes compatible with the array length, or can be an integer value, in which case the
        block size will be applied to the weight's in_channels dimension, and per channel will be used for the weight's
        out_channels dimension.

        A block size value of -1 for a particular dimension is equivalent to a block size equal
        to the size of that particular dimension.

    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim
        >>> # Allows setting of all Linear and Conv weight quantizers to block_size 64 in the input_channels dimension:
        >>> set_blockwise_quantization_for_weights(sim=sim,
        ...                                        arg=[torch.nn.Linear, torch.nn.Conv2d],
        ...                                        bitwidth=4,
        ...                                        symmetric=True,
        ...                                        block_size=64)
        >>> # Allows setting of specific model layers' weight quantizer block_size to 64 in the input_channels dimension:
        >>> set_blockwise_quantization_for_weights(sim=sim,
        ...                                        arg=[sim.model.conv2, sim.model.linear1],
        ...                                        bitwidth=4,
        ...                                        symmetric=True,
        ...                                        block_size=64)
        >>> # Allows setting of only Convolution layers with input channels dim == 128 to block_size 64 in the input_channels dimension
        >>> set_blockwise_quantization_for_weights(sim=sim,
        ...                                        arg=lambda module: isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] == 128,
        ...                                        bitwidth=4,
        ...                                        symmetric=True,
        ...                                        block_size=64)
    """
    condition = _parse_arg_for_condition(arg)
    _set_blockwise_quantization_for_weights(sim, condition, bitwidth, symmetric, block_size)


def _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size):
    layer_to_quantizer_shape_dict = {}
    invalid_layers_for_block_size = []
    for name, quant_layer in sim.named_qmodules():
        if condition(quant_layer) and 'weight' in quant_layer.param_quantizers and \
                quant_layer.param_quantizers['weight'] is not None:
            assert hasattr(quant_layer, 'weight')
            layer_to_quantizer_shape_dict[quant_layer] = \
                _get_weight_quantizer_shape_from_block_size(quant_layer, block_size)
            if layer_to_quantizer_shape_dict[quant_layer] is None:
                invalid_layers_for_block_size.append((name, quant_layer.weight.shape))

    if invalid_layers_for_block_size:
        for name, shape in invalid_layers_for_block_size:
            error_str = f"Quant layer {name} has shape {shape} which does not align with block_size {block_size}. " \
                        f"Each dimension's shape must divide evenly with the corresponding block size."
            raise RuntimeError(error_str)

    return layer_to_quantizer_shape_dict


def _set_blockwise_quantization_for_weights(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                            bitwidth: int, symmetric: bool, block_size: Union[int, Tuple[int, ...]]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to blockwise """
    model_device = get_device(sim.model)
    layer_to_quantizer_shape_dict = _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size)
    for layer, quantizer_shape in layer_to_quantizer_shape_dict.items():
        layer_block_size = _get_block_size_array_for_module(layer, block_size)
        device = _get_quantizer_device_or_default(layer.param_quantizers['weight'], model_device)
        layer.param_quantizers['weight'] = QuantizeDequantize(quantizer_shape, bitwidth, symmetric, None,
                                                              layer_block_size).to(device)


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Union[int, Tuple[int, ...]],
                                                   block_grouping: Union[int, Tuple[int, ...]] = -1):
    """ Set weight parameter quantizers of the given module type to grouped blockwise """


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, qmodules: List[torch.nn.Module],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Union[int, Tuple[int, ...]],
                                                   block_grouping: Union[int, Tuple[int, ...]] = -1):
    """ Set weight parameter quantizers of the given modules to grouped blockwise """


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel,
                                                   condition: Callable[[torch.nn.Module], bool],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Union[int, Tuple[int, ...]],
                                                   block_grouping: Union[int, Tuple[int, ...]] = -1):
    """ Set weight parameter quantizers of modules that satisfy the given condition to grouped blockwise """


def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, arg,
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Union[int, Tuple[int, ...]],
                                                   block_grouping: Union[int, Tuple[int, ...]] = -1):
    """
    Set weight parameter quantizers of modules to grouped blockwise.

    :param sim: Quantsim to set weight quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:

        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set

        2. A list of torch.nn.Modules, in which case all modules in the list will be set

        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param bitwidth: Bitwidth for affine quantization
    :param symmetric: True if affine quantization is symmetric, False otherwise
    :param decompressed_bw: Decompressed bw for grouped block quantization
    :param block_size: Block size for affine quantization. This can be an array in which case all layers identified
        by arg must have weight shapes compatible with the array length, or can be an integer value, in which case the
        block size will be applied to the weight's in_channels dimension and per channel will be used for the weight's
        out_channels dimension.

        A block size value of -1 for a particular dimension is equivalent to a block size equal
        to the size of that particular dimension.
    :param block_grouping: Block grouping for grouped block quantization. This can be an array in which case all layers
        identified by arg must have weight shapes compatible with the array length, or can be an integer value, in which
        case the block grouping will be applied to the weight's in_channels dimension, and no other dimensions will
        experience block grouping.

        A block grouping value of -1 for a particular dimension is equivalent to a block
        grouping equal to the number of blocks for that particular dimension.

    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim
        >>> # Allows setting of all Linear and Conv weight quantizers to LPBQ with block_size 64 in the input_channels dimension:
        >>> set_grouped_blockwise_quantization_for_weights(sim=sim,
        ...                                                arg=[torch.nn.Linear, torch.nn.Conv2d],
        ...                                                bitwidth=4,
        ...                                                symmetric=True,
        ...                                                decompressed_bw=8,
        ...                                                block_size=64,
        ...                                                block_grouping=-1)
        >>> # Allows setting of specific model layers' weight quantizer to LPBQ with block_size 64 in the input_channels dimension:
        >>> set_grouped_blockwise_quantization_for_weights(sim=sim,
        ...                                                arg=[sim.model.conv2, sim.model.linear1],
        ...                                                bitwidth=4,
        ...                                                symmetric=True,
        ...                                                decompressed_bw=8,
        ...                                                block_size=64,
        ...                                                block_grouping=-1)
        >>> # Allows setting of only Convolution layers with input channels dim == 128 to LPBQ with block_size 64 in the input_channels dimension:
        >>> set_grouped_blockwise_quantization_for_weights(sim=sim,
        ...                                                arg=lambda module: isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] == 128,
        ...                                                bitwidth=4,
        ...                                                symmetric=True,
        ...                                                decompressed_bw=8,
        ...                                                block_size=64,
        ...                                                block_grouping=-1)
    """
    condition = _parse_arg_for_condition(arg)
    _set_grouped_blockwise_quantization_for_weights(sim, condition, bitwidth, symmetric, decompressed_bw, block_size,
                                                    block_grouping)


def _set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel,
                                                    condition: Callable[[torch.nn.Module], bool],
                                                    bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                    block_size: Union[int, Tuple[int, ...]],
                                                    block_grouping: Union[int, Tuple[int, ...]]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to grouped blockwise """
    model_device = get_device(sim.model)
    layer_to_quantizer_shape_dict = _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size)
    for layer, quantizer_shape in layer_to_quantizer_shape_dict.items():
        layer_block_size = _get_block_size_array_for_module(layer, block_size)
        layer_block_grouping = _get_block_grouping_array_for_module(layer, block_grouping)
        device = _get_quantizer_device_or_default(layer.param_quantizers['weight'], model_device)
        layer.param_quantizers['weight'] = GroupedBlockQuantizeDequantize(quantizer_shape, bitwidth, symmetric,
                                                                          decompressed_bw, None, layer_block_size,
                                                                          layer_block_grouping).to(device)
