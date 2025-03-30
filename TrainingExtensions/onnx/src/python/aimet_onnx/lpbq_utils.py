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
""" Utility functions for applying LPBQ quantization """

from typing import List, Tuple, Sequence
import numpy as np

from aimet_common.quantsim import compute_min_max_given_delta_offset
from aimet_common import libpymo


def _split_blocks(encoding: np.ndarray, block_grouping) -> np.ndarray:
    """
    Get expanded scale shape which breaks each scale dimension into a pair of dimensions with sizes
    (original_shape / block_grouping, block_grouping).

    :return: Expanded scale shape
    """
    expanded_shape = []
    for idx, block_group in enumerate(block_grouping):
        # Block group of -1 is equivalent to grouping all blocks together
        if block_group == -1:
            expanded_shape.append(1)
            expanded_shape.append(encoding.shape[idx])
        else:
            expanded_shape.append(encoding.shape[idx] // block_group)
            expanded_shape.append(block_group)
    return encoding.reshape(expanded_shape)

def _get_per_group_scale_factor(scale: np.ndarray,
                               block_grouping: Sequence[int],
                               scale_bitwidth: int) -> np.ndarray:
    """
    Get per channel scale.

    :param scale: Scale array
    :param block_grouping: Number of indices to group together for each dimension of scale
    :return: Per-group scale factor
    """
    grouped_scale = _split_blocks(scale, block_grouping)
    group_axes = tuple(range(1, len(grouped_scale.shape), 2))
    max_scale = np.max(grouped_scale, axis=group_axes, keepdims=True)
    per_group_scale = max_scale / 2 ** scale_bitwidth
    return per_group_scale

def grouped_dynamic_quantize(input_array: np.ndarray,
                             grouping,
                             bitwidth) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize input array between (1, 2 ** bitwidth) based on the maximum value in each group.

    :input_array: numpy array to quantize
    :param grouping: Number of indices per group for each axis of the input_array
    :bitwidth: Quantization bitwidth
    :return: Tuple of quantized input and quantization scale
    """
    dynamic_scale = _get_per_group_scale_factor(input_array, grouping, bitwidth)
    grouped_scale = _split_blocks(input_array, grouping)
    # Note: following aimet_torch implementation, clip to 2 ** bitwidth
    quantized_input = np.clip(np.round(grouped_scale / dynamic_scale), 1, 2 ** bitwidth).astype(np.int32)
    return quantized_input.reshape(input_array.shape), dynamic_scale

def compress_encoding_scales(encodings: List[libpymo.TfEncoding],
                             encoding_shape: Sequence[int],
                             block_grouping: Sequence[int],
                             scale_bitwidth: int) -> List[libpymo.TfEncoding]:
    """
    Performs dynamic quantize-dequantization on encodings with the granularity specified in block_grouping

    :param encodings: Encodings to quantize-dequantize
    :param encoding_shape: Shape of encodings
    :param block_grouping: Number of indices at each axis of the encoding_shape to be grouped together
    :param scale_bitwidth: Bitwidth of quantize-dequantize operation to be performed on the encoding scales
    """
    assert len(encoding_shape) == len(block_grouping)
    scale, offset = encodings_to_scale_offset_arrays(encodings, encoding_shape)
    compressed_scales = _compress_encoding_scales(scale, block_grouping, scale_bitwidth)
    new_encodings = scale_offset_arrays_to_encodings(compressed_scales, offset, encodings[0].bw)
    return new_encodings

def _compress_encoding_scales(scale: np.ndarray,
                              block_grouping: Sequence[int],
                              scale_bitwidth: int) -> np.ndarray:
    int_scale, per_group_scale_factor = grouped_dynamic_quantize(scale, block_grouping, scale_bitwidth)
    grouped_int_scale = _split_blocks(int_scale, block_grouping)
    dequantized_scale = grouped_int_scale * per_group_scale_factor
    return dequantized_scale.reshape(scale.shape)

def encodings_to_scale_offset_arrays(encodings: List[libpymo.TfEncoding], shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts list of TfEncoding objects to scale & offset numpy arrays with the given shape
    """
    num_encodings = np.prod(shape)
    assert len(encodings) == num_encodings
    scale = np.array([enc.delta for enc in encodings]).reshape(shape)
    offset = np.array([enc.offset for enc in encodings]).reshape(shape)
    return scale, offset

def scale_offset_arrays_to_encodings(scales: np.ndarray, offsets: np.ndarray, bitwidth) -> List[libpymo.TfEncoding]:
    """
    Converts scale offset arrays to a list of TfEncoding objects
    """
    encodings = []
    for scale, offset in zip(scales.flatten().tolist(), offsets.flatten().tolist()):
        min_val, max_val = compute_min_max_given_delta_offset(scale, offset, bitwidth, False, False)
        encoding = libpymo.TfEncoding()

        encoding.bw = bitwidth
        encoding.min = min_val
        encoding.max = max_val
        encoding.delta = scale
        encoding.offset = offset
        encodings.append(encoding)

    return encodings
