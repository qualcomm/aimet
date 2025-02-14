# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for quantization """

from typing import Dict
import numpy as np

from aimet_common.defs import QuantizationDataType, EncodingType
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def get_conv_accum_bounds(weights: np.ndarray, quant_bw: int, accum_bw: int):
    """
    Get upper and lower bounds for accumulator for a given layer
    :param weights: Weight tensor in OIHW format
    :param quant_bw: Quantization bitwidth
    :param accum_bw: Accumulator bitwidth
    :return: Tuple of (was accumulator range exceeded, most accumulator range used)
    """

    # Max integer value
    max_int_value = (2 ** quant_bw) - 1
    max_accum_value = 2 ** (accum_bw - 1)

    # Calculate min and max (absolute)
    quant_min = min(np.min(weights), 0)
    quant_max = max(np.max(weights), 0)
    quant_scale = 2 * max(abs(quant_min), abs(quant_max)) / max_int_value
    if quant_scale == 0:
        quant_scale = 1e-5      # Prevent divide by zero for degenerate layers

    most_accum_range_used = 0
    was_accum_range_exceeded = False

    for out_chan_index in range(weights.shape[0]):

        accum_max = np.sum(max_int_value * np.maximum(np.round(weights[out_chan_index] / quant_scale), 0))
        accum_min = np.sum(max_int_value * np.minimum(np.round(weights[out_chan_index] / quant_scale), 0))

        if accum_max / max_accum_value > most_accum_range_used:
            most_accum_range_used = accum_max / max_accum_value

        if accum_min / -max_accum_value > most_accum_range_used:
            most_accum_range_used = accum_min / -max_accum_value

        if (accum_max >= max_accum_value) or (accum_min < -max_accum_value):
            was_accum_range_exceeded = True
            _logger.info("Accumulator range potentially exceeded in channel %d", out_chan_index)

    return was_accum_range_exceeded, most_accum_range_used

def _convert_encoding_format_0_6_1_to_1_0_0(encodings: Dict[str, Dict]):
    """
    Convert encoding dictionary in 0.6.1 format to 1.0.0 format (list of encodings)
    """
    new_encodings = []
    assert isinstance(encodings, dict)
    for tensor_name, encoding in encodings.items():
        new_encoding = {'name': tensor_name}
        new_encoding['bw'] = encoding[0]['bitwidth']
        if encoding[0]['dtype'] == 'float':
            new_encoding['dtype'] = QuantizationDataType.float.name.upper()
            new_encoding['enc_type'] = EncodingType.PER_TENSOR.name
        else:
            new_encoding['dtype'] = QuantizationDataType.int.name.upper()
            new_encoding['is_sym'] = encoding[0]['is_symmetric'] == 'True'
            if len(encoding) == 1:
                new_encoding['enc_type'] = EncodingType.PER_TENSOR.name
            else:
                new_encoding['enc_type'] = EncodingType.PER_CHANNEL.name

            scale = []
            offset = []
            for enc in encoding:
                scale.append(enc['scale'])
                offset.append(enc['offset'])
            new_encoding['scale'] = scale
            new_encoding['offset'] = offset
        new_encodings.append(new_encoding)

    return new_encodings
