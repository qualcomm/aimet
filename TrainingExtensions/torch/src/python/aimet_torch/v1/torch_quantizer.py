# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities that are used for different quantization simulation features """

import math
from typing import Tuple, Union
import torch
from packaging import version

from aimet_common import libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantizationDataType
from aimet_torch.v1.tensor_quantizer import StaticGridTensorQuantizer, LearnedGridTensorQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

torch_quantizer_zero_ponit_data_type = torch.int64
if version.parse(torch.__version__) > version.parse('1.10.2'):
    torch_quantizer_zero_ponit_data_type = torch.int32


def calc_params_for_native_torch_quantizer(quantizer, ch_axis, device: torch.device) \
        -> Tuple[Union[torch.Tensor, float], Union[torch.Tensor, int], int, int]:
    """
    This function merely transforms previously computed quant encodings to the expected pytorch function format
    :param quantizer: input or output quantizer
    :param ch_axis: Channel axis is used for per-channel quant. Otherwise it will be set to None
    :param device: device on which model is
    :return: tuple of quantization parameters used to set up native torch quantizer
    """

    numSteps = pow(2, quantizer.bitwidth) - 1
    encodings = quantizer.encoding

    if quantizer.use_strict_symmetric:
        error_msg = 'Strict symmetric is not supported by native torch quantizer'
        logger.error(error_msg)
        raise ValueError(error_msg)

    if ch_axis is None:
        # Per tensor quantization
        scale = float(encodings.delta)
        zero_point = int(-encodings.offset)
        if quantizer.use_symmetric_encodings and (encodings.min < 0
                                                  or (not quantizer.use_unsigned_symmetric)):
            # Symmetric quantization
            q_max = math.floor(numSteps / 2)
            q_min = -math.ceil(numSteps / 2)
            zero_point = 0
        else:
            # Unsigned symmetric
            q_min, q_max = 0, numSteps

    else:
        # Per Channel quantization
        scale = torch.tensor([encoding.delta for encoding in encodings], device=device)
        zero_point = torch.tensor([int(-encoding.offset) for encoding in encodings], device=device, dtype=torch_quantizer_zero_ponit_data_type)
        if quantizer.use_symmetric_encodings and (all(encoding.min < 0 for encoding in encodings)
                                                  or (not quantizer.use_unsigned_symmetric)):
            # Symmetric quantization
            q_max = math.floor(numSteps / 2)
            q_min = -math.ceil(numSteps / 2)
            zero_point = torch.zeros_like(zero_point, dtype=torch_quantizer_zero_ponit_data_type)
        else:
            # Unsigned symmetric
            q_min, q_max = 0, numSteps

    return scale, zero_point, q_max, q_min


class TorchQuantizer:
    """
    A Quantizer using native torch quantization nodes
    """
    def __init__(self, quantizer: Union[StaticGridTensorQuantizer, LearnedGridTensorQuantizer],
                 device: torch.device):
        """
        Constructor
        :param post_training_module: StaticGridQuantWrapper wrapped module
        :param device: device on which model is
        """
        super().__init__()
        self.device = device
        self.enabled = quantizer.enabled
        self.data_type = quantizer.data_type
        self.bitwidth = quantizer.bitwidth
        self._ch_axis = None

        if self.data_type == QuantizationDataType.float and self.bitwidth != 16:
            raise ValueError('Only FP16 quantizers are supported by TorchQuantizer')
        encodings = quantizer.encoding
        # To aviod quantizer.enabled is True but quantizer.encoding is None
        if quantizer.enabled and quantizer.encoding:
            if not isinstance(encodings, libpymo.TfEncoding):
                # pylint: disable=protected-access
                self._ch_axis = quantizer._ch_axis
            self.scale, self.zero_point, self.q_max, self.q_min = calc_params_for_native_torch_quantizer(quantizer, self._ch_axis, device)

    def quantize_dequantize(self, tensor: torch.Tensor):
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """
        if self.enabled:
            if self.data_type == QuantizationDataType.float:
                quantized_tensor = tensor.half()
                quantized_tensor = quantized_tensor.float()
                return quantized_tensor
            if self._ch_axis is None:
                return torch.fake_quantize_per_tensor_affine(tensor, self.scale, self.zero_point,
                                                             self.q_min, self.q_max)

            return torch.fake_quantize_per_channel_affine(tensor, self.scale, self.zero_point,
                                                          self._ch_axis, self.q_min, self.q_max)

        return tensor
