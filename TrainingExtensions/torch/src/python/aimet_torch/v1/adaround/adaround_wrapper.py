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

""" Custom Wrapper for quantizing weights using Adaround """

import contextlib
from typing import Tuple
import torch
import torch.nn

# Import AIMET specific modules
import aimet_common.aimet_tensor_quantizer as AimetTensorQuantizer
from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO
from aimet_torch._base.adaround.adaround_wrapper import AdaroundWrapperBase
from aimet_torch.v1.tensor_quantizer import StaticGridPerChannelQuantizer
from aimet_torch.v1.quantsim_straight_through_grad import broadcast_to_tensor


class AdaroundWrapper(AdaroundWrapperBase):
    """
    Adaround wrapper class for AIMET v1
    """

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns original module so that we can check its
        module type or access its weight
        """
        # pylint: disable=protected-access
        return self.module_to_wrap._module_to_wrap

    @contextlib.contextmanager
    def _disable_weight_quantizer(self):
        """
        Temporarily disable weight quantizer
        """
        weight_quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        is_enabled = weight_quantizer.enabled
        weight_quantizer.enabled = False
        yield
        weight_quantizer.enabled = is_enabled

    def _is_weight_quantizer_enabled(self) -> bool:
        """
        Returns true if the weight quantizer is enabled
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.enabled

    def _get_weight_quantizer_channel_axis(self) -> int:
        """
        Returns channel axis of the current weight quantizer
        """
        # pylint: disable = protected-access
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer, StaticGridPerChannelQuantizer):
            return quantizer._ch_axis
        return 0

    def _get_weight_quantizer_delta_and_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta and offset of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer.encoding, list):
            # pylint: disable = protected-access
            cpp_op = AimetTensorQuantizer.AimetTensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[quantizer.quant_scheme])
            delta, offset = cpp_op.makeDeltaOffsetTensor(self.weight.device, quantizer.encoding)
        else:
            delta, offset = quantizer.encoding.delta, quantizer.encoding.offset

        ch_axis = self._get_weight_quantizer_channel_axis()
        return broadcast_to_tensor(self.weight, delta, ch_axis), broadcast_to_tensor(self.weight, offset, ch_axis)
