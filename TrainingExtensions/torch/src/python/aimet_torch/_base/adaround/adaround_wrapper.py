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
""" Defines AdaroundWrapperBase shared across aimet_torch.v1 and v2 """

import abc
from typing import Tuple
import torch

from aimet_common.defs import AdaroundConstants
from aimet_torch._base.quantsim import _QuantizedModuleProtocol
from aimet_torch.v2.utils import patch_attr

class AdaroundWrapperBase(abc.ABC, torch.nn.Module):
    """
    Adaround base class
    """
    def __init__(self, module: _QuantizedModuleProtocol):
        super().__init__()
        assert self.weight_name in module.param_quantizers
        self.module_to_wrap = module
        self._init_param()

    def forward(self, *args, **kwargs):
        """
        Temporarily replace weight of the wrapped module by adarounded weight
        and run forward function of wrapped module
        """
        origianl_module = self.get_original_module()
        weight = self.weight
        if self._is_weight_quantizer_enabled():
            weight = self.apply_adaround(weight)

        with self._disable_weight_quantizer(), \
            patch_attr(origianl_module, self.weight_name, weight):
            return self.module_to_wrap.forward(*args, **kwargs)

    def apply_adaround(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply adaround to the input tensor
        """
        input_dtype = tensor.dtype
        alpha = self.alpha.to(device=tensor.device, dtype=tensor.dtype)

        # Scale the tensor
        tensor = torch.floor(tensor / self.broadcasted_delta)

        # Soft rounding maps alpha parameter between zero and one using
        # rectified sigmoid function and hard rounding maps it to exactly zero or one
        if self.use_soft_rounding:
            h_alpha = torch.clamp(torch.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                  AdaroundConstants.GAMMA, 0, 1)
        else:
            h_alpha = (alpha >= 0).to(tensor.dtype)

        # Adaround the tensor
        tensor = tensor + h_alpha

        # Quantize and de-quantize the tensor
        tensor_quant = torch.clamp(tensor - self.broadcasted_offset, self.clip_min, self.clip_max)
        tensor_dequant = (tensor_quant + self.broadcasted_offset) * self.broadcasted_delta

        return tensor_dequant.to(input_dtype)

    def _get_weight_quantizer_bitwidth(self) -> int:
        """
        Returns bitwidth of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.bitwidth

    @staticmethod
    def _generate_alpha_parameter(tensor: torch.Tensor, delta: torch.Tensor) -> torch.nn.Parameter:
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        tensor_floor = torch.floor(tensor / delta)
        tensor = (tensor / delta) - tensor_floor
        alpha = - torch.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

        # Even if the input is float16, alpha has to be kept in float32
        # in order to be updated by the optimizer
        return torch.nn.Parameter(alpha.float(), requires_grad=True)

    @property
    def weight(self) -> torch.Tensor:
        """
        Returns the weight of the original model
        """
        return getattr(self.get_original_module(), self.weight_name)

    def _init_param(self):
        """
        Initialize adaround parameter using the original module
        """
        self.broadcasted_delta, self.broadcasted_offset = self._get_weight_quantizer_delta_and_offset()
        self.alpha = self._generate_alpha_parameter(self.weight, self.broadcasted_delta)
        self.bitwidth = self._get_weight_quantizer_bitwidth()
        self.use_soft_rounding = True
        self.clip_max = 2 ** self.bitwidth - 1
        self.clip_min = 0

    @property
    def weight_name(self) -> str:
        """
        Returns the name of the weight to apply adaround
        """
        return 'weight'

    @abc.abstractmethod
    def _disable_weight_quantizer(self):
        """
        Temporarily disable weight quantizer
        """

    @abc.abstractmethod
    def _is_weight_quantizer_enabled(self) -> bool:
        """
        Returns true if the weight quantizer is enabled
        """

    @abc.abstractmethod
    def get_original_module(self) -> torch.nn.Module:
        """
        Returns wrapped module
        """

    @abc.abstractmethod
    def _get_weight_quantizer_channel_axis(self) -> int:
        """
        Returns channel axis of the current weight quantizer
        """

    @abc.abstractmethod
    def _get_weight_quantizer_delta_and_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta and offset of the weight quantizer
        """
