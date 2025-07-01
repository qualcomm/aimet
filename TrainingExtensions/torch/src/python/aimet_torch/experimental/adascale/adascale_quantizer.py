# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
"""Adascale quantizer"""

from abc import abstractmethod
from typing import Optional

import torch

from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.affine.backends import torch_builtins

use_adascale_lwc: bool = True


class AdaScaleQuantizeDequantize(QuantizeDequantize):
    """Base class for AdaScale QDQ"""

    beta: torch.nn.Parameter
    gamma: torch.nn.Parameter

    def __init__(self, qdq: QuantizeDequantize):
        super().__init__(
            qdq.shape,
            qdq.bitwidth,
            qdq.symmetric,
            qdq.encoding_analyzer,
            qdq.block_size,
        )

        self.register_parameter("beta", torch.nn.Parameter(torch.zeros(self.shape)))
        self.register_parameter("gamma", torch.nn.Parameter(torch.zeros(self.shape)))

        self.set_range(qdq.min, qdq.max)
        self.min.requires_grad = False
        self.max.requires_grad = False

    def get_adascale_trainable_parameters(self):
        """Method to query all the trainable parameters of AdaScale QDQ"""
        return self._get_beta_gamma(), self._get_learnable_scales()

    def get_qdq(self) -> QuantizeDequantize:
        """
        Return the Quantized QDQ object for the sim object to be restored to original condition.
        S2, S3 are not used to create QDQ object. This needs to be folded into the weights before converting to QDQ.
        """
        q = QuantizeDequantize(
            self.shape,
            self.bitwidth,
            self.symmetric,
            self.encoding_analyzer,
            self.block_size,
        )
        q.set_range(self.get_min(), self.get_max())
        return q

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        scale = (
            torch.exp(self.gamma) * self.max.to(dtype)
            - torch.exp(self.beta) * self.min.to(dtype)
        ) / self._get_num_steps()
        return scale

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        return torch.zeros_like(self.min, requires_grad=False, dtype=dtype)

    def get_folded_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return the folded weight of the layer. This method along with get_qdq can be used to convert AdaScale
        QDQ object into regular QDQ object
        """
        for scale in self._get_learnable_scales():
            weight = weight / torch.exp(scale)
        return weight

    def _get_beta_gamma(self) -> list[torch.Tensor]:
        """lwc trainable parameters introduced in omniquant"""
        return [self.beta, self.gamma]

    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """
        Performs QDQ on the input tensor based on the learnt scales by using the parameters min and max

        :param input: Input tensor to be QDQ
        :return: Dequantized tensor after applying AdaScale QDQ
        """
        for scale in self._get_learnable_scales():
            input = input / torch.exp(scale)
        return super().forward(input)

    @abstractmethod
    def _get_learnable_scales(self) -> list[torch.Tensor]:
        """learnable scales corresponding to the module type"""


class AdaScaleLinearQuantizeDequantize(AdaScaleQuantizeDequantize):
    """Specialized class for AdaScale Linear QDQ"""

    s2: torch.nn.Parameter
    s3: torch.nn.Parameter

    def __init__(self, qdq: QuantizeDequantize, weight_shape: torch.Size):
        """
        Creates the AdaScale QDQ object. This quantizer should be substituted in place of Linear weight qdq object

        :param qdq: QuantizeDequantize object using which the Adascale object needs to be created
        :param weight_shape: Shape of the weight tensor
        """

        assert use_adascale_lwc, "Flexround QDQ is not yet implemented."
        assert qdq.symmetric is True, "Only symmetric quantization is supported"
        super().__init__(qdq=qdq)

        if qdq.block_size is not None:
            self.register_parameter(
                "s2",
                torch.nn.Parameter(
                    torch_builtins.reshape_tensor_for_blocks(
                        torch.zeros(weight_shape), qdq.shape, self.block_size
                    ).squeeze(1)
                ),
            )
            self.register_parameter(
                "s3", torch.nn.Parameter(torch.zeros(self.shape).unsqueeze(-1))
            )
        else:
            self.register_parameter("s2", torch.nn.Parameter(torch.zeros(weight_shape)))
            self.register_parameter("s3", torch.nn.Parameter(torch.zeros(self.shape)))

    def _get_learnable_scales(self) -> list[torch.Tensor]:
        """learnable scales corresponding to Linear layer"""
        return [self.s2, self.s3]


class AdaScaleConv2dQuantizeDequantize(AdaScaleQuantizeDequantize):
    """Specialized class for AdaScale Conv2d QDQ"""

    s2: torch.nn.Parameter
    s3: torch.nn.Parameter
    s4: torch.nn.Parameter

    def __init__(self, qdq: QuantizeDequantize, weight_shape: torch.Size):
        """
        Creates the AdaScale QDQ object. This quantizer should be substituted in place of Conv2d weight qdq object

        :param qdq: QuantizeDequantize object using which the Adascale object needs to be created
        :param weight_shape: Shape of the weight tensor
        """

        assert use_adascale_lwc, "Flexround QDQ is not yet implemented."
        assert qdq.symmetric is True, "Only symmetric quantization is supported"
        super().__init__(qdq=qdq)

        out_ch, in_ch, _, _ = weight_shape

        if qdq.block_size is not None:
            self.register_parameter(
                "s2",
                torch.nn.Parameter(
                    torch_builtins.reshape_tensor_for_blocks(
                        torch.zeros(weight_shape), qdq.shape, self.block_size
                    ).squeeze(1)
                ),
            )
            self.register_parameter(
                "s3", torch.nn.Parameter(torch.zeros((out_ch, 1, 1, 1)))
            )
            self.register_parameter(
                "s4", torch.nn.Parameter(torch.zeros((1, in_ch, 1, 1)))
            )
        else:
            self.register_parameter("s2", torch.nn.Parameter(torch.zeros(weight_shape)))
            self.register_parameter(
                "s3", torch.nn.Parameter(torch.zeros((out_ch, 1, 1, 1)))
            )
            self.register_parameter(
                "s4", torch.nn.Parameter(torch.zeros((1, in_ch, 1, 1)))
            )

    def _get_learnable_scales(self) -> list[torch.Tensor]:
        """learnable scales corresponding to Linear layer"""
        return [self.s2, self.s3, self.s4]
