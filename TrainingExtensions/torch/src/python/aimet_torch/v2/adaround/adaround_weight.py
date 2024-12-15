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

""" Top level API for Adaptive Rounding - Post-Training Quantization (PTQ) """

import contextlib
import itertools
from typing import Tuple, Union
import torch
from torch import nn

from aimet_common.defs import QuantScheme
from aimet_torch._base.adaround.adaround_wrapper import AdaroundWrapperBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch._base.adaround.adaround_weight import (
    AdaroundBase,
    AdaroundParameters,
    AdaroundSupportedModules,
)


__all__ = [
    'Adaround',
    'AdaroundParameters',
    'AdaroundSupportedModules',
]


class Adaround(AdaroundBase):
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    Subclass for AIMET v2 compatibility
    """
    @staticmethod
    def _get_quantsim(model: torch.nn.Module, dummy_input: torch.Tensor,
                      quant_scheme: QuantScheme, default_param_bw: int, config_file: str):
        return QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=quant_scheme,
                                    default_param_bw=default_param_bw,
                                    config_file=config_file)

    @staticmethod
    def _get_adaround_wrapper(quant_module: BaseQuantizationMixin):
        return AdaroundWrapper(quant_module)

    @classmethod
    def _remove_quantization_wrappers(cls, module: BaseQuantizationMixin):
        for module_name, module_ref in module.named_children():
            if isinstance(module_ref, BaseQuantizationMixin):
                setattr(module, module_name, module_ref.get_original_module())  # pylint: disable=protected-access
            else:
                cls._remove_quantization_wrappers(module_ref)

    @staticmethod
    def _get_quant_wrapper(quant_sim_model: torch.nn.Module, module_name: str) -> Union[BaseQuantizationMixin, None]:
        """
        For given module name, get the quantized module from the QuantSim model
        :param quant_sim_model: Model with simulation ops
        :param module_name: Module name
        :return: Quantized wrapper module or None
        """
        quant_module = None

        for name, module in quant_sim_model.named_modules():
            if name == module_name and isinstance(module, BaseQuantizationMixin):
                quant_module = module
                break

        return quant_module

    @staticmethod
    def _compute_param_encodings(quant_sim: QuantizationSimModel):
        """
        Compute encodings for parameters, needed for initializing Adaround quantizers
        :param quant_sim: Quant sim
        """
        for quant_module in quant_sim.model.modules():
            if isinstance(quant_module, BaseQuantizationMixin):
                # Adaround requires input and output quantizers to be disabled
                quant_module.input_quantizers = nn.ModuleList([None for _ in quant_module.input_quantizers])
                quant_module.output_quantizers = nn.ModuleList([None for _ in quant_module.output_quantizers])

                for name, param_quantizer in quant_module.param_quantizers.items():
                    if not param_quantizer:
                        continue

                    param = getattr(quant_module, name)
                    with param_quantizer.compute_encodings():
                        _ = param_quantizer(param)

    @staticmethod
    def _validate_quant_module_for_adaround(quant_module: BaseQuantizationMixin):
        assert quant_module.param_quantizers['weight'], '%s does not have weight parameter.' % quant_module
        assert quant_module.param_quantizers['weight'].is_initialized(), '%s encoding needs to be set.' % quant_module

    @staticmethod
    def _check_input_output_quantizers_for_adaround(quant_model: torch.nn.Module):
        for module in quant_model.modules():
            if isinstance(module, BaseQuantizationMixin):
                for quantizer in itertools.chain(module.input_quantizers, module.output_quantizers):
                    assert quantizer is None

    @staticmethod
    def _get_lowest_weight_bw(quant_model: torch.nn.Module):
        param_quantizers = []
        for module in quant_model.modules():
            if isinstance(module, BaseQuantizationMixin):
                for quantizer in module.param_quantizers.values():
                    if isinstance(quantizer, AffineQuantizerBase):
                        param_quantizers.append(quantizer)

        return min(
            quantizer.bitwidth for quantizer in param_quantizers
        )


class AdaroundWrapper(AdaroundWrapperBase):
    """
    Basic adaround wrapper class for AIMET v2
    """
    @contextlib.contextmanager
    def _disable_weight_quantizer(self):
        """
        Temporarily disable weight quantizer
        """
        weight_quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        self.module_to_wrap.param_quantizers[self.weight_name] = None
        yield
        self.module_to_wrap.param_quantizers[self.weight_name] = weight_quantizer

    def _is_weight_quantizer_enabled(self) -> bool:
        """
        Returns true if the weight quantizer is enabled
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return bool(quantizer)

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns wrapped module
        """
        return self.module_to_wrap

    def _get_weight_quantizer_channel_axis(self) -> int:
        """
        Returns channel axis of the current weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        for idx, dim in enumerate(quantizer.shape):
            if dim != 1:
                return idx
        return 0

    def _get_weight_quantizer_delta_and_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta and offset of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.get_scale().expand_as(self.weight), quantizer.get_offset().expand_as(self.weight)

    def _init_param(self):
        super()._init_param()
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if quantizer.signed:
            self.clip_max = 2 ** (self.bitwidth - 1) - 1
            self.clip_min = - 2 ** (self.bitwidth - 1)
