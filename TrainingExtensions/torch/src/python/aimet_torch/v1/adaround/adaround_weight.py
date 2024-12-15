# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import itertools
from typing import Union
import torch

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType

from aimet_torch import utils
from aimet_torch.save_utils import SaveUtils
from aimet_torch._base.adaround.adaround_weight import (
    AdaroundBase,
    AdaroundParameters,
    AdaroundSupportedModules,
)
from aimet_torch.v1.quantsim import QuantizationSimModel, QcQuantizeWrapper
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper, QcQuantizeOpMode
from aimet_torch.v1.adaround.adaround_wrapper import AdaroundWrapper


__all__ = [
    'Adaround',
    'AdaroundParameters',
    'AdaroundSupportedModules',
]

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class Adaround(AdaroundBase):
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """
    @staticmethod
    def _compute_param_encodings(quant_sim: QuantizationSimModel):
        """
        Compute encodings for parameters, needed for initializing Adaround quantizers
        :param quant_sim: Quant sim
        """
        for quant_module in quant_sim.model.modules():
            if isinstance(quant_module, StaticGridQuantWrapper):
                # Adaround requires input and output quantizers to be disabled
                for quatizer in quant_module.input_quantizers:
                    quatizer.enabled = False
                for quatizer in quant_module.output_quantizers:
                    quatizer.enabled = False

                # pylint: disable=protected-access
                for name, param in quant_module._module_to_wrap.named_parameters():
                    param_quantizer = quant_module.param_quantizers[name]
                    param_quantizer.reset_encoding_stats()
                    param_quantizer.update_encoding_stats(param.data)
                    param_quantizer.compute_encoding()

                # Wrapper mode must be set to ACTIVE because the wrapper's quantize_dequantize_params() will only call
                # into the param tensor quantizer's quantize_dequantize() if the mode is not PASSTHROUGH.
                quant_module.set_mode(QcQuantizeOpMode.ACTIVE)

    @staticmethod
    def _get_quantsim(model: torch.nn.Module, dummy_input: torch.Tensor,
                      quant_scheme: QuantScheme, default_param_bw: int, config_file: str):
        return QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=quant_scheme,
                                    default_param_bw=default_param_bw,
                                    config_file=config_file)

    @staticmethod
    def _get_adaround_wrapper(quant_module: QcQuantizeWrapper):
        return AdaroundWrapper(quant_module)

    @staticmethod
    def _remove_quantization_wrappers(module: torch.nn.Module):
        SaveUtils.remove_quantization_wrappers(module)

    @staticmethod
    def _validate_quant_module_for_adaround(quant_module: StaticGridQuantWrapper):
        assert quant_module.param_quantizers['weight'], '%s does not have weight parameter.' % quant_module
        assert quant_module.param_quantizers['weight'].encoding, '%s encoding needs to be set.' % quant_module

    @staticmethod
    def _check_input_output_quantizers_for_adaround(quant_model: torch.nn.Module):
        _, input_quantizers, output_quantizers = utils.get_all_quantizers(quant_model)
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            assert not quantizer.enabled

    @staticmethod
    def _get_lowest_weight_bw(quant_model: torch.nn.Module):
        param_quantizers, _, _ = utils.get_all_quantizers(quant_model)
        return min(
            quantizer.bitwidth for quantizer in param_quantizers
            if quantizer.enabled and quantizer.data_type == QuantizationDataType.int
        )

    @staticmethod
    def _get_quant_wrapper(quant_sim_model: torch.nn.Module, module_name: str) -> Union[StaticGridQuantWrapper, None]:
        """
        For given module name, get the quantized wrapper module from the QuantSim model
        :param quant_sim_model: Model with simulation ops
        :param module_name: Module name
        :return: Quantized wrapper module or None
        """
        quant_module = None

        for name, module in quant_sim_model.named_modules():
            if name == module_name and isinstance(module, StaticGridQuantWrapper):
                quant_module = module
                break

        return quant_module
