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

""" Implementation for handling LoRA adapters added using PEFT """
# pylint: disable=import-error
# pylint: disable=no-name-in-module
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin, lora as qlora


def _get_lora_layer_except_base_layer(sim: QuantizationSimModel):
    part_of_lora_layer_except_base = set()
    for module in sim.model.modules():
        if isinstance(module, (qlora.QuantizedLinear, qlora.QuantizedConv)):
            for m in module.modules():
                if isinstance(m, BaseQuantizationMixin) and m != module.base_layer:
                    part_of_lora_layer_except_base.add(m)
    return part_of_lora_layer_except_base

def _freeze_quantizer(quantizer):
    """
    Disables compute encodings and gradient update for a quantizer

    :param quantizer: Param, output or Input quantizer
    """
    # pylint:disable = protected-access
    quantizer._allow_overwrite = False
    quantizer.requires_grad_(False)

def freeze_base_model_param_quantizers(sim: QuantizationSimModel):
    """
    Freeze parameter quantizers of base model

    :param sim: QuantSim model
    """
    def _freeze(module):
        for param_quantizer in module.param_quantizers.values():
            if param_quantizer:
                _freeze_quantizer(param_quantizer)

    part_of_lora_layer_except_base = _get_lora_layer_except_base_layer(sim)
    for module in sim.model.modules():
        if isinstance(module, BaseQuantizationMixin) and module not in part_of_lora_layer_except_base:
            _freeze(module)

def freeze_base_model_activation_quantizers(sim: QuantizationSimModel):
    """
    Freeze activation quantizers of base model

    :param sim: QuantSim model
    """
    def _freeze(module):
        for input_quantizer, output_quantizer in zip(module.input_quantizers, module.output_quantizers):
            if input_quantizer:
                _freeze_quantizer(input_quantizer)
            if output_quantizer:
                _freeze_quantizer(output_quantizer)

    part_of_lora_layer_except_base = _get_lora_layer_except_base_layer(sim)
    for module in sim.model.modules():
        if isinstance(module, BaseQuantizationMixin) and module not in part_of_lora_layer_except_base:
            _freeze(module)

def freeze_base_model(sim: QuantizationSimModel):
    """
    Freeze entire base model

    :param sim: QuantSim model
    """
    freeze_base_model_param_quantizers(sim)
    freeze_base_model_activation_quantizers(sim)
