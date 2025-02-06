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
import tempfile
import torch
from peft import PeftMixedModel, PeftModel
from peft import LoraConfig, get_peft_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.lora.peft_utils import freeze_base_model_activation_quantizers, freeze_base_model_param_quantizers


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


def two_adapter_model():
    model = DummyModel()
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,
        bias="none",
        target_modules=["linear"],
    )

    peft_model = PeftMixedModel(model, lora_config)
    peft_model.add_adapter('default_new', lora_config)
    peft_model.set_adapter(['default', 'default_new'])
    return peft_model


class TestLoraAdapterPeft:
    def test_freeze_base_model(self):
        model = two_adapter_model()
        dummy_inputs = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)
        print(sim)
        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        sim.compute_encodings(forward_pass, None)
        qc_lora = sim.model.base_model.model.linear

        assert not _is_frozen(qc_lora.base_layer.param_quantizers['weight'])
        freeze_base_model_param_quantizers(sim)
        freeze_base_model_activation_quantizers(sim)

        assert _is_frozen(qc_lora.base_layer.param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_A['default'].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_A['default_new'].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_B['default'].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_B['default_new'].param_quantizers['weight'])

        assert _is_frozen(qc_lora.base_layer.output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_A['default'].output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_B['default_new'].output_quantizers[0])

    def test_lora_flow(self):
        model = two_adapter_model()
        dummy_inputs = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        sim.compute_encodings(forward_pass, None)

        # Export lora model
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=True,
                       filename_prefix_encodings='encodings')

def _is_frozen(quantizer):
    return quantizer._allow_overwrite == False and\
           quantizer.min.requires_grad == False and\
           quantizer.max.requires_grad == False
