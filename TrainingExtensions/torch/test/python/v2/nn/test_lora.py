#!/usr/bin/env python3
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

import pytest
import torch
from torch import nn
import peft.tuners.lora.layer as lora

import aimet_torch.v2 as aimet
from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.experimental import lora as qlora


class TestQuantizedLinear:
    def test_quantsim_construction(self):
        model = lora.Linear(nn.Linear(10, 10), adapter_name='adapter_0', r=1).cuda()
        dummy_input = torch.randn(10, 10, device="cuda:0")
        sim = QuantizationSimModel(model, dummy_input)

        """
        When: Create quantsim with lora.Linear
        Then: 1) lora.Linear should be converted to QuantizedLinear
              2) Mul and Add modules should have input and output quantizers as necessary
        """
        assert isinstance(sim.model, qlora.QuantizedLinear)
        assert isinstance(sim.model.mul['adapter_0'].input_quantizers[1], affine.QuantizeDequantize)
        assert isinstance(sim.model.mul['adapter_0'].output_quantizers[0], affine.QuantizeDequantize)
        assert isinstance(sim.model.add['adapter_0'].output_quantizers[0], affine.QuantizeDequantize)

        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        assert sim.model.mul['adapter_0'].input_quantizers[1].is_initialized()
        assert sim.model.mul['adapter_0'].output_quantizers[0].is_initialized()
        assert sim.model.add['adapter_0'].output_quantizers[0].is_initialized()


    @pytest.mark.skip(reason="To be discussed")
    def test_update_layer(self):
        """
        When: Add a new lora adapter with "update_layer" API
        Then: The new added adapters should be aimet.nn.QuantizedLinear with
              param and output quantizers instantiated as necessary
        """
        model = lora.Linear(nn.Linear(10, 10), adapter_name='adapter_0', r=1).cuda()
        dummy_input = torch.randn(10, 10, device="cuda:0")
        sim = QuantizationSimModel(model, dummy_input)

        sim.model.update_layer("new_adapter", ...)
        new_lora_a = sim.model.lora_A["new_adapter"]
        new_lora_b = sim.model.lora_B["new_adapter"]

        assert isinstance(new_lora_a, aimet.nn.QuantizedLinear)
        assert isinstance(new_lora_a.param_quantizers['weight'], affine.QuantizeDequantize)
        assert isinstance(new_lora_a.output_quantizers[0], affine.QuantizeDequantize)

        assert isinstance(new_lora_b, aimet.nn.QuantizedLinear)
        assert isinstance(new_lora_b.param_quantizers['weight'], affine.QuantizeDequantize)
        assert isinstance(new_lora_b.output_quantizers[0], affine.QuantizeDequantize)

        assert isinstance(sim.model.mul['new_adapter'].input_quantizers[1], affine.QuantizeDequantize)
        assert isinstance(sim.model.mul['new_adapter'].output_quantizers[0], affine.QuantizeDequantize)
        assert isinstance(sim.model.add['new_adapter'].output_quantizers[0], affine.QuantizeDequantize)
