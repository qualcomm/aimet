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
import json
import os
import torch
from torch import nn
import peft.tuners.lora.layer as lora
import tempfile

import aimet_torch.v2 as aimet
from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.experimental import lora as qlora


class TestQuantizedLinear:
    def test_quantsim_basics(self):
        model = lora.Linear(nn.Linear(10, 10), adapter_name='adapter_0', r=1)
        dummy_input = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input)

        """
        When: Create quantsim with lora.Linear
        Then: 1) lora.Linear should be converted to QuantizedLinear
              2) Mul and Add modules should have input and output quantizers as necessary
              3) All lora adapters (lora_A, B) and base layer should be converted to aimet.nn.QuantizedLinear
        """
        assert isinstance(sim.model, qlora.QuantizedLinear)
        assert isinstance(sim.model.mul['adapter_0'].input_quantizers[1], affine.QuantizeDequantize)
        assert isinstance(sim.model.mul['adapter_0'].output_quantizers[0], affine.QuantizeDequantize)
        assert isinstance(sim.model.add['adapter_0'].output_quantizers[0], affine.QuantizeDequantize)

        lora_A = sim.model.lora_A["adapter_0"]
        assert isinstance(lora_A, aimet.nn.QuantizedLinear)
        assert isinstance(lora_A.param_quantizers['weight'], affine.QuantizeDequantize)
        assert isinstance(lora_A.output_quantizers[0], affine.QuantizeDequantize)

        lora_B = sim.model.lora_B["adapter_0"]
        assert isinstance(lora_B, aimet.nn.QuantizedLinear)
        assert isinstance(lora_B.param_quantizers['weight'], affine.QuantizeDequantize)
        assert isinstance(lora_B.output_quantizers[0], affine.QuantizeDequantize)

        base_layer = sim.model.base_layer
        assert isinstance(base_layer, aimet.nn.QuantizedLinear)
        assert isinstance(base_layer.param_quantizers['weight'], affine.QuantizeDequantize)
        assert isinstance(base_layer.output_quantizers[0], affine.QuantizeDequantize)

        """
        When: compute_encodings
        Then: All quantizers should be initialized
        """
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        for qtzr in sim.model.modules():
            if isinstance(qtzr, QuantizerBase):
                assert qtzr.is_initialized()

        """
        When: Export
        Then: The generated encoding file should contain all entries properly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, 'model', dummy_input=dummy_input)
            with open(os.path.join(tmpdir, 'model_torch.encodings')) as f:
                encodings = json.load(f)

        expected_schema = {
            'activation_encodings': {
                'base_layer':       {'input': {'0': ...}, 'output': ...},
                'lora_A.adapter_0': {'input': {'0': ...}, 'output': ...},
                'lora_B.adapter_0': {                     'output': ...},
                'mul.adapter_0':    {'input': {'1': ...}, 'output': ...},
                'add.adapter_0':    {                     'output': ...},
            },
            'param_encodings': {
                'base_layer.weight': ...,
                'lora_A.adapter_0.weight': ...,
                'lora_B.adapter_0.weight': ...,
            }
        }

        def _assert_same_keys(d: dict, expected: dict):
            assert d.keys() == expected.keys()

            for k in d:
                v1, v2 = d[k], expected[k]
                if isinstance(v2, dict):
                    _assert_same_keys(v1, v2)

        _assert_same_keys(encodings['activation_encodings'], expected_schema['activation_encodings'])
        _assert_same_keys(encodings['param_encodings'], expected_schema['param_encodings'])

    @pytest.mark.skip(reason="To be discussed")
    def test_update_layer(self):
        """
        When: Add a new lora adapter with "update_layer" API
        Then: The new added adapters should be aimet.nn.QuantizedLinear with
              param and output quantizers instantiated as necessary
        """
        model = lora.Linear(nn.Linear(10, 10), adapter_name='adapter_0', r=1)
        dummy_input = torch.randn(10, 10)
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
