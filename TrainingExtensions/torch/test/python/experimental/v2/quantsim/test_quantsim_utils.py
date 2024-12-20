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
""" Test experimental utilities for QuantizationSimModel """

import json
import os
import torch
import tempfile
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine.backends.torch_builtins import quantize
from aimet_torch.v2.experimental import clip_weights_to_7f7f
from ....models.test_models import SingleResidualWithAvgPool

def test_clip_weights_to_7f7f():
    torch.manual_seed(0)
    model = SingleResidualWithAvgPool().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    # Force all weights to positive to guarantee max quantized value will be > 32639
    for module in model.modules():
        if hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.copy_(torch.abs(module.weight))

    quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
            json.dump(quantsim_config, f)

        qsim = QuantizationSimModel(model, dummy_input, config_file=os.path.join(tempdir, 'quantsim_config.json'), default_param_bw=16)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    affected_quant_layers = []
    for _, quant_layer in qsim.named_qmodules():
        if 'weight' in quant_layer.param_quantizers and quant_layer.param_quantizers['weight'] is not None:
            encoding = quant_layer.param_quantizers['weight'].get_encodings()
            quantized_weight = quantize(quant_layer.weight, encoding.scale, encoding.offset, -32768, 32767)
            assert torch.equal(torch.max(quantized_weight), torch.tensor(32767))
            affected_quant_layers.append(quant_layer)
        assert affected_quant_layers

    clip_weights_to_7f7f(qsim)

    for quant_layer in affected_quant_layers:
        encoding = quant_layer.param_quantizers['weight'].get_encodings()
        quantized_weight = quantize(quant_layer.weight, encoding.scale, encoding.offset, -32768, 32767)
        assert torch.equal(torch.max(quantized_weight), torch.tensor(32639))
