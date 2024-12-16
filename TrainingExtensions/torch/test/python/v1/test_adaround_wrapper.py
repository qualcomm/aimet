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

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper
from aimet_torch.v1.adaround.adaround_wrapper import AdaroundWrapper


@pytest.mark.parametrize('module_factory', [lambda: torch.nn.Linear(12, 8),
                                            lambda: torch.nn.Conv2d(12, 8, 3),
                                            lambda: torch.nn.ConvTranspose2d(12, 8, 3)])
def test_adaround_tensor_quantizer(module_factory):
    """ Test the Adarounding of a Tensor """
    nearest_encoding = libpymo.TfEncoding()
    nearest_encoding.bw = 4
    nearest_encoding.max = 10.0
    nearest_encoding.min = 0.19699306
    nearest_encoding.offset = -127.0
    nearest_encoding.delta = 0.001551126479

    module = module_factory()
    weight_tensor = torch.randn(module.weight.shape)
    wrapper = StaticGridQuantWrapper(module, weight_bw=4, activation_bw=4, round_mode='nearest', quant_scheme=QuantScheme.post_training_tf_enhanced)
    wrapper.param_quantizers['weight'].encoding = nearest_encoding
    ada_wrapper = AdaroundWrapper(wrapper)
    ada_quantized = ada_wrapper.apply_adaround(weight_tensor)
    assert not torch.equal(weight_tensor, ada_quantized)
