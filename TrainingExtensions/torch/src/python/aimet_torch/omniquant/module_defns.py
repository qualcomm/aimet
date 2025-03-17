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

# pylint: disable=missing-module-docstring
import torch
from torch import nn
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.llama.modelling_llama import LlamaRMSNorm
except ImportError:
    class LlamaRMSNorm(torch.nn.Module):
        '''
        Define llama rms norm if import from transformers fails
        '''
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # pylint: disable=missing-function-docstring
        def forward(self, x):
            return self.weight * self._norm(x)

@QuantizationMixin.implements(LlamaRMSNorm)
class QuantizedLlamaRMSNorm(QuantizationMixin, LlamaRMSNorm):
    '''
        Define QuantizedLlamaRMSNorm
    '''
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    # pylint: disable=arguments-differ
    def forward(self, hidden_states):
        # Quantize input tensors
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret

try:
    from transformers.models.gemma.modeling_gemma.py import GemmaRMSNorm
except ImportError:
    class GemmaRMSNorm(nn.Module):
        '''
        Define gemma rms norm if import from transformers fails
        '''
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.zeros(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # pylint: disable=arguments-differ,missing-function-docstring
        def forward(self, x):
            output = self._norm(x.float())
            # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
            # See https://github.com/huggingface/transformers/pull/29402
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)

@QuantizationMixin.implements(GemmaRMSNorm)
class QuantizedGemmaNorm(QuantizationMixin, GemmaRMSNorm):
    '''
        Define QuantizedGemmaNorm
    '''
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])
        self.bias = 1 # TODO bias is a bad name, change to something else

    # pylint: disable=arguments-differ
    def forward(self, hidden_states):
        weight = self.weight
        bias = self.bias
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        if self.param_quantizers.weight:
            weight = self.param_quantizers.weight(weight)

        ret = self._norm(hidden_states.float())
        ret = ret * (bias+weight)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret
