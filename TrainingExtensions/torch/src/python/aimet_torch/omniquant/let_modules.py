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

#from transformers.models.llama.modelling_llama import LlamaRMSNorm
from aimet_torch.v2.nn import QuantizedLinear
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import copy

'''
TODO:
    1. Add helper method @QuantizedLETModule for get_let_params
    2. Add comments
    3. implemenet let conv
'''
class LETModule():
    def __init__(self):
        self.init_let_module()

    def init_let_module(self):
        self.prev_scale = None
        self.prev_shift = None
        self.prev_prep_fn = torch.nn.Identity()
        self.foll_scale = None
        self.foll_shift = None
        self.foll_prep_fn = torch.nn.Identity()

    def get_let_params(self):
        let_params = {
            "prev_scale": self.prev_scale,
            "prev_shift": self.prev_shift,
            "prev_prep_fn": self.prev_prep_fn,
            "foll_scale": self.foll_scale,
            "foll_shift": self.foll_shift,
            "foll_prep_fn": self.foll_prep_fn,
        }
        return let_params

    def register_let_params(self, p_scale, p_shift, p_prep_fn, f_scale, f_shift, f_prep_fn):
        self.prev_scale = p_scale
        self.prev_shift = p_shift
        self.prev_prep_fn = p_prep_fn
        self.foll_scale = f_scale
        self.foll_shift = f_shift
        self.foll_prep_fn = f_prep_fn
        if p_shift is not None or f_shift is not None:
            assert self.bias is not None

    def fold_let_params(self):
        self.fold()
        self.prev_scale = None
        self.prev_shift = None
        self.prev_prep_fn = torch.nn.Identity()
        self.foll_scale = None
        self.foll_shift = None
        self.foll_prep_fn = torch.nn.Identity()

    def update_quantizers(self, source):
        self.input_quantizers = copy.deepcopy(source.input_quantizers)
        self.output_quantizers = copy.deepcopy(source.output_quantizers)
        self.param_quantizers = copy.deepcopy(source.param_quantizers)

class QuantizedLETConv(LETModule, torch.nn.Conv2d):
    def __quant_init__(self):
        super().__quant_init__()
        self.param_quantizers = nn.ModuleDict({})
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, input):
        pass

    def fold(elf):
        pass

class LETLinear(QuantizedLinear, LETModule):
    def __quant_init__(self):
        #QuantizedLinear.__quant_init__()
        print("xxxx")
        super().__quant_init__()
        LETModule.__init__(self)


    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        bias = self.bias
        print("&&&&&&&&", self.weight)
        
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            if bias is not None:
                if self.prev_shift is not None:
                    prev_shift = self.prev_prep_fn(self.prev_shift)
                    bias = bias - prev_shift
                bias = bias / prev_scale

            weight = weight / prev_scale.unsqueeze(1)

        if self.foll_scale is not None:
            foll_scale = self.foll_prep_fn(self.foll_scale)
            if bias is not None:
                if self.foll_shift is not None:
                    foll_shift = self.fprep_fn(self.foll_shift)
                    bias = bias + torch.matmul(weight, foll_shift)

            weight = weight * foll_scale.unsqueeze(0)
        

        '''
        if self.param_quantizers.weight:
            w = self.param_quantizers["weight"](w)
        '''
        print("1", input, self.weight, self.bias)
        print("2",input, weight, bias)
        # self.weight = nn.Parameter(weight)
        # self.bias = nn.Parameter(bias)
        self.weight.data.copy_(weight)
        self.bias.data.copy_(bias)
        print("3", input, self.weight, self.bias)
        print("4",input, weight, bias)
        # TODO, does weight need to be assigned back to self.weight (or is it pointer like?)
        #breakpoint()
        out = super().forward(input)
        out1 = F.linear(input, self.weight, self.bias)
        breakpoint()
        
        '''
        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)
        '''
        return out

    def fold(self):
        weight = self.weight
        bias = self.bias

        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            if bias is not None:
                if self.prev_shift is not None:
                    prev_shift = self.prev_prep_fn(self.prev_shift)
                    bias.data -= prev_shift
                bias.data /= prev_scale
            weight.data /= prev_scale.unsqueeze(1)

        if self.foll_scale is not None:
            foll_scale = self.foll_prep_fn(self.foll_scale)
            if bias is not None:
                if self.foll_shift is not None:
                    foll_shift = self.foll_prep_fn(foll_shift)
                    bias.data += torch.matmul(weight, foll_shift)
            weight.data *= foll_scale.unsqueeze(0)


#     @classmethod
#     def initialize_from_original_module(cls, orig_module):
#         bias = True if orig_module.bias is not None else False
#         new_module = cls(
#             in_features=orig_module.in_features,
#             out_features=orig_module.out_features,
#             bias=bias,
#             dtype=orig_module.weight.dtype
#         )
#         new_module.weight.data.copy_(orig_module.weight.detach())
#         if bias:
#             new_module.bias.data.copy_(orig_module.bias.detach())
#         return new_module

# class QuantizedLETLayerNorm(QuantizedLETModule, QuantizedLayerNorm):u
#     def __quant_init__(self):
#         super().__quant_init__()
#         QuantizedLETModule.__init__(self)
#         self.param_quantizers = nn.ModuleDict({})
#         self.input_quantizers = nn.ModuleList([None])
#         self.output_quantizers = nn.ModuleList([None])

#     def forward(self, input: Tensor) -> Tensor:
#         w = self.weight
#         bias = self.bias
#         if self.p_scale is not None:
#             p_scale = self.p_prep_fn(self.p_scale)
#             w = w / p_scale
#             if b is not None:
#                 b = b / p_scale

#         if self.input_quantizers[0]:
#             hidden_states = self.input_quantizers[0](hidden_states)

#         if self.param_quantizers.weight:
#             w = self.param_quantizers.weight(w)

#         output = F.layer_norm(hidden_states, self.normalized_shape, w, b, self.eps)

#         if self.output_quantizers[0]:
#             output = self.output_quantizers[0](output)
#         return output

#     def fold(self):
#         pass

# class Norm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.zeros(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         return self.weight * self._norm(x)

# class GemmaRmsNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
    
#     def forward(self, x):
#         return (self.weight + 1) * self._norm(x)

# LlamaRmsNorm = Norm

# class QuantizedLETGemmaRMSNorm(QuantizationMixin,QuantizedLETModule, GemmaRmsNorm):
#     def __quant_init__(self):
#         super().__quant_init__()
#         QuantizedLETModule.__init__(self)
#         self.param_quantizers = nn.ModuleDict({})
#         self.input_quantizers = nn.ModuleList([None])
#         self.output_quantizers = nn.ModuleList([None])

#     def forward(self, hidden_states):
#         w = self.weight
#         b = 1
#         if self.p_scale is not None:
#             p_scale = self.p_prep_fn(self.p_scale)
#             w = w / p_scale
#             b = b / p_scale

#         if self.input_quantizers[0]:
#             hidden_states = self.input_quantizers[0](hidden_states)

#         if self.param_quantizers.weight:
#             w = self.param_quantizers.weight(w + b)

#         # == super().forward() ==
#         #output = self._norm(hidden_states.float())
#         #output = output * (b + w)
#         # TODO check the d-types
#         output = super.forward(hidden_states)
#         if self.output_quantizers[0]:
#             output = self.output_quantizers[0](output)
#         return output

#     def fold(self):
#         w = self.weight
#         if self.p_scale is not None:
#             p_scale = self.p_prep_fn(self.p_scale)
#             w.data = w.data / p_scale + 1 / p_scale - 1

#     @torch.no_grad()
#     def get_original_module(self):
#         hidden_size = self.weight.shape[0]
#         eps = self.eps
#         orig_module = GemmaRMSNorm(hidden_size, eps)
#         orig_module.weight.copy_(self.weight)
#         return orig_module

# @QuantizationMixin.implements(LlamaRmsNorm)
# class QuantizedLETLlamaRMSNorm(QuantizationMixin, QuantizedLETModule, LlamaRmsNorm):
#     def __quant_init__(self):
#         super().__quant_init__()
#         QuantizedLETModule.__init__(self)
#         self.param_quantizers = nn.ModuleDict({})
#         self.input_quantizers = nn.ModuleList([None])
#         self.output_quantizers = nn.ModuleList([None])

#     def forward(self, hidden_states):
#         w = self.weight

#         if self.p_scale is not None:
#             p_scale = self.p_prep_fn(self.p_scale)
#             w = w / p_scale

#         if self.input_quantizers[0]:
#             hidden_states = self.input_quantizers[0](hidden_states)

#         if self.param_quantizers.weight:
#             w = self.param_quantizers.weight(w)
        
#         #TODO check the d-types
#         '''
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)


#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         output = w * hidden_states.to(input_dtype)
#         '''

#         output = super().forward(hidden_states)

#         if self.output_quantizers[0]:
#             output = self.output_quantizers[0](output)
#         return output

#     def fold(self):
#         w = self.weight
#         if self.p_scale is not None:
#             p_scale = self.p_prep_fn(self.p_scale)
#             w.data /= p_scale

#     @classmethod
#     def initialize_from_original_module(cls, orig_module):
#         ##breakpoint()
#         hidden_size = orig_module.weight.shape[0]
#         eps = 1e-6#orig_module.eps
#         new_module = cls(hidden_size, eps)
#         #new_module.weight.copy_(orig_module.weight.detach())
#         new_module.weight = nn.Parameter(orig_module.weight.detach())
#         return new_module

