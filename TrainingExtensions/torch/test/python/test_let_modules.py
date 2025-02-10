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
"""Test Let modules"""
import itertools
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Union, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


from aimet_common import quantsim
from aimet_torch.omniquant.let_modules import LETLinear
#from aimet_torch.omniquant.let_modules import Norm
from aimet_torch.utils import is_vector_encoding
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantization.affine import VectorEncoding
from aimet_torch.v2.quantsim import QuantizationSimModel
from torch import nn
import copy

'''
TODO:
1. add test for gemmarmsnorm->linearpair
2. add more comments
2. add test for layernorm->linear pair
'''

def _copy_quantizers(source, target):
        target.input_quantizers = copy.deepcopy(source.input_quantizers)
        target.output_quantizers = copy.deepcopy(source.output_quantizers)
        target.param_quantizers = copy.deepcopy(source.param_quantizers)

class LinearLinearPair(torch.nn.Module):
    def __init__(self):
        super(LinearLinearPair, self).__init__()
        self.l1 = torch.nn.Linear(1, 2)
        self.l1.weight.data.fill_(0.5)
        self.l1.bias.data.fill_(0.5)
        self.l2 = torch.nn.Linear(2, 1)
        #self.l2.weight.data.fill_(1.5)
        self.l2.weight.data = torch.tensor([[150., 6.]])
        self.l2.bias.data.fill_(10)

    def forward(self, input):
        x = self.l1(input)
        x = self.l2(x)
        return x

# class RmsNormLinearPair(torch.nn.Module):
#     def __init__(self):
#         super(RmsNormLinearPair, self).__init__()
#         self.n1 = Norm((3,))
#         torch.nn.init.uniform_(self.n1.weight, -0.5, 0.5)
#         self.l1 = torch.nn.Linear(3, 4)

#     def forward(self, input):
#         x = self.n1(input)
#         x = self.l1(x)
#         return x

# @QuantizationMixin.implements(Norm)
# class QuantizedNorm(QuantizationMixin, Norm):
#     def __quant_init__(self):
#         super().__quant_init__()
#         self.param_quantizers = nn.ModuleDict({})
#         self.input_quantizers = nn.ModuleList([None])
#         self.output_quantizers = nn.ModuleList([None])

#     def forward(self, hidden_states):
#         if self.input_quantizers[0]:
#             hidden_states = self.input_quantizers[0](hidden_states)

#         with self._patch_quantized_parameters():
#             hidden_states = super().forward(hidden_states)

#         if self.output_quantizers[0]:
#             hidden_states = self.output_quantizers[0](hidden_states)
#         return hidden_states

def test_linear_linear_pair():
    model = LinearLinearPair().eval()
    #inp = torch.rand(1, 1)
    inp = torch.ones(1, 1)
    out = model(inp)
    sim = QuantizationSimModel(model, inp)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model 
    # Replace with let module
    new_module1 = LETLinear(sim.model.l1.weight.shape[1], sim.model.l1.weight.shape[0])
    new_module1.load_state_dict(sim.model.l1.state_dict())
    # new_module1.weight = copy.deepcopy(sim.model.l1.weight)
    # new_module1.bias = copy.deepcopy(sim.model.l1.bias)
    new_module1.update_quantizers(sim.model.l1)

    new_module2 = LETLinear(sim.model.l2.weight.shape[1],sim.model.l2.weight.shape[0])
    new_module2.load_state_dict(sim.model.l2.state_dict())
    # load_state_dict not copying the quantizers from sim to nw module
    # when calling forward in the sim model with let module with the updated weight the o/p is incorrect (l2.wt[0] is not getting updated)
    # new_module2.weight = copy.deepcopy(sim.model.l2.weight)
    # new_module2.bias = copy.deepcopy(sim.model.l2.bias)
    new_module2.update_quantizers(sim.model.l2)

    setattr(sim.model, 'l1', new_module1)
    setattr(sim.model, 'l2',  new_module2)
    
    #breakpoint()
    # forward pass through toy model with let module
    sim.model.l1.prev_scale = torch.tensor([2])
    sim.model.l2.foll_scale = torch.tensor([2])
    out_without_scale = sim.model(inp) 
    breakpoint()
    # # No scale and shift has been set so out1 should be similar to out2
    # assert torch.allclose(sim_out, out_without_scale, atol=0.01)

    # # Set scale
    # sim.model.l1.p_scale = torch.tensor([2])
    # sim.model.l2.f_scale = torch.tensor([3])

    # out_with_randn_scale = sim.model(inp)

    # # out1 and out3 should differ 
    # assert not torch.allclose(sim_out, out_with_randn_scale, atol=0.01)

    # #set scale = 1.
    # sim.model.l1.p_scale = torch.tensor([1])
    # sim.model.l2.f_scale = torch.tensor([1])

    # out4_with_scale_1 = sim.model(inp)


    # w1 = sim.model.l1.weight
    # b1 = sim.model.l1.bias
    # w2 = sim.model.l2.weight
    # b2 = sim.model.l2.bias

    # # since scale = 1 out1 should b similar to out4
    # assert torch.allclose(sim_out, out4_with_scale_1, atol=0.01)

    
    # sim.model.l1.p_scale = torch.tensor([1.03])
    # sim.model.l2.f_scale = torch.tensor([1.03])

    # out_with_scale = sim.model(inp) #quantized with let modules toymodel 
    # assert torch.allclose(sim_out, out_with_scale, atol=0.01)

    # outref = (((inp @ w1.T) + b1) @ w2.T) + b2
    # #remove the qunatizers
    # for name, module in sim.model.named_modules():
    #     if isinstance(module, QuantizationMixin):
    #         module._remove_all_quantizers()

    # out_quantizers_disabled = sim.model(inp)
    # assert torch.allclose(out, out_quantizers_disabled, atol=0.01)

# def test_llamarmsnorm_linear_pair(self):
#     model = RmsNormLinearPair().eval()
#     inp = torch.rand(2, 3)
#     outref_fp = model(inp)
#     sim = QuantizationSimModel(model, inp)
#     sim.compute_encodings(lambda model, _: model(inp), None)
#     out = sim.model(inp) #Quantized toy model 
    
#     # Replace with let module
#     new_module1 = QuantizedLETLlamaRMSNorm.initialize_from_original_module(sim.model.n1)
#     _copy_quantizers(sim.model.n1, new_module1)

#     new_module2 = QuantizedLETLinear.initialize_from_original_module(sim.model.l1)
#     _copy_quantizers(sim.model.l1, new_module2)

#     setattr(sim.model, 'n1', new_module1)
#     setattr(sim.model, 'l1',  new_module2)
#     sim_out = sim.model(inp) 

#     # No scale and shift has been set so out1 should be similar to out2
#     assert torch.allclose(out, sim_out, atol=0.01)


#     # Set scale
#     sim.model.n1.p_scale = torch.tensor([2])
#     sim.model.l1.f_scale = torch.tensor([3])


#     out_with_randn_scale = sim.model(inp)

#     # out1 and out3 should differ 
#     assert not torch.allclose(sim_out, out_with_randn_scale, atol=0.01)


#     #set scale = 1.
#     sim.model.n1.p_scale = torch.tensor([1])
#     sim.model.l1.f_scale = torch.tensor([1])

#     out_with_scale_1 = sim.model(inp)


#     w1 = sim.model.n1.weight
#     #b1 = sim.model.l1.bias
#     w2 = sim.model.l1.weight
#     b2 = sim.model.l1.bias

#     # since scale = 1 out1 should b similar to out4
#     assert torch.allclose(sim_out, out_with_scale_1, atol=0.01)

#     sim.model.n1.p_scale = torch.tensor([1.02])
#     sim.model.l1.f_scale = torch.tensor([1.02])

#     out_with_scale = sim.model(inp) #quantized with let modules toymodel 

#     assert torch.allclose(sim_out, out_with_scale, atol=0.01)

#     #remove the qunatizers
#     for name, module in sim.model.named_modules():
#         if isinstance(module, QuantizationMixin):
#             module._remove_all_quantizers()

#     out_without_quantizers = sim.model(inp)
#     assert torch.allclose(out, out_without_quantizers, atol=0.01)

test_linear_linear_pair()