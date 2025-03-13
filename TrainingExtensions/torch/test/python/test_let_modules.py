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
from aimet_torch.omniquant.let_modules import LETModule#LETQuantizedLinear, QuantizedLETLlamaRMSNorm, LETLayerNorm, QuantizedLETGemmaRMSNorm
from aimet_torch.omniquant.let_modules import LETQuantizedLinear, LETQuantizedLlamaRMSNorm
from aimet_torch.omniquant.let_modules import Norm,GemmaRmsNorm
from aimet_torch.utils import is_vector_encoding
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantization.affine import VectorEncoding
from aimet_torch.v2.quantsim import QuantizationSimModel
from torch import nn
from aimet_torch.v2.quantization.affine import QuantizeDequantize
import copy
#TODO ananmukh check this
from llama_model.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm
config_file = "/prj/qct/compute_aisw/ananmukh/morpheus/remote_dev/aimet-main/aimet/config/htp_quantsim_config_v73.json"
'''
TODO:
1. add test for gemmarmsnorm->linearpair
2. add more comments
2. add test for layernorm->linear pair
'''

def _reset_quantizers(source, target):
        breakpoint()
        target.input_quantizers = copy.deepcopy(source.input_quantizers)
        target.output_quantizers = copy.deepcopy(source.output_quantizers)
        target.param_quantizers = copy.deepcopy(source.param_quantizers)


# ananmukh use quantized_linear = QuantizationMixin.from_module(linear)
def _reset_model(source, target):
    shape = source.param_quantizers['weight'].shape
    if source.param_quantizers:
        target.param_quantizers['weight'] = QuantizeDequantize(shape=shape, bitwidth=8, symmetric=True)
    if source.input_quantizers[0]:
        target.input_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    if source.output_quantizers[0]:
        target.output_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    target.load_state_dict(source.state_dict())

class LinearLinearPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearLinearPair, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        #self.l1.weight.data.fill_(2)
        #self.l1.bias.data.fill_(1)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)
        #self.l2.weight.data = torch.tensor([[150., 6.]])
        #self.l2.bias.data.fill_(10)

    def forward(self, input):
        x = self.l1(input)
        x = self.l2(x)
        return x

class RmsNormLinearPair(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RmsNormLinearPair, self).__init__()
        self.n1 = LlamaRMSNorm((input_dim,))
        self.l1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, input):
        x = self.n1(input)
        x = self.l1(x)
        return x

class GemmaRmsNormLinearPair(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GemmaRmsNormLinearPair, self).__init__()
        self.gemmarmsnorm = GemmaRmsNorm(input_dim)
        torch.nn.init.uniform_(self.gemmarmsnorm.weight, -0.5, 0.5)
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.gemmarmsnorm(x)
        x = self.linear(x)
        return x

class LayernormLinearPair(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LayernormLinearPair, self).__init__()
        self.layernorm  = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        x = self.layernorm(input)
        x = self.linear(x)
        return x

#TODO ananmukh Add a test for linear layer bias = false
def test_linear_linear_pair():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    model = LinearLinearPair(input_dim, hidden_dim, output_dim).eval()
    inp = torch.rand(1, 10)
    #inp = torch.ones(1, 1)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model 

    #Creating LET Quantized modules from quantized module
    new_module1 = LETQuantizedLinear(sim.model.l1)
    new_module2 = LETQuantizedLinear(sim.model.l2)

    setattr(sim.model, 'l1', new_module1)
    setattr(sim.model, 'l2',  new_module2)
    # forward pass through toy model with let module
    sim_out_with_no_scale = sim.model(inp) 

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.equal(sim_out, sim_out_with_no_scale)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([20])
    sim.model.l1.register_let_params(prev_scale = prev_scale)
    sim.model.l2.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp) 

    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)

    # # Set scale to 2
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.l1.register_let_params(prev_scale = prev_scale)
    sim.model.l2.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough 
    assert  torch.allclose(sim_out, out_with_scale_2, atol=1e-05)

    #remove the qunatizers
    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()

    out_with_quantizers_disabled = sim.model(inp)
    # out_with_quantizers_disabled and out_fp should be same as quantizers were disabled
    assert torch.equal(out_fp, out_with_quantizers_disabled)

    # Test fold
    l1_let_params = sim.model.l1.get_let_params()
    l2_let_params = sim.model.l2.get_let_params()
    orig_wt_l1 = sim.model.l1.weight.cpu().detach().clone()
    orig_wt_l2 = sim.model.l2.weight.cpu().detach().clone()
    # Fold the scale into the weights
    sim.model.l1.fold_let_params()
    sim.model.l2.fold_let_params()
    scale_folded_wts_l1 = sim.model.l1.weight.cpu().detach()
    scale_folded_wts_l2 = sim.model.l2.weight.cpu().detach()

    '''
    On folding the LET scale to weights we update the original model weights  
    l1.w = w/s
    l2.w = w*s
    '''
    assert torch.equal(orig_wt_l1, scale_folded_wts_l1 * l1_let_params['prev_scale'])
    assert torch.equal(orig_wt_l2, scale_folded_wts_l2 / l2_let_params['foll_scale'])


@QuantizationMixin.implements(LlamaRMSNorm)
class QuantizedLlamaRMSNorm(QuantizationMixin, LlamaRMSNorm):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

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

def test_llamarmsnorm_linear_pair():
    input_dim = 3
    output_dim = 2
    model = RmsNormLinearPair(input_dim, output_dim).eval()
    #inp = torch.rand(1, input_dim)
    inp = torch.ones(1, input_dim)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model

    #Creating LET Quantized modules from quantized module
    new_module1 = LETQuantizedLlamaRMSNorm(sim.model.n1)
    new_module2 = LETQuantizedLinear(sim.model.l1)
 
    setattr(sim.model, 'n1', new_module1)
    setattr(sim.model, 'l1',  new_module2)
    # forward pass through toy model with let module
    sim_out_no_scale = sim.model(inp)

    # sim_out_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.equal(sim_out, sim_out_no_scale)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([3])
    sim.model.n1.register_let_params(prev_scale = prev_scale)
    sim.model.l1.register_let_params(foll_scale = foll_scale)

    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp)
    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)

    #set scale = 2.
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.n1.register_let_params(prev_scale = prev_scale)
    sim.model.l1.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough
    assert torch.allclose(sim_out, out_with_scale_2, atol=0.01)

    #remove the qunatizers
    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()
    out_without_quantizers = sim.model(inp)
    # out_quantizers_disabled and out_fp should be same as quantizers were disabled
    assert torch.allclose(out_fp, out_without_quantizers, atol=0.03)

    # Test fold
    n1_let_params = sim.model.n1.get_let_params()
    l1_let_params = sim.model.l1.get_let_params()
    orig_wt_n1 = sim.model.n1.weight.cpu().detach().clone()
    orig_wt_l1 = sim.model.l1.weight.cpu().detach().clone()
    # Fold the scale into the weights
    sim.model.n1.fold_let_params()
    sim.model.l1.fold_let_params()
    scale_folded_wts_n1 = sim.model.n1.weight.cpu().detach()
    scale_folded_wts_l1 = sim.model.l1.weight.cpu().detach()
    '''
    On folding the LET scale to weights we update the original model weights  
    l1.w = w/s
    l2.w = w*s
    '''
    assert torch.equal(orig_wt_n1, scale_folded_wts_n1 * n1_let_params['prev_scale'])
    assert torch.equal(orig_wt_l1, scale_folded_wts_l1 / l1_let_params['foll_scale'])


def test_layernorm_linear_pair():
    input_dim = 4
    output_dim = 2
    model = LayernormLinearPair(input_dim, output_dim).eval()
    inp = torch.rand(1, input_dim)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model

    # Replace with let module
    new_module1 = LETModule.from_quantized_module(sim.model.layernorm)
    new_module2 = LETModule.from_quantized_module(sim.model.linear)
    setattr(sim.model, 'layernorm', new_module1)
    setattr(sim.model, 'linear',  new_module2)

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    sim_out_no_scale = sim.model(inp)
    assert torch.allclose(sim_out, sim_out_no_scale, atol=0.01)

    sim.model.layernorm.prev_scale = torch.tensor([2])
    sim.model.linear.foll_scale = torch.tensor([2])
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out_updated_params = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough 
    assert torch.allclose(sim_out, sim_out_updated_params, atol=0.02)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    sim.model.layernorm.prev_scale = torch.tensor([10])
    sim.model.linear.foll_scale = torch.tensor([100])
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp)
    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)

    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()
            module.reset_let_params()
    sim_out_quantizers_disabled = sim.model(inp)
    # This should be equal to out_fp quantizers were disabled and let params set to none
    assert torch.allclose(out_fp, sim_out_quantizers_disabled, atol=0.01)

def test_gemmarmsnorm_linear_pair():
    input_dim = 3
    output_dim = 2
    model = GemmaRmsNormLinearPair(input_dim, output_dim).eval()
    inp = torch.rand(1, input_dim)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model 
    # Replace with let module
    # new_module1 = QuantizedLETGemmaRMSNorm.initialize_from_original_module(sim.model.gemmarmsnorm)
    # _copy_quantizers(sim.model.gemmarmsnorm, new_module1)
    # new_module2 = LETLinear(input_dim, output_dim)
    # _update_quantizers(sim.model.linear, new_module2)

    new_module1 = LETModule.from_quantized_module(sim.model.gemmarmsnorm)
    new_module2 = LETModule.from_quantized_module(sim.model.linear)
    setattr(sim.model, 'gemmarmsnorm', new_module1)
    setattr(sim.model, 'linear',  new_module2)

    sim_out_no_scale = sim.model(inp) 
    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.allclose(sim_out_no_scale, sim_out, atol=0.01)


    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    sim.model.gemmarmsnorm.prev_scale = torch.tensor([2])
    sim.model.linear.foll_scale = torch.tensor([3])
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp)
    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    breakpoint()
    assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)


    #set scale = 1.
    sim.model.gemmarmsnorm.prev_scale = torch.tensor([2])
    sim.model.linear.foll_scale = torch.tensor([2])
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    breakpoint()
    # sim_out and out_with_scale_2 should be close enough
    assert torch.allclose(sim_out, out_with_scale_2, atol=0.01)

    #remove the qunatizers
    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()
            module.reset_let_params()

    out_without_quantizers = sim.model(inp)
    breakpoint()
    # out_quantizers_disabled and out_fp should be same as quantizers were disabled
    #assert torch.allclose(out_fp, out_without_quantizers, atol=0.01)



class LinearToyModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearToyModel, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, output_dim)
        self.l1.weight.data.fill_(2)
        self.l1.bias.data.fill_(1)

    def forward(self, input):
        x = self.l1(input)
        return x
def test_compute_encodings():
    model = LinearToyModel(1,1)
    x = [i for i in range(1,10)]
    y = [i for i in range(10, 0, -1)]
    input = x+y
    for item in input:
        inp = torch.tensor(item, dtype=torch.float32)
        out_fp = model(inp.unsqueeze(0))
        print("ANANMUKH****** ",out_fp)
    dummy_inp = torch.tensor(1, dtype=torch.float32).unsqueeze(0)
    sim = QuantizationSimModel(model, dummy_inp, config_file=config_file)
    breakpoint()
    inp1 = torch.tensor(input[0], dtype=torch.float32)
    sim.compute_encodings(lambda model, _: model(inp1.unsqueeze(0)), None)
    sim_out = sim.model(inp1.unsqueeze(0))
    print("ANANMUKH SIMMMM 1****** ",sim_out, sim.model.l1.input_quantizers[0].min, sim.model.l1.input_quantizers[0].max)
    for item in input[1:-1]:
        
        inp = torch.tensor(item, dtype=torch.float32)
        if item ==10:
            sim.compute_encodings(lambda model, _: model(inp.unsqueeze(0)), None)
        sim_out = sim.model(inp.unsqueeze(0))
        #breakpoint()
        print("ANANMUKH SIMMMM ****** ",sim_out, sim.model.l1.input_quantizers[0].min, sim.model.l1.input_quantizers[0].max)

    inp2 = torch.tensor(input[-1], dtype=torch.float32)
    sim.compute_encodings(lambda model, _: model(inp2.unsqueeze(0)), None)
    sim_out = sim.model(inp1.unsqueeze(0))
    print("ANANMUKH SIMMMM 2****** ",sim_out, sim.model.l1.input_quantizers[0].min, sim.model.l1.input_quantizers[0].max)
    # for item in input:
    #     inp = torch.tensor(item, dtype=torch.float32)
    #     sim.compute_encodings(lambda model, _: model(inp.unsqueeze(0)), None)
    #     sim_out = sim.model(inp.unsqueeze(0))
    #     #breakpoint()
    #     print("ANANMUKH SIMMMM ****** ",sim_out, sim.model.l1.input_quantizers[0].min, sim.model.l1.input_quantizers[0].max)


test_llamarmsnorm_linear_pair()
test_linear_linear_pair()