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
from aimet_torch.omniquant.let_modules import (
    LETModule,
    LETQuantizedLinear,
    LETQuantizedLlamaRMSNorm,
    LETQuantizedLayerNorm,
    LETQuantizedConv2d,
    LETQuantizedGemmaNorm,
    QuantizedGemmaNorm,
    QuantizedLlamaRMSNorm,
)

from aimet_torch.utils import is_vector_encoding
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from torch import nn
from aimet_torch.v2.quantization.affine import QuantizeDequantize
import copy
config_file = "/prj/qct/compute_aisw/ananmukh/morpheus/remote_dev/aimet-main/aimet/config/htp_quantsim_config_v73.json"
'''
TODO: ananmukh 
add doc string comments
'''

class LinearLinearPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearLinearPair, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.l1(input)
        x = self.l2(x)
        return x

class ConvConvPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvConvPair, self).__init__()
        self.conv1  = torch.nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU() 

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
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
        self.gemmarmsnorm = GemmaRMSNorm(input_dim)
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
    inp = torch.rand(1, input_dim)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model 

    # Creating LET Quantized modules from quantized module
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

    # Set scale to 2
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.l1.register_let_params(prev_scale = prev_scale)
    sim.model.l2.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough 
    assert  torch.allclose(sim_out, out_with_scale_2, atol=1e-05)

    # remove the qunatizers
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

def test_conv_conv_pair():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    model = ConvConvPair(input_dim, hidden_dim, output_dim).eval()
    inp = torch.rand(1, input_dim, 32, 32)
    out_fp = model(inp)
    
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model

    #Creating LET Quantized modules from quantized module
    new_module1 = LETQuantizedConv2d(sim.model.conv1)
    new_module2 = LETQuantizedConv2d(sim.model.conv2)

    setattr(sim.model, 'conv1', new_module1)
    setattr(sim.model, 'conv2',  new_module2)
    # forward pass through toy model with let module
    sim_out_with_no_scale = sim.model(inp) 

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.equal(sim_out, sim_out_with_no_scale)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([20])
    sim.model.conv1.register_let_params(prev_scale = prev_scale)
    sim.model.conv2.register_let_params(foll_scale = foll_scale)

    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_radn_scale = sim.model(inp)

    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_radn_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_radn_scale, atol=0.01)

    # Set scale to 2
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.conv1.register_let_params(prev_scale = prev_scale)
    sim.model.conv2.register_let_params(foll_scale = foll_scale)
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
    conv1_let_params = sim.model.conv1.get_let_params()
    conv2_let_params = sim.model.conv2.get_let_params()
    orig_wt_conv1 = sim.model.conv1.weight.cpu().detach().clone()
    orig_wt_conv2 = sim.model.conv2.weight.cpu().detach().clone()
    # Fold the scale into the weights
    sim.model.conv1.fold_let_params()
    sim.model.conv2.fold_let_params()
    scale_folded_wts_conv1 = sim.model.conv1.weight.cpu().detach()
    scale_folded_wts_conv2 = sim.model.conv2.weight.cpu().detach()
    '''
    On folding the LET scale to weights we update the original model weights  
    l1.w = w/s
    l2.w = w*s
    '''
    assert torch.allclose(orig_wt_conv1, scale_folded_wts_conv1 * conv1_let_params['prev_scale'])
    assert torch.allclose(orig_wt_conv2, scale_folded_wts_conv2 / conv2_let_params['foll_scale'])



try:
    from transformers.models.llama.modelling_llama import LlamaRMSNorm
except ImportError:
    class LlamaRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
            

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            return self.weight * self._norm(x)

QuantizedLlamaRMSNorm = QuantizationMixin.implements(LlamaRMSNorm)(QuantizedLlamaRMSNorm)

def test_llamarmsnorm_linear_pair():
    input_dim = 3
    output_dim = 2
    model = RmsNormLinearPair(input_dim, output_dim).eval()
    inp = torch.rand(1, input_dim)
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

    # Creating LET Quantized modules from quantized module
    new_module1 = LETQuantizedLayerNorm(sim.model.layernorm)
    new_module2 = LETQuantizedLinear(sim.model.linear)
    setattr(sim.model, 'layernorm', new_module1)
    setattr(sim.model, 'linear',  new_module2)

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    sim_out_no_scale = sim.model(inp)
    assert torch.equal(sim_out, sim_out_no_scale)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.tensor([10])
    foll_scale = torch.tensor([100])
    sim.model.layernorm.register_let_params(prev_scale = prev_scale)
    sim.model.linear.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp)
    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    #assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)

    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.layernorm.register_let_params(prev_scale = prev_scale)
    sim.model.linear.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough 
    assert torch.allclose(sim_out, out_with_scale_2, atol=0.02)

    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()

    out_with_quantizers_disabled = sim.model(inp)
    # out_with_quantizers_disabled and out_fp should be same as quantizers were disabled
    assert torch.allclose(out_fp, out_with_quantizers_disabled, atol=0.01)

    # Test fold
    ln_let_params = sim.model.layernorm.get_let_params()
    lin_let_params = sim.model.linear.get_let_params()
    orig_wt_ln = sim.model.layernorm.weight.cpu().detach().clone()
    orig_wt_lin = sim.model.linear.weight.cpu().detach().clone()
    # Fold the scale into the weights
    sim.model.layernorm.fold_let_params()
    sim.model.linear.fold_let_params()
    scale_folded_wts_layernorm = sim.model.layernorm.weight.cpu().detach()
    scale_folded_wts_linear = sim.model.linear.weight.cpu().detach()
    '''
    On folding the LET scale to weights we update the original model weights  
    l1.w = w/s
    l2.w = w*s
    '''
    assert torch.allclose(orig_wt_ln, scale_folded_wts_layernorm * ln_let_params['prev_scale'])
    assert torch.allclose(orig_wt_lin, scale_folded_wts_linear / lin_let_params['foll_scale'])

try:
    from transformers.models.gemma.modeling_gemma.py import GemmaRMSNorm
except ImportError:
    class GemmaRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.zeros(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            output = self._norm(x.float())
            # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
            # See https://github.com/huggingface/transformers/pull/29402
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)

QuantizedGemmaNorm = QuantizationMixin.implements(GemmaRMSNorm)(QuantizedGemmaNorm)

def test_gemmarmsnorm_linear_pair():
    input_dim = 3
    output_dim = 2
    model = GemmaRmsNormLinearPair(input_dim, output_dim).eval()
    inp = torch.rand(1, input_dim)
    out_fp = model(inp)
    sim = QuantizationSimModel(model, inp, config_file=config_file)
    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp) #Quantized toy model 
 
    #Creating LET Quantized modules from quantized module
    new_module1 = LETQuantizedGemmaNorm(sim.model.gemmarmsnorm)
    new_module2 = LETQuantizedLinear(sim.model.linear)
    setattr(sim.model, 'gemmarmsnorm', new_module1)
    setattr(sim.model, 'linear',  new_module2)
    # forward pass through toy model with let module
    sim_out_no_scale = sim.model(inp) 

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.equal(sim_out_no_scale, sim_out)


    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([3])
    sim.model.gemmarmsnorm.register_let_params(prev_scale = prev_scale)
    sim.model.linear.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_rand_scale = sim.model(inp)

    # Model params are updated due to non zero scale. 
    # Prev and foll scale are different, hence sim_out, out_with_rand_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_rand_scale, atol=0.01)


    #set scale = 1.
    prev_scale = torch.tensor([2])
    foll_scale = torch.tensor([2])
    sim.model.gemmarmsnorm.register_let_params(prev_scale = prev_scale)
    sim.model.linear.register_let_params(foll_scale = foll_scale)
    sim.compute_encodings(lambda model, _: model(inp), None)
    out_with_scale_2 = sim.model(inp)
    # sim_out and out_with_scale_2 should be close enough
    assert torch.allclose(sim_out, out_with_scale_2)

    #remove the qunatizers
    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()

    out_with_quantizers_disabled = sim.model(inp)
    # out_with_quantizers_disabled and out_fp should be same as quantizers were disabled
    assert torch.equal(out_fp, out_with_quantizers_disabled)

    # Test fold
    gemmarmsnorm_let_params = sim.model.gemmarmsnorm.get_let_params()
    linear_let_params = sim.model.linear.get_let_params()
    orig_wt_gemmarmsnorm = sim.model.gemmarmsnorm.weight.cpu().detach().clone()
    orig_wt_linear = sim.model.linear.weight.cpu().detach().clone()
    # Fold the scale into the weights
    sim.model.gemmarmsnorm.fold()
    sim.model.linear.fold_let_params()
    scale_folded_wts_gemmarmsnorm = sim.model.gemmarmsnorm.weight.cpu().detach()
    scale_folded_wts_linear = sim.model.linear.weight.cpu().detach()
    '''
    On folding the LET scale to weights we update the original model weights  
    l1.w = w/s
    l2.w = w*s
    '''
    assert torch.allclose(orig_wt_gemmarmsnorm, scale_folded_wts_gemmarmsnorm * gemmarmsnorm_let_params['prev_scale'])
    assert torch.allclose(orig_wt_linear, scale_folded_wts_linear / linear_let_params['foll_scale'])

