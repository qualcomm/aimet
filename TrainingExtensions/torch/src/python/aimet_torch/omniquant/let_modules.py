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
from aimet_torch.v2.nn import QuantizedLinear, QuantizedLayerNorm
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import copy
from aimet_torch.v2.utils import patch_attr
from abc import abstractmethod


class LETModule():
    def __init__(self, source: QuantizationMixin):
        self._reset_let_params()  
        if source.input_quantizers:
            self.input_quantizers = copy.deepcopy(source.input_quantizers)
        if source.output_quantizers:
            self.output_quantizers = copy.deepcopy(source.output_quantizers)
        if source.param_quantizers:
            self.param_quantizers = copy.deepcopy(source.param_quantizers)

    def _reset_let_params(self):
        self.prev_scale = None
        self.prev_prep_fn = torch.nn.Identity()
        self.foll_scale = None
        self.foll_prep_fn = torch.nn.Identity()

    def get_let_params(self):
        let_params = {
            "prev_scale": self.prev_scale,
            "prev_prep_fn": self.prev_prep_fn,
            "foll_scale": self.foll_scale,
            "foll_prep_fn": self.foll_prep_fn,
        }
        return let_params

    def register_let_params(self, p_scale, p_prep_fn, f_scale, f_prep_fn):
        self.prev_scale = p_scale
        self.prev_prep_fn = p_prep_fn
        self.foll_scale = f_scale
        self.foll_prep_fn = f_prep_fn

    def fold_let_params(self):
        '''
        Call (usually at the end) to fold the scales into the model params
        '''
        self._fold()
        self._reset_let_params()

    def _fold(self):
        params = self._update_parameters()
        with torch.no_grad():
            for k in params:
                getattr(self, k).copy_(params[k])

    @abstractmethod
    def _update_parameters(self):
        assert "Override in child class"
    
   
    @staticmethod # TODO delete
    def from_quantized_module(module):
        # copy w/bias
        # copy q/dq
        # assert module is a quantized module. in the change L -> QL -> QLetL

        #
        '''
        #https://github.com/quic/aimet/blob/6eea45a3b0f21543188598da8a533b6b4369af8e/TrainingExtensions/torch/src/python/aimet_torch/v2/nn/base.py#L383
        # do using load/statedict?
        '''
        assert isinstance(module, QuantizationMixin), f"LET is only supported for quantized modules"
        shape = module.param_quantizers['weight'].shape
        #breakpoint()
        if isinstance(module, QuantizedLinear):
            new_module = LETQuantizedLinear(module.weight.shape[1], module.weight.shape[0])
            breakpoint()
        elif isinstance(module, QuantizedLayerNorm):
            new_module = LETQuantizedLayerNorm(module.weight.shape)
        elif isinstance(module, QuantizedNorm):
            new_module = LETQuantizedRMSNorm(module.weight.shape)
        if isinstance(module, QuantizedGemmaNorm):
            new_module = LETQuantizedGemmaNorm(module.weight.shape)
        else:
            pass
            "TODO : ananmukh Throw descriptive error"
        if module.param_quantizers:
            new_module.param_quantizers['weight'] = QuantizeDequantize(shape=shape, bitwidth=8, symmetric=True)
        if module.input_quantizers[0]:
            new_module.input_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
        if module.output_quantizers[0]:
            new_module.output_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
        new_module.load_state_dict(module.state_dict())
        return new_module

class LETQuantizedLinear(QuantizedLinear, LETModule):

    # def __quant_init__(self):
    #     print("Iam called &&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #     super().__quant_init__()

    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape[1], module.weight.shape[0])
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        weight = self.weight
        bias = self.bias
        
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            if bias is not None:
                bias = bias / prev_scale

            weight = weight / prev_scale.unsqueeze(1)

        if self.foll_scale is not None:
            foll_scale = self.foll_prep_fn(self.foll_scale)
            weight = weight * foll_scale.unsqueeze(0)
        
        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()

        with patch_attr(self, 'weight', params['weight']):
             with patch_attr(self, 'bias', params['bias']):
                #TODO ananmukh ask kygyuen
                super().compute_param_encodings()
                out = super().__call__(*args, **kwargs)                
                return out


class LETQuantizedLayerNorm(QuantizedLayerNorm, LETModule):
    def __quant_init__(self):
        super().__quant_init__()
        LETModule.__init__(self)

    def _update_parameters(self):
        weight = self.weight
        bias = self.bias
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight = weight / prev_scale
            if bias is not None:
                bias = bias / prev_scale

        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        print("params from layernorm", params)
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                super().compute_param_encodings()
                out = super().__call__(*args, **kwargs)
                print("layer norm ", out)
                return out #super().__call__(*args, **kwargs)


class Norm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.rand(dim))
        

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x)

class GemmaRmsNorm(Norm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        self.bias = torch.tensor(1)
    
    def forward(self, x):
        #print("from GemmaRmsNorm", self.weight , self.bias)
        out = (self.weight + self.bias) * super()._norm(x)

        return out

LlamaRmsNorm = Norm

#https://github.qualcomm.com/qualcomm-ai/aimet/blob/6eea45a3b0f21543188598da8a533b6b4369af8e/TrainingExtensions/torch/src/python/aimet_torch/v2/nn/modules/custom.py#L484
@QuantizationMixin.implements(Norm)
class QuantizedNorm(QuantizationMixin, Norm):
    def __quant_init__(self):
        super().__quant_init__()
        self.param_quantizers = nn.ModuleDict({})
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, hidden_states):
        weight = self.weight
        
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)
        
        # if self.param_quantizers:
        #     self.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=True)
        
        
        with self._patch_quantized_parameters():
            out = super().forward(hidden_states)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](hidden_states)
        breakpoint()
        return out

@QuantizationMixin.implements(GemmaRmsNorm)
class QuantizedGemmaNorm(QuantizationMixin, GemmaRmsNorm):
    def __quant_init__(self):
        super().__quant_init__()
        self.param_quantizers = nn.ModuleDict({})
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, hidden_states):
        #weight = self.weight
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # if self.param_quantizers.weight:
        #     weight = self.param_quantizers.weight(weight)

        with self._patch_quantized_parameters():
            hidden_states = super().forward(hidden_states)

        if self.output_quantizers[0]:
            hidden_states = self.output_quantizers[0](hidden_states)
        return hidden_states

class LETQuantizedGemmaNorm(QuantizedGemmaNorm, LETModule):
    def __quant_init__(self):
        super().__quant_init__()
        LETModule.__init__(self)

    def _update_parameters(self):
        weight = self.weight
        bias = 1
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight = weight / prev_scale
            bias = bias / prev_scale

        return {'weight': weight, 'bias': bias}
    
    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        print("params from LETQuantizedGemmaNorm", params)
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                super().compute_param_encodings()
                out = super().__call__(*args, **kwargs)
                print("LETQuantizedGemmaNorm ", out)
                return out

    def _fold(self):
        # TODO: gemma fold can only be caled once
        # Gemma needs rethinking
        weight = self.weight
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight.data = weight.data / prev_scale + 1 / prev_scale - 1


class LETQuantizedRMSNorm(QuantizedNorm, LETModule):
    def __quant_init__(self):
        super().__quant_init__()
        LETModule.__init__(self)

    def _update_parameters(self):
        weight = self.weight
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight = weight / prev_scale

        return {'weight': weight}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        print("params from LETQuantizedRMSNorm", params)
        with patch_attr(self, 'weight', params['weight']):
            super().compute_param_encodings()
            out = super().__call__(*args, **kwargs)
            print("QuantizedLETLlamaRMSNorm ", out)
            return out #super().__call__(*args, **kwargs)


