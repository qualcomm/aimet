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
""" LET Quantized Module """
from aimet_torch.v2.nn import (
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedConv2d,
)
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import patch_attr
from aimet_torch.experimental.omniquant.module_defns import (
    QuantizedLlamaRMSNorm,
    QuantizedGemmaNorm,
)

import torch
import copy
from abc import abstractmethod

class LETModule():
    """
    LET modules implementation for omniquant
    """
    def __init__(self, source: QuantizationMixin):
        self._cached_prev_scale = None
        self._cached_foll_scale = None
        self._reset_let_params()
        # TODO in e2e integration decide what happens if some of the quantizers are None/missing
        # For now we assume all 3 values are present, else we throw an error
        for quantizers in ['input_quantizers', 'output_quantizers', 'param_quantizers']:
            src_quant = getattr(source, quantizers)
            assert src_quant, f'{quantizers} should not be none for LETModule'
            setattr(self, quantizers, copy.deepcopy(src_quant))

    def _reset_let_params(self):
        """ Set LET modules prev_scale/foll_scale to None"""
        self.prev_scale = None
        self.foll_scale = None

    def get_let_params(self):
        """ Query LET modules prev_scale/foll_scale """
        let_params = {
            "prev_scale": self.prev_scale,
            "foll_scale": self.foll_scale,
        }
        return let_params

    # pylint: disable=attribute-defined-outside-init
    def register_let_params(self, prev_scale = None, foll_scale = None, num_repeats = 1):
        """ Set prev_scale and foll_sclae to LET pairs. """
        if prev_scale is not None:
            self.prev_scale = prev_scale
        if foll_scale is not None:
            self.foll_scale = foll_scale
        # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
        # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
        # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
        self.num_repeats = num_repeats

    def _cache_train_scale(self):
        """ Cache trained scale to numpy tensor. """
        self._cached_prev_scale = self.prev_scale.data.cpu().numpy() if self.prev_scale is not None else None
        self._cached_foll_scale = self.foll_scale.data.cpu().numpy() if self.foll_scale is not None else None

    def fold_let_params(self):
        """ Call (usually at the end) to fold the scales into the model params, cache trained scale, reset let param to None. """
        self._fold()
        self._cache_train_scale()
        self._reset_let_params()

    @abstractmethod
    def _fold(self):
        params = self._update_parameters()
        with torch.no_grad():
            for k in params:
                param = getattr(self, k)
                if param is not None:
                    param.copy_(params[k])

    @abstractmethod
    def _update_parameters(self):
        assert False, "Override in child class"

    def get_source_quant_module(self):
        """ Create original quantize module with new quantizer and parameter. """
        source_quant_module = self._get_source_quant_module()
        for quantizers in ['input_quantizers', 'output_quantizers', 'param_quantizers']:
            let_quant = getattr(self, quantizers)
            assert let_quant, f'{quantizers} should not be none for LETModule'
            setattr(source_quant_module, quantizers, copy.deepcopy(let_quant))

        for w_b in ["weight", "bias"]:
            updated_param = getattr(self, w_b, None)
            if updated_param is not None:
                source_param = getattr(source_quant_module, w_b)
                source_param.copy_(updated_param)

        return source_quant_module

    @abstractmethod
    def _get_source_quant_module(self):
        assert False, "Override in child class"

    @staticmethod
    def get_let_module(mdl):
        """ Return corresponding LETQuantized module for different Quantized modules """
        if isinstance(mdl, QuantizedLinear):
            return LETQuantizedLinear(mdl)
        if isinstance(mdl, QuantizedLayerNorm):
            return LETQuantizedLayerNorm(mdl)
        if isinstance(mdl, QuantizedConv2d):
            return LETQuantizedConv2d(mdl)
        if isinstance(mdl, QuantizedLlamaRMSNorm):
            return LETQuantizedLlamaRMSNorm(mdl)
        if isinstance(mdl, QuantizedGemmaNorm):
            return LETQuantizedGemmaNorm(mdl)
        assert False, "Let Quantized module is not implemented"

# pylint: disable=too-many-ancestors
class LETQuantizedLinear(QuantizedLinear, LETModule):
    """ LET module implementation for QuantizedLinear """
    def __init__(self, module:QuantizationMixin):
        # TODO pass in all params to ctor
        super().__init__(module.weight.shape[1], module.weight.shape[0], bias=module.bias is not None)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        """
        Update the LETQuantizedLinear params with prev_scale/foll_scale
        """
        weight = self.weight
        bias = self.bias

        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            if bias is not None:
                bias = bias / prev_scale
            weight = weight / prev_scale.reshape(-1, 1)

        if self.foll_scale is not None:
            # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
            # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
            # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
            foll_scale = torch.repeat_interleave(self.foll_scale, dim=0, repeats=self.num_repeats)
            weight = weight * foll_scale

        return {'weight': weight, 'bias': bias}

    def _get_source_quant_module(self):
        return QuantizedLinear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # No need to call compute_encodings here as we don't want to calibrate
                # min-max when doing LET blockwise training.
                return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors
class LETQuantizedConv2d(QuantizedConv2d, LETModule):
    """ LET module implementation for QuantizedConv2d """
    def __init__(self, module:QuantizationMixin):
        # TODO pass in all params to ctor
        super().__init__(module.weight.shape[1], module.weight.shape[0], module.kernel_size, module.stride, module.padding, bias=module.bias is not None)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        """
        Update the LETQuantizedConv2d params with prev_scale/foll_scale
        """
        weight = self.weight
        bias = self.bias

        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            if bias is not None:
                bias = bias / prev_scale
            weight = weight / prev_scale.reshape(-1, 1, 1, 1)

        if self.foll_scale is not None:
            foll_scale = self.foll_scale
            weight = weight * foll_scale.reshape(1, -1, 1, 1)

        return {'weight': weight, 'bias': bias}

    def _get_source_quant_module(self):
        return QuantizedConv2d(self.weight.shape[1], self.weight.shape[0], self.kernel_size, self.stride, self.padding, bias=self.bias is not None)

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # No need to call compute_encodings here as we don't want to calibrate
                # min-max when doing LET blockwise training.
                return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors
class LETQuantizedLayerNorm(QuantizedLayerNorm, LETModule):
    """ LET module implementation for QuantizedLayerNorm """
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        """
        Update the LETQuantizedLayerNorm params with prev_scale/foll_scale
        """
        weight = self.weight
        bias = self.bias
        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            weight = weight / prev_scale
            if bias is not None:
                bias = bias / prev_scale

        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # No need to call compute_encodings here as we don't want to calibrate
                # min-max when doing LET blockwise training.
                return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors
class LETQuantizedLlamaRMSNorm(QuantizedLlamaRMSNorm, LETModule):
    """ LET module implementation for QuantizedLlamaRMSNorm """
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.variance_epsilon = module.variance_epsilon
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        """
        Update the LETQuantizedLlamaRMSNorm params with prev_scale/foll_scale
        """
        weight = self.weight
        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            weight = weight / prev_scale

        return {'weight': weight}

    def _get_source_quant_module(self):
        return QuantizedLlamaRMSNorm(self.weight.shape)

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            # No need to call compute_encodings here as we don't want to calibrate
            # min-max when doing LET blockwise training.
            return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors
class LETQuantizedGemmaNorm(QuantizedGemmaNorm, LETModule):
    """ LET module implementation for QuantizedGemmaNorm """
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        """
        Update the LETQuantizedGemmaNorm params with prev_scale/foll_scale
        """
        weight = self.weight
        bias = self.bias
        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            weight = weight / prev_scale
            bias = bias / prev_scale

        return {'weight': weight, 'bias': bias}

    def _get_source_quant_module(self):
        return QuantizedGemmaNorm(self.weight.shape)

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # No need to call compute_encodings here as we don't want to calibrate
                # min-max when doing LET blockwise training.
                return super().__call__(*args, **kwargs)

    def _fold(self):
        # Do not want bias to be copied.
        param = self._update_parameters()
        with torch.no_grad():
            self.weight.copy_(param['weight'])
