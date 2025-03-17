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
from aimet_torch.v2.nn import (
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedConv2d,
)
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import patch_attr
from aimet_torch.omniquant.module_defns import (
    GemmaRMSNorm,
    LlamaRMSNorm,
    QuantizedLlamaRMSNorm,
    QuantizedGemmaNorm,
)

import torch
import copy
from abc import abstractmethod


# TODO add docstring

# pylint: disable=missing-module-docstring, missing-class-docstring
class LETModule():
    def __init__(self, source: QuantizationMixin):
        self._reset_let_params()
        # TODO in e2e integration decide what happens if some of the quantizers are None/missing
        # For now we assume all 3 values are present, else we throw an error
        for quantizers in ['input_quantizers', 'output_quantizers', 'param_quantizers']:
            src_quant = getattr(source, quantizers)
            assert src_quant, f'{quantizers} should not be none for LETModule'
            setattr(self, quantizers, copy.deepcopy(src_quant))


    # TODO : ananmukh check if prep func can be removed from here
    def _reset_let_params(self):
        self.prev_scale = None
        self.prev_prep_fn = torch.nn.Identity()
        self.foll_scale = None
        self.foll_prep_fn = torch.nn.Identity()

    # pylint: disable=missing-function-docstring
    def get_let_params(self):
        let_params = {
            "prev_scale": self.prev_scale,
            "prev_prep_fn": self.prev_prep_fn,
            "foll_scale": self.foll_scale,
            "foll_prep_fn": self.foll_prep_fn,
        }
        return let_params

    # TODO ananmukh: scale is meant to be a nn.parameter shared between 2 nn.modules. Need to check
    # during end-to-end integration whats the best way to initialize scale
    # pylint: disable=missing-function-docstring, attribute-defined-outside-init
    def register_let_params(self, prev_scale = None, foll_scale = None):
        self.prev_scale = prev_scale
        self.foll_scale = foll_scale

    def fold_let_params(self):
        '''
        Call (usually at the end) to fold the scales into the model params
        '''
        self._fold()
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

# pylint: disable=too-many-ancestors, missing-class-docstring
class LETQuantizedLinear(QuantizedLinear, LETModule):
    def __init__(self, module:QuantizationMixin):
        # TODO pass in all params to ctor
        super().__init__(module.weight.shape[1], module.weight.shape[0], bias=module.bias is not None)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        weight = self.weight
        bias = self.bias

        #TODO: ananmukh check if unsqueeze is needed during end-to-end integration
        # currently system's team has a non scaler activation scale for a linear-linear pair
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            if bias is not None:
                bias = bias / prev_scale
            weight = weight / prev_scale

        if self.foll_scale is not None:
            foll_scale = self.foll_prep_fn(self.foll_scale)
            weight = weight * foll_scale

        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # TODO: ananmukh remove compute_param_encodings() from here
                # call it explicitly in training loop in a later PR
                super().compute_param_encodings()
                return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors, missing-class-docstring
class LETQuantizedConv2d(QuantizedConv2d, LETModule):
    def __init__(self, module:QuantizationMixin):
        # TODO pass in all params to ctor
        super().__init__(module.weight.shape[1], module.weight.shape[0], module.kernel_size, module.stride, module.padding, bias=module.bias is not None)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        weight = self.weight
        bias = self.bias

        #TODO: ananmukh check if unsqueeze is needed during end-to-end interation
        # currently system's team has a non scaler sctivation scale for a linear-linear pair
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            if bias is not None:
                bias = bias / prev_scale
            weight = weight / prev_scale

        if self.foll_scale is not None:
            foll_scale = self.foll_prep_fn(self.foll_scale)
            weight = weight * foll_scale

        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # TODO: ananmukh remove compute_param_encodings() from here
                # call it explicitly in training loop in a later PR
                super().compute_param_encodings()
                return super().__call__(*args, **kwargs)

# pylint: disable=too-many-ancestors, missing-class-docstring
class LETQuantizedLayerNorm(QuantizedLayerNorm, LETModule):
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

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
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                # TODO: ananmukh remove compute_param_encodings() from here
                # call it explicitly in training loop in a later PR
                super().compute_param_encodings()
                return super().__call__(*args, **kwargs)

# pylint: disable=missing-class-docstring
QuantizedLlamaRMSNorm = QuantizationMixin.implements(LlamaRMSNorm)(QuantizedLlamaRMSNorm)
class LETQuantizedLlamaRMSNorm(QuantizedLlamaRMSNorm, LETModule):
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        weight = self.weight
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight = weight / prev_scale

        return {'weight': weight}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            # TODO: ananmukh remove compute_param_encodings() from here
            # call it explicitly in training loop in a later PR
            super().compute_param_encodings()
            return super().__call__(*args, **kwargs)

# pylint: disable=missing-class-docstring
QuantizedGemmaNorm = QuantizationMixin.implements(GemmaRMSNorm)(QuantizedGemmaNorm)
class LETQuantizedGemmaNorm(QuantizedGemmaNorm, LETModule):
    def __init__(self, module:QuantizationMixin):
        super().__init__(module.weight.shape)
        LETModule.__init__(self, module)
        self.load_state_dict(module.state_dict())

    def _update_parameters(self):
        weight = self.weight
        bias = self.bias
        if self.prev_scale is not None:
            prev_scale = self.prev_prep_fn(self.prev_scale)
            weight = weight / prev_scale
            bias = bias / prev_scale

        return {'weight': weight, 'bias': bias}

    def __call__(self, *args, **kwargs):
        params = self._update_parameters()
        with patch_attr(self, 'weight', params['weight']):
            with patch_attr(self, 'bias', params['bias']):
                super().compute_param_encodings()
                return super().__call__(*args, **kwargs)

    def _fold(self):
        # Do not want bias to be copied.
        param = self._update_parameters()
        with torch.no_grad():
            self.weight.copy_(param['weight'])
