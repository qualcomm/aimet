#!/usr/bin/env python3
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
# pylint: skip-file

""" Quantized LoRA layers """

__all__ = ["QuantizedLinear"]

try:
    import peft.tuners.lora.layer as lora
except ImportError:
    lora = None
    QuantizedLinear = None
else:
    import torch
    from torch import nn
    from aimet_torch.v2.nn import QuantizationMixin, custom
    from aimet_torch.v2.nn.true_quant import _dispatch


    class _TensorDict(torch.nn.ParameterDict): # pylint: disable=abstract-method
        def __setitem__(self, key, value):
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            super().__setitem__(key, value.detach())

        def __getitem__(self, key) -> torch.Tensor:
            ret = super().__getitem__(key).detach()
            setattr(ret, '_consumer', key)
            return ret


    @QuantizationMixin.implements(lora.Linear)
    class QuantizedLinear(QuantizationMixin, lora.Linear): # pylint: disable=too-many-ancestors
        """
        Quantized lora.Linear.
        """

        # NOTE: The implementation of this class is tightly dependent on below assumptions
        #   1) LoRA scale (``self.scaling``) will be always multiplied with the output of lora adapters.
        #   2) The scaled output of LoRA adapters will be always added to  the output of the base layer.

        def __quant_init__(self):
            super().__quant_init__()

            # Quantized lora linear itself doesn't need input/output quantizers.
            self.input_quantizers = nn.ModuleList([])
            self.output_quantizers = nn.ModuleList([])

            self.scaling = _TensorDict(self.scaling)
            self.mul = nn.ModuleDict({
                adapter_name: custom.QuantizedMultiply() for adapter_name in self.lora_A.keys()
            })
            self.add = nn.ModuleDict({
                adapter_name: custom.QuantizedAdd() for adapter_name in self.lora_A.keys()
            })

        def _mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Implementation of elemeentwise add which will be dispatched in place of
            torch.Tensor.add and torch.add during forward.

            This function will invoke self.mul (type: QuantizedMultipy) if any of x and y
            is an entry of self.scaling.
            Otherwise, it will fall back to normal torch.Tensor.mul.
            """
            adapter_name = getattr(x, '_consumer', None)

            if adapter_name is None:
                adapter_name = getattr(y, '_consumer', None)

            if adapter_name is not None:
                # `x` or `y` is a scaling factor for adapter `adapter_name`.
                # Dispatch self.mul[adapter_name] in place of regular torch.Tensor.mul
                # so the scaling factor can be observed and quantzied properly
                out = self.mul[adapter_name](x, y)
                setattr(out, '_producer', adapter_name)
            else:
                # `x` or `y` is NOT a scaling factor.
                # Fall back to normal torch.Tensor.mul
                out = x * y

            return out

        def _add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Implementation of elemeentwise add which will be dispatched in place of
            torch.Tensor.add and torch.add during forward.

            This function will invoke self.add (type: QuantizedAdd) if any of x and y
            is the output of a lora adapter scaled by self.scaling.
            Otherwise, it will fall back to normal torch.Tensor.add.
            """
            adapter_name = getattr(x, '_producer', None)

            if adapter_name is None:
                adapter_name = getattr(y, '_producer', None)

            if adapter_name is not None:
                # `x` or `y` is an output of adapter `adapter_name`.
                # Dispatch self.add[adapter_name] in place of regular torch.Tensor.add
                # so the output can be observed and quantzied properly
                out = self.add[adapter_name](x, y)
            else:
                # `x` or `y` is NOT an output of any adapter.
                # Fall back to normal torch.Tensor.add
                out = x + y

            return out

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor: # pylint: disable=arguments-differ
            with _dispatch(torch.Tensor.mul, self._mul), _dispatch(torch.mul, self._mul),\
                    _dispatch(torch.Tensor.add, self._add), _dispatch(torch.add, self._add):
                return super().forward(x, *args, **kwargs)

        def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False): # pylint:disable=arguments-differ
            raise NotImplementedError

        def set_scale(self, adapter, scale):
            raise NotImplementedError

        def scale_layer(self, *args, **kwargs):
            raise NotImplementedError

        def unscale_layer(self, *args, **kwargs):
            raise NotImplementedError

        def merge(self, *args, **kwargs) -> None:
            raise NotImplementedError

        def unmerge(self, *args, **kwargs) -> None:
            raise NotImplementedError

        def get_delta_weight(self, adapter) -> torch.Tensor:
            raise NotImplementedError
