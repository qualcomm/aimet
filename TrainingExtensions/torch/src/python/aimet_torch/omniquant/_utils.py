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
""" Helping functions for Omniquant. """

from aimet_torch.v2.nn import (
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedConv2d,
)

from aimet_torch.omniquant.module_defns import (
    QuantizedLlamaRMSNorm,
    QuantizedGemmaNorm,
)
from aimet_torch.omniquant.let_modules import (
    LETModule,
    LETQuantizedLinear,
    LETQuantizedLlamaRMSNorm,
    LETQuantizedLayerNorm,
    LETQuantizedConv2d,
    LETQuantizedGemmaNorm,
)

_SUPPORT_QUANTIZED_MODULES = (QuantizedLinear, QuantizedLayerNorm, QuantizedConv2d, QuantizedLlamaRMSNorm, QuantizedGemmaNorm)

def get_let_module(mdl):
    """ Return corresponding LETQuantized module for different QuantizedLinear """
    match mdl:
        case QuantizedLinear():
            return LETQuantizedLinear
        case QuantizedLayerNorm():
            return LETQuantizedLayerNorm
        case QuantizedConv2d():
            return LETQuantizedConv2d
        case QuantizedLlamaRMSNorm():
            return LETQuantizedLlamaRMSNorm
        case QuantizedGemmaNorm():
            return LETQuantizedGemmaNorm
        case _:
            assert False, "Let Quantized module is not implemented"

def _convert_sim_to_letsim(sim):
    """ Convert sim to sim model with LET quantizers inplace. """
    for name, module in sim.model.named_modules():
        if isinstance(module, _SUPPORT_QUANTIZED_MODULES):
            let_module = get_let_module(module)(module)
            parent_module = ".".join(name.split(".")[:-1])
            leaf_module_name = name.split(".")[-1]
            setattr(sim.model.get_submodule(parent_module), leaf_module_name, let_module)


def _convert_letsim_to_sim(sim):
    """ Convert LET sim to original sim model inplace. """
    for name, module in sim.model.named_modules():
        if isinstance(module, LETModule):
            source_quant_module = module.get_source_quant_module()
            parent_module = ".".join(name.split(".")[:-1])
            leaf_module_name = name.split(".")[-1]
            setattr(sim.model.get_submodule(parent_module), leaf_module_name, source_quant_module)
