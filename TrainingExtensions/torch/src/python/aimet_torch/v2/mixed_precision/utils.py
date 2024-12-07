# /usr/bin/env python
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
"""Utilities to achieve mixed precision"""
# pylint: disable=logging-fstring-interpolation

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, List, Literal, Optional, Union

import torch

from aimet_common.defs import QuantizationDataType
from aimet_common.utils import AimetLogger
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.float.quantizer import FloatQuantizeDequantize
from aimet_torch.v2.cg_utils import ModuleProduct

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SupportedDType = Literal['int16', 'int8', 'int4', 'fp16']

@dataclass
class Precision:
    """ Internal data structure to represent quantization data type and bitwidth """
    data_type: QuantizationDataType
    bitwidth: int

    def __lt__(self, other):
        if self == other:
            return False
        if self.bitwidth != other.bitwidth:
            return self.bitwidth < other.bitwidth
        return self.data_type == QuantizationDataType.int and other.data_type != QuantizationDataType.int

    def __repr__(self):
        return f'{self.data_type.name}{self.bitwidth}'


TranslateUserDtypes = {
    'int16': Precision(QuantizationDataType.int, 16),
    'int8': Precision(QuantizationDataType.int, 8),
    'int4': Precision(QuantizationDataType.int, 4),
    'fp16': Precision(QuantizationDataType.float, 16),
}


@dataclass
class MpRequest:
    """ Internal data structure to save the request to act upon"""
    id: int = None  # original request ID
    input_candidates: List[Optional[Precision]] = None
    output_candidates: List[Optional[Precision]] = None
    param_candidate: Dict[str, Optional[Precision]] = None

    def __eq__(self, other):
        return self.id == other.id and self.is_same_precision(other)

    def fuse(self, other):
        """ Function to fuse two MpRequest objects, defaulting to self, and filling in None fields from other."""
        if not self:
            return other
        if not other:
            return self
        if not isinstance(self, MpRequest):
            raise NotImplementedError("Cannot add MpRequest object to non-MpRequest object.")

        fused_request = MpRequest()

        # We want to always defer to the newer request when there is a conflict
        newer_request = self if self.id > other.id else other
        older_request = other if newer_request is self else self

        # TODO: how do we handle id? defaulting to latest id
        fused_request.id = newer_request.id

        def fuse_lists(old_list, new_list):
            if old_list is None:
                return new_list
            if new_list is None:
                return old_list
            if len(new_list) != len(old_list):
                raise RuntimeError("Cannot combine two MpRequest objects with different number of candidates.")
            return [new if new else old for new, old in zip(new_list, old_list)]

        fused_request.input_candidates = fuse_lists(older_request.input_candidates, newer_request.input_candidates)
        fused_request.output_candidates = fuse_lists(older_request.output_candidates, newer_request.output_candidates)

        if older_request.param_candidate is None:
            fused_request.param_candidate = newer_request.param_candidate
        if newer_request.param_candidate is None:
            fused_request.param_candidate = older_request.param_candidate
        else:
            # Take param_candidate from RHS if possible, else default to LHS
            fused_request.param_candidate = older_request.param_candidate.update(newer_request.param_candidate)

        return fused_request

    def is_same_precision(self, other):
        """ Compare the precision between self and other"""
        return ((self.input_candidates == other.input_candidates) and
                (self.output_candidates == other.output_candidates) and
                (self.param_candidate == other.param_candidate))

class RequestType(Enum):
    """Enum to represent the type of request made by the user"""
    set_precision_by_module = 1
    set_precision_by_module_type = 2
    set_model_input_precision = 3
    set_model_output_precision = 4

@dataclass
class UserRequest:
    """ Data structure to store user requests"""
    request_type: RequestType
    module: Union[torch.nn.Module, Type, ModuleProduct, None] = None
    activation: Union[List[SupportedDType], SupportedDType, None] = None
    param: Optional[Dict[str, SupportedDType]] = None

def _is_qtzr_higher_precision_than_candidate(qtzr: BaseQuantizationMixin, candidate: Precision) -> bool:
    """ Helper function to determine if qtzr is higher precision than candidate """
    qtzr_dtype = QuantizationDataType.float if isinstance(qtzr, FloatQuantizeDequantize) else QuantizationDataType.int
    generated_candidate = Precision(qtzr_dtype, qtzr.bitwidth)
    return generated_candidate > candidate

def broadcast_tuples(inp_a, inp_b):
    """ Broadcast inp_a to match inp_b shape if possible, or raise RuntimeError """
    if not isinstance(inp_a, (tuple, list)) and not isinstance(inp_b, (tuple, list)):
        return inp_a

    if not isinstance(inp_a, (tuple, list)) and isinstance(inp_b, (tuple, list)):
        return tuple(broadcast_tuples(inp_a, inp_b_elem) for inp_b_elem in inp_b)

    if isinstance(inp_a, (tuple, list)) and isinstance(inp_b, (tuple, list)):
        if len(inp_a) == len(inp_b):
            return tuple(broadcast_tuples(inp_a_elem, inp_b_elem) for inp_a_elem, inp_b_elem in \
                         zip(inp_a, inp_b))

    raise RuntimeError("Incompatible tuple sizes.")
