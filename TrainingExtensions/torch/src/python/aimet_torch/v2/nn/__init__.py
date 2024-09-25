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

# pylint: disable=missing-docstring
import contextlib
import torch
from aimet_torch.v2.deepspeed_utils import _register_zero3_forward_hooks
from .fake_quant import *  # pylint: disable=import-error
from .true_quant import *  # pylint: disable=import-error
from .base import * # pylint: disable=import-error
from .modules import custom # pylint: disable=import-error


@contextlib.contextmanager
def compute_encodings(model: torch.nn.Module):
    """
    Compute encodings of all quantized modules in the model

    .. warning::
        Encodings of the quantizers loaded with :ref:`QuantizationSimModel.load_encodings`
        with ``allow_overwrite=False`` will be kept unchanged.
    """
    with _register_zero3_forward_hooks(model, use_dummy_params=False),\
            contextlib.ExitStack() as stack:
        for module in model.modules():
            if isinstance(module, BaseQuantizationMixin): # pylint: disable=undefined-variable
                ctx = module.compute_encodings()
                stack.enter_context(ctx)

        yield


def compute_param_encodings(model: torch.nn.Module):
    """
    Compute encodings of all parameter quantizers in the model

    .. warning::
        Encodings of the quantizers loaded with :ref:`QuantizationSimModel.load_encodings`
        with ``allow_overwrite=False`` will be kept unchanged.
    """
    for module in model.modules():
        if isinstance(module, BaseQuantizationMixin): # pylint: disable=undefined-variable
            module.compute_param_encodings()
