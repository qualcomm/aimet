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
# pylint: disable=redefined-builtin

""" Utilities to use deepspeed """
import contextlib
import torch
from deepspeed import comm as dist
from deepspeed.utils import safe_set_full_fp32_param


try:
    from deepspeed.runtime.zero import ZeroParamStatus, GatheredParameters


    class SafeGatheredParameters(GatheredParameters):
        """
        Ensure the synchronization of parameters in the bucket by overriding the `GatheredParameters`.
        """
        def __exit__(self, *exc):
            if not self.enabled:
                return
            if self.src_rank is None:
                self.params[0].partition(param_list=self.params, has_been_updated=False)
                return

            handles = [dist.broadcast(p.data, self.src_rank, group=p.ds_process_group, async_op=True) for p in self.params]
            for h in handles:
                h.wait()
            for param in self.params:
                safe_set_full_fp32_param(param, param.detach())
            self.params[0].partition(param_list=self.params, has_been_updated=True)


    def gathered_parameters(params, *args, **kwargs):
        """
        Shallow wrapper around ref:`GatheredParameters`.
        Unlike ref:`GatheredParameters`, this function can be also called
        with parameters that are already all-gathered by deepspeed zero3 or zero-offload runtime.
        """
        params = [
            p for p in params
            # Ignore if the parameter is already all-gathered.
            # deepspeed.zero.runtime.GatheredParameters assumes all the parameters to be "NOT_AVAILABLE"
            # and can fail if some of them were already "AVAILABLE".
            if getattr(p, 'ds_status', None) == ZeroParamStatus.NOT_AVAILABLE
        ]
        return SafeGatheredParameters(params, *args, **kwargs)

    @contextlib.contextmanager
    def _do_patch_dummy_parameters(module):
        orig_data = {
            name: p.data for name, p in module.named_parameters(recurse=False)
            # Ignore if the parameter is already all-gathered.
            # deepspeed.zero.runtime.GatheredParameters assumes all the parameters to be "NOT_AVAILABLE"
            # and can fail if some of them were already "AVAILABLE".
            if getattr(p, 'ds_status', None) == ZeroParamStatus.NOT_AVAILABLE
        }

        try:
            for name in orig_data:
                param = getattr(module, name)
                zeros = torch.zeros(size=param.ds_shape,
                                    dtype=param.dtype,
                                    device=param.device,
                                    requires_grad=param.requires_grad)
                param.data = zeros
            yield
        finally:
            for name, data in orig_data.items():
                getattr(module, name).data = data

except ImportError:
    def gathered_parameters(*args, **kwargs): # pylint: disable=unused-argument
        """ Dummy placeholder in case deepspeed doesn't exist """
        return contextlib.nullcontext()

    def _do_patch_dummy_parameters(module): # pylint: disable=unused-argument
        """ Dummy placeholder in case deepspeed doesn't exist """
        return contextlib.nullcontext()


_ds_ctx = {}

def _all_gather(module, _):
    ctx = gathered_parameters(module.parameters(recurse=False))
    ctx.__enter__()
    _ds_ctx[module] = ctx

def _patch_dummy_parameters(module, _):
    ctx = _do_patch_dummy_parameters(module)
    ctx.__enter__() # pylint: disable=no-member
    _ds_ctx[module] = ctx

def _restore(module, *_):
    ctx = _ds_ctx.pop(module, None)
    if ctx:
        ctx.__exit__(None, None, None)


@contextlib.contextmanager
def _register_zero3_forward_hooks(model: torch.nn.Module, use_dummy_params: bool):
    handles = []

    # Temporarily materialize parameters to make forward runnable
    materialize_parameters = _patch_dummy_parameters if use_dummy_params else _all_gather
    try:
        for module in model.modules():
            handle = module.register_forward_pre_hook(materialize_parameters)
            handles.append(handle)
            handle = module.register_forward_hook(_restore)
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.remove()


def _shallow_copy(dict_like):
    """
    Create a shallow copy for dict-like objects with variables
    """
    copy = dict_like.__new__(type(dict_like))
    copy.update(dict_like.items())
    if hasattr(dict_like, '__dict__'):
        copy.__dict__.update(dict_like.__dict__)

    return copy


def _get_shape(tensor: torch.Tensor):
    return getattr(tensor, 'ds_shape', tensor.shape)
