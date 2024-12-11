# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Common utility functions """
from typing import Callable, Tuple, Any
import functools
import itertools

import torch


def _is_expandable(src_shape: Tuple[int, ...],
                   target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def _is_reducible(src_shape: Tuple[int, ...],
                  target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return _is_expandable(target_shape, src_shape)  # pylint: disable=arguments-out-of-order


def reduce(input: torch.Tensor, shape: Tuple[int, ...], reduce_op: Callable):
    """
    Reduce input into given shape.

    :param input: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    """
    if not _is_reducible(input.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(input.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (
        *itertools.repeat(1, len(input.shape) - len(shape)),
        *shape
    )
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(input.permute(permute_dims).reshape(-1, *shape), dim=0, keepdim=False)


class _ContextManager:
    def __init__(self, action: Callable[[], Any], cleanup: Callable[[], Any]):
        self._action = action
        self._cleanup = cleanup

    def __enter__(self):
        self._action()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


def patch_attr(obj, attr_name, new_attr)-> _ContextManager:
    """
    Temporarily overwrite object attribute
    """
    if isinstance(obj, torch.nn.Module):
        if attr_name in obj._parameters or attr_name in obj._buffers: # pylint: disable=protected-access
            return _patch_param_or_buffer(obj, attr_name, new_attr)

    class _NullAttribute:
        pass

    old_attr = getattr(obj, attr_name, _NullAttribute())
    action = lambda: setattr(obj, attr_name, new_attr)

    def cleanup():
        try:
            delattr(obj, attr_name)
        except AttributeError:
            pass

        if not hasattr(obj, attr_name) and not isinstance(old_attr, _NullAttribute):
            setattr(obj, attr_name, old_attr)

    return _ContextManager(action, cleanup)


def _patch_param_or_buffer(module: torch.nn.Module,
                           param_or_buffer_name: str,
                           new_param_or_buffer: torch.Tensor):
    """
    Temporarily substitute the reference to the a parameter with the quantized parameter.
    Under the scope of this function, ``getattr(module, param_or_buffer_name)`` will return
    ``new_param_or_buffer`` instead of the original parameter.

    :param module: Module that owns the parameter
    :param param_or_buffer_name: Name of the parameter
    :param new_param_or_buffer: New parameter to replace the original parameter
    """
    # pylint: disable=protected-access

    orig_param_or_buffer = getattr(module, param_or_buffer_name)
    if orig_param_or_buffer is not None:
        assert new_param_or_buffer.shape == orig_param_or_buffer.shape

    # Modify module.__dict__.
    # module.__dict__ is the primary lookup table which has higher priority than __getattr__ method.
    # Once we overwrite module.__dict__[param_or_buffer_name] with quantized_params,
    # getattr(module, param_or_buffer_name) will return module.__dict__[param_or_buffer_name] directly
    # without falling back to torch.nn.Module's __getattr__ method which returns
    # the original parameter stored in module._parameters or module._buffers.
    action = lambda: module.__dict__.update({param_or_buffer_name: new_param_or_buffer})

    if param_or_buffer_name in module.__dict__:
        # Some non-standard modules (e.g. replicas of torch.nn.DataParallel) store their parameters
        # directly to module.__dict__. In that case, the cleanup function should restore the dict
        # so that module.__dict__[param_or_buffer_name] points back to the original parameter again.
        assert module.__dict__[param_or_buffer_name] is orig_param_or_buffer
        cleanup = lambda: module.__dict__.update({param_or_buffer_name: orig_param_or_buffer})
    else:
        if param_or_buffer_name in module._parameters:
            assert module._parameters[param_or_buffer_name] is orig_param_or_buffer
        elif param_or_buffer_name in module._buffers:
            assert module._buffers[param_or_buffer_name] is orig_param_or_buffer
        else:
            raise RuntimeError(f"'{param_or_buffer_name}' is not a valid name of parameter of buffer of {type(module)}.")

        cleanup = lambda: module.__dict__.pop(param_or_buffer_name)


    return _ContextManager(action, cleanup)


class _StraightThroughEstimator(torch.autograd.Function): # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs): # pylint: disable=arguments-differ
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)

class StatisticsNotFoundError(RuntimeError):
    '''
    Error raised when compute_encodings() is invoked without statistics
    '''


_ENABLE_RECOMPUTE = False


def _set_enable_recompute(mode: bool):
    original_mode = _ENABLE_RECOMPUTE

    def action():
        global _ENABLE_RECOMPUTE # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = mode

    def cleanup():
        global _ENABLE_RECOMPUTE # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = original_mode

    return _ContextManager(action, cleanup)


def is_recompute_enabled():
    """
    Returns True if recomputation for memory saving is enabled; False otherwise.
    """
    return _ENABLE_RECOMPUTE


def enable_recompute():
    """
    Enable recomputation for memory saving.
    """
    return _set_enable_recompute(True)


def no_recompute():
    """
    Disable recomputation for memory saving.
    """
    return _set_enable_recompute(False)


def allow_recompute(fn):
    """
    Allow recomputation of activation of the given function during training
    if recompute is enabled.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_recompute_enabled():
            # Enable activation recompute (a.k.a. activataion checkpointing)
            # to reduce memory footprint of training
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)
    return wrapper

def flatten_nn_module_list(module):
    """
    Flatten nested list of nn.Modules into a flat list
    """
    def flat_iter(mod):
        if isinstance(mod, (list, tuple, torch.nn.ModuleList)):
            for x in mod:
                yield from flat_iter(x)
        else:
            yield mod

    return list(flat_iter(module))


def docstring(doc: str):
    """
    Helper function to attach docstring
    """
    def decorator(fn_or_cls: Callable):
        fn_or_cls.__doc__ = doc
        return fn_or_cls
    return decorator


def _map_qmodule(modules, func):
    # pylint: disable=import-outside-toplevel
    # pylint: disable=protected-access, cyclic-import
    from aimet_torch.v2.nn import BaseQuantizationMixin
    contexts = []
    ctx = _ContextManager(action=lambda: None, cleanup=lambda:[context._cleanup() for context in contexts])

    if isinstance(modules, torch.nn.Module):
        modules = [modules]

    try:
        for module_elem in modules:
            for module in module_elem.modules():
                if isinstance(module, BaseQuantizationMixin):
                    context = func(module)
                    contexts.append(context)
    except Exception:
        ctx._cleanup()
        raise
    else:
        return ctx

def remove_input_quantizers(modules):
    '''
    Removes input quantizers for the modules provided
    '''
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_input_quantizers())

def remove_output_quantizers(modules):
    '''
    Removes output quantizers for the modules provided
    '''
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_output_quantizers())


def remove_param_quantizers(modules):
    '''
    Removes parameter quantizers for the modules provided
    '''
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_param_quantizers())

def remove_activation_quantizers(modules):
    '''
    Removes activation quantizers for the modules provided
    '''
    context_1 = remove_input_quantizers(modules)
    context_2 = remove_output_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(action=lambda: None,
                           cleanup=lambda: (context_1._cleanup(), context_2._cleanup()))

def remove_all_quantizers(modules):
    '''
    Removes all quantizers for the modules provided
    '''
    context_1 = remove_activation_quantizers(modules)
    context_2 = remove_param_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(action=lambda: None,
                           cleanup=lambda: (context_1._cleanup(), context_2._cleanup()))

def has_no_quantizers(module, ignore_params: bool = False) -> bool:
    """
    Helper function to check if a module has any quantizers enabled
    """
    return (all(inp_qtzr is None for inp_qtzr in module.input_quantizers) and
            all(out_qtzr is None for out_qtzr in module.output_quantizers) and
            (ignore_params or all(param_qtzr is None for param_qtzr in module.param_quantizers.values())))

def rgetattr(obj, attr):
    """ Drop in replacement for __getattr__ that can handle dotted attribute strings """
    return functools.reduce(getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    """ Drop in replacement for __setattr__ that can handle dotted attribute strings """
    pre, _, post = attr.rpartition('.')
    pre_obj = rgetattr(obj, pre) if pre else obj
    return setattr(pre_obj, post, val)

def apply_fn_recursively_to_all_elems(fn, container):
    """ Apply fn to all elements in recursively composed container """
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        return [apply_fn_recursively_to_all_elems(fn, elem) for elem in container]
    if isinstance(container, dict):
        return {key: apply_fn_recursively_to_all_elems(fn, elem) for key, elem in container.items()}
    return fn(container)

def flatten_list(container):
    """ Helper function to flatten nested list/tuple into 1D """
    if not container:
        return container
    if not isinstance(container, (list, tuple)):
        return [container]
    if isinstance(container[0], (list, tuple)):
        return flatten_list(container[0]) + flatten_list(container[1:])
    if len(container) == 1:
        return container
    return container[:1] + flatten_list(container[1:])
