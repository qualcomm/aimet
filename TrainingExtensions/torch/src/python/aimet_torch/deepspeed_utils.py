# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin

"""Utilities to use deepspeed"""

import contextlib
import torch

from aimet_torch.common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


try:
    from deepspeed.runtime.zero import ZeroParamStatus, GatheredParameters
    from deepspeed.utils import safe_set_local_fp32_param

    class SafeGatheredParameters:
        """
        Shallow wrapper around ref:`GatheredParameters`.
        Unlike ref:`GatheredParameters`, this function can be also called
        with parameters that are already all-gathered by deepspeed zero3 or zero-offload runtime.
        Additionally, this function ensure the synchronization of parameters.
        """

        # Keep track of the parameters that are already gathered to avoid double-gathering
        _already_gathered: set[int] = set()

        def __init__(self, params, *args, **kwargs):
            self.params = list(params)
            self.args = args
            self.kwargs = kwargs.copy()
            self._ctx = None

        def __enter__(self):
            params = [
                param
                for param in self.params
                if (ds_id := getattr(param, "ds_id", None)) is not None
                and ds_id not in self._already_gathered
            ]
            self._ctx = GatheredParameters(params, *self.args, **self.kwargs)

            if self._ctx.enabled:
                self._already_gathered |= set(param.ds_id for param in self._ctx.params)

            return self._ctx.__enter__()

        def __exit__(self, *exc):
            assert self._ctx is not None

            ret = self._ctx.__exit__(*exc)

            if not self._ctx.enabled:
                return ret

            self._already_gathered -= set(param.ds_id for param in self._ctx.params)

            if self._ctx.src_rank is not None:
                for param in self._ctx.params:
                    if hasattr(param, "_z3_optimizer"):
                        safe_set_local_fp32_param(param, param.ds_tensor)

            return ret

    @contextlib.contextmanager
    def _do_patch_dummy_parameters(module):
        orig_data = {
            name: p.data
            for name, p in module.named_parameters(recurse=False)
            # Ignore if the parameter is already all-gathered.
            # deepspeed.zero.runtime.GatheredParameters assumes all the parameters to be "NOT_AVAILABLE"
            # and can fail if some of them were already "AVAILABLE".
            if getattr(p, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE
        }

        try:
            for name in orig_data:
                param = getattr(module, name)
                zeros = torch.zeros(
                    size=param.ds_shape,
                    dtype=param.dtype,
                    device=param.device,
                    requires_grad=param.requires_grad,
                )
                param.data = zeros
            yield
        finally:
            for name, data in orig_data.items():
                getattr(module, name).data = data

# pylint: disable=broad-exception-caught
except Exception as e:
    if not isinstance(e, ImportError):
        import traceback
        import io

        f = io.StringIO()
        traceback.print_exc(file=f)
        _logger.warning(  # pylint: disable=logging-fstring-interpolation
            f"Found deepspeed package but failed to import due to {type(e).__name__}.\n\n"
            "Full traceback:\n"
            "==============================================================\n"
            f"{f.getvalue()}"
            "==============================================================\n\n"
        )

    class SafeGatheredParameters(contextlib.nullcontext):
        """Dummy placeholder in case deepspeed doesn't exist"""

        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            super().__init__()

    def _do_patch_dummy_parameters(module):  # pylint: disable=unused-argument
        """Dummy placeholder in case deepspeed doesn't exist"""
        return contextlib.nullcontext()

else:
    from deepspeed import __version__ as deepspeed_version
    from packaging import version

    if version.parse(deepspeed_version) == version.parse("0.19.8"):
        msg = [
            "-----------------------------------------------------------------",
            "  Detected deepspeed 0.19.8.",
            "  Deepspeed 0.19.8 has a known issue with ZeRO3 that causes a crash when used with AIMET QuantizationSimModel.",
            "  If you need to use ZeRO3 with AIMET, please consider switching to a different deepspeed version.",
            "  For more information, https://github.com/deepspeedai/DeepSpeed/issues/8639",
            "-----------------------------------------------------------------",
        ]
        _logger.warning("\n".join(msg))


_ds_ctx = {}


def _all_gather(module, _):
    ctx = SafeGatheredParameters(module.parameters(recurse=False))
    ctx.__enter__()  # pylint: disable=unnecessary-dunder-call
    _ds_ctx[module] = ctx


def _patch_dummy_parameters(module, _):
    ctx = _do_patch_dummy_parameters(module)
    ctx.__enter__()  # pylint: disable=no-member, unnecessary-dunder-call
    _ds_ctx[module] = ctx


def _restore(module, *_):
    ctx = _ds_ctx.pop(module, None)
    if ctx:
        ctx.__exit__(None, None, None)  # pylint: disable=unnecessary-dunder-call


@contextlib.contextmanager
def _register_zero3_forward_hooks(model: torch.nn.Module, use_dummy_params: bool):
    # Temporarily materialize parameters to make forward runnable
    handles = []
    materialize_parameters = (
        _patch_dummy_parameters if use_dummy_params else _all_gather
    )
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
    if hasattr(dict_like, "__dict__"):
        copy.__dict__.update(dict_like.__dict__)

    return copy


def _get_shape(tensor: torch.Tensor):
    return getattr(tensor, "ds_shape", tensor.shape)
