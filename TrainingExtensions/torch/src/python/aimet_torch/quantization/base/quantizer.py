# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Quantizer base class"""

import abc
import copy
from collections import OrderedDict
import contextlib
import weakref
from typing import Optional, List, Dict, TYPE_CHECKING, overload

import torch
from torch import nn
from packaging import version
from aimet_torch.common.utils import deprecated
from aimet_torch.quantization.base import EncodingBase
from aimet_torch.quantization.encoding_analyzer import EncodingAnalyzer
from aimet_torch.utils import _torch_compiler_is_compiling

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from aimet_torch.quantization.tensor import QuantizedTensorBase


__all__ = ["QuantizerBase"]


class QuantizerBase(abc.ABC, torch.nn.Module):
    """
    Quantizer base class
    """

    encoding_analyzer: EncodingAnalyzer

    def __init__(self):
        super().__init__()

        # param_name -> (weakref of initial parameter, version info of the initial parameter)
        # This info will be used for judging whether the current parameter has ever been
        # initialized after it was instantiated.
        self._initial_parameters = OrderedDict()
        self._is_overwrite_allowed: dict[str, bool] = OrderedDict()

    def forward(self, input: torch.Tensor) -> "QuantizedTensorBase":  # pylint: disable=redefined-builtin, useless-parent-delegation
        """
        Quantize the input tensor

        Args:
            input (torch.Tensor): Input tensor to quantize
        """
        # Call parent's forward to throw NotImplementedError
        return super().forward(input)

    @abc.abstractmethod
    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        """

    @abc.abstractmethod
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """

    @abc.abstractmethod
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings.
        """

    @abc.abstractmethod
    def get_encodings(self) -> Optional[EncodingBase]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """

    @deprecated(f"Use {get_encodings.__qualname__} instead")
    def get_encoding(self) -> Optional[EncodingBase]:
        """
        Alias of get_encodings
        """
        return self.get_encodings()

    def set_encodings(self, encodings: EncodingBase):
        """
        Set the quantizer's encodings
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_encodings(cls, encodings: EncodingBase) -> "QuantizerBase":
        """
        Create quantizer object from encoding object
        """

    def register_quantization_parameter(self, name: str, param: Optional[nn.Parameter]):
        """
        Register quantization parameter.
        """
        # pylint: disable=protected-access

        self.register_parameter(name, param)
        if param is not None:
            self._initial_parameters[name] = (weakref.ref(param), param._version)

    def is_initialized(self, name: str | None = None) -> bool:
        """
        Returns true if the quantization parameters are initialized.

        Args:
            name (str, optional): Name of the parameter to check. If None, check all parameters.
        """

        if name is None:
            named_parameters = list(self.named_parameters())
        else:
            named_parameters = [
                (param_name, param)
                for param_name, param in self.named_parameters()
                if param_name == name
            ]

        return all(
            self._is_initialized(param_name, param)
            for param_name, param in named_parameters
        )

    def _is_initialized(
        self, param_name: str, current_param: torch.nn.Parameter
    ) -> bool:
        # pylint: disable=protected-access

        initial_param_weakref, initial_param_version = self._initial_parameters.get(
            param_name, (None, None)
        )
        if initial_param_weakref is None:
            # parameters created using register_parameter need not be initialized
            return True

        initial_param = initial_param_weakref()

        if initial_param is None:
            # The initial parameter object doesn't exist in memory space anymore.
            return True

        # 1. Initial parameter has been replaced with a different object, or
        # 2. The version number has changed
        if current_param is not initial_param:
            return True

        # Tensor version is not available during compile
        if _torch_compiler_is_compiling():
            return True

        return current_param._version != initial_param_version

    def state_dict(self, *args, **kwargs):  # pylint: disable=arguments-differ
        state_dict = super().state_dict(*args, **kwargs)  # pylint: disable=missing-kwoa

        if version.parse(torch.__version__) < version.parse("1.10"):
            # This is for backward compatibility with torch < 1.10
            # which doesn't support get/set_extra_state() hooks
            prefix = kwargs.get("prefix", "")
            state_dict[f"{prefix}extra_state"] = self.get_extra_state()

        if (
            version.parse(torch.__version__) < version.parse("2")
            and torch.onnx.is_in_onnx_export()
        ):
            # During ONNX export in torch 1.x, state_dict cannot contain
            # non-tensor objects due to a bug in torch.onnx.export.
            # Skip adding extra_state because extra_state is unnecessary anyway
            # for ONNX export.
            prefix = kwargs.get("prefix", "")
            state_dict.pop(f"{prefix}_extra_state", None)

        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):  # pylint:disable=arguments-differ
        if "_extra_state" not in state_dict:
            is_initialized = OrderedDict(
                {
                    param_name: torch.tensor(True)
                    for param_name in state_dict
                    if param_name in self._parameters
                }
            )
            state_dict["_extra_state"] = is_initialized

        ret = super().load_state_dict(state_dict, *args, **kwargs)

        if version.parse(torch.__version__) < version.parse("1.10"):
            # This is for backward compatibility with torch < 1.10
            # which doesn't support get/set_extra_state() hooks
            self.set_extra_state(state_dict["_extra_state"])

        return ret

    def get_extra_state(self):
        """
        Get extra state that describes which parameters are initialized.
        """
        if torch.onnx.is_in_onnx_export():
            # Bypass get_extra_state during ONNX export.
            # ONNX export doesn't support non-tensor objects in state_dict
            # Return empty tensor since extra state is unnecessary for ONNX export anyway
            return torch.tensor([])

        return {
            param_name: torch.tensor(self._is_initialized(param_name, param))
            for param_name, param in self.named_parameters()
        }

    @torch.no_grad()
    def set_extra_state(self, state):
        """
        Set extra state that describes which parameters are initialized.
        """
        is_initialized = state
        for param_name, param in self._parameters.items():
            if param_name in is_initialized:
                self.register_quantization_parameter(param_name, param)

                if is_initialized[param_name]:
                    # If the parameter has been already initialized,
                    # artificially increment the parameter version to mark as initialized
                    param.mul_(1.0)

    @torch.no_grad()
    def __deepcopy__(self, memo):
        cls = type(self)
        self_copy = cls.__new__(cls)
        self_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        self_copy.set_extra_state(self.get_extra_state())
        return self_copy

    def __getstate__(self):
        getstate = getattr(super(), "__getstate__", self.__dict__.copy)
        state = getstate()
        state.pop("_initial_parameters")
        state["is_initialized"] = self.get_extra_state()
        return state

    @torch.no_grad()
    def __setstate__(self, state):
        self._initial_parameters = OrderedDict()
        is_initialized = state.pop("is_initialized")
        setstate = getattr(super(), "__setstate__", self.__dict__.update)
        setstate(state)
        self.set_extra_state(is_initialized)

    @overload
    def allow_overwrite(self, mode: bool):
        """Set allow_overwite flag"""

    @overload
    def allow_overwrite(self, **kwargs):
        """Set allow_overwite flag"""

    def allow_overwrite(self, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)

        if mode is not None:
            allow_overwrite = {
                param_name: mode for param_name in self._is_overwrite_allowed.keys()
            }
        else:
            allow_overwrite = kwargs.copy()

        expected_keys = self._is_overwrite_allowed.keys()
        unexpected_keys = allow_overwrite.keys() - expected_keys
        if unexpected_keys:
            unexpected_keys = sorted(list(unexpected_keys))
            expected_keys = sorted(list(expected_keys))
            raise RuntimeError(
                f"'allow_overwrite' expected param names {expected_keys};"
                f" got unexpected parameter names {unexpected_keys}"
            )

        self._is_overwrite_allowed.update(allow_overwrite)

    def is_overwrite_allowed(self, name: str | None = None) -> bool:
        if name is None:
            return any(self._is_overwrite_allowed.values())

        return self._is_overwrite_allowed[name]

    # Define _allow_overwrite getter/setter for backwards compatibility
    @property
    @deprecated(f"Use {is_overwrite_allowed.__qualname__}() function instead")
    def _allow_overwrite(self) -> bool:
        return self.is_overwrite_allowed()

    @_allow_overwrite.setter
    @deprecated(f"Use {allow_overwrite.__qualname__} function instead")
    def _allow_overwrite(self, mode: bool):
        self.allow_overwrite(mode)

    @contextlib.contextmanager
    def _precompute_encodings(self, dtype: torch.dtype | None = None):  # pylint: disable=unused-argument
        yield
