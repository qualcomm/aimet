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
"""Base class of quantized modules"""

import abc
import contextlib
import inspect
import itertools
from typing import Type, List, Dict, Union, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from aimet_torch.utils import is_vector_encoding
from aimet_torch.v2.quantization.affine.encoding import VectorEncoding, AffineEncoding

from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.utils import (
    patch_attr,
    _ContextManager,
    flatten_nn_module_list,
)
from aimet_torch.v2.deepspeed_utils import SafeGatheredParameters, _shallow_copy

def _no_op(in_tensor):
    return in_tensor


class BaseQuantizationMixin(abc.ABC):
    """Mixin that implements quantization on top of regular pytorch modules.

    Attributes:
        input_quantizers (nn.ModuleList): :class:`ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's input tensors
        output_quantizers (nn.ModuleList): :class:`ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's output tensors
        param_quantizers (nn.ModuleDict): :class:`ModuleDict` mapping parameter names to associated :class:`QuantizerBase`
            objects

    """

    input_quantizers: nn.ModuleList
    output_quantizers: nn.ModuleList
    param_quantizers: nn.ModuleDict

    cls_to_qcls: dict
    qcls_to_cls: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__quant_init__()

    def __quant_init__(self):
        """Initializer for quantized module. This method will be invoked right after :meth:`__init__`.

        This method initializes the :attr:`input_quantizers`, :attr:`output_quantizers`, and :attr:`param_quantizers`
        structures to the appropriate sizes based on the number of input tensors, output tensors, and parameters of the
        base :class:`nn.Module` class. All quantizers are initializd to ``None``.

        For custom quantized classes, this method should be overridden to set the appropriate lengths of
        :attr:`input_quantizers` and :attr:`output_quantizers` for the given base class.
        """
        self.param_quantizers = nn.ModuleDict({
            name: None for name, _ in self.named_parameters(recurse=False)
        })
        # Currently assume single input & output
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def __call__(self, *args, **kwargs):
        self._compute_param_encodings(overwrite=False)
        return super().__call__(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward function for quantized module.

        This method will replace the original forward function of the base :class:`nn.Module` class and is
        responsible for computing a quantized version of the base class' forward function using the configuration of
        the layer's :class:`QuantizerBase` objects.
        """
        return super().forward(*args, **kwargs)

    def _patch_quantized_parameters(self):
        stack = contextlib.ExitStack()
        for param_name, param_quantizer in self.param_quantizers.items():
            if param_quantizer and param_quantizer.is_initialized():
                orig_param = getattr(self, param_name)
                quantized_param = param_quantizer(orig_param)
                ctx = patch_attr(self, param_name, quantized_param)
                stack.enter_context(ctx)

        return stack

    def _compute_param_encodings(self, overwrite: bool):
        """
        :param bool overwrite: If True, the quantizers that are already initialized will also recompute encodings.
            Otherwise, only the uninitialized quantizers will compute encodings.
        """
        params = {}

        for param_name, param_quantizer in self.param_quantizers.items():
            if not param_quantizer:
                continue

            if not param_quantizer._allow_overwrite: # pylint: disable=protected-access
                continue

            if not param_quantizer.is_initialized() or overwrite:
                param = getattr(self, param_name)
                if param is not None:
                    params[param_quantizer] = param

        if not params:
            return

        with SafeGatheredParameters(params.values()):
            for param_qtzr, param in params.items():
                with patch_attr(param_qtzr, "forward", _no_op), param_qtzr.compute_encodings():
                    _ = param_qtzr(param)

    def compute_param_encodings(self):
        """ Compute encodings of parameter quantizers """
        self._compute_param_encodings(overwrite=True)

    @contextlib.contextmanager
    def compute_encodings(self):
        """Enters the :meth:`compute_encodings` context for all :class:`QuantizerBase` objects in the layer.

        Inside this context, each quantizer will observe all inputs passed to the quantizer and will compute
        quantization encodings upon exiting the context.

        Example:

            >>> qlinear = QuantizedLinear(10, 10)
            >>> qlinear.output_quantizers[0] = Quantize((), 8, symmetric=False)
            >>> with qlinear.compute_encodings():
            >>>     qlinear(torch.randn(16, 10))
            >>> print(qlinear.output_quantizers[0].is_initialized())
            True

        """
        self._compute_param_encodings(overwrite=True)

        with contextlib.ExitStack() as stack:
            input_quantizers = flatten_nn_module_list(self.input_quantizers)
            output_quantizers = flatten_nn_module_list(self.output_quantizers)

            for quantizer in itertools.chain(input_quantizers, output_quantizers):
                if not isinstance(quantizer, QuantizerBase):
                    continue

                if not quantizer._allow_overwrite: # pylint: disable=protected-access
                    continue

                # Set input/output quantizers into pass-through mode during compute_encodings
                # NOTE: This behavior is for backawrd-compatibility with V1 quantsim.
                stack.enter_context(patch_attr(quantizer, 'forward', _no_op))

                ctx = quantizer.compute_encodings()
                stack.enter_context(ctx)

            yield

    @classmethod
    @abc.abstractmethod
    def wrap(cls, module_cls: Type[nn.Module]):
        """
        Wrap a regular module class into a quantized module class
        """

    @classmethod
    def implements(cls, module_cls):
        """
        Decorator for registering quantized definition of the given base class.
        """

        def wrapper(quantized_cls):
            # pylint: disable=import-outside-toplevel
            cls.cls_to_qcls[module_cls] = quantized_cls
            cls.qcls_to_cls[quantized_cls] = module_cls

            # Update the mapping from torch module to onnx op
            # so v1 connected graph and quantsim configurator can properly handle quantized modules.
            from aimet_torch.onnx_utils import map_torch_types_to_onnx
            onnx_type = map_torch_types_to_onnx.get(module_cls, None)
            if onnx_type:
                map_torch_types_to_onnx[quantized_cls] = onnx_type

            # Update the mapping from torch module to backend op
            # so v1 connected graph and quantsim configurator can properly handle quantized modules.
            # TODO: This unfortunately relies on the **class name** of the module, not the real type
            #       of the module due to the limitation of v1 implementation.
            #       Should redefine `aimet_to_to_backend_op_name_map` as `Dict[Type[Module], str]`
            from aimet_torch.translation_mapping import aimet_op_to_backend_op_name_map # pylint:disable = cyclic-import
            backend_op_name = aimet_op_to_backend_op_name_map.get(module_cls, None)
            if backend_op_name:
                aimet_op_to_backend_op_name_map[quantized_cls] = backend_op_name

            return quantized_cls

        return wrapper

    @classmethod
    def _generate_code_example(cls, module_cls) -> str:
        mixin_clsname = cls.__name__
        module_clsname = module_cls.__name__
        forward_fn_signature = inspect.signature(module_cls.forward)
        _, *forward_fn_args = list(forward_fn_signature.parameters.keys())
        ret_type = forward_fn_signature.return_annotation

        if ret_type == inspect._empty: # pylint: disable=protected-access
            # if return annotation is unspecified, assume torch.Tensor as return type
            ret_type = torch.Tensor

        _quantize_inputs = "\n\n".join([
f"""
        if self.input_quantizers[{i}]:
            {varname} = self.input_quantizers[{i}]({varname})
""".strip('\n')
            for i, varname in enumerate(forward_fn_args)
        ])

        _quantize_outputs = \
"""
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
""".strip('\n') \
        if ret_type == torch.Tensor else \
"""
        # <TODO: Quantize `ret` as necessary>
""".strip('\n')

        return \
f"""
@{mixin_clsname}.implements({module_clsname})
class Quantized{module_clsname}({mixin_clsname}, {module_clsname}):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList({[None for _ in forward_fn_args]})
        self.output_quantizers = torch.nn.ModuleList({
            [None] if ret_type == torch.Tensor else "<TODO: declare the number of output quantizers here>"
        })

    def forward{forward_fn_signature}:
        # Quantize input tensors
{_quantize_inputs}

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward({", ".join(forward_fn_args)})

        # Quantize output tensors
{_quantize_outputs}

        return ret
""".strip('\n')

    @classmethod
    def from_module(cls, module: nn.Module):
        r"""Create an instance of quantized module from a regular module instance.

        The resulting quantized module contains the same attributes and parameters as the original module, but may
        be assigned input, output and parameter quantizers.

        :param module: Floating point module to quantize
        :return: Quantized version of the original module
        """
        # pylint: disable=protected-access
        module_cls = type(module)
        qtzn_module_cls = cls.cls_to_qcls.get(module_cls, None)

        if not qtzn_module_cls:
            api_reference_url = "https://quic.github.io/aimet-pages/releases/" \
                                "latest/torch_docs/api/" \
                                "nn.quantization_mixin.html#aimet_torch.v2.nn.QuantizationMixin.implements"

            raise RuntimeError(
                f'The quantized module definition of {module_cls} is not registered. '
                f'Please register the quantized module definition of {module_cls} '
                f'using `@{cls.__name__}.implements({module_cls.__name__})` decorator.\n\n'
                f"For example:\n\n{cls._generate_code_example(module_cls)}\n\n"
                f"For more details, please refer to the official API reference:\n{api_reference_url}"
            )

        qtzn_module = cls.__new__(qtzn_module_cls)

        qtzn_module.__dict__ = module.__dict__.copy()
        qtzn_module._modules = module._modules.copy()
        # NOTE: We use custom copy function _shallow_copy, which is a superset of dict.copy(),
        #       to circumvent an OOP failure in deepspeed where ZeROOrderedDict.copy()
        #       throws runtime error.
        # TODO: Revert this back to `module._parameters.copy()` once the OOP violation is
        #       fixed in deepspeed
        qtzn_module._parameters = _shallow_copy(module._parameters)
        qtzn_module._buffers = module._buffers.copy()

        qtzn_module.__quant_init__()
        return qtzn_module

    def export_input_encodings(self, encoding_version: str) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """
        input_encodings = []
        for quantizer in flatten_nn_module_list(self.input_quantizers):
            if isinstance(quantizer, QuantizerBase) and quantizer.is_initialized():
                input_encodings.append(quantizer.get_encodings().to_qnn_encoding_dict(encoding_version))
            else:
                input_encodings.append(None)
        return input_encodings

    def import_input_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import input encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }

        :param encodings: Dictionary mapping quantizer index (str) to encoding (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for i, quantizer in enumerate(list(self.input_quantizers)):
            if quantizer and not quantizer._allow_overwrite: # pylint: disable=protected-access
                continue
            encoding = encodings.get(str(i), None)
            if not encoding:
                if not partial:
                    # Dangling quantizers have to be removed when importing non-partial encodings
                    self.input_quantizers[i] = None
                continue
            if quantizer is None:
                if strict:
                    raise RuntimeError(f"Failed to import input encoding at index {i}: no quantizer present.")
                continue
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)

            if requires_grad is not None:
                quantizer.requires_grad_(requires_grad)

            quantizer.allow_overwrite(allow_overwrite)

    def export_output_encodings(self, encoding_version: str) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """
        output_encodings = []
        for quantizer in flatten_nn_module_list(self.output_quantizers):
            if isinstance(quantizer, QuantizerBase) and quantizer.is_initialized():
                output_encodings.append(quantizer.get_encodings().to_qnn_encoding_dict(encoding_version))
            else:
                output_encodings.append(None)
        return output_encodings

    def import_output_encodings(self,
                                encodings: Mapping[str, Mapping],
                                strict: bool,
                                partial: bool,
                                requires_grad: Optional[bool],
                                allow_overwrite: bool):
        """
        Import output encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }

        :param encodings: Dictionary mapping quantizer index (str) to encoding (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for i, quantizer in enumerate(list(self.output_quantizers)):
            if quantizer and not quantizer._allow_overwrite: # pylint: disable=protected-access
                continue
            encoding = encodings.get(str(i), None)
            if not encoding:
                if not partial:
                    # Dangling quantizers have to be removed when importing non-partial encodings
                    self.output_quantizers[i] = None
                continue
            if quantizer is None:
                if strict:
                    raise RuntimeError(f"Failed to import output encoding at index {i}: no quantizer present.")
                continue
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)

            if requires_grad is not None:
                quantizer.requires_grad_(requires_grad)

            quantizer.allow_overwrite(allow_overwrite)

    def export_param_encodings(self, encoding_version: str) -> Dict[str, List[Dict]]:
        """
        Returns a dict of {param name: param encodings}, with each encoding represented as a List of Dicts
        """
        encodings = {}
        for param_name, quantizer in self.param_quantizers.items():
            if isinstance(quantizer, QuantizerBase) and quantizer.is_initialized():
                encodings[param_name] = quantizer.get_encodings().to_qnn_encoding_dict(encoding_version)
            else:
                encodings[param_name] = None

        for param_name, quantizer in self.param_quantizers.items():
            param = getattr(self, param_name)
            if isinstance(quantizer, QuantizerBase):
                # Already taken care of by earlier for loop
                continue
            if isinstance(param, QuantizedTensorBase) and param.encoding is not None:
                # If parameter itself is an already-quantized tensor,
                # export the encoding held by the parameter
                e = param.encoding.to_qnn_encoding_dict(encoding_version) # pylint: disable=protected-access
            else:
                e = None
            encodings[param_name] = e

        return encodings

    def import_param_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import parameter encodings represented in below format:
        {
            'param_name_0': [dict, dict, ...],
            'param_name_1': [dict, dict, ...],
            ...
        }

        :param encodings: Dictionary mapping quantizer parameter name (str) to encodings (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for param_name, quantizer in dict(self.param_quantizers).items():
            if quantizer and not quantizer._allow_overwrite: # pylint: disable=protected-access
                continue
            encoding = encodings.get(param_name, None)

            if is_vector_encoding(encoding):
                # Vector encodings will be held directly by weights, not by quantizers.
                quantizer.set_legacy_encodings(encoding)
                param = getattr(self, param_name)
                rounded_weight = quantizer(param)
                # At this point, rounded_weight is a quantized tensor with affine encoding
                # since quantizer is an affine quantizer
                assert isinstance(rounded_weight, QuantizedTensorBase)
                assert isinstance(rounded_weight.encoding, AffineEncoding)
                e = rounded_weight.encoding
                # Convert affine encoding to vector encoding
                vector_encoding_properties = {
                    "rows_per_block": encoding[0]["rows_per_block"],
                    "cols_per_block": encoding[0]["cols_per_block"],
                    "vector_dim": encoding[0]["vector_dim"],
                    "vector_stride": encoding[0]["vector_stride"],
                    "index_bw": encoding[0]["index_bw"],
                }
                rounded_weight.encoding = VectorEncoding(e.scale,
                                                         e.offset,
                                                         e.bitwidth,
                                                         e.signed,
                                                         e.symmetry,
                                                         block_size=None,
                                                         **vector_encoding_properties)
                setattr(self, param_name, nn.Parameter(rounded_weight))
                # Remove associated quantizer since the weight is holding already-quantized values
                self.param_quantizers[param_name] = None

            if not encoding:
                if not partial:
                    # Dangling quantizers have to be removed when importing non-partial encodings
                    self.param_quantizers[param_name] = None
                continue
            if quantizer is None:
                if strict:
                    raise RuntimeError(f"Failed to import encoding for parameter {param_name}: no quantizer present.")
                continue
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)

            if requires_grad is not None:
                quantizer.requires_grad_(requires_grad)

            quantizer.allow_overwrite(allow_overwrite)

    def get_original_module(self) -> nn.Module:
        """Returns the floating point version of the quantized module

        Returns:
            A floating point module with quantizers removed

        Example:

            >>> qlinear = QuantizedLinear(10, 20, bias=False)
            >>> linear = qlinear.get_original_module()
            >>> linear
            Linear(in_features=10, out_features=20, bias=False)
            >>> linear.weight is qlinear.weight
            True

        """
        # pylint: disable=protected-access

        qtzn_module_cls = type(self)
        orig_module_cls = self.qcls_to_cls.get(qtzn_module_cls)

        orig_module = self.__new__(orig_module_cls)
        orig_module.__dict__ = self.__dict__.copy()
        orig_module.__dict__.pop('forward', None)

        # NOTE: We use custom copy function _shallow_copy, which is a superset of dict.copy(),
        #       to circumvent an OOP failure in deepspeed where ZeROOrderedDict.copy()
        #       throws runtime error.
        # TODO: Revert this back to `module._parameters.copy()` once the OOP violation is
        #       fixed in deepspeed
        orig_module._parameters = _shallow_copy(self._parameters)
        orig_module._buffers = self._buffers.copy()
        orig_module._modules = self._modules.copy()
        del orig_module._modules['input_quantizers']
        del orig_module._modules['output_quantizers']
        del orig_module._modules['param_quantizers']

        return orig_module

    def _remove_input_quantizers(self, indices: Union[int, Iterable[int]] = None):
        """
        Remove input quantizers
        :param indices: Indices of input quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(indices, int):
            indices = [indices]
        elif indices is None:
            indices = list(range(len(self.input_quantizers)))
        return _remove_quantizers(self.input_quantizers, indices)

    def _remove_param_quantizers(self, keys: Union[str, Iterable[str]] = None):
        """
        Remove parameter quantizers
        :param indices: Indices of parameter quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        elif keys is None:
            keys = list(self.param_quantizers.keys())
        return _remove_quantizers(self.param_quantizers, keys)

    def _remove_output_quantizers(self, indices: Union[int, Iterable[int]] = None):
        """
        Remove output quantizers
        :param indices: Indices of input quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(indices, int):
            indices = [indices]
        elif indices is None:
            indices = list(range(len(self.output_quantizers)))
        return _remove_quantizers(self.output_quantizers, indices)

    def _remove_activation_quantizers(self):
        """ Remove all activation quantizers """
        # pylint: disable=protected-access
        ctx_1 = self._remove_output_quantizers()
        ctx_2 = self._remove_input_quantizers()
        return _ContextManager(action=lambda: None,
                               cleanup=lambda: (ctx_1._cleanup(), ctx_2._cleanup()))

    def _remove_all_quantizers(self):
        """ Remove all quantizers """
        # pylint: disable=protected-access
        ctx_1 = self._remove_activation_quantizers()
        ctx_2 = self._remove_param_quantizers()
        return _ContextManager(action=lambda: None,
                               cleanup=lambda: (ctx_1._cleanup(), ctx_2._cleanup()))


def _remove_quantizers(quantizers, keys):
    orig_quantizers = {key: quantizers[key] for key in keys}

    def restore_quantizers():
        for key, orig_qtzr in orig_quantizers.items():
            quantizers[key] = orig_qtzr

    ctx = _ContextManager(action=lambda: None,
                          cleanup=restore_quantizers)

    try:
        for key in keys:
            quantizers[key] = None
    except Exception:
        ctx._cleanup() # pylint: disable=protected-access
        raise
    else:
        return ctx
