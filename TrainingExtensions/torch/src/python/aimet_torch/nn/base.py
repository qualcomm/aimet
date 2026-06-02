# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Base class of quantized modules"""

import abc
import contextlib
import inspect
import itertools
import threading
import weakref
from typing import Type, List, Dict, Union, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

from aimet_torch.utils import is_vector_encoding
from aimet_torch.quantization.affine.encoding import (
    AffineEncoding,
    GroupedBlockEncoding,
    VectorEncoding,
)
from aimet_torch.quantization.affine import (
    AffineQuantizerBase,
    GroupedBlockQuantizeDequantize,
    QuantizeDequantize,
)
from aimet_torch.quantization.float import FloatEncoding, FloatQuantizeDequantize

from aimet_torch.quantization.tensor import QuantizedTensorBase, DequantizedTensor
from aimet_torch.quantization.base import QuantizerBase
from aimet_torch.utils import (
    patch_attr,
    _ContextManager,
    flatten_nn_module_list,
    reduce,
)
from aimet_torch.deepspeed_utils import SafeGatheredParameters, _shallow_copy


def _no_op(in_tensor):
    return in_tensor


class UnknownModuleError(RuntimeError):
    """
    Exception thrown when an unknown module is encountered
    whose quantized definition isn't registered using @QuantizationMixin.implements().
    """

    module_cls: Type[torch.nn.Module]
    mixin_cls: Type["BaseQuantizationMixin"]
    api_reference_url = (
        "https://qualcomm.github.io/aimet-pages/releases/"
        "latest/apiref/torch/generated/"
        "aimet_torch.nn.QuantizationMixin.html#aimet_torch.nn.QuantizationMixin.implements"
    )

    def __init__(
        self,
        module_cls: Type[torch.nn.Module],
        mixin_cls: Type["BaseQuantizationMixin"],
    ):
        self.module_cls = module_cls
        self.mixin_cls = mixin_cls
        msg = self.generate_err_msg()
        super().__init__(msg)

    def generate_err_msg(self) -> str:
        """Generate error message"""
        module_cls = self.module_cls
        mixin_cls = self.mixin_cls
        code_example = self.generate_code_example()

        return (
            f"The quantized module definition of {module_cls} is not registered. "
            f"Please register the quantized module definition of {module_cls} "
            f"using `@{mixin_cls.__name__}.implements({module_cls.__name__})` decorator.\n\n"
            f"For example:\n\n{code_example}\n\n"
            "If you believe this module need not be quantized, please exclude it from quantization by calling "
            f"`QuantizationMixin.ignore({module_cls.__name__})`.\n\n"
            f"For more details, please refer to the official API reference:\n{self.api_reference_url}"
        )

    def generate_code_example(self) -> str:
        """Generate code example"""
        module_cls = self.module_cls
        mixin_cls = self.mixin_cls

        forward_fn_signature = inspect.signature(module_cls.forward)
        _, *forward_fn_args = list(forward_fn_signature.parameters.values())
        ret_type = forward_fn_signature.return_annotation

        if ret_type == inspect.Parameter.empty:
            # if return annotation is unspecified, assume torch.Tensor as return type
            ret_type = torch.Tensor

        positional_or_keyword_args = [
            arg
            for arg in forward_fn_args
            if arg.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if positional_or_keyword_args != forward_fn_args:
            # Module takes variable number of inputs (*args and/or **kwargs)
            # In this case, only the user knows the proper number of input quantizers
            _declare_input_quantizers = [
                "self.input_quantizers = torch.nn.ModuleList(",
                "    # <TODO: Declare the number of input quantizers here>",
                ")",
            ]
            _quantize_inputs = [
                "# <TODO: Quantize inputs as necessary>\n",
            ]
        else:
            _declare_input_quantizers = [
                f"self.input_quantizers = torch.nn.ModuleList({[None for _ in positional_or_keyword_args]})",
            ]
            _quantize_inputs = []
            for i, arg in enumerate(positional_or_keyword_args):
                _quantize_inputs += [
                    f"if self.input_quantizers[{i}]:",
                    f"    {arg.name} = self.input_quantizers[{i}]({arg.name})\n",
                ]

        if ret_type == torch.Tensor:
            _declare_output_quantizers = [
                "self.output_quantizers = torch.nn.ModuleList([None])",
            ]
            _quantize_outputs = [
                "if self.output_quantizers[0]:",
                "    ret = self.output_quantizers[0](ret)\n",
            ]
        else:
            _declare_output_quantizers = [
                "self.output_quantizers = torch.nn.ModuleList(",
                "    # <TODO: Declare the number of output quantizers here>",
                ")",
            ]
            _quantize_outputs = [
                "# <TODO: Quantize `ret` as necessary>\n",
            ]

        def format_arg(arg: inspect.Parameter):
            if arg.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return arg.name
            if arg.kind == inspect.Parameter.VAR_POSITIONAL:
                return f"*{arg.name}"
            if arg.kind == inspect.Parameter.KEYWORD_ONLY:
                return f"{arg.name}={arg.name}"
            if arg.kind == inspect.Parameter.VAR_KEYWORD:
                return f"**{arg.name}"
            raise RuntimeError

        return "\n".join(
            [
                f"@{mixin_cls.__name__}.implements({module_cls.__name__})",
                f"class Quantized{module_cls.__name__}({mixin_cls.__name__}, {module_cls.__name__}):",
                "    def __quant_init__(self):",
                "        super().__quant_init__()",
                "",
                "        # Declare the number of input/output quantizers",
                *(f"        {line}" for line in _declare_input_quantizers),
                *(f"        {line}" for line in _declare_output_quantizers),
                "",
                f"    def forward{forward_fn_signature}:",
                "        # Quantize input tensors",
                *(f"        {line}" for line in _quantize_inputs),
                "        # Run forward with quantized inputs and parameters",
                "        with self._patch_quantized_parameters():",
                f"            ret = super().forward({', '.join([format_arg(arg) for arg in forward_fn_args])})",
                "",
                "        # Quantize output tensors",
                *(f"        {line}" for line in _quantize_outputs),
                "        return ret",
            ]
        )


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

    _ignore_unknown_modules: bool = False
    _ignored_module_types = set()

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
        self.param_quantizers = nn.ModuleDict(
            {name: None for name, _ in self.named_parameters(recurse=False)}
        )
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
        # Early exit for stateless modules.
        # This helps mitigate dynamo tracing problems during torch.export.export
        if not any(self.param_quantizers.values()):
            return contextlib.nullcontext()

        stack = contextlib.ExitStack()
        for param_name, param_quantizer in self.param_quantizers.items():
            orig_param = getattr(self, param_name)

            if (
                isinstance(orig_param, QuantizedTensorBase)
                and torch.onnx.is_in_onnx_export()
            ):
                # Quantize-dequantize an already-quantized tensor in a possibly duplicate fashion.
                # If duplicate, the duplicate back-to-back QDQs will be removed by the graph pass
                # within aimet_torch.onnx.export
                orig_param = orig_param.encoding.quantize_dequantize(orig_param)

            if param_quantizer and param_quantizer.is_initialized():
                quantized_param = param_quantizer(orig_param)
                ctx = patch_attr(self, param_name, quantized_param)
                stack.enter_context(ctx)

        return stack

    # NOTE: Exclude from torch.compile as there was a bug observed
    # when trying to compile this function with autocast enabled.
    # TODO (kyunggeu): Triage this bug and fix it or file issue to PyTorch
    @torch.compiler.disable
    def _compute_param_encodings(self, overwrite: bool):
        """
        :param bool overwrite: If True, the quantizers that are already initialized will also recompute encodings.
            Otherwise, only the uninitialized quantizers will compute encodings.
        """
        params = {}

        for param_name, param_quantizer in self.param_quantizers.items():
            if not param_quantizer:
                continue

            if not param_quantizer._allow_overwrite:  # pylint: disable=protected-access
                continue

            if not param_quantizer.is_initialized() or overwrite:
                param = getattr(self, param_name)
                if param is not None:
                    params[param_quantizer] = param

        if not params:
            return

        with SafeGatheredParameters(params.values()):
            for param_qtzr, param in params.items():
                with (
                    patch_attr(param_qtzr, "forward", _no_op),
                    param_qtzr.compute_encodings(),
                ):
                    _ = param_qtzr(param)

                if param_qtzr.encoding_analyzer is not None:
                    param_qtzr.encoding_analyzer.reset_stats()

    def compute_param_encodings(self):
        """Compute encodings of parameter quantizers"""
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

                if not quantizer._allow_overwrite:  # pylint: disable=protected-access
                    continue

                # Set input/output quantizers into pass-through mode during compute_encodings
                # NOTE: This behavior is for backawrd-compatibility with V1 quantsim.
                stack.enter_context(patch_attr(quantizer, "forward", _no_op))

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
    def ignore(cls, module_cls):
        """
        Exclude given module type from quantization
        """
        if not issubclass(module_cls, torch.nn.Module):
            raise RuntimeError

        cls._ignored_module_types.add(module_cls)

        qcls = cls.cls_to_qcls.pop(module_cls, None)
        if qcls:
            cls.qcls_to_cls.pop(qcls, None)

        return module_cls

    @classmethod
    def ignore_unknown_modules(cls, ignore: bool = True):
        """
        Exclude all unkown module types from quantization
        """
        cls._ignore_unknown_modules = ignore

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
            from aimet_torch.translation_mapping import aimet_op_to_backend_op_name_map  # pylint:disable = cyclic-import

            backend_op_name = aimet_op_to_backend_op_name_map.get(module_cls, None)
            if backend_op_name:
                aimet_op_to_backend_op_name_map[quantized_cls] = backend_op_name

            return quantized_cls

        return wrapper

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
            if cls._ignore_unknown_modules or module_cls in cls._ignored_module_types:
                return module
            raise UnknownModuleError(module_cls, cls)

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
                input_encodings.append(
                    quantizer.get_encodings().to_qnn_encoding_dict(encoding_version)
                )
            else:
                input_encodings.append(None)
        return input_encodings

    def import_input_encodings(
        self,
        encodings: Mapping[str, Mapping],
        strict: bool,
        partial: bool,
        requires_grad: Optional[bool],
        allow_overwrite: bool,
    ):
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
            if quantizer and not quantizer._allow_overwrite:  # pylint: disable=protected-access
                continue
            encoding = encodings.get(str(i), None)
            if not encoding:
                if not partial:
                    # Dangling quantizers have to be removed when importing non-partial encodings
                    self.input_quantizers[i] = None
                continue
            if quantizer is None:
                if strict:
                    raise RuntimeError(
                        f"Failed to import input encoding at index {i}: no quantizer present."
                    )
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
                output_encodings.append(
                    quantizer.get_encodings().to_qnn_encoding_dict(encoding_version)
                )
            else:
                output_encodings.append(None)
        return output_encodings

    def import_output_encodings(
        self,
        encodings: Mapping[str, Mapping],
        strict: bool,
        partial: bool,
        requires_grad: Optional[bool],
        allow_overwrite: bool,
    ):
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
            if quantizer and not quantizer._allow_overwrite:  # pylint: disable=protected-access
                continue
            encoding = encodings.get(str(i), None)
            if not encoding:
                if not partial:
                    # Dangling quantizers have to be removed when importing non-partial encodings
                    self.output_quantizers[i] = None
                continue
            if quantizer is None:
                if strict:
                    raise RuntimeError(
                        f"Failed to import output encoding at index {i}: no quantizer present."
                    )
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
                encodings[param_name] = quantizer.get_encodings().to_qnn_encoding_dict(
                    encoding_version
                )
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
                e = param.encoding.to_qnn_encoding_dict(encoding_version)  # pylint: disable=protected-access
            else:
                e = None
            encodings[param_name] = e

        return encodings

    def import_param_encodings(
        self,
        encodings: Mapping[str, Mapping],
        strict: bool,
        partial: bool,
        requires_grad: Optional[bool],
        allow_overwrite: bool,
    ):
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
            if quantizer and not quantizer._allow_overwrite:  # pylint: disable=protected-access
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
                rounded_weight.encoding = VectorEncoding(
                    e.scale,
                    e.offset,
                    e.bitwidth,
                    e.signed,
                    e.symmetry,
                    block_size=None,
                    **vector_encoding_properties,
                )
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
                    raise RuntimeError(
                        f"Failed to import encoding for parameter {param_name}: no quantizer present."
                    )
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
        orig_module_cls = self.qcls_to_cls[qtzn_module_cls]

        orig_module = orig_module_cls.__new__(orig_module_cls)
        orig_module.__dict__ = self.__dict__.copy()
        orig_module.__dict__.pop("forward", None)

        # NOTE: We use custom copy function _shallow_copy, which is a superset of dict.copy(),
        #       to circumvent an OOP failure in deepspeed where ZeROOrderedDict.copy()
        #       throws runtime error.
        # TODO: Revert this back to `module._parameters.copy()` once the OOP violation is
        #       fixed in deepspeed
        orig_module._parameters = _shallow_copy(self._parameters)
        orig_module._buffers = self._buffers.copy()
        orig_module._modules = self._modules.copy()
        del orig_module._modules["input_quantizers"]
        del orig_module._modules["output_quantizers"]
        del orig_module._modules["param_quantizers"]

        return orig_module

    @torch.no_grad()
    def set_weight_quantizer_to_mxfp4_int8(self, block_size: int = 32):
        """
        Set weight quantizer to MXFP4 quantizer.

        Assumes weights are already quantize-dequantized to MXFP4 format.

        :param block_size: Block size along the last dimension for MXFP4 quantization.
        """
        # 0. If there is no "weight" parameter and/or param_quantizer, exit early
        if not hasattr(self, "weight") or self.weight is None:
            return
        if (
            "weight" not in self.param_quantizers
            or self.param_quantizers["weight"] is None
        ):
            return

        if not isinstance(self, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            raise RuntimeError(
                "MxFP4 is only implemented in QAIRT for nn.Linear and nn.Conv2d modules"
            )

        weight = self.weight

        # 1. Create a new float quantizer set to e2m1
        # pylint: disable=import-outside-toplevel
        from aimet_torch.quantsim.config_utils import (
            _get_block_size_array_for_module,
            _get_weight_quantizer_shape_from_block_size,
        )

        block_shape = _get_block_size_array_for_module(self, block_size)
        quantizer_shape = _get_weight_quantizer_shape_from_block_size(self, block_shape)

        if quantizer_shape is None:
            error_str = (
                f"Layer has shape {weight.shape} which does not align with block shape {block_shape}. "
                f"Each dimension's shape must divide evenly with the corresponding dimension in block shape."
            )
            raise RuntimeError(error_str)

        e2m1_qdq = FloatQuantizeDequantize(
            exponent_bits=2,
            mantissa_bits=1,
            finite=True,
            unsigned_zero=False,
            shape=quantizer_shape,
            block_size=block_shape,
        )

        # 2. Derive e8m0 scales from the weight values
        weight_max = reduce(
            weight.abs(),
            shape=quantizer_shape,
            block_size=block_shape,
            reduce_op=torch.amax,
        )

        # This is not intuitive but OCP microscaling spec suggests the following derivation
        # floor(log2(weight_max / 4))
        # Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        weight_max = weight_max / 4

        # Calculate the appropriate power of 2 for E8M0 scaling (vectorized)
        nonzero_mask = weight_max != 0
        safe_weight_max = torch.where(
            nonzero_mask, weight_max, torch.ones_like(weight_max)
        )
        exponent = torch.floor(torch.log2(safe_weight_max))
        block_scales = torch.where(
            nonzero_mask, torch.pow(2, exponent), torch.ones_like(weight_max)
        )

        # 3. We will QDQ the weight tensor so we end up with a dequantized tensor that holds the e2m1 scales.
        # There is no API to set the scale in FloatQuantizeDequantize, but we can set the quantization range to be [0, block_scales * 6] to
        # achieve the same effect since e2m1 format has 6.0 as the max representable value.
        e2m1_qdq.maxval = block_scales * 6
        dequantized_weight = e2m1_qdq(weight)

        # Todo: We can  perhaps add a check here to make sure that QDQ values are the same as the original weight values.

        # 4. Set the weight quantizer to per-channel symmetric INT8
        per_channel_shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
        self.param_quantizers["weight"] = QuantizeDequantize(
            shape=per_channel_shape,
            qmin=-128,
            qmax=127,
            symmetric=True,
        )

        # 5. Replace the weight tensor with the dequantized tensor from step 3, which holds the e2m1 encodings
        self.weight = torch.nn.Parameter(
            dequantized_weight, requires_grad=weight.requires_grad
        )

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
        """Remove all activation quantizers"""
        # pylint: disable=protected-access
        ctx_1 = self._remove_output_quantizers()
        ctx_2 = self._remove_input_quantizers()
        return _ContextManager(
            action=lambda: None, cleanup=lambda: (ctx_1._cleanup(), ctx_2._cleanup())
        )

    def _remove_all_quantizers(self):
        """Remove all quantizers"""
        # pylint: disable=protected-access
        ctx_1 = self._remove_activation_quantizers()
        ctx_2 = self._remove_param_quantizers()
        return _ContextManager(
            action=lambda: None, cleanup=lambda: (ctx_1._cleanup(), ctx_2._cleanup())
        )

    @torch.compiler.disable
    def _create_int32_bias_quantizer(self, input, _):  # pylint: disable=redefined-builtin
        assert hasattr(self, "bias")
        assert isinstance(self.bias, torch.Tensor)

        if isinstance(self.param_quantizers["weight"], GroupedBlockQuantizeDequantize):
            # NOTE: In LPBQ, bias encodings should be derived from per-channel weight scale
            weight_scale = self.param_quantizers["weight"].get_per_channel_scale()
        elif isinstance(self.param_quantizers["weight"], AffineQuantizerBase):
            weight_scale = self.param_quantizers["weight"].get_scale()
        else:
            weight_scale = None

        input_scale = None

        if len(input) == 1:
            (input,) = input
            if isinstance(self.input_quantizers[0], AffineQuantizerBase):
                input_scale = self.input_quantizers[0].get_scale()
            elif isinstance(input, QuantizedTensorBase) and isinstance(
                input.encoding, AffineEncoding
            ):
                input_scale = input.encoding.scale

        try:
            bias_scale = self._derive_bias_scale(input_scale, weight_scale)
        except NotImplementedError:
            bias_scale = None

        bias = self.bias
        qmin = -(2**31)
        qmax = 2**31 - 1

        if bias_scale is not None:
            bias_encoding_shape = bias_scale.shape
        elif weight_scale is not None and weight_scale.shape == ():
            # If weight is per-tensor quantized, bias should be also per-tensor quantized
            bias_encoding_shape = ()
        else:
            bias_encoding_shape = bias.shape

        bias_qtzr = QuantizeDequantize(
            shape=bias_encoding_shape, qmin=qmin, qmax=qmax, symmetric=True
        )
        bias_qtzr.to(dtype=bias.dtype, device=bias.device)
        self.param_quantizers["bias"] = bias_qtzr

        if bias_scale is not None:
            bias_qtzr.set_range(bias_scale * qmin, bias_scale * qmax)

        if not bias_qtzr.is_initialized():
            # Failed to derive bias encodings analytically from input and weight encodings.
            # Fall back to statistical bias encoding calibration.
            # This should be avoided as much as possible
            with bias_qtzr.compute_encodings():
                _ = bias_qtzr(bias)

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        raise NotImplementedError

    def fold_param_quantizers(self):
        """
        Fold parameter quantizers into their associated parameters to accelerate inference.

        Example:

          >>> qlinear = QuantizedLinear(10, 10)
          >>> qlinear.param_quantizers["weight"] = QuantizeDequantize((), -128, 127, symmetric=True)
          >>> type(qlinear.weight)
          <class 'torch.nn.parameter.Parameter'>
          >>> qlinear
          QuantizedLinear(
            in_features=10, out_features=10, bias=True
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
          )
          >>> qlinear.fold_param_quantizers()
          >>> type(qlinear.weight)
          <class 'aimet_torch.quantization.tensor.DequantizedTensor'>
          >>> qlinear
          QuantizedLinear(
            in_features=10, out_features=10, bias=True
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
          )
        """
        return self._fold_param_quantizers()

    def _fold_param_quantizers(self):
        self._compute_param_encodings(overwrite=False)

        for param_name, param_qtzr in self.param_quantizers.items():
            if not param_qtzr:
                continue

            param = getattr(self, param_name)

            with torch.no_grad():
                qdq_param = param_qtzr(param).dequantize()

            setattr(
                self,
                param_name,
                torch.nn.Parameter(qdq_param, requires_grad=param.requires_grad),
            )
            self.param_quantizers[param_name] = None

    @contextlib.contextmanager
    def _unfold_param_quantizers(self):
        """
        Re-instantiate param quantizers for ease of export
        """
        unfolded_param_encodings = {}

        try:
            for param_name, qdq_param in self.named_parameters(recurse=False):
                if not isinstance(qdq_param, DequantizedTensor):
                    continue

                if qdq_param.encoding is None:
                    continue

                if self.param_quantizers[param_name] is not None:
                    continue

                if isinstance(qdq_param.encoding, GroupedBlockEncoding):
                    param_qtzr = GroupedBlockQuantizeDequantize.from_encodings(
                        qdq_param.encoding
                    )
                elif isinstance(qdq_param.encoding, AffineEncoding):
                    param_qtzr = QuantizeDequantize.from_encodings(qdq_param.encoding)
                elif isinstance(qdq_param.encoding, FloatEncoding):
                    param_qtzr = FloatQuantizeDequantize.from_encodings(
                        qdq_param.encoding
                    )
                else:
                    raise ValueError

                unfolded_param_encodings[param_name] = qdq_param.encoding
                param = qdq_param.as_subclass(torch.Tensor)
                setattr(
                    self,
                    param_name,
                    torch.nn.Parameter(param, requires_grad=param.requires_grad),
                )
                self.param_quantizers[param_name] = param_qtzr

            yield
        finally:
            for param_name, orig_encoding in unfolded_param_encodings.items():
                self.param_quantizers[param_name] = None
                param = getattr(self, param_name)
                param = param.as_subclass(DequantizedTensor)
                param.encoding = orig_encoding
                setattr(self, param_name, torch.nn.Parameter(param))

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        """
        Returns true if the module is traceable by torch dynamo
        along with the reason if not traceable
        """
        return True, None

    @classmethod
    def _supports_dynamic_input_count(cls) -> bool:
        """
        Returns true if the module takes a dynamic number of inputs
        """
        return False


def _remove_quantizers(quantizers, keys, _registry=threading.local()):
    registry = _registry.__dict__.setdefault("snapshots", {})
    lid = id(quantizers)

    if lid in registry:
        stored_ref, _ = registry[lid]
        if stored_ref() is quantizers:
            # Same live container seen again - genuine tied-list case.
            return _ContextManager(action=lambda: None, cleanup=lambda: None)
        # Stale entry: the original container was GC'd and its address was reused.
        registry.pop(lid)

    orig = {key: quantizers[key] for key in keys}
    registry[lid] = (weakref.ref(quantizers), orig)

    def restore():
        registry.pop(lid, None)
        for key, q in orig.items():
            quantizers[key] = q

    ctx = _ContextManager(action=lambda: None, cleanup=restore)
    try:
        for key in keys:
            quantizers[key] = None
    except Exception:
        ctx._cleanup()  # pylint: disable=protected-access
        raise

    return ctx
