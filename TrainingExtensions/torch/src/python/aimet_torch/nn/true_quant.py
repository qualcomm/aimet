# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=too-many-lines, redefined-builtin
"""Quantized modules"""

from packaging import version
import contextlib
import itertools
from inspect import signature
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from typing import Type, Any, Optional, Callable, Set, Mapping, Tuple, Iterable
from weakref import WeakKeyDictionary
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import _mha_shape_check
from torch import Tensor
from torch.overrides import BaseTorchFunctionMode, get_overridable_functions
from torch._VF import (  # pylint: disable=no-name-in-module
    gru as _gru,
    gru_cell as _gru_cell,
    lstm as _lstm,
    lstm_cell as _lstm_cell,
    rnn_relu as _rnn_relu,
    rnn_tanh as _rnn_tanh,
    rnn_relu_cell as _rnn_relu_cell,
    rnn_tanh_cell as _rnn_tanh_cell,
)
from torch.utils._pytree import tree_map

from aimet_torch.quantization.base import QuantizerBase
from aimet_torch.quantization.tensor import QuantizedTensorBase
from aimet_torch.quantization.affine import (
    AffineQuantizerBase,
    AffineEncoding,
    GroupedBlockEncoding,
)
from aimet_torch.common.quantsim import (
    _adjust_weight_scale_against_bias_overflow,
    _adjust_weight_scale_against_scale_underflow,
)
from aimet_torch.utils import (
    patch_attr,
    _ContextManager,
    allow_recompute,
    _torch_compiler_is_exporting,
    _torch_compiler_is_compiling,
)
from .base import BaseQuantizationMixin
from aimet_torch.deepspeed_utils import SafeGatheredParameters


def _quantize_if_applicable(data: Any, quantizer: Optional[QuantizerBase]):
    """
    Quantize data if it is a quantizable type and quantize is not None
    """
    if quantizer and isinstance(data, Tensor) and data.is_floating_point():
        if isinstance(data, QuantizedTensorBase):
            data = data.dequantize()
        return quantizer(data)

    if isinstance(data, QuantizedTensorBase):
        return data.quantize()

    return data


def _dequantize_if_applicable(data: torch.Tensor):
    return data.dequantize() if isinstance(data, QuantizedTensorBase) else data


def _quantize_dequantize_if_applicable(data, quantizer):
    if quantizer and isinstance(data, Tensor) and data.is_floating_point():
        if isinstance(data, QuantizedTensorBase):
            data = data.dequantize()
        data = quantizer(data)

    if isinstance(data, QuantizedTensorBase):
        return data.dequantize()

    return data


_QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS = WeakKeyDictionary()


def _is_computing_encodings(qmodule):
    return _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS.get(qmodule, 0) > 0


def _enter_computing_encodings(qmodule):
    if qmodule not in _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS:
        _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] = 0
    _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] += 1


def _exit_compute_encodings(qmodule):
    assert _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] > 0
    _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] -= 1


class QuantizationMixinMeta(ABCMeta):
    """Sets :meth:`forward` to :meth:`quantized_forward` if only :meth:`quantized_forward` is defined"""

    def __new__(mcs, name, bases, namespace, **kwargs):
        if "quantized_forward" in namespace and "forward" not in namespace:
            warnings.warn(
                "Support for defining `quantized_forward` in place of `forward` method will be deprecated, "
                "please use `forward` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            namespace["forward"] = namespace["quantized_forward"]
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class QuantizationMixin(BaseQuantizationMixin, metaclass=QuantizationMixinMeta):  # pylint: disable=abstract-method
    """Quantization mixin class for torch.nn.Module.

    Specifically, a quantized module will quantize input, output, and parameter tensors with
    its held :class:`QuantizerBase` objects during the :meth:`forward` method and use the inherited :class:`torch.nn.Module`
    forward method to compute the layer operation. If all input, output, and parameter quantizers are ``None``, a
    quantized module will behave exactly the same as its parent :class:`torch.nn.Module`.

    Attributes:
        input_quantizers: :class:`torch.nn.ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's input tensors
        output_quantizers: :class:`torch.nn.ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's output tensors
        param_quantizers: :class:`torch.nn.ModuleDict` mapping parameter names to associated :class:`QuantizerBase`
            objects

    Examples:

        >>> qlinear = QuantizedLinear(in_features=10, out_features=10)
        >>> print(qlinear)
        QuantizedLinear(
          in_features=10, out_features=10, bias=True
          (param_quantizers): ModuleDict(
            (weight): None
            (bias): None
          )
          (input_quantizers): ModuleList(
            (0): None
          )
          (output_quantizers): ModuleList(
            (0): None
          )
        )

    """

    cls_to_qcls = OrderedDict()  # quantized class -> original class
    qcls_to_cls = OrderedDict()  # original class -> quantized class

    _default_kernel: Optional[Callable] = None
    _kernels = WeakKeyDictionary()  # instance -> instance_kernel

    def __nullary__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([])

    def __unary__(self):
        super().__quant_init__()

    def __binary__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])

    def __ternary__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None, None])

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Computes a quantized version of the parent module's forward method.

        The :meth:`forward` method should perform the following logic in order:

            1) Apply existing input quantizers to input tensors
            2) Apply existing param quantizers to the layer's parameters
            3) Call the inherited :class:`torch.nn.Module` forward method with quantized inputs and parameters
            4) Apply existing output quantizers to the outputs of the forward method

        If all input, output, and parameter quantizers are ``None``, this method will behave exactly the same as
        its parent module's forward pass.
        """
        return super().forward(*args, **kwargs)

    @classmethod
    def set_default_kernel(cls, kernel: Callable):
        """Set default kernel for the class.

        In general, this signature will follow the signature of the equivalent :mod:`torch.nn.functional` function,
        but should return a :class:`QuantizedTensor` object and take in the additional keyword argument ``output_encodings``.

        Once set, all instances of cls will call into kernel in the forward pass unless:

            1) The instance is within the :meth:`compute_encodings` context, or
            2) The kernel has been overridden by a :meth:`set_kernel` call

        Args:
            kernel: Callable object to be used as the default kernel by all the instances of this class.


        Example:

            >>> from aimet_torch import quantization as Q
            >>> def int_multiply(a, b, output_encodings=None):
            ...     encodings = [a.encoding, b.encoding, output_encodings]
            ...     if not all(enc.mapping == "affine" for enc in encodings):
            ...             raise NotImplementedError
            ...     q_output = (a.quantized_repr() + a.encoding.offset) * (b.quantized_repr() + b.encoding.offset)
            ...     dq_output = q_output *  (a.encoding.scale * b.encoding.scale)
            ...     return Q.QuantizedTensor(output_encodings.quantize(dq_output), encoding=output_encodings)
            ...
            >>> QuantizedMultiply.set_default_kernel(int_multiply)
            >>> qmult = QuantizedMultiply()
            >>> qmult.get_kernel()
            <function int_multiply at ...>

        """
        cls._default_kernel = kernel

    @classmethod
    def get_default_kernel(cls) -> Optional[Callable]:
        """Return the default kernel of the class

        Returns:
            Default kernel of the class. None if the default kernel is not set.

        """
        return cls._default_kernel

    def set_kernel(self, kernel: Callable):
        """Set kernel for this instance of quantized module.

        In general, this signature will follow the signature of the equivalent :mod:`torch.nn.functional` function,
        but should return a :class:`QuantizedTensor` object and take in the additional keyword argument ``output_encodings``.

        Once set, the layer will call into ``kernel`` in the forward pass unless within the :meth:`compute_encodings`
        context.

        Args:
            kernel: Callable object to be used as the underlying kernel.

        Example:

            >>> from aimet_torch import quantization as Q
            >>> def int_multiply(a, b, output_encodings=None):
            ...     encodings = [a.encoding, b.encoding, output_encodings]
            ...     if not all(enc.mapping == "affine" for enc in encodings):
            ...             raise NotImplementedError
            ...     q_output = (a.quantized_repr() + a.encoding.offset) * (b.quantized_repr() + b.encoding.offset)
            ...     dq_output = q_output *  (a.encoding.scale * b.encoding.scale)
            ...     return Q.QuantizedTensor(output_encodings.quantize(dq_output), encoding=output_encodings)
            ...
            >>> qmult = QuantizedMultiply()
            >>> qmult.set_kernel(int_multiply)

        """
        QuantizationMixin._kernels[self] = kernel

    def get_kernel(self) -> Optional[Callable]:
        """Return the kernel to be used by this instance of quantized module.

        If the current instance does not have any kernel set, it will retrieve the default kernel of the class.

        Returns:
            The kernel to be used by this instance.

        """
        if self in QuantizationMixin._kernels:
            return QuantizationMixin._kernels[self]
        return self.get_default_kernel()

    @contextlib.contextmanager
    def compute_encodings(self):  # pylint: disable=missing-function-docstring
        ctx = _ContextManager(
            action=lambda: _enter_computing_encodings(self),
            cleanup=lambda: _exit_compute_encodings(self),
        )
        with super().compute_encodings(), ctx:
            yield

    def _patch_dequantized_parameters(
        self, param_names: Optional[Iterable[str]] = None
    ) -> _ContextManager:
        # Early exit for stateless modules.
        # This helps mitigate dynamo tracing problems during torch.export.export
        if param_names is None:
            param_names = self.param_quantizers.keys()

        param_quantizers = {name: self.param_quantizers[name] for name in param_names}

        if not any(param_quantizers.values()):
            return contextlib.nullcontext()

        stack = contextlib.ExitStack()
        for param_name, _ in param_quantizers.items():
            qparam = getattr(self, param_name)
            dqparam = _dequantize_if_applicable(qparam)
            ctx = patch_attr(self, param_name, dqparam)
            stack.enter_context(ctx)

        return stack

    @classmethod
    def wrap(cls, module_cls: Type[nn.Module]) -> Type[nn.Module]:
        """
        Wrap a regular module class into a quantized module class
        """
        if not issubclass(module_cls, nn.Module):
            raise ValueError(
                "Expected module_cls to be a subclass of torch.nn.Module. "
                f"Got {module_cls}."
            )
        if module_cls in cls.cls_to_qcls:
            return cls.cls_to_qcls[module_cls]

        quantized_cls_name = f"Quantized{module_cls.__name__}"
        base_classes = (cls, module_cls)
        quantized_cls = type(quantized_cls_name, base_classes, {"__module__": __name__})
        return cls.implements(module_cls)(quantized_cls)

    @classmethod
    def from_module(cls, module: nn.Module):
        r"""Create an instance of quantized module from a regular module instance.

        The resulting quantized module contains the same attributes and parameters as the original module, but may
        be assigned input, output and parameter quantizers.

        :param module: Floating point module to quantize
        :return: Quantized version of the original module

        Example:

            >>> linear = torch.nn.Linear(10, 10)
            >>> quantized_linear = QuantizationMixin.from_module(linear)
            >>> print(quantized_linear.param_quantizers)
            QuantizedLinear(
              in_features=10, out_features=10, bias=True
              (param_quantizers): ModuleDict(
                (weight): None
                (bias): None
              )
              (input_quantizers): ModuleList(
                (0): None
              )
              (output_quantizers): ModuleList(
                (0): None
              )
            )
            >>> print(quantized_linear.weight is linear.weight)
            True
        """
        return super().from_module(module)

    @classmethod
    def ignore(cls, module_cls):
        """
        Exclude given module type from quantization

        .. note::
            This method will exclude `module_cls` from quantization
            even if its quantized module definition is already registered
            with :meth:`implements`.

        Example:

            >>> class MyModule(torch.nn.Module):
            ...     def forward(self, x):
            ...         return x ** 2
            >>> QuantizationMixin.ignore(MyModule)
            >>> model = torch.nn.Sequential(MyModule())
            >>> sim = aimet_torch.QuantizationSimModel(model, torch.randn(10, 10))
            >>> print(sim.model)
            Sequential(
              (0): MyModule()
            )
        """
        return super().ignore(module_cls)

    @classmethod
    def ignore_unknown_modules(cls, ignore: bool = True):
        """
        Exclude all unkown module types from quantization

        Example:

            >>> class MyModule(torch.nn.Module):
            ...     def forward(self, x):
            ...         return x ** 2
            >>> QuantizationMixin.ignore_unknown_modules(True)
            >>> model = torch.nn.Sequential(MyModule())
            >>> sim = aimet_torch.QuantizationSimModel(model, torch.randn(10, 10))
            >>> print(sim.model)
            Sequential(
              (0): MyModule()
            )
        """
        super().ignore_unknown_modules(ignore)

    @classmethod
    def implements(cls, module_cls):
        r"""
        Decorator for registering quantized definition of the given base class.

        Even though AIMET supports quantization of all built-in modules in torch.nn subpackage
        such as ``torch.nn.Conv2d`` or ``torch.nn.Linear`` that AIMET is already aware of,
        :class:`QuantizationSimModel` will throw a runtime error when it encounters custom modules
        defined by the users, asking the users to provide the quantized definition of the custom modules
        that AIMET doesn't know of.

        To declare the quantized definition of a module, :class:`QuantizationSimModel` requires you
        to define a subclass of your module decorated with :meth:`implements`,
        in which you will implement ``__quant_init__`` and ``forward`` methods.

        As an example, given a custom module as below::

            class MaskedAdd(torch.nn.Module):
               def forward(self, input: torch.Tensor, mask: torch.Tensor, value: torch.Tensor):
                   return input + mask * value

        its quantized definition should be declared before creating :class:`QuantizationSimModel`, typically as below::


            @QuantizationMixin.implements(MaskedAdd)
            class QuantizedMaskedAdd(QuantizationMixin, MaskedAdd):
                # The quantized definition of MaskedAdd should be a subclass of
                # QuantizationMixin and MaskedAdd (Order matters!)
                def __quant_init__(self):
                    super().__quant_init__()

                    # Declare the number of input/output quantizers
                    self.input_quantizers = torch.nn.ModuleList([None, None, None])
                    self.output_quantizers = torch.nn.ModuleList([None])

               def forward(self, input: torch.Tensor, mask: torch.Tensor, value: torch.Tensor):
                   input_qtzr  = self.input_quantizers[0]
                   _           = self.input_quantizers[1] # I don't want to quantize the boolean masks!
                   value_qtzr  = self.input_quantizers[2]
                   output_qtzr = self.output_quantizers[0]

                   if input_qtzr is not None:
                       input = input_qtzr(input)

                   if value_qtzr is not None:
                       value = value_qtzr(value)

                   output = super().forward(input, mask, value)

                   if output_qtzr is not None:
                       output = output_qtzr(output)

                   return output
        """
        return super().implements(module_cls)


# pylint: disable=too-many-ancestors


_dispatchable_torch_functions: Set[Callable]
_dispatchable_torch_functions = set(
    itertools.chain(*get_overridable_functions().values())
)


class _Dispatcher(BaseTorchFunctionMode):
    def __init__(self, dispatch_table: Mapping[Callable, Callable]):
        super().__init__()
        self._dispatch_table = dict(dispatch_table)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        impl = self._dispatch_table.get(func, None)

        if impl is None:
            impl = func

        return super().__torch_function__(impl, types, args, kwargs)


def _dispatch(torch_func: Callable, custom_impl: Callable) -> _Dispatcher:
    # Skip raising early exception during torch.compile or torch.export
    if not _torch_compiler_is_compiling() and not _torch_compiler_is_exporting():
        if torch_func not in _dispatchable_torch_functions:
            raise RuntimeError(f"PyTorch doesn't support overriding {torch_func}")

    dispatch_table = {torch_func: custom_impl}
    return _Dispatcher(dispatch_table)


class _DispatchMeta(QuantizationMixinMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Sanity check for class definitions of dispatch-based quantized modules
        """
        if "_builtin_torch_fn" in namespace:
            torch_fn = namespace["_builtin_torch_fn"]
            if torch_fn and torch_fn not in _dispatchable_torch_functions:
                raise RuntimeError(f"PyTorch doesn't support overriding {torch_fn}")
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class _DispatchMixin(metaclass=_DispatchMeta):
    _builtin_torch_fn: Optional[Callable] = None

    def _get_builtin_torch_fn(self):
        return type(self)._builtin_torch_fn

    def _is_dispatch_necessary(self) -> bool:
        # Dispatch is only strictly necessary when Module.forward doesn't merely
        # call the corresponding aten function but performs non-trivial
        # pre/post-processing to the inputs/outputs of aten function.
        return False

    def forward(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        kernel = self.get_kernel()
        builtin_torch_fn = self._get_builtin_torch_fn()

        if not kernel or _is_computing_encodings(self):
            if self._is_dispatch_necessary():
                kernel = self._builtin_torch_fn_helper(builtin_torch_fn)
            else:
                with (
                    self._patch_quantized_parameters(),
                    self._patch_dequantized_parameters(),
                ):
                    return self._forward_no_dispatch(super().forward, *args, **kwargs)
        else:
            kernel = self._custom_kernel_helper(kernel)

        with _dispatch(builtin_torch_fn, kernel):
            output = super().forward(*args, **kwargs)

        return _dequantize_if_applicable(output)

    # NOTE: Exclude from torch.compile as there was a bug observed
    # when trying to compile this function with autocast enabled.
    # TODO (kyunggeu): Triage this bug and fix it or file issue to PyTorch
    @torch.compiler.disable
    def _quantize_if_param(self, args, kwargs):
        params = {
            param: self.param_quantizers[name]
            if name in self.param_quantizers and self.param_quantizers[name]
            else None
            for name, param in self.named_parameters(recurse=False)
        }

        def quantize_if_param(tensor: Any):
            if not isinstance(tensor, torch.Tensor):
                return tensor

            param_qtzr = params.get(tensor, None)

            if (
                torch.onnx.is_in_onnx_export()
                and tensor in params
                and isinstance(tensor, QuantizedTensorBase)
            ):
                # Quantize-dequantize an already-quantized tensor in a possibly duplicate fashion.
                # If duplicate, the duplicate back-to-back QDQs will be removed by the graph pass
                # within aimet_torch.onnx.export
                tensor = tensor.encoding.quantize_dequantize(tensor)

            if param_qtzr:
                return param_qtzr(tensor)

            return tensor

        if isinstance(self, (nn.RNN, nn.GRU, nn.LSTM)):
            args, kwargs = tree_map(quantize_if_param, (args, kwargs))
        else:
            args = tuple(quantize_if_param(arg) for arg in args)
            kwargs = {key: quantize_if_param(value) for key, value in kwargs.items()}

        return args, kwargs

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def wrapper(*args, **kwargs):
            if any(self.param_quantizers.values()) or torch.onnx.is_in_onnx_export():
                args, kwargs = self._quantize_if_param(args, kwargs)
            return self._forward_no_dispatch(fn, *args, **kwargs)

        return wrapper

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def wrapper(*args, **kwargs):
            args, kwargs = self._quantize_if_param(args, kwargs)
            qtzd_args = (
                _quantize_if_applicable(x, qtzr)
                for x, qtzr in zip(args, self.input_quantizers)
            )
            others = args[len(self.input_quantizers) :]

            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )
            kwargs.update(output_encodings=output_encodings)
            return fn(*qtzd_args, *others, **kwargs)

        return wrapper

    def _forward_no_dispatch(self, forward_fn, *args, **kwargs):
        args = tuple(
            _quantize_dequantize_if_applicable(x, qtzr)
            if qtzr
            else _dequantize_if_applicable(x)
            for x, qtzr in itertools.zip_longest(args, self.input_quantizers)
        )
        kwargs = {
            key: _dequantize_if_applicable(value) for key, value in kwargs.items()
        }

        out = forward_fn(*args, **kwargs)

        if self.output_quantizers[0]:
            out = _quantize_dequantize_if_applicable(out, self.output_quantizers[0])

        return out


def _generate_docstring(parent_cls):
    return f"""
    Quantized subclass of torch.nn.{parent_cls.__name__}

    .. method:: forward{str(signature(parent_cls.forward))}
        :noindex:

        Quantized forward of torch.nn.{parent_cls.__name__}.

        The input(s), parameter(s) (if any), and output(s) will be quantized with
        ``self.input_quantizers``, ``self.param_quantizers``, and ``self.output_quantizers`` respectively.

        For more information, see :class:`QuantizationMixin`.
    """


def _derive_bias_scale(
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    bias_shape: torch.Size,
    channel_axis: int,
):
    if input_scale is None or weight_scale is None:
        return None

    bias_scale = input_scale.detach() * weight_scale.detach()

    # bias_scale.shape is not yet compatible with bias.shape due to one of the following reasons:
    #
    # 1. trivial reason (per-channel quantization):
    #      weight_scale and bias_scale are of shape [Cout, 1, 1, 1]
    #
    # 2. non-trivial reason (blockwise quantization):
    #      weight_scale and bias_scale are of shape [Cout, NUM_BLOCKS, 1, 1]
    #
    # In any case, we need to reduce bias_scale into a 1D vector of the same shape as bias (=[Cout])
    non_channel_axes = tuple(
        axis for axis in range(bias_scale.dim()) if axis != channel_axis
    )
    bias_scale = torch.amax(bias_scale, dim=non_channel_axes)

    if bias_scale.shape in ((), bias_shape):
        return bias_scale

    # This means channel_axis != output channel axis.
    # In this case, do not derive bias encoding from input and weight encodings
    return None


@QuantizationMixin.implements(nn.AdaptiveAvgPool1d)
class QuantizedAdaptiveAvgPool1d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool1d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(nn.AdaptiveAvgPool1d)
    _builtin_torch_fn = F.adaptive_avg_pool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AdaptiveAvgPool2d)
class QuantizedAdaptiveAvgPool2d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool2d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AdaptiveAvgPool2d)
    _builtin_torch_fn = F.adaptive_avg_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AdaptiveAvgPool3d)
class QuantizedAdaptiveAvgPool3d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool3d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AdaptiveAvgPool3d)
    _builtin_torch_fn = F.adaptive_avg_pool3d
    __quant_init__ = QuantizationMixin.__unary__


# @QuantizationMixin.implements(nn.AdaptiveLogSoftmaxWithLoss)
# class QuantizedAdaptiveLogSoftmaxWithLoss(_DispatchMixin, QuantizationMixin, nn.AdaptiveLogSoftmaxWithLoss):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.AdaptiveMaxPool1d)
class QuantizedAdaptiveMaxPool1d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool1d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AdaptiveMaxPool1d)
    _builtin_torch_fn = F.adaptive_max_pool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AdaptiveMaxPool2d)
class QuantizedAdaptiveMaxPool2d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool2d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AdaptiveMaxPool2d)
    _builtin_torch_fn = F.adaptive_max_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AdaptiveMaxPool3d)
class QuantizedAdaptiveMaxPool3d(
    _DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool3d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AdaptiveMaxPool3d)
    _builtin_torch_fn = F.adaptive_max_pool3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AlphaDropout)
class QuantizedAlphaDropout(_DispatchMixin, QuantizationMixin, nn.AlphaDropout):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AlphaDropout)
    _builtin_torch_fn = F.alpha_dropout
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AvgPool1d)
class QuantizedAvgPool1d(_DispatchMixin, QuantizationMixin, nn.AvgPool1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AvgPool1d)
    _builtin_torch_fn = F.avg_pool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, nn.AvgPool2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AvgPool2d)
    _builtin_torch_fn = F.avg_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.AvgPool3d)
class QuantizedAvgPool3d(_DispatchMixin, QuantizationMixin, nn.AvgPool3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.AvgPool3d)
    _builtin_torch_fn = F.avg_pool3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.BCELoss)
class QuantizedBCELoss(_DispatchMixin, QuantizationMixin, nn.BCELoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.BCELoss)
    _builtin_torch_fn = F.binary_cross_entropy
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.BCEWithLogitsLoss)
class QuantizedBCEWithLogitsLoss(
    _DispatchMixin, QuantizationMixin, nn.BCEWithLogitsLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.BCEWithLogitsLoss)
    _builtin_torch_fn = F.binary_cross_entropy_with_logits
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.BatchNorm1d)
class QuantizedBatchNorm1d(_DispatchMixin, QuantizationMixin, nn.BatchNorm1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.BatchNorm1d)
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.BatchNorm2d)
class QuantizedBatchNorm2d(_DispatchMixin, QuantizationMixin, nn.BatchNorm2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.BatchNorm2d)
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.BatchNorm3d)
class QuantizedBatchNorm3d(_DispatchMixin, QuantizationMixin, nn.BatchNorm3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.BatchNorm3d)
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Bilinear)
class QuantizedBilinear(_DispatchMixin, QuantizationMixin, nn.Bilinear):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Bilinear)
    _builtin_torch_fn = F.bilinear
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.CELU)
class QuantizedCELU(_DispatchMixin, QuantizationMixin, nn.CELU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.CELU)
    _builtin_torch_fn = F.celu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.CTCLoss)
class QuantizedCTCLoss(_DispatchMixin, QuantizationMixin, nn.CTCLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.CTCLoss)
    _builtin_torch_fn = F.ctc_loss
    __quant_init__ = QuantizationMixin.__unary__

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "F.ctc_loss isn't dynamo-traceable"


@QuantizationMixin.implements(nn.ChannelShuffle)
class QuantizedChannelShuffle(_DispatchMixin, QuantizationMixin, nn.ChannelShuffle):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ChannelShuffle)
    _builtin_torch_fn = F.channel_shuffle
    __quant_init__ = QuantizationMixin.__unary__


if version.parse(torch.__version__) >= version.parse("2.1.0"):

    @QuantizationMixin.implements(nn.CircularPad1d)
    class QuantizedCircularPad1d(QuantizationMixin, nn.CircularPad1d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.CircularPad1d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output

    @QuantizationMixin.implements(nn.CircularPad2d)
    class QuantizedCircularPad2d(QuantizationMixin, nn.CircularPad2d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.CircularPad2d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output

    @QuantizationMixin.implements(nn.CircularPad3d)
    class QuantizedCircularPad3d(QuantizationMixin, nn.CircularPad3d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.CircularPad3d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output


@QuantizationMixin.implements(nn.ConstantPad1d)
class QuantizedConstantPad1d(QuantizationMixin, nn.ConstantPad1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConstantPad2d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.ConstantPad2d)
class QuantizedConstantPad2d(QuantizationMixin, nn.ConstantPad2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConstantPad2d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.ConstantPad3d)
class QuantizedConstantPad3d(QuantizationMixin, nn.ConstantPad3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConstantPad3d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


# @QuantizationMixin.implements(nn.Container)
# class QuantizedContainer(_DispatchMixin, QuantizationMixin, nn.Container):
#     _builtin_torch_fn = ...


@contextlib.contextmanager
def _compute_encodings_with_overflow_protection(self):
    """
    Compute encodings with additional protection against overflow/underflow.

    Given
      y = round((xW + b) * sx * sw / sy - zy)

    HTP implements this equation as either eq 1 or eq 2, depending on hw version and other parameters
      y = round(( xW + b                ) * sx * sw / sy - zy)
        = round(( xW + b                ) *     s'       - zy)  ... eq 1
        = round(( xW + b -       zy/s'  ) *     s'           )
        ≈ round(( xW + b - round(zy/s') ) *     s'           )
        = round(( xW + b'               ) *     s'           )  ... eq 2

    where:

    | name |        description        |        dtype         |          equation          |
    |------|---------------------------|----------------------|----------------------------|
    |  x   | input                     | uint8 or uint16      | round(x_float / sx + zx)   |
    |  W   | weight                    | int4, int8, or int16 | round(W_float / sw)        |
    |  b   | bias                      | int32                | round(b_float / (sx * sw)) |
    |  y   | output                    | dtype(x)             |        given above         |
    |  sx  | input scale               | float                |             -              |
    |  zx  | input zero_point          | dtype(x)             |             -              |
    |  sw  | weight scale              | float                |             -              |
    |  sy  | output scale              | float                |             -              |
    |  zy  | output zero_point         | dtype(y)             |             -              |
    |  s'  | requantization scale      | float                |        sx * sw / sy        |
    |  b'  | combined accumulator bias | int32                |      b - round(zy / s')    |


    This function adjusts the weight scale to prevent 3 possible scenarios of overflow/underflow

    1. Bias Overflow
       - occurs if:   |b| > 2**31
       - bad because: Causes severe clipping error when exported as int32

    2. Requantization Scale Underflow
       - occurs if:   s' <= 2**-24
       - bad because: If exponent e <= -24, HexNN misinterprets s'=2**e as 2**(e+32)
                      due to internal type casting bug

    3. Accumulator Bias Overflow
       - occurs if:   |b'| > 2**31
       - bad because: HexNN internally stores b' as int32
    """
    input_encoding_producer = None

    def capture_input_encoding_producer_hook(self, inputs):
        nonlocal input_encoding_producer

        (input,) = inputs

        if hasattr(input, "encoding") and isinstance(input.encoding, AffineEncoding):
            input_encoding_producer = input.encoding.producer
        elif self.input_quantizers[0]:
            input_encoding_producer = self.input_quantizers[0]
        else:
            input_encoding_producer = None

    handle = None

    try:
        handle = self.register_forward_pre_hook(capture_input_encoding_producer_hook)

        with QuantizationMixin.compute_encodings(self):
            yield
    finally:
        if handle:
            handle.remove()

    weight_qtzr = self.param_quantizers["weight"]

    if not (
        isinstance(input_encoding_producer, AffineQuantizerBase)
        and isinstance(weight_qtzr, AffineQuantizerBase)
    ):
        return

    output_qtzr = self.output_quantizers[0]

    with SafeGatheredParameters(
        itertools.chain(
            weight_qtzr.parameters(),
            input_encoding_producer.parameters(),
            output_qtzr.parameters() if output_qtzr else [],
            [self.bias] if self.bias is not None else [],
        )
    ):
        if not (
            (input_encoding := input_encoding_producer.get_encodings())
            and (weight_encoding := weight_qtzr.get_encodings())
        ):
            return

        bias = (
            self.bias.clone().detach()
            if self.bias is not None
            else torch.zeros((), device=self.weight.device, dtype=self.weight.dtype)
        )

        if output_qtzr := self.output_quantizers[0]:
            output_encoding = output_qtzr.get_encodings()
        else:
            output_encoding = None

    if isinstance(weight_encoding, GroupedBlockEncoding):
        weight_scale = weight_encoding.per_channel_scale
    else:
        weight_scale = weight_encoding.scale

    if weight_scale.ndim == 0:
        bias = bias.abs().amax()
    elif bias.ndim == 0:
        bias = bias.expand_as(weight_scale).flatten().squeeze()

    non_singleton_axes = tuple(
        axis for axis, dim in enumerate(weight_scale.shape) if dim > 1
    )

    if len(non_singleton_axes) > 1:
        # Edge case: weight is quantized with more than 1 non-singleton dimension.
        # In this case, we can't analytically derive bias encoding
        return

    # Prevent bias overflow (1)
    adjusted_weight_scale = _adjust_weight_scale_against_bias_overflow(
        bias,
        input_encoding.scale,
        weight_scale.flatten().squeeze(),
        # Slightly discount from 2**31 to account for numerical instability
        num_steps=2**31 - 2**15,
    )
    adjusted_weight_scale = adjusted_weight_scale.reshape(weight_scale.shape)

    if not isinstance(output_encoding, AffineEncoding):
        return

    # Prevent requantization scale underflow (2)
    adjusted_weight_scale = _adjust_weight_scale_against_scale_underflow(
        input_encoding.scale,
        adjusted_weight_scale,
        output_encoding.scale,
    )

    # Prevent accumulator bias overflow (3)
    bias_scale = (input_encoding.scale * adjusted_weight_scale).flatten().squeeze()
    accumulator_bias = torch.round(bias / bias_scale) + torch.round(
        output_encoding.offset * output_encoding.scale / bias_scale
    )
    # Use slightly discounted num_steps to account for floating point precision error
    adjusted_weight_scale = _adjust_weight_scale_against_bias_overflow(
        accumulator_bias * bias_scale,
        input_encoding.scale,
        adjusted_weight_scale,
        num_steps=2**31 - 2**15,
    )

    if isinstance(weight_encoding, GroupedBlockEncoding):
        new_weight_encoding = GroupedBlockEncoding(
            scale=weight_encoding.per_block_int_scale * adjusted_weight_scale,
            offset=weight_encoding.offset,
            bitwidth=weight_encoding.bitwidth,
            block_size=weight_encoding.block_size,
            block_grouping=weight_encoding.block_grouping,
            decompressed_bw=weight_encoding.decompressed_bw,
            per_channel_scale=adjusted_weight_scale,
        )
    else:
        new_weight_encoding = AffineEncoding(
            adjusted_weight_scale,
            weight_encoding.offset,
            weight_encoding.qmin,
            weight_encoding.qmax,
            weight_encoding.symmetry,
            weight_encoding.block_size,
            weight_encoding.zero_point_shift,
        )

    weight_qtzr.set_range(
        min=new_weight_encoding.min,
        max=new_weight_encoding.max,
    )


@QuantizationMixin.implements(nn.Conv1d)
class QuantizedConv1d(_DispatchMixin, QuantizationMixin, nn.Conv1d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Conv1d)
    _builtin_torch_fn = F.conv1d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=0
        )


@QuantizationMixin.implements(nn.Conv2d)
class QuantizedConv2d(_DispatchMixin, QuantizationMixin, nn.Conv2d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Conv2d)
    _builtin_torch_fn = F.conv2d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=0
        )


@QuantizationMixin.implements(nn.Conv3d)
class QuantizedConv3d(_DispatchMixin, QuantizationMixin, nn.Conv3d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Conv3d)
    _builtin_torch_fn = F.conv3d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=0
        )


@QuantizationMixin.implements(nn.ConvTranspose1d)
class QuantizedConvTranspose1d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose1d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConvTranspose1d)
    _builtin_torch_fn = F.conv_transpose1d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=1
        )


@QuantizationMixin.implements(nn.ConvTranspose2d)
class QuantizedConvTranspose2d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose2d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConvTranspose2d)
    _builtin_torch_fn = F.conv_transpose2d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=1
        )


@QuantizationMixin.implements(nn.ConvTranspose3d)
class QuantizedConvTranspose3d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose3d):  # pylint: disable=too-many-ancestors
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ConvTranspose3d)
    _builtin_torch_fn = F.conv_transpose3d
    __quant_init__ = QuantizationMixin.__unary__

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=1
        )


@QuantizationMixin.implements(nn.CosineEmbeddingLoss)
class QuantizedCosineEmbeddingLoss(
    _DispatchMixin, QuantizationMixin, nn.CosineEmbeddingLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.CosineEmbeddingLoss)
    _builtin_torch_fn = F.cosine_embedding_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.CosineSimilarity)
class QuantizedCosineSimilarity(_DispatchMixin, QuantizationMixin, nn.CosineSimilarity):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.CosineSimilarity)
    _builtin_torch_fn = F.cosine_similarity
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.CrossEntropyLoss)
class QuantizedCrossEntropyLoss(_DispatchMixin, QuantizationMixin, nn.CrossEntropyLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.CrossEntropyLoss)
    _builtin_torch_fn = F.cross_entropy
    __quant_init__ = QuantizationMixin.__binary__


# @QuantizationMixin.implements(nn.CrossMapLRN2d)
# class QuantizedCrossMapLRN2d(_DispatchMixin, QuantizationMixin, nn.CrossMapLRN2d):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.Dropout)
class QuantizedDropout(_DispatchMixin, QuantizationMixin, nn.Dropout):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Dropout)
    _builtin_torch_fn = F.dropout
    __quant_init__ = QuantizationMixin.__unary__


if version.parse(torch.__version__) >= version.parse("1.12.0"):

    @QuantizationMixin.implements(nn.Dropout1d)
    class QuantizedDropout1d(_DispatchMixin, QuantizationMixin, nn.Dropout1d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.Dropout1d)
        _builtin_torch_fn = F.dropout1d
        __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Dropout2d)
class QuantizedDropout2d(_DispatchMixin, QuantizationMixin, nn.Dropout2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Dropout2d)
    _builtin_torch_fn = F.dropout2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Dropout3d)
class QuantizedDropout3d(_DispatchMixin, QuantizationMixin, nn.Dropout3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Dropout3d)
    _builtin_torch_fn = F.dropout3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.ELU)
class QuantizedELU(_DispatchMixin, QuantizationMixin, nn.ELU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ELU)
    _builtin_torch_fn = F.elu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Embedding)
class QuantizedEmbedding(_DispatchMixin, QuantizationMixin, nn.Embedding):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Embedding)
    _builtin_torch_fn = F.embedding
    __quant_init__ = QuantizationMixin.__nullary__


@QuantizationMixin.implements(nn.EmbeddingBag)
class QuantizedEmbeddingBag(_DispatchMixin, QuantizationMixin, nn.EmbeddingBag):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.EmbeddingBag)
    _builtin_torch_fn = F.embedding_bag

    def _is_dispatch_necessary(self) -> bool:
        return True

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def embedding_bag(
            input: Tensor,  # pylint: disable=redefined-builtin, too-many-arguments
            weight: Tensor,
            offsets: Optional[Tensor] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2,
            scale_grad_by_freq: bool = False,
            mode: str = "mean",
            sparse: bool = False,
            per_sample_weights: Optional[Tensor] = None,
            include_last_offset: bool = False,
            padding_idx: Optional[int] = None,
        ):
            if per_sample_weights is not None:
                qtzr = self.input_quantizers[0]
                per_sample_weights = _quantize_dequantize_if_applicable(
                    per_sample_weights, qtzr
                )

            weight_qtzr = self.param_quantizers["weight"]

            if weight_qtzr:
                weight = weight_qtzr(weight)

            output = fn(
                input,
                weight,
                offsets=offsets,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                mode=mode,
                sparse=sparse,
                per_sample_weights=per_sample_weights,
                include_last_offset=include_last_offset,
                padding_idx=padding_idx,
            )

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return embedding_bag

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def embedding_bag(
            input: Tensor,  # pylint: disable=redefined-builtin, too-many-arguments
            weight: Tensor,
            offsets: Optional[Tensor] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2,
            scale_grad_by_freq: bool = False,
            mode: str = "mean",
            sparse: bool = False,
            per_sample_weights: Optional[Tensor] = None,
            include_last_offset: bool = False,
            padding_idx: Optional[int] = None,
        ):
            if per_sample_weights is not None:
                qtzr = self.input_quantizers[0]
                per_sample_weights = _quantize_if_applicable(per_sample_weights, qtzr)

            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )

            return fn(
                input,
                weight,
                offsets=offsets,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                mode=mode,
                sparse=sparse,
                per_sample_weights=per_sample_weights,
                include_last_offset=include_last_offset,
                padding_idx=padding_idx,
                output_encodings=output_encodings,
            )

        return embedding_bag


@QuantizationMixin.implements(nn.FeatureAlphaDropout)
class QuantizedFeatureAlphaDropout(
    _DispatchMixin, QuantizationMixin, nn.FeatureAlphaDropout
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.FeatureAlphaDropout)
    _builtin_torch_fn = F.feature_alpha_dropout
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Flatten)
class QuantizedFlatten(_DispatchMixin, QuantizationMixin, nn.Flatten):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Flatten)

    def _get_builtin_torch_fn(self):
        return Tensor.flatten

    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Fold)
class QuantizedFold(_DispatchMixin, QuantizationMixin, nn.Fold):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Fold)
    _builtin_torch_fn = F.fold
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.FractionalMaxPool2d)
class QuantizedFractionalMaxPool2d(
    _DispatchMixin, QuantizationMixin, nn.FractionalMaxPool2d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.FractionalMaxPool2d)
    _builtin_torch_fn = F.fractional_max_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.FractionalMaxPool3d)
class QuantizedFractionalMaxPool3d(
    _DispatchMixin, QuantizationMixin, nn.FractionalMaxPool3d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.FractionalMaxPool3d)
    _builtin_torch_fn = F.fractional_max_pool3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.GELU)
class QuantizedGELU(_DispatchMixin, QuantizationMixin, nn.GELU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GELU)
    _builtin_torch_fn = F.gelu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.GLU)
class QuantizedGLU(_DispatchMixin, QuantizationMixin, nn.GLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GLU)
    _builtin_torch_fn = F.glu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.GRU)
class QuantizedGRU(_DispatchMixin, QuantizationMixin, nn.GRU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GRU)
    _builtin_torch_fn = _gru

    def _is_dispatch_necessary(self) -> bool:
        return True

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _quantize_inputs(self, args, apply):
        if args[1].is_floating_point():
            input, hx, *others = args
            batch_sizes = None
        else:
            input, batch_sizes, hx, *others = args

        input = apply(input, self.input_quantizers[0])
        hx = apply(hx, self.input_quantizers[1])

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _gru
        apply = _quantize_dequantize_if_applicable

        def gru(*args):
            args = self._quantize_inputs(args, apply)
            args, _ = self._quantize_if_param(args, {})
            output, h_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
            )

        return gru

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def gru(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(
                qtzr and qtzr.get_encodings() for qtzr in self.output_quantizers
            )
            return fn(*args, output_encodings=output_encodings)

        return gru

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.nn.GRU isn't dynamo-traceable"


@QuantizationMixin.implements(nn.GRUCell)
class QuantizedGRUCell(_DispatchMixin, QuantizationMixin, nn.GRUCell):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GRUCell)
    _builtin_torch_fn = _gru_cell

    def _is_dispatch_necessary(self) -> bool:
        return True

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _gru_cell
        apply = _quantize_dequantize_if_applicable

        def gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
            w_ih = apply(w_ih, self.param_quantizers["weight_ih"])
            w_hh = apply(w_hh, self.param_quantizers["weight_hh"])
            if b_ih is not None:
                b_ih = apply(b_ih, self.param_quantizers["bias_ih"])
            if b_hh is not None:
                b_hh = apply(b_hh, self.param_quantizers["bias_hh"])
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output = fn(input, hx, w_ih, w_hh, b_ih, b_hh)
            return apply(output, self.output_quantizers[0])

        return gru_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def gru_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output_encodings = (
                self.output_quantizers[0] and self.output_quantizers[0].get_encodings()
            )
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return gru_cell


@QuantizationMixin.implements(nn.GaussianNLLLoss)
class QuantizedGaussianNLLLoss(_DispatchMixin, QuantizationMixin, nn.GaussianNLLLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GaussianNLLLoss)
    _builtin_torch_fn = F.gaussian_nll_loss
    __quant_init__ = QuantizationMixin.__ternary__

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "F.gaussian_nll_loss isn't dynamo-traceable"


@QuantizationMixin.implements(nn.GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, nn.GroupNorm):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.GroupNorm)
    _builtin_torch_fn = F.group_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Hardshrink)
class QuantizedHardshrink(_DispatchMixin, QuantizationMixin, nn.Hardshrink):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Hardshrink)
    _builtin_torch_fn = F.hardshrink
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Hardsigmoid)
class QuantizedHardsigmoid(QuantizationMixin, nn.Hardsigmoid):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Hardsigmoid)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.Hardswish)
class QuantizedHardswish(QuantizationMixin, nn.Hardswish):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Hardswish)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.Hardtanh)
class QuantizedHardtanh(_DispatchMixin, QuantizationMixin, nn.Hardtanh):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Hardtanh)
    _builtin_torch_fn = F.hardtanh
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.HingeEmbeddingLoss)
class QuantizedHingeEmbeddingLoss(
    _DispatchMixin, QuantizationMixin, nn.HingeEmbeddingLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.HingeEmbeddingLoss)
    _builtin_torch_fn = F.hinge_embedding_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.HuberLoss)
class QuantizedHuberLoss(_DispatchMixin, QuantizationMixin, nn.HuberLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.HuberLoss)
    _builtin_torch_fn = F.huber_loss
    __quant_init__ = QuantizationMixin.__binary__


QuantizationMixin.ignore(nn.Identity)


@QuantizationMixin.implements(nn.InstanceNorm1d)
class QuantizedInstanceNorm1d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.InstanceNorm1d)
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.InstanceNorm2d)
class QuantizedInstanceNorm2d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.InstanceNorm2d)
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.InstanceNorm3d)
class QuantizedInstanceNorm3d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.InstanceNorm3d)
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.KLDivLoss)
class QuantizedKLDivLoss(_DispatchMixin, QuantizationMixin, nn.KLDivLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.KLDivLoss)
    _builtin_torch_fn = F.kl_div
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.L1Loss)
class QuantizedL1Loss(_DispatchMixin, QuantizationMixin, nn.L1Loss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.L1Loss)
    _builtin_torch_fn = F.l1_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.LPPool1d)
class QuantizedLPPool1d(_DispatchMixin, QuantizationMixin, nn.LPPool1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LPPool1d)
    _builtin_torch_fn = F.lp_pool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.LPPool2d)
class QuantizedLPPool2d(_DispatchMixin, QuantizationMixin, nn.LPPool2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LPPool2d)
    _builtin_torch_fn = F.lp_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.LSTM)
class QuantizedLSTM(_DispatchMixin, QuantizationMixin, nn.LSTM):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LSTM)
    _builtin_torch_fn = _lstm

    def _is_dispatch_necessary(self) -> bool:
        return True

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None, None])

    def _quantize_inputs(self, args, apply):
        if isinstance(args[1], Tensor):
            input, batch_sizes, hx, *others = args
        else:
            input, hx, *others = args
            batch_sizes = None

        input = apply(input, self.input_quantizers[0])
        h, c = hx
        h_qtzr, c_qtzr = self.input_quantizers[1:]
        hx = (apply(h, h_qtzr), apply(c, c_qtzr))

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _lstm
        apply = _quantize_dequantize_if_applicable

        def lstm(*args):
            args = self._quantize_inputs(args, apply)
            args, _ = self._quantize_if_param(args, {})
            output, h_n, c_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
                apply(c_n, self.output_quantizers[2]),
            )

        return lstm

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def lstm(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(
                qtzr and qtzr.get_encodings() for qtzr in self.output_quantizers
            )
            return fn(*args, output_encodings=output_encodings)

        return lstm

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.nn.LSTM isn't dynamo-traceable"


@QuantizationMixin.implements(nn.LSTMCell)
class QuantizedLSTMCell(_DispatchMixin, QuantizationMixin, nn.LSTMCell):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LSTMCell)
    _builtin_torch_fn = _lstm_cell

    def _is_dispatch_necessary(self) -> bool:
        return True

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _lstm_cell
        apply = _quantize_dequantize_if_applicable

        def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
            w_ih = apply(w_ih, self.param_quantizers["weight_ih"])
            w_hh = apply(w_hh, self.param_quantizers["weight_hh"])
            if b_ih is not None:
                b_ih = apply(b_ih, self.param_quantizers["bias_ih"])
            if b_hh is not None:
                b_hh = apply(b_hh, self.param_quantizers["bias_hh"])
            input = apply(input, self.input_quantizers[0])
            h, c = hx
            h_qtzr, c_qtzr = self.input_quantizers[1:]
            hx = (apply(h, h_qtzr), apply(c, c_qtzr))

            hx, cx = fn(input, hx, w_ih, w_hh, b_ih, b_hh)
            return (
                apply(hx, self.output_quantizers[0]),
                apply(cx, self.output_quantizers[1]),
            )

        return lstm_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def lstm_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            h, c = hx
            h_qtzr, c_qtzr = self.input_quantizers[1:]
            hx = (apply(h, h_qtzr), apply(c, c_qtzr))

            output_encodings = tuple(
                qtzr and qtzr.get_encodings() for qtzr in self.output_quantizers
            )
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return lstm_cell


@QuantizationMixin.implements(nn.LayerNorm)
class QuantizedLayerNorm(_DispatchMixin, QuantizationMixin, nn.LayerNorm):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LayerNorm)
    _builtin_torch_fn = F.layer_norm
    __quant_init__ = QuantizationMixin.__unary__


# @QuantizationMixin.implements(nn.LazyBatchNorm1d)
# class QuantizedLazyBatchNorm1d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm1d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyBatchNorm2d)
# class QuantizedLazyBatchNorm2d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm2d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyBatchNorm3d)
# class QuantizedLazyBatchNorm3d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm3d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv1d)
# class QuantizedLazyConv1d(_DispatchMixin, QuantizationMixin, nn.LazyConv1d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv2d)
# class QuantizedLazyConv2d(_DispatchMixin, QuantizationMixin, nn.LazyConv2d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv3d)
# class QuantizedLazyConv3d(_DispatchMixin, QuantizationMixin, nn.LazyConv3d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose1d)
# class QuantizedLazyConvTranspose1d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose1d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose2d)
# class QuantizedLazyConvTranspose2d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose2d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose3d)
# class QuantizedLazyConvTranspose3d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose3d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm1d)
# class QuantizedLazyInstanceNorm1d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm1d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm2d)
# class QuantizedLazyInstanceNorm2d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm2d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm3d)
# class QuantizedLazyInstanceNorm3d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm3d):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyLinear)
# class QuantizedLazyLinear(_DispatchMixin, QuantizationMixin, nn.LazyLinear):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.LeakyReLU)
class QuantizedLeakyReLU(_DispatchMixin, QuantizationMixin, nn.LeakyReLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LeakyReLU)
    _builtin_torch_fn = F.leaky_relu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Linear)
class QuantizedLinear(_DispatchMixin, QuantizationMixin, nn.Linear):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Linear)
    _builtin_torch_fn = F.linear
    __quant_init__ = QuantizationMixin.__unary__

    # Only allow activation recompute (a.k.a activation checkpointing) for QuantizedLinear.
    # This is mainly to reduce memory footprint of QAT of large language models.
    @allow_recompute
    def forward(self, *args, **kwargs):
        if _torch_compiler_is_exporting():
            return super().forward(*args, **kwargs)

        if not _torch_compiler_is_compiling() and not _torch_compiler_is_exporting():
            # Workaround for deepspeed.
            # Deepspeed zero3 sometimes forcefully mokey-patches F.linear to torch.addmm,
            # which collides with the core assumption of our dispatch mechanism
            # that nn.Linear invokes F.linear.
            # To circumvent this issue, we temporarily restore the original F.linear
            # before running forward.
            ctx = patch_attr(F, "linear", type(self)._builtin_torch_fn)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return super().forward(*args, **kwargs)

    compute_encodings = _compute_encodings_with_overflow_protection

    def _derive_bias_scale(
        self, input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]
    ):
        return _derive_bias_scale(
            input_scale, weight_scale, self.bias.shape, channel_axis=0
        )


@QuantizationMixin.implements(nn.LocalResponseNorm)
class QuantizedLocalResponseNorm(
    _DispatchMixin, QuantizationMixin, nn.LocalResponseNorm
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LocalResponseNorm)
    _builtin_torch_fn = F.local_response_norm
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.LogSigmoid)
class QuantizedLogSigmoid(_DispatchMixin, QuantizationMixin, nn.LogSigmoid):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LogSigmoid)
    _builtin_torch_fn = F.logsigmoid
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.LogSoftmax)
class QuantizedLogSoftmax(_DispatchMixin, QuantizationMixin, nn.LogSoftmax):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.LogSoftmax)
    _builtin_torch_fn = F.log_softmax
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MSELoss)
class QuantizedMSELoss(_DispatchMixin, QuantizationMixin, nn.MSELoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MSELoss)
    _builtin_torch_fn = F.mse_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.MarginRankingLoss)
class QuantizedMarginRankingLoss(
    _DispatchMixin, QuantizationMixin, nn.MarginRankingLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MarginRankingLoss)
    _builtin_torch_fn = F.margin_ranking_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.MaxPool1d)
class QuantizedMaxPool1d(_DispatchMixin, QuantizationMixin, nn.MaxPool1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxPool1d)
    _builtin_torch_fn = F.max_pool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MaxPool2d)
class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, nn.MaxPool2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxPool2d)
    _builtin_torch_fn = F.max_pool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MaxPool3d)
class QuantizedMaxPool3d(_DispatchMixin, QuantizationMixin, nn.MaxPool3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxPool3d)
    _builtin_torch_fn = F.max_pool3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MaxUnpool1d)
class QuantizedMaxUnpool1d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxUnpool1d)
    _builtin_torch_fn = F.max_unpool1d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MaxUnpool2d)
class QuantizedMaxUnpool2d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxUnpool2d)
    _builtin_torch_fn = F.max_unpool2d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MaxUnpool3d)
class QuantizedMaxUnpool3d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MaxUnpool3d)
    _builtin_torch_fn = F.max_unpool3d
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Mish)
class QuantizedMish(_DispatchMixin, QuantizationMixin, nn.Mish):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Mish)
    _builtin_torch_fn = F.mish
    __quant_init__ = QuantizationMixin.__unary__


# @QuantizationMixin.implements(nn.Module)
# class QuantizedModule(_DispatchMixin, QuantizationMixin, nn.Module):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ModuleDict)
# class QuantizedModuleDict(_DispatchMixin, QuantizationMixin, nn.ModuleDict):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ModuleList)
# class QuantizedModuleList(_DispatchMixin, QuantizationMixin, nn.ModuleList):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.MultiLabelMarginLoss)
class QuantizedMultiLabelMarginLoss(
    _DispatchMixin, QuantizationMixin, nn.MultiLabelMarginLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MultiLabelMarginLoss)
    _builtin_torch_fn = F.multilabel_margin_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MultiLabelSoftMarginLoss)
class QuantizedMultiLabelSoftMarginLoss(
    _DispatchMixin, QuantizationMixin, nn.MultiLabelSoftMarginLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MultiLabelSoftMarginLoss)
    _builtin_torch_fn = F.multilabel_soft_margin_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MultiMarginLoss)
class QuantizedMultiMarginLoss(_DispatchMixin, QuantizationMixin, nn.MultiMarginLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.MultiMarginLoss)
    _builtin_torch_fn = F.multi_margin_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.MultiheadAttention)
class QuantizedMultiheadAttention(QuantizationMixin, nn.MultiheadAttention):
    def __quant_init__(self):
        super().__quant_init__()
        self.param_quantizers.clear()
        self.input_quantizers = torch.nn.ModuleList()
        self.output_quantizers = torch.nn.ModuleList()

        if self.in_proj_weight is None:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight
        else:
            q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3)

        if self.in_proj_bias is None:
            q_proj_bias = k_proj_bias = v_proj_bias = None
        else:
            q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3)

        self.q_proj = nn.Linear(
            in_features=q_proj_weight.shape[1],
            out_features=q_proj_weight.shape[0],
            bias=q_proj_bias is not None,
        )
        self.q_proj.weight = torch.nn.Parameter(q_proj_weight)
        if q_proj_bias is not None:
            self.q_proj.bias = torch.nn.Parameter(q_proj_bias)

        self.k_proj = nn.Linear(
            in_features=k_proj_weight.shape[1],
            out_features=k_proj_weight.shape[0],
            bias=k_proj_bias is not None,
        )
        self.k_proj.weight = torch.nn.Parameter(k_proj_weight)
        if k_proj_bias is not None:
            self.k_proj.bias = torch.nn.Parameter(k_proj_bias)

        self.v_proj = nn.Linear(
            in_features=v_proj_weight.shape[1],
            out_features=v_proj_weight.shape[0],
            bias=v_proj_bias is not None,
        )
        self.v_proj.weight = torch.nn.Parameter(v_proj_weight)
        if v_proj_bias is not None:
            self.v_proj.bias = torch.nn.Parameter(v_proj_bias)

        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias
        self.out_proj = nn.Linear(
            in_features=out_proj_weight.shape[1],
            out_features=out_proj_weight.shape[0],
            bias=out_proj_bias is not None,
        )
        self.out_proj.weight = out_proj_weight
        self.out_proj.bias = out_proj_bias

        from .modules import custom

        self.qk_matmul = custom.MatMul()
        self.mask_add = custom.Add()
        self.key_padding_mask_add = custom.Add()
        self.softmax = nn.Softmax(dim=-1)
        self.qkv_matmul = custom.MatMul()
        self.mean = custom.Mean()
        self.cat_bias_k = custom.Concat(axis=0)
        self.cat_bias_v = custom.Concat(axis=0)
        self.attn_mask_value = -float("inf")

    def _validate_inputs(  # pylint: disable=unused-argument
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        need_weights: bool,
        attn_mask: Optional[Tensor],
        average_attn_weights: bool,
        is_causal: bool,
    ):
        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)
        num_heads = self.num_heads

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

    def forward(  # pylint: disable=arguments-differ
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        H = self.num_heads
        D = self.head_dim
        batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, H)

        if batched and self.batch_first:
            # Transpose from (N, L, E) to (L, N, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if batched:
            L, N, E = query.size()
        else:
            N = 1
            L, E = query.size()
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        self._validate_inputs(
            query,
            key,
            value,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )

        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = None

        if key_padding_mask is not None and not key_padding_mask.is_floating_point():
            key_padding_mask = torch.zeros_like(
                key_padding_mask, dtype=query.dtype, device=query.device
            ).masked_fill_(key_padding_mask, self.attn_mask_value)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # (1, L, L)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None and self.bias_v is not None:
            bias_k = self.bias_k.repeat_interleave(N, dim=1)
            k = self.cat_bias_k(k, bias_k)
            bias_v = self.bias_v.repeat_interleave(N, dim=1)
            v = self.cat_bias_v(v, bias_v)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Transpose to (N * H, L, D) for attention computation
        q = q.view(-1, N * H, D).transpose(0, 1)
        k = k.view(-1, N * H, D).transpose(0, 1)
        v = v.view(-1, N * H, D).transpose(0, 1)

        if self.add_zero_attn:
            k = F.pad(k, (0, 0, 0, 1, 0, 0), value=0)
            v = F.pad(v, (0, 0, 0, 1, 0, 0), value=0)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        if attn_mask is not None and not attn_mask.is_floating_point():
            attn_mask = torch.zeros_like(
                attn_mask, dtype=query.dtype, device=query.device
            ).masked_fill_(attn_mask, self.attn_mask_value)

        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.view(-1, 1, 1, k.size(1))
                .expand(-1, H, -1, -1)
                .reshape(-1, 1, k.size(1))
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = self.key_padding_mask_add(attn_mask, key_padding_mask)

        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = torch.full(
                (q.size(1), k.size(1)),
                self.attn_mask_value,
                dtype=query.dtype,
                device=query.device,
            ).triu(diagonal=1)

        # Scaled dot-product attention
        attn_scores = self.qk_matmul(q, k.transpose(-2, -1)) / (D**0.5)  # (N * H, L, L)

        # Apply attention mask if provided
        attn_scores = (
            self.mask_add(attn_scores, attn_mask)
            if attn_mask is not None
            else attn_scores
        )

        attn_weights = self.softmax(attn_scores)  # (N * H, L, L)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = self.qkv_matmul(attn_weights, v)  # (N * H, L, D)

        # Transpose and reshape back to (L, N, E)
        attn_output = attn_output.transpose(0, 1).reshape(-1, E)

        # Output projection
        output = self.out_proj(attn_output)
        output = output.view(L, N, -1)  # (L, N, E)

        attn_weights = attn_weights.view(N, H, L, -1)

        if batched:
            if self.batch_first:
                output = output.permute(1, 0, 2)  # (N, L, E)
        else:
            output = output.squeeze(1)  # (L, E)

        # Handle need_weights and average_attn_weights
        if need_weights:
            # Average attention weights over heads if requested
            if average_attn_weights:
                # (N, H, L, L) -> (N, L, L)
                attn_weights = self.mean(attn_weights, dim=1)
            if not batched:
                attn_weights = attn_weights.squeeze(0)  # (H, L, L)

            return output, attn_weights
        else:
            return output, None


@QuantizationMixin.implements(nn.NLLLoss)
class QuantizedNLLLoss(_DispatchMixin, QuantizationMixin, nn.NLLLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.NLLLoss)
    _builtin_torch_fn = F.nll_loss
    __quant_init__ = QuantizationMixin.__unary__


# # Suppress FutureWarning when accessing nn.NLLLoss2d (deprecated in PyTorch, alias for NLLLoss)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    _NLLLoss2d = nn.NLLLoss2d


@QuantizationMixin.implements(_NLLLoss2d)
class QuantizedNLLLoss2d(_DispatchMixin, QuantizationMixin, nn.NLLLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.NLLLoss)
    _builtin_torch_fn = F.nll_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(torch.nn.modules.linear.NonDynamicallyQuantizableLinear)
class QuantizedNonDynamicallyQuantizableLinear(
    QuantizedLinear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear
):
    pass


@QuantizationMixin.implements(nn.PReLU)
class QuantizedPReLU(_DispatchMixin, QuantizationMixin, nn.PReLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.PReLU)
    _builtin_torch_fn = F.prelu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.PairwiseDistance)
class QuantizedPairwiseDistance(_DispatchMixin, QuantizationMixin, nn.PairwiseDistance):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.PairwiseDistance)
    _builtin_torch_fn = F.pairwise_distance
    __quant_init__ = QuantizationMixin.__binary__


# @QuantizationMixin.implements(nn.ParameterDict)
# class QuantizedParameterDict(_DispatchMixin, QuantizationMixin, nn.ParameterDict):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ParameterList)
# class QuantizedParameterList(_DispatchMixin, QuantizationMixin, nn.ParameterList):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.PixelShuffle)
class QuantizedPixelShuffle(_DispatchMixin, QuantizationMixin, nn.PixelShuffle):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.PixelShuffle)
    _builtin_torch_fn = F.pixel_shuffle
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.PixelUnshuffle)
class QuantizedPixelUnshuffle(_DispatchMixin, QuantizationMixin, nn.PixelUnshuffle):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.PixelUnshuffle)
    _builtin_torch_fn = F.pixel_unshuffle
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.PoissonNLLLoss)
class QuantizedPoissonNLLLoss(_DispatchMixin, QuantizationMixin, nn.PoissonNLLLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.PoissonNLLLoss)
    _builtin_torch_fn = F.poisson_nll_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.RNN)
class QuantizedRNN(_DispatchMixin, QuantizationMixin, nn.RNN):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.RNN)

    def _is_dispatch_necessary(self) -> bool:
        return True

    def _get_builtin_torch_fn(self):
        assert self.mode in ("RNN_TANH", "RNN_RELU")
        if self.mode == "RNN_TANH":
            return _rnn_tanh
        return _rnn_relu

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _quantize_inputs(self, args, apply):
        if args[1].is_floating_point():
            input, hx, *others = args
            batch_sizes = None
        else:
            input, batch_sizes, hx, *others = args

        input = apply(input, self.input_quantizers[0])
        hx = apply(hx, self.input_quantizers[1])

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn in (_rnn_tanh, _rnn_relu)
        apply = _quantize_dequantize_if_applicable

        def rnn(*args):
            args = self._quantize_inputs(args, apply)
            args, _ = self._quantize_if_param(args, {})
            output, h_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
            )

        return rnn

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def rnn(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(
                qtzr and qtzr.get_encodings() for qtzr in self.output_quantizers
            )
            return fn(*args, output_encodings=output_encodings)

        return rnn

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.nn.RNN isn't dynamo-traceable"


# @QuantizationMixin.implements(nn.RNNBase)
# class QuantizedRNNBase(_DispatchMixin, QuantizationMixin, nn.RNNBase):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.RNNCell)
class QuantizedRNNCell(_DispatchMixin, QuantizationMixin, nn.RNNCell):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.RNNCell)

    def _is_dispatch_necessary(self) -> bool:
        return True

    def _get_builtin_torch_fn(self):
        assert self.nonlinearity in ("tanh", "relu")

        if self.nonlinearity == "tanh":
            return _rnn_tanh_cell
        return _rnn_relu_cell

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn in (_rnn_tanh_cell, _rnn_relu_cell)
        apply = _quantize_dequantize_if_applicable

        def rnn_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
            w_ih = apply(w_ih, self.param_quantizers["weight_ih"])
            w_hh = apply(w_hh, self.param_quantizers["weight_hh"])
            if b_ih is not None:
                b_ih = apply(b_ih, self.param_quantizers["bias_ih"])
            if b_hh is not None:
                b_hh = apply(b_hh, self.param_quantizers["bias_hh"])
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output = fn(input, hx, w_ih, w_hh, b_ih, b_hh)
            return apply(output, self.output_quantizers[0])

        return rnn_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def rnn_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output_encodings = (
                self.output_quantizers[0] and self.output_quantizers[0].get_encodings()
            )
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return rnn_cell


# @QuantizationMixin.implements(nn.RNNCellBase)
# class QuantizedRNNCellBase(_DispatchMixin, QuantizationMixin, nn.RNNCellBase):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.RReLU)
class QuantizedRReLU(_DispatchMixin, QuantizationMixin, nn.RReLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.RReLU)
    _builtin_torch_fn = F.rrelu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.ReLU)
class QuantizedReLU(_DispatchMixin, QuantizationMixin, nn.ReLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReLU)
    _builtin_torch_fn = F.relu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.ReLU6)
class QuantizedReLU6(_DispatchMixin, QuantizationMixin, nn.ReLU6):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReLU6)
    _builtin_torch_fn = F.hardtanh
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.ReflectionPad1d)
class QuantizedReflectionPad1d(QuantizationMixin, nn.ReflectionPad1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReflectionPad1d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.ReflectionPad2d)
class QuantizedReflectionPad2d(QuantizationMixin, nn.ReflectionPad2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReflectionPad2d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


if version.parse(torch.__version__) >= version.parse("1.10.0"):

    @QuantizationMixin.implements(nn.ReflectionPad3d)
    class QuantizedReflectionPad3d(QuantizationMixin, nn.ReflectionPad3d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.ReflectionPad3d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output


@QuantizationMixin.implements(nn.ReplicationPad1d)
class QuantizedReplicationPad1d(QuantizationMixin, nn.ReplicationPad1d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReplicationPad1d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.ReplicationPad2d)
class QuantizedReplicationPad2d(QuantizationMixin, nn.ReplicationPad2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReplicationPad2d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.ReplicationPad3d)
class QuantizedReplicationPad3d(QuantizationMixin, nn.ReplicationPad3d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ReplicationPad3d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.SELU)
class QuantizedSELU(_DispatchMixin, QuantizationMixin, nn.SELU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.SELU)
    _builtin_torch_fn = F.selu
    __quant_init__ = QuantizationMixin.__unary__


# @QuantizationMixin.implements(nn.Sequential)
# class QuantizedSequential(_DispatchMixin, QuantizationMixin, nn.Sequential):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.SiLU)
class QuantizedSiLU(_DispatchMixin, QuantizationMixin, nn.SiLU):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.SiLU)
    _builtin_torch_fn = F.silu
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Sigmoid)
class QuantizedSigmoid(_DispatchMixin, QuantizationMixin, nn.Sigmoid):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Sigmoid)
    _builtin_torch_fn = torch.sigmoid
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.SmoothL1Loss)
class QuantizedSmoothL1Loss(_DispatchMixin, QuantizationMixin, nn.SmoothL1Loss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.SmoothL1Loss)
    _builtin_torch_fn = F.smooth_l1_loss
    __quant_init__ = QuantizationMixin.__binary__


@QuantizationMixin.implements(nn.SoftMarginLoss)
class QuantizedSoftMarginLoss(_DispatchMixin, QuantizationMixin, nn.SoftMarginLoss):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.SoftMarginLoss)
    _builtin_torch_fn = F.soft_margin_loss
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softmax)
class QuantizedSoftmax(_DispatchMixin, QuantizationMixin, nn.Softmax):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softmax)
    _builtin_torch_fn = F.softmax
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softmax2d)
class QuantizedSoftmax2d(_DispatchMixin, QuantizationMixin, nn.Softmax2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softmax2d)
    _builtin_torch_fn = F.softmax
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softmin)
class QuantizedSoftmin(_DispatchMixin, QuantizationMixin, nn.Softmin):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softmin)
    _builtin_torch_fn = F.softmin
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softplus)
class QuantizedSoftplus(_DispatchMixin, QuantizationMixin, nn.Softplus):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softplus)
    _builtin_torch_fn = F.softplus
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softshrink)
class QuantizedSoftshrink(_DispatchMixin, QuantizationMixin, nn.Softshrink):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softshrink)
    _builtin_torch_fn = F.softshrink
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Softsign)
class QuantizedSoftsign(_DispatchMixin, QuantizationMixin, nn.Softsign):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Softsign)
    _builtin_torch_fn = F.softsign
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.SyncBatchNorm)
class QuantizedSyncBatchNorm(QuantizationMixin, nn.SyncBatchNorm):
    __doc__ = _generate_docstring(parent_cls=nn.SyncBatchNorm)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        with self._patch_quantized_parameters():
            output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.Tanh)
class QuantizedTanh(_DispatchMixin, QuantizationMixin, nn.Tanh):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Tanh)
    _builtin_torch_fn = torch.tanh
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Tanhshrink)
class QuantizedTanhshrink(_DispatchMixin, QuantizationMixin, nn.Tanhshrink):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Tanhshrink)
    _builtin_torch_fn = F.tanhshrink
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Threshold)
class QuantizedThreshold(QuantizationMixin, nn.Threshold):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Threshold)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


# @QuantizationMixin.implements(nn.Transformer)
# class QuantizedTransformer(_DispatchMixin, QuantizationMixin, nn.Transformer):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerDecoder)
# class QuantizedTransformerDecoder(_DispatchMixin, QuantizationMixin, nn.TransformerDecoder):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerDecoderLayer)
# class QuantizedTransformerDecoderLayer(_DispatchMixin, QuantizationMixin, nn.TransformerDecoderLayer):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerEncoder)
# class QuantizedTransformerEncoder(_DispatchMixin, QuantizationMixin, nn.TransformerEncoder):
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerEncoderLayer)
# class QuantizedTransformerEncoderLayer(_DispatchMixin, QuantizationMixin, nn.TransformerEncoderLayer):
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.TripletMarginLoss)
class QuantizedTripletMarginLoss(
    _DispatchMixin, QuantizationMixin, nn.TripletMarginLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.TripletMarginLoss)
    _builtin_torch_fn = F.triplet_margin_loss
    __quant_init__ = QuantizationMixin.__ternary__


@QuantizationMixin.implements(nn.TripletMarginWithDistanceLoss)
class QuantizedTripletMarginWithDistanceLoss(
    QuantizationMixin, nn.TripletMarginWithDistanceLoss
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.TripletMarginWithDistanceLoss)
    __quant_init__ = QuantizationMixin.__ternary__

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            anchor = self.input_quantizers[0](anchor)

        if self.input_quantizers[1]:
            positive = self.input_quantizers[1](positive)

        if self.input_quantizers[2]:
            negative = self.input_quantizers[2](negative)

        output = super().forward(anchor, positive, negative)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.Unflatten)
class QuantizedUnflatten(QuantizationMixin, nn.Unflatten):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Unflatten)

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


@QuantizationMixin.implements(nn.Unfold)
class QuantizedUnfold(_DispatchMixin, QuantizationMixin, nn.Unfold):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Unfold)
    _builtin_torch_fn = F.unfold
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.Upsample)
class QuantizedUpsample(_DispatchMixin, QuantizationMixin, nn.Upsample):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.Upsample)
    _builtin_torch_fn = F.interpolate
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.UpsamplingBilinear2d)
class QuantizedUpsamplingBilinear2d(
    _DispatchMixin, QuantizationMixin, nn.UpsamplingBilinear2d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.UpsamplingBilinear2d)
    _builtin_torch_fn = F.interpolate
    __quant_init__ = QuantizationMixin.__unary__


@QuantizationMixin.implements(nn.UpsamplingNearest2d)
class QuantizedUpsamplingNearest2d(
    _DispatchMixin, QuantizationMixin, nn.UpsamplingNearest2d
):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.UpsamplingNearest2d)
    _builtin_torch_fn = F.interpolate
    __quant_init__ = QuantizationMixin.__unary__


if version.parse(torch.__version__) >= version.parse("2.1.0"):

    @QuantizationMixin.implements(nn.ZeroPad1d)
    class QuantizedZeroPad1d(QuantizationMixin, nn.ZeroPad1d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.ZeroPad1d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output


@QuantizationMixin.implements(nn.ZeroPad2d)
class QuantizedZeroPad2d(QuantizationMixin, nn.ZeroPad2d):
    # pylint: disable=missing-class-docstring
    __doc__ = _generate_docstring(parent_cls=nn.ZeroPad2d)
    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            input = self.input_quantizers[0](input)

        output = super().forward(input)

        if self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output


if version.parse(torch.__version__) >= version.parse("2.1.0"):

    @QuantizationMixin.implements(nn.ZeroPad3d)
    class QuantizedZeroPad3d(QuantizationMixin, nn.ZeroPad3d):
        # pylint: disable=missing-class-docstring
        __doc__ = _generate_docstring(parent_cls=nn.ZeroPad3d)
        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, input: torch.Tensor):  # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                input = self.input_quantizers[0](input)

            output = super().forward(input)

            if self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output
