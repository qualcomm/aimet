# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin, too-many-lines
"""Affine quantizers"""

from itertools import chain, repeat
from typing import (
    Any,
    Dict,
    List,
    Optional,
    overload,
    Protocol,
    runtime_checkable,
    Tuple,
)
import contextlib
import functools

import torch
from torch import nn

from aimet_torch.v2.utils import (
    patch_attr,
    _is_expandable,
    StatisticsNotFoundError,
    docstring,
)
from aimet_torch.v2.quantization.encoding_analyzer import (
    EncodingAnalyzer,
    MinMaxEncodingAnalyzer,
    _flag_extreme_min_max,
)
from aimet_torch.v2.quantization.affine import AffineEncoding, GroupedBlockEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine.backends import (
    quantize,
    quantize_dequantize,
    dequantize,
    torch_builtins,
    _derive_qmin_qmax,
)
from aimet_torch.v2.utils import ste_round
from aimet_torch.v2.deepspeed_utils import SafeGatheredParameters
from aimet_common.quantsim import _get_minimum_scale
from ._utils import _GridMixin, _register_signature


__all__ = [
    "AffineQuantizerBase",
    "Dequantize",
    "GroupedBlockQuantizeDequantize",
    "MinMaxQuantizer",
    "Quantize",
    "QuantizeDequantize",
    "ScaleOffsetQuantizer",
]


class AffineQuantizerBase(QuantizerBase, _GridMixin):  # pylint: disable=too-many-instance-attributes
    """
    Base class for linear quantization modules.

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)

    """

    _init_signatures = []

    @overload
    @_register_signature(_init_signatures)
    def __init__(
        self,
        shape,
        qmin: int,
        qmax: int,
        symmetric: bool,
        encoding_analyzer: EncodingAnalyzer = None,
        block_size: Optional[Tuple[int, ...]] = None,
        zero_point_shift: Optional[float] = None,
    ): ...

    @overload
    @_register_signature(_init_signatures)
    def __init__(
        self,
        shape,
        bitwidth: int,
        symmetric: bool,
        encoding_analyzer: EncodingAnalyzer = None,
        block_size: Optional[Tuple[int, ...]] = None,
        zero_point_shift: Optional[float] = None,
    ): ...

    def __init__(self, shape, *args, **kwargs):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        full_args = (shape, *args)

        # Pad positional args with None's such that len(args) == 6
        args = tuple(chain(args, repeat(None, 6 - len(args))))
        arg0 = kwargs.pop("qmin", kwargs.pop("bitwidth", args[0]))
        arg1 = kwargs.pop("qmax", args[1])

        if arg1 is not None and not isinstance(arg1, bool):
            # (arg0, arg1, arg2) == (qmin, qmax, symmetric)
            qmin, qmax = arg0, arg1
            symmetric = kwargs.pop("symmetric", args[2])

            if (qmin is None) or (qmax is None) or (symmetric is None):
                raise self._arg_parsing_error(full_args, kwargs)

            encoding_analyzer = kwargs.pop("encoding_analyzer", args[3])
            block_size = kwargs.pop("block_size", args[4])
            zero_point_shift = kwargs.pop("zero_point_shift", args[5])
        else:
            # (arg0, arg1) == (bitwidth, symmetric)
            bitwidth = arg0
            symmetric = kwargs.pop("symmetric", args[1])

            if (bitwidth is None) or (symmetric is None):
                raise self._arg_parsing_error(full_args, kwargs)

            # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
            qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=symmetric)
            encoding_analyzer = kwargs.pop("encoding_analyzer", args[2])
            block_size = kwargs.pop("block_size", args[3])
            zero_point_shift = kwargs.pop("zero_point_shift", args[4])

        assert qmin is not None
        assert qmax is not None

        if kwargs:
            cls = type(self).__qualname__
            unexpected_keys = ", ".join(kwargs.keys())
            raise TypeError(
                f"{cls}.__init__ got unexpected keyword argument: {unexpected_keys}"
            )

        if qmin >= qmax:
            raise ValueError(
                f"qmax should be strictly larger than qmin. Got qmax={qmax}, qmin={qmin}"
            )

        self.qmin = qmin
        self.qmax = qmax
        self._symmetric = symmetric
        self.block_size = block_size

        self.zero_point_shift = zero_point_shift or 0.0
        if self.zero_point_shift not in [0.0, 0.5]:
            raise ValueError(
                f"zero_point_shift should be 0.0 or 0.5. Got {self.zero_point_shift}"
            )

        self.encoding_analyzer = encoding_analyzer or MinMaxEncodingAnalyzer(
            torch_builtins.get_encoding_shape_with_blocks(self.shape, self.block_size)
        )

        if self.block_size is None and not _is_expandable(
            self.encoding_analyzer.observer.shape, self.shape
        ):
            raise RuntimeError(
                f"Encoding analyzer of shape {self.encoding_analyzer.observer.shape} "
                f"is incompatible with quantizer of shape {self.shape}."
            )

        self._reparametrize_to_min_max()
        self._is_overwrite_allowed.update(
            {
                "min": True,
                "max": True,
            }
        )

    def _is_scale_offset_quantizer(self):
        return "scale" in self._parameters and "offset" in self._parameters

    def _is_min_max_quantizer(self):
        return "min" in self._parameters and "max" in self._parameters

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            if (name in ("min", "max") and self._is_scale_offset_quantizer()) or (
                name in ("scale", "offset") and self._is_min_max_quantizer()
            ):
                param_names = "/".join(self._parameters.keys())
                msg = (
                    f"'{type(self).__qualname__}' object has no attribute '{name}' "
                    f"because it's parametrized with {param_names}. "
                    f"To get '{name}' of this quantizer, use qtzr.get_{name}() instead. "
                    "To assign a new input range to this quantizer, use qtzr.set_range() instead"
                )
                raise AttributeError(msg) from e

            raise e

    def _reparametrize_to_scale_offset(self):
        # pylint: disable=attribute-defined-outside-init
        if self._is_scale_offset_quantizer():
            return

        is_initialized = self.is_initialized()

        self.register_quantization_parameter(
            "scale", nn.Parameter(torch.ones(self.shape))
        )
        self.register_quantization_parameter(
            "offset",
            None
            if self.symmetric
            else torch.nn.Parameter(
                _get_symmetric_offset(
                    self.qmin, self.qmax, self.shape, torch.float32, "cpu"
                )
            ),
        )

        if self._is_min_max_quantizer():
            min = self._parameters.pop("min")
            max = self._parameters.pop("max")
            self.requires_grad_(min.requires_grad or max.requires_grad)
            # NOTE: Only follow the device, but NOT the dtype of min & max.
            #       Scale & offset should be always kept in float32 for numerical stability
            self.to(device=min.device, dtype=torch.float32)

            if is_initialized:
                self.set_range(min, max)

        # Dummy attribute for runtime-checkable protocols in python >=3.12
        if hasattr(self, "_is_min_max"):
            delattr(self, "_is_min_max")
        setattr(self, "_is_scale_offset", True)

    def _reparametrize_to_min_max(self):
        # pylint: disable=attribute-defined-outside-init
        if self._is_min_max_quantizer():
            return

        is_initialized = self.is_initialized()

        self.register_quantization_parameter(
            "min", nn.Parameter(-torch.ones(self.shape))
        )
        self.register_quantization_parameter(
            "max", nn.Parameter(torch.ones(self.shape))
        )

        if self._is_scale_offset_quantizer():
            scale = self._parameters.pop("scale")
            offset = self._parameters.pop("offset")
            self.requires_grad_(
                scale.requires_grad or getattr(offset, "requires_grad", False)
            )
            self.to(device=scale.device, dtype=scale.dtype)

            if is_initialized:
                min, max = _get_min_max(scale, offset, self.qmin, self.qmax)
                self.set_range(min, max)

        # Dummy attribute for runtime-checkable protocols in python >=3.12
        if hasattr(self, "_is_scale_offset"):
            delattr(self, "_is_scale_offset")
        setattr(self, "_is_min_max", True)

    def get_min(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :param dtype: dtype of the computed min. Use of self.min.dtype by default.
        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        return self.get_scale(dtype) * (
            self.get_offset(dtype) + self.qmin + self.zero_point_shift
        )

    def get_max(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :param dtype: dtype of the computed max. Use of self.min.dtype by default.
        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        return self.get_scale(dtype) * (
            self.get_offset(dtype) + self.qmax + self.zero_point_shift
        )

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.
        Return None if the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed scale

        Returns:
            Quantization scale
        """
        if not self.is_initialized():
            return None

        dtype = dtype or torch.float32

        if self._is_scale_offset_quantizer():
            scale = self.scale
        else:
            num_steps = self.qmax - self.qmin
            scale = (self.max.to(dtype) - self.min.to(dtype)) / num_steps

        minimum_scale = _get_minimum_scale(self.qmax - self.qmin)
        return scale.abs().clamp_min_(minimum_scale).to(dtype)

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.
        Return None if the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed offset

        Returns:
            Quantization offset
        """
        return self._get_offset(dtype=dtype)

    def _get_offset(self, scale=None, dtype=None) -> Optional[torch.Tensor]:
        if not self.is_initialized():
            return None

        dtype = dtype or torch.float32
        device = next(p.device for p in self.parameters())

        if self.symmetric:
            offset = _get_symmetric_offset(
                self.qmin, self.qmax, self.shape, dtype, device
            )
        elif self._is_scale_offset_quantizer():
            offset = ste_round(self.offset)
        else:
            scale = scale if scale is not None else self.get_scale(dtype)
            offset = torch.clamp(
                ste_round(self.min / scale) - self.qmin, -self.qmax, -self.qmin
            )

        return offset.to(dtype)

    @torch.no_grad()
    def set_range(
        self,
        min: Optional[torch.Tensor],
        max: Optional[torch.Tensor],
    ):
        """
        Set quantization parameters to the given min-max range.
        If only one of min or max is set to None, this function will fix only one end of the encoding range.
        If both are set to None, this function is no-op.

        Args:
            min (torch.Tensor | None): Minimum value of quantization encoding range
            max (torch.Tensor | None): Maximum value of quantization encoding range
        """
        if min is None and max is None:
            return

        if min is not None and not isinstance(min, torch.Tensor):
            min = torch.tensor(min)

        if max is not None and not isinstance(max, torch.Tensor):
            max = torch.tensor(max)

        if self._is_min_max_quantizer():
            with SafeGatheredParameters(
                self.parameters(recurse=False), modifier_rank=0
            ):
                if min is not None:
                    self.min.copy_(min)
                if max is not None:
                    self.max.copy_(max)
        else:
            if min is None:
                min = self.get_min()

            if max is None:
                max = self.get_max()

            if min is None or max is None:
                raise ValueError(
                    f"{type(self).__name__}.set_range(...) with partial input, "
                    "such as `set_range(min=0.0, max=None)` or `set_range(min=None, max=1.0), "
                    "cannot be supported for quantizers parametrized with scale & offset."
                )

            # Compute scale/offset with float32 for numerical stability
            scale, offset = _get_scale_offset(
                min.to(torch.float32),
                max.to(torch.float32),
                qmin=self.qmin,
                qmax=self.qmax,
                symmetric=self.symmetric,
            )

            with SafeGatheredParameters(
                self.parameters(recurse=False), modifier_rank=0
            ):
                self.scale.copy_(scale)
                if not self.symmetric:
                    self.offset.copy_(offset)

    def get_encodings(self) -> Optional[AffineEncoding]:
        """
        Return the quantizer's encodings as an AffineEncoding object
        """
        if self.is_initialized():
            scale = self.get_scale(dtype=torch.float32)
            offset = self._get_offset(scale=scale, dtype=torch.float32)

            return AffineEncoding(
                scale,
                offset,
                self.qmin,
                self.qmax,
                self._symmetric,
                self.block_size,
                self.zero_point_shift,
            )
        return None

    @classmethod
    def from_encodings(cls, encodings: AffineEncoding) -> "AffineQuantizerBase":
        if not isinstance(encodings, AffineEncoding):
            raise TypeError(f"Expected {AffineEncoding}; got {type(encodings)}")

        qtzr = cls(
            shape=encodings.scale.shape,
            qmin=encodings.qmin,
            qmax=encodings.qmax,
            symmetric=encodings.symmetry,
            block_size=encodings.block_size,
        )

        qtzr.set_range(encodings.min, encodings.max)

        return qtzr

    @torch.no_grad()
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """
        # pylint: disable=redefined-builtin, protected-access

        if not self.is_initialized():
            return None

        return self.get_encodings()._to_legacy_format()

    @torch.no_grad()
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            ...
        ]
        """

        def str_to_bool(s: str):
            s = s.lower()
            if s == "false":
                return False
            if s == "true":
                return True
            raise ValueError

        bitwidth = encodings[0]["bitwidth"]
        symmetric = str_to_bool(encodings[0]["is_symmetric"])
        # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
        self.qmin, self.qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=symmetric)
        self.symmetric = symmetric
        # Note: We can only accurately infer signed-ness in the symmetric case, but AIMET uses unsigned for asymmetric
        min_ = torch.tensor([e["min"] for e in encodings]).view(self.shape)
        max_ = torch.tensor([e["max"] for e in encodings]).view(self.shape)
        self.set_range(min_, max_)

    def extra_repr(self) -> str:
        extra_repr = f"shape={self.shape}"

        if self.block_size is not None:
            extra_repr += f", block_size={self.block_size}"

        extra_repr += (
            f", qmin={self.qmin}, qmax={self.qmax}, symmetric={self.symmetric}"
        )
        return extra_repr

    @property
    def symmetric(self) -> bool:
        """
        Indicates whether this quantizer uses symmetric quantization
        """
        if self._is_min_max_quantizer():
            return self._symmetric

        return self.offset is None

    @symmetric.setter
    def symmetric(self, symmetric: bool):
        """
        Set the quantizer symmetry

        :param symmetric: If True, use symmetric encodings. Else, use asymmetric encodings
        """
        if self._is_min_max_quantizer():
            self._symmetric = symmetric
            return

        if symmetric and not self.symmetric:
            self.offset = None
            return

        if not symmetric and self.symmetric:
            offset = _get_symmetric_offset(
                self.qmin, self.qmax, self.shape, self.scale.dtype, self.scale.device
            )
            self.offset = torch.nn.Parameter(
                offset, requires_grad=self.scale.requires_grad
            )

    @property
    @docstring(_GridMixin._get_bitwidth.__doc__)
    def bitwidth(self) -> int:  # pylint: disable=missing-function-docstring
        return self._get_bitwidth()

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        self._set_bitwidth(bitwidth)

    @property
    @docstring(_GridMixin._get_signed.__doc__)
    def signed(self) -> bool:  # pylint: disable=missing-function-docstring
        return self._get_signed()

    @signed.setter
    def signed(self, signed: bool):
        self._set_signed(signed)

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not any(self._is_overwrite_allowed.values()):
            yield
            return

        original_forward = self.forward
        shape = self.shape

        try:
            dtype, device = next((p.dtype, p.device) for p in self.parameters())
        except StopIteration as e:
            raise RuntimeError from e

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            input = input.as_subclass(torch.Tensor)
            expanded_input = torch_builtins.reshape_tensor_for_blocks(
                input, shape, self.block_size
            )
            batch_statistics = self.encoding_analyzer.update_stats(expanded_input)
            num_steps = self.qmax - self.qmin
            if self.zero_point_shift == 0.5:
                num_steps -= 1
            dynamic_min, dynamic_max = (
                self.encoding_analyzer.compute_encodings_from_stats(
                    batch_statistics, num_steps, self.symmetric
                )
            )
            if self.block_size is not None:
                dynamic_min = dynamic_min.view(shape)
                dynamic_max = dynamic_max.view(shape)
            dynamic_min = dynamic_min.to(dtype=dtype, device=device).expand(shape)
            dynamic_max = dynamic_max.to(dtype=dtype, device=device).expand(shape)

            if self._is_min_max_quantizer():
                with (
                    patch_attr(self, "min", dynamic_min),
                    patch_attr(self, "max", dynamic_max),
                ):
                    ret = original_forward(input)
            else:
                # Compute scale/offset with float32 for numerical stability
                dynamic_scale, dynamic_offset = _get_scale_offset(
                    dynamic_min.to(torch.float32),
                    dynamic_max.to(torch.float32),
                    qmin=self.qmin,
                    qmax=self.qmax,
                    symmetric=self.symmetric,
                )
                with (
                    patch_attr(self, "scale", dynamic_scale),
                    patch_attr(self, "offset", dynamic_offset),
                ):
                    ret = original_forward(input)

            return ret

        self.encoding_analyzer.reset_stats()

        try:
            with patch_attr(self, "forward", forward_wrapper):
                yield
        except:  # pylint: disable=try-except-raise
            raise

        try:
            num_steps = self.qmax - self.qmin
            if self.zero_point_shift == 0.5:
                num_steps -= 1
            enc_min, enc_max = self.encoding_analyzer.compute_encodings(
                num_steps, self.symmetric
            )
            if self.block_size is not None:
                enc_min = enc_min.view(shape)
                enc_max = enc_max.view(shape)
            _flag_extreme_min_max(enc_min, enc_max)

        except StatisticsNotFoundError:
            return

        if not self._is_overwrite_allowed["min"]:
            enc_min = None

        if not self._is_overwrite_allowed["max"]:
            enc_max = None

        self.set_range(enc_min, enc_max)


def _get_symmetric_offset(qmin, qmax, shape, dtype, device):
    return torch.full(
        shape,
        fill_value=-round((qmin + qmax) / 2),
        requires_grad=False,
        dtype=dtype,
        device=device,
    )


def _get_min_max(
    scale: torch.Tensor, offset: Optional[torch.Tensor], qmin: int, qmax: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offset is None:
        offset = _get_symmetric_offset(
            qmin, qmax, scale.shape, torch.int32, scale.device
        )

    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32)

    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor(offset, dtype=torch.int32)

    out_dtype = scale.dtype
    scale = scale.to(torch.float32)
    offset = offset.to(torch.int32)

    min = scale * (offset + qmin)
    max = scale * (offset + qmax)
    return min.to(out_dtype), max.to(out_dtype)


def _get_scale_offset(
    min: torch.Tensor, max: torch.Tensor, qmin: int, qmax: int, symmetric: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_steps = qmax - qmin

    if not isinstance(min, torch.Tensor):
        min = torch.tensor(min, dtype=torch.float32)

    if not isinstance(max, torch.Tensor):
        max = torch.tensor(max, dtype=torch.float32)

    out_dtype = min.dtype
    min = min.to(torch.float32)
    max = max.to(torch.float32)

    scale = (max - min).div_(num_steps)

    if symmetric:
        offset = torch.full_like(
            min, fill_value=-round((qmin + qmax) / 2), requires_grad=False
        )
    else:
        offset = ste_round(min / scale) - qmin

    return scale.to(out_dtype), offset.to(out_dtype)


@runtime_checkable
class MinMaxQuantizer(Protocol):
    """
    Affine quantizer protocol parametrized with min and max
    """

    _is_min_max: Any

    shape: Tuple[int, ...]
    qmin: int
    qmax: int
    symmetric: bool


@runtime_checkable
class ScaleOffsetQuantizer(Protocol):
    """
    Affine quantizer protocol parametrized with scale and offset
    """

    _is_scale_offset: Any

    shape: Tuple[int, ...]
    qmin: int
    qmax: int
    symmetric: bool


class Quantize(AffineQuantizerBase):
    r"""Applies quantization to the input.

    Precisely,

    .. math::
        out = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    where :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)
        block_size (Tuple[int, ...], optional): Block size

    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`Quantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> q.is_initialized()
        False
        >>> with q.compute_encodings():
        ...     _ = q(input)
        ...
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[129.,  64., 255., 122.,   0., 192., 106.,  94., 255.,   0.],
                         [  0., 145., 181., 255., 144., 255., 194.,   0.,  74.,  86.],
                         [122.,   0., 255., 150.,  33., 103., 103.,   0.,  37., 255.],
                         [255., 111., 237., 218.,   0.,  49., 155., 255.,   0., 179.],
                         [  0.,  66., 255.,  89., 110.,  17.,  36.,  83., 255.,   0.]],
                        grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> q.is_initialized()
        False
        >>> q.min = torch.nn.Parameter(-torch.ones_like(q.min))
        >>> q.max = torch.nn.Parameter(torch.ones_like(q.max))
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[187., 186., 131.,   0., 203.,  64.,  80.,   0., 143., 152.],
                         [ 16.,   0., 255.,   0.,   0., 150.,   0., 255.,  32., 255.],
                         [255., 226.,   0., 255.,  55., 172.,   0., 255., 145., 255.],
                         [207., 146., 216., 238.,   0.,   0., 141., 178., 255., 188.],
                         [ 63.,  59.,  19., 162.,  30., 255., 109., 255.,   0., 255.]],
                        grad_fn=<AliasBackward0>)
    """

    # NOTE: Deepspeed has a bug where it will inadvertently patch __init__ method permanently
    #       unless each leaf class explicitly defines its own __init__ separately.
    #       As a temporary workaround, we define __init__ to avoid triggering this bug.
    # pylint: disable=useless-super-delegation
    def __init__(self, shape, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)

        if self.zero_point_shift != 0.0:
            raise RuntimeError("Nonzero quant shift not supported for Quantize")

    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        """Quantizes the input tensor

        Args:
            input (torch.Tensor): Input to quantize

        Returns:
            Quantized output

        """
        if not self.is_initialized():
            raise RuntimeError(
                "Failed to run Quantize since quantization parameters are not initialized."
                " Please initialize the quantization parameters using `compute_encodings()`."
            )

        encoding = self.get_encodings()

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        input = input.as_subclass(torch.Tensor)

        output = quantize(
            input,
            encoding.scale,
            encoding.offset,
            encoding.qmin,
            encoding.qmax,
            block_size=self.block_size,
        )
        output = output.as_subclass(QuantizedTensor)
        output.encoding = encoding
        return output


class QuantizeDequantize(AffineQuantizerBase):
    r"""Applies fake-quantization by quantizing and dequantizing the input.

    Precisely,

    .. math::
        out = (\overline{input} + offset) * scale

    where

    .. math::
        \overline{input} = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    and :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} &= (\overline{input}_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}\\
        \overline{input}_{j_0 \cdots j_{D-1}} &= clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    Args:
        shape (tuple): Shape of the quantization parameters
        bitwidth (int): Quantization bitwidth
        symmetric (bool): If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        encoding_analyzer (EncodingAnalyzer, optional): Encoding analyzer for calibrating quantization encodings
                                                        (default: absolute min-max encoding analyzer)
        block_size (Tuple[int, ...], optional): Block size

    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`QuantizeDequantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 2), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> qdq.is_initialized()
        False
        >>> with qdq.compute_encodings():
        ...     _ = qdq(input)
        ...
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[-0.2771,  0.3038,  1.0819,  0.9700,  0.9487, -0.1307,
                            -1.7894, -0.1709, -0.2212,  0.7741],
                           [-1.0295, -1.2265, -1.0295,  1.0564,  0.6177, -1.0386,
                            -0.0176, -2.6054,  1.8836, -0.1232],
                           [-0.8229,  0.5540,  0.3992, -0.2363,  1.2546, -1.0036,
                             0.2355,  0.1741,  1.6079,  0.6247],
                           [-1.0115,  1.2458,  0.9157, -1.4694, -0.0639, -0.2568,
                             0.0680,  1.6695,  0.7932, -0.1889],
                           [ 0.0158,  0.5695,  0.5220,  0.1977, -1.4475, -0.0424,
                            -1.1128, -0.8796, -0.1060,  1.5897]],
                          grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 2), bitwidth=8, symmetric=False, block_size=(1, 5))
        >>> qdq.is_initialized()
        False
        >>> qdq.min = torch.nn.Parameter(-torch.ones_like(qdq.min))
        >>> qdq.max = torch.nn.Parameter(torch.ones_like(qdq.max))
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[-0.6196, -0.9961,  0.0549, -0.6431,  1.0039, -0.8706,
                             1.0039,  0.4706, -0.2353,  0.8078],
                           [ 0.3451, -0.1176, -0.9961, -0.4549, -0.0549, -0.0471,
                            -0.5255, -0.2353,  1.0039, -0.9961],
                           [-0.4157,  0.0784,  0.5333,  0.1647, -0.9961, -0.9961,
                            -0.2118, -0.2196,  0.9176,  0.9490],
                           [ 1.0039, -0.7765,  0.4784, -0.8706,  1.0039,  0.6039,
                            -0.4157, -0.2118, -0.9961,  0.3137],
                           [ 1.0039,  0.3216, -0.2353, -0.7765, -0.9961,  0.8000,
                             1.0039,  0.4157,  0.4392,  0.4863]],
                          grad_fn=<AliasBackward0>)
    """

    # NOTE: Deepspeed has a bug where it will inadvertently patch __init__ method permanently
    #       unless each leaf class explicitly defines its own __init__ separately.
    #       As a temporary workaround, we define __init__ to avoid triggering this bug.
    # pylint: disable=useless-super-delegation
    def __init__(self, shape, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """Quantizes and dequantizes the input tensor

        Args:
            input (torch.Tensor): Input to quantize and dequantize

        Returns:
            Quantize-dequantized output

        """
        if not self.is_initialized():
            raise RuntimeError(
                "Failed to run QuantizeDequantize since quantization parameters are not initialized."
                " Please initialize the quantization parameters using `compute_encodings()`."
            )

        encoding = self.get_encodings()

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        input = input.as_subclass(torch.Tensor)

        output = quantize_dequantize(
            input,
            encoding.scale,
            encoding.offset,
            encoding.qmin,
            encoding.qmax,
            block_size=self.block_size,
            zero_point_shift=self.zero_point_shift,
        )
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output


class Dequantize(AffineQuantizerBase):  # pylint: disable=missing-class-docstring
    def forward(self, input):
        if not self.is_initialized():
            raise RuntimeError(
                "Failed to run Dequantize since quantization parameters are not initialized."
                " Please initialize the quantization parameters using `compute_encodings()`."
            )

        if self.zero_point_shift != 0.0:
            raise RuntimeError("Nonzero quant shift not supported for Dequantize")

        encoding = self.get_encodings()

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        input = input.as_subclass(torch.Tensor)

        output = dequantize(
            input, encoding.scale, encoding.offset, block_size=self.block_size
        )
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output


class GroupedBlockQuantizeDequantize(QuantizeDequantize):  # pylint: disable=too-many-ancestors
    """Class for performing Grouped Block Quantize Dequantize"""

    def __init__(
        self,
        shape,
        bitwidth: int,
        symmetric: bool,
        decompressed_bw: int,
        encoding_analyzer: EncodingAnalyzer = None,
        block_size: Optional[Tuple[int, ...]] = None,
        block_grouping: Optional[Tuple[int, ...]] = None,
    ):
        """
        Grouped Block Quantize Dequantize constructor.

        :param shape: Shape of the quantization parameters
        :type shape: tuple
        :param bitwidth: Quantization bitwidth
        :type bitwidth: int
        :param symmetric: If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        :type symmetric: bool
        :param decompressed_bw: Bitwidth used for decompression
        :type decompressed_bw: int
        :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings
                                  (default: absolute min-max encoding analyzer)
        :type encoding_analyzer: EncodingAnalyzer, optional
        :param block_size: Block size per dimension.
        :type block_size: Tuple
        :param block_grouping: Block grouping per dimension. If provided, every set of block_group scales will be
                               grouped together, and the maximum scale for all blocks in the group will be used to find
                               the scale in the decompressed_grid to be shared by all blocks in the group.
                               If no block_grouping is provided, default behavior uses a block group of 1 for all dims,
                               equivalent to Blockwise Quantization.
                               A value of -1 for a block group for a dimension is equivalent to grouping all blocks in
                               the dimension in one group. This is also equivalent to a block group value equal to the
                               number of blocks for that dimension.
        :type block_grouping: Tuple
        """
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer, block_size)
        self.decompressed_bw = decompressed_bw
        self.block_grouping = block_grouping
        if self.block_grouping is None:
            # Default to BQ behavior with 1 for all block grouping dims if not provided
            self.block_grouping = tuple(1 for _ in enumerate(self.shape))

        if block_grouping is not None:
            if len(block_grouping) != len(shape):
                raise RuntimeError(
                    f"Length of block grouping {block_grouping} must equal length of shape {shape}."
                )
            for idx, block_group in enumerate(block_grouping):
                if block_group != -1 and shape[idx] % block_group != 0:
                    raise RuntimeError(
                        f"Quantizer shape dimensions must divide evenly with corresponding block "
                        f"grouping values for shapes {shape} and block grouping {block_grouping}."
                    )

        if self.decompressed_bw < self.bitwidth:
            raise RuntimeError(
                f"Decompressed bitwidth {decompressed_bw} cannot be smaller than self.bitwidth "
                f"{bitwidth}"
            )

        if not symmetric:
            raise RuntimeError(
                "GroupedBlockQuantizeDequantize only supports symmetric quantization."
            )

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        r"""
        Compute quantization scale to be used for forward pass.
        Overrides QuantizeDequantize self.get_scale() to apply the grouped block algorithm for calculating modified
        scales.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: Updated scale
        """
        lpbq_scale, _ = self._get_scale(dtype)
        return lpbq_scale

    def get_per_channel_scale(self, dtype=None) -> Optional[torch.Tensor]:
        r"""
        Returns per-channel scale such that

        :math:`scale = per_chanel_scale * per_block_int_scale`
        """
        raw_scale = super().get_scale(dtype)
        if raw_scale is None:
            return None
        return self._get_per_channel_scale(raw_scale)

    def _get_scale(
        self, dtype=None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        raw_scale = super().get_scale(dtype)
        if raw_scale is None:
            return None, None

        per_channel_scale = self._get_per_channel_scale(raw_scale)

        lpbq_scale = quantize_dequantize(
            tensor=raw_scale,
            scale=per_channel_scale,
            offset=torch.zeros_like(per_channel_scale),
            qmin=1,
            qmax=2 ** (self.decompressed_bw - self.bitwidth),
            block_size=self.block_grouping,
        )
        return lpbq_scale, per_channel_scale

    def _get_per_channel_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        per_channel_scale_shape = [
            s_dim // group_size if group_size != -1 else 1
            for s_dim, group_size in zip(raw_scale.shape, self.block_grouping)
        ]
        reshaped_scale = torch_builtins.reshape_tensor_for_blocks(
            raw_scale, per_channel_scale_shape, self.block_grouping
        )
        max_scale = torch.amax(
            reshaped_scale, dim=tuple(range(1, reshaped_scale.dim(), 2))
        )
        per_channel_scale = max_scale / 2 ** (self.decompressed_bw - self.bitwidth)
        return per_channel_scale

    def get_encodings(self) -> Optional[GroupedBlockEncoding]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """
        if self.is_initialized():
            lpbq_scale, per_channel_scale = self._get_scale(dtype=torch.float32)
            return GroupedBlockEncoding(
                scale=lpbq_scale,
                offset=self.get_offset(dtype=torch.float32),
                bitwidth=self.bitwidth,
                signed=self.signed,
                symmetry=self.symmetric,
                block_size=self.block_size,
                block_grouping=self.block_grouping,
                decompressed_bw=self.decompressed_bw,
                per_channel_scale=per_channel_scale,
            )
        return None

    @classmethod
    def from_encodings(
        cls, encodings: GroupedBlockEncoding
    ) -> "GroupedBlockQuantizeDequantize":
        if not isinstance(encodings, GroupedBlockEncoding):
            raise TypeError(f"Expected {GroupedBlockEncoding}; got {type(encodings)}")

        qtzr = cls(
            shape=encodings.scale.shape,
            bitwidth=encodings.bitwidth,
            symmetric=encodings.symmetry,
            decompressed_bw=encodings.decompressed_bw,
            block_size=encodings.block_size,
            block_grouping=encodings.block_grouping,
        )

        qtzr.set_range(encodings.min, encodings.max)

        return qtzr
