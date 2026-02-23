# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
from __future__ import annotations
from collections import namedtuple
from typing import Optional, Mapping
import torch
import onnx


class _finfo(
    namedtuple("_finfo", ("exponent_bits", "mantissa_bits", "finite", "unsigned_zero"))
):
    def to_torch_dtype(self) -> Optional[torch.dtype]:
        return _finfo_to_torch_dtype.get(self)

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype) -> "_finfo":
        try:
            return _torch_dtype_to_finfo[dtype]
        except KeyError as e:
            msg = " ".join(
                [
                    f"Expected dtype to be one of {list(_torch_dtype_to_finfo.keys())};",
                    f"got {dtype}",
                ]
            )
            raise ValueError(msg) from e

    def to_onnx_dtype(self) -> onnx.TensorProto.DataType | None:
        return _finfo_to_onnx_dtype.get(self)

    @classmethod
    def from_onnx_dtype(cls, dtype: int) -> "_finfo":
        try:
            return _onnx_dtype_to_finfo[dtype]
        except KeyError as e:
            msg = " ".join(
                [
                    f"Expected dtype to be one of {list(_onnx_dtype_to_finfo.keys())};",
                    f"got {dtype}",
                ]
            )
            raise ValueError(msg) from e

    def to_str(self) -> str:
        torch_dtype = self.to_torch_dtype()

        if torch_dtype:
            _, typename = str(torch_dtype).split(".")
            return typename

        e, m, fn, uz = self
        fn = "fn" if fn else ""
        uz = "uz" if uz else ""

        return f"float{e + m + 1}_e{e}m{m}{fn}{uz}"

    def is_float16(self) -> bool:
        return self == _float16

    def is_bfloat16(self) -> bool:
        return self == _bfloat16

    @property
    def max(self) -> float:
        torch_dtype = self.to_torch_dtype()

        if torch_dtype:
            return torch.finfo(torch_dtype).max

        if self == _float4_e2m1fn:
            return 6.0

        if not self.finite and not self.unsigned_zero:
            return self._ieee_float_max_representable_value()

        raise RuntimeError(f"Maximum representable value of {self.to_str()} is unkown")

    def _ieee_float_max_representable_value(self):
        exponent_bits, mantissa_bits, _, _ = self
        exponent_max = 2**exponent_bits - 1
        exponent_bias = exponent_max // 2
        return (2 - 2**-mantissa_bits) * 2 ** (exponent_max - exponent_bias - 1)


_float16 = _finfo(exponent_bits=5, mantissa_bits=10, finite=False, unsigned_zero=False)
_bfloat16 = _finfo(exponent_bits=8, mantissa_bits=7, finite=False, unsigned_zero=False)

_finfo_to_torch_dtype: Mapping[_finfo, torch.dtype] = {
    _float16: torch.float16,
    _bfloat16: torch.bfloat16,
}

_finfo_to_onnx_dtype: Mapping[torch.dtype, _finfo] = {
    _float16: onnx.TensorProto.FLOAT16,
    _bfloat16: onnx.TensorProto.BFLOAT16,
}

_float8_e4m3fn = _finfo(
    exponent_bits=4, mantissa_bits=3, finite=True, unsigned_zero=False
)

if hasattr(torch, "float8_e4m3fn"):
    _finfo_to_torch_dtype.update({_float8_e4m3fn: torch.float8_e4m3fn})

if hasattr(onnx.TensorProto, "FLOAT8E4M3FN"):
    _finfo_to_onnx_dtype.update({_float8_e4m3fn: onnx.TensorProto.FLOAT8E4M3FN})

_float8_e4m3fnuz = _finfo(
    exponent_bits=4, mantissa_bits=3, finite=True, unsigned_zero=True
)

if hasattr(torch, "float8_e4m3fnuz"):
    _finfo_to_torch_dtype.update({_float8_e4m3fnuz: torch.float8_e4m3fnuz})

if hasattr(onnx.TensorProto, "FLOAT8E4M3FNUZ"):
    _finfo_to_onnx_dtype.update({_float8_e4m3fnuz: onnx.TensorProto.FLOAT8E4M3FNUZ})

_float8_e5m2 = _finfo(
    exponent_bits=5, mantissa_bits=2, finite=False, unsigned_zero=False
)

if hasattr(torch, "float8_e5m2"):
    _finfo_to_torch_dtype.update({_float8_e5m2: torch.float8_e5m2})

if hasattr(onnx.TensorProto, "FLOAT8E5M2"):
    _finfo_to_onnx_dtype.update({_float8_e5m2: onnx.TensorProto.FLOAT8E5M2})

_float8_e5m2fnuz = _finfo(
    exponent_bits=5, mantissa_bits=2, finite=True, unsigned_zero=True
)

if hasattr(torch, "float8_e5m2fnuz"):
    _finfo_to_torch_dtype.update({_float8_e5m2fnuz: torch.float8_e5m2fnuz})

if hasattr(onnx.TensorProto, "FLOAT8E5M2FNUZ"):
    _finfo_to_onnx_dtype.update({_float8_e5m2fnuz: onnx.TensorProto.FLOAT8E5M2FNUZ})

_float4_e2m1fn = _finfo(
    exponent_bits=2, mantissa_bits=1, finite=True, unsigned_zero=False
)

if hasattr(onnx.TensorProto, "FLOAT4E2M1"):
    _finfo_to_onnx_dtype.update({_float4_e2m1fn: onnx.TensorProto.FLOAT4E2M1})


_torch_dtype_to_finfo: Mapping[torch.dtype, _finfo] = {
    torch_dtype: finfo for finfo, torch_dtype in _finfo_to_torch_dtype.items()
}
_onnx_dtype_to_finfo = {
    onnx_dtype: finfo for finfo, onnx_dtype in _finfo_to_onnx_dtype.items()
}
