# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from packaging.version import parse
import onnx
import onnx_ir
from . import opset13


class QuantizeLinear(opset13.QuantizeLinear):
    OPSET = 19
    SUPPORTED_DTYPES = {
        **opset13.QuantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.15.0"):
        SUPPORTED_DTYPES.update(
            {
                "float8e4m3fn": onnx_ir.DataType.FLOAT8E4M3FN,
                "float8e4m3fnuz": onnx_ir.DataType.FLOAT8E4M3FNUZ,
                "float8e5m2": onnx_ir.DataType.FLOAT8E5M2,
                "float8e5m2fnuz": onnx_ir.DataType.FLOAT8E5M2FNUZ,
            }
        )


class DequantizeLinear(opset13.DequantizeLinear):
    OPSET = 19
    SUPPORTED_DTYPES = {
        **opset13.DequantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.15.0"):
        SUPPORTED_DTYPES.update(
            {
                "float8e4m3fn": onnx_ir.DataType.FLOAT8E4M3FN,
                "float8e4m3fnuz": onnx_ir.DataType.FLOAT8E4M3FNUZ,
                "float8e5m2": onnx_ir.DataType.FLOAT8E5M2,
                "float8e5m2fnuz": onnx_ir.DataType.FLOAT8E5M2FNUZ,
            }
        )
