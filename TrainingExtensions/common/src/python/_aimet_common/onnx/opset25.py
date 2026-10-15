# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from packaging.version import parse
import onnx
import onnx_ir
from . import opset23


class QuantizeLinear(opset23.QuantizeLinear):
    OPSET = 25
    SUPPORTED_DTYPES = {
        **opset23.QuantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.20.0"):
        SUPPORTED_DTYPES.update(
            {
                "int2": onnx_ir.DataType.INT2,
                "uint2": onnx_ir.DataType.UINT2,
            }
        )


class DequantizeLinear(opset23.DequantizeLinear):
    OPSET = 25
    SUPPORTED_DTYPES = {
        **opset23.DequantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.20.0"):
        SUPPORTED_DTYPES.update(
            {
                "int2": onnx_ir.DataType.INT2,
                "uint2": onnx_ir.DataType.UINT2,
            }
        )
