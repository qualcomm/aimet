# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from packaging.version import parse
import onnx
import onnx_ir
from . import opset21


class QuantizeLinear(opset21.QuantizeLinear):
    OPSET = 23
    SUPPORTED_DTYPES = {
        **opset21.QuantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.18.0"):
        SUPPORTED_DTYPES.update({"float4e2m1": onnx_ir.DataType.FLOAT4E2M1})


class DequantizeLinear(opset21.DequantizeLinear):
    OPSET = 23
    SUPPORTED_DTYPES = {
        **opset21.DequantizeLinear.SUPPORTED_DTYPES,
    }

    if parse(onnx.__version__) >= parse("1.18.0"):
        SUPPORTED_DTYPES.update({"float4e2m1": onnx_ir.DataType.FLOAT4E2M1})
