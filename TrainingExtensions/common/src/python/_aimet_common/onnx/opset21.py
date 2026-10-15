# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from typing import Iterable, Optional
import onnx_ir
from . import opset19


class QuantizeLinear(opset19.QuantizeLinear):
    OPSET = 21
    SUPPORTED_DTYPES = {
        **opset19.QuantizeLinear.SUPPORTED_DTYPES,
        "int4": onnx_ir.DataType.INT4,
        "uint4": onnx_ir.DataType.UINT4,
        "int16": onnx_ir.DataType.INT16,
        "uint16": onnx_ir.DataType.UINT16,
    }

    @classmethod
    def make_node(
        cls,
        name: str,
        inputs: Iterable[onnx_ir.Value],
        output: onnx_ir.Value,
        dtype: str,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> onnx_ir.Node:
        cls._check_dtype(dtype)

        if axis is None and block_size is not None:
            raise RuntimeError(
                "axis must be specified if block_size is not None; "
                f"got axis={axis}, block_size={block_size}"
            )

        return onnx_ir.node(
            "QuantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            # NOTE: Don't pass output_dtype explicitly; ORT has a bug
            #       where per-tensor int8 QuantizeLinear
            #       fails with output_dtype explicitly specified as INT8
            # "output_dtype": cls._ir_dtype(dtype),
            attributes={"axis": axis, "block_size": block_size},
        )


class DequantizeLinear(opset19.DequantizeLinear):
    OPSET = 21
    SUPPORTED_DTYPES = {
        **opset19.DequantizeLinear.SUPPORTED_DTYPES,
        "int4": onnx_ir.DataType.INT4,
        "uint4": onnx_ir.DataType.UINT4,
        "int16": onnx_ir.DataType.INT16,
        "uint16": onnx_ir.DataType.UINT16,
    }

    @classmethod
    def make_node(
        cls,
        name: str,
        inputs: Iterable[onnx_ir.Value],
        output: onnx_ir.Value,
        dtype: str,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> onnx_ir.Node:
        cls._check_dtype(dtype)

        if axis is None and block_size is not None:
            raise RuntimeError(
                "axis must be specified if block_size is not None; "
                f"got axis={axis}, block_size={block_size}"
            )

        return onnx_ir.node(
            "DequantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            attributes={"axis": axis, "block_size": block_size},
        )
