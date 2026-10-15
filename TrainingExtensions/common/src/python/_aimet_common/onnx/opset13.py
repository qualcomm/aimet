# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from typing import Iterable, Optional
import onnx_ir
from . import opset10


class QuantizeLinear(opset10.QuantizeLinear):
    OPSET = 13

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
        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return onnx_ir.node(
            "QuantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            attributes={"axis": axis},
        )


class DequantizeLinear(opset10.DequantizeLinear):
    OPSET = 13

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
        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return onnx_ir.node(
            "DequantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            attributes={"axis": axis},
        )
