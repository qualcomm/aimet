# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnx
import onnx_ir
import pytest
from onnx import TensorProto, helper, numpy_helper

try:
    try:
        from aimet_onnx.common.onnx._utils import (
            _derive_data_movement_op_encodings,
            _derive_data_movement_op_encodings_ir,
        )
    except ImportError:
        from aimet_torch.common.onnx._utils import (
            _derive_data_movement_op_encodings,
            _derive_data_movement_op_encodings_ir,
        )
except ImportError:
    pass
else:

    def _masked_softmax_model(opset: int, bias: float) -> onnx.ModelProto:
        inputs = [
            helper.make_tensor_value_info("qk", TensorProto.FLOAT, [1, 2, 4, 4]),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, [1, 1, 1, 4]),
        ]
        output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 4, 4])
        initializers = [numpy_helper.from_array(np.array(bias, np.float32), "B")]
        if opset == 18:
            initializers.append(
                numpy_helper.from_array(np.array([-1], np.int64), "axes")
            )

        if opset == 17:
            reduce_min = helper.make_node(
                "ReduceMin",
                ["qk"],
                ["rmin"],
                axes=[-1],
                keepdims=1,
                name="rmin_node",
            )
        else:
            reduce_min = helper.make_node(
                "ReduceMin",
                ["qk", "axes"],
                ["rmin"],
                keepdims=1,
                name="rmin_node",
            )

        nodes = [
            reduce_min,
            helper.make_node("Add", ["rmin", "B"], ["inv"], name="add"),
            helper.make_node("Where", ["mask", "qk", "inv"], ["w"], name="where"),
            helper.make_node("Softmax", ["w"], ["out"], axis=-1, name="softmax"),
        ]
        graph = helper.make_graph(
            nodes, "masked_softmax", inputs, [output], initializers
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        onnx.checker.check_model(model)
        return model

    def _qk_encoding() -> dict:
        return {
            "name": "qk",
            "output_dtype": "uint8",
            "y_scale": 0.1,
            "y_zero_point": 0,
        }

    @pytest.mark.parametrize("opset", [17, 18])
    def test_masked_softmax_reducemin_not_derived_proto(opset: int) -> None:
        model = _masked_softmax_model(opset, bias=-10000.0)
        assert "rmin" not in _derive_data_movement_op_encodings(
            model, {"qk": dict(_qk_encoding())}
        )

    @pytest.mark.parametrize("opset", [17, 18])
    def test_masked_softmax_reducemin_not_derived_ir(opset: int) -> None:
        model = _masked_softmax_model(opset, bias=-10000.0)
        ir_model = onnx_ir.from_proto(model)
        assert "rmin" not in _derive_data_movement_op_encodings_ir(
            ir_model, {"qk": dict(_qk_encoding())}
        )

    @pytest.mark.parametrize("opset", [17, 18])
    def test_non_masked_softmax_reducemin_derived(opset: int) -> None:
        model = _masked_softmax_model(opset, bias=-1.0)
        assert "rmin" in _derive_data_movement_op_encodings(
            model, {"qk": dict(_qk_encoding())}
        )
        ir_model = onnx_ir.from_proto(model)
        assert "rmin" in _derive_data_movement_op_encodings_ir(
            ir_model, {"qk": dict(_qk_encoding())}
        )
