# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import onnx
import onnx_ir
import onnxruntime as ort
import copy
import numpy as np
import aimet_onnx

from aimet_onnx import ir_utils
from aimet_onnx.common.onnx._utils import _get_all_constants_ir
from .models import models_for_tests


@pytest.mark.parametrize(
    "model",
    (
        models_for_tests.build_dummy_model(),
        models_for_tests.single_residual_model().model,
        models_for_tests.multi_input_model().model,
        models_for_tests.transposed_conv_model().model,
        models_for_tests.concat_model().model,
        models_for_tests.hierarchical_model().model,
        models_for_tests.elementwise_op_model().model,
        models_for_tests.instance_norm_model().model,
        models_for_tests.layernorm_model(),
        models_for_tests.matmul_with_constant_first_input(),
        models_for_tests.model_with_split_matmul(),
    ),
)
def test_remove_aimet_quantizers(model: onnx.ModelProto):
    sim = aimet_onnx.QuantizationSimModel(copy.deepcopy(model))
    ir_sim_model = onnx_ir.from_proto(sim.model.model)
    ir_utils.remove_aimet_quantizers(ir_sim_model)
    for node in ir_sim_model.graph.all_nodes():
        assert node.op_type != "QcQuantizeOp"

    proto: onnx.ModelProto = onnx_ir.to_proto(ir_sim_model)
    onnx.checker.check_model(proto)
    assert len(proto.graph.node) == len(model.graph.node)
    for orig_inp, new_inp in zip(model.graph.input, proto.graph.input):
        assert orig_inp.name == new_inp.name
    for orig_out, new_out in zip(model.graph.output, proto.graph.output):
        assert orig_out.name == new_out.name

    node_name_to_inputs_orig = {node.name: node.input for node in model.graph.node}
    node_name_to_inputs_new = {node.name: node.input for node in proto.graph.node}
    for node_name in node_name_to_inputs_orig:
        assert node_name in node_name_to_inputs_new
        assert node_name_to_inputs_orig[node_name] == node_name_to_inputs_new[node_name]

    dummy_input = aimet_onnx.utils.make_dummy_input(sim.model.model)
    sess_orig = ort.InferenceSession(model.SerializeToString())
    sess_new = ort.InferenceSession(proto.SerializeToString())
    orig_outs = sess_orig.run(None, dummy_input)
    new_outs = sess_new.run(None, dummy_input)
    for orig_out, new_out in zip(orig_outs, new_outs):
        assert np.allclose(orig_out, new_out, atol=1e-6)


def test_set_static_tensor_constant_after_constants_scan():
    """
    Given: A Constant node output whose const_value was cached by _get_all_constants_ir
    When: set_static_tensor overwrites it
    Then: The serialized Constant node holds the new data
    """
    weight = onnx.numpy_helper.from_array(np.ones(2, dtype=np.float32), "w")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Constant", [], ["w"], value=weight),
            onnx.helper.make_node("Add", ["x", "w"], ["y"]),
        ],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    model = onnx_ir.from_proto(onnx.helper.make_model(graph))
    constants = _get_all_constants_ir(model.graph)

    ir_utils.set_static_tensor(constants["w"], np.full(2, 7, dtype=np.float32))

    proto = onnx_ir.to_proto(model)
    (const_node,) = [n for n in proto.graph.node if n.op_type == "Constant"]
    assert np.array_equal(onnx.numpy_helper.to_array(const_node.attribute[0].t), [7, 7])


def test_static_shape_of_initializer():
    """
    Given: An initializer with a declared shape
    When: static_shape is called on its value
    Then: The concrete shape is returned as a tuple
    """
    weight = onnx.numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), "w")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["w"], ["y"])],
        "g",
        [],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3])],
        initializer=[weight],
    )
    model = onnx_ir.from_proto(onnx.helper.make_model(graph))
    constants = _get_all_constants_ir(model.graph)

    assert ir_utils.static_shape(constants["w"]) == (2, 3)


def test_static_shape_of_constant_output():
    """
    Given: A Constant node output whose shape is unset but whose const_value is cached
    When: static_shape is called on it
    Then: The shape is taken from the constant value
    """
    weight = onnx.numpy_helper.from_array(np.ones((4,), dtype=np.float32), "w")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Constant", [], ["w"], value=weight),
            onnx.helper.make_node("Identity", ["w"], ["y"]),
        ],
        "g",
        [],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4])],
    )
    model = onnx_ir.from_proto(onnx.helper.make_model(graph))
    constants = _get_all_constants_ir(model.graph)
    constants["w"].shape = None

    assert ir_utils.static_shape(constants["w"]) == (4,)


def test_static_shape_without_shape_returns_none():
    """
    Given: A value with no shape and no constant value
    When: static_shape is called on it
    Then: None is returned
    """
    value = onnx_ir.Value(name="v")

    assert ir_utils.static_shape(value) is None


def test_float_dtype_of_none_is_float32():
    """
    Given: No value
    When: float_dtype is called on None
    Then: float32 is returned
    """
    assert ir_utils.float_dtype(None) == np.float32


def test_float_dtype_of_untyped_value_is_float32():
    """
    Given: A value whose dtype is unset
    When: float_dtype is called on it
    Then: float32 is returned
    """
    assert ir_utils.float_dtype(onnx_ir.Value(name="v")) == np.float32


def test_float_dtype_of_float16_value():
    """
    Given: A value of type FLOAT16
    When: float_dtype is called on it
    Then: float16 is returned
    """
    value = onnx_ir.Value(name="v", type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT16))

    assert ir_utils.float_dtype(value) == np.float16
