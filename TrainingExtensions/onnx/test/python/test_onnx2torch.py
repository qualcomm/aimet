# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os

import numpy as np
import onnx_ir
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from torch import nn as nn

from aimet_onnx.adascale.model_converter import get_pt_block

# Importing onnx2torch_ext (done transitively by model_converter) registers the
# custom converters (ScatterElements, Flatten, NonZero, OneHot, Trilu, Clip)
# on the onnx2torch registry.
from .models import models_for_tests
from .utils import tmp_dir  # noqa: F401  pytest fixture


@pytest.fixture(autouse=True)
def _seed():
    """Seed RNGs so each test's random inputs are deterministic."""
    np.random.seed(0)
    torch.manual_seed(0)


class ScatterElementsModel(nn.Module):
    """Emits an ONNX ScatterElements node for the given reduce mode.

    ``reduce`` is a torch.scatter_reduce reduce name ("sum"/"prod"/"amax"/"amin")
    or ``None`` for a plain (overwrite) scatter. torch.onnx.export lowers these to
    a ScatterElements node whose ONNX ``reduction`` attribute the converter reads.
    """

    def __init__(self, axis: int, reduce: str | None):
        super().__init__()
        self.axis = axis
        self.reduce = reduce

    def forward(self, data, indices, updates):
        if self.reduce is None:
            return torch.scatter(data, self.axis, indices, updates)
        return data.scatter_reduce(
            self.axis,
            indices,
            updates,
            reduce=self.reduce,
            include_self=True,
        )


def _export_and_convert(model, inputs, tmp_dir, name, input_names):
    model.eval()
    onnx_path = os.path.join(tmp_dir, f"{name}.onnx")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        dynamo=True,
        verbose=False,
    )
    onnx_model = onnx_ir.load(onnx_path)
    pt_block, _ = get_pt_block(onnx_model, (input_names, ["output"]))
    return onnx_path, pt_block


def _run_ort(onnx_path, feed):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {name: t.numpy() for name, t in feed.items()})[0]


class ReduceL2Model(nn.Module):
    """Emits an ONNX ReduceL2 node (L2 norm over ``dim``)."""

    def __init__(self, dim, keepdim: bool):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, data):
        return torch.linalg.vector_norm(data, ord=2, dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_reduce_l2(dim, keepdim, tmp_dir):
    """Three-way equivalence for ReduceL2 across dim x keepdim."""

    data = torch.randn(2, 3, 4)

    model = ReduceL2Model(dim=dim, keepdim=keepdim)
    onnx_path, pt_block = _export_and_convert(
        model, (data,), tmp_dir, f"reduce_l2_dim{dim}_keep{keepdim}", ["data"]
    )

    # 1) original torch model, 2) ONNX Runtime, 3) onnx2torch-converted block.
    orig_out = model(data).detach().numpy()
    onnx_out = _run_ort(onnx_path, {"data": data})
    converted_out = pt_block(data).detach().numpy()

    np.testing.assert_allclose(orig_out, onnx_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(orig_out, converted_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reduce", [None, "sum", "prod", "amax", "amin"])
def test_scatter_elements(axis, reduce, tmp_dir):
    """Three-way equivalence across every axis x reduce mode.

    The original torch model, ONNX Runtime on the exported graph, and the
    onnx2torch-converted block must all agree. Duplicate indices along the
    scattered axis ensure the reduction actually folds multiple updates.
    """
    # (3, 3) data; indices duplicate along `axis` so the reduction actually folds.
    data = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    if axis == 0:
        indices = torch.tensor([[0, 0, 1], [0, 2, 1], [2, 2, 0]])
    else:
        indices = torch.tensor([[0, 0, 1], [2, 2, 0], [1, 1, 2]])
    updates = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])

    model = ScatterElementsModel(axis=axis, reduce=reduce)
    onnx_path, pt_block = _export_and_convert(
        model,
        (data, indices, updates),
        tmp_dir,
        f"scatter_ax{axis}_{reduce}",
        ["data", "indices", "updates"],
    )

    # 1) original torch model, 2) ONNX Runtime, 3) onnx2torch-converted block.
    orig_out = model(data, indices, updates).detach().numpy()
    onnx_out = _run_ort(
        onnx_path, {"data": data, "indices": indices, "updates": updates}
    )
    converted_out = pt_block(data, indices, updates).detach().numpy()

    np.testing.assert_array_equal(orig_out, onnx_out)
    np.testing.assert_array_equal(orig_out, converted_out)


def _assert_single_node_matches_ort(node, feed, output_type, initializers=()):
    """
    Assert the onnx2torch-converted one-node graph matches ONNX Runtime.
    """
    input_infos = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape
        )
        for name, arr in feed.items()
    ]
    graph = helper.make_graph(
        [node],
        "g",
        input_infos,
        [helper.make_tensor_value_info("output", output_type, None)],
        list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    sess = ort.InferenceSession(model.SerializeToString())
    onnx_out, *_ = sess.run(None, feed)
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (list(feed), ["output"]))
    converted_out = (
        pt_block(*(torch.from_numpy(arr) for arr in feed.values())).detach().numpy()
    )

    np.testing.assert_allclose(onnx_out, converted_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    "input_shape, axis",
    [
        ((2, 3), 2),  # converted forward errors out with stock onnx2torch
        ((2, 3, 4), 0),
        ((2, 3, 4), 2),
        ((2, 3, 4), -1),
    ],
)
def test_flatten(input_shape, axis, dtype):
    """Flatten matches ONNX Runtime for any axis."""
    node = helper.make_node("Flatten", ["x"], ["output"], axis=axis)
    feed = {"x": np.random.randn(*input_shape).astype(dtype)}
    _assert_single_node_matches_ort(
        node, feed, helper.np_dtype_to_tensor_dtype(feed["x"].dtype)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("input_shape", [(3, 4), (2, 3, 4)])
def test_nonzero(input_shape, dtype):
    """NonZero matches ONNX Runtime (output is (rank, num_nonzero))."""
    node = helper.make_node("NonZero", ["x"], ["output"])
    feed = {"x": (np.random.rand(*input_shape) > 0.5).astype(dtype)}
    _assert_single_node_matches_ort(node, feed, TensorProto.INT64)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_one_hot(axis, dtype):
    """OneHot matches ONNX Runtime across insertion axes."""
    values_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    node = helper.make_node("OneHot", ["ind", "depth", "values"], ["output"], axis=axis)
    feed = {"ind": np.random.randint(0, 4, (8, 2)).astype(np.int64)}
    initializers = [
        helper.make_tensor("depth", TensorProto.INT64, [], [4]),
        helper.make_tensor(
            "values", values_type, [2], np.array([0.0, 1.0], dtype=dtype)
        ),
    ]
    _assert_single_node_matches_ort(node, feed, values_type, initializers)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("upper", [0, 1])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_trilu(upper, k, dtype):
    """Trilu matches ONNX Runtime for upper/lower across diagonals."""
    node = helper.make_node("Trilu", ["x", "k"], ["output"], upper=upper)
    feed = {"x": np.random.randn(4, 5).astype(dtype)}
    initializers = [helper.make_tensor("k", TensorProto.INT64, [], [k])]
    _assert_single_node_matches_ort(
        node, feed, helper.np_dtype_to_tensor_dtype(feed["x"].dtype), initializers
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("exclusive", [0, 1])
@pytest.mark.parametrize("reverse", [None, 0, 1])
def test_cumsum(reverse, exclusive, dtype):
    """CumSum matches ONNX Runtime, including when ``reverse`` is omitted.

    The ONNX spec default for ``reverse`` is 0, but torch.onnx.export omits
    the attribute entirely when reverse=False (the common case). Stock
    onnx2torch defaults a missing ``reverse`` to 1, silently producing a
    reversed cumsum; ``reverse=None`` here exercises that omitted-attribute
    case.
    """
    node_kwargs = {"exclusive": exclusive}
    if reverse is not None:
        node_kwargs["reverse"] = reverse
    node = helper.make_node("CumSum", ["x", "axis"], ["output"], **node_kwargs)
    feed = {"x": np.random.randn(3, 4).astype(dtype)}
    initializers = [helper.make_tensor("axis", TensorProto.INT64, [], [1])]
    _assert_single_node_matches_ort(
        node, feed, helper.np_dtype_to_tensor_dtype(feed["x"].dtype), initializers
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (0.0, None),  # rejected by stock onnx2torch
        (None, 1.0),
        (-0.5, 0.5),
    ],
)
def test_clip(min_val, max_val, dtype):
    """Clip matches ONNX Runtime with an omitted optional min/max."""
    tensor_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inputs = ["x"]
    initializers = []
    for name, val in (("min", min_val), ("max", max_val)):
        if val is None:
            inputs.append("")  # omitted optional input
        else:
            inputs.append(name)
            initializers.append(
                helper.make_tensor(name, tensor_type, [], np.array([val], dtype=dtype))
            )
    node = helper.make_node("Clip", inputs, ["output"])
    feed = {"x": np.random.randn(3, 4).astype(dtype)}
    _assert_single_node_matches_ort(node, feed, tensor_type, initializers)


@pytest.mark.parametrize(
    "op_type, attrs, weight_shape",
    [
        # Depthwise causal conv, as in the gated-delta-net linear attention
        ("Conv", {"group": 4, "pads": [3, 0]}, (4, 1, 4)),
        ("Cos", {}, None),
        ("Sin", {}, None),
        ("Softplus", {}, None),
    ],
)
def test_opset22_converters(op_type, attrs, weight_shape):
    """
    Given: An opset 22 graph using an op whose schema was bumped at 22 (bfloat16 added)
    When: The model is converted to torch
    Then: The converted model matches ONNX Runtime, as the previous version's converter
          is registered at 22
    """
    feed = {"x": np.random.randn(1, 4, 16).astype(np.float32)}
    inputs, initializers = ["x"], []
    if weight_shape is not None:
        inputs.append("w")
        initializers.append(
            numpy_helper.from_array(
                np.random.randn(*weight_shape).astype(np.float32), "w"
            )
        )
    graph = helper.make_graph(
        [helper.make_node(op_type, inputs, ["output"], **attrs)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, feed["x"].shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    onnx_out, *_ = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (["x"], ["output"]))
    converted_out = pt_block(torch.from_numpy(feed["x"])).detach().numpy()

    np.testing.assert_allclose(onnx_out, converted_out, rtol=1e-5, atol=1e-5)


def _scan_model(
    body, scan_inputs, scan_outputs, opset, initializers=(), nodes=(), **attrs
):
    """Model running body over scan_inputs with one Scan, after nodes."""
    scan = helper.make_node(
        "Scan",
        [info.name for info in scan_inputs],
        [info.name for info in scan_outputs],
        body=body,
        **attrs,
    )
    graph = helper.make_graph(
        [*nodes, scan], "g", scan_inputs, scan_outputs, list(initializers)
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    return model


def _assert_scan_matches_ort(model, feed):
    onnx_outs = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    output_names = [output.name for output in model.graph.output]
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (list(feed), output_names))
    converted_outs = pt_block(*(torch.from_numpy(arr) for arr in feed.values()))
    for onnx_out, converted_out in zip(onnx_outs, converted_outs):
        assert np.allclose(
            onnx_out, converted_out.detach().numpy(), rtol=1e-5, atol=1e-6
        )


def _float_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_scan():
    """An opset 18 Scan whose body reads an initializer and a node output of the enclosing
    graph, and contains Transpose, which has no converter at the latest opset"""
    trip_count = 32
    body = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w2"], ["m"]),
            helper.make_node("Transpose", ["m"], ["t"], perm=[1, 0]),
            helper.make_node("Add", ["t", "w"], ["u"]),
            helper.make_node("Add", ["c", "u"], ["s"]),
            helper.make_node("Tanh", ["s"], ["c_out"]),
            helper.make_node("Mul", ["c_out", "x"], ["y"]),
        ],
        "body",
        [_float_info("c", [4, 4]), _float_info("x", [4, 4])],
        [_float_info("c_out", [4, 4]), _float_info("y", [4, 4])],
    )
    model = _scan_model(
        body,
        [_float_info("c0", [4, 4]), _float_info("xs", [trip_count, 4, 4])],
        [_float_info("c_final", [4, 4]), _float_info("ys", [trip_count, 4, 4])],
        opset=18,
        initializers=[
            helper.make_tensor("w", TensorProto.FLOAT, [4, 4], np.random.randn(16))
        ],
        nodes=[helper.make_node("Add", ["w", "w"], ["w2"])],
        num_scan_inputs=1,
    )
    feed = {
        "c0": np.random.randn(4, 4).astype(np.float32),
        "xs": np.random.randn(trip_count, 4, 4).astype(np.float32),
    }

    """
    When: The model is converted to torch
    Then: The converted model matches ORT, as the body converts with the model's opset
    """
    _assert_scan_matches_ort(model, feed)


def test_scan_with_if_reading_enclosing_graph():
    """A Scan body holding an If whose branch reads a tensor of the graph enclosing the Scan"""
    if_node = helper.make_node(
        "If",
        ["cond"],
        ["b"],
        then_branch=helper.make_graph(
            [helper.make_node("Add", ["x", "top"], ["t"])],
            "then",
            [],
            [_float_info("t", [4, 4])],
        ),
        else_branch=helper.make_graph(
            [helper.make_node("Neg", ["x"], ["e"])],
            "else",
            [],
            [_float_info("e", [4, 4])],
        ),
    )
    body = helper.make_graph(
        [
            if_node,
            helper.make_node("Add", ["c", "b"], ["c_out"]),
            helper.make_node("Relu", ["b"], ["y"]),
        ],
        "body",
        [_float_info("c", [4, 4]), _float_info("x", [4, 4])],
        [_float_info("c_out", [4, 4]), _float_info("y", [4, 4])],
    )
    model = _scan_model(
        body,
        [_float_info("c0", [4, 4]), _float_info("xs", [3, 4, 4])],
        [_float_info("c_final", [4, 4]), _float_info("ys", [3, 4, 4])],
        opset=18,
        nodes=[helper.make_node("Relu", ["top_in"], ["top"])],
        num_scan_inputs=1,
    )
    model.graph.input.extend(
        [
            _float_info("top_in", [4, 4]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
        ]
    )
    for cond in (True, False):
        feed = {
            "c0": np.random.randn(4, 4).astype(np.float32),
            "xs": np.random.randn(3, 4, 4).astype(np.float32),
            "top_in": np.random.randn(4, 4).astype(np.float32),
            "cond": np.array([cond]),
        }
        _assert_scan_matches_ort(model, feed)


@pytest.mark.parametrize(
    "model_fn, cond_names",
    [
        (models_for_tests.if_model, ["cond"]),
        (models_for_tests.weighted_if_model, ["cond"]),
        (models_for_tests.chained_if_model, ["cond0", "cond1"]),
        (models_for_tests.predicated_experts_model, ["route0", "route1"]),
    ],
)
def test_if_matches_ort(model_fn, cond_names):
    """
    Given: A model with Ifs, whose branches read tensors and weights of the enclosing graph
    When: The model is converted to torch
    Then: The converted model matches ORT for every combination of conds, i.e. it runs the
          branch ORT runs
    """
    model = model_fn()
    x = np.random.randn(
        *[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    )
    feed_names = [model.graph.input[0].name, *cond_names]
    output_name = model.graph.output[0].name
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (feed_names, [output_name]))
    session = ort.InferenceSession(model.SerializeToString())
    for conds in itertools.product([True, False], repeat=len(cond_names)):
        feed = {feed_names[0]: x.astype(np.float32)}
        feed.update({name: np.array([c]) for name, c in zip(cond_names, conds)})
        onnx_out = session.run(None, feed)[0]
        converted_out = pt_block(*(torch.from_numpy(arr) for arr in feed.values()))
        np.testing.assert_allclose(
            onnx_out, converted_out.detach().numpy(), rtol=1e-5, atol=1e-6
        )


def test_if_branch_weights_convert_to_linear():
    """
    Given: Experts in If branches, whose MatMuls read top-level weights
    When: The block is converted to torch
    Then: Each expert MatMul is an nn.Linear mapped to its weight, so AdaScale optimizes it
    """
    pt_block, param_map = get_pt_block(
        onnx_ir.from_proto(models_for_tests.predicated_experts_model()),
        (["x", "route0", "route1"], ["y1"]),
    )
    modules = dict(pt_block.named_modules())
    for module_name, weight in (
        ("if_0.then_branch.t0_mm", "w0"),
        ("if_1.then_branch.t1_mm", "w1"),
    ):
        assert isinstance(modules[module_name], nn.Linear)
        assert param_map[module_name] == weight


def _reduce_model(op_type, axes_source, axes, **attrs):
    """Opset 18 model applying op_type to x over axes read from axes_source."""
    x_info = _float_info("x", [2, 3, 4])
    axes_array = np.array(axes, dtype=np.int64)
    inputs, nodes, initializers = [x_info], [], []
    if axes_source == "graph_input":
        inputs.append(
            helper.make_tensor_value_info("axes", TensorProto.INT64, [len(axes)])
        )
    elif axes_source == "identity":
        initializers.append(numpy_helper.from_array(axes_array, "axes_init"))
        nodes.append(helper.make_node("Identity", ["axes_init"], ["axes"]))
    else:
        initializers.append(numpy_helper.from_array(axes_array, "axes"))
    nodes.append(helper.make_node(op_type, ["x", "axes"], ["output"], **attrs))
    graph = helper.make_graph(
        nodes, "g", inputs, [_float_info("output", None)], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


@pytest.mark.parametrize("op_type", ["ReduceMean", "ReduceL2", "ReduceMax"])
@pytest.mark.parametrize("axes_source", ["initializer", "identity", "graph_input"])
@pytest.mark.parametrize("axes", [[1], [-1], [0, 2]])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_axes(op_type, axes_source, axes, keepdims):
    """
    Given: An opset 18 reduction whose axes input is an initializer, or is only known
           at runtime (graph input or Identity output)
    When: The model is converted to torch
    Then: The converted model reduces over the given axes, matching ORT
    """
    model = _reduce_model(op_type, axes_source, axes, keepdims=keepdims)
    feed = {"x": np.random.randn(2, 3, 4).astype(np.float32)}
    if axes_source == "graph_input":
        feed["axes"] = np.array(axes, dtype=np.int64)
    onnx_out, *_ = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (list(feed), ["output"]))
    converted_out = pt_block(*(torch.from_numpy(arr) for arr in feed.values()))

    np.testing.assert_allclose(
        onnx_out, converted_out.detach().numpy(), rtol=1e-6, atol=1e-6
    )


def test_reduce_l2_noop_with_empty_axes():
    """
    Given: An opset 18 ReduceL2 with empty axes and noop_with_empty_axes=1
    When: The model is converted to torch
    Then: The converted model returns |x|, since only the reduction step is a no-op

    Compared against the ONNX reference implementation, as ORT returns x unchanged.
    """
    model = _reduce_model("ReduceL2", "initializer", [], noop_with_empty_axes=1)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    pt_block, _ = get_pt_block(onnx_ir.from_proto(model), (["x"], ["output"]))

    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {"x": x})[0], np.abs(x)
    )
    np.testing.assert_array_equal(pt_block(torch.from_numpy(x)).numpy(), np.abs(x))
