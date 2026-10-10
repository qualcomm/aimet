# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import onnx_ir
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper
from torch import nn as nn

from aimet_onnx.adascale.model_converter import get_pt_block

# Importing onnx2torch_ext (done transitively by model_converter) registers the
# custom converters (ScatterElements, Flatten, NonZero, OneHot, Trilu, Clip)
# on the onnx2torch registry.
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
