# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import pytest
from aimet_torch.nn.modules import custom
from aimet_torch.experimental.export._utils import (
    _is_torch_ao_q_node,
    _is_torch_ao_dq_node,
    _is_torch_ao_qdq_node,
)
import aimet_torch.quantization as Q
from aimet_torch.experimental.export.exported_program import (
    ExportedProgram as AIMETExportedProgram,
)


def _assert_graph_fully_quantized(
    graph: torch.fx.Graph,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    activation_dtype: torch.dtype,
):
    for node in graph.nodes:
        _assert_node_fully_quantized(
            node,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            activation_dtype=activation_dtype,
        )


def _assert_node_fully_quantized(
    node: torch.fx.Node,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    activation_dtype: torch.dtype,
):
    if node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    ):
        return

    for producer in node.all_input_nodes:
        assert _is_torch_ao_dq_node(producer)

    for user in node.users:
        if node.meta["val"].is_floating_point():
            assert _is_torch_ao_q_node(user)
        else:
            assert _is_torch_ao_dq_node(user)

        *_, qtype = user.args
        assert qtype == (
            weight_dtype
            if node.name.startswith("p_weight")
            else bias_dtype
            if node.name.startswith("p_bias")
            else activation_dtype
        )


def _get_min_max_range(node: torch.fx.Node) -> tuple[float, float]:
    if not _is_torch_ao_qdq_node(node):
        raise ValueError(f"Node {node.name} is not a quantize/dequantize node")

    _, scale, zero_point, qmin, qmax, *_ = node.args
    min = (qmin - zero_point) * scale
    max = (qmax - zero_point) * scale
    return min, max


@pytest.fixture
def module(request):
    cls = request.param

    match cls:
        case torch.nn.Conv2d:
            return cls(3, 3, 3).eval()
        case torch.nn.ReLU:
            return cls().eval()
        case torch.nn.Softmax:
            return cls().eval()
        case custom.Mean:
            return cls().eval()
        case _:
            raise NotImplementedError


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv2d,
        torch.nn.ReLU,
        torch.nn.Softmax,
        custom.Mean,
    ],
    indirect=True,
)
def test_fully_quantized(module):
    example_inputs = (torch.randn(1, 3, 10, 10),)
    ep = torch.export.export(module, example_inputs)
    ep = AIMETExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(
        param_bw=8, activation_bw=16, target_backend="htp_v69"
    ):
        output = ep.module()(*example_inputs)
        assert isinstance(output, Q.DequantizedTensor)

    _assert_graph_fully_quantized(
        ep.graph,
        weight_dtype=torch.int8,
        bias_dtype=torch.int32,
        activation_dtype=torch.uint16,
    )

    (output_dq,) = ep.graph.output_node().all_input_nodes

    if isinstance(module, torch.nn.Softmax):
        min, max = _get_min_max_range(output_dq)
        assert min == 0.0
        assert np.allclose(max, 1.0)
    elif isinstance(module, torch.nn.ReLU):
        min, _ = _get_min_max_range(output_dq)
        assert min == 0.0
