# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch.export import ExportedProgram
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
    ep: ExportedProgram,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    activation_dtype: torch.dtype,
):
    for node in ep.graph.nodes:
        _assert_node_fully_quantized(
            ep,
            node,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            activation_dtype=activation_dtype,
        )


def _assert_node_fully_quantized(
    ep: ExportedProgram,
    node: torch.fx.Node,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    activation_dtype: torch.dtype,
):
    if _is_torch_ao_qdq_node(node):
        return

    if all(
        _is_torch_ao_qdq_node(user) and node in user.args[1:3] for user in node.users
    ):
        # Current node is 2nd (scale) or 3rd (zero_point) input of QDQ node.
        return

    for producer in node.all_input_nodes:
        assert _is_torch_ao_dq_node(producer)

    for user in node.users:
        if node.meta["val"].is_floating_point():
            assert _is_torch_ao_q_node(user)
        else:
            assert _is_torch_ao_dq_node(user)

        *_, qtype = user.args

        if _is_weight_node(ep, node):
            assert qtype == weight_dtype
            assert (
                user.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
        elif _is_bias_node(ep, node):
            assert qtype == bias_dtype
            assert (
                user.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
        else:
            assert qtype == activation_dtype


def _is_weight_node(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node.target in ep.state_dict
        and "weight" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
    )


def _is_bias_node(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node.target in ep.state_dict
        and "bias" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
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
        case torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 3),
                torch.nn.ReLU(),
                torch.nn.Softmax(),
                custom.Mean(),
            ).eval()
        case _:
            raise NotImplementedError


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv2d,
        torch.nn.ReLU,
        torch.nn.Softmax,
        custom.Mean,
        torch.nn.Sequential,
    ],
    indirect=True,
)
def test_fully_quantized(tmp_path, module):
    """
    Given: nn.Module included in quantization IR
    When: compute missing encodings with target backend specified
    Then: All nodes in the graph should be quantized with the correct dtype
    """
    example_inputs = (torch.randn(1, 3, 10, 10),)
    ep = torch.export.export(module, example_inputs)
    ep = AIMETExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(
        param_bw=8, activation_bw=16, target_backend="htp_v69"
    ):
        output = ep.module()(*example_inputs)
        assert isinstance(output, Q.DequantizedTensor)

    # Make sure ExportedProgram object can be saved and loaded correctly
    torch.export.save(ep, tmp_path / "ep.pt2")
    ep = torch.export.load(tmp_path / "ep.pt2")

    _assert_graph_fully_quantized(
        ep,
        weight_dtype=torch.int8,
        bias_dtype=torch.int32,
        activation_dtype=torch.uint16,
    )

    if isinstance(module, torch.nn.Softmax):
        """
        Softmax output should be set to [0, 1] range after quantization
        """
        (output_dq,) = ep.graph.output_node().all_input_nodes
        min, max = _get_min_max_range(output_dq)
        assert min == 0.0
        assert np.allclose(max, 1.0)

    if isinstance(module, torch.nn.ReLU):
        """
        Relu output should be set to [0, ?] range after quantization
        """
        (output_dq,) = ep.graph.output_node().all_input_nodes
        min, _ = _get_min_max_range(output_dq)
        assert min == 0.0

    if isinstance(module, (torch.nn.ReLU, custom.Mean)):
        """
        Relu and ReduceMean's input range should be euqal to the output range
        """
        (input,) = ep.graph.find_nodes(op="placeholder")
        (input_q,) = input.users
        (output_dq,) = ep.graph.output_node().all_input_nodes
        input_range = _get_min_max_range(input_q)
        output_range = _get_min_max_range(output_dq)
        assert input_range == output_range
