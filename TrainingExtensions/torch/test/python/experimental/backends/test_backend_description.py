# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import operator
from functools import lru_cache
import numpy as np
import yaml
import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputSpec
import pytest
from aimet_torch.nn.modules import custom
from aimet_torch.experimental.backends import OpSchema, BackendDescription
from aimet_torch.experimental.export._utils import (
    _is_torch_ao_q_node,
    _is_torch_ao_dq_node,
    _is_torch_ao_qdq_node,
)
import aimet_torch.quantization as Q
from aimet_torch.experimental.export.exported_program import (
    ExportedProgram as AIMETExportedProgram,
)
from aimet_torch.experimental.backends.schema import BackendDescription, qtype


def _assert_graph_fully_quantized(
    ep: ExportedProgram,
    w8a16_op_schemas: dict[str, OpSchema],
):
    quantizable_nodes = [
        node
        for node in ep.graph.nodes
        if node.op != "placeholder"
        and node.op != "output"
        and node.target != operator.getitem
        and not _is_torch_ao_qdq_node(node)
    ]
    assert quantizable_nodes, "No quantizable nodes found in the graph"

    for node in quantizable_nodes:
        assert node.target.name() in w8a16_op_schemas
        expected_schema = w8a16_op_schemas[node.target.name()]
        _assert_op_schema(node, expected_schema)


def _assert_op_schema(node: torch.fx.Node, expected_schema: OpSchema):
    for input_node, arg_spec in zip(node.args, expected_schema.signature.inputs):
        if not arg_spec.qtype:
            continue
        assert isinstance(input_node, torch.fx.Node)
        *_, qtype = input_node.args
        expected_qtype = getattr(torch, arg_spec.qtype.dtype)
        assert qtype == expected_qtype

        if arg_spec.qtype.channel_axis is None:
            assert input_node.target.overloadpacket in (
                torch.ops.quantized_decomposed.quantize_per_tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor,
            )
        else:
            assert input_node.target.overloadpacket in (
                torch.ops.quantized_decomposed.quantize_per_channel,
                torch.ops.quantized_decomposed.dequantize_per_channel,
            )
            _, _, _, channel_axis, *_ = input_node.args
            assert channel_axis == arg_spec.qtype.channel_axis

    if isinstance(node.meta["val"], (tuple, list)):
        # Node returns a list of tensors.
        # In this case, `node` will be followed by getitem nodes for each output tensor as below
        #
        #                              +-------> getitem --------> y0
        #   x -> node -> (y0, y1, y2) -+-------> getitem --------> y1
        #                              +-------> getitem --------> y2
        #                                  (`all_output_nodes`)
        all_output_nodes = list(node.users)
    else:
        all_output_nodes = [node]

    for output_node, ret_spec in zip(
        all_output_nodes, expected_schema.signature.outputs
    ):
        if not ret_spec.qtype:
            continue
        assert isinstance(output_node, torch.fx.Node)
        (q,) = output_node.users
        assert _is_torch_ao_q_node(q)
        *_, qtype = q.args
        expected_qtype = getattr(torch, ret_spec.qtype.dtype)
        assert qtype == expected_qtype

        if ret_spec.qtype.channel_axis is None:
            assert q.target.overloadpacket in (
                torch.ops.quantized_decomposed.quantize_per_tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor,
            )
        else:
            assert q.target.overloadpacket in (
                torch.ops.quantized_decomposed.quantize_per_channel,
                torch.ops.quantized_decomposed.dequantize_per_channel,
            )
            _, _, _, channel_axis, *_ = q.args
            assert channel_axis == ret_spec.qtype.channel_axis


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
        case torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d:
            return cls(3, 3, 3).eval()
        case (
            torch.nn.ConvTranspose1d
            | torch.nn.ConvTranspose2d
            | torch.nn.ConvTranspose3d
        ):
            return cls(3, 3, 3).eval()
        case torch.nn.Linear:
            return cls(10, 10).eval()
        case torch.nn.ReLU | torch.nn.PReLU:
            return cls().eval()
        case torch.nn.Softmax:
            return cls().eval()
        case custom.Sin | custom.Cos | torch.nn.Tanh:
            return cls().eval()
        case custom.MatMul:
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
        case torch.nn.LayerNorm:
            return cls(10).eval()
        case torch.nn.InstanceNorm2d:
            return cls(num_features=3, affine=True).eval()
        case torch.nn.BatchNorm2d:
            return torch.nn.BatchNorm2d(
                num_features=3,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ).eval()
        case _:
            raise NotImplementedError


@pytest.fixture
def example_inputs(module):
    match module:
        case torch.nn.Conv1d() | torch.nn.ConvTranspose1d():
            return (torch.randn(1, 3, 10),)
        case torch.nn.Conv3d() | torch.nn.ConvTranspose3d():
            return (torch.randn(1, 3, 10, 10, 10),)
        case torch.nn.Linear():
            return (torch.randn(1, 3, 10),)
        case custom.MatMul():
            return (torch.randn(10, 10), torch.randn(10, 10))
        case _:
            return (torch.randn(1, 3, 10, 10),)


@pytest.fixture(scope="session")
def w8a16_op_schemas():
    htp_v69 = BackendDescription.from_backend_id("htp_v69")
    return {
        f"{domain}::{op_type}": schema
        for (domain, op_type), schemas in htp_v69.all_ops.items()
        for schema in schemas
        if schema.signature.tag in ("w8a16", "a16", "a16xa8")
    }


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.PReLU,
        torch.nn.Softmax,
        custom.Sin,
        custom.Cos,
        torch.nn.Tanh,
        custom.MatMul,
        custom.Mean,
        torch.nn.BatchNorm2d,
        torch.nn.LayerNorm,
        torch.nn.InstanceNorm2d,
        torch.nn.Sequential,
    ],
    indirect=True,
)
def test_fully_quantized(tmp_path, module, example_inputs, w8a16_op_schemas):
    """
    Given: nn.Module included in quantization IR
    When: compute missing encodings with target backend specified
    Then: All nodes in the graph should be quantized with the correct dtype
    """
    ep = torch.export.export(module, example_inputs)
    ep = AIMETExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(
        param_bw=8, activation_bw=16, target_backend="htp_v69"
    ):
        outputs = ep.module()(*example_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for output in outputs:
            if output.numel() > 0:
                assert isinstance(output, Q.DequantizedTensor)

    # Make sure ExportedProgram object can be saved and loaded correctly
    torch.export.save(ep, tmp_path / "ep.pt2")
    ep = torch.export.load(tmp_path / "ep.pt2")
    _assert_graph_fully_quantized(ep, w8a16_op_schemas)

    if isinstance(module, (torch.nn.Softmax, custom.Sin, custom.Cos, torch.nn.Tanh)):
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


def test_wildcard_overwrite():
    yaml_config = yaml.safe_load("""
name: test
ops:
  - op_type: "*"
    type_constraints:
      a8: {"*": "uint8"}
  - op_type: "aten::relu"
    type_constraints:
      a16: {"*": "uint16"}
""")
    bd = BackendDescription.from_yaml(yaml_config)

    for (domain, op_type), schemas in bd.all_ops.items():
        expected_qtype = (
            qtype("uint16") if (domain, op_type) == ("aten", "relu") else qtype("uint8")
        )
        assert all(
            arg_spec.qtype in (None, expected_qtype)
            for schema in schemas
            for arg_spec in itertools.chain(
                schema.signature.inputs, schema.signature.outputs
            )
        )
        # To prevent trivial test passing
        assert any(
            arg_spec.qtype == expected_qtype
            for schema in schemas
            for arg_spec in itertools.chain(
                schema.signature.inputs, schema.signature.outputs
            )
        )
