# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from functools import lru_cache
import numpy as np
import yaml
import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputSpec
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
from aimet_torch.experimental.backends.schema import BackendDescription, qtype


def _assert_graph_fully_quantized(
    ep: ExportedProgram,
    expected_weight_dtype: torch.dtype,
    expected_bias_dtype: torch.dtype,
    expected_activation_dtype: torch.dtype,
    expected_channel_axis: int,
):
    for node in ep.graph.nodes:
        _assert_node_fully_quantized(
            ep,
            node,
            expected_weight_dtype,
            expected_bias_dtype,
            expected_activation_dtype,
            expected_channel_axis,
        )


def _assert_node_fully_quantized(
    ep: ExportedProgram,
    node: torch.fx.Node,
    expected_weight_dtype: torch.dtype,
    expected_bias_dtype: torch.dtype,
    expected_activation_dtype: torch.dtype,
    expected_channel_axis: int,
):
    if _is_torch_ao_qdq_node(node):
        return

    if all(
        _is_torch_ao_qdq_node(user) and node in user.args[1:3] for user in node.users
    ):
        # Current node is 2nd (scale) or 3rd (zero_point) input of QDQ node.
        return

    for producer in node.all_input_nodes:
        if _output_can_be_quantized(ep, producer):
            assert _is_torch_ao_dq_node(producer)

    if not _output_can_be_quantized(ep, node):
        return

    for user in node.users:
        fake_output = node.meta["val"]

        if fake_output.is_floating_point():
            assert _is_torch_ao_q_node(user)
        else:
            assert _is_torch_ao_dq_node(user)

        *_, qtype = user.args

        if _is_weight_node(ep, node):
            assert qtype == expected_weight_dtype
            assert (
                user.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
            _, _, _, channel_axis, *_ = user.args
            assert channel_axis == expected_channel_axis
        elif _is_bias_node(ep, node):
            assert qtype == expected_bias_dtype
            assert (
                user.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
            _, _, _, channel_axis, *_ = user.args
            # NOTE: Bias is 1D, so its channel axis is always 0
            assert channel_axis == 0
        else:
            assert qtype == expected_activation_dtype


def _output_can_be_quantized(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    fake_output = node.meta["val"]
    return (
        isinstance(fake_output, torch.Tensor)
        and fake_output.numel() > 0
        and fake_output.is_floating_point()
        and not _is_running_mean(ep, node)
        and not _is_running_var(ep, node)
    )


def _is_weight_node(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node in _get_input_specs(ep)
        and "weight" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
    )


def _is_bias_node(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node in _get_input_specs(ep)
        and "bias" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
    )


def _is_running_mean(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node in _get_input_specs(ep)
        and "running_mean" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
    )


def _is_running_var(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        node in _get_input_specs(ep)
        and "running_var" in node.name
        and "scale" not in node.name
        and "zero_point" not in node.name
    )


@lru_cache(maxsize=16)
def _get_input_specs(ep: ExportedProgram) -> dict[torch.fx.Node, InputSpec]:
    input_specs = {
        input_spec.arg.name: input_spec for input_spec in ep.graph_signature.input_specs
    }
    return {
        node: input_spec
        for node in ep.graph.nodes
        if (input_spec := input_specs.get(node.target))
    }


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
        case _:
            return (torch.randn(1, 3, 10, 10),)


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
        custom.Mean,
        torch.nn.BatchNorm2d,
        torch.nn.LayerNorm,
        torch.nn.InstanceNorm2d,
        torch.nn.Sequential,
    ],
    indirect=True,
)
def test_fully_quantized(tmp_path, module, example_inputs):
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

    if isinstance(
        module,
        (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d),
    ):
        expected_channel_axis = 1
    else:
        expected_channel_axis = 0

    _assert_graph_fully_quantized(
        ep,
        expected_weight_dtype=torch.int8,
        expected_bias_dtype=torch.int32,
        expected_activation_dtype=torch.uint16,
        expected_channel_axis=expected_channel_axis,
    )

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
      - {"*": "uint8"}
  - op_type: "aten::relu"
    type_constraints:
      - {"*": "uint16"}
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
