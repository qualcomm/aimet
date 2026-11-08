# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme
from aimet_onnx.meta.connectedgraph import ConnectedGraph

import numpy as np
import pytest

from aimet_onnx.graph_passes.utils import is_unidirectionally_broadcastable
from ..models.test_models import (
    rmsnorm_model,
    rmsnorm_with_cast_model,
    rmsnorm_followed_by_mul_model,
)
from ..utils import patch_fuse_supergroups
from .utils import assert_on_const_quantizers, assert_on_output_quantizers


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("mul_for_pow", [True, False])
@pytest.mark.parametrize(
    "mul_rsqrt_pattern",
    ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt", "mul_pow_neg_half"],
)
def test_rmsnorm(elementwise_affine, mul_for_pow, mul_rsqrt_pattern):
    dim = 32
    model = rmsnorm_model(
        dim=dim,
        elementwise_affine=elementwise_affine,
        mul_for_pow=mul_for_pow,
        mul_rsqrt_pattern=mul_rsqrt_pattern,
    )
    graph = ConnectedGraph(model)

    input_data = {"x": np.random.rand(1, 3, dim, dim).astype(np.float32)}
    with patch_fuse_supergroups(False):
        sim = QuantizationSimModel(
            model,
            input_data,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            config_file="htp_v81",
        )

    all_ops = graph.ordered_ops
    # Check if quantization is disabled for RMSNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[:-1], sim.qc_quantize_op_dict)
    # Check if quantization is enabled for last op of RMSNormalization sub-graph
    assert_on_output_quantizers(all_ops[-1:], sim.qc_quantize_op_dict, enabled=True)

    # Check if quantization is disabled for RMSNormalization sub-graph constant ops except weight
    if elementwise_affine:
        layernorm_weight = all_ops[-1]
        all_ops.remove(layernorm_weight)
        assert_on_const_quantizers(
            [layernorm_weight], sim.qc_quantize_op_dict, enabled=True
        )

    assert_on_const_quantizers(all_ops, sim.qc_quantize_op_dict)


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("mul_for_pow", [True, False])
@pytest.mark.parametrize(
    "mul_rsqrt_pattern",
    ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt", "mul_pow_neg_half"],
)
def test_rmsnorm_with_cast(elementwise_affine, mul_for_pow, mul_rsqrt_pattern):
    """Test RMSNorm pattern matching when Cast ops are present.

    Models like Qwen2RMSNorm cast to float32 for numerical stability,
    inserting Cast ops that the graph pass must skip to find the weight Mul.
    """
    dim = 32
    model = rmsnorm_with_cast_model(
        dim=dim,
        elementwise_affine=elementwise_affine,
        mul_for_pow=mul_for_pow,
        mul_rsqrt_pattern=mul_rsqrt_pattern,
    )
    graph = ConnectedGraph(model)

    input_data = {"x": np.random.rand(1, 3, dim, dim).astype(np.float16)}
    with patch_fuse_supergroups(False):
        sim = QuantizationSimModel(
            model,
            input_data,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            config_file="htp_v81",
        )

    all_ops = graph.ordered_ops
    # Filter out Cast ops — they are pass-through and should not affect quantization
    non_cast_ops = [op for op in all_ops if op.type != "Cast"]

    # All intermediate ops (everything except last) should have quantization disabled
    assert_on_output_quantizers(non_cast_ops[:-1], sim.qc_quantize_op_dict)
    # Last op (weight Mul or final norm output) should have quantization enabled
    assert_on_output_quantizers(
        non_cast_ops[-1:], sim.qc_quantize_op_dict, enabled=True
    )

    # Constant quantizers: weight should stay enabled, all others disabled
    if elementwise_affine:
        weight_op = non_cast_ops[-1]
        non_cast_ops.remove(weight_op)
        assert_on_const_quantizers([weight_op], sim.qc_quantize_op_dict, enabled=True)

    assert_on_const_quantizers(non_cast_ops, sim.qc_quantize_op_dict)


@pytest.mark.parametrize("fuse_supergroups", [True, False])
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "other_shape, is_valid_scale_shape",
    [
        # Activation of the same shape as the input is not a valid scale
        ((1, 3, 32, 32), False),
        # Activation of the normalized shape is a valid scale
        ((32,), True),
    ],
)
def test_rmsnorm_followed_by_mul(
    fuse_supergroups, elementwise_affine, other_shape, is_valid_scale_shape
):
    """
    The Mul after an RMSNorm is part of it only if the RMSNorm has no weight yet and the
    other input of the Mul has a valid scale shape as per the ONNX spec
    """
    model = rmsnorm_followed_by_mul_model(
        dim=32, elementwise_affine=elementwise_affine, other_shape=other_shape
    )
    input_data = {
        "x": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "y": np.random.rand(*other_shape).astype(np.float32),
    }
    with patch_fuse_supergroups(fuse_supergroups):
        sim = QuantizationSimModel(
            model,
            input_data,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            config_file="htp_v81",
        )

    # Mul by y is taken as the RMSNorm scale Mul only if y is a valid scale and the RMSNorm
    # has no weight
    absorbed = is_valid_scale_shape and not elementwise_affine
    consumer_of_y = next(
        op
        for op in sim.connected_graph.ordered_ops
        if any(inp.name == "y" for inp in op.inputs)
    )
    # Output of the op that consumes y is quantized, whether it is the Mul by y or the
    # RMSNorm that absorbed it
    assert sim.qc_quantize_op_dict[consumer_of_y.outputs[0].name].enabled

    if not absorbed:
        # Mul by y consumes the RMSNorm output, so the RMSNorm output is quantized
        assert consumer_of_y.type == "Mul"
        norm_output = next(inp for inp in consumer_of_y.inputs if inp.name != "y")
        assert sim.qc_quantize_op_dict[norm_output.name].enabled
    elif fuse_supergroups:
        assert consumer_of_y.type == "RMSNormalization"
    else:
        # Intermediate outputs of the RMSNorm are not quantized
        norm_output = next(inp for inp in consumer_of_y.inputs if inp.name != "y")
        assert not sim.qc_quantize_op_dict[norm_output.name].enabled

    if fuse_supergroups:
        fused = [
            op
            for op in sim.connected_graph.ordered_ops
            if op.type == "RMSNormalization"
        ]
        assert len(fused) == 1
        assert len(fused[0].inputs) == (2 if elementwise_affine or absorbed else 1)


@pytest.mark.parametrize(
    "shape, target_shape, expected",
    [
        ([32], [32], True),
        # Dim of 1 broadcasts
        ([1, 32], [3, 32], True),
        # Same symbolic dim
        (["hidden"], ["seq", "hidden"], True),
        # Rank of the shape is higher than the rank of the target shape
        ([1, 3, 32], [3, 32], False),
        # Dim does not match the target dim
        ([16], [32], False),
        # Unknown shape
        (None, [32], False),
        # Symbolic dims with different names
        (["seq"], ["seq", "hidden"], False),
        # Rank decides even if the dims are unknown
        ([None, None], [None], False),
    ],
)
def test_is_unidirectionally_broadcastable(shape, target_shape, expected):
    assert is_unidirectionally_broadcastable(shape, target_shape) is expected
