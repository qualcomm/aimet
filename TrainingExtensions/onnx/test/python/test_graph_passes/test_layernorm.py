# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import onnx_ir
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme

import numpy as np
import pytest

from ..models.test_models import layernorm_model
from ..utils import patch_fuse_supergroups
from .utils import assert_on_const_quantizers, assert_on_output_quantizers


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_layer_norm(elementwise_affine, bias):
    dim = 32
    model = layernorm_model(dim=dim, elementwise_affine=elementwise_affine, bias=bias)
    graph = ConnectedGraph(onnx_ir.from_proto(model))

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
    # Check if quantization is disabled for LayerNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[:-1], sim.qc_quantize_op_dict)
    # Check if quantization is enabled for last op of LayerNormalization sub-graph
    assert_on_output_quantizers(all_ops[-1:], sim.qc_quantize_op_dict, enabled=True)

    # Check if quantization is disabled for LayerNormalization sub-graph constant ops except layernorm.weight
    if elementwise_affine:
        layernorm_weight = all_ops[-2 if bias else -1]
        all_ops.remove(layernorm_weight)
        assert_on_const_quantizers(
            [layernorm_weight], sim.qc_quantize_op_dict, enabled=True
        )

    assert_on_const_quantizers(all_ops, sim.qc_quantize_op_dict)


def test_layer_norm_intermediate():
    dim = 32
    model = layernorm_model(dim=dim, include_add_ops=True)
    graph = ConnectedGraph(onnx_ir.from_proto(model))

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
    # Check if quantization is disabled for LayerNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[1:-2], sim.qc_quantize_op_dict)
    # Check if quantization is enabled for last op of LayerNormalization sub-graph
    assert_on_output_quantizers(all_ops[-2:-1], sim.qc_quantize_op_dict, enabled=True)
    # Check if quantization is disabled for LayerNormalization sub-graph constant ops except layernorm.weight
    layernorm_weight = all_ops[-3]
    all_ops.remove(layernorm_weight)
    assert_on_const_quantizers(all_ops, sim.qc_quantize_op_dict)
    assert_on_const_quantizers(
        [layernorm_weight], sim.qc_quantize_op_dict, enabled=True
    )
