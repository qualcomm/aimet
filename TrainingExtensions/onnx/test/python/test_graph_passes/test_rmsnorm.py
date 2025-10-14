# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme
from aimet_onnx.meta.connectedgraph import ConnectedGraph

import numpy as np
import pytest

from ..models.test_models import rmsnorm_model
from .utils import assert_on_const_quantizers, assert_on_output_quantizers


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("mul_for_pow", [True, False])
@pytest.mark.parametrize(
    "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
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
