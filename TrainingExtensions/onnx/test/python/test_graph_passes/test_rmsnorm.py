# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
from ..models.test_models import rmsnorm_model
from .utils import assert_on_const_quantizers, assert_on_output_quantizers, get_dummy_qc_quantize_op_dict

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.graph_passes.pass_registry import apply_graph_passes
import pytest


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("mul_for_pow", [True, False])
@pytest.mark.parametrize("separate_mul_div", [True, False])
def test_rmsnorm(elementwise_affine, mul_for_pow, separate_mul_div):

    model = rmsnorm_model(dim=32, elementwise_affine=elementwise_affine, mul_for_pow=mul_for_pow, separate_mul_div=separate_mul_div)
    graph = ConnectedGraph(model)
    qc_quantize_op_dict = get_dummy_qc_quantize_op_dict(graph)

    quantization_status = [q_op.enabled for q_op in list(qc_quantize_op_dict.values())]
    # Check if quantization is enabled for all ops
    assert all(quantization_status)
    apply_graph_passes(model, graph, qc_quantize_op_dict, ["RMSNormalization"])

    all_ops = graph.ordered_ops
    # Check if quantization is disabled for RMSNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[:-1], qc_quantize_op_dict)
    # Check if quantization is enabled for last op of RMSNormalization sub-graph
    assert_on_output_quantizers(all_ops[-1:], qc_quantize_op_dict, enabled=True)

    # Check if quantization is disabled for RMSNormalization sub-graph constant ops except weight
    if elementwise_affine:
        layernorm_weight = all_ops[-1]
        all_ops.remove(layernorm_weight)
        assert_on_const_quantizers([layernorm_weight], qc_quantize_op_dict, enabled=True)

    assert_on_const_quantizers(all_ops, qc_quantize_op_dict)
