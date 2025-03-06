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
from aimet_onnx.graph_passes.pass_registry import register_pass, apply_graph_passes
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.utils import get_const_input_names, get_output_names
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_common.connected_graph.operation import Op
import pytest
from ..models.models_for_tests import build_dummy_model
from .utils import get_dummy_qc_quantize_op_dict


@register_pass("DummyTestGraphPass")
class DummyTestGraphPass(SupergroupGraphPass):
    def match_pattern(self, op: Op):
        self.disable_quantizers = get_const_input_names(op_list=[op]) + get_output_names(op_list=[op])
        return True


def test_register_and_apply_graph_pass():
    model = build_dummy_model()
    graph = ConnectedGraph(model)
    qc_quantize_ops = get_dummy_qc_quantize_op_dict(graph)
    apply_graph_passes(graph, qc_quantize_ops, ["DummyTestGraphPass"])
    qc_quantize_ops = list(qc_quantize_ops.values())
    quantization_status = [q_op.enabled for q_op in qc_quantize_ops]

    # Ensure all output quantizers are disabled
    assert not any(quantization_status)


def test_error_on_unregistered_graph_pass():
    model = build_dummy_model()
    graph = ConnectedGraph(model)
    qc_quantize_ops = get_dummy_qc_quantize_op_dict(graph)

    with pytest.raises(
        ValueError, match="Graph pass requested but not found:"
    ):
        apply_graph_passes(graph, qc_quantize_ops, ["UnRegisteredGraphPass"])
