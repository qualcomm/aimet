# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

from aimet_common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.quantizer_groups import find_op_groups, find_quantizer_group
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from models.test_models import model_small_mnist, model_with_split, single_residual_model, concat_model, linear_layer_model


class TestQuantizerGroups:

    def test_simple_mnist_netwrok(self):
        model = model_small_mnist()
        connected_graph = ConnectedGraph(model)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 7
        # Check if there is one parent and one child
        for parent, child in op_groups.items():
            assert not isinstance(parent, tuple)
            assert len(child) == 1

        assert '/conv2/Conv' in op_groups
        assert '/relu3/Relu' in op_groups['/fc1/Gemm']

    def test_model_with_one_split(self):
        model = model_with_split()
        connected_graph = ConnectedGraph(model)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 3
        assert len(op_groups['/conv1/Conv']) == 2

    def test_single_residual_network(self):
        model = single_residual_model()
        connected_graph = ConnectedGraph(model)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 11
        count = 0

        add_op = [op for op in connected_graph.get_all_ops().values() if op.type == 'Add'][0]
        for _, child in op_groups.items():
            if tuple(child)[0] == add_op.name:
                count += 1
        assert count == 2
        assert op_groups['/Add'] == ['/relu3/Relu']

    def test_concat_model(self):
        model = concat_model()
        conn_graph = ConnectedGraph(model)
        op_groups = find_op_groups(conn_graph)
        assert len(op_groups) == 4
        assert op_groups['/conv1/Conv'] == ['/Concat']
        assert op_groups['/conv2/Conv'] == ['/Concat']
        assert op_groups['/conv3/Conv'] == ['/Concat']

    def test_find_quantizer_groups(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 10
        conv3_group = avg_pool_group = None
        for group in quantizer_groups:
            if "conv3.weight" in group.parameter_quantizers:
                conv3_group = group
            if "/avgpool/AveragePool_output_0" in group.activation_quantizers:
                avg_pool_group = group

        assert conv3_group.activation_quantizers[0] == '/relu2/Relu_output_0'
        assert avg_pool_group.parameter_quantizers[0] == 'fc.weight'

    def test_find_quantizer_groups_first_param_quantizer_disabled(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        # disable first param quantizer
        list(sim.qc_quantize_op_dict.values())[0].enabled = False
        _, quantizer_groups = find_quantizer_group(sim)
        input_group = conv3_group = avg_pool_group = None
        for group in quantizer_groups:
            if "input" in group.activation_quantizers:
                input_group = group
            if "conv3.weight" in group.parameter_quantizers:
                conv3_group = group
            if "/avgpool/AveragePool_output_0" in group.activation_quantizers:
                avg_pool_group = group

        assert len(quantizer_groups) == 10
        assert len(input_group.parameter_quantizers) == 0
        assert conv3_group.activation_quantizers[0] == '/relu2/Relu_output_0'
        assert avg_pool_group.parameter_quantizers[0] == 'fc.weight'

    def test_set_and_get_bitwidth_quantizer_groups(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        op_name_to_quantizer_dict, quantizer_groups = find_quantizer_group(sim)
        quantizer_group = None
        for group in quantizer_groups:
            if "conv3.weight" in group.parameter_quantizers:
                quantizer_group = group
                break
        candidate = ((8, QuantizationDataType.int), (16, QuantizationDataType.float))
        quantizer_group.set_quantizers_to_candidate(op_name_to_quantizer_dict, candidate)
        found_candidate = quantizer_group.get_candidate(op_name_to_quantizer_dict)
        assert candidate == found_candidate

        number_of_active_quantizers = len(quantizer_group.get_active_quantizers(op_name_to_quantizer_dict))
        assert number_of_active_quantizers == 2

    def test_linear_layer_model(self):
        model = linear_layer_model()
        sim = QuantizationSimModel(model)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 3
        assert quantizer_groups[-1].activation_quantizers[0] == '/layers.1/Gemm_output_0'
