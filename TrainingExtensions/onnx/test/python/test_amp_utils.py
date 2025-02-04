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

import pytest

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.utils import find_layer_database_for_mac_calculation, create_mac_dict, find_bit_ops_reduction, \
    calculate_running_bit_ops, find_param_name_to_parent_name_dict, get_quantizer_to_op_type_dict
from aimet_onnx.amp.quantizer_groups import QuantizerGroup
from .models.test_models import single_residual_model


class TestAMPUtils:
    def test_find_layer_database_and_mac_cost(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        layer_db = find_layer_database_for_mac_calculation(sim)
        assert len(layer_db) == 5
        assert layer_db['/conv1/Conv'].output_shape == (1, 32, 18, 18)
        assert layer_db['/conv3/Conv'].weight_shape == [8, 16, 2, 2]
        mac_dict = create_mac_dict(sim)
        for node_name in mac_dict:
            cost = layer_db[node_name].output_shape[2] * layer_db[node_name].output_shape[3] \
                   * layer_db[node_name].weight_shape[0] * layer_db[node_name].weight_shape[1]
            if len(layer_db[node_name].weight_shape) == 4:
                cost *= layer_db[node_name].weight_shape[2] * layer_db[node_name].weight_shape[3]
            assert mac_dict[node_name] == cost

    def test_find_parent_name_dict(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        param_op_name_dict = find_param_name_to_parent_name_dict(sim.connected_graph)
        assert len(param_op_name_dict) == 5
        assert '/conv1/Conv' in param_op_name_dict.values()

    def test_calculate_running_bit_ops(self):
        """ Test calculate running bit ops """
        mac_dict = {'Conv_0': 124416, 'Conv_3': 100352, 'Conv_5': 12800, 'Conv_6': 50176, 'Gemm_14': 720}
        param_op_name_dict = {'36': 'Conv_0', 'conv4.weight': 'Conv_6', 'fc.weight': 'Gemm_14', '39': 'Conv_3', 'conv3.weight': 'Conv_5'}

        # 1) activation + weight
        quantizer_group = QuantizerGroup(parameter_quantizers=('36',), activation_quantizers=('input',))

        op_bitwidth_dict = {}
        max_bw = ((16, None), (16, None))
        new_bw = ((8, None), (8, None))
        starting_bit_ops = 63700992
        running_bit_ops = calculate_running_bit_ops(mac_dict, quantizer_group, param_op_name_dict,
                                                    op_bitwidth_dict, max_bw, new_bw, starting_bit_ops)

        assert running_bit_ops == starting_bit_ops + mac_dict['Conv_0'] * new_bw[0][0] * new_bw[1][0] \
               - mac_dict['Conv_0'] * max_bw[0][0] * max_bw[1][0]

        # 2) only weight
        quantizer_group = QuantizerGroup(parameter_quantizers=('fc.weight',), activation_quantizers=())

        op_bitwidth_dict = {}
        running_bit_ops = calculate_running_bit_ops(mac_dict, quantizer_group, param_op_name_dict,
                                                    op_bitwidth_dict, max_bw, new_bw, starting_bit_ops)

        assert running_bit_ops == starting_bit_ops + mac_dict['Gemm_14'] * max_bw[0][0] * new_bw[1][0] - \
               mac_dict['Gemm_14'] * max_bw[0][0] * max_bw[1][0]

    def test_find_bit_ops_reduction(self):
        """ Test find bit ops reduction """

        mac_dict = {'Conv_0': 124416, 'Conv_3': 100352, 'Conv_5': 12800, 'Conv_6': 50176, 'Gemm_14': 720}
        param_op_name_dict = {'36': 'Conv_0', 'conv4.weight': 'Conv_6', 'fc.weight': 'Gemm_14', '39': 'Conv_3',
                              'conv3.weight': 'Conv_5'}

        quantizer_group = QuantizerGroup(parameter_quantizers=('36',), activation_quantizers=('input',))

        bit_ops_reduction = find_bit_ops_reduction(quantizer_group, mac_dict,  param_op_name_dict,
                                                   ((16, None), (16, None)),
                                                   ((8, None), (8, None)))

        assert bit_ops_reduction == 124416 * 16 * 16 - 124416 * 8 * 8

        # 2) only weight
        quantizer_group = QuantizerGroup(parameter_quantizers=('fc.weight',), activation_quantizers=())

        bit_ops_reduction = find_bit_ops_reduction(quantizer_group, mac_dict, param_op_name_dict,
                                                   ((16, None), (16, None)),
                                                   ((8, None), (8, None)))

        assert bit_ops_reduction == 720 * 16 * 16 - 720 * 16 * 8

    @pytest.mark.skip
    def test_get_quantizer_to_op_type_dict(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        d = get_quantizer_to_op_type_dict(sim)
        quantizer_to_op_type = {'input': ['Conv'], 'onnx::Conv_45': ['Conv'], 'onnx::Conv_46': ['Conv'], '/conv1/Conv_output_0': ['Conv'], '/relu1/Relu_output_0': ['Relu'], '/maxpool/MaxPool_output_0': ['MaxPool'], 'onnx::Conv_48': ['Conv'], 'onnx::Conv_49': ['Conv'], '/conv2/Conv_output_0': ['Conv'], '/relu2/Relu_output_0': ['Relu'], 'conv3.weight': ['Conv'], '/conv3/Conv_output_0': ['Conv'], 'conv4.weight': ['Conv'], 'conv4.bias': ['Conv'], '/conv4/Conv_output_0': ['Conv'], '/ada/Pad_output_0': ['Pad'], '/ada/AveragePool_output_0': ['AveragePool'], '/Add_output_0': ['Add'], '/relu3/Relu_output_0': ['Relu'], '/avgpool/Pad_output_0': ['Pad'], '/avgpool/AveragePool_output_0': ['AveragePool'], 'fc.weight': ['Gemm'], 'fc.bias': ['Gemm'], 'output': ['Gemm']}
        assert d == quantizer_to_op_type
