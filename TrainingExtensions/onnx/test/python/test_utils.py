# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import onnx
import torch
from packaging import version

import aimet_onnx.utils as utils
from aimet_onnx.utils import ParamUtils
from aimet_onnx.adaround.utils import ModelData, read_attributes_for_op

from .models import models_for_tests


class TestUtils:
    """
    Test functions in utils
    """
    def test_remove_nodes(self):
        """
        Test remove nodes by given type
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm']
        # Remove first layer of dummy model
        utils.remove_nodes_with_type('Conv', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['Relu', 'MaxPool', 'Flatten', 'Gemm']
        # Remove last layer of dummy model
        utils.remove_nodes_with_type('Gemm', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['Relu', 'MaxPool', 'Flatten']
        # Check connection of each layer
        onnx.checker.check_model(model)

    def test_replace_nodes(self):
        """
        Test replace op type of nodes with given op type
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm']

        utils.replace_node_with_op('Conv', 'CustomOp', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['CustomOp', 'Relu', 'MaxPool', 'Flatten', 'Gemm']

    def test_get_weights(self):
        """
        Test get weights
        """
        model = models_for_tests.build_dummy_model()
        for node in model.graph.initializer:
            assert node.raw_data == utils.get_weights(node.name, model.graph)

    def test_list_nodes(self):
        """
        Test get nodes with ordered
        """
        model = models_for_tests.build_dummy_model()
        node_dict = utils.get_ordered_dict_of_nodes(model.graph)
        node_keys = list(node_dict.keys())

        for i, node in enumerate(model.graph.node):
            assert node_keys[i] == node.name
            assert node_dict[node.name] == node

    def test_weight_utils(self):
        model = models_for_tests.build_dummy_model()
        for node in model.graph.node:
            if node.op_type == 'Conv':
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [1]
                assert weights_shape == [1, 3, 3, 3]
                assert weights.name == 'conv_w'
                assert bias.name == 'conv_b'

            if node.op_type == 'Gemm':
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [10]
                assert weights_shape == [256, 10]
                assert weights.name == 'fc_w'
                assert bias.name == 'fc_b'

    def test_utils_transposed_conv_model(self):
        model = models_for_tests.transposed_conv_model()
        model = model.model
        for node in model.graph.node:
            if node.op_type == 'ConvTranspose':
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [10]
                assert weights_shape == [10, 10, 3, 3]
                assert weights.name == 'conv1.weight'
                assert bias.name == 'conv1.bias'
                break

    def test_utils_const_param_model(self):
        model = models_for_tests.const_param_model()
        for node in model.graph.node:
            if node.op_type == 'InstanceNormalization':
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [32]
                assert weights_shape == [32]
                assert weights.name == '/down_blocks.0/resnets.0/norm1/Constant_1_output_0'
                assert bias.name == '/down_blocks.0/resnets.0/norm1/Constant_2_output_0'
                break

    def test_remove_node(self):
        """
        Test remove node from model
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm']
        gemm_node = model.graph.node[-1]
        utils.remove_node(gemm_node, model.graph)

        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten']
        assert model.graph.output[0].name in model.graph.node[-1].output


    def test_get_attribute(self):
        """
        Test get attribute value from node
        """
        model = models_for_tests.build_dummy_model()
        conv_layer = model.graph.node[0]
        assert utils.get_node_attribute(conv_layer, "pads") == [1, 1, 1, 1]
        assert utils.get_node_attribute(conv_layer, "kernel_shape") == [3, 3]

    def test_replace_relu6_with_relu(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.depthwise_conv_model_with_relu6()
            relu6_count = 0
            original_relu_count = 0
            for node in model.model.graph.node:
                if node.op_type == 'Clip':
                    relu6_count += 1
                if node.op_type == 'Relu':
                    original_relu_count += 1

            utils.replace_relu6_with_relu(model)

            relu_count = 0
            for node in model.model.graph.node:
                if node.op_type == 'Relu':
                    relu_count += 1

            assert relu_count - original_relu_count == relu6_count

    def test_create_model_data_single_residual_model(self):
        model = models_for_tests.transposed_conv_model_without_bn()
        model_data = ModelData(model.model)
        assert len(model_data.module_to_info) == 3
