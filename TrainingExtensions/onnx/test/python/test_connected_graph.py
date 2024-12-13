# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import itertools
import pytest
import torch
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_ops_with_constant_inputs
from aimet_onnx.meta.connectedgraph import ConnectedGraph, CONSTANT_TYPE, OPS_WITH_PARAMS
from aimet_onnx.utils import ParamUtils
from models import models_for_tests


class TestConnectedGraph:

    @pytest.mark.parametrize("model", (models_for_tests.build_dummy_model(),
                                       models_for_tests.single_residual_model().model,
                                       models_for_tests.multi_input_model().model,
                                       models_for_tests.transposed_conv_model().model,
                                       models_for_tests.concat_model().model,
                                       models_for_tests.hierarchical_model().model,
                                       models_for_tests.elementwise_op_model().model,
                                       models_for_tests.instance_norm_model().model,
                                       models_for_tests.layernorm_model(),
                                       models_for_tests.matmul_with_constant_first_input()
                                       ))
    def test_model_representation(self, model):
        """
        Given: A ConnectedGraph constructed for a given model
        Then: 1) All non-constant nodes should be represented as ops in the CG
              2) All tensors should be represented as products in the CG
        """
        nodes = {node.name for node in model.graph.node if node.op_type != "Constant"}
        tensors = {tensor for node in model.graph.node for tensor in itertools.chain(node.input, node.output) if tensor}
        cg = ConnectedGraph(model)
        ops = cg.get_all_ops()
        assert ops.keys() == nodes
        products = cg.get_all_products()
        assert products.keys() == tensors

        for _, product in products.items():
            for node in model.graph.node:
                if node.op_type == "Constant":
                    continue
                """
                When: A tensor is the output of a node
                Then: The corresponding Op should be the product's producer
                """
                if product.name in node.output:
                    assert node == product.producer.get_module()

                """
                When: A tensor is the input of a node
                Then: The corresponding Op should appear in the product's consumers
                """
                if product.name in node.input:
                    assert node in [op.get_module() for op in product.consumers]

            """
            When: A tensor is a model input
            Then: The corresponding product should have product.is_model_input set to True
            """
            if product.name in {t.name for t in model.graph.input}:
                assert product.is_model_input
                assert not product.producer
            else:
                assert not product.is_model_input

            """
            When: A tensor has no producer
            Then: The tensor is either a model input, constant, or parameter
            """
            if not product.producer:
                assert product.is_model_input or product.is_const or product.is_parm
            else:
                assert not (product.is_model_input or product.is_const or product.is_parm)

        for op_name, op in ops.items():
            node = op.get_module()
            """
            When: A tensor is input idx of node
            Then: The tensor's product should be input idx of the corresponding op
            """
            for idx, tensor in enumerate(node.input):
                assert op.inputs[idx] is products[tensor]

            """
            When: A tensor is output[0] of a node
            Then: The tensor's product should be the corresponding op's output
            """
            if node.output:
                assert op.output is products[node.output[0]]

    def test_single_residual_model(self):
        model = models_for_tests.single_residual_model()
        conn_graph = ConnectedGraph(model)
        operator_names = {node.name for node in model.nodes() if node.op_type not in CONSTANT_TYPE}
        assert operator_names == conn_graph.get_all_ops().keys()

        model_weights = {node.input[1] for node in model.graph().node if node.op_type == "Conv"}
        products = conn_graph.get_all_products()

        for weight in model_weights:
            assert products[weight].is_parm

        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 1
        for op in conn_graph.ordered_ops:
            if op.type == "Gemm":
                assert op.transposed_params

    def test_multi_inputs_model(self):
        model = models_for_tests.multi_input_model()
        conn_graph = ConnectedGraph(model)
        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 2

    def test_concat_model(self):
        model = models_for_tests.concat_model()
        conn_graph = ConnectedGraph(model)
        ops = conn_graph.get_all_ops()
        assert len(ops['/Concat'].inputs) == 3

    def test_hierarchical_model(self):
        model = models_for_tests.hierarchical_model()
        conn_graph = ConnectedGraph(model)
        ordered_ops = conn_graph.ordered_ops
        name_to_index = {}
        for index, op in enumerate(ordered_ops):
            name_to_index[op.name] = index

        # Check in the graph that if A & B are connected and A comes before B in the graph then that should be the case
        # in ordered graphs as well
        assert name_to_index['/conv1/conv/Conv'] < name_to_index['/nm1/tm1/Reshape']
        assert name_to_index['/sq/seq_list/seq_list.0/Conv'] < name_to_index['/sq/seq_list/seq_list.5/Conv']
        assert name_to_index['/conv2/conv/Conv'] < name_to_index['/nm2/tm1/conv3/Conv']

    def test_matmul_layer_param_creation(self):
        torch.manual_seed(10)
        torch_model = models_for_tests.BNBeforeFlattenLinear()

        torch_model.eval()

        input_shape = (2, 10, 24, 24)

        model = models_for_tests._convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))

        cg = ConnectedGraph(model)
        for op in cg.ordered_ops:
            if op.type == 'MatMul':
                assert 'fc2.weight' in op.parameters
                break
        else:
            assert False

    def test_constant_elementwise_inputs(self):
        """ Test that constant inputs to elementwise ops are identified correctly """
        model = models_for_tests.elementwise_op_model()
        cg = ConnectedGraph(model)

        assert len(get_all_ops_with_constant_inputs(cg)) == 2
        for product in cg.ordered_ops[0].inputs:
            if product.name == "input":
                assert not product.is_const
                assert product.is_model_input
            else:
                assert product.is_const

        for product in cg.ordered_ops[1].inputs:
            assert not product.is_model_input
            if product.producer == cg.ordered_ops[0]:
                assert not product.is_const
            else:
                assert product.is_const

    def test_instance_norm_model(self):
        model = models_for_tests.instance_norm_model()
        cg = ConnectedGraph(model)
        assert cg.ordered_ops[-2].type == 'InstanceNormalization'

    def test_layer_norm_model(self):
        model = models_for_tests.layernorm_model()
        cg = ConnectedGraph(model)
        layernorm_cg_op = cg.ordered_ops[-1]
        assert layernorm_cg_op.type == 'LayerNormalization'
        assert ['layernorm.scale', 'layernorm.bias'] == list(layernorm_cg_op.parameters.keys())

    def test_malformed_model(self):
        model = models_for_tests.layernorm_model()
        model.graph.node.pop(1) # Remove constant node
        with pytest.raises(RuntimeError):
            cg = ConnectedGraph(model)
