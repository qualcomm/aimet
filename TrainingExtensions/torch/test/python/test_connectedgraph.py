# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains unit tests for testing ConnectedGraph module for PyTorch. """
import copy

import pytest
import unittest.mock
import torch
import torch.nn as nn
import torch.nn.functional as F

import aimet_torch.utils
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops,\
    get_all_ops_with_constant_inputs, CG_SPLIT
from .models import test_models
from aimet_common.connected_graph.product import Product
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.meta.operation import Op
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
import aimet_torch._base.nn.modules.custom as aimet_modules


class TestConnectedGraph(unittest.TestCase):
    """ Unit tests for testing ConnectedGraph module"""

    def test_single_residual(self):
        """ Test building ConnectedGraph on single residual model """
        # pylint: disable=protected-access
        model = test_models.SingleResidual()
        model.eval()
        inp_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(inp_shape, get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(17, len(conn_graph.ordered_ops))
        # Split count of 2 due to residual as well as reshape having a split
        self.assertEqual(2, conn_graph._split_count)
        # All ops will include 2 inserted split ops
        self.assertEqual(19, len(conn_graph.get_all_ops().keys()))
        input_ops = get_all_input_ops(conn_graph)
        self.assertEqual(1, len(input_ops))
        self.assertEqual(model.conv1, input_ops[0].get_module())
        output_ops = get_all_output_ops(conn_graph)
        self.assertEqual(1, len(output_ops))
        self.assertEqual(model.fc, output_ops[0].get_module())

    def test_multi_input(self):
        """ Test building ConnectedGraph on a model with multiple inputs """
        # pylint: disable=protected-access
        model = test_models.MultiInput()
        model.eval()
        inp_shape_1 = (1, 3, 32, 32)
        inp_shape_2 = (1, 3, 20, 20)
        inp_tensor_list = create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2], get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(11, len(conn_graph.ordered_ops))
        # Split count of 1 due to reshape having a split
        self.assertEqual(1, conn_graph._split_count)
        conv1 = conn_graph.get_op_from_module_name('MultiInput.conv1')
        self.assertEqual(model.conv1, conv1.get_module())
        self.assertEqual(2, len(conv1.inputs))
        conv2 = conn_graph.get_op_from_module_name('MultiInput.conv2')
        self.assertEqual(model.conv2, conv2.get_module())
        self.assertEqual(3, len(conv2.inputs))
        conv3 = conn_graph.get_op_from_module_name('MultiInput.conv3')
        self.assertEqual(model.conv3, conv3.get_module())
        self.assertEqual(3, len(conv3.inputs))

        input_ops = get_all_input_ops(conn_graph)
        input_modules = [op.get_module() for op in input_ops]
        self.assertEqual(2, len(input_ops))
        self.assertTrue(model.conv1 in input_modules)
        self.assertTrue(model.conv3 in input_modules)
        output_ops = get_all_output_ops(conn_graph)
        self.assertEqual(1, len(output_ops))
        self.assertEqual(model.fc, output_ops[0].get_module())

    def test_module_list(self):
        """ Test building ConnectedGraph on a model with module list """
        model = test_models.ModuleListModel()
        model.eval()
        inp_data_1 = torch.rand(1, 3, 8, 8)
        conn_graph = ConnectedGraph(model, (inp_data_1,))
        self.assertEqual(10, len(conn_graph.ordered_ops))
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.mod_list.4'), conn_graph.ordered_ops[0])
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.seq_list.2'), conn_graph.ordered_ops[1])
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.mod_list.1'), conn_graph.ordered_ops[2])
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.mod_list.0'), conn_graph.ordered_ops[3])
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.mod_list.2'), conn_graph.ordered_ops[4])
        self.assertEqual(conn_graph.get_op_from_module_name('ModuleListModel.seq_list.0'), conn_graph.ordered_ops[5])

    def test_concat(self):
        """ Test building ConnectedGraph on a model with concat """
        model = test_models.ConcatModel()
        model.eval()
        inp_shape_1 = (1, 3, 8, 8)
        inp_shape_2 = (1, 3, 8, 8)
        inp_shape_3 = (1, 3, 8, 8)
        inp_tensor_list = create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2, inp_shape_3], get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        concat_op = [op for op in conn_graph.get_all_ops().values() if op.type == 'Concat'][0]
        self.assertEqual(4, len(concat_op.inputs))
        self.assertEqual(14, concat_op.output_shape[1])

    def test_dropouts(self):
        """ Test building ConnectedGraph on a model with dropouts """
        # pylint: disable=protected-access
        model = test_models.ModelWithDropouts()
        model.eval()
        inp_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(inp_shape, get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(9, len(conn_graph.ordered_ops))
        # Split count of 2 due to residual as well as reshape having a split
        self.assertEqual(1, conn_graph._split_count)
        # All ops will include 2 inserted split ops
        self.assertEqual(10, len(conn_graph.get_all_ops().keys()))
        dropout_1_op = conn_graph.get_all_ops()['Dropout_3']
        dropout_2_op = conn_graph.get_all_ops()['Dropout_4']
        self.assertEqual(model.dropout1, dropout_1_op.get_module())
        self.assertEqual(model.dropout2, dropout_2_op.get_module())

    def test_sequential(self):
        # pylint: disable=protected-access
        """ Test building ConnectedGraph on a model constructed with nn.Sequential Module """
        model = test_models.SequentialModel()
        model.eval()
        inp_data_1 = torch.rand(1, 3, 8, 8)
        conn_graph = ConnectedGraph(model, (inp_data_1,))
        self.assertEqual(10, len(conn_graph.ordered_ops))
        # Expect 1 split for the reshape operation
        self.assertEqual(1, conn_graph._split_count)

    def test_hierarchical_model(self):
        """ Test building ConnectedGraph on model which multi-level aggregation of nn.Modules  """
        # pylint: disable=protected-access
        model = test_models.HierarchicalModel()
        model.eval()
        conv_shape = (1, 64, 32, 32)
        inp_shape = (1, 3, 32, 32)
        seq_shape = (1, 3, 8, 8)
        device = get_device(model)
        inp_tensor_list = create_rand_tensors_given_shapes([conv_shape, inp_shape, conv_shape, inp_shape, seq_shape], device)
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(95, len(conn_graph.ordered_ops))
        self.assertEqual(5, conn_graph._split_count)
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.conv1.conv'), conn_graph.ordered_ops[0])
        self.assertEqual(conn_graph.ordered_ops[0].residing_module, model.conv1)
        self.assertEqual(conn_graph.ordered_ops[4].type, 'narrow')
        self.assertEqual(conn_graph.ordered_ops[4].get_module(), None)
        self.assertEqual(conn_graph.ordered_ops[4].residing_module, model)
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.nm1.tm1.conv1'), conn_graph.ordered_ops[5])
        self.assertEqual(conn_graph.ordered_ops[5].residing_module, model.nm1.tm1)
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.nm1.tm2.conv1'), conn_graph.ordered_ops[20])
        self.assertEqual(conn_graph.ordered_ops[35].type, 'Concat')
        self.assertEqual(conn_graph.ordered_ops[35].get_module(), None)
        self.assertEqual(conn_graph.ordered_ops[35].residing_module, model.nm1)
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.conv2.conv'), conn_graph.ordered_ops[36])
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.multi_conv.seq_list.0.conv'), conn_graph.ordered_ops[40])
        self.assertEqual(conn_graph.ordered_ops[40].residing_module, model.multi_conv.seq_list[0])
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.nm2.tm1.conv1'), conn_graph.ordered_ops[53])
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.nm2.tm2.conv1'), conn_graph.ordered_ops[68])
        self.assertEqual(conn_graph.get_op_from_module_name('HierarchicalModel.sq.seq_list.0'), conn_graph.ordered_ops[84])
        self.assertEqual(conn_graph.ordered_ops[84].residing_module, model.sq.seq_list)
        self.assertEqual(conn_graph.ordered_ops[92].type, 'view')
        self.assertEqual(conn_graph.ordered_ops[92].get_module(), None)
        self.assertEqual(conn_graph.ordered_ops[92].residing_module, model.sq)

    def test_passthrough_op_last_module(self):
        """ Test building a connected graph on a model where a torch.nn.Identity is the last module in the graph. """
        model = test_models.PassThroughOpLastLayerModel()
        model.eval()
        inp_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(inp_shape, get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(1, len(conn_graph.ordered_ops))

    def test_multi_output_model(self):
        """ Test multi-output model with Tuple Tensor as intermediate  output. """
        model = test_models.MultiOutputModel()
        inp_data = torch.rand(1, 3, 8, 8)
        conn_graph = ConnectedGraph(model, (inp_data,))
        self.assertEqual(7, len(conn_graph.ordered_ops))
        self.assertEqual(6, len([op for op in conn_graph.get_all_ops().keys() if 'Conv' in op]))
        self.assertEqual(0, len([op for op in conn_graph.get_all_ops().keys() if 'Tuple' in op]))
        self.assertEqual(0, len([product for product in conn_graph.get_all_products().keys() if 'Tuple' in product]))
        self.assertEqual('Concat', conn_graph.ordered_ops[-1].type)

    def test_multi_output_with_unuse_model(self):
        """ Test multi-output model with Tuple Tensor as intermediate output and with one of tuple tensor not used """

        class MultiOutputWithUnuseModel(torch.nn.Module):
            """
            Model with Tuple of Tensors as output with one output tensor unused
            """
            def __init__(self):
                super(MultiOutputWithUnuseModel, self).__init__()
                self.layer = test_models.TupleOutputModel()
                self.conv1 = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(6, 4, kernel_size=3, padding=1)

            def forward(self, *inputs):
                x, _, z = self.layer(inputs[0])
                x1 = self.conv1(x)
                z1 = self.conv2(z)
                return torch.cat([x1, z1], 1)

        inp_data = torch.rand(1, 3, 8, 8)
        model = MultiOutputWithUnuseModel()
        conn_graph = ConnectedGraph(model, (inp_data,))
        self.assertEqual(6, len(conn_graph.ordered_ops))
        self.assertEqual(5, len([op for op in conn_graph.get_all_ops().keys() if 'Conv' in op]))
        self.assertEqual(0, len([op for op in conn_graph.get_all_ops().keys() if 'Tuple' in op]))
        self.assertEqual('Concat', conn_graph.ordered_ops[-1].type)

        conv0 = conn_graph.get_op_from_module_name('MultiOutputWithUnuseModel.layer.conv1')
        conv2 = conn_graph.get_op_from_module_name('MultiOutputWithUnuseModel.layer.conv3')
        conv3 = conn_graph.get_op_from_module_name('MultiOutputWithUnuseModel.conv1')
        conv4 = conn_graph.get_op_from_module_name('MultiOutputWithUnuseModel.conv2')
        concat = conn_graph.ordered_ops[-1]

        expected_products = [
            # layer #1 to conv1,conv2
            (conv0, conv3),
            (conv2, conv4),

            # conv1,conv2 to cat
            (conv3, concat),
            (conv4, concat)]

        products = conn_graph.get_all_products().values()
        for product in products:
            if (product.producer, product.consumers[0]) in expected_products:
                self.assertEqual(product.shape, product.producer.output_shape)
                expected_products.remove((product.producer, product.consumers[0]))
        self.assertEqual(0, len(expected_products))

    def test_multi_output_with_matched_layers(self):
        """ Test a multiple layer multi-output model with intermediate Tuple Tensors in order """
        class MultiOutputLayersModel(torch.nn.Module):
            """
            Model with Tuple of Tensors as output in order between layers
            """
            def __init__(self):
                super(MultiOutputLayersModel, self).__init__()
                self.layer1 = test_models.ConfigurableTupleOutputModel(channels=(1, 2, 3))
                self.layer2 = test_models.ConfigurableTupleOutputModel(channels=(1, 2, 3))
                self.layer3 = test_models.ConfigurableTupleOutputModel(channels=(1, 2, 3))

            def forward(self, *inputs):
                x1, x2, x3 = self.layer1(inputs[0], inputs[1], inputs[2])
                y1, y2, y3 = self.layer2(x1, x2, x3)
                z1, z2, z3 = self.layer3(y1, y2, y3)
                return torch.cat([z1, z2, z3], 1)

        model = MultiOutputLayersModel()
        inp_tensor_list = create_rand_tensors_given_shapes([(1, 1, 8, 8), (1, 2, 8, 8), (1, 3, 8, 8)], get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(10, len(conn_graph.ordered_ops))
        self.assertEqual(9, len([op for op in conn_graph.get_all_ops().keys() if 'Conv' in op]))
        self.assertEqual(0, len([op for op in conn_graph.get_all_ops().keys() if 'Tuple' in op]))
        self.assertEqual('Concat', conn_graph.ordered_ops[-1].type)

        product_names = conn_graph.get_all_products().keys()
        self.assertEqual(0, len([product for product in product_names if 'Tuple' in product]))
        conv0 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer1.conv1')
        conv1 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer1.conv2')
        conv2 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer1.conv3')
        conv3 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer2.conv1')
        conv4 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer2.conv2')
        conv5 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer2.conv3')
        conv6 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer3.conv1')
        conv7 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer3.conv2')
        conv8 = conn_graph.get_op_from_module_name('MultiOutputLayersModel.layer3.conv3')
        concat = conn_graph.ordered_ops[-1]

        expected_products = [
            # layer #1 to layer #2
            (conv0, conv3),
            (conv1, conv4),
            (conv2, conv5),

            # layer #2 to layer #3
            (conv3, conv6),
            (conv4, conv7),
            (conv5, conv8),

            # layer #3 to cat
            (conv6, concat),
            (conv7, concat),
            (conv8, concat)]

        products = conn_graph.get_all_products().values()
        for product in products:
            if (product.producer, product.consumers[0]) in expected_products:
                self.assertEqual(product.shape, product.producer.output_shape)
                expected_products.remove((product.producer, product.consumers[0]))
        self.assertEqual(0, len(expected_products))

    def test_multi_output_with_shuffled_layers(self):
        """ Test a multiple layer multi-output model with intermediate Tuple Tensors shuffled """
        class MultiOutputShuffledModel(torch.nn.Module):
            """
            Model with Tuple of Tensors as output shuffled between layers
            """
            def __init__(self):
                super(MultiOutputShuffledModel, self).__init__()
                self.layer1 = test_models.ConfigurableTupleOutputModel(channels=(1, 2, 3))
                self.layer2 = test_models.ConfigurableTupleOutputModel(channels=(2, 3, 1))
                self.layer3 = test_models.ConfigurableTupleOutputModel(channels=(3, 1, 2))

            def forward(self, *inputs):
                x1, x2, x3 = self.layer1(inputs[0], inputs[1], inputs[2])
                y2, y3, y1 = self.layer2(x2, x3, x1)
                z3, z1, z2 = self.layer3(y3, y1, y2)
                return torch.cat([z1, z2, z3, x1], 1)

        model = MultiOutputShuffledModel()
        inp_tensor_list = create_rand_tensors_given_shapes([(1, 1, 8, 8), (1, 2, 8, 8), (1, 3, 8, 8)], get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        self.assertEqual(10, len(conn_graph.ordered_ops))
        self.assertEqual(9, len([op for op in conn_graph.get_all_ops().keys() if 'Conv' in op]))
        self.assertEqual(0, len([op for op in conn_graph.get_all_ops().keys() if 'Tuple' in op]))
        self.assertEqual('Concat', conn_graph.ordered_ops[-1].type)

        conv0 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer1.conv1')
        conv1 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer1.conv2')
        conv2 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer1.conv3')
        conv3 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer2.conv1')
        conv4 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer2.conv2')
        conv5 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer2.conv3')
        conv6 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer3.conv1')
        conv7 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer3.conv2')
        conv8 = conn_graph.get_op_from_module_name('MultiOutputShuffledModel.layer3.conv3')
        concat = conn_graph.ordered_ops[-1]
        split = [op for op in conn_graph.get_all_ops().values() if op.type == CG_SPLIT][0]

        expected_products = [
            # layer #1 to layer #2
            (conv0, split),
            (conv1, conv3),
            (conv2, conv4),

            # layer #2 to layer #3
            (conv3, conv8),
            (conv4, conv6),
            (conv5, conv7),

            # layer #3 to cat
            (conv6, concat),
            (conv7, concat),
            (conv8, concat)]

        products = conn_graph.get_all_products().values()
        for product in products:
            if (product.producer, product.consumers[0]) in expected_products:
                self.assertEqual(product.shape, product.producer.output_shape)
                expected_products.remove((product.producer, product.consumers[0]))
        self.assertEqual(0, len(expected_products))
        split_product = conn_graph.get_all_products()['CG_Split_0__to__multiple_ops']
        self.assertTrue(conv5 in split_product.consumers)
        self.assertTrue(concat in split_product.consumers)

    def test_submodules_with_sequence_and_module_list(self):
        """ Test building ConnectedGraph on a model with sequence and module list """

        class ModuleListAndSequentialModel(torch.nn.Module):
            def __init__(self):
                super(ModuleListAndSequentialModel, self).__init__()
                self.mod_list = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        test_models.BasicConv2d(kernel_size=3),
                        test_models.BasicConv2d(kernel_size=3)
                    ),
                    torch.nn.Sequential(
                        torch.nn.Sequential(
                            test_models.BasicConv2d(kernel_size=3),
                            test_models.BasicConv2d(kernel_size=3)
                        ),
                    ),
                    torch.nn.ModuleList([
                        torch.nn.ModuleList([
                       test_models.BasicConv2d(kernel_size=3)
                        ])
                    ]),
                    test_models.ModuleListModel()]
                )

            def forward(self, *inputs):
                s1 = self.mod_list[0](inputs[0])
                s2 = self.mod_list[1](inputs[0])
                m1 = self.mod_list[2][0][0](inputs[0])
                m2 = self.mod_list[3](inputs[1])
                return s1, s2, m1,m2
        inp_data_1 = torch.rand(1, 64, 8, 8)
        inp_data_2 = torch.rand(1, 3, 8, 8)
        conn_graph = ConnectedGraph(ModuleListAndSequentialModel(), (inp_data_1, inp_data_2))
        self.assertEqual(30, len(conn_graph.ordered_ops))
        self.assertEqual(0, len([op for op in conn_graph.get_all_ops().keys() if 'Tuple' in op]))

    def test_module_reuse_model(self):
        class ReuseReluLeafModel(torch.nn.Module):
            """ A model with Relu instance used multiple times
            Expected one input of size (1, 64, 8, 8) """

            def __init__(self):
                super(ReuseReluLeafModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.relu(x)
                x = self.conv2(x)
                return self.relu(x)

        inp_data = torch.rand(1, 64, 8, 8)
        model = ReuseReluLeafModel()
        conn_graph = ConnectedGraph(model, (inp_data,))
        self.assertEqual(4, len(conn_graph.ordered_ops))
        self.assertEqual(2, len([op for name, op in conn_graph.get_all_ops().items()
                                 if 'Relu' in name and
                                 op.get_module() == model.relu]))

        class ReluModel(torch.nn.Module):
            def __init__(self):
                super(ReluModel, self).__init__()
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, *inputs):
                return self.relu( inputs[0])

        class ReuseReluLayerModel(torch.nn.Module):
            """ A model with Relu Layer instance used multiple times
            Expected one input of size (1, 64, 8, 8) """

            def __init__(self):
                super(ReuseReluLayerModel, self).__init__()
                self.conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.layer = ReluModel()

            def forward(self, *inputs):
                x = self.layer(inputs[0])
                x = self.conv(x)
                return self.layer(x)

        layer_model = ReuseReluLayerModel()
        conn_graph = ConnectedGraph(layer_model, (inp_data,))
        self.assertEqual(3, len(conn_graph.ordered_ops))
        self.assertEqual(2, len([op for name, op in conn_graph.get_all_ops().items()
                                 if 'Relu' in name and
                                 op.get_module() == layer_model.layer.relu]))

    def test_dict_input(self):
        """ Test building ConnectedGraph on a model with multiple inputs """
        # pylint: disable=protected-access
        model = test_models.DictInputModel()
        model.eval()
        inp_shape_1 = (1, 3, 32, 32)
        inp_shape_2 = (1, 3, 20, 20)
        inp_tensor_list = create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2], get_device(model))
        dict_input = {'inp_1': inp_tensor_list[0], 'inp_2': inp_tensor_list[1]}
        conn_graph = ConnectedGraph(model, dict_input)
        self.assertEqual(13, len(conn_graph.ordered_ops))

        # Split count of 1 due to reshape having a split
        self.assertEqual(1, conn_graph._split_count)
        conv1 = conn_graph.get_op_from_module_name('DictInputModel.conv1')
        self.assertEqual(model.conv1, conv1.get_module())
        self.assertEqual(2, len(conv1.inputs))
        conv2 = conn_graph.get_op_from_module_name('DictInputModel.conv2')
        self.assertEqual(model.conv2, conv2.get_module())
        self.assertEqual(3, len(conv2.inputs))
        conv3 = conn_graph.get_op_from_module_name('DictInputModel.conv3')
        self.assertEqual(model.conv3, conv3.get_module())
        self.assertEqual(3, len(conv3.inputs))

        input_ops = get_all_input_ops(conn_graph)
        self.assertEqual(2, len(input_ops))

        self.assertTrue(model.conv1 is input_ops[0].output.consumers[0].get_module())
        self.assertTrue(model.conv3 is input_ops[1].output.consumers[0].get_module())
        output_ops = get_all_output_ops(conn_graph)
        self.assertEqual(1, len(output_ops))
        self.assertEqual(model.fc, output_ops[0].get_module())

    def test_nested_sequential(self):
        # pylint: disable=protected-access
        """ Test building ConnectedGraph on a model constructed with nested nn.Sequential Module """
        model = test_models.NestedSequentialModel()
        model.eval()
        inp_data_1 = torch.rand(1, 3, 8, 8)
        conn_graph = ConnectedGraph(model, (inp_data_1,))
        self.assertEqual(10, len(conn_graph.ordered_ops))
        # Expect 1 split for the reshape operation
        self.assertEqual(1, conn_graph._split_count)

    def test_lstm_with_tuple_input(self):
        model = test_models.LinearAndLSTMModel()
        model.eval()
        h = torch.randn((2, 1, 5))
        c = torch.randn((2, 1, 5))
        rand_inp = torch.randn((5, 10))
        conn_graph = ConnectedGraph(model, model_input=(rand_inp, (h, c)))
        self.assertEqual(4, len(conn_graph.ordered_ops))
        self.assertEqual(9, len(conn_graph.get_all_products()))

    def test_graph_construction_with_split_module(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = test_models.ModelWithReluAfterSplit().to(device)
        model.eval()
        dummy_input = torch.randn(6, 2, device=device)

        connected_graph = ConnectedGraph(model, model_input=dummy_input)
        # Four ops from PyTorch modules (Split, ReLU, ReLU, ReLU) and one SplitOp inserted by ConnectedGraph construction
        self.assertEqual(4 + 1, len(connected_graph.get_all_ops()))

        # ordered_ops filter out SplitOp inserted by ConnectedGraph
        self.assertEqual(4, len(connected_graph.ordered_ops))

    def test_graph_input_structure(self):
        model = test_models.SingleResidual()
        model.eval()
        inp_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(inp_shape, get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)

        self.assertEqual(model.conv1, conn_graph.input_structure[0].op.get_module())

    def test_graph_output_structure(self):
        model = test_models.SingleResidual()
        model.eval()
        inp_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(inp_shape, get_device(model))
        conn_graph = ConnectedGraph(model, inp_tensor_list)

        self.assertEqual(model.fc, conn_graph.output_structure.op.get_module())

    def test_graph_input_structure_with_nested_inputs_and_outputs(self):
        class ModelWithMultiInputMultiOutput(nn.Module):
            def __init__(self):
                super(ModelWithMultiInputMultiOutput, self).__init__()
                self.conv1_a_i = nn.Conv2d(1, 10, kernel_size=5)
                self.maxpool1_a = nn.MaxPool2d(2)
                self.relu1_a = nn.ReLU()

                self.conv1_a_ii = nn.Conv2d(1, 10, kernel_size=5)

                self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
                self.maxpool1_b = nn.MaxPool2d(2)
                self.relu1_b = nn.ReLU()

                self.conv1_c = nn.Conv2d(1, 10, kernel_size=5)
                self.maxpool1_c = nn.MaxPool2d(2)
                self.relu1_c = nn.ReLU()

                self.conv2_a = nn.Conv2d(10, 20, kernel_size=5)
                self.maxpool2_a = nn.MaxPool2d(2)
                self.relu2_a = nn.LeakyReLU()

                self.conv2_b = nn.Conv2d(10, 20, kernel_size=5)
                self.maxpool2_b = nn.MaxPool2d(2)
                self.relu2_b = nn.LeakyReLU()

                self.softmax_1 = nn.LogSoftmax(dim=1)
                self.softmax_2 = nn.LogSoftmax(dim=1)

            def forward(self, x1, x23):
                x2, x3 = x23

                x1 = self.relu1_a(self.maxpool1_a(self.conv1_a_i(x1) + self.conv1_a_ii(x1)))
                x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
                x3 = self.relu1_c(self.maxpool1_c(self.conv1_c(x3)))
                y1 = x1 + x2
                y2 = x2 + x3

                y1 = y1.transpose(2, 3)
                y1 = self.relu2_a(self.maxpool2_a(self.conv2_a(y1)))
                y1 = self.softmax_1(y1)

                y2 = self.relu2_b(self.maxpool2_b(self.conv2_b(y2)))
                y2 = self.softmax_2(y2)

                return y1, y2, (x1, x2, x3)

        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        input_tensor = create_rand_tensors_given_shapes([input_shape, [input_shape, input_shape]], get_device(model))
        conn_graph = ConnectedGraph(model, input_tensor)

        self.assertEqual(model.conv1_a_i, conn_graph.input_structure[0][0].op.get_module())
        self.assertEqual(model.conv1_a_ii, conn_graph.input_structure[0][1].op.get_module())
        self.assertEqual(model.conv1_b, conn_graph.input_structure[1][0][0].op.get_module())
        self.assertEqual(model.conv1_c, conn_graph.input_structure[1][1][0].op.get_module())
        self.assertEqual(model.relu1_a, conn_graph.output_structure[2][0].op.get_module())
        self.assertEqual(model.relu1_b, conn_graph.output_structure[2][1].op.get_module())
        self.assertEqual(model.relu1_c, conn_graph.output_structure[2][2].op.get_module())
        self.assertEqual(model.softmax_1, conn_graph.output_structure[0].op.get_module())
        self.assertEqual(model.softmax_2, conn_graph.output_structure[1].op.get_module())
        return

    def test_graph_input_output_with_passthrough(self):
        class ModelWithPassthroughOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3)

            def forward(self, x, y):
                x = self.conv(x)
                return x, y

        model = ModelWithPassthroughOutput()
        input_shape = (1, 3, 3, 3)
        input_tensor_list = create_rand_tensors_given_shapes([input_shape, input_shape], get_device(model))
        conn_graph = ConnectedGraph(model, input_tensor_list)

        self.assertEqual(model.conv, conn_graph.input_structure[0][0].op.get_module())
        self.assertEqual([None], conn_graph.input_structure[1])
        self.assertEqual(model.conv, conn_graph.output_structure[0].op.get_module())
        self.assertEqual(None, conn_graph.output_structure[1])

    def test_graph_input_output_with_conditional(self):
        class LinearConditionalModel(torch.nn.Module):
            def __init__(self):
                super(LinearConditionalModel, self).__init__()
                self.l1 = torch.nn.Linear(in_features=128, out_features=128)
                self.l2 = torch.nn.Linear(in_features=128, out_features=128)
                self.l3 = torch.nn.Linear(in_features=128, out_features=128)

            def forward(self, x, y):
                if y:
                    return self.l1(x)
                else:
                    return self.l2(x) + self.l3(x)

        model = LinearConditionalModel()
        model.eval()

        dummy_input = (torch.randn(128, 128), torch.tensor(data=True, dtype=torch.bool))
        conn_graph = ConnectedGraph(model, dummy_input)

        self.assertEqual(model.l1, conn_graph.input_structure[0][0].op.get_module())
        self.assertEqual(model.l1, conn_graph.output_structure.op.get_module())

    def test_graph_input_output_indices(self):
        class ModelWithMatMul(nn.Module):
            def __init__(self):
                super(ModelWithMatMul, self).__init__()
                self.matmul = aimet_modules.MatMul()

            def forward(self, x, y):
                z = self.matmul(y, x)
                return z

        model = ModelWithMatMul()
        model.eval()

        dummy_input = (torch.randn(128, 128), torch.randn(128, 128))
        conn_graph = ConnectedGraph(model, dummy_input)

        self.assertEqual(model.matmul, conn_graph.input_structure[0][0].op.get_module())
        self.assertEqual(model.matmul, conn_graph.input_structure[1][0].op.get_module())
        self.assertEqual(model.matmul, conn_graph.output_structure.op.get_module())

        self.assertEqual(1, conn_graph.input_structure[0][0].index)
        self.assertEqual(0, conn_graph.input_structure[1][0].index)
        self.assertEqual(0, conn_graph.output_structure.index)


class ModelWithMultipleActivations(nn.Module):
    def __init__(self):
        super(ModelWithMultipleActivations, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.hardshrink = nn.Hardshrink()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.tanhshrink = nn.Tanhshrink()
        self.conv4 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.mish = nn.Mish()
        self.conv5 = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.softmax2d = nn.Softmax2d()

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.hardshrink(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.tanhshrink(x)
        x = self.conv4(x)
        x = self.mish(x)
        x = self.conv5(x)
        x = self.softmax2d(x)
        return x


class TestConnectedGraphUtils(unittest.TestCase):
    """ Unit tests for testing connectedgraph_utils module"""

    def test_get_module_act_func_pair_with_modules(self):
        """ Test get module activation function pair - activations are nn.Modules """

        model = test_models.TinyModel().eval()
        inp_tensor_list = [torch.randn(1, 3, 32, 32)]

        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, inp_tensor_list)

        # 12 modules
        self.assertEqual(len(module_act_func_pair), 12)

        # followed by relu case
        self.assertTrue(isinstance(module_act_func_pair[model.bn1], torch.nn.ReLU))
        self.assertTrue(isinstance(module_act_func_pair[model.bn2], torch.nn.ReLU))
        self.assertTrue(isinstance(module_act_func_pair[model.conv3], torch.nn.ReLU))

        # not followed by relu case
        self.assertEqual(module_act_func_pair[model.conv1], None)
        self.assertEqual(module_act_func_pair[model.conv2], None)

        # final module case
        self.assertEqual(module_act_func_pair[model.fc], None)

    def test_get_module_act_func_pair_for_activations(self):
        model = ModelWithMultipleActivations().eval()
        inp_tensor_list = [torch.randn(1, 3, 32, 32)]

        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, inp_tensor_list)

        # followed by activation case
        self.assertTrue(isinstance(module_act_func_pair[model.bn1], torch.nn.Hardshrink))
        self.assertTrue(isinstance(module_act_func_pair[model.bn2], torch.nn.GELU))
        self.assertTrue(isinstance(module_act_func_pair[model.conv3], torch.nn.Tanhshrink))
        self.assertTrue(isinstance(module_act_func_pair[model.conv4], torch.nn.Mish))
        self.assertTrue(isinstance(module_act_func_pair[model.conv5], torch.nn.Softmax2d))

        # followed by non-activation case
        self.assertEqual(module_act_func_pair[model.conv1], None)
        self.assertEqual(module_act_func_pair[model.hardshrink], None)

    def test_get_ops_with_missing_modules(self):
        """ Check that get ops with missing modules reports ops with missing modules correctly """

        model = test_models.ModelWithFunctionalOps()
        rand_inp = torch.randn(1, 3, 32, 32)
        ops_with_missing_modules = connectedgraph_utils.get_ops_with_missing_modules(model, rand_inp)
        self.assertEqual(2, len(ops_with_missing_modules))

    def test_find_nodes_in_forward_pass_for_elementwise_ops(self):
        """ Check _find_nodes_in_forward_pass() method for elementwise_ops """
        # 1) aimet_modules.Add()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.Add(), dummy_input)
        nodes =  ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 2) aimet_modules.Subtract()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.Subtract(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 3) aimet_modules.Multiply()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.Multiply(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 4) aimet_modules.Divide()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.Divide(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 5) aimet_modules.MatMul()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.MatMul(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 6) aimet_modules.Concat()
        dummy_input = (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4))
        trace = torch.jit.trace(aimet_modules.Concat(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

    @pytest.mark.cuda
    def test_find_nodes_in_forward_pass_for_custom_module(self):
        """ Check _find_nodes_in_forward_pass() method for custom module """

        class CustomModule(torch.nn.Module):
            @staticmethod
            def forward(x: torch.Tensor):
                y = x.detach()
                return y * torch.nn.functional.softplus(x).relu()

        dummy_input = torch.randn(1, 3, 4, 4)
        trace = torch.jit.trace(CustomModule().cuda(), dummy_input.cuda())
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        # mulitply, softplus and relu ops.
        # detach is considered as passthrough op.
        assert len(nodes) == 3

    def test_find_nodes_in_forward_pass_for_custom_conv2d_module(self):
        """ Check _find_nodes_in_forward_pass() method for custom module """

        class CustomModule(torch.nn.Module):
            @staticmethod
            def forward(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
                return torch.nn.functional.conv2d(inp, weight, bias)

        dummy_input = torch.randn(1, 3, 4, 4)
        dummy_weight = torch.randn(32, 3, 1, 1)
        dummy_bias = torch.randn((32,))
        dummy_input = (dummy_input, dummy_weight, dummy_bias)
        trace = torch.jit.trace(CustomModule(), dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

    def test_find_nodes_in_forward_pass_for_torch_nn_module(self):
        """ Check _find_nodes_in_forward_pass() method for torch.nn modules """

        # 1) Conv2d
        dummy_input = torch.randn(1, 3, 4, 4)
        conv = torch.nn.Conv2d(3, 3, 2).eval()
        print(conv.__class__)
        trace = torch.jit.trace(conv, dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 2) ReLU
        dummy_input = torch.randn(1, 3, 4, 4)
        relu = torch.nn.ReLU(inplace=True).eval()
        trace = torch.jit.trace(relu, dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

        # 3) BatchNorm2d
        dummy_input = torch.randn(1, 3, 4, 4)
        bn = torch.nn.BatchNorm2d(3).eval()
        trace = torch.jit.trace(bn, dummy_input)
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(trace)
        assert len(nodes) == 1

    def test_find_nodes_in_forward_pass_for_unused_module(self):
        """ test _find_nodes_in_forward_pass() for unused module """

        class MultiOutputWithUnuseModel(torch.nn.Module):
            """
            Model with Tuple of Tensors as output with one output tensor unused
            """
            def __init__(self):
                super(MultiOutputWithUnuseModel, self).__init__()
                self.layer = test_models.TupleOutputModel()
                self.conv1 = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(6, 4, kernel_size=3, padding=1)

            def forward(self, *inputs):
                x, _, z = self.layer(inputs[0])
                x1 = self.conv1(x)
                z1 = self.conv2(z)
                return torch.cat([x1, z1], 1)

        dummy_input = torch.rand(1, 3, 8, 8)
        model = MultiOutputWithUnuseModel().eval()
        print(model.conv1.__class__)
        trace = torch.jit.trace(model, dummy_input)
        trace = getattr(trace, "layer")

        # Conv2 is unused in forward pass.
        conv2_trace = getattr(trace, "conv2")
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(conv2_trace)
        assert len(nodes) == 0

        # Conv1 and Conv3 are used in forward pass.
        conv1_trace = getattr(trace, "conv1")
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(conv1_trace)
        assert len(nodes) == 1

    def test_find_nodes_in_forward_pass_for_undefined_graph(self):
        """ test _find_nodes_in_forward_pass() for undefined trace graph """

        dummy_input = torch.rand(1, 3, 8, 8)
        model = test_models.NestedSequentialModel().eval()
        print(model.inner_seq[1].__class__)
        trace = torch.jit.trace(model, dummy_input)

        inner_seq_trace = getattr(trace, "inner_seq")
        bn_trace = getattr(inner_seq_trace, "1")
        nodes = ConnectedGraph._find_aten_nodes_in_forward_pass(bn_trace)
        assert len(nodes) == 0

        with pytest.raises(RuntimeError):
            _ = bn_trace.graph

    def test_constant_elementwise_inputs(self):
        """ Test that constant inputs to elementwise ops are identified correctly """
        class ConstantElementwiseInputModel(torch.nn.Module):
            def __init__(self):
                super(ConstantElementwiseInputModel, self).__init__()
                self.add = aimet_modules.Add()
                self.mul = aimet_modules.Multiply()

            def forward(self, inp):
                x = self.add(inp, torch.tensor(2.0))
                x = self.mul(torch.tensor(3.0), x)
                return x

        model = ConstantElementwiseInputModel()
        cg = ConnectedGraph(model, model_input=torch.randn(1, 6))
        assert len(get_all_ops_with_constant_inputs(cg)) == 2
        assert not cg.ordered_ops[0].inputs[0].is_const
        assert cg.ordered_ops[0].inputs[1].is_const
        assert cg.ordered_ops[1].inputs[0].is_const
        assert not cg.ordered_ops[1].inputs[1].is_const

    def test_constant_single_input(self):
        class ConstantSingleInputModel(torch.nn.Module):
            def __init__(self):
                super(ConstantSingleInputModel, self).__init__()
                self.relu = torch.nn.ReLU()
                self.relu2 = torch.nn.ReLU()
                self.add = aimet_modules.Add()
                self.add2 = aimet_modules.Add()
                self.register_buffer('constant_1', torch.tensor([3.0, 4.0]))

            def forward(self, inp):
                x = self.relu(torch.tensor([-1.0, 1.0]))
                y = self.relu2(self.constant_1)
                x = self.add(x, inp)
                x = self.add2(x, y)
                return x

        model = ConstantSingleInputModel()
        dummy_input = torch.randn(1, 2)
        cg = ConnectedGraph(model, model_input=dummy_input)
        assert cg.ordered_ops[0].inputs[0].is_const
        assert cg.ordered_ops[1].inputs[0].is_const
        assert not cg.ordered_ops[2].inputs[0].is_const
        assert not cg.ordered_ops[2].inputs[1].is_const
        assert not cg.ordered_ops[3].inputs[0].is_const
        assert not cg.ordered_ops[3].inputs[1].is_const

    def test_model_with_non_leaf_candidate_0(self):
        """
        Model: [C1 -> L1] -> L2
        Where [C1 -> L1] is non leaf module wrapped under ConvLinearModel
        ConvLinearModel is made to be considered a leaf
        """
        class ConvLinearModel(torch.nn.Module):
            def __init__(self):
                super(ConvLinearModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 12, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
                self.linear1 = nn.Linear(32, 32, bias=False)

            def forward(self, inp):
                y = self.conv1(inp)
                y = self.linear1(y)
                return y

        class TopLevelModel(torch.nn.Module):
            def __init__(self):
                super(TopLevelModel, self).__init__()
                self.layer1 = ConvLinearModel()
                self.linear1 = nn.Linear(32, 64, bias=False)

            def forward(self, inp):
                y = self.layer1(inp)
                y = self.linear1(y)
                return y

        model = TopLevelModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)

        aimet_torch.utils.modules_to_treat_as_leaf = [ConvLinearModel]

        cg_1 = ConnectedGraph(model, model_input=dummy_input)
        assert len(cg_1.ordered_ops) == 2

        assert len(cg_1.ordered_ops[0].inputs) == 3
        assert cg_1.ordered_ops[0].inputs[0].name == 'input_0_to_ConvLinearModel_0'
        assert cg_1.ordered_ops[0].inputs[0].is_model_input == True

        assert cg_1.ordered_ops[0].inputs[1].name == 'TopLevelModel.layer1.conv1.weight'
        assert cg_1.ordered_ops[0].inputs[1].is_model_input == False
        assert cg_1.ordered_ops[0].inputs[1].is_parm == True

        assert cg_1.ordered_ops[0].inputs[2].name == 'TopLevelModel.layer1.linear1.weight'
        assert cg_1.ordered_ops[0].inputs[2].is_model_input == False
        assert cg_1.ordered_ops[0].inputs[2].is_parm == True

    def test_model_with_non_leaf_candidate_1(self):
        """
        Model: L1 -> [C1 -> L2] -> L3
        Where [C1 -> L2] is non leaf module wrapped under ConvLinearModel
        ConvLinearModel is made to be considered a leaf
        """
        class ConvLinearModel(torch.nn.Module):
            def __init__(self):
                super(ConvLinearModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 12, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
                self.linear1 = nn.Linear(3, 3, bias=False)

            def forward(self, inp):
                y = self.conv1(inp)
                y = self.linear1(y)
                return y

        class TopLevelModel(torch.nn.Module):
            def __init__(self):
                super(TopLevelModel, self).__init__()
                self.linear1 = nn.Linear(3, 3, bias=False)
                self.layer1 = ConvLinearModel()
                self.linear2 = nn.Linear(3, 6, bias=False)

            def forward(self, inp):
                y = self.linear1(inp)
                y = self.layer1(y)
                y = self.linear2(y)
                return y

        model = TopLevelModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 3, 3)

        #out = model(dummy_input)
        aimet_torch.utils.modules_to_treat_as_leaf = [ConvLinearModel]

        cg_1 = ConnectedGraph(model, model_input=dummy_input)

        # three ops in total: linear1, layer1(non-leaf), linear2
        assert len(cg_1.ordered_ops) == 3

        assert len(cg_1.ordered_ops[1].inputs) == 3

        assert cg_1.ordered_ops[0].inputs[0].name == 'input_0_to_Gemm_0'
        assert cg_1.ordered_ops[0].inputs[0].is_model_input == True
        assert cg_1.ordered_ops[0].inputs[1].name == 'TopLevelModel.linear1.weight'
        assert cg_1.ordered_ops[0].inputs[1].is_parm == True

        assert cg_1.ordered_ops[1].inputs[0].name == 'Gemm_0_to_ConvLinearModel_1'
        assert cg_1.ordered_ops[1].inputs[0].is_model_input == False
        assert cg_1.ordered_ops[1].inputs[1].is_parm == True
        assert cg_1.ordered_ops[1].inputs[2].is_parm == True

        assert cg_1.ordered_ops[2].inputs[0].name == 'ConvLinearModel_1_to_Gemm_2'
        assert cg_1.ordered_ops[2].inputs[0].is_model_input == False
        assert cg_1.ordered_ops[2].inputs[1].is_parm == True

    def test_conv_bn_mangle_nodes(self):
        class ModelWithConvBNMangleNodes(torch.nn.Module):
            def __init__(self):
                super(ModelWithConvBNMangleNodes, self).__init__()
                self.inner_seq = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(16)
                )
                self.seq_list = nn.Sequential(
                    self.inner_seq,
                    nn.ReLU(inplace=True)
                )

            def forward(self, inp):
                return self.inner_seq(inp)

        dummy_input = torch.randn(1, 3, 8, 8)
        model = ModelWithConvBNMangleNodes()
        cg = ConnectedGraph(model, dummy_input)
        assert len(cg.ordered_ops[0].inputs) == 2
        assert len(cg.ordered_ops[1].inputs) == 5
        for inp in cg.ordered_ops[0].inputs[1:]:
            assert inp.is_parm
            assert not inp.is_const
        for inp in cg.ordered_ops[1].inputs[1:]:
            assert inp.is_parm
            assert not inp.is_const

    def test_remove_inputs_for_ops(self):
        class MockConnectedGraph(ConnectedGraph):
            def __init__(self):
                self._ops = {}
                self._products = {}

        mcg = MockConnectedGraph()

        conv_1 = Op('conv_1', 'conv_1', None, False, 'Conv', None)
        p1 = Product('p1', None)
        p2 = Product('p2', None)
        conv_1.add_input(p1)
        conv_1.add_input(p2)
        mcg._ops[conv_1.name] = conv_1
        mcg._products[p1.name] = p1
        mcg._products[p2.name] = p2

        add_1 = Op('add_1', 'add_1', None, False, 'Add', None)
        p3 = Product('p3', None)
        p4 = Product('p4', None)
        p5 = Product('p5', None)
        add_1.add_input(p3)
        add_1.add_input(p4)
        add_1.add_input(p5)
        mcg._ops[add_1.name] = add_1
        mcg._products[p3.name] = p3
        mcg._products[p4.name] = p4
        mcg._products[p5.name] = p5

        assert len(conv_1.inputs) == 2
        assert conv_1.inputs == [p1, p2]
        assert p1.name in mcg._products.keys()

        assert len(add_1.inputs) == 3
        assert add_1.inputs == [p3, p4, p5]
        assert p5.name in mcg._products.keys()

        mcg._remove_inputs_for_ops()

        assert len(conv_1.inputs) == 1
        assert conv_1.inputs == [p2]
        assert p1.name not in mcg._products.keys()

        assert len(add_1.inputs) == 2
        assert add_1.inputs == [p3, p4]
        assert p5.name not in mcg._products.keys()

    def test_custom_ops_to_treat_as_leaf_module(self):
        class RMSNormModel(torch.nn.Module):
            def __init__(self):
                super(RMSNormModel, self).__init__()
                self.rms_norm_0 = aimet_modules.RmsNorm([5, 2, 3], [2], 1e-5)

            def forward(self, inp):
                x = self.rms_norm_0(inp)
                return x

        model = RMSNormModel()
        dummy_input = torch.randn(5, 2, 3)

        cg = ConnectedGraph(model, dummy_input)
        assert len(cg.get_all_ops()) == 1
        assert cg.ordered_ops[0].inputs[0].is_model_input
        assert cg.ordered_ops[0].get_module() == model.rms_norm_0
