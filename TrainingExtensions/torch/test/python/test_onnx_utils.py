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

import contextlib
import copy
import logging
import os
import tempfile
from collections import defaultdict

import aimet_torch._base.nn.modules.custom as aimet_modules
import onnx
import pytest
import torch
from aimet_common.utils import AimetLogger
from aimet_torch import onnx_utils
from aimet_torch.model_preparer import prepare_model
from aimet_torch.onnx_utils import (
    OnnxSaver,
    get_pytorch_name_from_onnx_name,
    restore_onnx_graph_initializers,
    save_initializer_restored_onnx_graph,
)
from torchvision import models

from .models.test_models import (
    InputOutputDictModel,
    ModelWithModuleList,
    ModelWithReusedInitializers,
    MultiplePReluModel,
    NestedModelWithOverlappingNames,
    NestedSeqModel,
    RoiModel,
)


class OutOfOrderModel(torch.nn.Module):

    def __init__(self):
        super(OutOfOrderModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(32, 32, 3)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        y1 = self.conv2(x)
        y1 = self.bn2(y1)
        y1 = self.relu2(y1)

        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)

        return x1 + y1

class MultiplyModuleAndFunctional(torch.nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.mul = aimet_modules.Multiply()
        self.factor = factor

    def forward(self, x):
        return self.mul(x, self.factor) * 3

class HierarchicalMultiplyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul1 = MultiplyModuleAndFunctional(4)
        self.mul2 = MultiplyModuleAndFunctional(5)

    def forward(self, x):
        x = self.mul1(x)
        return self.mul2(x) * 6


class TestOnnxUtils:

    @staticmethod
    def check_onnx_node_name_uniqueness(onnx_model):
        """
        utility to check if node names are unique
        """
        onnx_node_names = [node.name for node in onnx_model.graph.node]
        assert len(onnx_node_names) == len(set(onnx_node_names)), f'list size mismatch, check if names are unique'

    def test_add_pytorch_node_names_to_onnx_resnet(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        model_name = 'resnet18'
        model = models.resnet18(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, model_name + '.onnx'), model, dummy_input,
                                                is_conditional=False, module_marker_map={})

            onnx_model = onnx.load(os.path.join(tmp_dir,  model_name + '.onnx'))
            self.check_onnx_node_names(onnx_model)

    def check_onnx_node_names(self, onnx_model):
        name_to_bn_node_map = {}
        for node in onnx_model.graph.node:
            if node.op_type in ('Conv', 'Gemm', 'MaxPool'):
                assert node.name
            if node.op_type == 'BatchNormalization':
                name_to_bn_node_map[node.name] = node

            for in_tensor in node.input:
                if in_tensor.endswith('weight'):
                    tensor_context = in_tensor[:-7]
                    assert node.name == tensor_context or tensor_context in name_to_bn_node_map

    def test_add_pytorch_node_names_to_onnx_ooo(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        model_name = 'out_of_order'
        model = OutOfOrderModel()
        dummy_input = torch.randn(1, 16, 20, 20)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir,  model_name + '.onnx'), model, dummy_input,
                                                is_conditional=False, module_marker_map={})

            onnx_model = onnx.load(os.path.join(tmp_dir,  model_name + '.onnx'))
            self.check_onnx_node_names(onnx_model)

    def test_onnx_node_name_to_input_output_names_util(self):
        """ test onxx based utility to find mapping between onnx node names and io tensors"""
        model = models.resnet18(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.onnx.export(model, dummy_input, os.path.join(tmp_dir, 'resnet18.onnx'))
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'resnet18.onnx'), model, dummy_input,
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'resnet18.onnx'))

            # Get Dict mapping node name to the input and output names
            node_to_io_dict,_ = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

            node_0 = onnx_model.graph.node[0]
            assert node_0.input == node_to_io_dict[node_0.name].inputs
            assert node_0.output == node_to_io_dict[node_0.name].outputs

    def test_single_pytorch_module_mapping_to_many_onnx_nodes(self):
        """ test onxx based utility to find mapping between onnx node names and io tensors
        when more than one onnx node maps to the same torch module
        """

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        class TwoLayerLstmModel(torch.nn.Module):
            """
            Model using torch.nn.LSTM module
            """
            def __init__(self):
                super(TwoLayerLstmModel, self).__init__()
                self.lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=3)

            def forward(self, x, hx=None):
                return self.lstm(x, hx)

        model_name = 'multilayer_lstm'
        model = TwoLayerLstmModel()
        dummy_input = torch.randn(10, 1, 3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.onnx.export(model, dummy_input, os.path.join(tmp_dir, model_name + '.onnx'))
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, model_name + '.onnx'), model, dummy_input,
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, model_name + '.onnx'))

            lstm_nodes = [node for node in onnx_model.graph.node if node.op_type == 'LSTM']
            assert 3 == len(lstm_nodes)

            node_to_io_dict, _ = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
            assert isinstance(node_to_io_dict['lstm#root_node'], list)
            assert 3 == len(node_to_io_dict['lstm#root_node'])

    def test_onnx_export_complex_model(self):

        from aimet_torch._base.nn.modules.custom import Add

        class ResidualLayer1(torch.nn.Module):
            def __init__(self):
                super(ResidualLayer1, self).__init__()
                self.conv1 = torch.nn.Conv2d(20, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.relu2 = torch.nn.ReLU()

                self.conv3 = torch.nn.Conv2d(20, 20, 3)
                self.relu3 = torch.nn.ReLU()
                self.conv4 = torch.nn.Conv2d(20, 20, 3)
                self.relu4 = torch.nn.ReLU()

            def forward(self, x):
                y1 = self.conv1(x)
                y1 = self.relu1(y1)
                y2 = self.conv2(x)
                y2 = self.relu2(y2)

                y1 = self.conv3(y1)
                y1 = self.relu3(y1)
                y2 = self.conv4(y2)
                y2 = self.relu4(y2)

                return y1 + y2

        class ResidualLayer2(torch.nn.Module):
            def __init__(self):
                super(ResidualLayer2, self).__init__()
                self.conv1 = torch.nn.Conv2d(20, 20, 3, padding=(1, 1))
                self.relu1 = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(20, 20, 3, padding=(1, 1))
                self.relu3 = torch.nn.ReLU()

                self.conv2 = torch.nn.Dropout2d()

                self.conv4 = torch.nn.Conv2d(20, 20, 3, padding=(1, 1))
                self.relu4 = torch.nn.ReLU()
                self.add = Add()

            def forward(self, x):
                y1 = self.conv1(x)
                y1 = self.relu1(y1)
                y1 = self.conv3(y1)
                y1 = self.relu3(y1)

                y2 = self.conv2(x)

                y3 = self.conv4(x)
                y3 = self.relu4(y3)

                y2 = self.add(y1, y2)
                return y2 + y3

        class TwoLevelLayer(torch.nn.Module):
            def __init__(self):
                super(TwoLevelLayer, self).__init__()
                self.conv1 = torch.nn.Conv2d(20, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(20, 20, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                return x

        class CustomLayer(torch.nn.Module):
            def __init__(self):
                super(CustomLayer, self).__init__()

            def forward(self, x):
                x = x * 10
                x = torch.nn.functional.relu(x)
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv0 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.custom = CustomLayer()
                self.block1 = TwoLevelLayer()
                self.block2 = ResidualLayer1()
                self.block3 = ResidualLayer2()
                self.add = Add()
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                z = self.conv0(x)
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.custom(x)
                x = self.conv2(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.add(x, x)
                x = x[:, :, 0, 0]
                x = x.reshape(x.shape[0], x.shape[1], 1, 1)
                x = self.relu1(x)
                return x

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))
            expected_conv_names = ['conv1', 'conv2', 'block1.conv1', 'block1.conv2', 'block2.conv1', 'block2.conv2',
                                   'block2.conv3', 'block2.conv4', 'block3.conv1', 'block3.conv3', 'block3.conv4']
            expected_other_node_names = ['relu1', 'add', 'block1.relu1', 'block2.relu1', 'block2.relu2', 'block2.relu3',
                                         'block2.relu4']
            not_expected_names = ['conv0']

            module_names = { module_name for module_name, _ in model.named_modules()}
            for node in onnx_model.graph.node:
                if not node.name.startswith('.'):
                    name = node.name.split('#')[0]
                    assert '.'.join(name.split('.')[:-1]) in module_names

                if node.op_type == 'Conv':
                    assert node.name in expected_conv_names

            actual_node_names = [node.name for node in onnx_model.graph.node]
            for name in expected_other_node_names:
                assert name in actual_node_names
            for name in not_expected_names:
                assert name not in actual_node_names

    def test_onnx_export_model_input_empty_layer(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.drop0 = torch.nn.Dropout2d()
                self.drop1 = torch.nn.Dropout2d()
                self.conv0 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)

            def forward(self, x):
                x = self.drop0(x)
                x = self.drop1(x)
                x = self.conv0(x)
                x = self.conv2(x)

                return x

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv0', 'conv2']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            for name in expected_nodes:
                assert name in actual_nodes

    def test_onnx_export_model_output_empty_layer(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv0 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.drop0 = torch.nn.Dropout2d()
                self.drop1 = torch.nn.Dropout2d()

            def forward(self, x):
                x = self.conv0(x)
                x = self.conv2(x)
                x = self.drop0(x)
                x = self.drop1(x)

                return x

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv0', 'conv2']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            for name in expected_nodes:
                assert name in actual_nodes

    def test_onnx_export_model_empty_layer_consumed_by_multiple_nodes(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv0 = torch.nn.Conv2d(10, 20, 3)
                self.drop0 = torch.nn.Dropout2d()
                self.drop1 = torch.nn.Dropout2d()
                self.conv1 = torch.nn.Conv2d(20, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)

            def forward(self, x):
                x = self.conv0(x)
                x = self.drop0(x)
                x = self.drop1(x)
                y1 = self.conv1(x)
                y2 = self.conv2(x)

                return y1, y2

        model = MyModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv0', 'conv1', 'conv2']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            for name in expected_nodes:
                assert name in actual_nodes

    def test_onnx_export_model_input_empty_layer_consumed_by_multiple_nodes(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.drop0 = torch.nn.Dropout2d()
                self.drop1 = torch.nn.Dropout2d()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(10, 20, 3)

            def forward(self, x):
                x = self.drop0(x)
                x = self.drop1(x)
                y1 = self.conv1(x)
                y2 = self.conv2(x)

                return y1, y2

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv1', 'conv2']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            for name in expected_nodes:
                assert name in actual_nodes

    def test_onnx_export_intermediate_tensor_also_model_output(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.conv3 = torch.nn.Conv2d(20, 20, 3)

            def forward(self, x):
                x = self.conv1(x)

                y1 = self.conv2(x)
                y2 = self.conv3(x)

                return y1, y2, x

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model, dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv1', 'conv2', 'conv3']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            for name in expected_nodes:
                assert name in actual_nodes

    def test_onnx_export_intermediate_tensor_also_model_output_via_empty_marker(self):

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.conv3 = torch.nn.Conv2d(20, 20, 3)
                self.drop1 = torch.nn.Dropout2d()
                self.drop2 = torch.nn.Dropout2d()

            def forward(self, x):
                x = self.conv1(x)

                y1 = self.conv2(x)
                y2 = self.conv3(x)
                y3 = self.drop1(x)
                y4 = self.drop2(x)

                return y1, y2, y3, y4

        model = MyModel()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'MyModel.onnx'), model,
                                                dummy_input=torch.rand(1, 10, 24, 24),
                                                is_conditional=False, module_marker_map={})
            onnx_model = onnx.load(os.path.join(tmp_dir, 'MyModel.onnx'))

            expected_nodes = ['conv1', 'conv2', 'conv3']
            actual_nodes = [node.name for node in onnx_model.graph.node]
            assert len(actual_nodes) == len(expected_nodes)

            self.check_onnx_node_name_uniqueness(onnx_model)

            for name in expected_nodes:
                assert name in actual_nodes

    @pytest.mark.parametrize("export_args", [None, {"opset_version": 12}])
    def test_set_node_name_for_matmul_add_linear(self, export_args):
        """
        Test that node names are set correctly for linear ops turned into matmul/add in onnx.
        """
        class Linear(torch.nn.Module):
            def __init__(self):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, inp):
                x = self.linear(inp)
                return x

        model = Linear()
        # Using an input to linear op with dimension != 2 causes torch to use matmul->add instead of gemm op
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'MyModel.onnx')
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input=torch.randn(1, 1, 3), onnx_export_args=copy.deepcopy(export_args))
            onnx_model = onnx.load(onnx_path)
            expected_node_names = ['linear', 'linear#1.end']

            actual_node_names = [node.name for node in onnx_model.graph.node]
            for name in expected_node_names:
                assert name in actual_node_names

            expected_param_names = ['linear.weight', 'linear.bias']
            _, valid_param_set = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
            for name in expected_param_names:
                assert name in valid_param_set

            # Check that gemm still works as expected
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input=torch.randn(1, 3), onnx_export_args=copy.deepcopy(export_args))
            onnx_model = onnx.load(onnx_path)

            actual_node_names = [node.name for node in onnx_model.graph.node]
            assert 'linear' in actual_node_names
            assert 'linear#1' not in actual_node_names

            expected_param_names = ['linear.weight', 'linear.bias']
            _, valid_param_set = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
            for name in expected_param_names:
                assert name in valid_param_set

            self.check_onnx_node_name_uniqueness(onnx_model)

    # @pytest.mark.skipif(Version(torch.__version__) < Version('1.10.0'),
    #                     reason="Need Pytorch1.10.0 https://github.com/pytorch/pytorch/pull/60244")
    @pytest.mark.skip(reason="This test causes sporadic subprocess abort failures due to possible excess process/memory usage.")
    def test_set_node_name_for_large_model(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                self.fc = torch.nn.Linear(1024, 1024*1024)
            def forward(self, input):
                return self.fc(input)

        model = LargeModel()
        dummy_input = torch.randn(1024)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "MyModel.onnx")
            export_args = onnx_utils.OnnxExportApiArgs(opset_version=11)
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input, onnx_export_args=export_args)
            assert os.path.exists(onnx_path)

    def test_set_unique_node_names(self):
        """
        Test that node names are uniquely set.
        """
        class TwoLayerLstmModel(torch.nn.Module):
            """
            Model using torch.nn.LSTM module
            """
            def __init__(self):
                super(TwoLayerLstmModel, self).__init__()
                self.lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=3)

            def forward(self, x, hx=None):
                return self.lstm(x, hx)

        model = TwoLayerLstmModel()
        dummy_input = torch.randn(10, 1, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'MyModel.onnx')

            torch.onnx.export(model, dummy_input, onnx_path)
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)

            onnx_model = onnx.load(onnx_path)
            self.check_onnx_node_name_uniqueness(onnx_model)

    def test_non_leaf_module_names(self):

        """
        Test that node names are uniquely set.
        """
        class Net(torch.nn.Module):
            """
            Model using multiply as functional and module at different depths
            """
            def __init__(self):
                super().__init__()
                self.layer = HierarchicalMultiplyModule()

            def forward(self, x):
                return self.layer(x)

        model = Net()
        dummy_input = torch.randn(10, 1, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'MyModel.onnx')

            torch.onnx.export(model, dummy_input, onnx_path)
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.check_onnx_node_name_uniqueness(onnx_model)

            expected_names = [
                # names compatible with torch 1.9.1 version (should be removed in the future)
                'layer.mul1.mul',
                'layer.mul1.Mul_7',
                'layer.mul2.mul',
                'layer.mul2.Mul_15',
                'layer.Mul_18',

                # names compatible with torch 1.13.1 version
                '/layer/mul1/Mul',
                '/layer/mul2/Mul',
                '/layer/Mul'
            ]
            for node in onnx_model.graph.node:
                assert 'Constant' in node.name or node.name in expected_names

    def test_model_with_input_last_onnx_node(self):
        """
        Test that adversial case when the first input is feed to last node in onnx sub-graph
        """

        roi_model = RoiModel(height=7, width=7, scale=0.25)
        x = torch.rand(1, 1, 6, 6)
        rois = torch.tensor([ [0, -2.0, -2.0, 22.0, 22.0], ])
        dummy_input = (x, rois)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_utils.OnnxSaver.set_node_names(os.path.join(tmp_dir, 'roi.onnx'), roi_model, dummy_input, is_conditional=False,
                                                module_marker_map={},
                                                onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11))
                                                )
            onnx_model = onnx.load(os.path.join(tmp_dir, 'roi.onnx'))
            end_nodes = [ n.name for n in onnx_model.graph.node if 'end' in n.name]
            assert len(end_nodes) == 1

    def test_export_dict_input_output(self):
        """ test dictionary input and output  layers"""


        class Net(torch.nn.Module):
            """
            Model using multiply as functional and module at different depths
            """
            def __init__(self):
                super().__init__()
                self.layer = InputOutputDictModel()

            def forward(self, x):
                return self.layer(x)

        model = Net()

        # Add an empty dictionary as the last element to not treat as named arguments.
        # see torch.onnx.export() API for more details.
        dummy_input = (
            {'a': torch.randn(1, 10, 10, 10),
             'b': torch.randn(1, 10, 10, 10),
             'c': torch.randn(1, 10, 10, 10)
             }, {}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'MyModel.onnx')

            torch.onnx.export(model, dummy_input, onnx_path)
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.check_onnx_node_name_uniqueness(onnx_model)

            for node in onnx_model.graph.node:
                print(node.name)
                assert node.name.startswith('layer')

    def test_kwargs_input_dict_output(self):
        """ test dictionary input as kwargs in an intermediate layer """

        class KwargModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mul = aimet_modules.Multiply()

            def forward(self, a, b, c):
                ab = a * b
                bc = b * c
                ca = self.mul(c, a)

                return {'ab': ab, 'bc': bc, 'ca': ca}

        class Net(torch.nn.Module):
            """
            Model using multiply as functional and module at different depths
            """
            def __init__(self):
                super().__init__()
                self.layer = KwargModel()

            def forward(self, x):
                return self.layer(**x)

        model = Net()

        # Add an empty dictionary as the last element to not treat as named arguments.
        # see torch.onnx.export() API for more details.
        dummy_input = (
            {'a': torch.randn(1, 10, 10, 10),
             'b': torch.randn(1, 10, 10, 10),
             'c': torch.randn(1, 10, 10, 10)
             }, {}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'MyModel.onnx')

            torch.onnx.export(model, dummy_input, onnx_path)
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.check_onnx_node_name_uniqueness(onnx_model)

            for node in onnx_model.graph.node:
                assert node.name.startswith('layer') or node.name.startswith('/layer')

            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def test_naming_for_model_with_deep_graph(self):
        """ test naming works for model have large number of sequential nodes """

        model = models.resnet152(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path= os.path.join(tmp_dir, model.__class__.__name__ + '.onnx')
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input,
                                                is_conditional=False, module_marker_map={})

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.check_onnx_node_names(onnx_model)

            counts = defaultdict(int)
            top_level_nodes = tuple(['conv1', 'bn1', 'relu', 'maxpool', 'avgpool', 'Flatten_', '/Flatten', 'fc'])
            for node in onnx_model.graph.node:
                if node.name.startswith(top_level_nodes):
                    continue
                elif '.' in node.name:
                    layer_name = '.'.join(node.name.split('#')[0].split('.')[:-1])
                    counts[layer_name] += 1
                elif node.name.startswith('/'):
                    layer_name = '.'.join(node.name.split('/')[1:-1])
                    counts[layer_name] += 1

            for name, counts in counts.items():
                if 'downsample' in name:
                    assert counts == 2
                else:
                    print(name, counts)
                    assert counts == 10

    @pytest.mark.parametrize("num_parameters", [1, 8])
    def test_set_node_name_for_multiple_p_relu_model(self, num_parameters):
        model = MultiplePReluModel(num_parameters)
        dummy_input = torch.randn(4, 3, 28, 28)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = f"{tmp_dir}/multiple_p_relu_model.onnx"

            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)
            onnx_model = onnx.load(onnx_path)

        expected_initializer_names = ["act1.weight", "act2.weight", "act3.weight"]
        actual_initializer_names = {x.name for x in onnx_model.graph.initializer}
        for name in expected_initializer_names:
            assert name in actual_initializer_names

        _, valid_param_set = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
        for name in expected_initializer_names:
            assert name in valid_param_set

        self.check_onnx_node_name_uniqueness(onnx_model)

    def test_save_initializer_restored_onnx_graph(self):
        model = MultiplePReluModel()
        dummy_input = torch.randn(4, 3, 28, 28)

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/multiple_p_relu_model.onnx"
            torch.onnx.export(model, dummy_input, original_model_path)

            original_model = onnx.load(original_model_path)
            identity_node_outputs = {x.output[0] for x in original_model.graph.node if x.op_type == "Identity"}
            p_relu_slopes = [x.input[1] for x in original_model.graph.node if x.op_type == "PRelu"]

            # At least one slope is related to the identity node in original ONNX model
            assert any([slope in identity_node_outputs for slope in p_relu_slopes])

            restored_model_path = f"{tmp_dir}/restored_multiple_p_relu_model.onnx"
            save_initializer_restored_onnx_graph(original_model_path, restored_model_path)
            restored_model = onnx.load(restored_model_path)

        # All slope should be initializers in restored ONNX model
        restored_model_initializers = {x.name for x in restored_model.graph.initializer}
        restored_p_relu_slopes = [x.input[1] for x in restored_model.graph.node if x.op_type == "PRelu"]

        assert all([slope in restored_model_initializers for slope in restored_p_relu_slopes])

    @pytest.mark.parametrize("inplace", [True, False])
    def test_restore_onnx_graph_initializers(self, inplace):
        model = MultiplePReluModel()
        dummy_input = torch.randn(4, 3, 28, 28)

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/multiple_p_relu_model.onnx"
            torch.onnx.export(model, dummy_input, original_model_path)

            original_model = onnx.load(original_model_path)
            restored_model = restore_onnx_graph_initializers(original_model, inplace=inplace)

            # Both inplace=True and inplace=False, restored model should have separate initializers
            restored_model_initializers = {x.name for x in restored_model.graph.initializer}
            restored_p_relu_slopes = [x.input[1] for x in restored_model.graph.node if x.op_type == "PRelu"]
            assert all([slope in restored_model_initializers for slope in restored_p_relu_slopes])

            if inplace:
                assert id(original_model) == id(restored_model)
            else:
                # Original model shouldn't be modified if inplace=False
                identity_node_outputs = {x.output[0] for x in original_model.graph.node if x.op_type == "Identity"}
                p_relu_slopes = [x.input[1] for x in original_model.graph.node if x.op_type == "PRelu"]
                assert any([slope in identity_node_outputs for slope in p_relu_slopes])
                assert id(original_model) != id(restored_model)

    def test_get_pytorch_name_from_onnx_name(self):
        """ test get_pytorch_name_from_onnx_name utility """
        # model w/ nested sequntials
        model = NestedSeqModel().eval()
        dummy_input = torch.randn(4, 1, 4, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/model.onnx"
            torch.onnx.export(model, dummy_input, original_model_path)
            original_model = onnx.load(original_model_path)
            for node in original_model.graph.node:
                pytorch_name = get_pytorch_name_from_onnx_name(node.name)
                print(pytorch_name, "-->", node.name)
                assert isinstance(model.get_submodule(pytorch_name), torch.nn.Module)

        # model w/ modules having overlapping names
        model = NestedModelWithOverlappingNames().eval()
        dummy_input = torch.randn(4, 1, 4, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/model.onnx"
            torch.onnx.export(model, dummy_input, original_model_path)
            original_model = onnx.load(original_model_path)
            for node in original_model.graph.node:
                pytorch_name = get_pytorch_name_from_onnx_name(node.name)
                print(pytorch_name, "-->", node.name)
                assert isinstance(model.get_submodule(pytorch_name), torch.nn.Module)

        # model w/ modulelist
        model = ModelWithModuleList().eval()
        dummy_input = torch.randn(4, 1, 4, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/model.onnx"
            torch.onnx.export(model, dummy_input, original_model_path)
            original_model = onnx.load(original_model_path)
            for node in original_model.graph.node:
                pytorch_name = get_pytorch_name_from_onnx_name(node.name)
                print(pytorch_name, "-->", node.name)
                assert isinstance(model.get_submodule(pytorch_name), torch.nn.Module)

    @pytest.mark.parametrize("inplace", [True])
    def test_restore_onnx_graph_reused_initializers(self, inplace):
        """ test to verify that Initializers are added and correponding Identity nodes are removed correctly """
        repetition = 2
        model = ModelWithReusedInitializers(repetition).eval()
        dummy_input = torch.randn(1, 256)
        prepared_model = prepare_model(model)
        assert torch.equal(model(dummy_input), prepared_model(dummy_input))

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_model_path = f"{tmp_dir}/reused_initializers.onnx"
            torch.onnx.export(prepared_model, dummy_input, original_model_path)

            original_model = onnx.load(original_model_path)
            identity_nodes = [node for node in original_model.graph.node if node.op_type == "Identity"]
            assert len(identity_nodes) == 6
            initializers = [ini for ini in original_model.graph.initializer]
            assert len(initializers) == 6

            restored_model = restore_onnx_graph_initializers(original_model, inplace=inplace)
            identity_nodes = [node for node in restored_model.graph.node if node.op_type == "Identity"]
            # There shouldn't be any "Identity" type nodes in the restored model.
            assert len(identity_nodes) == 0
            initializers = [ini for ini in restored_model.graph.initializer]
            # There will be 6 more initializers added for newly added modules.
            assert len(initializers) == 6 * repetition

            # Ensure that the graph is correct
            self.check_onnx_node_name_uniqueness(restored_model)
            onnx.checker.check_model(restored_model)

    def test_get_unique_node_output_name(self):
        # Case 1. If updated_name is occurred first time
        # Return its name without modification
        # node_output_name_counter will have an entry and set to value as 0
        # param_name_to_updated_name will be updated with param_name to unique_node_output_name
        updated_name = '/down_blocks.0/conv1_1/Conv_output_0'
        param_name = '/down_blocks.0/marked_module/conv1/marked_module_1/Conv_output_0'
        base_node_name_counter = {}
        param_name_to_updated_name = {}
        assert (
            OnnxSaver._get_unique_node_output_name(
                updated_name=updated_name,
                param_name=param_name,
                node_output_name_counter=base_node_name_counter,
                param_name_to_updated_name=param_name_to_updated_name,
            )
            == updated_name
        )
        assert base_node_name_counter[updated_name] == 0
        assert (
            param_name_to_updated_name[param_name]
            == updated_name
        )

        # Case 2. If updated_name is not first time
        # Postfix with count will be appended to node output name (e.g., Conv_output_0 -> Conv_output_0_dup1)
        # base_node_name_counter will be incremented (x to x + 1)
        # param_name_to_updated_name will be updated with param_name to unique_node_output_name
        updated_name = '/down_blocks.0/Add_1/Add_output_0'
        param_name = '/down_blocks.0/marked_module/Add/marked_module_1/Add_output_0'
        base_node_name_counter = {updated_name: 0}
        param_name_to_updated_name = {
            '/down_blocks.0/marked_module/Add_1/marked_module/Add_0_output_0': updated_name
        }
        assert (
            OnnxSaver._get_unique_node_output_name(
                updated_name=updated_name,
                param_name=param_name,
                node_output_name_counter=base_node_name_counter,
                param_name_to_updated_name=param_name_to_updated_name,
            )
            == '/down_blocks.0/Add_1/Add_output_0_dup1'
        )
        assert base_node_name_counter[updated_name] == 1
        assert (
            param_name_to_updated_name[param_name] == '/down_blocks.0/Add_1/Add_output_0_dup1'
        )

    def test_node_names(self):
        """ Check if the 'marked_module' string is removed correctly """
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv1(x)


        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.onnx.export(pt_model.eval(),
                              dummy_input,
                              os.path.join(tmp_dir, "model.onnx"),
                              training=torch.onnx.TrainingMode.EVAL,
                              export_params=True,
                              input_names=['input'],
                              output_names=['output'])
            model = onnx.load_model(os.path.join(tmp_dir, "model.onnx"))

            # Add 'marked_module' string in input and output field of onnx.GraphProto object.
            model.graph.input[0].name = model.graph.input[0].name + '/marked_module'
            model.graph.output[0].name = model.graph.input[0].name + '/marked_module'

            # An exception should be raised since the node input/output names are not consistent.
            from onnx.checker import ValidationError
            with pytest.raises(ValidationError):
                onnx.checker.check_model(model)

            # Remove the 'marked_module' string
            OnnxSaver._remove_marked_module_string_from_node_inp_out_names(model.graph)

            # model should be consistent.
            onnx.checker.check_model(model)
