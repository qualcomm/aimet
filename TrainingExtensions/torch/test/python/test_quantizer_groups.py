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
import torch

from aimet_common.defs import QuantizationDataType
from aimet_common.amp.quantizer_groups import reformat_supported_kernels
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from aimet_torch.examples.test_models import SingleResidual, ConcatModel
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.amp.quantizer_groups import find_quantizer_group, find_op_groups, find_supported_candidates, \
    QuantizerGroup
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch import onnx_utils
from aimet_torch.v1.nn.modules import custom
from torchvision.models import mobilenet_v3_large as mobilenetv3
from models import test_models


class TestQuantizerGroups:

    def test_simple_mnist_network(self):
        model = test_models.SmallMnist().to(device='cpu')
        input_shapes = (1, 1, 10, 10)
        inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes, torch.device('cpu'))
        connected_graph = ConnectedGraph(model, inp_tensor_list)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 11
        # Check if there is one parent and one child
        for parent, child in op_groups.items():
            assert not isinstance(parent, tuple)
            assert len(child) == 1
        assert 'input_ops' in op_groups
        assert op_groups['input_ops'] == ['SmallMnist.conv1']
        assert op_groups['output_ops'] == ['SmallMnist.log_softmax']
        assert 'SmallMnist.conv1' in op_groups
        assert 'SmallMnist.relu2' in op_groups['SmallMnist.conv2_drop']

    def test_model_with_one_split(self):
        model = test_models.ModelWithOneSplit().to(device='cpu')
        input_shapes = (1, 1, 10, 10)
        inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes, torch.device('cpu'))
        connected_graph = ConnectedGraph(model, inp_tensor_list)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 3
        assert len(op_groups['ModelWithOneSplit.conv1']) == 2

    def test_single_residual_network(self):
        model = SingleResidual()
        input_shapes = (1, 3, 32, 32)
        device = utils.get_device(model)
        inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes, device)
        connected_graph = ConnectedGraph(model, inp_tensor_list)
        op_groups = find_op_groups(connected_graph)
        assert len(op_groups) == 14
        count = 0

        add_op = [op for op in connected_graph.get_all_ops().values() if op.type == 'Add'][0]
        for _, child in op_groups.items():
            if tuple(child)[0] == add_op.name:
                count += 1
        assert count == 2
        assert op_groups['SingleResidual.relu1'] == ['SingleResidual.conv2', 'SingleResidual.conv4']
        assert op_groups['SingleResidual.conv4'] == ['SingleResidual.ada']

    def test_concat_model(self):
        model = ConcatModel()
        inp_shape_1 = (1, 3, 8, 8)
        inp_shape_2 = (1, 3, 8, 8)
        inp_shape_3 = (1, 3, 8, 8)
        device = utils.get_device(model)
        inp_tensor_list = utils.create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2, inp_shape_3], device)
        conn_graph = ConnectedGraph(model, inp_tensor_list)
        op_groups = find_op_groups(conn_graph)
        assert len(op_groups) == 6
        count = 0
        concat_op = [op for op in conn_graph.get_all_ops().values() if op.type == 'Concat'][0]
        for _, child in op_groups.items():
            if tuple(child)[0] == concat_op.name:
                count += 1

        assert count == 3

    def test_find_quantizer_groups(self):
        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        # Temporary hack before dropout is disabled in default_config.json
        sim.model.conv2_drop.output_quantizers[0].enabled = False
        sim.model.dropout.output_quantizers[0].enabled = False
        _, quantizer_groups = find_quantizer_group(sim)

        assert len(quantizer_groups) == 8
        assert len(quantizer_groups[0].input_quantizers) == 1
        assert quantizer_groups[1].output_quantizers == ('relu1',)
        assert quantizer_groups[3].parameter_quantizers == ('fc1',)
        assert len(quantizer_groups[-1].output_quantizers) == 1

    def test_multiple_inputs_qg(self):
        model = test_models.ModelWithMatMul3()
        dummy_input = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input=(dummy_input, dummy_input))
        _, quantizer_groups = find_quantizer_group(sim)
        assert(quantizer_groups[0].get_input_quantizer_modules() == ("matmul_1", ))
        assert (quantizer_groups[0].input_quantizers == (
        'matmul_1_input_quantizer_idx_0', 'matmul_1_input_quantizer_idx_1'))

    def test_quantizer_groups_for_model_with_two_inputs_and_two_outputs(self):
        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        model = test_models.ModelWithTwoInputsTwoOutputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 6
        assert len(quantizer_groups[0].input_quantizers) == 1
        assert len(quantizer_groups[1].input_quantizers) == 1
        assert len(quantizer_groups[-1].output_quantizers) == 1
        assert len(quantizer_groups[-2].output_quantizers) == 1

    def test_quantizer_groups_inverted_residuals(self):
        model = mobilenetv3()
        dummy_input = torch.randn(1, 3, 224, 224)
        fold_all_batch_norms(model, (1, 3, 224, 224))
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          "./model_single_residual.onnx",  # where to save the model (can be a file or file-like object)
                          training=torch.onnx.TrainingMode.EVAL,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert quantizer_groups[4].to_list() == [('output', 'features.1.block.1.0')]
        assert quantizer_groups[15].to_list() == [('output', 'features.4.block.2.avgpool'), ('weight', 'features.4.block.2.fc1')]
        assert len(quantizer_groups[-1].output_quantizers) == 1
        assert len(quantizer_groups) == 122

    def test_reformat_supported_kernels_1(self):
        supported_kernels = {'defaults': [{'activation': {'bitwidth': 8, 'dtype': QuantizationDataType.int},
                                           'param': {'bitwidth': 8, 'dtype': QuantizationDataType.int}},
                                          {'activation': {'bitwidth': 16, 'dtype': QuantizationDataType.float},
                                            'param': {'bitwidth': 16, 'dtype': QuantizationDataType.float}}],
                            'Conv': [{'activation': {'bitwidth': 16, 'dtype': QuantizationDataType.float},
                                     'param': {'bitwidth': 16, 'dtype': QuantizationDataType.float}}]}

        formated_supported_kernels = reformat_supported_kernels(supported_kernels)
        assert len(formated_supported_kernels['defaults']) == 2
        assert len(formated_supported_kernels['Conv']) == 1

        def_candidates = formated_supported_kernels['defaults']

        candidate = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
        assert candidate in def_candidates

        candidate = ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        assert candidate in def_candidates

        conv_candidates = formated_supported_kernels['Conv']
        candidate = ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        assert candidate in conv_candidates

    def test_reformat_supported_kernels_2(self):
        supported_kernels = {}
        formated_supported_kernels = reformat_supported_kernels(supported_kernels)
        assert not formated_supported_kernels

    def test_find_supported_candidates_1(self):
        """
        Test to verify use_all_amp_candidates option. When set to true, the return values supported_candidates in
        quantizers_with_supported_candidates and max_candidate_options should be equal to amp_candidates passed in.
        """

        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        module_name_to_module_dict, quantizer_groups = find_quantizer_group(sim)

        amp_candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
        ]

        supported_kernels = {} # does not matter for this test

        quantizers_with_supported_candidates, max_candidate_options = find_supported_candidates(quantizer_groups,
                                                                                                amp_candidates,
                                                                                                supported_kernels,
                                                                                                module_name_to_module_dict,
                                                                                                use_all_amp_candidates=True)
        for candidate_list in quantizers_with_supported_candidates.values():
            assert amp_candidates == candidate_list

        assert amp_candidates == max_candidate_options

    def test_find_supported_candidates_2(self):
        """
        Test to verify the output when supported_kernels is empty
        """

        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        module_name_to_module_dict, quantizer_groups = find_quantizer_group(sim)

        amp_candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
        ]

        supported_kernels = {} # does not matter for this test

        quantizers_with_supported_candidates, max_candidate_options = find_supported_candidates(quantizer_groups,
                                                                                                amp_candidates,
                                                                                                supported_kernels,
                                                                                                module_name_to_module_dict,
                                                                                                use_all_amp_candidates=False)
        for candidate_list in quantizers_with_supported_candidates.values():
            assert amp_candidates == candidate_list

        assert amp_candidates == max_candidate_options


    def test_find_supported_candidates_3(self):
        """
        Test to verify test asserts when "defaults" is not present in supported_kernels
        """

        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        module_name_to_module_dict, quantizer_groups = find_quantizer_group(sim)

        amp_candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
        ]

        candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        ]

        supported_kernels = {'Conv': candidates}

        with pytest.raises(ValueError):
            find_supported_candidates(quantizer_groups,
                                      amp_candidates,
                                      supported_kernels,
                                      module_name_to_module_dict,
                                      use_all_amp_candidates=False)

    def test_find_supported_candidates_4(self):
        """
        Test to verify that find_supported_candidates asserts if no combination of candidates can be computed with the
        given combination of supported_kernels and quantizer_groups
        """

        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        module_name_to_module_dict, quantizer_groups = find_quantizer_group(sim)

        amp_candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
        ]

        candidates = [
            ((4, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        ]

        supported_kernels = {'defaults': candidates,
                             'Conv': candidates}

        with pytest.raises(ValueError):
            find_supported_candidates(quantizer_groups,
                                      amp_candidates,
                                      supported_kernels,
                                      module_name_to_module_dict,
                                      use_all_amp_candidates=False)

    def test_find_supported_candidates_5(self):
        """
        Test to verify that find_supported_candidates returns correct combination of candidates
        """

        model = test_models.SmallMnist()
        dummy_input = torch.randn(1, 1, 10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        module_name_to_module_dict, quantizer_groups = find_quantizer_group(sim)

        amp_candidates = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        ]

        candidates_default = [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        ]

        candidates_conv = [
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        ]

        supported_kernels = {'defaults': candidates_default,
                             'Conv': candidates_conv}

        quantizer_groups_with_supported_candidates, max_candidate_options = find_supported_candidates(quantizer_groups,
                                                                                                amp_candidates,
                                                                                                supported_kernels,
                                                                                                module_name_to_module_dict,
                                                                                                use_all_amp_candidates=False)

        assert ((16, QuantizationDataType.float), (16, QuantizationDataType.float)) in max_candidate_options

        for quantizer_group, candidates in quantizer_groups_with_supported_candidates.items():
            supported_kernel_ops = quantizer_group.supported_kernel_ops
            onnx_types = []
            for op in supported_kernel_ops:
                onnx_types.append(
                    onnx_utils.map_torch_types_to_onnx.get(type(module_name_to_module_dict[op]._module_to_wrap)))

            # verify to make sure the candidates returned is always part of amp_candidates and they are part of
            # either "Conv" or "Defaults"
            for c in candidates:
                assert c in amp_candidates

            if ['Conv'] in onnx_types:
                for c in candidates:
                    assert c in candidates_conv
            else:
                for c in candidates:
                    assert c in candidates_default

    def test_resnet18_quantizer_groups(self):
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        # NOTE: resnet18 has several relu layers reused which are not addressed in AMP directly.
        # Please do not use resnet18 without going through Model Preparer (Pro)
        dummy_input = torch.randn(1, 3, 224, 224)
        fold_all_batch_norms(model, (1, 3, 224, 224))

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 23
        assert quantizer_groups[21].output_quantizers == ('avgpool', )
        assert quantizer_groups[21].parameter_quantizers == ('fc', )
        assert quantizer_groups[22].output_quantizers == ('fc', )


    def test_model_with_flatten(self):
        model = test_models.ModelWithFlatten()
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 4
        assert quantizer_groups[2].output_quantizers == ('relu_1',)
        assert quantizer_groups[2].parameter_quantizers == ('fc_1',)
        assert quantizer_groups[3].output_quantizers == ('fc_1',)

    def test_quantizer_groups_with_diff_combinations(self):
        # tests split, two consecutive data movement ops, data movement op in the end of a branch
        model = test_models.ModelWithSeveralDataMovementOps()
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 5

    def test_supported_kernel_ops(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.relu1 = torch.nn.ReLU()
                self.relu2 = torch.nn.ReLU()
                self.add = custom.Add()
                self.relu3 = torch.nn.ReLU()
                self.relu4 = torch.nn.ReLU()

            def forward(self, inp, inp2):
                x1 = self.relu1(inp)
                x2 = self.relu2(inp2)
                x = self.add(x1, x2)
                x1 = self.relu3(x)
                x2 = self.relu4(x)
                return x1, x2

        model = Model()
        dummy_input = (torch.randn(1, 3), torch.randn(1, 3))
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 7

        expected_groups = [QuantizerGroup(input_quantizers=('relu1_input_quantizer_idx_0',),
                                          supported_kernel_ops=('relu1',)),
                           QuantizerGroup(input_quantizers=('relu2_input_quantizer_idx_0',),
                                          supported_kernel_ops=('relu2',)),
                           QuantizerGroup(output_quantizers=('relu1',),
                                          supported_kernel_ops=('add',)),
                           QuantizerGroup(output_quantizers=('relu2',),
                                          supported_kernel_ops=('add',)),
                           QuantizerGroup(output_quantizers=('add',),
                                          supported_kernel_ops=('relu3', 'relu4')),
                           QuantizerGroup(output_quantizers=('relu3',),
                                          supported_kernel_ops=tuple()),
                           QuantizerGroup(output_quantizers=('relu4',),
                                          supported_kernel_ops=tuple())]
        for group in expected_groups:
            assert group in quantizer_groups

    def test_find_supported_kernels(self):
        quantizer_groups_to_test = [QuantizerGroup(input_quantizers=('inp1_input_quantizer_idx_0', 'inp2_input_quantizer_idx_0'),
                                                   supported_kernel_ops=tuple()),
                                    QuantizerGroup(input_quantizers=('inp1_input_quantizer_idx_0', 'inp2_input_quantizer_idx_0'),
                                                   supported_kernel_ops=('op1',)),
                                    QuantizerGroup(input_quantizers=('inp1_input_quantizer_idx_0', 'inp2_input_quantizer_idx_0'),
                                                   supported_kernel_ops=('op1', 'op2')),
                                    QuantizerGroup(output_quantizers=('inp1', 'inp2'),
                                                   supported_kernel_ops=('op1', 'op2')),
                                    QuantizerGroup(parameter_quantizers=('inp1', 'inp2'),
                                                   supported_kernel_ops=('op1', 'op2'))]
        amp_candidates = [((2, QuantizationDataType.int), (2, QuantizationDataType.int)),
                          ((3, QuantizationDataType.int), (3, QuantizationDataType.int)),
                          ((4, QuantizationDataType.int), (4, QuantizationDataType.int)),
                          ((5, QuantizationDataType.int), (5, QuantizationDataType.int))]
        supported_kernels = {
            'defaults': [((2, QuantizationDataType.int), (2, QuantizationDataType.int)),
                         ((3, QuantizationDataType.int), (3, QuantizationDataType.int)),
                         ((4, QuantizationDataType.int), (4, QuantizationDataType.int))],
            'Conv': [((2, QuantizationDataType.int), (2, QuantizationDataType.int)),
                     ((3, QuantizationDataType.int), (3, QuantizationDataType.int))],
            'Relu': [((2, QuantizationDataType.int), (2, QuantizationDataType.int))],
        }

        class MockWrapper:
            def __init__(self, module_to_wrap):
                self._module_to_wrap = module_to_wrap

        module_name_to_module_dict = {
            'op1': MockWrapper(torch.nn.Conv2d(3, 8, (2, 2))),
            'op2': MockWrapper(torch.nn.ReLU())
        }

        supported_kernel_dict, _ = find_supported_candidates(quantizer_groups_to_test,
                                                             amp_candidates,
                                                             supported_kernels,
                                                             module_name_to_module_dict,
                                                             False)

        assert set(supported_kernel_dict[quantizer_groups_to_test[0]]) == {
            ((2, QuantizationDataType.int), (2, QuantizationDataType.int)),
            ((3, QuantizationDataType.int), (3, QuantizationDataType.int)),
            ((4, QuantizationDataType.int), (4, QuantizationDataType.int))}

        assert set(supported_kernel_dict[quantizer_groups_to_test[1]]) == {
            ((2, QuantizationDataType.int), (2, QuantizationDataType.int)),
            ((3, QuantizationDataType.int), (3, QuantizationDataType.int))}

        assert set(supported_kernel_dict[quantizer_groups_to_test[2]]) == {
            ((2, QuantizationDataType.int), (2, QuantizationDataType.int))}

        assert set(supported_kernel_dict[quantizer_groups_to_test[3]]) == {
            ((2, QuantizationDataType.int), (2, QuantizationDataType.int))}

        assert set(supported_kernel_dict[quantizer_groups_to_test[4]]) == {
            ((2, QuantizationDataType.int), (2, QuantizationDataType.int))}
