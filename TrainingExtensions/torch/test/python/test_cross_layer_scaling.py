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

import copy
import numpy as np
from itertools import chain
import pytest
from contextlib import contextmanager
import torch
from torchvision import models

from aimet_common.utils import AimetLogger
from aimet_torch import batch_norm_fold
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
import aimet_torch.cross_layer_equalization as cle
from aimet_torch.cross_layer_equalization import CrossLayerScaling, HighBiasFold, equalize_model,\
    ClsSetInfo, GraphSearchUtils
from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch.utils import get_layer_name
from models.mobilenet import MockMobileNetV2, MockMobileNetV1
from models.test_models import Float32AndInt64InputModel

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class TwoInputsModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(TwoInputsModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2)
        self.ada = torch.nn.AdaptiveAvgPool2d(18)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = torch.nn.Linear(1600, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs[1])
        x2 = self.bn2(x2)
        x2 = self.conv3(x2)
        x2 = self.ada(x2)
        x = x1 + x2
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 20, 3)
        self.relu2 = torch.nn.PReLU()
        self.conv3 = torch.nn.Conv2d(20, 20, 3)
        self.conv4 = torch.nn.Conv2d(20, 20, 3)
        # 1D conv layers
        self.conv5 = torch.nn.Conv1d(20, 20, 3)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.relu3 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv1d(20, 20, 3)
        self.relu4 = torch.nn.ReLU()
        # Transposed conv layers
        self.conv7 = torch.nn.ConvTranspose1d(20, 20, 3)
        self.relu5 = torch.nn.ReLU()
        self.conv8 = torch.nn.ConvTranspose1d(20, 20, 3)
        self.conv8a = torch.nn.Conv1d(20, 20, 3)
        self.relu5a = torch.nn.ReLU()
        # Depthwise separable conv 1D layers
        self.relu6 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv1d(20, 20, 3, groups=20)
        self.relu7 = torch.nn.ReLU()
        self.conv10 = torch.nn.Conv1d(20, 20, 1)
        self.relu8 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(5040, 10)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.relu1(x)
        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)
        # Case where BN can fold into an immediate downstream conv
        x = self.conv3(x)
        # No fold if there is a split between conv and BN
        x = self.conv4(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.relu3(x)
        x = self.conv6(x)
        x = self.relu4(x)
        x = self.conv7(x)
        x = self.relu5(x)
        x = self.conv8(x)
        x = self.conv8a(x)
        x = self.relu5a(x)
        x = self.conv9(x)
        x = self.relu7(x)
        x = self.conv10(x)
        x = self.relu8(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super(TransposedConvModel, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

@contextmanager
def _use_python_impl(flag: bool):
    orig_flag = cle.USE_PYTHON_IMPL
    try:
        cle.USE_PYTHON_IMPL = flag
        yield
    finally:
        cle.USE_PYTHON_IMPL = orig_flag


@pytest.fixture(params=[True, False])
def use_python_impl(request):
    param: bool = request.param

    with _use_python_impl(param):
        yield


class TestTrainingExtensionsCrossLayerScaling:

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_verify_cross_layer_scaling(self, use_python_impl, device):
        # Get trained MNIST model
        torch.manual_seed(10)
        model = MyModel().eval().to(device)
        random_input = torch.rand((2, 10, 24, 24)).to(device)
        baseline_output = model(random_input).detach().cpu().numpy()

        params_before = [param.clone() for param in chain(model.conv1.parameters(), model.conv1.parameters())]
        CrossLayerScaling.scale_cls_set_with_conv_layers((model.conv1, model.conv2))
        params_after = [param.clone() for param in chain(model.conv1.parameters(), model.conv1.parameters())]

        # Ensure that parameters before and after scaling are not bit-exact
        assert not all(torch.equal(w_prev, w_after) for w_prev, w_after in zip(params_before, params_after))

        output_after_scaling = model(random_input).detach().cpu().numpy()
        range_conv1_after_scaling = np.amax(np.abs(model.conv1.weight.detach().cpu().numpy()), axis=(1, 2, 3))
        range_conv2_after_scaling = np.amax(np.abs(model.conv2.weight.detach().cpu().numpy()), axis=(0, 2, 3))

        assert (np.allclose(range_conv1_after_scaling, range_conv2_after_scaling))
        assert (np.allclose(baseline_output, output_after_scaling, rtol=1.e-2))

    def test_top_level_api(self, use_python_impl):
        torch.manual_seed(10)
        model = MyModel().eval()
        input_shape = (2, 10, 24, 24)
        random_input = torch.rand((2, 10, 24, 24))
        baseline_output = model(random_input).detach().numpy()

        folded_pairs = batch_norm_fold.fold_all_batch_norms(model, [input_shape])

        cls_sets = CrossLayerScaling.scale_model(model, input_shape, random_input)
        # cls_sets is empty!
        pass

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_verify_cross_layer_for_multiple_pairs(self, use_python_impl, device):
        # Get trained MNIST model
        model = MyModel().eval().to(device)
        # Call API
        consecutive_layer_list = [(model.conv1, model.conv2),
                                  (model.conv3, model.conv4)]
        w1 = model.conv1.weight.clone()
        w2 = model.conv2.weight.clone()
        w3 = model.conv3.weight.clone()

        CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

        # check if weights are updating
        assert not torch.equal(model.conv1.weight, w1)
        assert not torch.equal(model.conv2.weight, w2)
        assert not torch.equal(model.conv3.weight, w3)

    def test_verify_cross_layer_scaling_depthwise_separable_layer_mobilnet(self, use_python_impl):
        torch.manual_seed(10)

        model = MockMobileNetV1().eval()
        model = model.to(torch.device('cpu'))
        model.model[0][0].bias = torch.nn.Parameter(torch.rand(model.model[0][0].weight.data.size()[0]))
        model.model[1][0].bias = torch.nn.Parameter(torch.rand(model.model[1][0].weight.data.size()[0]))
        model.model[1][3].bias = torch.nn.Parameter(torch.rand(model.model[1][3].weight.data.size()[0]))
        model.model[2][0].bias = torch.nn.Parameter(torch.rand(model.model[2][0].weight.data.size()[0]))
        model.model[2][3].bias = torch.nn.Parameter(torch.rand(model.model[2][3].weight.data.size()[0]))

        random_input = torch.rand(1, 3, 224, 224)
        baseline_output = model(random_input).detach().numpy()

        consecutive_layer_list = [(model.model[0][0], model.model[1][0], model.model[1][3]),
                                  (model.model[1][3], model.model[2][0], model.model[2][3])]

        for consecutive_layer in consecutive_layer_list:
            CrossLayerScaling.scale_cls_set_with_depthwise_layers(consecutive_layer)
            r1 = np.amax(np.abs(consecutive_layer[0].weight.detach().cpu().numpy()), axis=(1, 2, 3))
            r2 = np.amax(np.abs(consecutive_layer[1].weight.detach().cpu().numpy()), axis=(1, 2, 3))
            r3 = np.amax(np.abs(consecutive_layer[2].weight.detach().cpu().numpy()), axis=(0, 2, 3))
            assert (np.allclose(r1, r2))
            assert (np.allclose(r2, r3))

        output_after_scaling = model(random_input).detach().numpy()

        assert (np.allclose(baseline_output, output_after_scaling, rtol=1.e-2))

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_verify_cross_layer_scaling_depthwise_separable_layer_multiple_triplets(self, use_python_impl, device):
        torch.manual_seed(10)

        model = MockMobileNetV1().eval().to(device)
        consecutive_layer_list = [(model.model[0][0], model.model[1][0], model.model[1][3]),
                                  (model.model[1][3], model.model[2][0], model.model[2][3])]
        w1 = model.model[0][0].weight.clone()
        w2 = model.model[1][3].weight.clone()
        w3 = model.model[2][3].weight.clone()

        CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

        assert not torch.equal(model.model[0][0].weight, w1)
        assert not torch.equal(model.model[1][3].weight, w2)
        assert not torch.equal(model.model[2][3].weight, w3)

    def test_find_layer_groups_to_scale_for_network_with_residuals(self, use_python_impl):
        torch.manual_seed(10)
        model = MockMobileNetV2()
        model.eval()

        fold_all_batch_norms(model, (1, 3, 224, 224))
        input_shape = (1, 3, 224, 224)
        random_input = torch.rand(*input_shape)
        graph_search = GraphSearchUtils(model, (1, 3, 224, 224),random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()
        assert 4 == len(layer_groups)
        assert [model.features[3].conv[0], model.features[3].conv[3], model.features[3].conv[6]] in layer_groups
        assert [model.features[4].conv[0], model.features[4].conv[3], model.features[4].conv[6]] in layer_groups
        assert [model.features[5].conv[0], model.features[5].conv[3], model.features[5].conv[6],
                model.features[6][0]] in layer_groups

        for layer_group in layer_groups:
            print("Group ------- ")
            for module in layer_group:
                print("   " + get_layer_name(model, module))

    def test_find_layer_groups_to_scale(self):
        """
        test conv+depthwise+conv combination
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3), groups=20),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 40, (3, 3))
        )
        model.eval()
        random_input = torch.rand(1, 10, 24, 24)
        graph_search = GraphSearchUtils(model, (1, 10, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 1 == len(cls_sets)
        assert (model[0], model[2], model[4]) in cls_sets

    def test_find_layer_groups_to_scale_2(self):
        """
        verify conv+conv cls sets
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3))
        )
        model.eval()
        input_shape = (1, 10, 24, 24)
        random_input = torch.rand(*input_shape)
        graph_search = GraphSearchUtils(model, (1, 10, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 2 == len(cls_sets)
        assert model[0] in cls_sets[0]
        assert model[2] in cls_sets[0]
        assert model[2] in cls_sets[1]
        assert model[4] in cls_sets[1]

    def test_find_layer_groups_to_scale_3(self):
        """
        verify depthwiseConv2D+conv cls sets
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(20, 20, (3, 3), groups=20),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3))
        )
        model.eval()
        random_input = torch.rand((1, 20, 24, 24))
        graph_search = GraphSearchUtils(model, (1, 20, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 2 == len(cls_sets)
        assert model[0] in cls_sets[0]
        assert model[2] in cls_sets[0]
        assert model[2] in cls_sets[1]
        assert model[4] in cls_sets[1]

    def test_find_layer_groups_to_scale_4(self):
        """
        verify depthwiseConv2D+conv cls sets
        - test ignore depthwise+depthwise combo which is unsupported
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(20, 20, (3, 3), groups=20),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3), groups=20),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (3, 3))
        )
        model.eval()
        random_input = torch.rand((1, 20, 24, 24))
        graph_search = GraphSearchUtils(model, (1, 20, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 1 == len(cls_sets)
        assert model[2] in cls_sets[0]
        assert model[4] in cls_sets[0]

    def test_find_layer_groups_to_scale_5(self):
        """
        Test invalid case
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, (3, 3), groups=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, (3, 3), groups=4),
            torch.nn.ReLU(),
        )
        model.eval()
        random_input = torch.rand((1, 2, 24, 24))
        graph_search = GraphSearchUtils(model, (1, 2, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 0 == len(cls_sets)

    def test_find_layer_groups_to_scale_6(self):
        """
        Test invalid case
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2, 4, (3, 3), groups=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, (3, 3), groups=4),
            torch.nn.ReLU()
        )
        model.eval()
        random_input = torch.rand((1, 2, 24, 24))
        graph_search = GraphSearchUtils(model, (1, 2, 24, 24), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        assert 0 == len(cls_sets)

    def test_find_cls_sets_vgg16(self):

        torch.manual_seed(10)
        model = models.vgg16()
        print(model)
        model.eval()
        random_input = torch.rand((1, 3, 224, 224))
        graph_search = GraphSearchUtils(model, (1, 3, 224, 224), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()
        assert 5 == len(layer_groups)
        assert [model.features[0], model.features[2]] in layer_groups
        assert [model.features[5], model.features[7]] in layer_groups
        assert [model.features[10], model.features[12], model.features[14]] in layer_groups
        assert [model.features[17], model.features[19], model.features[21]] in layer_groups
        assert [model.features[24], model.features[26], model.features[28]] in layer_groups

        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        for cls_set in cls_sets:
            print(cls_set)

        assert 8 == len(cls_sets)
        assert (model.features[0], model.features[2]) in cls_sets
        assert (model.features[5], model.features[7]) in cls_sets
        assert (model.features[10], model.features[12]) in cls_sets
        assert (model.features[12], model.features[14]) in cls_sets
        assert (model.features[17], model.features[19]) in cls_sets
        assert (model.features[19], model.features[21]) in cls_sets
        assert (model.features[24], model.features[26]) in cls_sets
        assert (model.features[26], model.features[28]) in cls_sets

        result = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)
        print(result)

    def test_find_cls_sets_mobilenetv1(self):

        torch.manual_seed(10)

        model = MockMobileNetV1()
        model.eval()

        fold_all_batch_norms(model, (1, 3, 224, 224))
        random_input = torch.rand(1, 3, 224, 224)
        graph_search = GraphSearchUtils(model, (1, 3, 224, 224), random_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        assert 1 == len(layer_groups)
        assert ([model.model[0][0],
                       model.model[1][0],
                       model.model[1][3],
                       model.model[2][0],
                       model.model[2][3],
                       model.model[3][0],
                       model.model[3][3],
                       model.model[4][0],
                       model.model[4][3],
                       model.model[5][0],
                       model.model[5][3],
                       model.model[6][0],
                       model.model[6][3],
                       model.model[7][0],
                       model.model[7][3],
                       model.model[8][0],
                       model.model[8][3],
                       ], layer_groups)

        layer_pairs = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_groups[0])
        for layer_tuple in layer_pairs:
            print(layer_tuple)

    def test_auto_mobilenetv1(self, use_python_impl):
        torch.manual_seed(10)
        model = MockMobileNetV1()
        model.eval()

        # BN fold
        fold_all_batch_norms(model, (1, 3, 224, 224))
        random_input = torch.rand((1, 3, 224, 224))
        scale_factors = CrossLayerScaling.scale_model(model, (1, 3, 224, 224), random_input)
        assert 8 == len(scale_factors)

    def test_auto_cls_custom_model(self, use_python_impl):
        torch.manual_seed(10)
        model = MyModel()
        model.eval()
        random_input = torch.rand(2, 10, 24, 24)
        output_before_scale = model(random_input)

        # BN fold
        fold_all_batch_norms(model, (2, 10, 24, 24))

        scale_factors = CrossLayerScaling.scale_model(model, (2, 10, 24, 24), random_input)
        assert 8 == len(scale_factors)
        assert scale_factors[0].cls_pair_info_list[0].relu_activation_between_layers
        assert scale_factors[1].cls_pair_info_list[0].relu_activation_between_layers
        assert not scale_factors[2].cls_pair_info_list[0].relu_activation_between_layers

        output_after_scale = model(random_input)
        assert torch.allclose(output_before_scale, output_after_scale)

    @pytest.mark.cuda
    def test_auto_cls_custom_model_multi_gpu(self, use_python_impl):

        torch.manual_seed(10)
        model = MyModel()
        model.eval()

        model = torch.nn.DataParallel(model, device_ids=[0])
        random_input = torch.rand(2, 10, 24, 24)
        output_before_scale = model(random_input)

        # BN fold
        fold_all_batch_norms(model, (2, 10, 24, 24))

        scale_factors = CrossLayerScaling.scale_model(model, (2, 10, 24, 24), random_input)

        output_after_scale = model(random_input)
        assert torch.allclose(output_before_scale, output_after_scale, rtol=1.e-2)

    def test_auto_cle_custom_model(self, use_python_impl):

        torch.manual_seed(10)
        model = MyModel()
        model.eval()
        random_input = torch.rand(2, 10, 24, 24)
        output_before_equalize = model(random_input)

        equalize_model(model, (2, 10, 24, 24), dummy_input=random_input)

        output_after_equalize = model(random_input)
        assert torch.allclose(output_before_equalize, output_after_equalize)

    @pytest.mark.cuda
    def test_auto_cle_custom_model_multi_gpu(self, use_python_impl):

        torch.manual_seed(10)
        model = MyModel()
        model.eval()
        model = torch.nn.DataParallel(model, device_ids=[0])
        input_shapes = (2, 10, 24, 24)
        random_input = torch.rand(*input_shapes)
        random_input = torch.rand(2, 10, 24, 24)
        output_before_equalize = model(random_input)

        equalize_model(model, input_shapes, dummy_input=random_input)

        output_after_equalize = model(random_input)
        assert torch.allclose(output_before_equalize, output_after_equalize, rtol=1.e-2)

    def test_auto_cle_two_inputs_model(self, use_python_impl):

        model = TwoInputsModel().eval()
        model_copy = copy.deepcopy(model)
        inp_shapes = [(1, 3, 32, 32), (1, 3, 20, 20)]
        model_input_list = create_rand_tensors_given_shapes(inp_shapes, get_device(model))

        output_before_equalize = model(*model_input_list)
        equalize_model(model, inp_shapes)
        output_after_equalize = model(*model_input_list)
        assert torch.allclose(output_before_equalize, output_after_equalize)

        output_before_equalize = model_copy(*model_input_list)
        equalize_model(model_copy, inp_shapes, dummy_input=model_input_list)
        output_after_equalize = model_copy(*model_input_list)
        assert torch.allclose(output_before_equalize, output_after_equalize)

    def test_auto_transposed_conv2d_model(self, use_python_impl):

        torch.manual_seed(10)
        model = TransposedConvModel()
        model.eval()
        random_input = torch.rand((10, 10, 4, 4))

        baseline_output = model(random_input).detach().numpy()
        scale_factors = CrossLayerScaling.scale_model(model, (10, 10, 4, 4), random_input)

        output_after_scaling = model(random_input).detach().numpy()
        assert np.allclose(baseline_output, output_after_scaling, rtol=1.e-2)
        assert 10 == len(scale_factors[0].cls_pair_info_list[0].scale_factor)

    def test_auto_depthwise_transposed_conv_model(self, use_python_impl):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(5, 10, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(10, 10, 3, groups=10),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 24, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(24, 32, 3),
        )
        model.eval()
        random_input = torch.rand((1, 5, 32, 32))

        baseline_output = model(random_input).detach().numpy()
        scale_factors = CrossLayerScaling.scale_model(model, (1, 5, 32, 32), random_input)

        output_after_scaling = model(random_input).detach().numpy()
        output_diff = abs(baseline_output - output_after_scaling)

        assert np.all(max(output_diff) < 1e-6)
        assert 2 == len(scale_factors)
        assert 2 == len(scale_factors[0].cls_pair_info_list)

    def test_cle_for_float32_and_int64_input_model(self, use_python_impl):
        model = Float32AndInt64InputModel().to(torch.device('cpu'))
        model.eval()

        inp_shapes = [(1, 3, 32, 32), (3, 20, 20), (3, 20, 20)]

        a = torch.rand(1, 3, 32, 32)
        b = torch.randint(0, 10, (3, 20, 20))
        c = torch.randint(0, 10, (3, 20, 20))
        input_tuple = (a, b, c)

        output_before_cle = model(*input_tuple)

        # equalize the model
        equalize_model(model, input_shapes=inp_shapes, dummy_input=input_tuple)

        output_after_cle = model(*input_tuple)

        assert torch.allclose(output_before_cle, output_after_cle, rtol=1.e-2)

class TestTrainingExtensionsCrossLayerScalingPythonOnly:

    @pytest.mark.cuda
    def test_cle_using_python_impl(self):
        """ Compare MO and python implementation for CLE """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            random_input = torch.rand(2, 10, 24, 24).cuda()
            model = MyModel().eval().cuda()
            model_copy = copy.deepcopy(model).eval()
            # original outputs
            output = model(random_input)

            # equalize using MO
            cle.USE_PYTHON_IMPL = False
            equalize_model(model, (2, 10, 24, 24), dummy_input=random_input)
            output_using_mo = model(random_input)

            # equalize using python
            cle.USE_PYTHON_IMPL = True
            equalize_model(model_copy, (2, 10, 24, 24), dummy_input=random_input)
            output_using_python = model_copy(random_input)

            assert torch.allclose(output_using_mo, output_using_python)
            assert torch.allclose(output, output_using_mo)
            assert torch.allclose(output, output_using_python)
        finally:
            cle.USE_PYTHON_IMPL = flag

    @pytest.mark.cuda
    def test_scale_cls_set_with_conv_layers_using_python_impl(self):
        """ Compare scale_cls_set_with_conv_layers API """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = MyModel().cuda().eval()
            model_copy = copy.deepcopy(model).eval()
            random_input = torch.rand((2, 10, 24, 24)).cuda()

            # original outputs
            output = model(random_input)

            # Invoke MO implementation
            cle.USE_PYTHON_IMPL = False
            CrossLayerScaling.scale_cls_set_with_conv_layers((model.conv1, model.conv2))
            output_using_mo = model(random_input)

            # Invoke python implementation
            cle.USE_PYTHON_IMPL = True
            CrossLayerScaling.scale_cls_set_with_conv_layers((model_copy.conv1, model_copy.conv2))
            output_using_python = model_copy(random_input)

            # Verify the outputs.
            assert torch.allclose(output_using_mo, output_using_python)
            assert torch.allclose(output, output_using_mo)
            assert torch.allclose(output, output_using_python)

            # Verify the weights.
            assert torch.allclose(model.conv1.weight, model_copy.conv1.weight)
            assert torch.allclose(model.conv2.weight, model_copy.conv2.weight)
        finally:
            cle.USE_PYTHON_IMPL = flag

    @pytest.mark.cuda
    def test_cls_using_python_impl_mobilenetv1(self):
        """ Compare MO and python implementation for CLS """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = MockMobileNetV1().cuda().eval()
            model_copy = copy.deepcopy(model).cuda().eval()
            dummy_input = torch.rand((1, 3, 224, 224)).cuda()

            # BN fold
            fold_all_batch_norms(model, (1, 3, 224, 224), dummy_input=dummy_input)
            fold_all_batch_norms(model_copy, (1, 3, 224, 224), dummy_input=dummy_input)

            # CLS using MO and python.
            cle.USE_PYTHON_IMPL = False
            scale_factors_mo = CrossLayerScaling.scale_model(model, (1, 3, 224, 224), dummy_input)
            cle.USE_PYTHON_IMPL = True
            scale_factors_python = CrossLayerScaling.scale_model(model_copy, (1, 3, 224, 224), dummy_input)

            assert len(scale_factors_mo) == 8
            assert len(scale_factors_python) == 8

            # Verify the outputs.
            assert torch.allclose(model(dummy_input), model_copy(dummy_input))

            # Verify the weights
            assert torch.allclose(model.model[0][0].weight, model_copy.model[0][0].weight)
        finally:
            cle.USE_PYTHON_IMPL = flag

    @pytest.mark.cuda
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_bias_fold_using_python_impl(self, device):
        """ Verify bias fold API using python implementation """
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2)
                self.bn1 = torch.nn.BatchNorm2d(16)
                self.conv2 = torch.nn.ConvTranspose2d(16, 32, kernel_size=3)
                self.bn2 = torch.nn.BatchNorm2d(32)
                self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3)
                for m in self.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        torch.nn.init.normal_(m.weight)
                        torch.nn.init.constant_(m.bias, 4)
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.conv3(x)
                return x

        def _verify_bias_fold(model, dummy_input):

            folded_pairs = batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224), dummy_input=dummy_input)
            bn_dict = {}
            for conv, bn in folded_pairs:
                bn_dict[conv] = bn

            # Create a list of consecutive conv layers to be equalized and scale them.
            consecutive_layer_list = [(model.conv1, model.conv2), (model.conv2, model.conv3)]
            scaling_factor_list = CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

            cls_set_info_list = \
                [ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(model.conv1, model.conv2, scaling_factor_list[0], True)),
                 ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(model.conv2, model.conv3, scaling_factor_list[1], True))]

            conv1_bias_before = model.conv1.bias.clone().cpu()
            conv2_bias_before = model.conv2.bias.clone().cpu()
            conv3_bias_before = model.conv3.bias.clone().cpu()

            # Fold the biases.
            HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

            conv1_bias = model.conv1.bias.detach().cpu()
            conv2_bias = model.conv2.bias.detach().cpu()
            conv3_bias = model.conv3.bias.detach().cpu()

            assert not torch.equal(conv1_bias_before, conv1_bias)
            assert not torch.equal(conv2_bias_before, conv2_bias)
            assert not torch.equal(conv3_bias_before, conv3_bias)

            return conv1_bias, conv2_bias, conv3_bias

        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = Model().eval().to(device)
            model_copy = copy.deepcopy(model).to(device)
            dummy_input = torch.randn(1, 3, 10, 10).to(device)

            # invoke with MO (c++) implementation
            cle.USE_PYTHON_IMPL = False
            conv1_bias_mo, conv2_bias_mo, conv3_bias_mo = _verify_bias_fold(model, dummy_input)
            # invoke with python implementation
            cle.USE_PYTHON_IMPL = True
            conv1_bias_p, conv2_bias_p, conv3_bias_p = _verify_bias_fold(model_copy, dummy_input)

            assert torch.allclose(conv1_bias_mo, conv1_bias_p)
            assert torch.allclose(conv2_bias_mo, conv2_bias_p)
            assert torch.allclose(conv3_bias_mo, conv3_bias_p)
            assert torch.allclose(model(dummy_input), model_copy(dummy_input), rtol=1.e-2)
        finally:
            cle.USE_PYTHON_IMPL = flag

    @pytest.mark.parametrize("groups", [1, 10])
    def test_compare_scale_factors(self, groups):
        """ compare scale factors using with MO and python implementation """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(10, 10, 3, groups=groups),
                torch.nn.Conv2d(10, 10, 3),
            ).eval()

            with torch.no_grad():
                model[0].weight *= model[0].weight * 100

            model_copy = copy.deepcopy(model).eval()
            dummy_input = torch.rand((1, 10, 32, 32))
            cle.USE_PYTHON_IMPL = True
            py_scale_factors = CrossLayerScaling.scale_model(model_copy, dummy_input=dummy_input)
            cle.USE_PYTHON_IMPL = False
            mo_scale_factors = CrossLayerScaling.scale_model(model, dummy_input=dummy_input)
            for py, mo in zip(py_scale_factors[0].cls_pair_info_list[0].scale_factor,
                              mo_scale_factors[0].cls_pair_info_list[0].scale_factor):
                assert np.isclose(py, mo)
        finally:
            cle.USE_PYTHON_IMPL = flag

        def _verify_ranges(module_0, module_1):
            if isinstance(module_0, torch.nn.ConvTranspose2d) and module_0.groups == 1:
                weight_0 = module_0.weight.detach().permute(1, 0, 2, 3)
            else:
                weight_0 = module_0.weight.detach()
            range_conv1_after_scaling = np.max(np.abs(weight_0.cpu().numpy()), axis=(1, 2, 3))
            range_conv2_after_scaling = np.max(np.abs(module_1.weight.detach().cpu().numpy()), axis=(0, 2, 3))
            assert np.allclose(range_conv1_after_scaling, range_conv2_after_scaling)

        # Verify that weights are scaled back to similar ranges
        _verify_ranges(model[0], model[1])
        _verify_ranges(model_copy[0], model_copy[1])

    def test_divide_by_zero(self, use_python_impl):
        """ Ensure scale factors are computed using with MO and python implementation """
        torch.manual_seed(10)
        model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(10, 10, 3, groups=10),
            torch.nn.Conv2d(10, 10, 3),
        ).eval()
        dummy_input = torch.rand((1, 10, 32, 32))
        with torch.no_grad():
            model[0].weight[0, :, :, :] = 0
        CrossLayerScaling.scale_model(model, dummy_input=dummy_input)
        assert not torch.isnan(model[0].weight).any()

    def test_bias_fold_for_convtranspose1d(self):
        """ Verify bias fold for ConvTranspose1d """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = torch.nn.Sequential(
                torch.nn.Conv1d(10, 10, 3),
                torch.nn.BatchNorm1d(10),
                torch.nn.ConvTranspose1d(10, 10, 3)
            ).eval()
            # Initialize BN parameters
            torch.nn.init.normal_(model[1].weight)
            torch.nn.init.normal_(model[1].bias)
            dummy_input = torch.randn(1, 10, 32)
            model_copy = copy.deepcopy(model).eval()
            cle.USE_PYTHON_IMPL = True
            equalize_model(model, dummy_input=dummy_input)
            cle.USE_PYTHON_IMPL = False
            equalize_model(model_copy, dummy_input=dummy_input)
            assert torch.allclose(model[0].bias, model_copy[0].bias, rtol=1.e-1)
            assert torch.allclose(model[2].bias, model_copy[2].bias, rtol=1.e-1)
        finally:
            cle.USE_PYTHON_IMPL = flag

    def test_divide_by_zero_with_depthwise(self):
        """ Ensure scale factors are computed using with MO and python implementation """
        flag = cle.USE_PYTHON_IMPL
        try:
            torch.manual_seed(10)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(10, 10, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, 3, groups=10),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, 1),
                torch.nn.ReLU(),
            ).eval()
            dummy_input = torch.randn(1, 10, 32, 32)
            with torch.no_grad():
                model[2].weight[0, :, :, :] = 0

            model_copy = copy.deepcopy(model).eval()
            cle.USE_PYTHON_IMPL = True
            CrossLayerScaling.scale_model(model, dummy_input=dummy_input)
            cle.USE_PYTHON_IMPL = False
            CrossLayerScaling.scale_model(model_copy, dummy_input=dummy_input)

            assert not torch.isnan(model[0].weight).any()
            assert not torch.isnan(model[2].weight).any()
            assert not torch.isnan(model[4].weight).any()
            assert torch.allclose(model[0].weight, model_copy[0].weight)
            assert torch.allclose(model[2].weight, model_copy[2].weight)
            assert torch.allclose(model[4].weight, model_copy[4].weight)

            with torch.no_grad():
                assert torch.allclose(model(dummy_input), model_copy(dummy_input), rtol=1.e-2)
        finally:
            cle.USE_PYTHON_IMPL = flag
