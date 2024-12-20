# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Cross Layer Equalization acceptance tests for ResNet model. """

import os
import tempfile
import pytest
import copy
import torch
import numpy as np
from torchvision import models

from aimet_torch import batch_norm_fold
from aimet_torch.cross_layer_equalization import CrossLayerScaling, HighBiasFold, equalize_model
from aimet_torch import visualize_model
from models.mobilenet import MobileNetV2


class TestCrossLayerEqualization:
    """ Acceptance tests related to winnowing ResNet models. """

    def test_cross_layer_equalization_resnet(self):

        torch.manual_seed(10)
        model = models.resnet18().eval()

        folded_pairs = batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        assert not isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d)

        w1 = model.layer1[0].conv1.weight.clone()
        w2 = model.layer1[0].conv2.weight.clone()
        w3 = model.layer1[1].conv1.weight.clone()

        cls_set_info_list = CrossLayerScaling.scale_model(model, (1, 3, 224, 224))

        # check if weights are updating
        assert not torch.allclose(model.layer1[0].conv1.weight, w1)
        assert not torch.allclose(model.layer1[0].conv2.weight, w2)
        assert not torch.allclose(model.layer1[1].conv1.weight, w3)

        b1 = model.layer1[0].conv1.bias.clone()
        b2 = model.layer1[1].conv2.bias.clone()

        HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

        for i in range(len(model.layer1[0].conv1.bias)):
            assert model.layer1[0].conv1.bias[i] <= b1[i]

        for i in range(len(model.layer1[1].conv2.bias)):
            assert model.layer1[1].conv2.bias[i] <= b2[i]

    def test_cross_layer_equalization_mobilenet_v2(self):
        torch.manual_seed(10)
        model = MobileNetV2().to(torch.device('cpu'))
        model = model.eval()
        equalize_model(model, (1, 3, 224, 224))

    def test_cross_layer_equalization_vgg(self):
        torch.manual_seed(10)
        model = models.vgg16().to(torch.device('cpu'))
        model = model.eval()
        equalize_model(model, (1, 3, 224, 224))

    @pytest.mark.skip("Takes 1 min 42 secs to run")
    def test_cross_layer_equalization_mobilenet_v2_visualize_after_optimization(self):
        torch.manual_seed(10)
        model = MobileNetV2().to(torch.device('cpu'))
        model = model.eval()
        model_copy = copy.deepcopy(model)
        results_dir = 'artifacts'
        if not os.path.exists('artifacts'):
            os.makedirs('artifacts')

        # model_copy_again = copy.deepcopy(model)
        batch_norm_fold.fold_all_batch_norms(model_copy, (1, 3, 224, 224))
        equalize_model(model, (1, 3, 224, 224))
        visualize_model.visualize_changes_after_optimization(model_copy, model, results_dir)

    def test_cross_layer_equalization_resnet18_visualize_to_identify_problem_layers(self):
        torch.manual_seed(10)
        model = models.resnet18().eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file = os.path.join(tmp_dir, 'visualize_relative_weight_ranges_to_identify_problematic_layers.html')

            batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

            visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, tmp_dir)
            assert os.path.isfile(file)

    def test_cle_transposed_conv2D(self):
        class TransposedConvModel(torch.nn.Module):
            def __init__(self):
                super(TransposedConvModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose2d(20, 10, 3)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.ConvTranspose2d(10, 15, 3)
                self.bn2 = torch.nn.BatchNorm2d(15)

            def forward(self, x):
                # Regular case - conv followed by bn
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x

        torch.manual_seed(10)
        model = TransposedConvModel()

        w_shape_1 = copy.deepcopy(model.conv1.weight.shape)
        w_shape_2 = copy.deepcopy(model.conv2.weight.shape)
        model = model.eval()

        input_shapes = (1, 20, 3, 4)
        random_input = torch.rand(input_shapes)
        output_before_cle = model(random_input).detach().numpy()

        folded_pairs = batch_norm_fold.fold_all_batch_norms(model, input_shapes)
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        cls_set_info_list = CrossLayerScaling.scale_model(model, input_shapes)
        HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

        assert w_shape_1 == model.conv1.weight.shape
        assert w_shape_2 == model.conv2.weight.shape

        output_after_cle = model(random_input).detach().numpy()
        assert np.allclose(output_before_cle, output_after_cle, rtol=1.e-2)

    def test_cle_depthwise_transposed_conv2D(self):

        class TransposedConvModel(torch.nn.Module):
            def __init__(self):
                super(TransposedConvModel, self).__init__()
                self.conv = torch.nn.Conv2d(20, 10, 3)
                self.bn = torch.nn.BatchNorm2d(10)
                self.relu = torch.nn.ReLU()
                self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.ConvTranspose2d(10, 15, 3)
                self.bn2 = torch.nn.BatchNorm2d(15)

            def forward(self, x):
                # Regular case - conv followed by bn
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x

        torch.manual_seed(10)
        model = TransposedConvModel()

        w_shape_1 = copy.deepcopy(model.conv1.weight.shape)
        w_shape_2 = copy.deepcopy(model.conv2.weight.shape)
        model = model.eval()

        input_shapes = (1, 20, 3, 4)
        random_input = torch.rand(input_shapes)
        output_before_cle = model(random_input).detach().numpy()

        folded_pairs = batch_norm_fold.fold_all_batch_norms(model, input_shapes)
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        cls_set_info_list = CrossLayerScaling.scale_model(model, input_shapes)
        HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

        assert w_shape_1 == model.conv1.weight.shape
        assert w_shape_2 == model.conv2.weight.shape

        output_after_cle = model(random_input).detach().numpy()
        assert np.allclose(output_before_cle, output_after_cle, rtol=1.e-2)

    def test_cle_for_maskrcnn(self):
        class JITTraceableWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs):
                outputs = self.model(inputs)
                return outputs[0]["masks"]

        model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        model = JITTraceableWrapper(model).eval()
        input_shapes = (1, 3, 224, 224)
        dummy_input = torch.rand(input_shapes)

        output_before_cle = model(dummy_input)
        equalize_model(model, input_shapes)
        output_after_cle = model(dummy_input)

        assert output_before_cle.shape == output_after_cle.shape
