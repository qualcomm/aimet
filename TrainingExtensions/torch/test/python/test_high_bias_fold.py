# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest.mock
import copy
import torch
from torchvision import models

import numpy as np

from aimet_torch.cross_layer_equalization import HighBiasFold, ClsSetInfo
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from models.test_models import TransposedConvModel


class TestTrainingExtensionHighBiasFold(unittest.TestCase):

    def test_high_bias_fold(self):
        np.random.seed(1)
        torch.random.manual_seed(10)
        model = models.resnet18()

        model = model.eval()
        # random_input = torch.rand(1, 3, 224, 224)
        bn_dict = {model.layer1[0].conv1: model.layer1[0].bn1}

        output_channels = model.layer1[0].conv1.weight.shape[0]
        scale_factor = np.array(np.random.randn(output_channels))
        cls_pair_info = ClsSetInfo.ClsSetLayerPairInfo(model.layer1[0].conv1, model.layer1[0].conv2,
                                                       scale_factor, True)
        cls_set_info = ClsSetInfo(cls_pair_info)

        model.layer1[0].conv1.bias = torch.nn.Parameter(torch.rand(output_channels))
        bias = model.layer1[0].conv1.bias.data
        model.layer1[0].conv2.bias = torch.nn.Parameter(torch.rand(output_channels))

        HighBiasFold.bias_fold([cls_set_info], bn_dict)

        for i in range(len(model.layer1[0].conv1.bias)):
            self.assertTrue(model.layer1[0].conv1.bias.data[i] <= bias.data[i])

    def test_auto_hbf_transposed_conv2d_model(self):
        torch.manual_seed(10)
        model = TransposedConvModel()
        model.eval()

        bn_dict = {model.conv1: model.bn1}
        fold_all_batch_norms(model, (10, 10, 4, 4))

        scale_factor = np.array(np.random.randn(10))
        cls_pair_info = ClsSetInfo.ClsSetLayerPairInfo(model.conv1, model.conv2,
                                                       scale_factor, True)
        cls_set_info = ClsSetInfo(cls_pair_info)
        bias = copy.deepcopy(model.conv1.bias.data)

        HighBiasFold.bias_fold([cls_set_info], bn_dict)

        for i in range(len(model.conv1.bias)):
            self.assertTrue(model.conv1.bias.data[i] <= bias.data[i])