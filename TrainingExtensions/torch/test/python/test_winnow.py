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

"""
Contains unit tests to test the winnowing of a model.

We have captured following scenarios so far for Residual Networks.
Scenario 1 : conv2d/Linear with zero planes right in between Add and Split layer (Non - optimized)
Scenario 2 : conv2d/Linear with zero planes right below an split layer
Scenario 3 : conv2d/Linear with zero planes right below an Add and split layer
Scenario 4 : conv2d/Linear with zero planes right below an Add layer
 """

import os
import tempfile
import pytest
import unittest
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from aimet_common.utils import AimetLogger
from aimet_common.winnow.winnow_utils import OpConnectivity, ConnectivityType
from .models.test_models import ModuleListModel, SingleResidual
from aimet_torch.winnow.winnow import winnow_model
from aimet_torch.winnow.mask_propagation_winnower import MaskPropagationWinnower
from aimet_torch.winnow.winnow_utils import zero_out_input_channels, search_for_zero_planes, DownsampleLayer
from aimet_torch.utils import get_layer_name
from aimet_torch.meta.connectedgraph import ConnectedGraph
from .models.mobilenet import MobileNetV2, MobileNetV1

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

# pylint: disable=too-many-lines

def mobilenetv1(pretrained=False, **_):
    """Constructs a MobileNetV1 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    """
    model = MobileNetV1()
    if pretrained:
        logger.info("Pretrained model is not available to eliminate dependency on external directories")
    return model


def mobilenetv2(pretrained=False, **kwargs):
    """Constructs a MobileNetV2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        model_file (str):  If pretrained=True, this path is used to find/load the model file

    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        logger.info("Pretrained model is not available to eliminate dependency on external directories")
    return model


###############################################################################
# This code within this comment block is from platform systems team.
# Brought over to AIMET to test Concat.
def get_pre_stage_net():
    """ Get pre stage network dict """
    network_dict = {'block_pre_stage': [{'conv4_3_CPM': [512, 256, 3, 1, 1]},
                                        {'conv4_4_CPM': [256, 128, 3, 1, 1]}]}
    return network_dict


def get_shared_network_dict():
    """ Get shared network dict """
    network_dict = get_pre_stage_net()
    stage_channel = [0, 128, 185, 185, 185, 185, 185]
    for i in range(1, 3):
        network_dict['block%d_shared' % i] = [{'Mconv1_stage%d_L1' % i: [stage_channel[i], 128, 7, 1, 3]},
                                              {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]}]

        network_dict['block%d_1' % i] = [{'Mconv3_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv4_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv5_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                         {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}]
        network_dict['block%d_2' % i] = [{'Mconv3_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv4_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv5_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                         {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}]
    return network_dict


def get_network_dict():
    """ Get network dict """
    network_dict = get_pre_stage_net()
    stage_channel = [0, 128, 185, 185, 185, 185, 185]
    for i in range(1, 3):
        network_dict['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [stage_channel[i], 128, 7, 1, 3]},
                                         {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                         {'Mconv3_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv4_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv5_stage%d_L1' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                         {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}]
        network_dict['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [stage_channel[i], 128, 7, 1, 3]},
                                         {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                         {'Mconv3_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv4_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv5_stage%d_L2' % i: [128, 128, 3, 1, 1]},
                                         {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                         {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}]
    return network_dict


def get_model(share_weights=False, upsample=False):     # pylint: disable=too-many-statements
    """ Return a network dict for the model """
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
              {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1_stage1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]},
              {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2_stage1': [2, 2, 0]},
              {'conv3_1': [128, 256, 3, 1, 1]},
              {'conv3_2': [256, 256, 3, 1, 1]},
              {'conv3_3': [256, 256, 3, 1, 1]},
              {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3_stage1': [2, 2, 0]},
              {'conv4_1': [256, 512, 3, 1, 1]},
              {'conv4_2': [512, 512, 3, 1, 1]}]

    if share_weights:
        print("defining network with shared weights")
        network_dict = get_shared_network_dict()
    else:
        network_dict = get_network_dict()

    def define_base_layers(block, layer_size):
        layers = []
        for i in range(layer_size):
            one_ = block[i]
            for k, v in zip(one_.keys(), one_.values()):
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return layers

    def define_stage_layers(cfg_dict):
        layers = define_base_layers(cfg_dict, len(cfg_dict) - 1)
        one_ = cfg_dict[-1].keys()
        k = list(one_)[0]
        v = cfg_dict[-1][k]
        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]
        return nn.Sequential(*layers)

    # create all the layers of the model
    base_layers = define_base_layers(block0, len(block0))
    pre_stage_layers = define_base_layers(network_dict['block_pre_stage'], len(network_dict['block_pre_stage']))
    blocks = {'block0': nn.Sequential(*base_layers),
              'block_pre_stage': nn.Sequential(*pre_stage_layers)}
    if share_weights:
        shared_layers_s1 = define_base_layers(network_dict['block1_shared'], len(network_dict['block1_shared']))
        shared_layers_s2 = define_base_layers(network_dict['block2_shared'], len(network_dict['block2_shared']))
        blocks['block1_shared'] = nn.Sequential(*shared_layers_s1)
        blocks['block2_shared'] = nn.Sequential(*shared_layers_s2)

    for k, v in zip(network_dict.keys(), network_dict.values()):
        if 'shared' not in k and 'pre_stage' not in k:
            blocks[k] = define_stage_layers(v)

    class PoseModel(nn.Module):
        """ Pose Model class """
        def __init__(self, model_dict, upsample=False):
            super(PoseModel, self).__init__()
            self.upsample = upsample
            self.basemodel = model_dict['block0']
            self.pre_stage = model_dict['block_pre_stage']
            if share_weights:
                self.stage1_shared = model_dict['block1_shared']
            self.stage1_1 = model_dict['block1_1']
            self.stage2_1 = model_dict['block2_1']
            # self.stage3_1 = model_dict['block3_1']
            # self.stage4_1 = model_dict['block4_1']
            # self.stage5_1 = model_dict['block5_1']
            # self.stage6_1 = model_dict['block6_1']
            if share_weights:
                self.stage2_shared = model_dict['block2_shared']
            self.stage1_2 = model_dict['block1_2']
            self.stage2_2 = model_dict['block2_2']
            # self.stage3_2 = model_dict['block3_2']
            # self.stage4_2 = model_dict['block4_2']
            # self.stage5_2 = model_dict['block5_2']
            # self.stage6_2 = model_dict['block6_2']

        def forward(self, *inputs):
            out1_vgg = self.basemodel(inputs[0])
            out1 = self.pre_stage(out1_vgg)
            if share_weights:
                out1_shared = self.stage1_shared(out1)
            else:
                out1_shared = out1
            out1_1 = self.stage1_1(out1_shared)
            out1_2 = self.stage1_2(out1_shared)

            out2 = torch.cat([out1_1, out1_2, out1], 1)
            if share_weights:
                out2_shared = self.stage2_shared(out2)
            else:
                out2_shared = out2
            out2_1 = self.stage2_1(out2_shared)
            out2_2 = self.stage2_2(out2_shared)
            # out3 = torch.cat([out2_1, out2_2, out1], 1)

            # out3_1 = self.stage3_1(out3)
            # out3_2 = self.stage3_2(out3)
            # out4 = torch.cat([out3_1, out3_2, out1], 1)
            #
            # out4_1 = self.stage4_1(out4)
            # out4_2 = self.stage4_2(out4)
            # out5 = torch.cat([out4_1, out4_2, out1], 1)
            #
            # out5_1 = self.stage5_1(out5)
            # out5_2 = self.stage5_2(out5)
            # out6 = torch.cat([out5_1, out5_2, out1], 1)
            #
            # out6_1 = self.stage6_1(out6)
            # out6_2 = self.stage6_2(out6)

            if self.upsample:
                # parameters to check for up-sampling: align_corners = True, mode='nearest'
                upsampler = nn.Upsample(scale_factor=2, mode='bilinear')
                out2_1_up = upsampler(out2_1)
                out2_2_up = upsampler(out2_2)
                return out1_1, out1_2, out2_1, out2_2, out2_1_up, out2_2_up
            return out1_1, out1_2, out2_1, out2_2

    model = PoseModel(blocks, upsample=upsample)
    return model

###############################################################################


class AnotherSingleResidual(nn.Module):     # pylint: disable=too-many-instance-attributes
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(AnotherSingleResidual, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # The output ofBatchNorm2d layer above(bn33) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(160000, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        # ada = nn.AdaptiveAvgPool2d(14)
        # residual = ada(residual)
        x += residual
        x = self.relu3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SingleResidualScenario4(nn.Module):       # pylint: disable=too-many-instance-attributes
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidualScenario4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # The output ofBatchNorm2d layer above(bn33) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(160000, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        # Save the output of MaxPool as residual.
        residual = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DualResidual(nn.Module):      # pylint: disable=too-many-instance-attributes
    """ A model with a two residual connections.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(DualResidual, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ada1 = nn.AdaptiveAvgPool2d(56)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        # The following layers are considered as single block.
        self.conv4 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ada2 = nn.AdaptiveAvgPool2d(7)

        # The output of Conv2d layer above(conv4) is added with the the residual from
        # MaxPool2d (maxpool2) and then fed to the relu layer below.
        self.relu5 = nn.ReLU(inplace=True)

        # All layers below are same as ResNet
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Save the output of MaxPool as residual1.
        residual1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual1 = self.ada1(residual1)
        x += residual1
        x = self.relu3(x)

        x = self.conv4(x)
        # Save the output of conv4 as residual2.
        residual2 = x

        x = self.bn3(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual2 = self.ada2(residual2)

        x += residual2
        x = self.relu5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ###################################################################################################


__all__ = ['BNInception']

pretrained_settings = {
    'bnincept': {
        'imagenet': {
            # Was ported using python2 (may trigger warning)
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bnincept-239d2248.pth',
            # 'url': 'http://yjxiong.me/others/bnincept-9f5701afb96c8044.pth',
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 255],
            'mean': [104, 117, 128],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}


class BNInceptionModule(nn.Module):     # pylint: disable=too-many-instance-attributes
    """ BNInception submodule class """
    def __init__(self, in_dim, dim1x1, dimr3x3, dim3x3,     # pylint: disable=too-many-arguments
                 dimdr3x3, dimd3x3, dimp, pool='avg'):
        """
            pool: 'avg' or 'max'
        """
        super(BNInceptionModule, self).__init__()
        inplace = True
        self.relu = nn.ReLU(inplace)
        self.bnincept_1x1 = nn.Conv2d(
            in_dim, dim1x1, kernel_size=1, stride=1)
        self.bnincept_1x1_bn = nn.BatchNorm2d(dim1x1, momentum=0.1)
        self.bnincept_3x3_reduce = nn.Conv2d(
            in_dim, dimr3x3, kernel_size=1, stride=1)
        self.bnincept_3x3_reduce_bn = nn.BatchNorm2d(
            dimr3x3, momentum=0.1)
        self.bnincept_3x3 = nn.Conv2d(
            dimr3x3, dim3x3, kernel_size=3, stride=1, padding=1)
        self.bnincept_3x3_bn = nn.BatchNorm2d(dim3x3, momentum=0.1)
        self.bnincept_double_3x3_reduce = nn.Conv2d(
            in_dim, dimdr3x3, kernel_size=1, stride=1)
        self.bnincept_double_3x3_reduce_bn = nn.BatchNorm2d(
            dimdr3x3, momentum=0.1)
        self.bnincept_double_3x3_1 = nn.Conv2d(
            dimdr3x3, dimd3x3, kernel_size=3, stride=1, padding=1)
        self.bnincept_double_3x3_1_bn = nn.BatchNorm2d(
            dimd3x3, momentum=0.1)
        self.bnincept_double_3x3_2 = nn.Conv2d(
            dimd3x3, dimd3x3, kernel_size=3, stride=1, padding=1)
        self.bnincept_double_3x3_2_bn = nn.BatchNorm2d(
            dimd3x3, momentum=0.1)
        if pool == 'avg':
            self.bnincept_pool = nn.AvgPool2d(
                3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        else:
            self.bnincept_pool = nn.MaxPool2d(
                3, stride=1, padding=1, ceil_mode=True)
        self.bnincept_pool_proj = nn.Conv2d(
            in_dim, dimp, kernel_size=1, stride=1)
        self.bnincept_pool_proj_bn = nn.BatchNorm2d(dimp, momentum=0.1)

    def forward(self, *inputs):     # pylint: disable=too-many-locals
        out_1x1 = self.bnincept_1x1(inputs[0])
        out_1x1_bn = self.bnincept_1x1_bn(out_1x1)
        _ = self.relu(out_1x1_bn)
        out_3x3_reduce = self.bnincept_3x3_reduce(inputs[0])
        out_3x3_reduce_bn = self.bnincept_3x3_reduce_bn(out_3x3_reduce)
        _ = self.relu(out_3x3_reduce_bn)
        out_3x3 = self.bnincept_3x3(out_3x3_reduce_bn)
        out_3x3_bn = self.bnincept_3x3_bn(out_3x3)
        _ = self.relu(out_3x3_bn)
        out_d3x3_reduce = self.bnincept_double_3x3_reduce(inputs[0])
        out_d3x3_reduce_bn = self.bnincept_double_3x3_reduce_bn(
            out_d3x3_reduce)
        _ = self.relu(out_d3x3_reduce_bn)
        out_d3x3_1 = self.bnincept_double_3x3_1(out_d3x3_reduce_bn)
        out_d3x3_1_bn = self.bnincept_double_3x3_1_bn(out_d3x3_1)
        _ = self.relu(out_d3x3_1_bn)
        out_d3x3_2 = self.bnincept_double_3x3_2(out_d3x3_1_bn)
        out_d3x3_2_bn = self.bnincept_double_3x3_2_bn(out_d3x3_2)
        _ = self.relu(out_d3x3_2_bn)
        out_pool = self.bnincept_pool(inputs[0])
        out_pool_proj = self.bnincept_pool_proj(out_pool)
        out_pool_proj_bn = self.bnincept_pool_proj_bn(out_pool_proj)
        _ = self.relu(out_pool_proj_bn)
        output = torch.cat(
            [out_1x1_bn, out_3x3_bn, out_d3x3_2_bn, out_pool_proj_bn], 1)
        return output


class BNInceptionStrideModule(nn.Module):       # pylint: disable=too-many-instance-attributes
    """ BNInception Stride Module class """
    def __init__(self, in_dim, dimr3x3, dim3x3, dimdr3x3, dimd3x3):
        super(BNInceptionStrideModule, self).__init__()
        inplace = True
        self.relu = nn.ReLU(inplace)
        self.bnincept_3x3_reduce = nn.Conv2d(
            in_dim, dimr3x3, kernel_size=1, stride=1)
        self.bnincept_3x3_reduce_bn = nn.BatchNorm2d(dimr3x3, momentum=0.1)
        self.bnincept_3x3 = nn.Conv2d(
            dimr3x3, dim3x3, kernel_size=3, stride=2, padding=1)
        self.bnincept_3x3_bn = nn.BatchNorm2d(dim3x3, momentum=0.1)
        self.bnincept_double_3x3_reduce = nn.Conv2d(
            in_dim, dimdr3x3, kernel_size=1, stride=1)
        self.bnincept_double_3x3_reduce_bn = nn.BatchNorm2d(
            dimdr3x3, momentum=0.1)
        self.bnincept_double_3x3_1 = nn.Conv2d(
            dimdr3x3, dimd3x3, kernel_size=3, stride=1, padding=1)
        self.bnincept_double_3x3_1_bn = nn.BatchNorm2d(dimd3x3, momentum=0.1)
        self.bnincept_double_3x3_2 = nn.Conv2d(
            dimd3x3, dimd3x3, kernel_size=3, stride=2, padding=1)
        self.bnincept_double_3x3_2_bn = nn.BatchNorm2d(dimd3x3, momentum=0.1)
        self.bnincept_pool = nn.MaxPool2d(
            (3, 3), stride=2, dilation=1, ceil_mode=True)

    def forward(self, *inputs):
        out_3x3_red = self.bnincept_3x3_reduce(inputs[0])
        out_3x3_red_bn = self.bnincept_3x3_reduce_bn(out_3x3_red)
        _ = self.relu(out_3x3_red_bn)
        out_3x3 = self.bnincept_3x3(out_3x3_red_bn)
        out_3x3_bn = self.bnincept_3x3_bn(out_3x3)
        _ = self.relu(out_3x3_bn)
        out_d3x3_red = self.bnincept_double_3x3_reduce(inputs[0])
        out_d3x3_red_bn = self.bnincept_double_3x3_reduce_bn(out_d3x3_red)
        _ = self.relu(out_d3x3_red_bn)
        out_d3x3_1 = self.bnincept_double_3x3_1(out_d3x3_red_bn)
        out_d3x3_1_bn = self.bnincept_double_3x3_1_bn(out_d3x3_1)
        _ = self.relu(out_d3x3_1_bn)
        out_d3x3_2 = self.bnincept_double_3x3_2(out_d3x3_1_bn)
        out_d3x3_2_bn = self.bnincept_double_3x3_2_bn(out_d3x3_2)
        _ = self.relu(out_d3x3_2_bn)
        out_pool = self.bnincept_pool(inputs[0])
        output = torch.cat([out_3x3_bn, out_d3x3_2_bn, out_pool], 1)
        return output


class BNInception(nn.Module):       # pylint: disable=too-many-instance-attributes
    """ BNInception class """
    def __init__(self, num_classes=1000, reduction=True):
        super(BNInception, self).__init__()
        inplace = True

        input_dim = 3
        if reduction:
            self.reduction_conv = nn.Conv2d(
                3, 6, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3))
            self.reduction_bn = nn.BatchNorm2d(6, momentum=0.1)
            self.relu = nn.ReLU(inplace)
            self.stage0 = nn.Sequential(
                # self.reduction_conv, self.reduction_bn, self.relu)
                nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3)),
                nn.BatchNorm2d(6, momentum=0.1),
                nn.ReLU(inplace))
            input_dim = 6
        self.reduction = reduction

        # 1/4 scale output
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace),
            nn.MaxPool2d((3, 3), stride=2, dilation=1, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, momentum=0.1),
            nn.ReLU(inplace))

        self.bnincept_3a = BNInceptionModule(
            in_dim=192, dim1x1=64, dimr3x3=64, dim3x3=64,
            dimdr3x3=64, dimd3x3=96, dimp=32)
        self.bnincept_3b = BNInceptionModule(
            in_dim=256, dim1x1=64, dimr3x3=64, dim3x3=96,
            dimdr3x3=64, dimd3x3=96, dimp=64)

        # 1/8 scale output
        self.stage2 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=2, dilation=1, ceil_mode=True),
            self.bnincept_3a, self.bnincept_3b)

        self.bnincept_3c = BNInceptionStrideModule(
            in_dim=320, dimr3x3=128, dim3x3=160, dimdr3x3=64, dimd3x3=96)
        self.bnincept_4a = BNInceptionModule(
            in_dim=576, dim1x1=224, dimr3x3=64, dim3x3=96,
            dimdr3x3=96, dimd3x3=128, dimp=128)
        self.bnincept_4b = BNInceptionModule(
            in_dim=576, dim1x1=192, dimr3x3=96, dim3x3=128,
            dimdr3x3=96, dimd3x3=128, dimp=128)
        self.bnincept_4b = BNInceptionModule(
            in_dim=576, dim1x1=192, dimr3x3=96, dim3x3=128,
            dimdr3x3=96, dimd3x3=128, dimp=128)
        self.bnincept_4c = BNInceptionModule(
            in_dim=576, dim1x1=160, dimr3x3=128, dim3x3=160,
            dimdr3x3=128, dimd3x3=160, dimp=128)
        self.bnincept_4d = BNInceptionModule(
            in_dim=608, dim1x1=96, dimr3x3=128, dim3x3=192,
            dimdr3x3=160, dimd3x3=192, dimp=128)

        # 1/16 scale output
        self.stage3 = nn.Sequential(
            self.bnincept_3c, self.bnincept_4a, self.bnincept_4b,
            self.bnincept_4c, self.bnincept_4d)

        self.bnincept_4e = BNInceptionStrideModule(
            in_dim=608, dimr3x3=128, dim3x3=192, dimdr3x3=192, dimd3x3=256)
        self.bnincept_5a = BNInceptionModule(
            in_dim=1056, dim1x1=352, dimr3x3=192, dim3x3=320,
            dimdr3x3=160, dimd3x3=224, dimp=128)
        self.bnincept_5b = BNInceptionModule(
            in_dim=1024, dim1x1=352, dimr3x3=192, dim3x3=320,
            dimdr3x3=192, dimd3x3=224, dimp=128, pool='max')

        # 1/32 scale output
        self.stage4 = nn.Sequential(
            self.bnincept_4e, self.bnincept_5a, self.bnincept_5b)

        self.global_pool = nn.AvgPool2d(
            7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, inp):
        """ Features function """
        if self.reduction:
            inp = self.stage0(inp)
        stage1 = self.stage1(inp)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage4

    def logits(self, features):
        """ Logits function """
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, *inputs):
        x = self.features(inputs[0])
        x = self.logits(x)
        return x


def bninception(num_classes=1000, pretrained='imagenet'):
    """BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper. """
    model = BNInception(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['bnincept'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        # model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']     # pylint: disable=attribute-defined-outside-init
        model.input_size = settings['input_size']       # pylint: disable=attribute-defined-outside-init
        model.input_range = settings['input_range']     # pylint: disable=attribute-defined-outside-init
        model.mean = settings['mean']                   # pylint: disable=attribute-defined-outside-init
        model.std = settings['std']                     # pylint: disable=attribute-defined-outside-init
    return model

# ########################################################################################################


def load_model(path):
    """ Loads a model from the location of the path. """
    model = torch.load(path)
    return model


def save_model(model: nn.Module, path):
    """ Saves a model to the location of the path. """
    torch.save(model, path)


def load_checkpoint(model, path):
    """ Loads the state of model from the location of the path to the given model. """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)


def load_model_vgg16(local_path):
    """ Loads the VGG-16 """
    # see if a local copy of the model is present
    if os.path.exists(local_path):
        model = load_model(local_path)
    else:
        # download (~15 sec) and save locally
        model = torchvision.models.vgg16()
        save_model(model, local_path)
    return model


class TestTrainingExtensionsWinnow(unittest.TestCase):      # pylint: disable=too-many-public-methods
    """ Unit test cases for winnowing. """

    def test_winnowing_partial(self):
        """Validate output of a single winnowed module and the modules impacted by it."""

        model = models.vgg16(pretrained=False)
        input_shape = [1, 3, 224, 224]

        input_channels_to_prune = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                                   100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                   120, 121, 122, 123]

        list_of_modules_to_winnow = [(model.features[10], input_channels_to_prune)]

        new_model, _ = winnow_model(model, input_shape, list_of_modules_to_winnow, verbose=True)

        self.assertEqual(new_model.features[10].in_channels, 64)
        self.assertEqual(new_model.features[7].out_channels, 64)

    def test_vgg_winnowing(self):
        """Validate output of entire model with a single winnowed module."""

        model = models.vgg16(pretrained=False)

        batch_size = 1
        input_shape = [batch_size, 3, 224, 224]

        input_channels_to_prune = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                                   100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                   120, 121, 122, 123]

        list_of_modules_to_winnow = [(model.features[10], input_channels_to_prune)]

        new_model, _ = winnow_model(model, input_shape, list_of_modules_to_winnow, verbose=True)

        self.assertTrue(new_model.features[10].in_channels == 64)

    def test_vgg_back_to_back_conv_winnowing(self):
        """Validate output of entire model with a single winnowed module."""

        model = models.vgg16(pretrained=False)
        model.eval()
        input_shape = (1, 3, 224, 224)

        # For the Conv2d at VGG.features[10], zero out 15, 29, 24, 28, 33, 47
        input_channels_to_prune = [15, 29, 24, 28, 33, 47]
        zero_out_input_channels(model.features[10], input_channels_to_prune)

        # For the Conv2d at VGG.features[12], zero out input channels 5, 9, 14, 18, 23, 27, 32, 36, 41, 44, 54
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        zero_out_input_channels(model.features[12], input_channels_to_prune)

        list_of_modules_to_winnow = search_for_zero_planes(model)
        new_model, _ = winnow_model(model, input_shape, list_of_modules_to_winnow, verbose=True)

        # compare zeroed out and pruned model output
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)
        is_cuda = next(new_model.parameters()).is_cuda
        print("\n\n Winnowed Model is CUDA:      ", is_cuda)
        test_output = new_model(input_tensor)
        self.assertTrue(test_output.shape == validation_output.shape)

        # VGG/Sequential[features]/Conv2d[7]
        # 6 output channels will be impacted by VGG/Sequential[features]/Conv2d[10] input channel pruning
        # input channel
        self.assertEqual(new_model.features[7].weight.shape[1], 128)
        # output channel
        self.assertEqual(new_model.features[7].weight.shape[0], 122)
        self.assertEqual(new_model.features[7].in_channels, 128)
        self.assertEqual(new_model.features[7].out_channels, 122)
        # VGG/Sequential[features]/Conv2d[10]
        # 6 input channels are pruned of VGG/Sequential[features]/Conv2d[10]
        # 11 output channels will be impacted by VGG/Sequential[features]/Conv2d[12] input channel pruning
        # input channel
        self.assertEqual(new_model.features[10].weight.shape[1], 122)
        # output channel
        self.assertEqual(new_model.features[10].weight.shape[0], 245)
        self.assertEqual(new_model.features[10].in_channels, 122)
        self.assertEqual(new_model.features[10].out_channels, 245)
        # VGG/Sequential[features]/Conv2d[12]
        # 11 input channels are pruned of VGG/Sequential[features]/Conv2d[12]
        # input channel
        self.assertEqual(new_model.features[12].weight.shape[1], 245)
        # output channel
        self.assertEqual(new_model.features[12].weight.shape[0], 256)
        self.assertEqual(new_model.features[12].in_channels, 245)
        self.assertEqual(new_model.features[12].out_channels, 256)

    def test_zero_out_input_channels_resnet18(self):
        """Zero out input channels of one of the Conv2d layer of Resnet-18 model"""
        model = models.resnet18(pretrained=False)
        # zero out input channels 0 and 2
        input_channels_to_prune = [0, 2]
        zero_out_input_channels(model.layer1[0].conv1, input_channels_to_prune)
        self.assertTrue(np.all(model.layer1[0].conv1.weight[:, 0, :, :].detach().numpy() == 0))
        self.assertTrue(np.all(model.layer1[0].conv1.weight[:, 2, :, :].detach().numpy() == 0))

    def test_single_residual(self):
        """ Tests the simple single residual model. """
        model = SingleResidual()
        model.eval()
        input_shape = [1, 3, 32, 32]

        input_channels_to_prune = [1, 3]
        list_of_modules_to_winnow = [(model.conv3, input_channels_to_prune)]

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        # compare model output
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)

        # validate winnowed net
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)
        # Since the model classifies 10 classes
        self.assertTrue(test_output.size()[-1] == 10)

        # In the winnowed model, conv3 has in_channels = 14, out_channels = 8
        self.assertTrue(new_model.conv3.in_channels == 14)
        self.assertTrue(new_model.conv3.out_channels == 8)

        # The winnowed model's bn2 layer has 14 num_features
        self.assertEqual(new_model.bn2.num_features, 14)
        self.assertEqual(list(new_model.bn2.weight.shape), [14])
        self.assertEqual(list(new_model.bn2.bias.shape), [14])
        self.assertEqual(list(new_model.bn2.running_mean.shape), [14])
        self.assertEqual(list(new_model.bn2.running_var.shape), [14])

        # In the winnowed model, conv2 has in_channels = 32, out_channels = 14 (impacted by layer3 pruning)
        self.assertTrue(new_model.conv2.in_channels == 32)
        self.assertTrue(new_model.conv2.out_channels == 14)

    def test_dual_residual(self):
        """ Tests the dual residual model. """

        model = DualResidual()
        model.eval()
        input_shape = [1, 3, 896, 896]

        # For conv5 layer
        list_of_modules_to_winnow = []
        input_channels_to_prune = [3, 4, 5, 9, 10, 11]
        module_mask_pair = (model.conv5, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        # In the winnowed model, conv5 has in_channels = 58, out_channels = 64
        self.assertTrue(new_model.conv5[1].in_channels == 58)
        self.assertTrue(new_model.conv5[1].out_channels == 64)

    def test_resnetlike_single_residual_(self):
        """ Tests the simple single residual model. """
        model = AnotherSingleResidual()
        model.eval()
        input_shape = [1, 3, 224, 224]

        # For conv3 layer, zero out input channels 1 and 3
        input_channels_to_prune = [1, 3]

        list_of_modules_to_winnow = [(model.conv3, input_channels_to_prune)]

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        # compare zeroed out and pruned model output
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)
        # validate winnowed net
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)
        self.assertTrue(test_output.allclose(validation_output))

        # In the winnowed model, conv3 has in_channels = 62, out_channels = 64
        self.assertTrue(new_model.conv3.in_channels == 62)
        self.assertTrue(new_model.conv3.out_channels == 64)

        # The winnowed model's bn2 layer has 62 num_features
        self.assertEqual(new_model.bn2.num_features, 62)
        self.assertEqual(list(new_model.bn2.weight.shape), [62])
        self.assertEqual(list(new_model.bn2.bias.shape), [62])
        self.assertEqual(list(new_model.bn2.running_mean.shape), [62])
        self.assertEqual(list(new_model.bn2.running_var.shape), [62])

        # In the winnowed model, conv2 has in_channels = 64, out_channels = 62 (impacted by layer3 pruning)
        self.assertTrue(new_model.conv2.in_channels == 64)
        self.assertTrue(new_model.conv2.out_channels == 62)

    def test_winnowing_resnet18_onnx_export(self):
        """ Tests winnowing resnet18 with multiple layers  with zero planes. """

        model = models.resnet18()
        model.eval()

        input_shape = [1, 3, 224, 224]
        list_of_modules_to_winnow = []

        # For layer4[1].conv2 layer
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        module_mask_pair = (model.layer4[1].conv2, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer4[0].conv1 layer
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        module_mask_pair = (model.layer4[0].conv1, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer3[1].conv2 layer
        input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
        module_mask_pair = (model.layer3[1].conv2, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer2[1].conv2 layer
        input_channels_to_prune = [33, 44, 55]
        module_mask_pair = (model.layer2[1].conv2, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer2[0].conv2 layer
        input_channels_to_prune = [11, 12, 13, 14, 15]
        module_mask_pair = (model.layer2[0].conv2, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer1[1].conv1 layer
        input_channels_to_prune = [55, 56, 57, 58, 59]
        module_mask_pair = (model.layer1[1].conv1, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # For layer1[0].conv2 layer
        input_channels_to_prune = [42, 44, 46]
        module_mask_pair = (model.layer1[0].conv2, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        input_tensor = torch.rand(input_shape)

        # Save the model as ONNX.
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'winnowed_index_select' + '.onnx'
            final_path = os.path.join(tmpdir, filename)
            torch.onnx.export(new_model, input_tensor, final_path, verbose=True, export_params=True,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
            print("Saved the winnowed model as ONNX")
            self.assertEqual(0, 0)

    def test_winnowing_single_layer_below_add_single_residual_scenario4(self):
        """ Tests the simple single residual model for Scenario 4. """
        model = SingleResidualScenario4()
        model.eval()
        input_shape = [1, 3, 224, 224]
        # conv4 layer is right below Add layer
        # For conv4 layer, zero out input channels 1 and 3
        list_of_modules_to_winnow = []
        module = model.conv4
        input_channels_to_prune = [1, 3]
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)
        # validate winnowed net
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)
        self.assertTrue(test_output.allclose(validation_output))

        # In the winnowed model, conv4[1] has in_channels = 62, out_channels = 64
        self.assertTrue(new_model.conv4[1].in_channels == 62)
        self.assertTrue(new_model.conv4[1].out_channels == 64)

    def test_pose_model_with_concat(self):
        """ Test winnowing pose model """
        pose_model = get_model()
        # print("Original Pose Model with Concat Op", pose_model)

        input_shape = [1, 3, 368, 368]

        list_of_modules_to_winnow = []
        # Winnow some input channels for one of the Conv2d layers.
        print("Pose Model, basemodel[14] Conv2d's Weight shape", pose_model.basemodel[14].weight.shape, pose_model.basemodel[14])
        input_channels_to_prune_1 = [0, 5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 255]
        module_mask_pair = (pose_model.basemodel[14], input_channels_to_prune_1)
        list_of_modules_to_winnow.append(module_mask_pair)

        # This Conv2d layer is just below the Concat operation.
        # Since Concat models are not supported yet, this scenario is expected to fail.
        print("Pose Model, Stage2_1[0] Conv2d's Weight shape", pose_model.stage2_1[0].weight.shape, pose_model.stage2_1[0])
        input_channels_to_prune_2 = [15, 19, 24, 28, 33, 37, 42, 46, 51, 55, 64]
        module_mask_pair = (pose_model.stage2_1[0], input_channels_to_prune_2)
        list_of_modules_to_winnow.append(module_mask_pair)

        # Winnow some input channels for one of the Conv2d layers with 1x1 kernel
        print("Pose Model, Stage1_1[12] Conv2d's Weight shape", pose_model.stage1_1[12].weight.shape, pose_model.stage1_1[12])
        input_channels_to_prune_3 = [0, 25, 29, 24, 38, 43, 47, 52, 56, 61, 65, 74, 127]
        module_mask_pair = (pose_model.stage1_1[12], input_channels_to_prune_3)
        list_of_modules_to_winnow.append(module_mask_pair)

        new_model, _ = winnow_model(pose_model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        self.assertTrue(new_model)
        self.assertTrue(new_model.basemodel[14].in_channels == 244)
        self.assertTrue(new_model.stage2_1[0][1].in_channels == 174)
        self.assertTrue(new_model.stage1_1[12].in_channels == 115)

    def test_winnow_model_api_resnet18(self):
        """ Tests the winnow_model() API for winnowing resnet18 with
        multiple layers  with zero planes."""

        model = models.resnet18()
        model.eval()
        zeroed_model = zero_out_select_input_channels(model)
        input_shape = (1, 3, 224, 224)
        # winnow_model() API.
        list_of_modules_to_winnow = search_for_zero_planes(model)
        winnowed_model, _ = winnow_model(zeroed_model, input_shape, list_of_modules_to_winnow)

        # compare zeroed out and pruned model output
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)

        # validate winnowed net
        test_output = winnowed_model(input_tensor)
        self.assertTrue(test_output.shape == validation_output.shape)

    @pytest.mark.cuda
    def test_winnow_model_api_resnet18_memory_check(self):
        """
        Tests the winnow_model() API and check the memory leak
        """

        model = models.resnet18().cuda()
        model.eval()
        zeroed_model = zero_out_select_input_channels(model)
        input_shape = (1, 3, 224, 224)

        for i in range(5):
            mem_before = torch.cuda.memory_allocated()
            list_of_modules_to_winnow = search_for_zero_planes(model)
            _, _ = winnow_model(zeroed_model, input_shape, list_of_modules_to_winnow, in_place=False)
            mem_after = torch.cuda.memory_allocated()

            # get the diff in MB
            diff = (mem_after/(1024 * 1024)) - (mem_before/(1024 * 1024))

            print("memory before: ", mem_before/(1024 * 1024))
            print("memory after: ", mem_after/(1024 * 1024))
            print("diff: ", diff)

            # for the first iteration the memory consumption will be doubled because of in_place = False
            # for rest of the iterations, diff should be zero.
            if i > 0:
                self.assertTrue(diff < 50)

    def test_winnow_model_api_resnet18_no_reshaping(self):
        """ Tests the winnow_model() API for winnowing resnet18 with
        multiple layers  with zero planes."""

        model = models.resnet18()
        zeroed_model = zero_out_select_input_channels(model)
        input_shape = (1, 3, 224, 224)
        # winnow_model() API.
        list_of_modules_to_winnow = search_for_zero_planes(model)
        winnowed_model, _ = winnow_model(zeroed_model, input_shape, list_of_modules_to_winnow, reshape=False)

        # compare zeroed out and pruned model output
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)

        # validate winnowed net
        test_output = winnowed_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)

    def test_winnow_model_inplace_known_zero_planes_api(self):
        """ Tests winnowing resnet18 with multiple layers  with zero planes. """

        model = models.resnet18()
        model.eval()
        input_shape = (1, 3, 224, 224)
        module_zero_channels_list = create_list_of_modules_to_winnow(model)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, mod_list = winnow_model(model, input_shape, module_zero_channels_list, in_place=True,
                                                verbose=True)
        print("Order of modified modules after winnowing", [get_layer_name(winnowed_model, mod) for _, mod in mod_list])

        # Test the forward pass
        winnowed_model(torch.rand(input_shape))

        self.assertTrue(winnowed_model.layer4[1].conv2.in_channels == 501)
        self.assertTrue(winnowed_model.layer4[1].conv2.out_channels == 512)
        self.assertTrue(winnowed_model.layer4[1].conv1.out_channels == 501)

        self.assertTrue(winnowed_model.layer1[1].conv1[1].in_channels == 59)

        self.assertTrue(winnowed_model.layer4[0].conv1[1].in_channels == 245)

        self.assertTrue(winnowed_model.layer2[0].conv2.in_channels == 117)
        self.assertTrue(winnowed_model.layer2[0].conv2.out_channels == 128)
        self.assertTrue(winnowed_model.layer2[0].conv1.out_channels == 117)

    def test_winnow_copy_of_model_known_zero_planes_api(self):
        """ Tests winnowing resnet18 with multiple layers  with zero planes. """

        model = models.resnet18()
        model.eval()
        input_shape = (1, 3, 224, 224)
        module_zero_channels_list = create_list_of_modules_to_winnow(model)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, mod_list = winnow_model(model, input_shape,
                                                module_zero_channels_list, reshape=True,
                                                in_place=True, verbose=True)
        print("Order of modified modules after winnowing", [get_layer_name(winnowed_model, mod) for _, mod in mod_list])

        self.assertEqual(8, len(mod_list))
        set_of_modified_layers = {get_layer_name(winnowed_model, mod) for _, mod in mod_list}
        self.assertSetEqual(set_of_modified_layers, {'layer4.1.conv1',
                                                     'layer2.0.conv1',
                                                     'layer4.1.bn1',
                                                     'layer2.0.bn1',
                                                     'layer2.0.conv2',
                                                     'layer4.0.conv1',
                                                     'layer4.1.conv2',
                                                     'layer1.1.conv1'})

    def test_winnow_mobilenet_v1(self):
        """ Test winnowing mobilenet v1 model """
        mobnet_model = mobilenetv1()
        mobnet_model.eval()
        input_shape = (1, 3, 224, 224)

        list_of_modules_to_winnow = []

        module = mobnet_model.model[1][3]
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 31]
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        print("\nOrder of modules in in the API:", [get_layer_name(mobnet_model, m) for m, _ in list_of_modules_to_winnow])
        # API version 2.
        winnowed_model, mod_list = winnow_model(mobnet_model, input_shape,
                                                list_of_modules_to_winnow,
                                                reshape=True, in_place=True, verbose=True)
        print(winnowed_model)
        print("Order of modified modules after winnowing", [get_layer_name(winnowed_model, mod) for _, mod in mod_list])

        # The pointwise convolution layer getting winnowed.
        self.assertTrue(winnowed_model.model[1][3].in_channels == 25)
        self.assertTrue(winnowed_model.model[1][3].out_channels == 64)
        self.assertEqual(winnowed_model.model[1][3].weight.shape[0], 64)
        self.assertEqual(winnowed_model.model[1][3].weight.shape[1], 25)

        # The depthwise convolution layer whose in_channels, out_channels and groups must be the same
        self.assertTrue(winnowed_model.model[1][0].in_channels == 25)
        self.assertTrue(winnowed_model.model[1][0].out_channels == 25)
        self.assertTrue(winnowed_model.model[1][0].groups == 25)
        self.assertTrue(winnowed_model.model[1][0].out_channels == 25)
        self.assertEqual(winnowed_model.model[1][0].weight.shape[0], 25)

        # The very first Conv layer of the model, whose output channel must be winnowed.
        self.assertTrue(winnowed_model.model[0][0].out_channels == 25)

        # Test the forward pass
        winnowed_model(torch.rand(input_shape))

    def test_vgg_with_winnow_model_api(self):
        """ Test winnowing vgg model """
        model = models.vgg16(pretrained=False)
        input_shape = (1, 3, 224, 224)

        # For the Conv2d at VGG.features[10]
        input_channels_to_prune_1 = [15, 29, 24, 28, 33, 47]

        # For the Conv2d at VGG.features[12]
        input_channels_to_prune_2 = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]

        list_of_modules_to_winnow = [(model.features[10], input_channels_to_prune_1),
                                     (model.features[12], input_channels_to_prune_2)]

        # Call the Winnow API.
        new_model, _ = winnow_model(model, input_shape, list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        # compare zeroed out and pruned model output
        input_tensor = torch.rand(input_shape)

        validation_output = model(input_tensor)

        # validate winnowed net

        test_output = new_model(input_tensor)
        self.assertTrue(test_output.shape == validation_output.shape)

        # VGG/Sequential[features]/Conv2d[7]
        # 6 output channels will be impacted by VGG/Sequential[features]/Conv2d[10] input channel pruning
        # input channel
        self.assertEqual(new_model.features[7].weight.shape[1], 128)
        # output channel
        self.assertEqual(new_model.features[7].weight.shape[0], 122)
        self.assertEqual(new_model.features[7].in_channels, 128)
        self.assertEqual(new_model.features[7].out_channels, 122)

        # VGG/Sequential[features]/Conv2d[10]
        # 6 input channels are pruned of VGG/Sequential[features]/Conv2d[10]
        # 11 output channels will be impacted by VGG/Sequential[features]/Conv2d[12] input channel pruning
        # input channel
        self.assertEqual(new_model.features[10].weight.shape[1], 122)
        # output channel         self.assertEqual(new_model.features[10].weight.shape[0], 245)
        self.assertEqual(new_model.features[10].in_channels, 122)
        self.assertEqual(new_model.features[10].out_channels, 245)

        # VGG/Sequential[features]/Conv2d[12]
        # 11 input channels are pruned of VGG/Sequential[features]/Conv2d[12]
        # input channel
        self.assertEqual(new_model.features[12].weight.shape[1], 245)
        # output channel
        self.assertEqual(new_model.features[12].weight.shape[0], 256)
        self.assertEqual(new_model.features[12].in_channels, 245)
        self.assertEqual(new_model.features[12].out_channels, 256)

    @pytest.mark.cuda
    def test_winnowing_resnet18_on_cuda(self):
        """ Tests winnowing resnet18 with multiple layers  with zero planes. """

        model = models.resnet18()
        model.eval()
        model = model.cuda()
        input_shape = (1, 3, 224, 224)
        module_zero_channels_list = create_list_of_modules_to_winnow(model)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, mod_list = winnow_model(model, input_shape,
                                                module_zero_channels_list, in_place=True,
                                                verbose=True)
        print("Order of modified modules after winnowing", [get_layer_name(winnowed_model, mod) for _, mod in mod_list])

        # Test the forward pass
        winnowed_model(torch.rand(input_shape).cuda())

        self.assertTrue(winnowed_model.layer4[1].conv2.in_channels == 501)
        self.assertTrue(winnowed_model.layer4[1].conv2.out_channels == 512)
        self.assertTrue(winnowed_model.layer4[1].conv1.out_channels == 501)

        self.assertTrue(winnowed_model.layer1[1].conv1[1].in_channels == 59)

        self.assertTrue(winnowed_model.layer4[0].conv1[1].in_channels == 245)

        self.assertTrue(winnowed_model.layer2[0].conv2.in_channels == 117)
        self.assertTrue(winnowed_model.layer2[0].conv2.out_channels == 128)
        self.assertTrue(winnowed_model.layer2[0].conv1.out_channels == 117)

    @pytest.mark.cuda
    def test_inception_model_conv_below_split_on_cuda(self):
        """ Test winnowing inception model for a conv module below a split """
        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict['relu'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['max_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['adaptive_avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['dropout'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['flatten'] = ConnectivityType.skip
        model = models.inception_v3()
        model.eval()

        # Purposefully, model is not switched to CUDA at this time.
        input_shape = (1, 3, 299, 299)
        list_of_modules_to_winnow = []

        # Scenario: This conv layer is directly below split layer
        module = model.Mixed_5b.branch3x3dbl_1.conv
        input_channels_to_prune = [1, 3, 5, 7, 9, 15, 32, 45]
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # Call the Winnow API.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         list_of_modules_to_winnow,
                                         reshape=True, in_place=False, verbose=True)

        if winnowed_model:
            # Purposefully, the model is now switched to CUDA.
            winnowed_model = winnowed_model.cuda()
            # winnowed_model = winnowed_model.cpu()

            winnowed_model = winnowed_model.eval()

            # Test the forward pass
            winnowed_model(torch.rand(input_shape).cuda())
        self.assertEqual(0, 0)

    def test_inception_model_winnowing_multiple_modules(self):      # pylint: disable=too-many-locals
        """ Test winnowing multiple modules on inception model """
        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict['relu'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['max_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['adaptive_avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['dropout'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['flatten'] = ConnectivityType.skip
        model = models.inception_v3()
        model.eval()

        # Purposefully, model is not switched to CUDA at this time.
        input_shape = (1, 3, 299, 299)
        num_channels = 35

        test_conv0 = model.Mixed_5b.branch1x1.conv
        input_channels_to_prune0 = list(range(0, 170))

        test_conv1 = model.Mixed_5b.branch5x5_1.conv
        input_channels_to_prune1 = list(range(0, num_channels))

        test_conv2 = model.Mixed_5b.branch5x5_2.conv
        input_channels_to_prune2 = list(range(0, num_channels))

        test_conv3 = model.Mixed_5b.branch3x3dbl_1.conv
        input_channels_to_prune3 = list(range(0, num_channels))

        test_conv4 = model.Mixed_5b.branch3x3dbl_2.conv
        input_channels_to_prune4 = list(range(0, num_channels))

        test_conv5 = model.Mixed_5b.branch3x3dbl_3.conv
        input_channels_to_prune5 = list(range(0, num_channels))

        test_conv6 = model.Mixed_5b.branch_pool.conv
        input_channels_to_prune6 = list(range(0, num_channels))

        test_conv7 = model.Mixed_7c.branch_pool.conv
        input_channels_to_prune7 = list(range(1, 10))

        list_of_modules_to_winnow = [
            [test_conv0, input_channels_to_prune0],
            [test_conv1, input_channels_to_prune1],
            [test_conv2, input_channels_to_prune2],
            [test_conv3, input_channels_to_prune3],
            [test_conv4, input_channels_to_prune4],
            [test_conv5, input_channels_to_prune5],
            [test_conv6, input_channels_to_prune6],
            [test_conv7, input_channels_to_prune7]
        ]

        # Call the Winnow API.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         list_of_modules_to_winnow,
                                         reshape=True, in_place=False, verbose=True)

        if winnowed_model:
            winnowed_model = winnowed_model.cpu()

            winnowed_model = winnowed_model.eval()

            # Test the forward pass
            winnowed_model(torch.rand(input_shape))
        self.assertEqual(0, 0)

    @pytest.mark.cuda
    def test_inception_model_conv_has_upstream_concat(self):
        """
        Test winnow on inception model for conv module with concat above
        """
        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict['relu'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['max_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['adaptive_avg_pool2d'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['dropout'] = ConnectivityType.direct
        OpConnectivity.pytorch_dict['flatten'] = ConnectivityType.skip
        model = models.inception_v3()
        model.eval()

        # Purposefully, model is not switched to CUDA at this time.
        input_shape = (1, 3, 299, 299)
        list_of_modules_to_winnow = []

        # Scenario: This conv layer is directly below split layer
        module = model.Mixed_6d.branch_pool.conv
        input_channels_to_prune = [15, 33, 45, 57, 99, 115, 132, 191]
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # Call the Winnow API.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         list_of_modules_to_winnow,
                                         reshape=False, in_place=False, verbose=True)

        if winnowed_model is not None:
            # Purposefully, the model is now switched to CUDA.
            winnowed_model = winnowed_model.cuda()
            winnowed_model = winnowed_model.eval()

            # Test the forward pass
            winnowed_model(torch.rand(input_shape).cuda())
        else:
            # If reshape is set to False, no model will be returned.
            print("\nNo model is returned as expected.")
        self.assertEqual(0, 0)

    def test_winnowing_all_channels_in_a_module_single_residual(self):
        """ Tests the simple single residual model. """
        model = SingleResidual()

        # Test forward pass on the copied model before zering out channels of layers.
        input_shape = (1, 3, 32, 32)

        # For conv3 layer, winnow all channels.
        list_of_modules_to_winnow = []
        module = model.conv3
        input_channels_to_prune = list(range(0, 64))
        print(input_channels_to_prune)
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # Call the Winnow API.
        # This should result in an error being logged explaining winnowing all channels in a module
        # is not allowed.
        with self.assertRaises(ValueError):
            _, _ = winnow_model(model, input_shape,
                                list_of_modules_to_winnow,
                                reshape=True, in_place=False, verbose=True)

    @unittest.skip
    def test_bn_inception_model(self):
        """
        This BN Inception is a model used by the Avante team (Korea). This test was used to recreate the issue
        reported. Not deleting this test. As Winnowing is enhanced to handle more scenarios, this test would be useful.
        """
        model = bninception()

        # Test forward pass on the copied model before zeroing out channels of layers.
        input_shape = (1, 3, 448, 448)
        inp = torch.rand(input_shape)
        #
        model.eval()
        _ = model(inp)

        list_of_modules_to_winnow = []
        module = model.stage1[0]
        input_channels_to_prune = [5]

        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)
        _, mod_list = winnow_model(model, input_shape,
                                   list_of_modules_to_winnow,
                                   in_place=True, verbose=True)
        print(mod_list)
        self.assertEqual(0, 0)

    def test_resnet18_downsample_winnowing(self):
        """ Tests the winnow_model() API for winnowing resnet18 with
        multiple layers  with zero planes."""

        model = models.resnet18()
        input_shape = (1, 3, 224, 224)

        test_conv1 = model.layer2[0].conv1
        input_channels_to_prune1 = list(range(8, 10))

        test_conv2 = model.layer2[0].downsample[0]
        input_channels_to_prune2 = list(range(1, 32))

        list_of_modules_to_winnow = [
            [test_conv1, input_channels_to_prune1],
            [test_conv2, input_channels_to_prune2]
        ]

        (new_model, module_list) = winnow_model(model, input_shape,
                                                list_of_modules_to_winnow,
                                                verbose=True)

        # compare zeroed out and pruned model output

        input_tensor = torch.rand(input_shape)
        _ = model(input_tensor)

        # validate winnowed net
        _ = new_model(input_tensor)

        print("\n\nmodule_list: %s" % module_list)
        print("\nWeight shape of the original Conv2d layer: ", model.layer2[0])
        print("\nWeight shape of the downsampled Conv2d layer: ", new_model.layer2[0])
        self.assertEqual(0, 0)

    def test_resnet18_winnow_first_module(self):
        """ Tests for asserting on winnowing input to first module, using resnet18 """
        model = models.resnet18()
        input_shape = (1, 3, 224, 224)

        # model.conv1 is first module for resnet18
        first_module = model.conv1
        input_channels_to_prune = [1]

        list_of_modules_to_winnow = [
            [first_module, input_channels_to_prune]
        ]

        with self.assertRaises(NotImplementedError):
            _, _ = winnow_model(model,
                                input_shape,
                                list_of_modules_to_winnow,
                                verbose=True)

    def test_resnet18_no_modules_winnowed(self):
        """ Tests for returning unchanged model when no channels are to be winnowed """
        model = models.resnet18()
        input_shape = (1, 3, 224, 224)

        list_of_modules_to_winnow = [
        ]

        # Below test assumes in_place = False.  If in_place is set to True, this test will always pass
        # regardless of whether the module was changed in any way or not.
        (new_model, module_list) = winnow_model(model,
                                                input_shape,
                                                list_of_modules_to_winnow,
                                                verbose=False)

        self.assertTrue(repr(model), repr(new_model))
        self.assertEqual(module_list, None)

    def test_resnet18_winnow_non_conv2d_module(self):
        """ Tests for asserting on attempting to winnow non conv2d module """
        model = models.resnet18()
        input_shape = (1, 3, 224, 224)

        fc_module = model.fc
        input_channels_to_prune = [1]

        list_of_modules_to_winnow = [
            [fc_module, input_channels_to_prune]
        ]

        with self.assertRaises(NotImplementedError):
            _, _ = winnow_model(model,
                                input_shape,
                                list_of_modules_to_winnow,
                                verbose=False)

    def test_winnow_mobilenet_v2(self):
        """ Tests basic winnowing with mobilenet_v2 """
        mobnet_model = mobilenetv2()
        mobnet_model.eval()
        input_shape = (1, 3, 224, 224)

        list_of_modules_to_winnow = []
        module = mobnet_model.features[1].conv[3]
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 31]
        module_mask_pair = (module, input_channels_to_prune)
        list_of_modules_to_winnow.append(module_mask_pair)

        # API version 2.
        winnowed_model, _ = winnow_model(mobnet_model, input_shape,
                                         list_of_modules_to_winnow,
                                         reshape=True, in_place=True, verbose=False)

        # The pointwise convolution layer getting winnowed.
        self.assertTrue(winnowed_model.features[1].conv[3].in_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[3].out_channels == 16)
        self.assertEqual(winnowed_model.features[1].conv[3].weight.shape[0], 16)
        self.assertEqual(winnowed_model.features[1].conv[3].weight.shape[1], 25)

        # The depthwise convolution layer whose in_channels, out_channels and groups must be the same
        self.assertTrue(winnowed_model.features[1].conv[0].in_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].out_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].groups == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].out_channels == 25)
        self.assertEqual(winnowed_model.features[1].conv[0].weight.shape[0], 25)

        # The very first Conv layer of the model, whose output channel must be winnowed.
        self.assertTrue(winnowed_model.features[0][0].out_channels == 25)

        # Test the forward pass
        winnowed_model(torch.rand(input_shape))

    def test_winnow_mobilenet_v2_with_zero_planes(self):
        """" Tests winnowing with mobilenet_v2 with zero planes set explicitly"""
        mobnet_model = mobilenetv2()
        mobnet_model.eval()
        input_shape = (1, 3, 224, 224)

        module = mobnet_model.features[1].conv[3]
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 31]
        zero_out_input_channels(module, input_channels_to_prune)

        # API version 2.
        list_of_modules_to_winnow = search_for_zero_planes(mobnet_model)
        winnowed_model, _ = winnow_model(mobnet_model, input_shape,
                                         list_of_modules_to_winnow,
                                         reshape=True, in_place=True, verbose=False)

        # The pointwise convolution layer getting winnowed.
        self.assertTrue(winnowed_model.features[1].conv[3].in_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[3].out_channels == 16)
        self.assertEqual(winnowed_model.features[1].conv[3].weight.shape[0], 16)
        self.assertEqual(winnowed_model.features[1].conv[3].weight.shape[1], 25)

        # The depthwise convolution layer whose in_channels, out_channels and groups must be the same
        self.assertTrue(winnowed_model.features[1].conv[0].in_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].out_channels == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].groups == 25)
        self.assertTrue(winnowed_model.features[1].conv[0].out_channels == 25)
        self.assertEqual(winnowed_model.features[1].conv[0].weight.shape[0], 25)

        # The very first Conv layer of the model, whose output channel must be winnowed.
        self.assertTrue(winnowed_model.features[0][0].out_channels == 25)

        # Test the forward pass
        winnowed_model(torch.rand(input_shape))

    def test_winnow_modulelist(self):
        """ Test winnowing a model with ops defined using ModuleList """
        model = ModuleListModel()
        model.eval()
        input_shape = (1, 3, 8, 8)

        module = model.mod_list[2]
        input_channels_to_prune = [2, 3, 5, 7, 11, 13]
        list_of_modules_to_winnow = [(module, input_channels_to_prune)]
        _, _ = winnow_model(model, input_shape,
                            list_of_modules_to_winnow,
                            reshape=True, in_place=True, verbose=False)
        self.assertEqual(10, model.mod_list[4].weight.shape[0])
        self.assertEqual(10, model.seq_list[2].weight.shape[0])
        self.assertEqual(10, model.mod_list[2].weight.shape[1])

    def test_convTranspose2d(self):
        """ Test model with ConvTranspose2d """
        class ConvTranspose(nn.Module):
            """ Conv Transpose model """
            def __init__(self):
                super(ConvTranspose, self).__init__()

                self.conv1 = nn.Conv2d(1, 4, 3, stride=2, padding=1)
                self.conv2 = nn.Conv2d(4, 2, 3, stride=2, padding=1)
                self.deconv = nn.ConvTranspose2d(2, 16, 3, stride=2)
                self.fc = nn.Linear((16*15*15), 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.deconv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = ConvTranspose()
        model.eval()

        # ONNX export.
        input_shape = (1, 1, 28, 28)
        input_tensor = torch.rand(input_shape)

        graph = ConnectedGraph(model, (input_tensor,))
        op = graph.get_op_from_module_name('ConvTranspose.deconv')
        self.assertTrue(torch.nn.ConvTranspose2d, type(op.get_module()))
        self.assertTrue(op.dotted_name == 'ConvTranspose.deconv')

    def test_winnowing_parallel_convs(self):
        """ Test winnowing a case where convs in parallel branches are winnowed with certain intersecting channels
        winnowed """
        model = SingleResidual()
        model.eval()
        input_shape = [1, 3, 32, 32]
        input_tensor = torch.rand(input_shape)
        validation_output = model(input_tensor)

        list_of_modules_to_winnow = [(model.conv2, [20, 30]), (model.conv4, [20])]

        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        # compare model output
        test_output = new_model(input_tensor)
        self.assertTrue(test_output.shape == validation_output.shape)
        # Since the model classifies 10 classes
        self.assertTrue(test_output.size()[-1] == 10)

        # In the winnowed model, conv2 has in_channels = 14, out_channels = 16
        self.assertTrue(isinstance(new_model.conv2[0], DownsampleLayer))
        self.assertTrue(new_model.conv2[1].in_channels == 30)
        self.assertTrue(new_model.conv2[1].out_channels == 16)

        # In the winnowed model, conv4 has in_channels = 15, out_channels = 8
        self.assertTrue(new_model.conv4.in_channels == 31)
        self.assertTrue(new_model.conv4.out_channels == 8)

        # In the winnowed model, conv1 has in_channels = 3, out_channels = 62 (impacted by conv2 and conv4 pruning)
        self.assertTrue(new_model.conv1.in_channels == 3)
        self.assertTrue(new_model.conv1.out_channels == 31)

    # pylint: disable=too-many-locals
    # pylint: disable=protected-access
    def test_winnowing_with_downsample(self):
        """ Test winnowing a model that has a downsample layer already inserted from a previous winnow pass """
        model = SingleResidual()
        model.eval()
        input_shape = [1, 3, 32, 32]
        input_tensor = torch.rand(input_shape)

        list_of_modules_to_winnow = [(model.conv2, [20, 30])]
        new_model, _ = winnow_model(model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        list_of_modules_to_winnow = [(new_model.conv4, [20])]
        new_model, _ = winnow_model(new_model, input_shape,
                                    list_of_modules_to_winnow,
                                    in_place=True, verbose=True)

        conv_2 = new_model.conv2[1]
        conv_4 = new_model.conv4[1]
        self.assertTrue(isinstance(new_model.conv2[0], DownsampleLayer))
        self.assertTrue(isinstance(new_model.conv4[0], DownsampleLayer))
        self.assertEqual(conv_2.in_channels, 30)
        self.assertEqual(conv_4.in_channels, 31)

        mask_winnower = MaskPropagationWinnower(new_model, input_shape, in_place=True)
        conv_2_conn_graph_op = mask_winnower._mask_propagator._graph._module_to_op_dict[conv_2]
        conv_4_conn_graph_op = mask_winnower._mask_propagator._graph._module_to_op_dict[conv_4]
        downsample_1_conn_graph_op = conv_2_conn_graph_op.inputs[0].producer
        downsample_2_conn_graph_op = conv_4_conn_graph_op.inputs[0].producer
        downsample_1_mask = mask_winnower._mask_propagator.op_to_mask_dict[downsample_1_conn_graph_op]
        downsample_2_mask = mask_winnower._mask_propagator.op_to_mask_dict[downsample_2_conn_graph_op]
        self.assertEqual(downsample_1_mask._num_in_channels, 32)
        self.assertEqual(downsample_1_mask._num_out_channels, 30)
        self.assertEqual(downsample_2_mask._num_in_channels, 32)
        self.assertEqual(downsample_2_mask._num_out_channels, 31)

        _ = new_model(input_tensor)


def hook_verifier(_a, _b, _c):
    """ Print to verify hook """
    print("Backward Pass Verified.")


def create_list_of_modules_to_winnow(model):
    """ Function to reuse in multiple tests """

    list_of_modules_to_winnow = []

    module = model.layer4[1].conv2
    input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]

    # To test winnowing the first module of a model (which is not allowed), uncomment the following 2 lines.
    # module = model.conv1
    # input_channels_to_prune = [2]

    module_mask_pair = (module, input_channels_to_prune)
    list_of_modules_to_winnow.append(module_mask_pair)

    module = model.layer1[1].conv1
    input_channels_to_prune = [55, 56, 57, 58, 59]
    module_mask_pair = (module, input_channels_to_prune)
    list_of_modules_to_winnow.append(module_mask_pair)

    module = model.layer4[0].conv1
    input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
    module_mask_pair = (module, input_channels_to_prune)
    list_of_modules_to_winnow.append(module_mask_pair)

    module = model.layer2[0].conv2
    input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
    module_mask_pair = (module, input_channels_to_prune)
    list_of_modules_to_winnow.append(module_mask_pair)

    return list_of_modules_to_winnow


def zero_out_select_input_channels(model):
    """ Set specified input channels to have zeroed out parameters """
    input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
    zero_out_input_channels(model.layer4[1].conv2, input_channels_to_prune)

    input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
    zero_out_input_channels(model.layer4[0].conv1, input_channels_to_prune)

    input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
    zero_out_input_channels(model.layer3[1].conv2, input_channels_to_prune)

    input_channels_to_prune = [33, 44, 55]
    zero_out_input_channels(model.layer2[1].conv2, input_channels_to_prune)

    input_channels_to_prune = [11, 12, 13, 14, 15]
    zero_out_input_channels(model.layer2[0].conv2, input_channels_to_prune)

    input_channels_to_prune = [55, 56, 57, 58, 59]
    zero_out_input_channels(model.layer1[1].conv1, input_channels_to_prune)

    input_channels_to_prune = [42, 44, 46]
    zero_out_input_channels(model.layer1[0].conv2, input_channels_to_prune)

    return model
