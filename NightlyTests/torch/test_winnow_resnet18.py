# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Winnowing acceptance tests for ResNet models."""

import unittest
import torch
from torchvision import models
from aimet_torch.winnow.winnow import winnow_model
from aimet_torch.common.winnow.winnow_utils import OpConnectivity, ConnectivityType


class WinnowResNet18Test(unittest.TestCase):
    """Acceptance tests related to winnowing ResNet models."""

    def test_winnowing_multiple_zeroed_resnet34(self):
        """Tests winnowing resnet18 with multiple layers with zero planes."""

        model = models.resnet34(pretrained=False)
        model.eval()

        # Test forward pass on the copied model before zeroing out channels in any layer..
        input_shape = [1, 3, 224, 224]

        list_of_modules_to_winnow = []

        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[0].conv1, input_channels_to_prune)
        )

        input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
        list_of_modules_to_winnow.append(
            (model.layer3[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [33, 44, 55]
        list_of_modules_to_winnow.append(
            (model.layer2[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [11, 12, 13, 14, 15]
        list_of_modules_to_winnow.append(
            (model.layer2[0].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [55, 56, 57, 58, 59]
        list_of_modules_to_winnow.append(
            (model.layer1[1].conv1, input_channels_to_prune)
        )

        input_channels_to_prune = [42, 44, 46]
        list_of_modules_to_winnow.append(
            (model.layer1[0].conv2, input_channels_to_prune)
        )

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        # compare zeroed out and pruned model output
        # use double precision for lower absolute error
        input_tensor = torch.rand(input_shape).double()
        model.double()
        model.eval()
        validation_output = model(input_tensor)

        # validate winnowed net
        new_model.double()
        new_model.eval()
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)

        # layer1.0.conv2 input channels pruned from 64 --> 61
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[0].conv2.in_channels, 61)
        self.assertEqual(list(new_model.layer1[0].conv2.weight.shape), [64, 61, 3, 3])
        self.assertEqual(new_model.layer1[0].conv1.out_channels, 61)

        # layer1.1.conv1 output channels pruned from 64 --> 59
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[1].conv1[1].in_channels, 59)
        self.assertEqual(
            list(new_model.layer1[1].conv1[1].weight.shape), [64, 59, 3, 3]
        )

        # layer2.0.conv2 input channels pruned from 128 --> 123
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[0].conv2.in_channels, 123)
        self.assertEqual(list(new_model.layer2[0].conv2.weight.shape), [128, 123, 3, 3])
        self.assertEqual(new_model.layer2[0].conv1.out_channels, 123)

        # layer2.1.conv2 input channels pruned from 128 --> 125
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[1].conv2.in_channels, 125)
        self.assertEqual(list(new_model.layer2[1].conv2.weight.shape), [128, 125, 3, 3])
        self.assertEqual(new_model.layer2[1].conv1.out_channels, 125)

        # layer3.1.conv2 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer3[1].conv2.in_channels, 245)
        self.assertEqual(list(new_model.layer3[1].conv2.weight.shape), [256, 245, 3, 3])
        self.assertEqual(new_model.layer3[1].conv1.out_channels, 245)

        # layer4.0.conv1 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[0].conv1[1].in_channels, 245)
        self.assertEqual(
            list(new_model.layer4[0].conv1[1].weight.shape), [512, 245, 3, 3]
        )

        # layer4.1.conv2 input channels pruned from 512 --> 501
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[1].conv2.in_channels, 501)
        self.assertEqual(list(new_model.layer4[1].conv2.weight.shape), [512, 501, 3, 3])
        self.assertEqual(new_model.layer4[1].conv1.out_channels, 501)

    def test_winnowing_multiple_zeroed_resnet50(self):
        """Tests winnowing resnet18 with multiple layers  with zero planes."""

        model = models.resnet50(pretrained=False)
        model.eval()

        input_shape = [1, 3, 224, 224]
        list_of_modules_to_winnow = []

        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[0].conv1, input_channels_to_prune)
        )

        input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
        list_of_modules_to_winnow.append(
            (model.layer3[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [33, 44, 55]
        list_of_modules_to_winnow.append(
            (model.layer2[1].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [11, 12, 13, 14, 15]
        list_of_modules_to_winnow.append(
            (model.layer2[0].conv2, input_channels_to_prune)
        )

        input_channels_to_prune = [55, 56, 57, 58, 59]
        list_of_modules_to_winnow.append(
            (model.layer1[1].conv1, input_channels_to_prune)
        )

        input_channels_to_prune = [42, 44, 46]
        list_of_modules_to_winnow.append(
            (model.layer1[0].conv2, input_channels_to_prune)
        )

        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        # compare zeroed out and pruned model output
        # use double precision for lower absolute error
        input_tensor = torch.rand(input_shape).double()
        model.double()
        model.eval()
        validation_output = model(input_tensor)

        # validate winnowed net
        new_model.double()
        new_model.eval()
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)

        # layer1.0.conv2 input channels pruned from 64 --> 61
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[0].conv2.in_channels, 61)
        self.assertEqual(list(new_model.layer1[0].conv2.weight.shape), [64, 61, 3, 3])
        self.assertEqual(new_model.layer1[0].conv1.out_channels, 61)

        # layer1.1.conv1 output channels pruned from 64 --> 59
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[1].conv1[1].in_channels, 251)
        self.assertEqual(
            list(new_model.layer1[1].conv1[1].weight.shape), [64, 251, 1, 1]
        )

        # layer2.0.conv2 input channels pruned from 128 --> 123
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[0].conv2.in_channels, 123)
        self.assertEqual(list(new_model.layer2[0].conv2.weight.shape), [128, 123, 3, 3])
        self.assertEqual(new_model.layer2[0].conv1.out_channels, 123)

        # layer2.1.conv2 input channels pruned from 128 --> 125
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[1].conv2.in_channels, 125)
        self.assertEqual(list(new_model.layer2[1].conv2.weight.shape), [128, 125, 3, 3])
        self.assertEqual(new_model.layer2[1].conv1.out_channels, 125)

        # layer3.1.conv2 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer3[1].conv2.in_channels, 245)
        self.assertEqual(list(new_model.layer3[1].conv2.weight.shape), [256, 245, 3, 3])
        self.assertEqual(new_model.layer3[1].conv1.out_channels, 245)

        # layer4.0.conv1 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[0].conv1[1].in_channels, 1013)
        self.assertEqual(
            list(new_model.layer4[0].conv1[1].weight.shape), [512, 1013, 1, 1]
        )

        # layer4.1.conv2 input channels pruned from 512 --> 501
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[1].conv2.in_channels, 501)
        self.assertEqual(list(new_model.layer4[1].conv2.weight.shape), [512, 501, 3, 3])
        self.assertEqual(new_model.layer4[1].conv1.out_channels, 501)

    def test_winnowing_multiple_zeroed_resnet101(self):
        """Tests winnowing resnet18 with multiple layers  with zero planes."""

        model = models.resnet101(pretrained=False)
        model.eval()

        input_shape = [1, 3, 224, 224]

        list_of_modules_to_winnow = []

        # For layer4[1].conv2 layer, zero out input channels 5, 9, 14, 18, 23, 27, 32, 36, 41, 44, 54
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[1].conv2, input_channels_to_prune)
        )

        # For layer4[0].conv1 layer, zero out input channels 5, 9, 14, 18, 23, 27, 32, 36, 41, 44, 54
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[0].conv1, input_channels_to_prune)
        )

        # For layer3[1].conv2 layer, zero out input channels 15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9
        input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
        list_of_modules_to_winnow.append(
            (model.layer3[1].conv2, input_channels_to_prune)
        )

        # For layer2[1].conv2 layer, zero out input channels 33, 44, 55
        input_channels_to_prune = [33, 44, 55]
        list_of_modules_to_winnow.append(
            (model.layer2[1].conv2, input_channels_to_prune)
        )

        # For layer2[0].conv2 layer, zero out input channels 1, 12, 13, 14, 15
        input_channels_to_prune = [11, 12, 13, 14, 15]
        list_of_modules_to_winnow.append(
            (model.layer2[0].conv2, input_channels_to_prune)
        )

        # For layer1[1].conv1 layer, zero out input channels 55, 56, 57, 58, 59
        input_channels_to_prune = [55, 56, 57, 58, 59]
        list_of_modules_to_winnow.append(
            (model.layer1[1].conv1, input_channels_to_prune)
        )

        # For layer1[0].conv2 layer, zero out input channels 42, 44, 36
        input_channels_to_prune = [42, 44, 46]
        list_of_modules_to_winnow.append(
            (model.layer1[0].conv2, input_channels_to_prune)
        )

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        input_tensor = torch.rand(input_shape).double()
        model.double()
        model.eval()
        validation_output = model(input_tensor)

        # validate winnowed net
        new_model.double()
        new_model.eval()
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)

        # layer1.0.conv2 input channels pruned from 64 --> 61
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[0].conv2.in_channels, 61)
        self.assertEqual(list(new_model.layer1[0].conv2.weight.shape), [64, 61, 3, 3])
        self.assertEqual(new_model.layer1[0].conv1.out_channels, 61)

        # layer1.1.conv1 output channels pruned from 64 --> 59
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[1].conv1[1].in_channels, 251)
        self.assertEqual(
            list(new_model.layer1[1].conv1[1].weight.shape), [64, 251, 1, 1]
        )

        # layer2.0.conv2 input channels pruned from 128 --> 123
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[0].conv2.in_channels, 123)
        self.assertEqual(list(new_model.layer2[0].conv2.weight.shape), [128, 123, 3, 3])
        self.assertEqual(new_model.layer2[0].conv1.out_channels, 123)

        # layer2.1.conv2 input channels pruned from 128 --> 125
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[1].conv2.in_channels, 125)
        self.assertEqual(list(new_model.layer2[1].conv2.weight.shape), [128, 125, 3, 3])
        self.assertEqual(new_model.layer2[1].conv1.out_channels, 125)

        # layer3.1.conv2 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer3[1].conv2.in_channels, 245)
        self.assertEqual(list(new_model.layer3[1].conv2.weight.shape), [256, 245, 3, 3])
        self.assertEqual(new_model.layer3[1].conv1.out_channels, 245)

        # layer4.0.conv1 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[0].conv1[1].in_channels, 1013)
        self.assertEqual(
            list(new_model.layer4[0].conv1[1].weight.shape), [512, 1013, 1, 1]
        )

        # layer4.1.conv2 input channels pruned from 512 --> 501
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[1].conv2.in_channels, 501)
        self.assertEqual(list(new_model.layer4[1].conv2.weight.shape), [512, 501, 3, 3])
        self.assertEqual(new_model.layer4[1].conv1.out_channels, 501)

    def test_winnowing_multiple_zeroed_resnet152(self):
        """Tests winnowing resnet18 with multiple layers  with zero planes."""

        model = models.resnet152(pretrained=False)
        model.eval()

        # Test forward pass on the copied model before zeroing out channels in any layer..
        input_shape = [1, 3, 224, 224]

        list_of_modules_to_winnow = []

        # For layer4[1].conv2 layer, zero out input channels 5, 9, 14, 18, 23, 27, 32, 36, 41, 44, 54
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[1].conv2, input_channels_to_prune)
        )

        # For layer4[0].conv1 layer, zero out input channels 5, 9, 14, 18, 23, 27, 32, 36, 41, 44, 54
        input_channels_to_prune = [5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 54]
        list_of_modules_to_winnow.append(
            (model.layer4[0].conv1, input_channels_to_prune)
        )

        # For layer3[1].conv2 layer, zero out input channels 15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9
        input_channels_to_prune = [15, 29, 24, 28, 33, 47, 2, 3, 1, 5, 9]
        list_of_modules_to_winnow.append(
            (model.layer3[1].conv2, input_channels_to_prune)
        )

        # For layer2[1].conv2 layer, zero out input channels 33, 44, 55
        input_channels_to_prune = [33, 44, 55]
        list_of_modules_to_winnow.append(
            (model.layer2[1].conv2, input_channels_to_prune)
        )

        # For layer2[0].conv2 layer, zero out input channels 1, 12, 13, 14, 15
        input_channels_to_prune = [11, 12, 13, 14, 15]
        list_of_modules_to_winnow.append(
            (model.layer2[0].conv2, input_channels_to_prune)
        )

        # For layer1[1].conv1 layer, zero out input channels 55, 56, 57, 58, 59
        input_channels_to_prune = [55, 56, 57, 58, 59]
        list_of_modules_to_winnow.append(
            (model.layer1[1].conv1, input_channels_to_prune)
        )

        # For layer1[0].conv2 layer, zero out input channels 42, 44, 36
        input_channels_to_prune = [42, 44, 46]
        list_of_modules_to_winnow.append(
            (model.layer1[0].conv2, input_channels_to_prune)
        )

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        # compare zeroed out and pruned model output:
        # use double precision for lower absolute error
        input_tensor = torch.rand(input_shape).double()
        model.double()
        model.eval()
        validation_output = model(input_tensor)

        # validate winnowed net
        new_model.double()
        new_model.eval()
        test_output = new_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)

        # layer1.0.conv2 input channels pruned from 64 --> 61
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[0].conv2.in_channels, 61)
        self.assertEqual(list(new_model.layer1[0].conv2.weight.shape), [64, 61, 3, 3])
        self.assertEqual(new_model.layer1[0].conv1.out_channels, 61)

        # layer1.1.conv1 output channels pruned from 64 --> 59
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer1[1].conv1[1].in_channels, 251)
        self.assertEqual(
            list(new_model.layer1[1].conv1[1].weight.shape), [64, 251, 1, 1]
        )

        # layer2.0.conv2 input channels pruned from 128 --> 123
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[0].conv2.in_channels, 123)
        self.assertEqual(list(new_model.layer2[0].conv2.weight.shape), [128, 123, 3, 3])
        self.assertEqual(new_model.layer2[0].conv1.out_channels, 123)

        # layer2.1.conv2 input channels pruned from 128 --> 125
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer2[1].conv2.in_channels, 125)
        self.assertEqual(list(new_model.layer2[1].conv2.weight.shape), [128, 125, 3, 3])
        self.assertEqual(new_model.layer2[1].conv1.out_channels, 125)

        # layer3.1.conv2 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer3[1].conv2.in_channels, 245)
        self.assertEqual(list(new_model.layer3[1].conv2.weight.shape), [256, 245, 3, 3])
        self.assertEqual(new_model.layer3[1].conv1.out_channels, 245)

        # layer4.0.conv1 input channels pruned from 256 --> 245
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[0].conv1[1].in_channels, 1013)
        self.assertEqual(
            list(new_model.layer4[0].conv1[1].weight.shape), [512, 1013, 1, 1]
        )

        # layer4.1.conv2 input channels pruned from 512 --> 501
        # weight (Tensor) : [out_channels, in_channels, kernel_size, kernel_size]
        self.assertEqual(new_model.layer4[1].conv2.in_channels, 501)
        self.assertEqual(list(new_model.layer4[1].conv2.weight.shape), [512, 501, 3, 3])
        self.assertEqual(new_model.layer4[1].conv1.out_channels, 501)

    def test_inception_model_conv_below_conv(self):
        """Test winnowing inception model conv below conv"""

        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict["relu"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["max_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["adaptive_avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["dropout"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["flatten"] = ConnectivityType.skip

        model = models.Inception3()
        model.eval()
        input_shape = [1, 3, 299, 299]
        input_channels_to_prune = [1, 3, 5, 7, 9, 15, 32, 45]

        list_of_modules_to_winnow = [
            (model.Mixed_5b.branch3x3dbl_2.conv, input_channels_to_prune)
        ]

        print(model.Mixed_5b.branch3x3dbl_1.conv.out_channels)

        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        self.assertEqual(new_model.Mixed_5b.branch3x3dbl_1.conv.out_channels, 56)
        self.assertEqual(
            list(new_model.Mixed_5b.branch3x3dbl_1.conv.weight.shape), [56, 192, 1, 1]
        )
        del model
        del new_model

    def test_inception_model_conv_below_split(self):
        """Test winnowing inception model with conv below split"""

        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict["relu"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["max_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["adaptive_avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["dropout"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["flatten"] = ConnectivityType.skip

        model = models.Inception3()
        model.eval()
        input_shape = [1, 3, 299, 299]
        input_channels_to_prune = [1, 3, 5, 7, 9, 15, 32, 45]

        list_of_modules_to_winnow = [
            (model.Mixed_5b.branch3x3dbl_1.conv, input_channels_to_prune)
        ]

        print(model.Mixed_5b.branch3x3dbl_1.conv.out_channels)

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        del model
        del new_model

        model = models.Inception3()
        model.eval()
        input_shape = [1, 3, 299, 299]
        input_channels_to_prune = [1, 3, 5, 7, 9, 15, 32, 45]
        list_of_modules_to_winnow = [
            (model.Mixed_5b.branch1x1.conv, input_channels_to_prune)
        ]
        print(model.Mixed_5b.branch3x3dbl_1.conv.out_channels)

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        del model
        del new_model
        self.assertEqual(0, 0)

    def test_inception_model_conv_below_avgpool(self):
        """Test winnowing inception model with conv below avgpool"""
        # These modules are included as a hack to allow tests using inception model to pass,
        # as the model uses functionals instead of modules.
        OpConnectivity.pytorch_dict["relu"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["max_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["adaptive_avg_pool2d"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["dropout"] = ConnectivityType.direct
        OpConnectivity.pytorch_dict["flatten"] = ConnectivityType.skip
        model = models.Inception3()
        model.eval()
        input_shape = [1, 3, 299, 299]
        input_channels_to_prune = [1, 3, 5, 7, 9, 15, 32, 45]
        list_of_modules_to_winnow = [
            (model.Mixed_5b.branch_pool.conv, input_channels_to_prune)
        ]
        print(model.Mixed_5b.branch_pool.conv)
        print(
            model.Mixed_5b.branch_pool.conv.out_channels,
            model.Mixed_5b.branch_pool.conv.in_channels,
        )

        # Call the Winnow API.
        new_model, _ = winnow_model(
            model,
            input_shape,
            list_of_modules_to_winnow,
            reshape=True,
            in_place=False,
            verbose=True,
        )

        self.assertEqual(new_model.Mixed_5b.branch_pool.conv[1].out_channels, 32)
        self.assertEqual(
            list(new_model.Mixed_5b.branch_pool.conv[1].weight.shape), [32, 184, 1, 1]
        )
        del model
        del new_model
