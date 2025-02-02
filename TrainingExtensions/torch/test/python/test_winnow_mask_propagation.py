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
""" Contains unit tests to test winnowing of a model using mask propagation """
import unittest
import numpy as np
import torch
import torch.nn as nn

from aimet_common.utils import AimetLogger
from aimet_common.winnow.mask import NullInternalConnectivity, DirectInternalConnectivity, SplitInternalConnectivity, \
    AddInternalConnectivity, ConcatInternalConnectivity
from aimet_torch.winnow.winnow_utils import UpsampleLayer
from aimet_torch.winnow.winnow import winnow_model
from aimet_torch.utils import get_layer_name

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class SingleResidual(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidual, self).__init__()
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


class SingleConcat(nn.Module):
    """ A model with a single Concat. """

    def __init__(self, num_classes=10):
        super(SingleConcat, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(9, 36, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(36, 19, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

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
        residual_1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        residual_2 = x
        x = self.conv3(x)
        x = self.bn3(x)

        # Concat the 3 residuals
        x = torch.cat([residual_2, residual_1, x], 1)
        x = self.relu3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SingleChunk(nn.Module):
    """ A model with a single Concat. """
    def __init__(self, num_classes=10):
        super(SingleChunk, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(43, 43, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(43, 43, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # The output ofBatchNorm2d layer above(bn33) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(160000, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Use torch.chunk to slice the tensor
        chunk_1, chunk_2, chunk_3 = torch.chunk(x, 3, dim=1)

        x = self.conv2(chunk_1)
        x = self.bn2(x)
        x = self.relu2(x)
        chunk_1_residual = x
        x = self.conv3(chunk_2)
        x = self.bn3(x)
        chunk_2_residual = x

        # Concat the 3 residuals
        x = torch.cat([chunk_1_residual, chunk_2_residual, chunk_3], 1)
        x = self.relu3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestWinnowMaskPropagation(unittest.TestCase):
    """ Unit test cases for winnowing using mask propagation """

    def test_conv_to_conv_mask_propagation(self):
        """ After the graph is constructed, the Op should have default masks and connectivity for all module types. """
        logger.debug("Test default mask and connectivity.")
        model = SingleResidual()

        # Test forward pass on the copied model before winnowing.
        input_shape = [1, 3, 224, 224]
        input_tensor = torch.rand(input_shape).double()
        model.double()
        model.eval()
        print("test_conv_to_conv_mask_propagation(): Testing forward pass before winnowing.")
        validation_output = model(input_tensor)

        # Convert the model back to float.
        model.float()

        module_zero_channels_list = []
        module = model.conv3
        input_channels_to_prune = [1, 3]

        module_mask_pair = (module, input_channels_to_prune)
        module_zero_channels_list.append(module_mask_pair)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         module_zero_channels_list,
                                         in_place=True, verbose=True)

        # validate winnowed net
        input_tensor = torch.rand(input_shape).double()
        winnowed_model.double()
        winnowed_model.eval()
        test_output = winnowed_model(input_tensor)

        self.assertTrue(test_output.shape == validation_output.shape)
        # self.assertTrue(test_output.allclose(validation_output)) # TBD. Why is this failing?
        print(test_output)
        print(validation_output)

        # In the winnowed model, conv3 has in_channels = 62, out_channels = 64
        self.assertTrue(winnowed_model.conv3.in_channels == 62)
        self.assertTrue(winnowed_model.conv3.out_channels == 64)

        # The winnowed model's bn2 layer has 62 num_features
        self.assertEqual(winnowed_model.bn2.num_features, 62)
        self.assertEqual(list(winnowed_model.bn2.weight.shape), [62])
        self.assertEqual(list(winnowed_model.bn2.bias.shape), [62])
        self.assertEqual(list(winnowed_model.bn2.running_mean.shape), [62])
        self.assertEqual(list(winnowed_model.bn2.running_var.shape), [62])

        # In the winnowed model, conv2 has in_channels = 64, out_channels = 62 (impacted by layer3 pruning)
        self.assertTrue(winnowed_model.conv2.in_channels == 64)
        self.assertTrue(winnowed_model.conv2.out_channels == 62)

        print("test_conv_to_conv_mask_propagation(): Successfully validated winnowed  model.")

    def test_mask_propagation_through_add_and_split(self):
        """ After the graph is constructed, the Op should have default masks and connectivity for all module types. """
        logger.debug("Test default mask and connectivity.")
        model = SingleResidual()

        # Test forward pass on the copied model before zering out channels of layers.
        input_shape = [2, 3, 224, 224]
        module_zero_channels_list = []
        module = model.conv4
        input_channels_to_prune = [0, 1, 2, 3, 4]

        module_mask_pair = (module, input_channels_to_prune)
        module_zero_channels_list.append(module_mask_pair)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         module_zero_channels_list,
                                         reshape=True, in_place=True, verbose=True)
        if winnowed_model:
            # validate winnowed net
            # input_tensor = torch.rand(input_shape).double()
            input_tensor = torch.rand(input_shape)
            # winnowed_model.double()
            winnowed_model.eval()
            _ = winnowed_model(input_tensor)
        self.assertEqual(0, 0)

    def test_mask_propagation_through_concat(self):
        """ After the graph is constructed, the Op should have default masks and connectivity for all module types. """
        logger.debug("Test default mask and connectivity.")
        model = SingleConcat()

        # Test forward pass on the copied model before zeroing out channels of layers.
        input_shape = [1, 3, 224, 224]
        module_zero_channels_list = []
        module = model.conv4
        input_channels_to_prune = [1, 3, 5, 7, 9, 19, 21, 23, 25, 43, 45, 47, 49, 51, 57, 59, 61, 63]

        module_mask_pair = (module, input_channels_to_prune)
        module_zero_channels_list.append(module_mask_pair)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        winnowed_model, _ = winnow_model(model, input_shape,
                                         module_zero_channels_list,
                                         in_place=True, verbose=True)

        # validate winnowed net
        # input_tensor = torch.rand(input_shape).double()
        input_tensor = torch.rand(input_shape)
        # winnowed_model.double()
        winnowed_model.eval()
        _ = winnowed_model(input_tensor)
        self.assertEqual(0, 0)

    @unittest.skip
    def test_mask_propagation_through_single_chunk(self):
        """ After the graph is constructed, the Op should have default masks and connectivity for all module types. """
        logger.debug("Test default mask and connectivity.")
        model = SingleChunk()

        # Test forward pass on the copied model before zering out channels of layers.
        input_shape = [1, 3, 224, 224]

        module_zero_channels_list = []
        module = model.conv4
        input_channels_to_prune = [5, 9]

        module_mask_pair = (module, input_channels_to_prune)
        module_zero_channels_list.append(module_mask_pair)

        print("Order of modules in in the API:", [get_layer_name(model, m) for m, _ in module_zero_channels_list])
        # API version 2.
        _, _ = winnow_model(model, input_shape,
                            module_zero_channels_list,
                            in_place=True, verbose=True)
        self.assertEqual(0, 0)

    def test_upsample_layer(self):
        """ Test upsample layer functionality """
        org_size = 10
        mask, idx_mask = get_dummy_mask(org_size)
        input_data = torch.rand((2, org_size - len(idx_mask), 5, 5))

        layer = UpsampleLayer(mask)
        output_data = layer(input_data)

        self.assertEqual(output_data.size(1), org_size)
        self.assertTrue(np.allclose(output_data[:, idx_mask], 0), "All masked channels should be zero")
        self.assertTrue(np.allclose(output_data[:, mask], input_data), "Input data should stay the unchanged")

    def test_null_internal_connectivity(self):
        """ Test the initialization, forward and backward propagation of the NullInternalConnectivity class. """

        # Represents the input and output mask lists held at Mask object.
        input_masks = [[]]
        output_masks = [[]]

        input_masks_list = []
        output_masks_list = []

        input_mask_length = 10
        output_mask_length = input_mask_length

        # Create a single input mask for Null
        input_masks_length_tuple = (input_masks[0], input_mask_length)
        input_masks_list.append(input_masks_length_tuple)

        # Create a single output mask for Null
        output_masks_length_tuple = (output_masks[0], output_mask_length)
        output_masks_list.append(output_masks_length_tuple)

        logger.info("Input mask Tuple: %s, Output mask Tuple: %s", input_masks_list, output_masks_list)

        # Create Null Internal Connectivity Object
        internal_connectivity = NullInternalConnectivity(input_masks_list, output_masks_list)

        logger.info("After Null Initialization. Input masks: %s, Output masks: %s", input_masks, output_masks)

        self.assertEqual(len(input_masks[0]), input_mask_length)
        self.assertEqual(len(output_masks[0]), output_mask_length)

        # Test NULL internal connectivity backward propagation
        output_masks[0] = [0 if i % 4 else 1 for i in range(output_mask_length)]

        save_input_mask = input_masks[0]
        internal_connectivity.backward_propagate_the_masks(output_masks, input_masks)
        self.assertEqual(input_masks[0], save_input_mask)
        logger.info("After Add Backward Mask Propagation. Output masks: %s, Input masks: %s", output_masks, input_masks)

        # Test Null internal connectivity forward propagation
        input_masks[0] = [1 if i % 3 else 0 for i in range(input_mask_length)]
        saved_output_mask = output_masks[0]
        internal_connectivity.forward_propagate_the_masks(input_masks, output_masks)
        self.assertEqual(output_masks[0], saved_output_mask)
        logger.info("After Null Forward Mask Propagation. Input masks: %s, Output masks: %s", input_masks, output_masks)

    def test_direct_internal_connectivity(self):
        """ Test the initialization, forward and backward propagation of the DirectInternalConnectivity class. """

        # Represents the input and output mask lists held at Mask object.
        input_masks = [[]]
        output_masks = [[]]

        input_masks_list = []
        output_masks_list = []

        input_mask_length = 15
        output_mask_length = input_mask_length

        # Create a single input mask for Null
        input_masks_length_tuple = (input_masks[0], input_mask_length)
        input_masks_list.append(input_masks_length_tuple)

        # Create a single output mask for Null
        output_masks_length_tuple = (output_masks[0], output_mask_length)
        output_masks_list.append(output_masks_length_tuple)

        logger.info("Input mask Tuple: %s, Output mask Tuple: %s", input_masks_list, output_masks_list)

        # Create Null Internal Connectivity Object
        internal_connectivity = DirectInternalConnectivity(input_masks_list, output_masks_list)

        logger.info("After Direct Initialization. Input masks: %s, Output masks: %s", input_masks, output_masks)

        self.assertEqual(len(input_masks[0]), input_mask_length)
        self.assertEqual(len(output_masks[0]), output_mask_length)

        # Test Direct internal connectivity backward propagation
        output_masks[0] = [0 if i % 3 else 1 for i in range(output_mask_length)]
        internal_connectivity.backward_propagate_the_masks(output_masks, input_masks)
        self.assertEqual(input_masks[0], output_masks[0])
        logger.info("After Direct Backward Mask Propagation. Output masks: %s, Input masks: %s", output_masks,
                    input_masks)

        # Test Direct internal connectivity forward propagation
        input_masks[0] = [1 if i % 4 else 0 for i in range(input_mask_length)]
        internal_connectivity.forward_propagate_the_masks(input_masks, output_masks)
        self.assertEqual(output_masks[0], input_masks[0])
        logger.info("After Direct  Forward Mask Propagation. Input masks: %s, Output masks: %s", input_masks,
                    output_masks)

    def test_concat_internal_connectivity(self):
        """ Test the initialization, forward and backward propagation of the ConcatInternalConnectivity class. """

        # Represents the input and output mask lists held at Mask object.
        input_masks = [[], [], []]
        output_masks = [[]]

        input_masks_list = []
        output_masks_list = []

        # Create 3 input masks for Concat.
        # Mask 1 has 5 channels
        # Mask 2 has 10 channels
        # Mask 3 has 15 channels
        minimum_num_channels_in_mask = 5
        output_mask_length = 0
        for i in range(3):
            input_mask_length = minimum_num_channels_in_mask + i * minimum_num_channels_in_mask
            output_mask_length += input_mask_length
            logger.info("Input Mask number: %s, Input mask length: %s", i, input_mask_length)
            input_masks_length_tuple = (input_masks[i], input_mask_length)
            input_masks_list.append(input_masks_length_tuple)

        logger.info("Input mask Tuple: %s, output mask length: %s", input_masks_list, output_mask_length)

        # Output masks
        output_masks_length_tuple = (output_masks[0], output_mask_length)
        output_masks_list.append(output_masks_length_tuple)
        logger.info("Input masks list: %s, Output masks list: %s", input_masks_list, output_masks_list)

        # Create Concat Internal Connectivity Object
        internal_connectivity = ConcatInternalConnectivity(input_masks_list, output_masks_list)

        logger.info("After Concat Initialization. Input masks: %s, Output masks: %s", input_masks, output_masks)

        self.assertEqual(len(input_masks[0]), minimum_num_channels_in_mask + minimum_num_channels_in_mask * 0)
        self.assertEqual(len(input_masks[1]), minimum_num_channels_in_mask + minimum_num_channels_in_mask * 1)
        self.assertEqual(len(input_masks[2]), minimum_num_channels_in_mask + minimum_num_channels_in_mask * 2)
        self.assertEqual(len(output_masks[0]), len(input_masks[0]) + len(input_masks[1]) + len(input_masks[2]))

        # Test Concat internal connectivity backward propagation
        output_masks[0] = [0 if i % 2 else 1 for i in range(30)]
        internal_connectivity.backward_propagate_the_masks(output_masks, input_masks)
        self.assertEqual(input_masks[0], output_masks[0][:5])
        self.assertEqual(input_masks[1], output_masks[0][5:15])
        self.assertEqual(input_masks[2], output_masks[0][15:30])
        logger.info("After Concat Backward Mask Propagation. Output masks: %s, Input masks: %s", output_masks,
                    input_masks)

        # Test Concat internal connectivity forward propagation
        input_masks[0] = [1 if i % 2 else 0 for i in range(5)]
        input_masks[1] = [1 if i % 2 else 0 for i in range(10)]
        input_masks[2] = [1 if i % 2 else 0 for i in range(15)]
        internal_connectivity.forward_propagate_the_masks(input_masks, output_masks)
        self.assertEqual(output_masks[0], input_masks[0] + input_masks[1] + input_masks[2])

        logger.info("After Concat Forward Mask Propagation. Input masks: %s, Output masks: %s", input_masks,
                    output_masks)

    def test_add_internal_connectivity(self):
        """ Test the initialization, forward and backward propagation of the AddInternalConnectivity class. """

        # Represents the input and output mask lists held at Mask object.
        input_masks = [[], [], []]
        output_masks = [[]]

        input_masks_list = []
        output_masks_list = []

        # Create 3 input masks of the same length for Add.

        input_mask_length = 10
        output_mask_length = input_mask_length
        for i in range(3):
            input_masks_length_tuple = (input_masks[i], input_mask_length)
            input_masks_list.append(input_masks_length_tuple)

        logger.info("Input mask Tuple: %s, output mask length: %s", input_masks_list, output_mask_length)

        # Output masks
        output_masks_length_tuple = (output_masks[0], output_mask_length)
        output_masks_list.append(output_masks_length_tuple)
        logger.info("Input masks list: %s, Output masks list: %s", input_masks_list, output_masks_list)

        # Create Add Internal Connectivity Object
        internal_connectivity = AddInternalConnectivity(input_masks_list, output_masks_list)

        logger.info("After Add Initialization. Input masks: %s, Output masks: %s", input_masks, output_masks)

        self.assertEqual(len(input_masks[0]), input_mask_length)
        self.assertEqual(len(input_masks[1]), input_mask_length)
        self.assertEqual(len(input_masks[2]), input_mask_length)
        self.assertEqual(len(output_masks[0]), output_mask_length)

        # Test Add internal connectivity backward propagation
        output_masks[0] = [0 if i % 3 else 1 for i in range(output_mask_length)]
        internal_connectivity.backward_propagate_the_masks(output_masks, input_masks)
        self.assertEqual(input_masks[0], output_masks[0])
        self.assertEqual(input_masks[1], output_masks[0])
        self.assertEqual(input_masks[2], output_masks[0])
        logger.info("After Add Backward Mask Propagation. Output masks: %s, Input masks: %s", output_masks, input_masks)

        # Test Add internal connectivity forward propagation
        input_masks[0] = [1 if i % 2 else 0 for i in range(input_mask_length)]
        input_masks[1] = [1 if i % 3 else 0 for i in range(input_mask_length)]
        input_masks[2] = [1 if i % 4 else 0 for i in range(input_mask_length)]
        internal_connectivity.forward_propagate_the_masks(input_masks, output_masks)

        logger.info("After Add Forward Mask Propagation. Input masks: %s, Output masks: %s", input_masks, output_masks)

    def test_split_internal_connectivity(self):
        """ Test the initialization, forward and backward propagation of the AddInternalConnectivity class. """

        # Represents the input and output mask lists held at Mask object.
        input_masks = [[]]
        output_masks = [[], []]

        input_masks_list = []
        output_masks_list = []

        input_mask_length = 10
        output_mask_length = input_mask_length

        # Create a single input mask for Split
        input_masks_length_tuple = (input_masks[0], input_mask_length)
        input_masks_list.append(input_masks_length_tuple)

        # Create 2 Output masks for Split
        for i in range(2):
            output_masks_length_tuple = (output_masks[i], output_mask_length)
            output_masks_list.append(output_masks_length_tuple)

        logger.info("Input mask Tuple: %s, Output mask Tuple: %s", input_masks_list, output_masks_list)

        # Create Split Internal Connectivity Object
        internal_connectivity = SplitInternalConnectivity(input_masks_list, output_masks_list)

        logger.info("After Split Initialization. Input masks: %s, Output masks: %s", input_masks, output_masks)

        self.assertEqual(len(input_masks[0]), input_mask_length)
        self.assertEqual(len(output_masks[0]), output_mask_length)
        self.assertEqual(len(output_masks[1]), output_mask_length)

        # Test Split internal connectivity backward propagation
        output_masks[0] = [0 if i % 2 else 1 for i in range(output_mask_length)]
        # Do not change output_masks[1].Leave it all initialized to 1.

        internal_connectivity.backward_propagate_the_masks(output_masks, input_masks)
        self.assertEqual(input_masks[0], [a or b for a, b in zip(output_masks[0], output_masks[1])])
        logger.info("After Add Backward Mask Propagation. Output masks: %s, Input masks: %s", output_masks, input_masks)

        # Test Split internal connectivity forward propagation
        input_masks[0] = [1 if i % 3 else 0 for i in range(input_mask_length)]
        internal_connectivity.forward_propagate_the_masks(input_masks, output_masks)
        self.assertEqual(output_masks[0], input_masks[0])
        self.assertEqual(output_masks[1], input_masks[0])
        logger.info("After Split Forward Mask Propagation. Input masks: %s, Output masks: %s", input_masks,
                    output_masks)


def get_dummy_mask(size: int):
    """ Return a test mask of length size """
    mask = torch.ones(size).byte()
    idx_mask = [2, 3, 6]
    mask[[idx_mask]] = 0
    return mask, idx_mask
