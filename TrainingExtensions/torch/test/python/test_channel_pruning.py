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

import unittest
import unittest.mock
import itertools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torchvision import models

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.winnow.winnow_utils import zero_out_input_channels
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.input_match_search import InputMatchSearch

from aimet_torch.data_subsampler import DataSubSampler
from aimet_torch.channel_pruning.weight_reconstruction import WeightReconstructor
from aimet_torch.channel_pruning.channel_pruner import InputChannelPruner
from .models.mnist_torch_model import Net as mnist_model
from aimet_torch.utils import to_numpy, create_fake_data_loader, get_layer_name, get_layer_by_name,\
    create_rand_tensors_given_shapes, get_device
from aimet_torch.layer_database import Layer, LayerDatabase


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ChannelPruning)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2(x), 2))
        x = x.view(x.view(0), -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class TestTrainingExtensionsChannelPruning(unittest.TestCase):

    def test_find_input_data_pixel_indices(self):
        """ Test utility to determine input data pixel height and width ranges are calculated correctly or not
        for given set of kernel_size, padding and stride combination"""
        in_channels = 1
        out_channels = 10
        input_data = np.random.rand(1, 1, 8, 8)
        strides = [[1, 1], [2, 2], [1, 2], [2, 1]]
        kernel_size_options = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]]
        padding_options = [[0, 0], [1, 1], [2, 2], [1, 2], [2, 1]]
        all_options = [kernel_size_options, padding_options, strides]

        for kernel_size, padding, stride in itertools.product(*all_options):

            # we don't consider padding larger than kernel_size
            for i, ks in enumerate(kernel_size):
                if ks == 1:
                    padding = copy.deepcopy(padding)
                    padding[i] = 0

            layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)

            input_h, input_w = input_data.shape[2], input_data.shape[3]
            filter_h, filter_w = layer.kernel_size[0], layer.kernel_size[1]
            stride_h, stride_w = layer.stride[0], layer.stride[1]
            padding_h, padding_w = layer.padding[0], layer.padding[1]

            check_h = (input_h - filter_h + 2 * padding_h) / stride_h
            check_w = (input_w - filter_w + 2 * padding_w) / stride_w

            # if the condition is not satisfied, ignore that particular combination of kernel_size, padding and stride
            if not check_h % layer.stride[0] == 0 or not check_w % layer.stride[1] == 0:
                continue

            # calculate output height and width max values
            out_height_max = int(check_h) + 1
            out_width_max = int(check_w) + 1

            size_options = [[x, y] for x in range(out_height_max) for y in range(out_width_max)]

            layer_attributes = (layer.kernel_size, layer.stride, layer.padding)

            # iterate over all the output pixels
            for size_opt in size_options:

                height = size_opt[0]
                width = size_opt[1]
                in_data_height_range, in_data_width_range = \
                    InputMatchSearch._find_pixel_range_for_input_data(input_data_shape=input_data.shape[1:],
                                                                      layer_attributes=layer_attributes,
                                                                      pixel=(height, width))

                # check input data height indices range
                self.assertEqual(in_data_height_range[0], max(0, (height * layer.stride[0]) - layer.padding[0]))
                self.assertEqual(in_data_height_range[1], (height * layer.stride[0]) - layer.padding[0] +
                                 layer.kernel_size[0])

                # check input data width indices range
                self.assertEqual(in_data_width_range[0], max(0, (width * layer.stride[1]) - layer.padding[1]))
                self.assertEqual(in_data_width_range[1], (width * layer.stride[1]) - layer.padding[1] +
                                 layer.kernel_size[1])

    def test_find_input_match(self):
        """ Test to determine input rectangle match given pixel ranges for input data and input match"""

        batch_index = 0
        input_data = np.array(range(8 * 8)).reshape([1, 1, 8, 8])
        kernel_size = (3, 3)

        layer_attributes = (kernel_size, )

        # bottom right corner
        pixel_range_for_data = ((6, 9), (6, 9))
        pixel_range_for_match = ((0, 2), (0, 2))

        input_match = InputMatchSearch._find_input_match(input_data[batch_index], layer_attributes,
                                                         pixel_range_for_data, pixel_range_for_match)

        self.assertEqual(np.sum(input_match), 54 + 55 + 62 + 63)

        # bottom left corner
        pixel_range_for_data = ((6, 9), (0, 2))
        pixel_range_for_match = ((0, 2), (1, 3))
        input_match = InputMatchSearch._find_input_match(input_data[batch_index], layer_attributes,
                                                         pixel_range_for_data, pixel_range_for_match)
        self.assertEqual(np.sum(input_match), 48 + 49 + 56 + 57)

        # top right corner
        pixel_range_for_data = ((0, 2), (6, 9))
        pixel_range_for_match = ((1, 3), (0, 2))
        input_match = InputMatchSearch._find_input_match(input_data[batch_index], layer_attributes,
                                                         pixel_range_for_data, pixel_range_for_match)

        self.assertEqual(np.sum(input_match), 6 + 7 + 14 + 15)

        # top left corner
        pixel_range_for_data = ((0, 2), (0, 2))
        pixel_range_for_match = ((1, 3), (1, 3))
        input_match = InputMatchSearch._find_input_match(input_data[batch_index], layer_attributes,
                                                         pixel_range_for_data, pixel_range_for_match)

        self.assertEqual(np.sum(input_match), 0 + 1 + 8 + 9)

        # middle
        pixel_range_for_data = ((3, 6), (3, 6))
        pixel_range_for_match = ((0, 3), (0, 3))
        input_match = InputMatchSearch._find_input_match(input_data[batch_index], layer_attributes,
                                                         pixel_range_for_data, pixel_range_for_match)

        self.assertEqual(np.sum(input_match), 27 + 28 + 29 + 35 + 36 + 37 + 43 + 44 + 45)

    def test__determine_output_pixel_height_width_range_for_random_selection(self):

        strides = [[1, 1], [2, 2], [1, 2], [2, 1]]
        kernel_size_options = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]]
        padding_options = [[0, 0], [1, 1], [2, 2], [1, 2], [2, 1], [3, 3]]

        all_options = [kernel_size_options, strides, padding_options]
        output_data_shape = (0, 1, 8, 8)
        for kernel_size, stride, padding in itertools.product(*all_options):

            layer_attributes = kernel_size, stride, padding
            height_range, width_range = \
                InputMatchSearch._determine_output_pixel_height_width_range_for_random_selection(
                    layer_attributes, output_data_shape)
            start, end = height_range
            if kernel_size[0] >= padding[0]:
                assert start == 0 and end == output_data_shape[2]
            else:
                assert start == padding[0] and end == (output_data_shape[2] - padding[0])

            start, end = width_range
            if kernel_size[1] >= padding[1]:
                assert start == 0 and end == output_data_shape[3]
            else:
                assert start == padding[1] and end == (output_data_shape[3] - padding[1])

    def test_find_input_match_for_pixel_from_output_data_baseline(self):

        batch_num = 0
        strides = [[1, 1], [2, 2], [1, 2], [2, 1]]
        kernel_size_options = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]]
        padding_options = [[0, 0], [1, 1], [2, 2], [1, 2], [2, 1]]

        max_height = 8
        max_width = 8

        input_frame = np.array(range(8 * 8)).reshape([1, 1, max_height, max_width])
        num_output_pixels = 5

        # randomly pick samples per image for height and width dimension
        heights = np.random.choice(range(2, (max_height - 2)), size=[num_output_pixels], replace=True)
        widths = np.random.choice(range(2, (max_width - 2)), size=[num_output_pixels], replace=True)

        size_options = [[a,b] for a, b in zip(heights, widths)]
        print("Size Options", size_options)

        all_options = [kernel_size_options, padding_options, size_options, strides]

        for kernel_size, padding, size_opt, stride in itertools.product(
                *all_options):

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, list) and len(stride) == 2:
                height, width = [size_opt[0]//stride[0], size_opt[1] // stride[1]]
            else:
                height, width = [size // stride for size in size_opt]

            output_data_pixel = (height, width)

            conv_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

            layer_attributes = (conv_filter.kernel_size, conv_filter.stride, conv_filter.padding)

            conv_filter.weight.data =\
                torch.FloatTensor(np.ones([1, 1, kernel_size[0], kernel_size[1]], dtype=np.float32))



            input_match = InputMatchSearch._find_input_match_for_output_pixel(input_frame[batch_num], layer_attributes,
                                                                              output_data_pixel)
            conv2d_out = functional.conv2d(torch.FloatTensor(input_frame), conv_filter.weight.data, stride=stride,
                                           padding=padding)

            predicted_output = np.sum(input_match)
            generated_output = conv2d_out[0, 0, height, width].detach().numpy()

            assert generated_output == predicted_output
            assert np.prod(input_match.shape) == kernel_size[0] * kernel_size[1]

    @unittest.skip
    def test_get_activation_data(self):
        """ Test to collect activations (input from model_copy and output from model for conv2 layer) and compare
        """
        orig_model = TestNet().cuda()
        comp_model = copy.deepcopy(orig_model)
        dataset_size = 1000
        batch_size = 10
        # max out number of batches
        number_of_batches = 100

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        conv2_input_data, conv2_output_data = DataSubSampler.get_sub_sampled_data(orig_layer=orig_model.conv2,
                                                                                  pruned_layer=comp_model.conv2,
                                                                                  orig_model=orig_model,
                                                                                  comp_model=comp_model,
                                                                                  data_loader=data_loader,
                                                                                  num_reconstruction_samples=
                                                                                  number_of_batches)
        iterator = data_loader.__iter__()

        for batch in range(number_of_batches):

            images_in_one_batch, _ = iterator.__next__()
            conv1_output = orig_model.conv1(images_in_one_batch.cuda())
            conv2_input = conv1_output
            conv2_output = orig_model.conv2(functional.relu(functional.max_pool2d(conv2_input, 2)))
            # compare the output from conv2 layer
            self.assertTrue(np.array_equal(to_numpy(conv2_output),
                                           conv2_output_data[batch * batch_size: (batch + 1) * batch_size, :, :, :]))

            conv1_output_copy = comp_model.conv1(images_in_one_batch.cuda())
            conv2_input_copy = functional.relu(functional.max_pool2d(conv1_output_copy, 2))
            # compare the inputs of conv2 layer
            self.assertTrue(np.array_equal(to_numpy(conv2_input_copy),
                                           conv2_input_data[batch * batch_size: (batch + 1) * batch_size, :, :, :]))

    @unittest.mock.patch('numpy.random.choice')
    def test_subsample_data(self, np_choice_function):
        """Test to subsample input match for random output pixel (1, 1) and corresponding input match"""
        # randomly selected output pixel (height, width) is fixed here and it is (1, 1)
        np_choice_function.return_value = [1]

        model = TestNet()

        input_data = np.arange(0, 1440).reshape((2, 5, 12, 12))
        output_data = np.arange(0, 1280).reshape((2, 10, 8, 8))

        conv2 = model.conv2
        layer_attributes = (conv2.kernel_size, conv2.stride, conv2.padding)

        sub_sample_input, sub_sample_output = InputMatchSearch.subsample_data(layer_attributes=layer_attributes,
                                                                              input_data=input_data,
                                                                              output_data=output_data,
                                                                              samples_per_image=1)

        # compare the inputs for both batches
        self.assertEqual(sub_sample_input.shape, (2, 5, 5, 5))
        self.assertTrue(np.array_equal(sub_sample_input[0, :, :, :], input_data[0, :, 1:6, 1:6]))
        self.assertTrue(np.array_equal(sub_sample_input[1, :, :, :], input_data[1, :, 1:6, 1:6]))

        # compare the output for batches
        output_pixel = (1, 1)
        self.assertEqual(sub_sample_output.shape, (2, 10))
        self.assertTrue(np.array_equal(sub_sample_output, output_data[:, :, output_pixel[0], output_pixel[1]]))

    def test_linear_regression(self):
        """Test weight reconstruction with data only"""

        number_of_images = 1000

        in_channels = 5
        out_channels = 10
        k_h = 5
        k_w = 5

        # 1) with fit_intercept = False (without bias)
        input_data = np.random.rand(number_of_images, in_channels, k_h, k_w)
        weight = np.random.rand(in_channels * k_h * k_w, out_channels)
        reshaped_input_data = input_data.reshape(number_of_images, in_channels * k_h * k_w)

        # calculate y = x * w
        output_data = np.matmul(reshaped_input_data, weight)

        new_w, _ = WeightReconstructor._linear_regression(input_data=reshaped_input_data, output_data=output_data,
                                                          bias=False)
        new_w = new_w.reshape(new_w.shape[0], np.prod(new_w.shape[1:4])).transpose()

        self.assertTrue(np.allclose(new_w, weight))

        # 2) with fit_intercept = True (with bias)
        input_data = np.random.rand(number_of_images, in_channels, k_h, k_w)
        weight = np.random.rand(in_channels * k_h * k_w, out_channels)
        bias = np.random.rand(out_channels)

        reshaped_input_data = input_data.reshape(number_of_images, in_channels * k_h * k_w)

        # calculate y = x * w + b
        output_data = np.matmul(reshaped_input_data, weight) + bias
        # fit_intercept = True
        new_w, new_b = WeightReconstructor._linear_regression(input_data=reshaped_input_data, output_data=output_data,
                                                              bias=True)
        new_w = new_w.reshape(new_w.shape[0], np.prod(new_w.shape[1:4])).transpose()

        # compare weight
        self.assertTrue(np.allclose(new_w, weight))
        # compare bias
        self.assertTrue(np.allclose(new_b, bias))

    def test_reconstruct_weight_and_bias_for_layer(self):
        """ """
        model = TestNet()
        layer = model.conv2

        # input shape should be [Ns, Nic, k_h, k_w]
        number_of_images = 500
        inputs = np.random.rand(number_of_images, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1])

        outputs = functional.conv2d(torch.FloatTensor(inputs), layer.weight, bias=layer.bias, stride=layer.stride,
                                    padding=layer.padding)

        # expecting output shape (number_of_images, layer.out_channels, 1, 1)
        self.assertEqual(np.prod(outputs.shape[1:4]), layer.out_channels)

        reshaped_outputs = outputs.reshape([outputs.shape[0], np.prod(outputs.shape[1:4])])

        WeightReconstructor.reconstruct_params_for_conv2d(layer=layer, input_data=inputs,
                                                          output_data=to_numpy(reshaped_outputs))

        new_outputs = functional.conv2d(torch.FloatTensor(inputs), layer.weight,
                                        bias=layer.bias, stride=layer.stride,
                                        padding=layer.padding)

        # if data is increased, choose tolerance wisely
        self.assertTrue(np.allclose(to_numpy(outputs), to_numpy(new_outputs), atol=1e-5))

    def test_data_sub_sampling_and_reconstruction(self):
        """Test end to end data sub sampling and reconstruction for MNIST conv2 layer"""
        orig_model = mnist_model()
        comp_model = copy.deepcopy(orig_model)

        dataset_size = 100
        batch_size = 10
        # max out number of batches
        number_of_batches = 10
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        cp = InputChannelPruner(data_loader=data_loader, input_shape=None,
                                num_reconstruction_samples=num_reconstruction_samples,
                                allow_custom_downsample_ops=True)

        cp._data_subsample_and_reconstruction(orig_layer=orig_model.conv2, pruned_layer=comp_model.conv2,
                                              orig_model=orig_model, comp_model=comp_model)

        self.assertEqual(comp_model.conv2.weight.data.shape, orig_model.conv2.weight.data.shape)
        self.assertEqual(comp_model.conv2.bias.data.shape, orig_model.conv2.bias.data.shape)

        # if you increase the data (data set size, number of batches or samples per image),
        # reduce the absolute tolerance

        self.assertTrue(np.allclose(to_numpy(comp_model.conv2.weight.data),
                                    to_numpy(orig_model.conv2.weight.data), atol=1e-0))

        self.assertTrue(np.allclose(to_numpy(comp_model.conv2.bias.data),
                                    to_numpy(orig_model.conv2.bias.data), atol=1e-0))

    def test_data_sub_sample_and_reconstruction_with_zero_channels(self):
        """Test end to end data sub sampling and reconstruction for MNIST conv2 layer"""
        orig_model = mnist_model()
        comp_model = copy.deepcopy(orig_model)

        dataset_size = 100
        batch_size = 10
        # max out number of batches
        number_of_batches = 10
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image
        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        cp = InputChannelPruner(data_loader=data_loader, input_shape=None,
                                num_reconstruction_samples=num_reconstruction_samples,
                                allow_custom_downsample_ops=True)

        input_channels_to_prune = [0, 1, 2, 3, 4, 5, 15, 29, 24, 28]
        zero_out_input_channels(comp_model.conv2, input_channels_to_prune)

        before_reconstruction = comp_model.conv2.weight.data

        cp._data_subsample_and_reconstruction(orig_layer=orig_model.conv2, pruned_layer=comp_model.conv2,
                                              orig_model=orig_model, comp_model=comp_model)

        after_reconstruction = comp_model.conv2.weight.data

        self.assertEqual(comp_model.conv2.weight.data.shape, orig_model.conv2.weight.data.shape)
        self.assertEqual(comp_model.conv2.bias.data.shape, orig_model.conv2.bias.data.shape)
        # make sure they are not same
        self.assertFalse(np.allclose(to_numpy(before_reconstruction), to_numpy(after_reconstruction)))

    def test_data_sub_sampling_and_reconstruction_without_bias(self):
        """Test end to end data sub sampling and reconstruction for MNIST conv2 layer (without bias)"""

        orig_model = mnist_model()
        # set bias to None
        orig_model.conv2.bias = None

        comp_model = copy.deepcopy(orig_model)

        dataset_size = 100
        batch_size = 10
        # max out number of batches
        number_of_batches = 10
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image
        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        conv2_pr_layer_name = get_layer_name(comp_model, comp_model.conv2)

        sampled_inp_data, sampled_out_data = DataSubSampler.get_sub_sampled_data(orig_layer=orig_model.conv2,
                                                                                 pruned_layer=comp_model.conv2,
                                                                                 orig_model=orig_model,
                                                                                 comp_model=comp_model,
                                                                                 data_loader=data_loader,
                                                                                 num_reconstruction_samples=
                                                                                 num_reconstruction_samples)

        conv_layer = get_layer_by_name(model=comp_model, layer_name=conv2_pr_layer_name)

        assert conv_layer == comp_model.conv2
        # original weight before reconstruction
        orig_weight = conv_layer.weight.data
        WeightReconstructor.reconstruct_params_for_conv2d(layer=conv_layer, input_data=sampled_inp_data,
                                                          output_data=sampled_out_data)
        # new weight after reconstruction
        new_weight = conv_layer.weight.data
        new_bias = conv_layer.bias

        self.assertEqual(new_weight.shape, orig_weight.shape)
        self.assertEqual(new_bias, None)
        # if you increase the data (data set size, number of batches or samples per image),
        # reduce the absolute tolerance
        self.assertTrue(np.allclose(to_numpy(new_weight), to_numpy(orig_weight), atol=1e-0))

    def test_select_inp_channels(self):

        orig_model = mnist_model()
        data_loader = unittest.mock.MagicMock()
        number_of_batches = unittest.mock.MagicMock()
        samples_per_image = unittest.mock.MagicMock()

        cp = InputChannelPruner(data_loader=data_loader, input_shape=None,
                                num_reconstruction_samples=number_of_batches,
                                allow_custom_downsample_ops=True)

        orig_model.conv2.weight.data = torch.FloatTensor(np.array(range(32 * 64 * 5 * 5)).reshape(64, 32, 5, 5))

        # in_channels = 32 and calculate remaining channels
        # 1) 32 * 0.25 = 8
        # 3) 32 * 0.50 = 16
        # 4) 32 * 0.75 = 24
        # 5) 32 * 1 = 32

        input_channels_indices = list(range(32))

        comp_ratio_prune_inp_channels_list = [(0.25, 24), (0.50, 16), (0.75, 8), (1, 0)]

        for comp_ratio, remaining_channels in comp_ratio_prune_inp_channels_list:
            prune_indices = cp._select_inp_channels(orig_model.conv2, comp_ratio=comp_ratio)
            self.assertTrue(isinstance(prune_indices, list))
            expected_indices = input_channels_indices[:remaining_channels]
            self.assertEqual(prune_indices, expected_indices)

        # 2) set particular input channels to higher values, so that they get picked up
        conv_layer = torch.nn.Conv2d(20, 50, 5)
        # set higher values to channel number 13 and 17
        conv_layer.weight.data[:, 13, :, :] = 4
        conv_layer.weight.data[:, 17, :, :] = 5


        # 10% comp_ratio select will select 18 input channels (20 * (1  - 0.10)) for pruning
        comp_ratio = 0.10

        prune_indices = cp._select_inp_channels(conv_layer, comp_ratio=comp_ratio)

        self.assertEqual(prune_indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19])

        # 3) check the worst case when number of input channels to keep becomes zero
        conv_layer = torch.nn.Conv2d(3, 50, 5)
        # set higher values to channel number 1
        conv_layer.weight.data[:, 1, :, :] = 4

        comp_ratio = 0.10

        prune_indices = cp._select_inp_channels(conv_layer, comp_ratio=comp_ratio)

        self.assertEqual(prune_indices, [0, 2])

    def test_sort_on_occurrence(self):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
                self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
                self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
                self.conv4 = nn.Conv2d(10, 10, kernel_size=3)
                self.fc1 = nn.Linear(490, 300)
                self.fc2 = nn.Linear(300, 10)

            def forward(self, x):
                x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
                x = functional.relu(self.conv2(x))
                x = functional.relu(self.conv3(x))
                x = functional.relu(self.conv4(x))
                x = x.view(x.size(0), -1)
                x = functional.relu(self.fc1(x))
                x = self.fc2(x)
                return functional.log_softmax(x, dim=1)

        orig_model = Net()

        data_loader = unittest.mock.MagicMock()
        number_of_batches = unittest.mock.MagicMock()
        samples_per_image = unittest.mock.MagicMock()

        input_channel_pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 1, 28, 28),
                                                  num_reconstruction_samples=number_of_batches,
                                                  allow_custom_downsample_ops=True)

        layer_comp_ratio_list = [LayerCompRatioPair(Layer(orig_model.conv4, None, None), None),
                                 LayerCompRatioPair(Layer(orig_model.conv1, None, None), None),
                                 LayerCompRatioPair(Layer(orig_model.conv3, None, None), None),
                                 LayerCompRatioPair(Layer(orig_model.conv2, None, None), None)]

        sorted_layer_comp_ratio_list = input_channel_pruner._sort_on_occurrence(orig_model, layer_comp_ratio_list)

        self.assertEqual(sorted_layer_comp_ratio_list[0].layer.module, orig_model.conv1)
        self.assertEqual(sorted_layer_comp_ratio_list[1].layer.module, orig_model.conv2)
        self.assertEqual(sorted_layer_comp_ratio_list[2].layer.module, orig_model.conv3)
        self.assertEqual(sorted_layer_comp_ratio_list[3].layer.module, orig_model.conv4)

        self.assertTrue(isinstance(sorted_layer_comp_ratio_list[0].layer, Layer))
        self.assertTrue(isinstance(sorted_layer_comp_ratio_list[1].layer, Layer))
        self.assertTrue(isinstance(sorted_layer_comp_ratio_list[2].layer, Layer))
        self.assertTrue(isinstance(sorted_layer_comp_ratio_list[3].layer, Layer))

    def test_prune_layer(self):

        orig_model = mnist_model()
        orig_model.eval()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(orig_model))
        orig_layer_db = LayerDatabase(orig_model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        dataset_size = 100
        batch_size = 10
        # max out number of batches
        number_of_batches = 10
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image
        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        input_channel_pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 1, 28, 28),
                                                  num_reconstruction_samples=num_reconstruction_samples,
                                                  allow_custom_downsample_ops=True)

        conv2 = comp_layer_db.find_layer_by_name('conv2')
        input_channel_pruner._prune_layer(orig_layer_db, comp_layer_db, conv2, 0.5, CostMetric.mac)

        self.assertTrue(comp_layer_db.model.conv2.in_channels, 16)
        self.assertTrue(comp_layer_db.model.conv2.out_channels, 64)

    def test_prune_layer_with_seq(self):
        """ Test end to end prune layer with resnet18"""

        batch_size = 2
        dataset_size = 1000
        number_of_batches = 1
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image

        model = models.resnet18().eval()

        # Create a layer database
        input_shape = (1, 3, 224, 224)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size,
                                              image_size=(3, 224, 224))

        input_channel_pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 3, 224, 224),
                                                  num_reconstruction_samples=num_reconstruction_samples,
                                                  allow_custom_downsample_ops=True)

        conv_below_split = comp_layer_db.find_layer_by_name('layer1.1.conv1')
        input_channel_pruner._prune_layer(orig_layer_db, comp_layer_db, conv_below_split, 0.25, CostMetric.mac)

        # 64 * 0.25 = 16
        self.assertEqual(comp_layer_db.model.layer1[1].conv1[1].in_channels, 16)
        self.assertEqual(comp_layer_db.model.layer1[1].conv1[1].out_channels, 64)
        self.assertEqual(list(comp_layer_db.model.layer1[1].conv1[1].weight.shape), [64, 16, 3, 3])

    def test_prune_model(self):
        """Test end to end prune model with Mnist"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
                self.max_pool2d = nn.MaxPool2d(2)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
                self.relu2 = nn.ReLU()
                self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
                self.relu3 = nn.ReLU()
                self.conv4 = nn.Conv2d(30, 40, kernel_size=3)
                self.relu4 = nn.ReLU()
                self.fc1 = nn.Linear(7 * 7 * 40, 300)
                self.relu5 = nn.ReLU()
                self.fc2 = nn.Linear(300, 10)
                self.log_softmax = nn.LogSoftmax(dim=1)

            def forward(self, x):
                x = self.relu1(self.max_pool2d(self.conv1(x)))
                x = self.relu2(self.conv2(x))
                x = self.relu3(self.conv3(x))
                x = self.relu4(self.conv4(x))
                x = x.view(x.size(0), -1)
                x = self.relu5(self.fc1(x))
                x = self.fc2(x)
                return self.log_softmax(x)

        orig_model = Net()
        orig_model.eval()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(orig_model))
        orig_layer_db = LayerDatabase(orig_model, dummy_input)

        dataset_size = 1000
        batch_size = 10
        # max out number of batches
        number_of_batches = 100
        samples_per_image = 10

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)

        input_channel_pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 1, 28, 28),
                                                  num_reconstruction_samples=number_of_batches,
                                                  allow_custom_downsample_ops=True)

        # keeping compression ratio = 0.5 for all layers
        layer_comp_ratio_list = [LayerCompRatioPair(Layer(orig_model.conv4, 'conv4', None), 0.5),
                                 LayerCompRatioPair(Layer(orig_model.conv3, 'conv3', None), 0.5),
                                 LayerCompRatioPair(Layer(orig_model.conv2, 'conv2', None), 0.5)]

        comp_layer_db = input_channel_pruner.prune_model(orig_layer_db, layer_comp_ratio_list, CostMetric.mac,
                                                            trainer=None)

        self.assertEqual(comp_layer_db.model.conv2.in_channels, 5)
        self.assertEqual(comp_layer_db.model.conv2.out_channels, 10)

        self.assertEqual(comp_layer_db.model.conv3.in_channels, 10)
        self.assertEqual(comp_layer_db.model.conv3.out_channels, 15)

        self.assertEqual(comp_layer_db.model.conv4.in_channels, 15)
        self.assertEqual(comp_layer_db.model.conv4.out_channels, 40)

    def test_prune_model_with_seq(self):
        """Test end to end prune model with resnet18"""

        batch_size = 2
        dataset_size = 1000
        number_of_batches = 1
        samples_per_image = 10
        num_reconstruction_samples = number_of_batches * batch_size * samples_per_image

        model = models.resnet18()
        model.eval()

        # Create a layer database
        input_shape = (1, 3, 224, 224)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size,
                                              image_size=(3, 224, 224))

        input_channel_pruner = InputChannelPruner(data_loader=data_loader, input_shape=(1, 3, 224, 224),
                                                  num_reconstruction_samples=num_reconstruction_samples,
                                                  allow_custom_downsample_ops=True)

        # keeping compression ratio = 0.5 for all layers
        layer_comp_ratio_list = [LayerCompRatioPair(Layer(model.layer4[1].conv1, 'layer4.1.conv1', None), 0.5),
                                 LayerCompRatioPair(Layer(model.layer3[1].conv1, 'layer3.1.conv1', None), 0.5),
                                 LayerCompRatioPair(Layer(model.layer2[1].conv1, 'layer2.1.conv1', None), 0.5),
                                 LayerCompRatioPair(Layer(model.layer1[1].conv1, 'layer1.1.conv1', None), 0.5),
                                 LayerCompRatioPair(Layer(model.layer1[0].conv2, 'layer1.0.conv2', None), 0.5)]

        comp_layer_db = input_channel_pruner.prune_model(orig_layer_db, layer_comp_ratio_list, CostMetric.mac,
                                                         trainer=None)

        # 1) not below split
        self.assertEqual(comp_layer_db.model.layer1[0].conv2.in_channels, 32)
        self.assertEqual(comp_layer_db.model.layer1[0].conv2.out_channels, 64)
        self.assertEqual(list(comp_layer_db.model.layer1[0].conv2.weight.shape), [64, 32, 3, 3])
        # impacted
        self.assertEqual(comp_layer_db.model.layer1[0].conv1.in_channels, 64)
        self.assertEqual(comp_layer_db.model.layer1[0].conv1.out_channels, 32)
        self.assertEqual(list(comp_layer_db.model.layer1[0].conv1.weight.shape), [32, 64, 3, 3])

        # 2) below split

        # 64 * .5
        self.assertEqual(comp_layer_db.model.layer1[1].conv1[1].in_channels, 32)
        self.assertEqual(comp_layer_db.model.layer1[1].conv1[1].out_channels, 64)
        self.assertEqual(list(comp_layer_db.model.layer1[1].conv1[1].weight.shape), [64, 32, 3, 3])

        # 128 * .5
        self.assertEqual(comp_layer_db.model.layer2[1].conv1[1].in_channels, 64)
        self.assertEqual(comp_layer_db.model.layer2[1].conv1[1].out_channels, 128)
        self.assertEqual(list(comp_layer_db.model.layer2[1].conv1[1].weight.shape), [128, 64, 3, 3])

        # 256 * .5
        self.assertEqual(comp_layer_db.model.layer3[1].conv1[1].in_channels, 128)
        self.assertEqual(comp_layer_db.model.layer3[1].conv1[1].out_channels, 256)
        self.assertEqual(list(comp_layer_db.model.layer3[1].conv1[1].weight.shape), [256, 128, 3, 3])

        # 512 * .5
        self.assertEqual(comp_layer_db.model.layer4[1].conv1[1].in_channels, 256)
        self.assertEqual(comp_layer_db.model.layer4[1].conv1[1].out_channels, 512)
        self.assertEqual(list(comp_layer_db.model.layer4[1].conv1[1].weight.shape), [512, 256, 3, 3])
