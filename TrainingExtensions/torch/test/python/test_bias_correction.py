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

import unittest.mock
import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .models import mnist_torch_model as mnist_model
from .models.mobilenet import MockMobileNetV11 as MockMobileNetV1
from .models.test_models import TransposedConvModel

from aimet_common.defs import QuantScheme

from aimet_torch import bias_correction
from aimet_torch.utils import to_numpy, create_fake_data_loader
from aimet_torch.cross_layer_equalization import get_ordered_list_of_conv_modules
from aimet_torch.bias_correction import find_all_conv_bn_with_activation
from aimet_torch import quantsim as qsim


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


class BatchIterator:
    def __init__(self, img_size):
        self.image_size = img_size

    def __iter__(self):
        img = torch.randn(*self.image_size)
        yield img, 0


class TestTrainingExtensionBnFold:
    def test_get_output_of_layer(self):
        model = TestNet()
        dataset_size = 2
        batch_size = 2
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)
        for images_in_one_batch, _ in data_loader:
            conv2_output_data = bias_correction.get_output_data(model.conv2, model, images_in_one_batch)

        # max out number of batches
        number_of_batches = 1
        iterator = data_loader.__iter__()

        for batch in range(number_of_batches):

            images_in_one_batch, _ = iterator.__next__()
            conv1_output = model.conv1(images_in_one_batch)
            conv2_input = conv1_output
            conv2_output = model.conv2(functional.relu(functional.max_pool2d(conv2_input, 2)))
            # compare the output from conv2 layer
            assert (np.allclose(to_numpy(conv2_output), np.asarray(conv2_output_data)[batch * batch_size: (batch + 1) *
                                                                                                          batch_size, :, :, :]))

    def test_get_ordering_of_nodes_in_model(self):
        model = mnist_model.ExtendedNet()
        dummy_input = torch.randn(1, 1, 28, 28)
        list_modules = get_ordered_list_of_conv_modules(model, dummy_input)
        assert list_modules[0][0] == 'conv1'
        assert list_modules[1][0] == 'conv2'

    def test_get_quantized_weight(self):
        model = mnist_model.Net()

        params = qsim.QuantParams(weight_bw=4, act_bw=4, round_mode="nearest",
                                       quant_scheme=QuantScheme.post_training_tf
                                 )
        use_cuda = False
        dataset_size = 2
        batch_size = 1
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)
        def pass_data_through_model(model, early_stopping_iterations=None, use_cuda=False):
        # forward pass for given number of batches for model
            for _, (images_in_one_batch, _) in enumerate(data_loader):
                model(images_in_one_batch)

        quantsim = qsim.QuantizationSimModel(model=model, quant_scheme=params.quant_scheme,
                                             rounding_mode=params.round_mode,
                                             default_output_bw=params.act_bw,
                                             default_param_bw=params.weight_bw,
                                             in_place=False,
                                             dummy_input=torch.rand(1, 1, 28, 28))
        quantsim.compute_encodings(pass_data_through_model, None)
        layer = quantsim.model.conv2
        quant_dequant_weights = bias_correction.get_quantized_dequantized_weight(layer)
        assert quant_dequant_weights.shape == torch.Size([64, 32, 5, 5])

    def test_bias_correction_analytical_and_empirical(self):
        torch.manual_seed(10)
        model = MockMobileNetV1()
        model = model.eval()
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 224, 224))

        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf
                                 )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2,
                                             conv_bn_dict, perform_only_empirical_bias_corr=False)
        assert analytical_mock.call_count == 9
        assert empirical_mock.call_count == 9
        assert model.model[1][0].bias.detach().cpu().numpy() is not None

    def test_bias_correction_empirical_with_config_file(self):
        # Using a dummy extension of MNIST
        torch.manual_seed(10)
        model = mnist_model.Net().eval()
        model_copy = copy.deepcopy(model)
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))

        # Takes default config file
        params = qsim.QuantParams(weight_bw=4, act_bw=4, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf, config_file=None
                                  )
        with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_correct_bias') as empirical_mock:
            bias_correction.correct_bias(model, params, 2, data_loader, 2)

        assert empirical_mock.call_count == 4
        assert np.allclose(model.conv1.bias.detach().cpu().numpy(),
                           model_copy.conv1.bias.detach().cpu().numpy())

        assert model.conv2.bias.detach().cpu().numpy() is not None
        assert model.fc1.bias.detach().cpu().numpy() is not None

    def test_layer_selection_bn_based_bc_no_residual(self):
        model = MockMobileNetV1()
        model = model.eval()
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))
        conv_2 = model.model[1][0]
        assert conv_bn_dict[conv_2].output_bn is None
        assert len(conv_bn_dict) == 18

    def test_bias_update(self):
        np.random.seed(1)
        layer = nn.Conv2d(3, 10, 5)
        bias_before = layer.bias.clone()

        shape = (10, 10, 5, 5)
        reference_output_batch = np.random.randn(*shape)
        quantized_model_output_batch = np.random.randn(*shape)

        bias_correction.call_empirical_correct_bias(layer, reference_output_batch, quantized_model_output_batch)

        bias_after = layer.bias.detach()

        # Assert bias has changed after running bias correction
        assert not torch.allclose(bias_before, bias_after)

    def test_bias_correction_analytical_and_empirical_ignore_layer(self):

        torch.manual_seed(10)
        model = MockMobileNetV1()
        model = model.eval()
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 224, 224))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf
                                  )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

        layer = model.model[0][0]
        layers_to_ignore = [layer]

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                             perform_only_empirical_bias_corr=False,
                                             layers_to_ignore = layers_to_ignore)

        assert analytical_mock.call_count == 8 # one layer ignored
        assert empirical_mock.call_count == 9
        assert model.model[1][0].bias.detach().cpu().numpy() is not None

    def test_hybrid_bias_correction_for_transposed_conv2d(self):
        torch.manual_seed(10)
        model = TransposedConvModel()
        model = model.eval()
        data_loader = BatchIterator((1, 10, 4, 4))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf
                                  )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(10, 10, 4, 4))

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                             perform_only_empirical_bias_corr=False,
                                             layers_to_ignore=[])

        assert analytical_mock.call_count == 2 # one layer ignored
        assert empirical_mock.call_count == 0

    def test_bias_correction_for_depthwise_transposed_conv2d(self):
        torch.manual_seed(10)
        model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(10, 10, 3, groups=10),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(10, 10, 3),
            torch.nn.BatchNorm2d(10),
        )
        model = model.eval()

        data_loader = BatchIterator((1, 10, 4, 4))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf
                                  )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 10, 4, 4))

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                             perform_only_empirical_bias_corr=False,
                                             layers_to_ignore=[])

        assert analytical_mock.call_count == 2 # one layer ignored
        assert empirical_mock.call_count == 0

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    @pytest.mark.parametrize("perform_empirical", [True, False])
    def test_python_only_bias_correction(self, device, perform_empirical):
        torch.manual_seed(10)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, 3),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 10, 3),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
        )
        # Initialize BN params.
        torch.nn.init.normal_(model[1].weight)
        torch.nn.init.normal_(model[1].bias)
        torch.nn.init.normal_(model[4].weight)
        torch.nn.init.normal_(model[4].bias)

        model = model.eval().to(device)
        dataset_size = 2
        batch_size = 1
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 32, 32))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf)

        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 32, 32))
        params_before = [m.bias.clone() for m in conv_bn_dict.keys()]
        bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                     perform_only_empirical_bias_corr=perform_empirical)
        params_after = [m.bias.clone() for m in conv_bn_dict.keys()]
        assert not all(torch.equal(b_prev, b_after) for b_prev, b_after in zip(params_before, params_after))

    @pytest.mark.cuda
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    @pytest.mark.parametrize("activation_cls", [torch.nn.ReLU, torch.nn.ReLU6])
    def test_python_only_analytical(self, device, activation_cls):
        torch.manual_seed(10)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, 3),
            torch.nn.BatchNorm2d(10),
            activation_cls(),
            torch.nn.Conv2d(10, 10, 3),
            torch.nn.BatchNorm2d(10),
            activation_cls(),
        )
        # Initialize BN params.
        torch.nn.init.normal_(model[1].weight)
        torch.nn.init.normal_(model[1].bias)
        torch.nn.init.normal_(model[4].weight)
        torch.nn.init.normal_(model[4].bias)

        model = model.eval().to(device)
        dataset_size = 2
        batch_size = 1
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size,
                                              image_size=(3, 32, 32))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme=QuantScheme.post_training_tf)

        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 32, 32))
        params_before = [m.bias.clone() for m in conv_bn_dict.keys()]
        bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                     perform_only_empirical_bias_corr=False)
        params_after = [m.bias.clone() for m in conv_bn_dict.keys()]
        assert not all(torch.equal(b_prev, b_after) for b_prev, b_after in zip(params_before, params_after))
