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

""" AdaRound Weights Unit Test Cases """

import pytest
import tempfile
from pathlib import Path
import os
import json
import logging
from unittest.mock import patch
import torch
import torch.nn.functional as functional
from torchvision import models

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import calculate_delta_offset
from aimet_torch.v1.adaround.adaround_wrapper import AdaroundWrapper
from aimet_torch.utils import create_fake_data_loader, create_rand_tensors_given_shapes, get_device
from .models_ import test_models
from aimet_torch._base.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_torch.v1.adaround.adaround_weight import AdaroundParameters
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.adaround import Adaround
from aimet_torch.v2.nn import BaseQuantizationMixin

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


@pytest.fixture
def disable_activation_caching():
    # Disable caching activation data
    orig_flag = AdaroundOptimizer.is_activation_caching_enabled
    try:
        AdaroundOptimizer.is_activation_caching_enabled = False
        yield
    finally:
        AdaroundOptimizer.is_activation_caching_enabled = orig_flag


def dummy_forward_pass(model, inp_shape):
    """ Dummy forward pass"""
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(inp_shape))
    return output


class ConvOnlyModel(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(ConvOnlyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        return x


class UnusedModule(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(UnusedModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2, bias=False)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        return x


class OutOfSequenceModule(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(OutOfSequenceModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)

    def forward(self, *inputs):
        x = self.conv2(inputs[0])
        x = self.conv1(x)
        return x


class ConvTransposeNet(torch.nn.Module):
    """ Model with ConvTranspose2d layer """
    def __init__(self):

        super(ConvTransposeNet, self).__init__()
        self.trans_conv1 = torch.nn.ConvTranspose2d(3, 6, 3, groups=3)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.reul1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.trans_conv1(x)
        x = self.bn1(x)
        x = self.reul1(x)

        return x


class MultiDataLoaders:
    """
    A simple implementation for supporting two data loaders, can be extended
     to support more than two data loaders as well.
    """
    def __init__(self, data_loader1, data_loader2):
        self._dl1 = data_loader1
        self._dl2 = data_loader2
        self.batch_size = self._dl1.batch_size

    def __len__(self):
        return len(self._dl1) + len(self._dl2)

    def __iter__(self):
        """
        yields batches alternatively one after the other from provided data loaders
        """
        dl1_iter = iter(self._dl1)
        dl2_iter = iter(self._dl2)
        for dl1_batch, dl2_batch in zip(dl1_iter, dl2_iter):
            yield dl1_batch
            yield dl2_batch

        # yield from remaining non-exhausted data loader
        yield from dl1_iter
        yield from dl2_iter


def save_config_file_for_per_channel_quantization(target_dir: Path) -> Path:
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
            },
            "per_channel_quantization": "True",
        },
        "params": {"bias": {
            "is_quantized": "False"
        }},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    target_file = Path(target_dir, "quantsim_config.json")
    with open(target_file, 'w') as f:
        json.dump(quantsim_config, f)
    return target_file


class SplittableModel(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(SplittableModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1)
        self.conv4 = torch.nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(36, 12)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def save_config_file_for_checkpoints(checkpoints_config, config_file):
    with open(config_file, 'w') as f:
        json.dump(checkpoints_config, f)


class TwoConvModel(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(TwoConvModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)

    def forward(self, *inputs):
        #return self.conv1(inputs[0])
        #return self.relu(self.conv1(inputs[0]))
        x = self.conv1(inputs[0])
        x =  self.relu(x)
        return self.conv2(x)


class MultiBlockModel(torch.nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(MultiBlockModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.block1 = TwoConvModel()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.block2 = TwoConvModel()
        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.block1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.block2(x)
        return self.conv3(x)


class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """

    def test_apply_adaround(self):
        """ test apply_adaround end to end using tiny model """
        torch.manual_seed(10)
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        out_before_ada = dummy_forward_pass(model, input_shape)

        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, inp_tensor_list, params, tempdir, 'dummy')
            out_after_ada = dummy_forward_pass(ada_rounded_model, input_shape)
            print(out_after_ada.detach().cpu().numpy()[0, :])
            assert not torch.all(torch.eq(out_before_ada, out_after_ada))

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            param_keys = list(encoding_data.keys())
            print(param_keys)
            assert param_keys[0] == "conv1.weight"
            assert isinstance(encoding_data["conv1.weight"], list)

    def test_adaround_with_and_without_checkpoints_config(self):
        def dummy_fwd(model, inputs):
            return model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
        torch.manual_seed(10)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=1, batch_size=1, image_size=(3, 32, 32))

        net = SplittableModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=dummy_fwd)
        checkpoints_config = {
            "grouped_modules": {
                "0": ["conv1", "bn1", "relu1", "maxpool"],
                "1": ["conv2", "bn2", "relu2"],
                "2": ["conv3", "relu3", "avgpool"],
                "3": ["conv4", "flatten", "fc"],
            },
            "include_static_inputs": [
                "False",
                "False",
                "False",
                "False"
            ],
            "cache_on_cpu": "False"
        }
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, 'test_checkpoints.json')
            save_config_file_for_checkpoints(checkpoints_config, config_file)
            ada_rounded_model = Adaround.apply_adaround(model, inp_tensor_list, params, tempdir, 'dummy')
            ada_rounded_model_ckpts = Adaround.apply_adaround_with_cache(model, inp_tensor_list, params, tempdir, 'dummy_checkpoints',
                                                                         checkpoints_config=config_file)

            for (name, param), (name_ckpts, param_ckpts) in zip(ada_rounded_model.named_parameters(),
                                                                ada_rounded_model_ckpts.named_parameters()):
                assert name == name_ckpts
                assert torch.equal(param, param_ckpts)

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            with open(os.path.join(tempdir, 'dummy_checkpoints.encodings')) as json_file:
                encoding_data_ckpts = json.load(json_file)['param_encodings']
                print(encoding_data_ckpts)

            assert list(encoding_data.keys()) == list(encoding_data_ckpts.keys())

            for key in list(encoding_data.keys()):
                enc = encoding_data[key][0]
                enc_ckpts = encoding_data_ckpts[key][0]
                assert list(enc.keys()) == list(enc_ckpts.keys())
                # Check all encodings are match
                for k in list(enc.keys()):
                    assert enc[k] == enc_ckpts[k]

    def test_adaround_with_disjoint_checkpoints_config(self):
        """ Test disjoint checkpoint for two blocks model """
        def dummy_fwd(model, inputs):
            return model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
        torch.manual_seed(10)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=1, batch_size=1, image_size=(3, 32, 32))

        net = MultiBlockModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)

        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5,
                                    forward_fn=dummy_fwd)
        checkpoints_config = {
            "checkpoint_type": "disjoint",
            "cached_blocks": [
                "block1",
                "block2"
            ],
            "cache_on_cpu": "False"
        }
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, 'disjoint_checkpoints.json')
            save_config_file_for_checkpoints(checkpoints_config, config_file)
            ada_rounded_model = Adaround.apply_adaround(model, inp_tensor_list, params, tempdir, 'dummy')
            ada_rounded_model_ckpts = Adaround.apply_adaround_with_cache(model, inp_tensor_list, params, tempdir,
                                                                         'dummy_checkpoints',
                                                                         checkpoints_config=config_file)

            for (name, param), (name_ckpts, param_ckpts) in zip(ada_rounded_model.named_parameters(),
                                                                ada_rounded_model_ckpts.named_parameters()):
                assert name == name_ckpts
                assert torch.equal(param, param_ckpts)

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            with open(os.path.join(tempdir, 'dummy_checkpoints.encodings')) as json_file:
                encoding_data_ckpts = json.load(json_file)['param_encodings']
                print(encoding_data_ckpts)

            assert list(encoding_data.keys()) == list(encoding_data_ckpts.keys())

            for key in list(encoding_data.keys()):
                enc = encoding_data[key][0]
                enc_ckpts = encoding_data_ckpts[key][0]
                assert list(enc.keys()) == list(enc_ckpts.keys())
                # Check all encodings are match
                for k in list(enc.keys()):
                    assert enc[k] == enc_ckpts[k]

    def test_apply_adaround_per_channel(self):
        """ test apply_adaround end to end using tiny model when using per-channel mode """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        model = test_models.TinyModel().eval()

        input_shape = (1, 3, 32, 32)
        out_before_ada = dummy_forward_pass(model, input_shape)

        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tempdir:

            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            ada_rounded_model = Adaround.apply_adaround(model, inp_tensor_list, params, tempdir, 'dummy',
                                                        default_config_file=os.path.join(tempdir, 'quantsim_config.json'))

            out_after_ada = dummy_forward_pass(ada_rounded_model, input_shape)

            assert not torch.all(torch.eq(out_before_ada, out_after_ada))

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            param_keys = list(encoding_data.keys())
            print(param_keys)
            assert param_keys[0] == "conv1.weight"
            assert len(encoding_data["conv1.weight"]) == 32

            # Test that encodings can be correctly loaded into sim
            sim = QuantizationSimModel(ada_rounded_model, torch.randn(input_shape),
                                       config_file=os.path.join(tempdir, 'quantsim_config.json'))
            sim.set_and_freeze_param_encodings(os.path.join(tempdir, 'dummy.encodings'))

            # Test that we after computing encodings, the param encodings are the same
            assert sim.model.conv1.param_quantizers["weight"].is_initialized()
            scale_before_compute_enc = sim.model.conv1.param_quantizers["weight"].get_scale().clone()
            sim.compute_encodings(dummy_forward_pass, input_shape)
            assert torch.equal(scale_before_compute_enc, sim.model.conv1.param_quantizers["weight"].get_scale())

    def test_before_opt_MSE(self):
        """  Check MSE of the output activations at the beginning of the optimization """
        model = test_models.TinyModel().eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        out_float32 = model.conv1(dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input, default_param_bw=4)

        for quant_wrapper in sim.model.modules():
            if isinstance(quant_wrapper, BaseQuantizationMixin):
                # Adaround requires input and output quantizers to be disabled
                quant_wrapper.input_quantizers = torch.nn.ModuleList([None for _ in quant_wrapper.input_quantizers])
                quant_wrapper.output_quantizers = torch.nn.ModuleList([None for _ in quant_wrapper.output_quantizers])

                for name, param in quant_wrapper.get_original_module().named_parameters():
                    # Compute encodings for parameters, needed for initializing Adaround quantizers
                    param_quantizer = quant_wrapper.param_quantizers[name]

                    if not param_quantizer:
                        continue

                    with param_quantizer.compute_encodings():
                        _ = param_quantizer(param.data)

        quant_module = sim.model.conv1

        # Get output using weight quantized with 'nearest' rounding mode, asymmetric encoding
        out_rounding_to_nearest = quant_module(dummy_input)

        # replace the tensor quantizer
        Adaround._replace_quantization_layer(sim.model, 'conv1')  # pylint: disable=protected-access
        quant_module = sim.model.conv1
        out_soft_quant = quant_module(dummy_input)

        soft_quant_rec = functional.mse_loss(out_soft_quant, out_float32)
        print('Reconstruction error before optimization (soft quant): ', float(soft_quant_rec))
        assert soft_quant_rec < 1

        # enable hard rounding
        quant_module.use_soft_rounding = False
        out_hard_quant = quant_module(dummy_input)
        hard_quant_rec = functional.mse_loss(out_hard_quant, out_rounding_to_nearest)

        print('Reconstruction error before optimization (hard quant): ', float(hard_quant_rec))
        assert hard_quant_rec < 1

    def test_adaround_conv_only_model_weight_binning(self):
        """ test AdaRound weight binning """
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = ConvOnlyModel().eval()
        model = net.to(torch.device('cpu'))
        param_bit_width = 4
        delta, offset = calculate_delta_offset(float(torch.min(model.conv1.weight)),
                                               float(torch.max(model.conv1.weight)),
                                               param_bit_width,
                                               use_symmetric_encodings=False,
                                               use_strict_symmetric=False)
        print(delta, offset)

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=10,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model, inp_tensor_list, params, path=tempdir, filename_prefix='dummy',
                                                default_param_bw=param_bit_width,
                                                default_quant_scheme=QuantScheme.post_training_tf,
                                                default_config_file=None)
            assert torch.allclose(model.conv1.weight, ada_model.conv1.weight, atol=2*delta)


    def test_unused_module_model(self):
        """ test AdaRound weight binning """
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = UnusedModule().eval()
        model = net.to(torch.device('cpu'))
        param_bit_width = 4
        delta, offset = calculate_delta_offset(float(torch.min(model.conv1.weight)),
                                               float(torch.max(model.conv1.weight)),
                                               param_bit_width,
                                               use_symmetric_encodings=False,
                                               use_strict_symmetric=False)
        print(delta, offset)

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=10,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model, inp_tensor_list, params, path=tempdir, filename_prefix='dummy',
                                                default_param_bw=param_bit_width,
                                                default_quant_scheme=QuantScheme.post_training_tf,
                                                default_config_file=None)
            # Only Conv1 must be AdaRounded.
            assert torch.allclose(model.conv1.weight, ada_model.conv1.weight, atol=2*delta)

            # Conv2 weights are not AdaRounded and should be the same
            assert torch.equal(model.conv2.weight, ada_model.conv2.weight)

    def test_out_of_sequence_module_model(self):
        """ test  out of sequence modules """
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = OutOfSequenceModule().eval()
        model = net.to(torch.device('cpu'))
        param_bit_width = 4
        delta, offset = calculate_delta_offset(float(torch.min(model.conv1.weight)),
                                               float(torch.max(model.conv1.weight)),
                                               param_bit_width,
                                               use_symmetric_encodings=False,
                                               use_strict_symmetric=False)
        print(delta, offset)

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=10,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model, inp_tensor_list, params, path=tempdir, filename_prefix='dummy',
                                                default_param_bw=param_bit_width,
                                                default_quant_scheme=QuantScheme.post_training_tf,
                                                default_config_file=None)
            # Both the modules must be AdaRounded
            assert torch.allclose(model.conv1.weight, ada_model.conv1.weight, atol=2*delta)
            assert torch.allclose(model.conv2.weight, ada_model.conv2.weight, atol=2*delta)

    def test_conv_transpose_2d_model(self):
        """ test a model that has a ConveTranspose2d module """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 24, 24))

        net = ConvTransposeNet().eval()
        model = net.to(torch.device('cpu'))

        param_bit_width = 4
        delta, offset = calculate_delta_offset(float(torch.min(model.trans_conv1.weight)),
                                               float(torch.max(model.trans_conv1.weight)),
                                               param_bit_width,
                                               use_symmetric_encodings=False,
                                               use_strict_symmetric=False)
        logger.info("For the ConvTranspose2d layer's weights, delta = %f, offset = %f", delta, offset)

        input_shape = (1, 3, 24, 24)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        # Test Forward Pass
        _ = model(*inp_tensor_list)

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=10,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model, inp_tensor_list, params, path=tempdir, filename_prefix='dummy',
                                                default_param_bw=param_bit_width,
                                                default_quant_scheme=QuantScheme.post_training_tf,
                                                default_config_file=None)

            # Test that forward pass works for the AdaRounded model
            _ = ada_model(*inp_tensor_list)

            # Assert that AdaRounded weights are not rounded more than one delta value up or down
            assert torch.allclose(model.trans_conv1.weight, ada_model.trans_conv1.weight, atol=1*delta)

    def test_conv_transpose_2d_model_per_channel(self):
        """ test a model that has a ConvTranspose2d module in per channel mode """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 24, 24))

        net = ConvTransposeNet().eval()
        model = net.to(torch.device('cpu'))

        param_bit_width = 4
        input_shape = (1, 3, 24, 24)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        # Test Forward Pass
        _ = model(*inp_tensor_list)

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=100,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_config_file_for_per_channel_quantization(Path(temp_dir))
            ada_model = Adaround.apply_adaround(model, inp_tensor_list, params, path=temp_dir, filename_prefix='dummy',
                                                default_param_bw=param_bit_width,
                                                default_quant_scheme=QuantScheme.post_training_tf,
                                                default_config_file=os.path.join(temp_dir, 'quantsim_config.json'))

            # Test that forward pass works for the AdaRounded model
            _ = ada_model(*inp_tensor_list)

            with open(os.path.join(temp_dir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            assert len(encoding_data['trans_conv1.weight']) == 2

    def test_overriding_default_parameter_bitwidths(self):
        """ Override the default parameter bitwidths for a model """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        # Create the override list with non-default parameter bitwidths 8 and 16
        param_bw_override_list = [(model.conv2, 8), (model.conv4, 16)]

        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model=model, dummy_input=inp_tensor_list, params=params,
                                                        path=tempdir, filename_prefix='dummy',
                                                        param_bw_override_list=param_bw_override_list)

            # Read exported param encodings JSON file
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            # Verify Conv2 weight encoding bitwidth is set to 8
            conv2_encoding = encoding_data["conv2.weight"][0]
            assert conv2_encoding.get('bitwidth') == 8

            # Verify Conv4 weight encoding bitwidth is set to 16
            conv4_encoding = encoding_data["conv4.weight"][0]
            assert conv4_encoding.get('bitwidth') == 16

    def test_overriding_default_parameter_bitwidths_with_empty_list(self):
        """ Override the default parameter bitwidths for a model """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        # Keep the parameter override list as empty
        param_bw_override_list = []
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model=model, dummy_input=inp_tensor_list, params=params,
                                                        path=tempdir, filename_prefix='dummy',
                                                        param_bw_override_list=param_bw_override_list)

            # Read exported param encodings JSON file
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            # Verify Conv2 weight encoding bitwidth is set to the default value of 4
            conv2_encoding = encoding_data["conv2.weight"][0]
            assert conv2_encoding.get('bitwidth') == 4

            # Verify Conv4 weight encoding bitwidth is set to the default value of 4
            conv4_encoding = encoding_data["conv4.weight"][0]
            assert conv4_encoding.get('bitwidth') == 4

    def test_ignoring_ops_for_quantization(self):
        """ Test ignoring certain layers from being quantized. """

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        sim = QuantizationSimModel(model, dummy_input=inp_tensor_list, default_param_bw=8)
        # sim.compute_encodings(dummy_forward_pass, forward_pass_callback_args=input_shape)

        # Before modifying the QuantSim, verify layers are wrapped
        assert isinstance(sim.model.maxpool, BaseQuantizationMixin)
        assert isinstance(sim.model.avgpool, BaseQuantizationMixin)
        assert isinstance(sim.model.conv2, BaseQuantizationMixin)
        assert isinstance(sim.model.relu3, BaseQuantizationMixin)

        # Skip the maxpool and avgpool layers.
        ignore_quant_ops_list = [model.maxpool, model.avgpool]
        Adaround._exclude_modules(model, sim, ignore_quant_ops_list)
        sim.compute_encodings(dummy_forward_pass, forward_pass_callback_args=input_shape)

        # Since maxpool and avgpool are skipped, they shouldn't be wrapped StaticGridQuantWrapper.
        assert not isinstance(sim.model.maxpool, BaseQuantizationMixin)
        assert not isinstance(sim.model.avgpool, BaseQuantizationMixin)

        # conv2 and relu3 must be remain wrapped in StaticGridQuantWrapper
        assert isinstance(sim.model.conv2, BaseQuantizationMixin)
        assert isinstance(sim.model.relu3, BaseQuantizationMixin)

    def test_apply_adaround_with_ignore_list(self):
        """ Test the apply_adaround() API with ignore list """

        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        ignore_quant_ops_list = [model.relu1, model.bn2]
        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model=model, dummy_input=inp_tensor_list, params=params,
                                                path=tempdir, filename_prefix='dummy',
                                                ignore_quant_ops_list=ignore_quant_ops_list)

            # Make sure model forwatd pass works.
            _ = ada_model(*inp_tensor_list)

            # Read exported param encodings JSON file
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            # Verify Conv2 weight encoding bitwidth is set to the default value of 4
            conv2_encoding = encoding_data["conv2.weight"][0]
            assert conv2_encoding.get('bitwidth') == 4

    def test_multi_data_loaders_example(self):
        """ Test order of getting data for multi data loader example """
        data_loader_1 = create_fake_data_loader(32, 16, image_size=(1, 2, 2))
        data_loader_2 = create_fake_data_loader(64, 16, image_size=(1, 3, 3))

        multi_data_loader = MultiDataLoaders(data_loader_1, data_loader_2)
        iterator = iter(multi_data_loader)
        batch, _ = next(iterator)   # batch1 from dl1
        assert batch.shape == (16, 1, 2, 2)

        batch, _ = next(iterator)   # batch1 from dl2
        assert batch.shape == (16, 1, 3, 3)

        batch, _ = next(iterator)   # batch2 from dl1
        assert batch.shape == (16, 1, 2, 2)

        batch, _ = next(iterator)   # batch2 from dl2
        assert batch.shape == (16, 1, 3, 3)

        batch, _ = next(iterator)   # batch3 from dl2
        assert batch.shape == (16, 1, 3, 3)

        batch, _ = next(iterator)  # batch4 from dl2
        assert batch.shape == (16, 1, 3, 3)

        with pytest.raises(StopIteration):
            batch, _ = next(iterator)  # exhausting dl2
            assert batch.shape == (16, 1, 3, 3)

    def test_apply_adaround_with_multi_data_loaders(self):
        """ Test Adaround with multiple data loaders """

        data_loader_1 = create_fake_data_loader(32, 16, image_size=(3, 32, 32))
        data_loader_2 = create_fake_data_loader(64, 16, image_size=(3, 32, 32))

        multi_data_loader = MultiDataLoaders(data_loader_1, data_loader_2)

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        input_shape = (1, 3, 32, 32)
        inp_tensor_list = create_rand_tensors_given_shapes(input_shape, get_device(model))

        params = AdaroundParameters(data_loader=multi_data_loader, num_batches=4, default_num_iterations=5)

        ignore_quant_ops_list = [model.relu1, model.bn2]
        with tempfile.TemporaryDirectory() as tempdir:
            ada_model = Adaround.apply_adaround(model=model, dummy_input=inp_tensor_list, params=params,
                                                path=tempdir, filename_prefix='dummy',
                                                ignore_quant_ops_list=ignore_quant_ops_list)

            # Make sure model forwatd pass works.
            _ = ada_model(*inp_tensor_list)

            # Read exported param encodings JSON file
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            # Verify Conv2 weight encoding bitwidth is set to the default value of 4
            conv2_encoding = encoding_data["conv2.weight"][0]
            assert conv2_encoding.get('bitwidth') == 4

    @pytest.mark.cuda
    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_apply_adaround_using_gpu(self, dtype):
        """ test apply_adaround end to end using tiny model """

        torch.manual_seed(10)
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(device=torch.device('cuda'), dtype=dtype)

        input_shape = (1, 3, 32, 32)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        dummy_input = [x.to(dtype=dtype) for x in dummy_input]
        out_before_ada = model(*dummy_input)

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=50)
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, dummy_input, params, tempdir, 'dummy')
            out_after_ada = ada_rounded_model(*dummy_input)

            assert not torch.all(torch.eq(out_before_ada, out_after_ada))

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            param_keys = list(encoding_data.keys())
            print(param_keys)
            assert param_keys[0] == "conv1.weight"
            assert isinstance(encoding_data["conv1.weight"], list)

    @pytest.mark.cuda
    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_apply_adaround_using_gpu_caching_disabled(self, dtype, disable_activation_caching):
        """ test apply_adaround end to end using tiny model """

        torch.manual_seed(10)
        AimetLogger.set_level_for_all_areas(logging.INFO)

        # create fake data loader with image size (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(device=torch.device('cuda'), dtype=dtype)

        input_shape = (1, 3, 32, 32)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        dummy_input = [x.to(dtype=dtype) for x in dummy_input]
        out_before_ada = model(*dummy_input)

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(model, dummy_input, params, tempdir, 'dummy')
            out_after_ada = ada_rounded_model(*dummy_input)

            assert not torch.all(torch.eq(out_before_ada, out_after_ada))

            # Test export functionality
            with open(os.path.join(tempdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']
                print(encoding_data)

            param_keys = list(encoding_data.keys())
            print(param_keys)
            assert param_keys[0] == "conv1.weight"
            assert isinstance(encoding_data["conv1.weight"], list)

    def test_adaround_with_modules_to_exclude(self):
        """ test adaround API with modules_to_exclude list with both leaf and non-leaf modules """
        model = models.resnet18().eval()
        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(input_shape)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=input_shape[1:])
        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)
        with tempfile.TemporaryDirectory() as tempdir:
            _ = Adaround.apply_adaround(model, dummy_input, params, path=tempdir, filename_prefix='resnet18',
                                        ignore_quant_ops_list=[model.layer1, model.layer2, model.layer3,
                                                               model.layer4, model.fc])
            with open(os.path.join(tempdir, 'resnet18.encodings')) as json_file:
                encoding_data = json.load(json_file)['param_encodings']

            assert len(encoding_data) == 1 # Only model.conv1 layer is adarounded.

    def test_adaround_default_values(self):
        model = models.resnet18().eval()
        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(input_shape)
        batch_size = 16
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=batch_size, image_size=input_shape[1:])
        params = AdaroundParameters(data_loader=data_loader, num_batches=4)

        with tempfile.TemporaryDirectory() as tempdir:
            for param_bw in (8, 16, 9):
                with patch.object(AdaroundOptimizer, "adaround_module") as adaround_module_fn_mock:
                    _ = Adaround.apply_adaround(model, dummy_input, params, path=tempdir, filename_prefix='resnet18',
                                                default_param_bw=8)
                _, _, _, _, _, _, _, opt_params, _ = adaround_module_fn_mock.call_args[0]
                # If adaround is performed with sub-8 bit weights, the default num_iterations should be 10K
                assert opt_params.num_iterations == 10000

            for param_bw in (4, 7):
                with patch.object(AdaroundOptimizer, "adaround_module") as adaround_module_fn_mock:
                    _ = Adaround.apply_adaround(model, dummy_input, params, path=tempdir, filename_prefix='resnet18',
                                                default_param_bw=param_bw)
                # If adaround is performed with sub-8 bit weights, the default num_iterations should be 15K
                _, _, _, _, _, _, _, opt_params, _ = adaround_module_fn_mock.call_args[0]
                assert opt_params.num_iterations == 15000

    def test_adaround_restore_tensor_quantizer_after_folding(self):
        torch.manual_seed(10)

        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=(3, 32, 32))

        net = test_models.TinyModel().eval()
        model = net.to(torch.device('cpu'))

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5)

        adaround_module = AdaroundOptimizer.adaround_module

        def _adaround_module(module, wrapper, model, sim_model, *args, **kwargs):
            # Assert all the wrappers are BaseQuantizationMixin
            # except for the wrapper that is currently being optimized
            for module_ in sim_model.modules():
                if not isinstance(module_, (BaseQuantizationMixin, AdaroundWrapper)):
                    continue

                if module_ is wrapper:
                    expected_weight_quantizer_cls = AdaroundWrapper
                else:
                    expected_weight_quantizer_cls = BaseQuantizationMixin

                assert isinstance(module_, expected_weight_quantizer_cls)

            return adaround_module(module, wrapper, model, sim_model, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(AdaroundOptimizer, 'adaround_module', _adaround_module):
                _ = Adaround.apply_adaround(model, torch.randn((1, 3, 32, 32)), params, tempdir, 'dummy')
