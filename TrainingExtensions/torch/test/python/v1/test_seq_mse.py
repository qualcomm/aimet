# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import json
import pytest
import tempfile
from pathlib import Path
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

from aimet_torch.utils import create_fake_data_loader
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper, QuantScheme
from aimet_torch.v1.seq_mse import  apply_seq_mse, get_candidates, optimize_module, SeqMseParams
from ..models.mnist_torch_model import Net

@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 1, 28, 28))


@pytest.fixture(scope="session")
def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([dummy_input[0, :] for _ in range(32)])
    return DataLoader(dataset)


def calibrate(model, inputs):

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    model.eval()
    with torch.no_grad():
        model(*inputs)


def save_config_file_for_checkpoints(target_dir: Path) -> Path:
    checkpoints_config = {
        "grouped_modules": {
            "0": ["conv1", "bn1", "relu1", "maxpool"],
            "1": ["conv2", "bn2", "relu2"],
            "2": ["conv3", "relu3", "avgpool"],
            "3": ["conv4", "flatten", "fc1", "fc2"],
        },
        "include_static_inputs": [
            "False",
            "False",
            "False",
            "False"
        ],
        "cache_on_cpu": "False"
    }

    target_file = Path(target_dir, 'test_checkpoints.json')
    with open(target_file, 'w') as f:
        json.dump(checkpoints_config, f)
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
        self.fc1 = torch.nn.Linear(36, 12)
        self.fc2 = torch.nn.Linear(12, 10)

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
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestSeqMse:

    def test_seq_mse(self):
        """ test get_candidates() """
        torch.manual_seed(0)
        linear = torch.nn.Linear(2, 4)
        x_max = torch.max(linear.weight.abs(), dim=1)[0]
        x_min = None
        candidates = get_candidates(20, x_max, x_min)
        for cand_max, cand_min in candidates:
            assert list(cand_max.size())[0] == linear.out_features
            assert list(cand_min.size())[0] == linear.out_features

    @pytest.mark.parametrize("enable_pcq", [True, False])
    @pytest.mark.parametrize("param_bw", [2, 31])
    @pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
    def test_optimize_module_linear(self, enable_pcq, param_bw, loss_fn):
        """ test optimize module for linear """
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 128)
        wrapper = StaticGridQuantWrapper(linear, param_bw, 16, 'nearest', QuantScheme.post_training_tf)
        wrapper.input_quantizers[0].enabled = False
        wrapper.output_quantizers[0].enabled = False
        if enable_pcq:
            wrapper.enable_per_channel_quantization()

        xq = torch.randn(32, 4, 32, 64)
        wrapper.param_quantizers['weight'].reset_encoding_stats()
        wrapper.param_quantizers['weight'].update_encoding_stats(wrapper.weight.data)
        wrapper.param_quantizers['weight'].compute_encoding()
        before = wrapper.param_quantizers['weight'].encoding
        params = SeqMseParams(num_batches=32, loss_fn=loss_fn)
        optimize_module(wrapper, xq, xq, params)
        after = wrapper.param_quantizers['weight'].encoding

        # If we use higher param_bw (for example 16, 31), then it should always choose larger candidates so
        # before and after param encodings should be almost same.
        if param_bw == 31:
            if enable_pcq:
                assert numpy.isclose(before[0].min, after[0].min)
                assert numpy.isclose(before[0].max, after[0].max)
            else:
                assert numpy.isclose(before.min, after.min)
                assert numpy.isclose(before.max, after.max)
        else:
            if enable_pcq:
                assert not numpy.isclose(before[0].min, after[0].min)
                assert not numpy.isclose(before[0].max, after[0].max)
            else:
                assert not numpy.isclose(before.min, after.min)
                assert not numpy.isclose(before.max, after.max)

    @pytest.mark.parametrize("enable_pcq", [True, False])
    @pytest.mark.parametrize("param_bw", [2, 31])
    @pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
    def test_optimize_module_conv(self, enable_pcq, param_bw, loss_fn):
        """ test optimize module for linear """
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 32, 3)
        wrapper = StaticGridQuantWrapper(conv, param_bw, 16, 'nearest', QuantScheme.post_training_tf)
        wrapper.input_quantizers[0].enabled = False
        wrapper.output_quantizers[0].enabled = False
        if enable_pcq:
            wrapper.enable_per_channel_quantization()

        xq = torch.randn(32, 1, 3, 10, 10)
        wrapper.param_quantizers['weight'].reset_encoding_stats()
        wrapper.param_quantizers['weight'].update_encoding_stats(wrapper.weight.data)
        wrapper.param_quantizers['weight'].compute_encoding()
        before = wrapper.param_quantizers['weight'].encoding
        params = SeqMseParams(num_batches=32, loss_fn=loss_fn)
        optimize_module(wrapper, xq, xq, params)
        after = wrapper.param_quantizers['weight'].encoding

        # If we use higher param_bw (for example 16, 31), then it should always choose larger candidates so
        # before and after param encodings should be almost same.
        if param_bw == 31:
            if enable_pcq:
                assert numpy.isclose(before[0].min, after[0].min)
                assert numpy.isclose(before[0].max, after[0].max)
            else:
                assert numpy.isclose(before.min, after.min)
                assert numpy.isclose(before.max, after.max)
        else:
            if enable_pcq:
                assert not numpy.isclose(before[0].min, after[0].min)
                assert not numpy.isclose(before[0].max, after[0].max)
            else:
                assert not numpy.isclose(before.min, after.min)
                assert not numpy.isclose(before.max, after.max)

    @pytest.mark.cuda()
    @pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
    @pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
    @pytest.mark.parametrize("qscheme", [QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init])
    def test_apply_seq_mse(self, unlabeled_data_loader, inp_symmetry, loss_fn, qscheme):
        """ test apply_seq_mse end-to-end """
        torch.manual_seed(0)
        model = Net().eval().cuda()
        dummy_input = torch.randn(1, 1, 28, 28).cuda()
        sim = QuantizationSimModel(model, dummy_input, default_param_bw=4, quant_scheme=qscheme)
        params = SeqMseParams(num_batches=2, inp_symmetry=inp_symmetry, loss_fn=loss_fn)
        apply_seq_mse(model, sim, unlabeled_data_loader, params)
        assert sim.model.fc1.param_quantizers['weight'].is_encoding_frozen
        assert sim.model.fc2.param_quantizers['weight'].is_encoding_frozen
        assert sim.model.conv1.param_quantizers['weight'].is_encoding_frozen
        assert sim.model.conv2.param_quantizers['weight'].is_encoding_frozen

        # Compute encodings for all the activations and remaining non-supported modules
        enc_before = sim.model.fc1.param_quantizers['weight'].encoding
        sim.compute_encodings(calibrate, dummy_input)
        enc_after = sim.model.fc1.param_quantizers['weight'].encoding
        assert enc_before.max == enc_after.max
        assert numpy.abs(enc_before.min - enc_after.min) < 1.05 * numpy.abs(enc_before.delta)

    @pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
    @pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
    @pytest.mark.parametrize("qscheme", [QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init])
    def test_seq_mse_with_and_without_checkpoints_config(self, inp_symmetry, loss_fn, qscheme):
        """ test apply_seq_mse end-to-end with and without checkpoints configs """
        torch.manual_seed(0)

        data_loader = create_fake_data_loader(dataset_size=2, batch_size=1, image_size=(3, 32, 32))
        model = SplittableModel().eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        sim_without = QuantizationSimModel(model, dummy_input, default_param_bw=4,
                                           quant_scheme=qscheme)
        sim_with = QuantizationSimModel(model, dummy_input, default_param_bw=4,
                                        quant_scheme=qscheme)
        params = SeqMseParams(num_batches=2, inp_symmetry=inp_symmetry, loss_fn=loss_fn)

        # Apply Sequential MSE without checkpoints config
        apply_seq_mse(model, sim_without, data_loader, params, modules_to_exclude=[model.fc1])
        assert not sim_without.model.fc1.param_quantizers['weight'].is_encoding_frozen
        assert sim_without.model.fc2.param_quantizers['weight'].is_encoding_frozen
        without_checkpoints_enc = sim_without.model.fc2.param_quantizers['weight'].encoding

        # Apply Sequential MSE with checkpoints config
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoints_config = save_config_file_for_checkpoints(Path(tmp_dir))
            apply_seq_mse(model, sim_with, data_loader, params, checkpoints_config=checkpoints_config,
                        modules_to_exclude=[model.fc1])
            assert not sim_with.model.fc1.param_quantizers['weight'].is_encoding_frozen
            assert sim_with.model.fc2.param_quantizers['weight'].is_encoding_frozen
            with_checkpoints_enc = sim_with.model.fc2.param_quantizers['weight'].encoding

        # encodings should be bit-exact
        assert without_checkpoints_enc.min == with_checkpoints_enc.min
        assert without_checkpoints_enc.max == with_checkpoints_enc.max
        assert without_checkpoints_enc.delta == with_checkpoints_enc.delta
        assert without_checkpoints_enc.offset == with_checkpoints_enc.offset

    @pytest.mark.parametrize("qscheme", [QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init])
    def test_apply_seq_mse_with_modules_to_exclude(self, unlabeled_data_loader, qscheme):
        """ test apply_seq_mse end-to-end with exclusion list """
        torch.manual_seed(0)
        model = Net().eval()
        dummy_input = torch.randn(1, 1, 28, 28)
        sim = QuantizationSimModel(model, dummy_input, default_param_bw=4, quant_scheme=qscheme)
        params = SeqMseParams(num_batches=2)
        apply_seq_mse(model, sim, unlabeled_data_loader, params, modules_to_exclude=[model.fc1])
        assert not sim.model.fc1.param_quantizers['weight'].is_encoding_frozen
        assert sim.model.fc2.param_quantizers['weight'].is_encoding_frozen
