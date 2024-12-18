# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import tempfile
from pathlib import Path

import pytest
import unittest
import copy
import time

import torch
import torch.nn as nn
from aimet_torch.v1.quantsim import QuantizationSimModel, QuantScheme
import aimet_torch._base.nn.modules.custom as aimet_modules
from aimet_common.utils import AimetLogger
from ..models import mnist_torch_model


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class ModelWithTwoInputsOneToAdd(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputsOneToAdd, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(10, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.add = aimet_modules.Add()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x1 = self.relu1_b(self.maxpool1_b(self.conv1_b(x1)))

        x = self.add(x1, x2)

        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


def forward_pass(model, args):
    torch.manual_seed(1)
    device = next(model.parameters()).device

    rand_input = torch.randn((10, 1, 28, 28)).to(device)
    model(rand_input)


class QuantizerCpuGpu(unittest.TestCase):

    @pytest.mark.cuda
    def test_and_compare_quantizer_no_fine_tuning_CPU_and_GPU(self):

        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        dummy_input = torch.rand(1, 1, 28, 28)
        dummy_input_cuda = dummy_input.cuda()

        start_time = time.time()

        # create model on CPU
        model_cpu = mnist_torch_model.Net().to('cpu').eval()

        model_gpu = copy.deepcopy(model_cpu).to('cuda')
        cpu_sim_model = QuantizationSimModel(model_cpu, quant_scheme='tf', in_place=True,
                                             dummy_input=dummy_input)
        # Quantize
        cpu_sim_model.compute_encodings(forward_pass, None)

        print("Encodings for cpu model calculated")
        print("Took {} secs".format(time.time() - start_time))
        start_time = time.time()

        # create model on GPU
        gpu_sim_model = QuantizationSimModel(model_gpu, quant_scheme='tf', in_place=True,
                                             dummy_input=dummy_input_cuda)
        # Quantize
        gpu_sim_model.compute_encodings(forward_pass, None)

        print("Encodings for gpu model calculated")
        print("Took {} secs".format(time.time() - start_time))

        # check the encodings only min and max
        # Test that first and second are approximately (or not approximately)
        # equal by computing the difference, rounding to the given number of
        # decimal places (default 7), and comparing to zero. Note that these
        # methods round the values to the given number of decimal places
        # (i.e. like the round() function) and not significant digits
        # excluding fc1 since it is part of Matmul->Relu supergroup
        # can't use assertEqual for FC2, so using assertAlmostEquals for FC2
        self.assertAlmostEqual(model_gpu.conv1.output_quantizers[0].encoding.min,
                               model_cpu.conv1.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.conv1.output_quantizers[0].encoding.max,
                               model_cpu.conv1.output_quantizers[0].encoding.max, delta=0.001)

        self.assertAlmostEqual(model_gpu.conv2.output_quantizers[0].encoding.min,
                               model_cpu.conv2.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.conv2.output_quantizers[0].encoding.max,
                               model_cpu.conv2.output_quantizers[0].encoding.max, delta=0.001)

        self.assertAlmostEqual(model_gpu.fc2.output_quantizers[0].encoding.min,
                               model_cpu.fc2.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.fc2.output_quantizers[0].encoding.max,
                               model_cpu.fc2.output_quantizers[0].encoding.max, delta=0.001)

        with tempfile.TemporaryDirectory() as tmp_dir:
            gpu_sim_model.export(tmp_dir, "quantizer_no_fine_tuning__GPU", dummy_input)
            cpu_sim_model.export(tmp_dir, "quantizer_no_fine_tuning__CPU", dummy_input)

        self.assertEqual(torch.device('cuda:0'), next(model_gpu.parameters()).device)
        self.assertEqual(torch.device('cpu'), next(model_cpu.parameters()).device)

    @pytest.mark.cuda
    def test_qc_trainable_wrapper_for_model_with_multiple_inputs_with_one_add(self):
        # NOTE: Use asymmetric quantization for parameter, which have gradients both encoding min/max
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = Path(tmp_dir, "quantsim_config.json")
            with open(config_file_path, "w") as f:
                json.dump(quantsim_config, f)

            dummy_input = (torch.rand(32, 1, 100, 100).cuda(), torch.rand(32, 10, 22, 22).cuda())

            def forward_pass(sim_model, _):
                sim_model.eval()
                with torch.no_grad():
                    sim_model(*dummy_input)

            model = ModelWithTwoInputsOneToAdd().cuda()

            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    config_file=config_file_path)
            # Enable input parameters to add (multiple input parameter exist)
            sim.model.add.input_quantizers[0].enabled = True
            sim.model.add.input_quantizers[1].enabled = True

            sim.compute_encodings(forward_pass, forward_pass_callback_args=None)

            assert len(sim.model.add.input_quantizers) == 2

            out = sim.model(*dummy_input)
            for _, params in sim.model.named_parameters():
                assert params.grad is None

            optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
            loss = out.flatten().sum()
            loss.backward()
            optimizer.step()
            # All parameters should have a gradient
            for params in sim.model.parameters():
                assert params.grad is not None
