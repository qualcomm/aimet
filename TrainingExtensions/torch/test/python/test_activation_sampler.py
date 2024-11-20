# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Unit tests for Adaround """

import unittest
import logging
import torch

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper, StaticGridQuantWrapper
from models.test_models import TinyModel
from aimet_torch.utils import create_fake_data_loader
from aimet_torch._base.adaround.activation_sampler import ActivationSampler
from aimet_torch.v1.adaround.adaround_wrapper import AdaroundWrapper

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class TestAdaroundActivationSampler(unittest.TestCase):
    """
    Adaround unit tests
    """
    def test_activation_sampler_conv(self):
        """ Test ActivationSampler for a Conv module """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 32, 32), quant_scheme='tf_enhanced',
                                   default_param_bw=4)

        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                for quantizer in module.input_quantizers + module.output_quantizers:
                    quantizer.enabled = False
                    quantizer.enabled = False

        for quantizer in sim.model.conv1.input_quantizers + sim.model.conv1.output_quantizers:
            self.assertFalse(quantizer.encoding)
        self.assertTrue(sim.model.conv1.param_quantizers['weight'])

        dataset_size = 100
        batch_size = 10
        image_size = (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size, batch_size, image_size)
        possible_batches = dataset_size // batch_size
        def forward_fn(model, inputs):
            inputs, _ = inputs
            model(inputs)

        act_sampler = ActivationSampler(model.conv1, sim.model.conv1, model, sim.model, forward_fn)
        quant_inp, orig_out = act_sampler.sample_and_place_all_acts_on_cpu(data_loader)

        self.assertEqual(list(quant_inp.shape), [batch_size * possible_batches, 3, 32, 32])
        self.assertEqual(list(orig_out.shape), [batch_size * possible_batches, 32, 18, 18])

    def test_activation_sampler_fully_connected_module(self):
        """ Test ActivationSampler for a fully connected module """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 32, 32), quant_scheme='tf_enhanced',
                                   default_param_bw=4)

        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                for quantizer in module.input_quantizers + module.output_quantizers:
                    quantizer.enabled = False
                    quantizer.enabled = False

        for quantizer in sim.model.conv1.input_quantizers + sim.model.conv1.output_quantizers:
            self.assertFalse(quantizer.encoding)
        self.assertTrue(sim.model.fc.param_quantizers['weight'])

        dataset_size = 100
        batch_size = 10
        image_size = (3, 32, 32)
        possible_batches = dataset_size // batch_size
        data_loader = create_fake_data_loader(dataset_size, batch_size, image_size)
        def forward_fn(model, inputs):
            inputs, _ = inputs
            model(inputs)

        act_sampler = ActivationSampler(model.fc, sim.model.fc, model, sim.model, forward_fn)
        quant_inp, orig_out = act_sampler.sample_and_place_all_acts_on_cpu(data_loader)

        self.assertEqual(list(quant_inp.shape), [batch_size * possible_batches, 36])
        self.assertEqual(list(orig_out.shape), [batch_size * possible_batches, 12])


    def test_adaround_tensor_quantizer(self):
        """ Test the Adarounding of a Tensor """
        modules_to_test = [torch.nn.Linear(12, 8),
                          torch.nn.Conv2d(12, 8, 3),
                          torch.nn.ConvTranspose2d(12, 8, 3)]

        nearest_encoding = libpymo.TfEncoding()
        nearest_encoding.bw = 4
        nearest_encoding.max = 10.0
        nearest_encoding.min = 0.19699306
        nearest_encoding.offset = -127.0
        nearest_encoding.delta = 0.001551126479

        for module in modules_to_test:
            with self.subTest(module):
                weight_tensor = torch.randn(module.weight.shape)
                wrapper = StaticGridQuantWrapper(module, weight_bw=4, activation_bw=4, round_mode='nearest', quant_scheme=QuantScheme.post_training_tf_enhanced)
                wrapper.param_quantizers['weight'].encoding = nearest_encoding
                ada_wrapper = AdaroundWrapper(wrapper)
                ada_quantized = ada_wrapper.apply_adaround(weight_tensor)
                self.assertFalse(torch.equal(weight_tensor, ada_quantized))
