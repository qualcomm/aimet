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

import pytest
import torch

from aimet_common.utils import AimetLogger
import aimet_torch.v1.quantsim as v1
import aimet_torch.v2.quantsim as v2
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from .models.test_models import TinyModel
from aimet_torch.utils import create_fake_data_loader
from aimet_torch._base.adaround.activation_sampler import ActivationSampler
from aimet_torch.v2.nn.base import BaseQuantizationMixin

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


@pytest.fixture
def model():
    return TinyModel().eval()


@pytest.fixture
def sim(request, model):
    QuantizationSimModel = request.param
    assert QuantizationSimModel in (v1.QuantizationSimModel, v2.QuantizationSimModel)
    sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 32, 32), quant_scheme='tf_enhanced',
                               default_param_bw=4)

    if QuantizationSimModel == v1.QuantizationSimModel:
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                for quantizer in module.input_quantizers + module.output_quantizers:
                    quantizer.enabled = False
                    quantizer.enabled = False
    else:
        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                module._remove_activation_quantizers()

    return sim


class TestAdaroundActivationSampler:
    """
    Adaround unit tests
    """
    @pytest.mark.parametrize('sim', [v1.QuantizationSimModel, v2.QuantizationSimModel], indirect=True)
    def test_activation_sampler_conv(self, sim, model):
        """ Test ActivationSampler for a Conv module """
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

        assert list(quant_inp.shape) == [batch_size * possible_batches, 3, 32, 32]
        assert list(orig_out.shape) == [batch_size * possible_batches, 32, 18, 18]

    @pytest.mark.parametrize('sim', [v1.QuantizationSimModel, v2.QuantizationSimModel], indirect=True)
    def test_activation_sampler_fully_connected_module(self, sim, model):
        """ Test ActivationSampler for a fully connected module """
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

        assert list(quant_inp.shape) == [batch_size * possible_batches, 36]
        assert list(orig_out.shape) == [batch_size * possible_batches, 12]
