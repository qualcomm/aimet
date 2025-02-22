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

""" Unit tests for Adaround Activation Sampler """

import numpy as np

from .models.models_for_tests import simple_relu_model
from aimet_onnx.adaround.activation_sampler import ActivationSampler
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import CachedDataset

class TestAdaroundActivationSampler:
    """
     AdaRound Activation Sampler Unit Test Cases
    """
    def test_activation_sampler_conv(self, tmp_path):
        """ Test ActivationSampler for a Conv op """
        np.random.seed(0)
        model = simple_relu_model()
        sim = QuantizationSimModel(model)
        activation_sampler = ActivationSampler('input', 'output', model, sim.model, True)
        data_loader = dataloader()
        cached_dataset = CachedDataset(data_loader, 1, tmp_path)
        all_inp_data, all_out_data = activation_sampler.sample_and_place_all_acts_on_cpu(cached_dataset)

        assert np.allclose(all_out_data, all_inp_data, atol=1e-5)
        assert all_inp_data[0].shape == (1, 3, 32, 32)

def dataloader():
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """
        def __init__(self, batch_size: int):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=2)
    return dummy_dataloader
