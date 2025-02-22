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

""" AdaRound Nightly Tests """
import json
import pytest
import random
import os
import numpy as np
import torch
import torch.cuda
from torchvision import models

from aimet_common.defs import QuantScheme
from aimet_torch.utils import create_fake_data_loader, create_rand_tensors_given_shapes
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.adaround.adaround_weight import Adaround, AdaroundParameters


def dummy_forward_pass(model, inp_shape):
    """ Dummy forward pass"""
    model.eval()
    with torch.no_grad():
        model_input = torch.randn(inp_shape).to(torch.device('cuda'))
        output = model(model_input)

    return output


class TestAdaround:
    """
    AdaRound test cases
    """
    @pytest.mark.cuda
    def test_adaround_resnet18_only_weights(self, tmp_path):
        """ test end to end adaround with only weight quantized """
        torch.cuda.empty_cache()
        seed_all(1000)

        model = models.resnet18().eval()
        model = model.to(torch.device('cuda'))
        input_shape = (1, 3, 224, 224)
        dummy_input = create_rand_tensors_given_shapes(input_shape, torch.device('cuda'))

        orig_output = dummy_forward_pass(model, input_shape)

        # create fake data loader with image size (3, 224, 224)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=input_shape[1:])

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        adarounded_model = Adaround.apply_adaround(model, dummy_input, params, path=tmp_path, filename_prefix='resnet18',
                                                   default_param_bw=4,
                                                   default_quant_scheme=QuantScheme.post_training_tf_enhanced)

        ada_output = dummy_forward_pass(adarounded_model, input_shape)
        assert not torch.equal(orig_output, ada_output)

        # Test exported encodings JSON file
        with open(tmp_path / 'resnet18.encodings') as json_file:
            encoding_data = json.load(json_file)['param_encodings']

        assert isinstance(encoding_data["conv1.weight"], list)

    @pytest.mark.cuda
    def test_adaround_resnet18_followed_by_quantsim(self, tmp_path):
        """ test end to end adaround with weight 4 bits and output activations 8 bits quantized """
        torch.cuda.empty_cache()
        seed_all(1000)

        model = models.resnet18().eval()
        model = model.to(torch.device('cuda'))
        input_shape = (1, 3, 224, 224)
        dummy_input = create_rand_tensors_given_shapes(input_shape, torch.device('cuda'))

        # create fake data loader with image size (3, 224, 224)
        data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=input_shape[1:])

        params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=5,
                                    default_reg_param=0.01, default_beta_range=(20, 2))
        # W4A8
        param_bw = 4
        output_bw = 8
        quant_scheme = QuantScheme.post_training_tf_enhanced

        adarounded_model = Adaround.apply_adaround(model, dummy_input, params, path=tmp_path,
                                                   filename_prefix='resnet18', default_param_bw=param_bw,
                                                   default_quant_scheme=quant_scheme)

        # Read exported param encodings JSON file
        with open(tmp_path / "resnet18.encodings") as json_file:
            encoding_data = json.load(json_file)['param_encodings']

        encoding = encoding_data["conv1.weight"]
        before_min = np.array([e["min"] for e in encoding])
        before_max = np.array([e["max"] for e in encoding])
        before_delta = np.array([e["scale"] for e in encoding])
        before_offset = np.array([e["offset"] for e in encoding])

        # Create QuantSim using adarounded_model, set and freeze parameter encodings and then invoke compute_encodings
        sim = QuantizationSimModel(adarounded_model, quant_scheme=quant_scheme, default_param_bw=param_bw,
                                   default_output_bw=output_bw, dummy_input=dummy_input)
        sim.set_and_freeze_param_encodings(encoding_path=tmp_path / 'resnet18.encodings')
        sim.compute_encodings(dummy_forward_pass, forward_pass_callback_args=input_shape)

        encoding = sim.model.conv1.param_quantizers['weight'].encoding
        after_min = np.array([e.min for e in encoding])
        after_max = np.array([e.max for e in encoding])
        after_delta = np.array([e.delta for e in encoding])
        after_offset = np.array([e.offset for e in encoding])

        # Quantization encoding should be same as used in Adaround optimization
        assert np.all(before_min == after_min)
        assert np.all(before_max == after_max)
        assert np.all(before_delta == after_delta)
        assert np.all(before_offset == after_offset)

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass


def seed_all(seed=1029):
    """ Setup seed """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
