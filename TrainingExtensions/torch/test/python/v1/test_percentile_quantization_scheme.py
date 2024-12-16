# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import pytest
import torch
import torch.nn as nn
from aimet_torch.v1.quantsim import QuantizationSimModel

class TestPercentileSchemeStaticGrid:
    """ Test Percentile quantization scheme """ 

    def test_model_with_percentile_scheme(self):
        """ Test pecentile scheme by setting different percentile values """

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding="same")
                self.conv2 = torch.nn.Conv2d(16, 16, 3, padding="same")

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 3, 224, 224)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        # set same percentile value for all the activation tensors
        sim.set_percentile_value(99.99)
        sim.compute_encodings(forward_pass, None)

        # Assign the same tensor for the outputs and check if the encodings are same
        tensor = torch.rand(1, 3, 224, 224)
        sim.model.conv1.output_quantizers[0].reset_encoding_stats()
        sim.model.conv1.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv1.set_percentile_value(99.999)
        sim.model.conv1.output_quantizers[0].compute_encoding()

        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.set_percentile_value(99.999)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv1.output_quantizers[0].encoding.max == sim.model.conv2.output_quantizers[0].encoding.max
        assert sim.model.conv1.output_quantizers[0].encoding.delta == sim.model.conv2.output_quantizers[0].encoding.delta

        # Set different percentile values for each layer and verify that the encoding are not same
        sim.model.conv1.output_quantizers[0].reset_encoding_stats()
        sim.model.conv1.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv1.set_percentile_value(99)
        sim.model.conv1.output_quantizers[0].compute_encoding()

        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.set_percentile_value(99.999)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv1.output_quantizers[0].encoding.max != sim.model.conv2.output_quantizers[0].encoding.max
        assert sim.model.conv1.output_quantizers[0].encoding.delta != sim.model.conv2.output_quantizers[0].encoding.delta 