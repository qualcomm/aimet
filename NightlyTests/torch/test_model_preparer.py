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
import torchvision
from packaging.version import Version
import pytest
import torch
from torchvision import models

from aimet_torch.model_preparer import prepare_model, _prepare_traced_model
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.v1.quantsim import QuantizationSimModel


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]

    model.eval()
    with torch.no_grad():
        model(*dummy_input)


class TestModelPreparer:

    @pytest.mark.cuda
    def test_inception_v3(self):
        """ Verify inception_v3 """
        model = models.inception_v3().eval().cuda()
        prepared_model = prepare_model(model)
        print(prepared_model)
        input_shape = (1, 3, 299, 299)
        dummy_input = torch.randn(*input_shape).cuda()

        # Verify bit-exact outputs.
        assert torch.equal(prepared_model(dummy_input), model(dummy_input))

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify with Quantization workflow.
        quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input)
        quant_sim.compute_encodings(evaluate, dummy_input)
        quant_sim.model(dummy_input)

    @pytest.mark.cuda
    def test_deeplab_v3(self):
        """ Verify deeplab_v3 """
        # Set the strict flag to False so that torch.jit.trace can be successful.
        from aimet_torch.meta import connectedgraph
        connectedgraph.jit_trace_args.update({"strict": False})
        if Version(torchvision.__version__) < Version('0.10.2'):
            model = models.segmentation.deeplabv3_resnet50(pretrained_backbone=False).eval().cuda()
        else:
            model = models.segmentation.deeplabv3_resnet50(weights_backbone=None).eval().cuda()
        prepared_model = prepare_model(model)
        print(prepared_model)
        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(*input_shape).cuda()

        # Verify bit-exact outputs.
        assert torch.equal(prepared_model(dummy_input)['out'], model(dummy_input)['out'])

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify with Quantization workflow.
        quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input)
        quant_sim.compute_encodings(evaluate, dummy_input)
        quant_sim.model(dummy_input)

    @pytest.mark.skipif(Version(torch.__version__) < Version('1.10.0'), reason="torch1.13.1 is required.")
    def test_fx_with_vit(self):
        """ Verify VIT """
        from transformers import ViTModel, ViTConfig
        from transformers.utils.fx import symbolic_trace

        # Set the strict flag to False so that torch.jit.trace can be successful.
        from aimet_torch.meta import connectedgraph
        connectedgraph.jit_trace_args.update({"strict": False})

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = ViTModel(ViTConfig()).to(device)
        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        if torch.cuda.is_available():
            model.cuda()
            dummy_input = dummy_input.cuda()

        traced_model = symbolic_trace(model, ["pixel_values"])
        _prepare_traced_model(traced_model)

        with torch.no_grad():
            outputs = model(dummy_input)
            outputs2 = traced_model(dummy_input)

        # Verify bit-exact outputs.
        assert torch.equal(dict(outputs)["last_hidden_state"], outputs2["last_hidden_state"])
        assert torch.equal(dict(outputs)["pooler_output"], outputs2["pooler_output"])

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(traced_model, dummy_input)

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass
