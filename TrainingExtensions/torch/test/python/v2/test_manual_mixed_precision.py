# /usr/bin/env python
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
import tempfile
import json
import os

import pytest
import torch
from torch import nn

from aimet_common.defs import QuantizationDataType
from aimet_torch.v2.quantization.base.quantizer import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator, SupportedDType, Precision
import aimet_torch.v1.nn.modules.custom as aimet_elementwise
from .models_.test_models import SingleResidual, ModelWithTwoInputs

class ModelWithMultiInputMultiOutput(nn.Module):
    def __init__(self):
        super(ModelWithMultiInputMultiOutput, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv1_c = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_c = nn.MaxPool2d(2)
        self.relu1_c = nn.ReLU()

        self.add_ab = aimet_elementwise.Add()
        self.add_bc = aimet_elementwise.Add()

        self.conv2_a = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2_a = nn.MaxPool2d(2)
        self.relu2_a = nn.LeakyReLU()

        self.conv2_b = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2_b = nn.MaxPool2d(2)
        self.relu2_b = nn.LeakyReLU()

        self.softmax_1 = nn.LogSoftmax(dim=1)
        self.softmax_2 = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2, x3):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        x3 = self.relu1_c(self.maxpool1_c(self.conv1_c(x3)))
        y1 = self.add_ab(x1, x2)
        y2 = self.add_bc(x2, x3)

        y1 = y1.transpose(2, 3)
        y1 = self.relu2_a(self.maxpool2_a(self.conv2_a(y1)))
        y1 = self.softmax_1(y1)

        y2 = self.relu2_b(self.maxpool2_b(self.conv2_b(y2)))
        y2 = self.softmax_2(y2)

        return y1, y2, x1, x2, x3


class ModelWithIntermediateOutput(nn.Module):
    def __init__(self):
        super(ModelWithIntermediateOutput, self).__init__()

        self.fc_1 = nn.Linear(10, 10)
        self.relu_1 = nn.ReLU()

        self.fc_2 = nn.Linear(10, 2)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        int_out = x = self.relu_1(self.fc_1(x))
        x = self.relu_2(self.fc_2(x))
        return x, int_out


class ModelWithIntermediateInput(nn.Module):
    def __init__(self):
        super(ModelWithIntermediateInput, self).__init__()

        self.fc_1 = nn.Linear(10, 10)
        self.relu_1 = nn.ReLU()

        self.add = aimet_elementwise.Add()

        self.fc_2 = nn.Linear(10, 2)
        self.relu_2 = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.relu_1(self.fc_1(x1))
        x = self.add(x1, x2)
        x = self.relu_2(self.fc_2(x))
        return x


class ModelWithExplicitDataMovementOp(nn.Module):
    def __init__(self):
        super(ModelWithExplicitDataMovementOp, self).__init__()
        self.fc_1 = nn.Linear(10, 10)
        self.relu_1 = nn.ReLU()

        self.transpose = aimet_elementwise.Permute()

        self.fc_2 = nn.Linear(10, 2)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.fc_1(x))
        x = self.transpose(x, [0, 1, 3, 2])
        x = self.relu_2(self.fc_2(x))
        return x


class TestManualMixedPrecisionConfigurator:

    def test_mp_1(self):
        """MMP Workflow """

        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        # 1. Create QuantSim object
        sim = QuantizationSimModel(model, input_tensor)

        # 2. Create the MixedPrecisionConfigurator object by passing in the QuantSim object
        mp_configurator = MixedPrecisionConfigurator(sim)

        # 3. Make set_precision/set_model_input_precision/set_model_output_precision calls
        mp_configurator.set_precision(sim.model.conv1, 'Int16', {'weight': 'Int16'})
        mp_configurator.set_precision(torch.nn.Conv2d, 'Int8', {'weight': 'Int8'})

        # 4. Call apply() method by passing in the config file and strict flag
        mp_configurator.apply()
        assert mp_configurator

        # 5. compute encodings and export


    def test_mp_2(self):
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv1, 'Int16', {'weight': 'Int16'})
        with pytest.raises(ValueError):
            mp_configurator.set_precision(sim.model.maxpool, activation='Int2')

    def test_mp_4(self):
        """
        Test over-writing old requests with new requests
        - test over-writing all Conv2d modules with Int8/Int8, after setting one to Int16/Int16
        """
        model = SingleResidual()

        torch.manual_seed(0)
        input_tensor = torch.randn((1, 3, 32, 32))
        sim = QuantizationSimModel(model, input_tensor)

        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv1, 'Int16', {'weight': 'Int16'})
        mp_configurator.set_precision(torch.nn.Conv2d, 'Int8', {'weight': 'Int8'})

        mp_requests = mp_configurator.mp_handler._process_user_requests(mp_configurator.user_requests)
        assert len(mp_requests) == 4
        for m, request in mp_requests.items():
            assert all(input_candidate ==
                       Precision(QuantizationDataType.int, 8) for input_candidate in request.input_candidates)
            assert all(output_candidate ==
                       Precision(QuantizationDataType.int, 8) for output_candidate in request.output_candidates)
            assert request.param_candidate == {'weight': Precision(QuantizationDataType.int, 8)}


    def test_mp_5(self):
        """
        Test over-writing old requests with new requests
        - test over-writing all modules with Fp16/Fp16, after setting few of them to different configurations
        """
        model = SingleResidual()

        torch.manual_seed(0)
        input_tensor = torch.randn((1, 3, 32, 32))
        sim = QuantizationSimModel(model, input_tensor)

        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv1, 'Int16', {'weight': 'Int16'})
        mp_configurator.set_precision(torch.nn.Conv2d, 'Int8', {'weight': 'Int8'})
        mp_configurator.set_precision(sim.model, 'Fp16', {'weight': 'Fp16'})

        mp_requests = mp_configurator.mp_handler._process_user_requests(mp_configurator.user_requests)
        assert len(mp_requests) == 13
        for m, request in mp_requests.items():
            assert all(input_candidate ==
                       Precision(QuantizationDataType.float, 16) for input_candidate in request.input_candidates)
            assert all(output_candidate ==
                       Precision(QuantizationDataType.float, 16) for output_candidate in request.output_candidates)
            assert request.param_candidate == {'weight': Precision(QuantizationDataType.float, 16)}

    def test_mp_6(self):
        """
        Test over-writing old requests with new requests
        - test over-riding Conv2d to Int8 after setting entire model to FP16
        """
        model = SingleResidual()

        torch.manual_seed(0)
        input_tensor = torch.randn((1, 3, 32, 32))
        sim = QuantizationSimModel(model, input_tensor)

        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model, 'Fp16', {'weight': 'Fp16'})
        mp_configurator.set_precision(torch.nn.Conv2d, 'Int8', {'weight': 'Int8'})

        mp_requests = mp_configurator.mp_handler._process_user_requests(mp_configurator.user_requests)
        assert len(mp_requests) == 13
        for m, request in mp_requests.items():
            if isinstance(m.get_original_module(), torch.nn.modules.Conv2d):
                assert all(input_candidate ==
                           Precision(QuantizationDataType.int, 8) for input_candidate in request.input_candidates)
                assert all(output_candidate ==
                           Precision(QuantizationDataType.int, 8) for output_candidate in request.output_candidates)
                assert request.param_candidate == {'weight': Precision(QuantizationDataType.int, 8)}
            else:
                assert all(input_candidate ==
                           Precision(QuantizationDataType.float, 16) for input_candidate in request.input_candidates)
                assert all(output_candidate ==
                           Precision(QuantizationDataType.float, 16) for output_candidate in request.output_candidates)
                assert request.param_candidate == {'weight': Precision(QuantizationDataType.float, 16)}

        mp_configurator.mp_handler.mp_requests = {}

    @pytest.mark.parametrize("candidate, qsim_bw", [('Int16', 8), ('Fp16', 8), ('Fp16', 16)])
    def test_mp_7(self, candidate: SupportedDType, qsim_bw: int):
        """ Basic test that user request was applied to model correctly """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor, default_output_bw=qsim_bw, default_param_bw=qsim_bw)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv1, candidate, {'weight': candidate})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.conv1.input_quantizers[0], sim.model.conv1.param_quantizers['weight']]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == qsim_bw

    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_8(self, candidate: SupportedDType):
        """
        Test that requests increasing bitwidth are applied properly
        - request should propagate upstream to affect output qtzr upstream node
        - request should not affect output qtzr at the requested node (since default bitwidth is lower than request)
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv3, candidate, {'weight': candidate})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.relu2.output_quantizers[0],
                              sim.model.conv3.param_quantizers['weight']]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    def test_mp_9(self):
        """
        Test that requests decreasing bitwidths are applied properly
        - request should propagate upstream to affect output qtzr upstream node
        - request should affect output qtzr at the requested node (since default bitwidth is higher than request)
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)
        sim = QuantizationSimModel(model, input_tensor, default_param_bw=16, default_output_bw=16)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv3, 'Int8', {'weight': 'Int8'})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.relu2.output_quantizers[0],
                              sim.model.conv3.output_quantizers[0],
                              sim.model.conv3.param_quantizers['weight']]:
                    assert module.bitwidth == 8
                else:
                    assert module.bitwidth == 16

    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_10(self, candidate: SupportedDType):
        """
        Test to make sure that requests at module inputs that have input quantizers do not propagate upwards
        """
        model = ModelWithTwoInputs()
        input_shape = (1, 1, 28, 28)

        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv2, candidate, {'weight': candidate})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.conv2.input_quantizers[0],
                              sim.model.conv2.param_quantizers['weight']]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_11(self, candidate: SupportedDType):
        """
        Test that requests are propagated to all parent modules
        """
        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.add_ab, candidate)
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.relu1_a.output_quantizers[0],
                              sim.model.relu1_b.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    def test_mp_12(self):
        """
        Test that requests are propagated to all parent nodes
        """
        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input, default_param_bw=16, default_output_bw=16)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv2_a, 'Int8', {'weight': 'Int8'})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.add_ab.output_quantizers[0],
                              sim.model.conv2_a.output_quantizers[0],
                              sim.model.conv2_a.param_quantizers['weight']]:
                    assert module.bitwidth == 8
                else:
                    assert module.bitwidth == 16

    @pytest.mark.skip("Skipping this test until a request from one child op generates a matching request at the other child op")
    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_13(self, candidate: SupportedDType):
        """
        Test that a request at a sibling op will affect the parent and the other sibling
        """
        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape))

        sim = QuantizationSimModel(model, dummy_input)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.add_ab, candidate, {'weight':candidate})

        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.relu1_a.output_quantizers[0],
                              sim.model.relu1_b.output_quantizers[0],
                              sim.model.add_ab.output_quantizers[0],
                              sim.model.add_bc.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.skip("Skipping this test until a contending child requests will raise an exception")
    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_14(self, candidate: SupportedDType):
        """
        Test that contending sibling requests will produce an error
        """
        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape))

        sim = QuantizationSimModel(model, dummy_input)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.add_ab, candidate, {'weight': candidate})
        mp_configurator.set_precision(sim.model.add_bc, candidate, {'weight': candidate})

        with pytest.raises(RuntimeError):
            mp_configurator.apply()

    @pytest.mark.skip("Skipping this test until MMP can determine model output quantizers")
    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_15(self, candidate: SupportedDType):
        """
        Test that requests at model output layers will be resolved even if they are at a higher precision than the
        rest of the model
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)
        sim = QuantizationSimModel(model, input_tensor)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.fc, candidate, {'weight': candidate})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.avgpool.output_quantizers[0],
                              sim.model.fc.output_quantizers[0],
                              sim.model.fc.param_quantizers['weight']]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_16(self, candidate: SupportedDType):
        """
        Test that request at node with multiple inputs will propagate up to parent nodes correctly, even if one of the
        inputs already has an input quantizer
        - this means that one of the input requests will have to propagate upwards but the other will not
        """
        model = ModelWithIntermediateInput()
        input_shape = (1, 1, 10, 10)

        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input)

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.add, candidate)
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.add.input_quantizers[1],
                              sim.model.relu_1.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.skip("Skipping this test until MMP can determine model output quantizers")
    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_17(self, candidate: SupportedDType):
        """
        Test that requests at model output layers will be resolved even if they are at a higher precision than the
        rest of the model
        """
        model = ModelWithIntermediateOutput()
        input_shape = (1, 1, 10, 10)

        torch.manual_seed(0)
        sim = QuantizationSimModel(model, torch.randn(*input_shape))

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.relu_1, candidate)
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.fc_1.output_quantizers[0],
                              sim.model.relu_1.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.skip("Skipping this test until MMP can apply backend awareness")
    def test_mp_18(self):
        """
        Basic backend awareness test
        - user specified activation bitwidth, but not param bitwidth. Param bitwidth will be selected automatically from
        provided config file
        """
        model = ModelWithTwoInputs()
        input_shape = (1, 1, 28, 28)

        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input)

        config = "" #TODO specify backend awareness in correct format (only allow 8x8 and 16x16 conv layers)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv2, 'Int16')
        mp_configurator.apply(config)

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.conv2.input_quantizers[0],
                              sim.model.conv2.param_quantizers['weight']]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.skip("Skipping this test until MMP can apply backend awareness, and MMP can handle siblings correctly")
    def test_mp_19(self):
        """
        Test for backend awareness. Same as test 8, except that the provided backend awareness file does not permit
        an op with two inputs at different precisions. So, the request at the sibling op will affect a larger set of
        qtzrs to realize the user request
        """
        model = ModelWithMultiInputMultiOutput()

        input_shape = (1, 1, 28, 28)
        torch.manual_seed(0)
        dummy_input = (torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape))
        sim = QuantizationSimModel(model, dummy_input)

        config = "" #TODO specify backend awareness in correct format (only allow inputs at same bitwidth in add layers)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.add_ab, 'Int16', {'weight': 'Int16'})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.relu1_a.output_quantizers[0],
                              sim.model.relu1_b.output_quantizers[0],
                              sim.model.relu1_c.output_quantizers[0],
                              sim.model.add_ab.output_quantizers[0],
                              sim.model.add_bc.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    @pytest.mark.skip("Skipping this test until MMP can handle supergroups correctly")
    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_20(self, candidate: SupportedDType):
        """
        Test that settings are applied to quantizer supergroups correctly
        """
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                }
            },
            "params": {
                "weight": {
                    "is_quantized": "True"
                }
            },
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Conv", "BatchNormalization", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            model = SingleResidual()

            torch.manual_seed(0)
            input_tensor = torch.randn((1, 3, 32, 32))
            sim = QuantizationSimModel(model, input_tensor, config_file=os.path.join(temp_dir, 'config.json'))
            mp_configurator = MixedPrecisionConfigurator(sim)

            mp_configurator.set_precision(sim.model.conv2, candidate, {'weight': candidate})
            mp_configurator.apply()

            for module in sim.model.modules():
                if isinstance(module, QuantizerBase):
                    if module in [sim.model.maxpool.output_quantizers[0],
                                  sim.model.conv2.param_quantizers['weight'],
                                  sim.model.relu2.output_quantizers[0]]:
                        assert module.bitwidth == 8
                    else:
                        assert module.bitwidth == 16


    def test_mp_21(self):
        """
        Tests that contending requests produce an error in strict mode
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv1, 'Int16', {'weight': 'Int16'})
        mp_configurator.set_precision(sim.model.relu1, 'Int4')

        with pytest.raises(RuntimeError):
            mp_configurator.apply()

    def test_mp_22(self):
        """
        Tests that contending requests do not produce an error in non-strict mode
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv3, 'Int16', {'weight': 'Int16'})
        mp_configurator.set_precision(sim.model.relu3, 'Int4')

        mp_configurator.apply(strict=False)

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.conv3.input_quantizers[0],
                              sim.model.conv3.param_quantizers['weight'],
                              sim.model.relu2.output_quantizers[0]]:
                    assert module.bitwidth == 16
                elif module in [sim.model.conv3.output_quantizers[0],
                                sim.model.relu3.output_quantizers[0],
                                sim.model.relu3.input_quantizers[0]]:
                    assert module.bitwidth == 4
                else:
                    assert module.bitwidth == 8

    def test_mp_23(self):
        """
        Tests that int quantizer can be converted successfully to a float quantizer, and that a float quantizer can be
        converted successfully to an int quantizer
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv1, 'Fp16', {'weight': 'Fp16'})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.conv1.input_quantizers[0], sim.model.conv1.param_quantizers['weight']]:
                    assert module.exponent_bits == 5
                    assert module.mantissa_bits == 10
                else:
                    assert module.bitwidth == 8

    @pytest.mark.parametrize("candidate", ['Int16', 'Fp16'])
    def test_mp_24(self, candidate: SupportedDType):
        """
        Test that upstream propagation can successfully skip explicit data movement ops
        """
        model = ModelWithExplicitDataMovementOp()
        input_shape = (1, 1, 10, 10)
        torch.manual_seed(0)
        sim = QuantizationSimModel(model, torch.randn(*input_shape))
        sim.model.transpose.output_quantizers[0] = None # doing this instead of signalling this via a qsim config file

        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.fc_2, candidate, {'weight': candidate})
        mp_configurator.apply()

        for module in sim.model.modules():
            if isinstance(module, QuantizerBase):
                if module in [sim.model.fc_2.param_quantizers['weight'],
                              sim.model.relu_1.output_quantizers[0]]:
                    assert module.bitwidth == 16
                else:
                    assert module.bitwidth == 8

    def test_mp_25(self):
        """
        Test that error is raised if invalid number of activation candidates are provided
        """
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)

        mp_configurator.set_precision(sim.model.conv3, ['Int16', 'Int16'])
        with pytest.raises(RuntimeError):
            mp_configurator.apply()