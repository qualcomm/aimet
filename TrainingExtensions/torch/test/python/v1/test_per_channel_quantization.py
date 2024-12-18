# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import tempfile
from pathlib import Path

import pytest
import torch
import json
import aimet_common.libpymo as libpymo
from aimet_common.defs import MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper, LearnedGridQuantWrapper
from ..models.test_models import ModelWithTwoInputs, ModelWithTransposeConv
from aimet_torch.v1.qc_quantize_op import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.quantsim_straight_through_grad import calculate_forward_pass
from aimet_torch.v1.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer, \
    LearnedGridTensorQuantizer, ParameterQuantizer


class ModelSingleChannel(torch.nn.Module):

    def __init__(self):
        super(ModelSingleChannel, self).__init__()
        self.conv1_a = torch.nn.Conv2d(3, 1, kernel_size=2)

    def forward(self, inp):
        x = self.conv1_a(inp)
        return x


class TestPerChannelQcQuantizeOpStaticGrid:
    def test_per_channel_symmetric_qdq(self):
        """ Test tensor quantizer symmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 3.81
            encodings[index].min = -3.84
            encodings[index].delta = 0.03
            encodings[index].offset = -128

        encodings[3].bw = 8
        encodings[3].max = 6.35
        encodings[3].min = -6.4
        encodings[3].delta = 0.05
        encodings[3].offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]],
                                  dtype=torch.float32)

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-6.4, -5, -3, 0, .1, 2.5]],
                                    dtype=torch.float32)
        assert torch.allclose(quant_out, expected_out, atol=1e-5)

    def test_per_channel_asymmetric_qdq(self):
        """ Test tensor quantizer asymmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 1.9999956
            encodings[index].min = -2.9999934
            encodings[index].delta = 0.0196078
            encodings[index].offset = -153

        encodings[3].bw = 8
        encodings[3].max = 2.404693
        encodings[3].min = -5.995262
        encodings[3].delta = 0.032941
        encodings[3].offset = -182

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]])

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-5.9953, -5.0070, -2.9976, 0, .09888, 2.4047]],
                                    dtype=torch.float32)
        assert torch.allclose(quant_out, expected_out, atol=0.0001)

    def test_per_channel_symmetric_compute_encodings(self):
        """ Test tensor quantizer symmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert quantizer.encoding[0].max == 7
        assert round(quantizer.encoding[0].min, 2) == -7.06

        assert quantizer.encoding[3].max == 5
        assert round(quantizer.encoding[3].min, 2) == -5.04

    def test_per_channel_asymmetric_compute_encodings(self):
        """ Test tensor quantizer asymmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf,
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert round(quantizer.encoding[0].max, 3) == 2.496
        assert round(quantizer.encoding[0].min, 3) == -7.004

        assert round(quantizer.encoding[3].max, 3) == 2.004
        assert round(quantizer.encoding[3].min, 3) == -4.996

    # -------------------------------------------
    def test_model_with_two_inputs_per_channel(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for _, wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        assert isinstance(sim.model.fc2.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.fc2.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.fc2.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        assert len(sim.model.conv1_a.param_quantizers['weight'].encoding) == 10
        assert len(sim.model.fc2.param_quantizers['weight'].encoding) == 10

        model(*dummy_input)

        # Check that different encodings are computed for different channels
        assert sim.model.conv1_a.param_quantizers['weight'].encoding[0] != \
               sim.model.conv1_a.param_quantizers['weight'].encoding[1]
        assert sim.model.fc2.param_quantizers['weight'].encoding[0] != \
               sim.model.fc2.param_quantizers['weight'].encoding[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'two_input_model_per_channel', dummy_input)

            with open(os.path.join(tmp_dir, "two_input_model_per_channel.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
            assert len(encodings['param_encodings']) == 5
            assert len(encodings['param_encodings']['conv1_a.weight']) == 10
            assert encodings['param_encodings']['conv1_a.weight'][1]['bitwidth'] == 8
            assert encodings['param_encodings']['conv1_a.weight'][1]['is_symmetric'] == 'True'

    def test_model_per_channel_single_channel(self):
        """Model with single channel conv """
        dummy_input = (torch.rand(1, 3, 28, 28),)

        def forward_pass(model, _):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelSingleChannel()
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.model.conv1_a.enable_per_channel_quantization()

        # Quantize
        sim.compute_encodings(forward_pass, None)
        assert sim.model.conv1_a.param_quantizers['weight'].encoding[0] is not None

    def test_set_and_freeze_param_encoding_per_channel(self):
        """ Test set and freeze parameter encoding for per-channel encodings """
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quant_module = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                              quant_scheme=QuantScheme.post_training_tf_enhanced,
                                              data_type=QuantizationDataType.int)
        quant_module.enable_per_channel_quantization()

        param_encodings = {'conv1.weight': [{'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'},
                                            {'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'}
                                            ]}

        quant_module.set_param_encoding('conv1', param_encodings)
        quant_module.freeze_param_encoding('conv1', param_encodings)

        assert len(quant_module.param_quantizers['weight'].encoding) == 4
        assert quant_module.param_quantizers['weight'].encoding[0].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[0].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[0].delta == 0.038
        assert quant_module.param_quantizers['weight'].encoding[3].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[3].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[3].delta == 0.038

        assert not quant_module.param_quantizers['weight'].use_symmetric_encodings
        assert quant_module.param_quantizers['weight'].bitwidth == 4

        # Reset encoding, Since encoding are frozen they should not be None after reset encoding
        quant_module.reset_encodings()

        assert len(quant_module.param_quantizers['weight'].encoding) == 4
        assert quant_module.param_quantizers['weight'].encoding[0].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[0].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[0].delta == 0.038
        assert quant_module.param_quantizers['weight'].encoding[3].bw == 4
        assert quant_module.param_quantizers['weight'].encoding[3].offset == -7.0
        assert quant_module.param_quantizers['weight'].encoding[3].delta == 0.038

    # -------------------------------------------
    def test_model_with_two_inputs_per_channel_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for _, wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    # -------------------------------------------

    def test_model_with_two_inputs_per_channel_fp16_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input, default_output_bw=16, default_param_bw=16,
                                   default_data_type=QuantizationDataType.float)

        for _, wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    def test_model_transposed_conv_per_channel_qat(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTransposeConv()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        for _, wrapper in sim.quant_wrappers():
            wrapper.enable_per_channel_quantization()

        assert isinstance(sim.model.conv1_a.param_quantizers['weight'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.param_quantizers['bias'], StaticGridPerChannelQuantizer)
        assert isinstance(sim.model.conv1_a.output_quantizers[0], StaticGridPerTensorQuantizer)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Pass some data in train mode
        sim.model.train()
        output = sim.model(*dummy_input)

        # Try a backward pass - all we are testing for is that nothing blows up functionally
        loss = output.flatten().sum()
        loss.backward()

    def test_transposed_conv_layer_per_channel(self):
        """Model with more than 1 input"""

        num_output_channels = 4
        layer = torch.nn.ConvTranspose2d(10, num_output_channels, kernel_size=5)
        # Fill all weight values with 1
        layer.weight.data.fill_(1.0)
        encodings = [libpymo.TfEncoding() for _ in range(num_output_channels)]
        for i in range(num_output_channels):
            layer.weight.data[:, i, :, :] *= (i+1)
            encodings[i].bw = 8
            encodings[i].max = i + 0.5
            encodings[i].min = 0
            encodings[i].delta = 0.0196078
            encodings[i].offset = -153

        quantization_wrapper = StaticGridQuantWrapper(layer, weight_bw=8, activation_bw=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                      data_type=QuantizationDataType.int)
        quantization_wrapper.enable_per_channel_quantization()

        weight_quantizer = quantization_wrapper.param_quantizers['weight']
        bias_quantizer = quantization_wrapper.param_quantizers['bias']

        assert isinstance(weight_quantizer, StaticGridPerChannelQuantizer)
        assert isinstance(bias_quantizer, StaticGridPerChannelQuantizer)
        assert len(weight_quantizer._cppOp) == num_output_channels

        weight_quantizer.update_encoding_stats(quantization_wrapper._module_to_wrap.weight)
        # Assign golden vector to encodings
        weight_quantizer.encoding = encodings
        round_mode = libpymo.RoundingMode.ROUND_NEAREST
        # Quantize Dequantize
        output = weight_quantizer.quantize_dequantize(quantization_wrapper._module_to_wrap.weight, round_mode)
        expected_output = layer.weight.data
        for i in range(num_output_channels):
            expected_output[:, i, :, :] -= 0.5

        assert torch.equal(output, expected_output)


class TestPerChannelQcQuantizeOpLearnedGrid:
    @pytest.mark.cuda
    def test_tensor_quantizer(self):
        torch.manual_seed(0)
        wrapper = create_learned_grid_wrapper()
        tensor_quantizer = wrapper.param_quantizers['weight']

        encodings = []
        for _ in range(3):
            encoding = libpymo.TfEncoding()
            encoding.bw, encoding.max, encoding.min, encoding.delta, encoding.offset = 8, 3, -2, 1, 0.2
            encodings.append(encoding)

        # Set encodings for tensor quantizer
        tensor_quantizer.encoding = encodings

        # Check getting of encodings
        new_encodings = tensor_quantizer.encoding
        # Get 1st tf encodings object to compare values
        enc_0 = new_encodings[0]
        for enc1, enc2 in zip(encodings, new_encodings):
            assert enc1.min == enc2.min
            assert enc1.max == enc2.max
            # Check that all delta offset got computed correctly
            assert enc2.delta == enc_0.delta
            assert enc2.offset == enc_0.offset
        print(tensor_quantizer)

    def test_quantize_dequantize_tensor_quantizer(self):
        torch.manual_seed(0)
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=False,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)
        tensor_quantizer._ch_axis = 0

        encoding_min = torch.nn.Parameter(torch.FloatTensor([0.0, 0.0, 0.0]))
        encoding_max = torch.nn.Parameter(torch.FloatTensor([1.0, 2.5, 3.5]))
        tensor = torch.ones((3, 1, 1, 2))
        tensor[0, :, :, :] *= 1.5
        tensor[2, :, :, :] *= 4
        quant_dequantized_output = tensor_quantizer.quantize_dequantize(tensor, encoding_min, encoding_max)
        expected_output = torch.ones((3, 1, 1, 2))
        expected_output[2, :, :, :] *= 3.5
        assert torch.all(expected_output.eq(quant_dequantized_output))
        assert encoding_min.grad == None
        assert encoding_max.grad == None

        optimizer = torch.optim.SGD([encoding_min, encoding_max], lr=0.05, momentum=0.5)

        loss = quant_dequantized_output.sum()
        loss.backward()
        optimizer.step()
        assert encoding_min.grad is not None
        assert encoding_max.grad is not None
        assert len(encoding_max) == len(encoding_max.grad)


    def test_replacement_of_wrapper(self):
        torch.manual_seed(0)
        conv1 = torch.nn.Conv2d(3, 3, kernel_size=5)
        post_training_module = StaticGridQuantWrapper(conv1, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=False,
                                                      is_output_quantized=False, activation_bw=8, weight_bw=8)

        post_training_module.enable_per_channel_quantization()

        encodings = []
        for _ in range(3):
            encoding = libpymo.TfEncoding()
            encoding.bw, encoding.max, encoding.min, encoding.delta, encoding.offset = 8, 3, -2, 1, 0.2
            encodings.append(encoding)

        post_training_module.input_quantizers[0].enabled = True
        post_training_module.input_quantizers[0].encoding = encodings[0]
        post_training_module.param_quantizers['weight'].enabled = True
        post_training_module.param_quantizers['weight'].encoding = encodings
        post_training_module.param_quantizers['bias'].enabled = False
        dummy_input = torch.randn(1, 3, 12, 12)
        sim = QuantizationSimModel(conv1, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)
        # sim.model.conv1.input_quantizer.enabled = True
        trainable_module = sim._construct_and_initialize_trainable_wrapper(post_training_module, device='cpu')

        assert trainable_module.output_quantizers[0].use_symmetric_encodings == False
        assert trainable_module.output_quantizers[0].enabled == False
        assert trainable_module.input0_encoding_min.item() == -2.0
        assert isinstance(trainable_module.param_quantizers['weight'].encoding, list)

    @pytest.mark.cuda
    def test_apply_gating_logic(self):
        torch.manual_seed(0)
        wrapper = create_learned_grid_wrapper()
        encodings = []
        for _ in range(32):
            encoding = libpymo.TfEncoding()
            encoding.bw, encoding.max, encoding.min, encoding.delta, encoding.offset = 8, -3, 2, 1, 0.2
            encodings.append(encoding)

        wrapper.output_quantizers[0].encoding = encodings[0]
        wrapper.input_quantizers[0].enabled = False
        wrapper.param_quantizers['weight'].encoding = encodings
        wrapper.param_quantizers['bias'].enabled = False
        wrapper.apply_gating_logic()
        assert wrapper.output_quantizers[0].encoding.min == 0.0
        assert abs(wrapper.output_quantizers[0].encoding.max - 1e-5) < 1e-6
        for enc in wrapper.param_quantizers['weight'].encoding:
            assert enc.min == 0.0
            assert abs(enc.max - 1e-5) < 1e-6

    @pytest.mark.cuda
    @pytest.mark.parametrize('dtype', (torch.float32, torch.float16, torch.bfloat16))
    def test_compute_gradients_Parameter_Quantizer(self, dtype):
        torch.manual_seed(0)
        wrapper = create_learned_grid_wrapper()
        encodings = []
        for _ in range(3):
            encoding = libpymo.TfEncoding()
            encoding.bw, encoding.max, encoding.min, encoding.delta, encoding.offset = 8, -3, 2, 1, 0.2
            encodings.append(encoding)

        wrapper.param_quantizers['weight'].encoding = encodings
        param_quantizer = wrapper.param_quantizers['weight']

        encoding_min = torch.nn.Parameter(torch.FloatTensor([0.0, 0.0, 0.0]).to('cuda', dtype), requires_grad=True)
        encoding_max = torch.nn.Parameter(torch.FloatTensor([1.0, 2.5, 3.5]).to('cuda', dtype), requires_grad=True)

        param_quantizer.scaling, param_quantizer.offset = param_quantizer.compute_scaling_offset(encoding_min, encoding_max)

        tensor = torch.ones((3, 1, 1, 2)).to('cuda', dtype)
        grad = torch.randn(3, 1, 1, 2).to('cuda', dtype)
        _, intermediate_result = calculate_forward_pass(tensor, param_quantizer, encoding_min, encoding_max)
        enc_min_grad, enc_max_grad = ParameterQuantizer.compute_gradients(tensor, grad, intermediate_result,
                                                                          param_quantizer.channel_axis)

        assert len(enc_min_grad) == len(enc_max_grad) == 3
        assert torch.all(torch.eq(enc_max_grad, -enc_min_grad))

    @pytest.mark.cuda
    @pytest.mark.parametrize('dtype', (torch.float32, torch.float16, torch.bfloat16))
    def test_compute_gradients_Parameter_Quantizer_bias(self, dtype):
        torch.manual_seed(0)
        wrapper = create_learned_grid_wrapper()

        param_quantizer = wrapper.param_quantizers['bias']

        encoding_min = torch.nn.Parameter(torch.FloatTensor([0.0, 0.0, 0.0]).to('cuda', dtype), requires_grad=True)
        encoding_max = torch.nn.Parameter(torch.FloatTensor([1.0, 2.5, 3.5]).to('cuda', dtype), requires_grad=True)

        param_quantizer.scaling, param_quantizer.offset = param_quantizer.compute_scaling_offset(encoding_min, encoding_max)

        tensor = torch.ones(3).to('cuda', dtype)
        grad = torch.randn(3).to('cuda', dtype)
        _, intermediate_result = calculate_forward_pass(tensor, param_quantizer, encoding_min, encoding_max)
        enc_min_grad, enc_max_grad = ParameterQuantizer.compute_gradients(tensor, grad, intermediate_result, param_quantizer.channel_axis)

        assert len(enc_min_grad) == len(enc_max_grad) == 3
        assert torch.all(torch.eq(enc_max_grad, -enc_min_grad))

    @pytest.mark.cuda
    def test_qc_trainable_wrapper(self):
        torch.manual_seed(0)
        trainable_module = create_learned_grid_wrapper()

        encodings = []
        for _ in range(3):
            encoding = libpymo.TfEncoding()
            encoding.bw, encoding.max, encoding.min, encoding.delta, encoding.offset = 8, 3, -2, 1, 0.2
            encodings.append(encoding)
        trainable_module.input_quantizers[0].enabled = False
        # trainable_module.input_quantizer.encoding = encodings[0]

        trainable_module.param_quantizers['weight'].enabled = True
        trainable_module.param_quantizers['weight'].encoding = encodings
        trainable_module.param_quantizers['bias'].enabled = True
        trainable_module.param_quantizers['bias'].encoding = encodings

        trainable_module.output_quantizers[0].enabled = True
        trainable_module.output_quantizers[0].encoding = encodings[0]

        inp = torch.rand((1, 2, 5, 5), requires_grad=True).to('cuda')
        out = trainable_module(inp)
        optimizer = torch.optim.SGD(trainable_module.parameters(), lr=0.05, momentum=0.5)
        loss = out.flatten().sum()
        loss.backward()
        optimizer.step()

        # Checking if encoding min max have changed
        assert not trainable_module.output0_encoding_min.item() == -2.0
        assert not trainable_module.output0_encoding_max.item() == 3.0

        for val in trainable_module.weight_encoding_min:
            assert not val.item() == -2.0
        for val in trainable_module.weight_encoding_max:
            assert not val.item() == 3.0

        for val in trainable_module.bias_encoding_min:
            assert not val.item() == -2.0
        for val in trainable_module.bias_encoding_max:
            assert not val.item() == 3.0

    def test_export_model_with_two_inputs(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = save_config_file_for_per_channel_quantization(Path(temp_dir))
            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                       quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       config_file=config_file)
            # Quantize
            sim.compute_encodings(forward_pass, None)
            assert len(sim.model.conv1_a.param_quantizers['weight'].encoding) == 10
            assert len(sim.model.fc2.param_quantizers['weight'].encoding) == 10

            model(*dummy_input)

            # Check that different encodings are computed for different channels
            assert sim.model.conv1_a.param_quantizers['weight'].encoding[0] != \
                   sim.model.conv1_a.param_quantizers['weight'].encoding[1]
            assert sim.model.fc2.param_quantizers['weight'].encoding[0] != \
                   sim.model.fc2.param_quantizers['weight'].encoding[1]

            sim.export(temp_dir, 'two_input_model_per_channel', dummy_input)

            with open(os.path.join(temp_dir, "two_input_model_per_channel.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
            assert len(encodings['param_encodings']) == 5
            assert len(encodings['param_encodings']['conv1_a.weight']) == 10
            assert encodings['param_encodings']['conv1_a.weight'][1]['bitwidth'] == 8

    def test_export_model_with_two_inputs_fp16(self):
        """Model with more than 1 input, fp16 mode"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_config_file_for_per_channel_quantization(temp_dir)
            sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                       default_param_bw=16, default_output_bw=16,
                                       default_data_type=QuantizationDataType.float,
                                       config_file=os.path.join(temp_dir, 'quantsim_config.json'))

            # Quantize
            sim.compute_encodings(forward_pass, None)
            model(*dummy_input)

            sim.export(temp_dir, 'results', dummy_input)

            with open(os.path.join(temp_dir, "results.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
            assert len(encodings['param_encodings']) == 5
            assert len(encodings['activation_encodings']) == 15

            for key in encodings['param_encodings'].keys():
                assert len(encodings['param_encodings'][key]) == 1
                assert len(encodings['param_encodings'][key][0]) == 2
                assert encodings['param_encodings'][key][0]['bitwidth'] == 16
                assert encodings['param_encodings'][key][0]['dtype'] == 'float'

            for key in encodings['activation_encodings'].keys():
                assert len(encodings['activation_encodings'][key]) == 1
                assert len(encodings['activation_encodings'][key][0]) == 2
                assert encodings['activation_encodings'][key][0]['bitwidth'] == 16
                assert encodings['activation_encodings'][key][0]['dtype'] == 'float'

    def test_model_with_two_inputs_in_manual_mixed_precision_mode(self):
        """
        Test manual mixed precision for a model with more than 1 input
        - set per-channel configuration
        - change the bitwidth and dtype of one of the layers to fp16
        - verify it is set correctly after compute_encodings and export of encodings
        """

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = save_config_file_for_per_channel_quantization(Path(temp_dir))
            sim = QuantizationSimModel(model, dummy_input=dummy_input, config_file=config_file)

            sim.model.conv1_a.param_quantizers['weight'].bitwidth = 16
            sim.model.conv1_a.param_quantizers['weight'].data_type = QuantizationDataType.float

            # Quantize
            sim.compute_encodings(forward_pass, None)
            model(*dummy_input)
            sim.export(temp_dir, 'results', dummy_input)

            with open(os.path.join(temp_dir, "results.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
            assert len(encodings['param_encodings']) == 5
            assert len(encodings['param_encodings']['conv1_a.weight']) == 1

            #verify the modified conv1_a params are set correctly
            assert len(encodings['param_encodings']['conv1_a.weight'][0]) == 2
            assert encodings['param_encodings']['conv1_a.weight'][0]['bitwidth'] == 16
            assert encodings['param_encodings']['conv1_a.weight'][0]['dtype'] == 'float'


def create_learned_grid_wrapper():
    conv1 = torch.nn.Conv2d(2, 3, kernel_size=5).to('cuda')

    wrapper = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                      is_symmetric=False, is_output_quantized=True, activation_bw=8,
                                      weight_bw=8, device='cuda', data_type=QuantizationDataType.int)
    return wrapper


def save_config_file_for_per_channel_quantization(path: Path) -> Path:
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {
            "bias": {
                "is_quantized": "False"
            }
        },
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    target_file = Path(path, 'quantsim_config.json')
    with open(target_file, 'w') as f:
        json.dump(quantsim_config, f)
    return target_file
