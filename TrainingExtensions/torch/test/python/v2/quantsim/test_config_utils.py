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
""" Test commonly used utilities for configuring quantsim """

import pytest
import torch
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantsim.config_utils import set_activation_quantizers_to_float, \
    set_blockwise_quantization_for_weights, set_grouped_blockwise_quantization_for_weights
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
import aimet_torch.v2.nn as aimet_nn
from aimet_torch.utils import get_device
from ...models.test_models import SingleResidualWithAvgPool, TransposedConvModel


devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

def test_set_activation_quantizers_to_float():
    model = SingleResidualWithAvgPool().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    starting_quantizer_count = 0
    for module in qsim.model.modules():
        if isinstance(module, QuantizerBase):
            starting_quantizer_count += 1

    other_layers = []
    relu_layers = []
    for _, module in qsim.named_qmodules():
        if isinstance(module, aimet_nn.QuantizedReLU):
            relu_layers.append(module)
        elif hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and \
                module.output_quantizers[0] is not None:
            other_layers.append(module)
    assert relu_layers
    assert other_layers

    for layer in relu_layers + other_layers:
        assert layer.output_quantizers[0].is_initialized()
        assert isinstance(layer.output_quantizers[0], QuantizeDequantize)

    with pytest.raises(RuntimeError):
        set_activation_quantizers_to_float(qsim, [1], dtype=torch.float16)

    set_activation_quantizers_to_float(qsim, [relu_layers[0]], dtype=torch.float16)
    assert isinstance(relu_layers[0].output_quantizers[0], FloatQuantizeDequantize)

    for relu_layer in relu_layers[1:]:
        assert isinstance(relu_layer.output_quantizers[0], QuantizeDequantize)

    set_activation_quantizers_to_float(qsim,
                                       lambda m: m == relu_layers[1],
                                       dtype=torch.float16)

    for relu_layer in relu_layers[:2]:
        assert isinstance(relu_layer.output_quantizers[0], FloatQuantizeDequantize)

    for relu_layer in relu_layers[2:]:
        assert isinstance(relu_layer.output_quantizers[0], QuantizeDequantize)

    set_activation_quantizers_to_float(qsim, [aimet_nn.QuantizedReLU], dtype=torch.float16)
    for relu_layer in relu_layers:
        assert isinstance(relu_layer.output_quantizers[0], FloatQuantizeDequantize)

    for other_layer in other_layers:
        assert isinstance(other_layer.output_quantizers[0], QuantizeDequantize)

    ending_quantizer_count = 0
    for module in qsim.model.modules():
        if isinstance(module, QuantizerBase):
            ending_quantizer_count += 1

    assert starting_quantizer_count == ending_quantizer_count

@pytest.mark.parametrize('device', devices)
def test_set_blockwise_quantization_for_weights(device):
    model = SingleResidualWithAvgPool().eval().to(device)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    conv_layers = [module for module in qsim.model.modules() if isinstance(module, aimet_nn.QuantizedConv2d)]

    # exclude the 1st conv layers since its in channels of 3 makes it inconvenient to set blockwise
    conv_layers = conv_layers[1:]

    for conv_layer in conv_layers:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    orig_device = get_device(conv_layers[0].param_quantizers['weight'])
    set_blockwise_quantization_for_weights(qsim, [conv_layers[0]], 4, True, [1, 4, -1, -1])
    assert get_device(conv_layers[0].param_quantizers['weight']) == orig_device
    assert not conv_layers[0].param_quantizers['weight'].is_initialized()
    assert conv_layers[0].param_quantizers['weight'].bitwidth == 4
    assert tuple(conv_layers[0].param_quantizers['weight'].shape) == (conv_layers[0].weight.shape[0],
                                                                      conv_layers[0].weight.shape[1] // 4,
                                                                      1,
                                                                      1)

    for conv_layer in conv_layers[1:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_blockwise_quantization_for_weights(qsim, lambda m: m == conv_layers[1], 4, True, [1, 4, -1, -1])

    for conv_layer in conv_layers[:2]:
        assert get_device(conv_layer.param_quantizers['weight']) == orig_device
        assert not conv_layer.param_quantizers['weight'].is_initialized()
        assert conv_layer.param_quantizers['weight'].bitwidth == 4
        assert tuple(conv_layer.param_quantizers['weight'].shape) == (conv_layer.weight.shape[0],
                                                                      conv_layer.weight.shape[1] // 4,
                                                                      1,
                                                                      1)

    for conv_layer in conv_layers[2:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8


    assert qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 8
    assert len(qsim.model.fc.param_quantizers['weight'].shape) == 0

    with pytest.raises(RuntimeError):
        # This should error out since the first conv's in_channels of 3 is invalid with block size 4
        set_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedConv2d], 4, True, [1, 4, -1, -1])

    set_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedLinear], 4, True, [1, 4])

    assert get_device(qsim.model.fc.param_quantizers['weight']) == orig_device
    assert not qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 4
    assert tuple(qsim.model.fc.param_quantizers['weight'].shape) == (qsim.model.fc.weight.shape[0],
                                                                     qsim.model.fc.weight.shape[1] // 4)

@pytest.mark.parametrize('device', devices)
def test_set_grouped_blockwise_quantization_for_weights(device):
    model = SingleResidualWithAvgPool().eval().to(device)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    conv_layers = [module for module in qsim.model.modules() if isinstance(module, aimet_nn.QuantizedConv2d)]

    # exclude the 1st conv layers since its in channels of 3 makes it inconvenient to set blockwise
    conv_layers = conv_layers[1:]

    for conv_layer in conv_layers:
        assert isinstance(conv_layer.param_quantizers['weight'], QuantizeDequantize)
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    orig_device = get_device(conv_layers[0])
    set_grouped_blockwise_quantization_for_weights(qsim, [conv_layers[0]], 4, True, 8, [1, 4, -1, -1], [1, -1, 1, 1])
    assert get_device(conv_layers[0].param_quantizers['weight']) == orig_device
    assert isinstance(conv_layers[0].param_quantizers['weight'], GroupedBlockQuantizeDequantize)
    assert not conv_layers[0].param_quantizers['weight'].is_initialized()
    assert conv_layers[0].param_quantizers['weight'].bitwidth == 4
    assert tuple(conv_layers[0].param_quantizers['weight'].shape) == (conv_layers[0].weight.shape[0],
                                                                      conv_layers[0].weight.shape[1] // 4,
                                                                      1,
                                                                      1)
    assert conv_layers[0].param_quantizers['weight'].decompressed_bw == 8
    assert tuple(conv_layers[0].param_quantizers['weight'].block_grouping) == (1, -1, 1, 1)

    for conv_layer in conv_layers[1:]:
        assert isinstance(conv_layer.param_quantizers['weight'], QuantizeDequantize)
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_grouped_blockwise_quantization_for_weights(qsim, lambda m: m == conv_layers[1], 4, True, 8, [1, 4, -1, -1],
                                                   [1, -1, 1, 1])

    for conv_layer in conv_layers[:2]:
        assert get_device(conv_layer.param_quantizers['weight']) == orig_device
        assert isinstance(conv_layer.param_quantizers['weight'], GroupedBlockQuantizeDequantize)
        assert not conv_layer.param_quantizers['weight'].is_initialized()
        assert conv_layer.param_quantizers['weight'].bitwidth == 4
        assert tuple(conv_layer.param_quantizers['weight'].shape) == (conv_layer.weight.shape[0],
                                                                      conv_layer.weight.shape[1] // 4,
                                                                      1,
                                                                      1)
        assert conv_layer.param_quantizers['weight'].decompressed_bw == 8
        assert tuple(conv_layer.param_quantizers['weight'].block_grouping) == (1, -1, 1, 1)

    for conv_layer in conv_layers[2:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 0
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    assert isinstance(qsim.model.fc.param_quantizers['weight'], QuantizeDequantize)
    assert qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 8
    assert len(qsim.model.fc.param_quantizers['weight'].shape) == 0

    with pytest.raises(RuntimeError):
        # This should error out since the first conv's in_channels of 3 is invalid with block size 4
        set_grouped_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedConv2d], 4, True, 8, [1, 4, -1, -1],
                                                       [1, -1, -1, -1])

    set_grouped_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedLinear], 4, True, 8, [1, 4], [1, -1])
    assert get_device(qsim.model.fc.param_quantizers['weight']) == orig_device
    assert isinstance(qsim.model.fc.param_quantizers['weight'], GroupedBlockQuantizeDequantize)
    assert not qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 4
    assert tuple(qsim.model.fc.param_quantizers['weight'].shape) == (qsim.model.fc.weight.shape[0],
                                                                     qsim.model.fc.weight.shape[1] // 4)
    assert qsim.model.fc.param_quantizers['weight'].decompressed_bw == 8
    assert tuple(qsim.model.fc.param_quantizers['weight'].block_grouping) == (1, -1)

def test_set_grouped_blockwise_with_single_value():
    model = SingleResidualWithAvgPool()
    dummy_input = torch.randn(1, 3, 32, 32)

    qsim = QuantizationSimModel(model, dummy_input)
    layers_to_set = [m for m in qsim.model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and \
                     m.weight.shape[1] != 3]
    assert layers_to_set
    set_grouped_blockwise_quantization_for_weights(qsim, lambda m: m in layers_to_set, 4, True, 8, 4, -1)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)
    _ = qsim.model(dummy_input)

    for m in layers_to_set:
        if isinstance(m, aimet_nn.QuantizedConv2d):
            assert m.param_quantizers['weight'].block_size == [1, 4, -1, -1]
            assert m.param_quantizers['weight'].block_grouping == [1, -1, 1, 1]
        else:
            assert m.param_quantizers['weight'].block_size == [1, 4]
            assert m.param_quantizers['weight'].block_grouping == [1, -1]

    model = TransposedConvModel()
    dummy_input = torch.randn(10, 10, 3, 4)
    qsim = QuantizationSimModel(model, dummy_input)
    layers_to_set = [m for m in qsim.model.modules() if isinstance(m, torch.nn.ConvTranspose2d)]
    assert layers_to_set
    set_grouped_blockwise_quantization_for_weights(qsim, lambda m: m in layers_to_set, 4, True, 8, 2, -1)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)
    _ = qsim.model(dummy_input)

    for m in layers_to_set:
        assert m.param_quantizers['weight'].block_size == [2, 1, -1, -1]
        assert m.param_quantizers['weight'].block_grouping == [-1, 1, 1, 1]

def test_invalid_single_value_block_size():
    class PreluModel(torch.nn.Module):
        def __init__(self):
            super(PreluModel, self).__init__()
            self.prelu = torch.nn.PReLU()
            self.relu = torch.nn.ReLU()

        def forward(self, inp):
            x = self.prelu(inp)
            return self.relu(x)

    model = PreluModel()
    dummy_input = torch.randn(1, 10)
    model.prelu.weight = torch.nn.Parameter(torch.randn(10))
    qsim = QuantizationSimModel(model, dummy_input)
    with pytest.raises(RuntimeError):
        set_blockwise_quantization_for_weights(qsim, lambda m: isinstance(m, torch.nn.PReLU), 4, True, 2)
