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

import json
import tempfile
import os
import onnx
import pytest
import torch
from aimet_common import libpymo
from onnx import numpy_helper

from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.weight_padding_utils import recompute_scale, recompute_encodings, weight_pad, WeightPaddingParams

from ..models.test_models import TinyModel, MultiInput


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    model.eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    with torch.no_grad():
        model(*dummy_input)


class TestWeightPadUtils:

    def test_recompute_encodings_assertion_error(self):
        bw_params = WeightPaddingParams(target_kernel_bw=4, simulated_bw=12)
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        quantizer.encoding = libpymo.TfEncoding()
        pytest.raises(AssertionError, recompute_encodings, quantizer, bw_params)

    def test_recompute_scale(self):
        scale = 2.5
        bw_params = WeightPaddingParams(target_kernel_bw=8, simulated_bw=4)
        updated_scale = recompute_scale(scale, bw_params)
        # 2.5 / 2 ** (4)
        assert updated_scale == 0.15625

    def test_recompute_scale_assertion_error(self):
        scale = 2.5
        bw_params = WeightPaddingParams(target_kernel_bw=2, simulated_bw=4)
        pytest.raises(AssertionError, recompute_scale, scale, bw_params)

    def test_weight_pad_tensor(self):
        # using B = 8 and b = 4
        a = [100.0, 23.0, -57.0, 127.0]
        input = torch.FloatTensor(a)

        # use b initially
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encoding = libpymo.TfEncoding()
        encoding.bw = 4
        encoding.min = torch.min(input)
        encoding.max = torch.max(input)
        encoding.delta = (encoding.max - encoding.min) / ((2 ** encoding.bw) - 1)
        encoding.offset = round(encoding.min / encoding.delta)
        quantizer.encoding = encoding

        # quant dequant with b
        quant_dequant = quantizer.quantize_dequantize(input, MAP_ROUND_MODE_TO_PYMO['nearest'])

        # recompute encoding with B
        updated_encoding = libpymo.TfEncoding()
        updated_encoding.bw = 8
        updated_encoding.min = torch.min(quant_dequant)
        updated_encoding.delta = quantizer.encoding.delta / 16
        updated_encoding.offset = round(updated_encoding.min / updated_encoding.delta)
        updated_encoding.max = (updated_encoding.delta * 255) + updated_encoding.min
        quantizer.encoding = updated_encoding

        # confirm weights are padded
        quant_output = quantizer.quantize(quant_dequant, MAP_ROUND_MODE_TO_PYMO['nearest'])
        for val in quant_output:
            assert val % 16 == 0

    def test_weight_pad_in_place_per_channel(self):
        model = MultiInput()
        model.eval()
        dummy_input = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))

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
                },
            },
            "op_type": {
                "Conv": {
                    "per_channel_quantization": "True"
                },
                "Gemm": {
                    "per_channel_quantization": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       default_param_bw=16, default_output_bw=16)
            sim.compute_encodings(evaluate, dummy_input)

            # confirms that per channel quantization is applied
            for name, module in sim.quant_wrappers():
                if 'weight' in module.param_quantizers:
                    assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)

            # populate bitwidths per layer in dict
            bw_dict = dict()
            for layer_name, layer in sim.quant_wrappers():
                bw_dict[layer_name] = WeightPaddingParams(simulated_bw=12, target_kernel_bw=16)

            weight_pad(sim, bw_dict)

            # confirm that all weights are properly padded in place
            for layer_name, layer in sim.quant_wrappers():
                param_quant_dict = layer.param_quantizers
                if 'weight' in param_quant_dict:
                    layer_weights = layer._module_to_wrap.weight
                    quant_tensor = param_quant_dict['weight'].quantize(layer_weights,MAP_ROUND_MODE_TO_PYMO['nearest'])
                    numpy_arr = quant_tensor.detach().numpy()
                    for val in numpy_arr.flatten():
                        assert val % 16 == 0

    def test_weight_pad_in_place_per_tensor(self):
        model = TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=16, default_output_bw=16)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(dummy_input)

        # populate bitwidths per layer in dict
        bw_dict = dict()
        for layer_name, layer in sim.quant_wrappers():
           bw_dict[layer_name] = WeightPaddingParams(simulated_bw=12, target_kernel_bw=16)

        weight_pad(sim, bw_dict)

        # confirm that all weights are properly padded in place
        for layer_name, layer in sim.quant_wrappers():
            param_quant_dict = layer.param_quantizers
            if 'weight' in param_quant_dict:
                quantizer = param_quant_dict['weight']
                if not quantizer.enabled:
                    continue
                layer_weights = layer._module_to_wrap.weight
                quant_tensor = quantizer.quantize(layer_weights,
                                                  MAP_ROUND_MODE_TO_PYMO['nearest'])
                numpy_arr = quant_tensor.detach().numpy()
                for val in numpy_arr.flatten():
                    assert val % 16 == 0

    def test_weight_pad_export_per_tensor(self):
        model = TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=16, default_output_bw=16)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(dummy_input)

        # populate bitwidths per layer in dict
        bw_dict = dict()
        for layer_name, layer in sim.quant_wrappers():
           bw_dict[layer_name] = WeightPaddingParams(simulated_bw=12, target_kernel_bw=16)

        # perform weight pad and export
        weight_pad(sim, bw_dict)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'weight_pad_model_per_tensor', dummy_input)

            quant_dict = dict()
            for layer_name, layer in sim.quant_wrappers():
                param_quant_dict = layer.param_quantizers
                if 'weight' in param_quant_dict:
                    quant_dict[layer_name + ".weight"] = param_quant_dict['weight']

            # confirm exported ONNX model has padded weights
            onnx_model = onnx.load(os.path.join(tmp_dir, 'weight_pad_model_per_tensor.onnx'))

            for node in onnx_model.graph.initializer:
                if "weight" in node.name:
                    weights = torch.Tensor(numpy_helper.to_array(node))
                    quantizer = quant_dict[node.name]

                    if not quantizer.enabled:
                        continue

                    quant_weights = quantizer.quantize(weights, round_mode=MAP_ROUND_MODE_TO_PYMO['nearest'])

                    numpy_arr = quant_weights.detach().numpy()
                    for val in numpy_arr.flatten():
                        assert val % 16 == 0

    def test_weight_pad_export_per_channel(self):
        model = MultiInput()
        model.eval()
        dummy_input = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))

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
                },
            },
            "op_type": {
                "Conv": {
                    "per_channel_quantization": "True"
                },
                "Gemm": {
                    "per_channel_quantization": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       default_param_bw=16, default_output_bw=16)
            sim.compute_encodings(evaluate, dummy_input)

            # confirms that per channel quantization is applied
            for name, module in sim.quant_wrappers():
                if 'weight' in module.param_quantizers:
                    assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)

            # populate bitwidths per layer in dict
            bw_dict = dict()
            for layer_name, layer in sim.quant_wrappers():
                bw_dict[layer_name] = WeightPaddingParams(simulated_bw=12, target_kernel_bw=16)

            # perform weight pad and export
            weight_pad(sim, bw_dict)
            sim.export(tmp_dir, 'weight_pad_model_per_channel', dummy_input)

            quant_dict = dict()
            for layer_name, layer in sim.quant_wrappers():
                param_quant_dict = layer.param_quantizers
                if 'weight' in param_quant_dict:
                    quant_dict[layer_name + ".weight"] = param_quant_dict['weight']

            # confirm exported ONNX model has padded weights
            onnx_model = onnx.load(os.path.join(tmp_dir, 'weight_pad_model_per_channel.onnx'))

            for node in onnx_model.graph.initializer:
                if "weight" in node.name:
                    weights = torch.Tensor(numpy_helper.to_array(node))
                    quantizer = quant_dict[node.name]
                    quant_weights = quantizer.quantize(weights, round_mode=MAP_ROUND_MODE_TO_PYMO['nearest'])

                    numpy_arr = quant_weights.detach().numpy()
                    for val in numpy_arr.flatten():
                        assert val % 16 == 0
