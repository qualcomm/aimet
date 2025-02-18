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
""" Module for checking QAT on quantsim v2 """

import os
import copy
import json
import tempfile
import pytest
import torch
from torchvision import models
from unittest.mock import patch


from aimet_common.defs import QuantScheme

from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization.affine.backends import torch_builtins
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.utils import is_leaf_module


class STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad


def autograd_based_qdq(tensor, scale, offset, qmin, qmax, block_size=None) -> torch.Tensor:
    orig_tensor_shape = tensor.shape
    tensor = torch_builtins.reshape_tensor_for_blocks(tensor, scale.shape, block_size)
    scale = scale.view(torch_builtins.get_encoding_shape_with_blocks(scale.shape, block_size))
    offset = offset.view(torch_builtins.get_encoding_shape_with_blocks(offset.shape, block_size))
    x_round = STE.apply(tensor / scale) - offset
    x_quant = torch.clamp(x_round, qmin, qmax)
    return ((x_quant + offset) * scale).view(orig_tensor_shape)


# Wrap resnet function to mimic model class instantiation
def resnet18():
    return models.resnet18(weights=None)

# Test util functions
def get_quantized_modules(model: torch.nn.Module):
    module_names = []
    for _, module in model.named_children():
        if isinstance(module, BaseQuantizationMixin):
            module_names.append(module)

        elif not is_leaf_module(module):
            module_names = module_names + get_quantized_modules(module)

    return module_names

def get_enabled_quantizers_from_quantized_module(module: BaseQuantizationMixin):
    quantizers = module.input_quantizers + module.output_quantizers \
        + list(module.param_quantizers.values())
    return [quantizer for quantizer in quantizers if quantizer is not None]

@pytest.fixture
def config_path():
    config_json = {
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
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {
            "is_input_quantized": "True"
        },
        "model_output": {}
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_path = os.path.join(temp_dir, "quantsim_config.json")
        with open(temp_config_path, 'w') as temp_config_file:
            json.dump(config_json, temp_config_file)
        yield temp_config_path


@pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                          QuantScheme.training_range_learning_with_tf_init,
                                          # QuantScheme.post_training_percentile, # TODO: not implemented
                                         ])
@pytest.mark.parametrize('model_cls, input_shape', [(resnet18, (1, 3, 224, 224))])
class TestQuantsimV2QAT:
    def test_requires_grad_correctness(self, model_cls, input_shape, quant_scheme, config_path):
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)
        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)

        for module in get_quantized_modules(sim.model):
            quantization_parameters = []
            quantizers = get_enabled_quantizers_from_quantized_module(module)
            for quantizer in quantizers:
                quantization_parameters.append(id(quantizer.min))
                quantization_parameters.append(id(quantizer.max))

            for _, param in module.named_parameters():
                # Assertions for quantization parameter
                if id(param) in quantization_parameters:
                    if quant_scheme in (QuantScheme.training_range_learning_with_tf_init,):
                        assert param.requires_grad

                    if quant_scheme in (QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_tf, \
                                        QuantScheme.post_training_percentile):
                        assert not param.requires_grad
                else:
                    # Assertion for model parameters
                    assert param.requires_grad

    # Based on https://github.com/quic/aimet/blob/75131de478af7ead6a8676dc36f4c46a6d3390ea/TrainingExtensions/torch/test/python/test_range_learning.py#L81
    def test_grad_correctness(self, model_cls, input_shape, quant_scheme, config_path):
        model = model_cls()
        dummy_input = torch.randn(input_shape)

        aimetgrad_qsim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)
        autograd_qsim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)

        aimetgrad_qsim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                                         forward_pass_callback_args=None)
        autograd_qsim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                                        forward_pass_callback_args=None)

        aimetgrad_qsim.model(dummy_input).sum().backward()
        # Patch backend of auto_grad_qsim to autograd-based backend
        with patch('aimet_torch.v2.quantization.affine.quantizer.quantize_dequantize', autograd_based_qdq):
            autograd_qsim.model(dummy_input).sum().backward()

        # Make sure that all the assertions are not being bypassed
        quantized_modules = get_quantized_modules(aimetgrad_qsim.model)
        assert len(quantized_modules) > 0

        for aimetgrad_param, autograd_param in zip(aimetgrad_qsim.model.parameters(),
                                                   autograd_qsim.model.parameters()):
            if not aimetgrad_param.requires_grad and not autograd_param.requires_grad:
                continue

            # Compare gradient between aimet-grad and autograd
            assert torch.allclose(aimetgrad_param.grad, autograd_param.grad, rtol=1e-3, atol=1e-3)

    def test_grad_accumulation(self, model_cls, input_shape, quant_scheme, config_path):
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)
        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)

        # First backward
        sim.model(dummy_input).sum().backward()
        grad_after_first_backward = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                grad_after_first_backward[param_name] = param.grad.clone()

        # Second backward without zero_grad()
        sim.model(dummy_input).sum().backward()
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(grad_after_first_backward[param_name] * 2, param.grad)

    def _test_weight_update(self, model_cls, input_shape, quant_scheme, config_path, device):
        model = model_cls().to(device)
        dummy_input = torch.randn(input_shape).to(device)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)
        sim.model.to(device)
        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)
        
        # Store initial parameter values
        initial_param_value = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                initial_param_value[param_name] = param.clone()
        
        # Update weights using gradient
        learning_rate = 0.01
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=learning_rate)
        sim.model(dummy_input).sum().backward()

        # Store gradient of parameters
        grad_after_backward = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                grad_after_backward[param_name] = param.grad.clone()
        optimizer.step()
        
        # Check weight update
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param, initial_param_value[param_name] - grad_after_backward[param_name] * learning_rate)

    def test_qat_on_cpu(self, model_cls, input_shape, quant_scheme, config_path):
        self._test_weight_update(model_cls, input_shape, quant_scheme, config_path, device=torch.device("cpu"))

    @pytest.mark.cuda
    def test_qat_on_gpu(self, model_cls, input_shape, quant_scheme, config_path):
        self._test_weight_update(model_cls, input_shape, quant_scheme, config_path, device=torch.device("cuda"))

    # Based on https://github.com/quic/aimet/blob/75131de478af7ead6a8676dc36f4c46a6d3390ea/TrainingExtensions/torch/test/python/test_quantizer.py#L1050
    @pytest.mark.cuda
    def test_multi_gpu(self, model_cls, input_shape, quant_scheme, config_path):
        device = torch.device("cuda")
        model = model_cls().to(device)
        dummy_input = torch.randn(input_shape).to(device)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme, config_file=config_path, default_param_bw=8, default_output_bw=4)
        sim.model.to(device)
        sim.compute_encodings(lambda sim_model, _: sim_model(dummy_input),
                              forward_pass_callback_args=None)

        # Get single GPU outputs and gradients
        single_gpu_output = sim.model(copy.deepcopy(dummy_input))
        single_gpu_output.sum().backward()
        single_gpu_grad = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                single_gpu_grad[param_name] = param.grad.clone()

        # Get multi-GPU outputs and gradients
        sim.model.zero_grad()
        sim.model = torch.nn.DataParallel(sim.model)
        multi_gpu_output = sim.model(copy.deepcopy(dummy_input))
        multi_gpu_output.sum().backward()

        # Compare single-gpu results and multi-gpu results
        assert torch.allclose(single_gpu_output, multi_gpu_output)
        for param_name, param in sim.model.named_parameters():
            if param_name.startswith("module."):
                param_name = ".".join(param_name.split(".")[1:])
            if param.requires_grad:
                assert torch.allclose(param.grad, single_gpu_grad[param_name], rtol=1e-3, atol=1e-3)

def test_autograd_based_qdq():
    torch.manual_seed(0)
    tensor = torch.randn(4, 8, 12)
    block_size = [2, 4, 3]
    scale = torch.randn(2, 2, 4)
    offset = torch.randint(low=-128, high=127, size=(2, 2, 4), dtype=scale.dtype)
    autograd_based_qdq_out = autograd_based_qdq(tensor, scale, offset, 0, 255, block_size)
    affine_qdq_out = affine.quantize_dequantize(tensor, scale, offset, bitwidth=8, signed=False,
                                                block_size=block_size)
    assert torch.equal(autograd_based_qdq_out, affine_qdq_out)
