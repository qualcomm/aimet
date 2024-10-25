# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import pytest
import json
import random
import os
import re
import copy
import numpy as np
import onnx
import torch
pytest.importorskip("torch", minversion="1.8") # Skip tests in this file if minimum torch version is not met
import torch.fx
torch.fx.wrap('len')
torch.fx.wrap('sqrt')

from torchvision import models
from math import sqrt
from torch.utils.data import DataLoader

from aimet_common.defs import QuantScheme
import aimet_torch.v1.nn.modules.custom as aimet_modules
from models.test_models import (
    ModelWithFunctionalReLU,
    SingleResidual,
    ModelWithDuplicateReLU,
    ConcatModel,
    CustomFunctionalConv,
)
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.v1.quantsim import QuantizationSimModel, QuantParams
from aimet_torch.utils import create_fake_data_loader, get_device, in_eval_mode
from aimet_torch.model_preparer import prepare_model, _find_functional_name_for_node
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import  equalize_model
from aimet_torch.v1.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch import bias_correction
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.model_preparer import prepare_pt_transformer_for_quantsim
from aimet_torch import onnx_utils


def train(model: torch.nn.Module, data_loader: DataLoader) -> torch.Tensor:
    """
    Helper function to train model given data loader
    :param model: torch model
    :param data_loader: torch data loader
    :return: total loss
    """
    total_loss = 0
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for (data, labels) in data_loader:
        optimizer.zero_grad()
        predicted = model(data)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss


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


@torch.fx.wrap
def custom_function_not_to_be_traced(x, y):
    """ Function which we do not want to be traced, when traced using torch FX API, call to this function will
    be inserted as call_function, and won't be traced through """
    for i in range(2):
        x += x
        y += y
    return x * x + y * y


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


class TestFX:

    def test_fx_with_relu(self):
        """
        test torch fx with functional ReLUs
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithFunctionalReLU().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert isinstance(model_transformed.module_relu, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_1, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_2, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_3, torch.nn.ReLU)

        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_add(self):
        """
        test torch fx with elementwise Add
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = SingleResidual().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert isinstance(model_transformed.module_add, aimet_modules.Add)
        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_duplicate_relu(self):
        """
        test torch fx with Duplicate/Reused module ReLU
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithDuplicateReLU().eval()
        model_transformed = prepare_model(model)

        assert isinstance(model_transformed.relu, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_1, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_2, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu_3, torch.nn.ReLU)

        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_resnet18(self):
        """
        test torch fx with torchvision Resnet18
        """
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(*input_shape)
        model = models.resnet18().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

        original_model_relu_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                original_model_relu_count += 1

        modified_model_relu_count = 0
        for module in model_transformed.modules():
            if isinstance(module, torch.nn.ReLU):
                modified_model_relu_count += 1

        # 8 duplicate ReLUs are replaced by new ReLUs in modified model
        assert original_model_relu_count + 8 == modified_model_relu_count

        assert model_transformed.layer1._modules['0'].module_relu_1.inplace == True
        assert model_transformed.layer1._modules['1'].module_relu_1.inplace == True
        assert model_transformed.layer2._modules['0'].module_relu_1.inplace == True
        assert model_transformed.layer2._modules['1'].module_relu_1.inplace == True
        assert model_transformed.layer3._modules['0'].module_relu_1.inplace == True
        assert model_transformed.layer3._modules['1'].module_relu_1.inplace == True
        assert model_transformed.layer4._modules['0'].module_relu_1.inplace == True
        assert model_transformed.layer4._modules['1'].module_relu_1.inplace == True

    @pytest.mark.cuda
    def test_fx_with_resnet18_with_cuda(self):
        """
        test torch fx with torchvision Resnet18 CUDA
        """
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(*input_shape).cuda()
        model = models.resnet18().cuda().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

        original_model_relu_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                original_model_relu_count += 1

        modified_model_relu_count = 0
        for module in model_transformed.modules():
            if isinstance(module, torch.nn.ReLU):
                modified_model_relu_count += 1

        # 8 duplicate ReLUs are replaced by new ReLUs in modified model
        assert original_model_relu_count + 8 == modified_model_relu_count
        assert get_device(model) == get_device(model_transformed)
        assert model.training == model_transformed.training

    def test_fx_with_functional_relu_quantsim(self):
        """
        test torch fx with QuantSim
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithFunctionalReLU().eval()
        model_transformed = prepare_model(model)

        quant_sim_for_modified_model = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        print(quant_sim_for_modified_model)

        # Conv --> ReLU supergroup is detected correctly
        assert quant_sim_for_modified_model.model.conv1.output_quantizers[0].enabled == False
        assert quant_sim_for_modified_model.model.module_relu.output_quantizers[0].enabled == True

        assert quant_sim_for_modified_model.model.conv2.output_quantizers[0].enabled == False
        assert quant_sim_for_modified_model.model.module_relu_1.output_quantizers[0].enabled == True

        assert quant_sim_for_modified_model.model.fc1.output_quantizers[0].enabled == False
        assert quant_sim_for_modified_model.model.module_relu_2.output_quantizers[0].enabled == True

        assert quant_sim_for_modified_model.model.fc2.output_quantizers[0].enabled == False
        assert quant_sim_for_modified_model.model.module_relu_3.output_quantizers[0].enabled == True

    def test_fx_with_functional_relu_quantsim_eval(self):
        """
        test torch fx with QuantSim evaluation
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithFunctionalReLU().eval()
        model_transformed = prepare_model(model)

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
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            quant_sim_for_original_model = QuantizationSimModel(model, dummy_input=input_tensor,
                                                                config_file=os.path.join(tmp_dir, 'quantsim_config.json'))

            quant_sim_for_modified_model = QuantizationSimModel(model_transformed, dummy_input=input_tensor,
                                                                config_file=os.path.join(tmp_dir, 'quantsim_config.json'))

            # Disable output activation quantizer for ReLUs to compare with original quantsim.model eval
            quant_sim_for_modified_model.model.module_relu.output_quantizers[0].enabled = False
            quant_sim_for_modified_model.model.module_relu_1.output_quantizers[0].enabled = False
            quant_sim_for_modified_model.model.module_relu_2.output_quantizers[0].enabled = False
            quant_sim_for_modified_model.model.module_relu_3.output_quantizers[0].enabled = False

            quant_sim_for_original_model.compute_encodings(evaluate, input_tensor)
            quant_sim_for_modified_model.compute_encodings(evaluate, input_tensor)

            # Eval for both models
            assert torch.allclose(quant_sim_for_original_model.model(input_tensor),
                                  quant_sim_for_modified_model.model(input_tensor))

            # Compare encodings for last layer for both models
            assert quant_sim_for_original_model.model.fc2.output_quantizers[0].encoding.min ==\
                   quant_sim_for_modified_model.model.fc2.output_quantizers[0].encoding.min

    def test_fx_with_elementwise_add_quantsim(self):
        """
        test torch fx with elementwise Add
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = SingleResidual().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert isinstance(model_transformed.module_add, aimet_modules.Add)
        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

        quant_sim_for_modified_model = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        print(quant_sim_for_modified_model)

        # Add + ReLU Supergroup
        # Add's output quantizer should be disabled, and ReLU's output quantizer should be enabled
        assert quant_sim_for_modified_model.model.module_add.output_quantizers[0].enabled == False
        assert quant_sim_for_modified_model.model.relu3.output_quantizers[0].enabled == True

    def test_fx_with_functional_relu_training(self):
        """
        test torch fx with training
        """
        model = ModelWithFunctionalReLU().eval()
        model_copy = copy.deepcopy(model)

        data_loader = create_fake_data_loader(32, 16, (3, 32, 32))
        before_training_weight = model.conv1.weight.clone()
        total_loss_for_original_model = train(model, data_loader)
        after_training_weight = model.conv1.weight

        # Weights should not be same before and after training
        assert not np.allclose(before_training_weight.detach().cpu().numpy(),
                               after_training_weight.detach().cpu().numpy())

        # Train modified model
        model_transformed = prepare_model(model_copy)
        total_loss_for_modified_model = train(model_transformed, data_loader)

        # Compare loss after one iteration of training
        assert total_loss_for_original_model.item() == total_loss_for_modified_model.item()

    def test_fx_with_functional_relu_qat(self):
        """
        test torch fx with QAT
        """
        model = ModelWithFunctionalReLU().eval()
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)

        model_transformed = prepare_model(model)

        # Compute encodings for both original and modified models
        quant_sim_for_original_model = QuantizationSimModel(model, dummy_input=input_tensor)
        quant_sim_for_modified_model = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        quant_sim_for_original_model.compute_encodings(evaluate, input_tensor)
        quant_sim_for_modified_model.compute_encodings(evaluate, input_tensor)

        before_training_weight = quant_sim_for_modified_model.model.conv1._module_to_wrap.weight.clone()
        print(before_training_weight[0, 0, :, :].detach().numpy())

        # QAT
        data_loader = create_fake_data_loader(32, 16, (3, 32, 32))
        total_loss_for_original_model = train(quant_sim_for_original_model.model, data_loader)
        total_loss_for_modified_model = train(quant_sim_for_modified_model.model, data_loader)

        after_training_weight = quant_sim_for_modified_model.model.conv1._module_to_wrap.weight
        print(after_training_weight[0, 0, :, :].detach().numpy())

        # Compare loss after one iteration of training
        # Since we are additionally adding noise for ReLUs now, loss will not be bit exact same
        assert torch.allclose(total_loss_for_original_model, total_loss_for_modified_model, atol=1e-2)

        # Weights should not be same before and after training
        assert not np.allclose(before_training_weight.detach().cpu().numpy(),
                               after_training_weight.detach().cpu().numpy())

    def test_fx_with_batch_norm_folding(self):
        """
        test torch fx with torchvision Resnet18 - BN fold
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = SingleResidual().eval()
        model_copy = copy.deepcopy(model)
        folded_pairs_for_original_model = fold_all_batch_norms(model, input_shape)

        # Apply BN fold for transformed model
        model_transformed = prepare_model(model_copy)
        folded_pairs_for_transformed_model = fold_all_batch_norms(model_transformed, input_shape)
        print(model_transformed)

        # folded pairs should be same for both original and transformed model
        assert len(folded_pairs_for_original_model) == len(folded_pairs_for_transformed_model)

        # forward pass for BN folded original and modified model
        # output should be close for both BN folded original and modified model
        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

        # compare weights for very first layer
        # Weights should be same
        original_model_conv1_weight = model.conv1.weight.clone()
        modified_model_conv1_weight = model_transformed.conv1.weight.clone()
        assert np.array_equal(original_model_conv1_weight.detach().cpu().numpy(),
                              modified_model_conv1_weight.detach().cpu().numpy())

        # compare weights for very last layer
        # Weights should be same
        original_model_fc_weight = model.fc.weight.clone()
        modified_model_fc_weight = model_transformed.fc.weight.clone()
        assert np.array_equal(original_model_fc_weight.detach().cpu().numpy(),
                              modified_model_fc_weight.detach().cpu().numpy())

    @pytest.mark.cuda
    def test_fx_with_cle(self):
        """
        test torch fx with torchvision Resnet18 - Cross layer equalization
        """
        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape).cuda()
        model = SingleResidual().cuda().eval()
        model_copy = copy.deepcopy(model)

        # Perform CLE - (BN fold, ReLU6 -> ReLU replacement, CLS, HBF)
        equalize_model(model, input_shape)

        # Apply CLE for transformed model
        model_transformed = prepare_model(model_copy)
        equalize_model(model_transformed, input_shape)

        # forward pass for equalized original and modified model
        # output should be close for both equalized original and modified model
        assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

        # compare weights for very first layer
        # Weights should be same
        original_model_conv1_weight = model.conv1.weight.clone()
        modified_model_conv1_weight = model_transformed.conv1.weight.clone()
        assert np.array_equal(original_model_conv1_weight.detach().cpu().numpy(),
                              modified_model_conv1_weight.detach().cpu().numpy())

        # compare weights for very last layer
        # Weights should be same
        original_model_fc_weight = model.fc.weight.clone()
        modified_model_fc_weight = model_transformed.fc.weight.clone()
        assert np.array_equal(original_model_fc_weight.detach().cpu().numpy(),
                              modified_model_fc_weight.detach().cpu().numpy())

    @pytest.mark.cuda
    def test_fx_with_adaround(self):
        """
        test torch fx with torchvision Resnet18 - adaround
        """
        seed_all(1)
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape).cuda()
        model = SingleResidual().cuda().eval()
        model_copy = copy.deepcopy(model)

        # create fake data loader with image size (3, 224, 224)
        data_loader = create_fake_data_loader(dataset_size=16, batch_size=16, image_size=input_shape[1:])
        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5)
        adarounded_original_model = Adaround.apply_adaround(model, dummy_input, params, path='./',
                                                            filename_prefix='resnet')
        # Apply Adaround for transformed model
        model_transformed = prepare_model(model_copy)
        adarounded_transformed_model = Adaround.apply_adaround(model_transformed, dummy_input, params, path='./',
                                                               filename_prefix='resnet')
        # compare weights for very first layer
        # Weights should be same
        original_model_conv1_weight = adarounded_original_model.conv1.weight.clone()
        modified_model_conv1_weight = adarounded_transformed_model.conv1.weight.clone()
        assert np.allclose(original_model_conv1_weight.detach().cpu().numpy(),
                           modified_model_conv1_weight.detach().cpu().numpy())

        # compare weights for very last layer
        # Weights should be same
        original_model_fc_weight = adarounded_original_model.fc.weight.clone()
        modified_model_fc_weight = adarounded_transformed_model.fc.weight.clone()
        assert np.allclose(original_model_fc_weight.detach().cpu().numpy(),
                           modified_model_fc_weight.detach().cpu().numpy())

    @pytest.mark.cuda
    def test_fx_with_bias_correction(self):
        """
        test torch fx with torchvision Resnet18 - bias correction
        """
        seed_all(1)
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape).cuda()
        model = SingleResidual().cuda().eval()
        model_copy = copy.deepcopy(model)

        # create fake data loader with image size (3, 224, 224)
        data_loader = create_fake_data_loader(dataset_size=16, batch_size=16, image_size=input_shape[1:])
        params = QuantParams(weight_bw=4, act_bw=4, round_mode="nearest",
                             quant_scheme=QuantScheme.post_training_tf)
        bias_correction.correct_bias(model, params, num_quant_samples=1, data_loader=data_loader,
                                     num_bias_correct_samples=1, perform_only_empirical_bias_corr=True)

        # Apply Bias correction for transformed model
        model_transformed = prepare_model(model_copy)
        bias_correction.correct_bias(model_transformed, params, num_quant_samples=1, data_loader=data_loader,
                                     num_bias_correct_samples=1, perform_only_empirical_bias_corr=True)

        # forward pass for bias corrected original and modified model
        # output should be close for both bias corrected original and modified model
        assert torch.allclose(model(dummy_input), model_transformed(dummy_input))

        # compare bias for very first layer
        # Bias should be same
        original_model_conv1_weight = model.conv1.bias.clone()
        modified_model_conv1_weight = model_transformed.conv1.bias.clone()
        assert np.array_equal(original_model_conv1_weight.detach().cpu().numpy(),
                              modified_model_conv1_weight.detach().cpu().numpy())

        # compare bias for very last layer
        # Bias should be same
        original_model_fc_weight = model.fc.bias.clone()
        modified_model_fc_weight = model_transformed.fc.bias.clone()
        assert np.array_equal(original_model_fc_weight.detach().cpu().numpy(),
                              modified_model_fc_weight.detach().cpu().numpy())

    def test_fx_with_save_and_load_entire_model(self):
        """
        test torch fx with torchvision Resnet18 - torch save and load the entire model
        """
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(*input_shape)
        model = models.resnet18().eval()
        model_transformed = prepare_model(model)

        torch.save(model_transformed, './modified_resnet18.pth')
        saved_model = torch.load('./modified_resnet18.pth')
        saved_model.eval()
        print(saved_model)

        # Eval for both models
        assert torch.allclose(model_transformed(input_tensor),
                              saved_model(input_tensor))

        if os.path.exists('./modified_resnet18.pth'):
            os.remove('./modified_resnet18.pth')

    def test_fx_with_save_and_load_state_dict(self):
        """
        test torch fx with torchvision Resnet18 - torch save and load only the state_dict
        """
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(*input_shape)
        model = models.resnet18().eval()
        model_transformed = prepare_model(model)

        # Save only the state_dict of transformed model
        torch.save(model_transformed.state_dict(), './modified_resnet18.pth')

        # 1) Load the state_dict in same transformed model
        model_transformed.load_state_dict(torch.load('./modified_resnet18.pth'))
        model_transformed.eval()

        # Eval for both models
        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        # 2) Load the dict in original model
        model.load_state_dict(torch.load('./modified_resnet18.pth'))
        model.eval()

        # Eval for both models
        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        if os.path.exists('./modified_resnet18.pth'):
            os.remove('./modified_resnet18.pth')

    def test_fx_with_quantsim_export(self):
        """
        test torch fx with torchvision Resnet18 - QuantSim export
        """
        input_shape = (1, 3, 224, 224)
        input_tensor = torch.randn(*input_shape)
        model = models.resnet18().eval()
        model_transformed = prepare_model(model)
        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))

        quant_sim.compute_encodings(forward_pass, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quant_sim.export(tmp_dir, filename_prefix='modified_resnet18', dummy_input=input_tensor)

            with open(os.path.join(tmp_dir, 'modified_resnet18.encodings')) as json_file:
                encoding_data = json.load(json_file)
                print(encoding_data)

            # Check the exported model
            loaded_model = torch.load(os.path.join(tmp_dir, 'modified_resnet18.pth'))

            # Load weights from exported model to load qdq weight
            model_transformed.load_state_dict(loaded_model.state_dict())

            # Eval for both models
            assert torch.allclose(model_transformed(input_tensor),
                                  loaded_model(input_tensor))

            # Check the onnx model
            onnx_model = onnx.load(os.path.join(tmp_dir, 'modified_resnet18.onnx'))
            node_for_relu = 0
            node_for_add = 0
            onnx.checker.check_model(onnx_model)
            module_names = { module_name for module_name, _ in model_transformed.named_modules()}
            for node in onnx_model.graph.node:
                if node.op_type == 'Relu':
                    node_for_relu += 1
                elif node.op_type == 'Add':
                    node_for_add += 1

                if not node.name.startswith('Flatten'):
                    name = node.name.split('#')[0]
                    assert '.'.join(name.split('.')[:-1]) in module_names

            # 8 new ReLUs are added.
            assert node_for_relu == 8 + 9

            # 8 new aimet_modules.Add are added.
            assert node_for_add == 8

    def test_fx_with_relu_relu6(self):
        """
        test torch fx with functional ReLU6 and ReLU
        """
        class ModelWithReLUReLU6(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

            def __init__(self):
                super(ModelWithReLUReLU6, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = torch.nn.functional.relu6(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.relu6(x)
                x = torch.nn.functional.relu(x)
                return x

        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithReLUReLU6().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_relu6, torch.nn.ReLU6)
        assert isinstance(model_transformed.module_relu, torch.nn.ReLU)
        assert isinstance(model_transformed.module_relu6_1, torch.nn.ReLU6)
        assert isinstance(model_transformed.module_relu_1, torch.nn.ReLU)

    def test_fx_with_elu(self):
        """
        test torch fx with functional ELU and alpha=0.2
        """
        class ModelWithNonLinearActivations(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

            def __init__(self):
                super(ModelWithNonLinearActivations, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = torch.nn.functional.elu(x, alpha=0.2)
                return x

        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithNonLinearActivations().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_elu, torch.nn.ELU)

    def test_fx_with_hardshrink(self):
        """
        test torch fx with functional hardshrink and lambda = 0.2
        """
        class ModelWithNonLinearActivations(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

            def __init__(self):
                super(ModelWithNonLinearActivations, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = torch.nn.functional.hardshrink(x, lambd=0.2)
                return x

        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithNonLinearActivations().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_hardshrink, torch.nn.Hardshrink)
        assert model_transformed.module_hardshrink.lambd == 0.2

    def test_fx_with_duplicate_elu(self):
        """
        test torch fx with functional reused/duplicate ELU
        """
        class ModelWithNonLinearActivations(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

            def __init__(self):
                super(ModelWithNonLinearActivations, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
                self.elu = torch.nn.ELU(alpha=0.2, inplace=True)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.elu(x)
                x = torch.nn.functional.relu(x, inplace=True)
                x = self.elu(x)
                return x

        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithNonLinearActivations().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_relu, torch.nn.ReLU)
        assert model_transformed.module_relu.inplace == True

        assert isinstance(model_transformed.module_elu_1, torch.nn.ELU)
        assert model_transformed.module_elu_1.alpha == 0.2
        assert model_transformed.module_elu_1.inplace == True
        assert model_transformed.module_elu_1.training == False

    def test_fx_with_functional_conv(self):
        class ModelWithFunctionalConv(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.parameter.Parameter(torch.randn(10, 10, 1, 1))
                self.b1 = torch.nn.parameter.Parameter(torch.randn(10))
                self.w2 = torch.nn.parameter.Parameter(torch.randn(5, 10, 1, 1))
                self.b2 = torch.nn.parameter.Parameter(torch.randn(5))
                self.conv = torch.nn.Conv2d(10, 10, 1)
                self.stride = self.conv.stride
                self.stride_2 = 2
                self.padding_2 = 2

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.conv2d(x, self.w1, self.b1, self.stride)
                x = torch.nn.functional.conv2d(x, self.w1, stride=self.stride)
                x = torch.nn.functional.conv2d(x, self.w1, None)
                x = torch.nn.functional.conv2d(x, self.w1, bias=None)
                x = torch.nn.functional.conv2d(weight=self.w2, bias=self.b2, input=x, stride=self.stride_2,
                                               padding=self.padding_2)
                return x

        input_shape = (1, 10, 10, 1)
        input_tensor = torch.randn(input_shape)
        model = ModelWithFunctionalConv().eval()
        model_transformed = prepare_model(model)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_conv2d, torch.nn.Conv2d)
        assert isinstance(model_transformed.module_conv2d_1, torch.nn.Conv2d)
        assert isinstance(model_transformed.module_conv2d_2, torch.nn.Conv2d)
        assert isinstance(model_transformed.module_conv2d_3, torch.nn.Conv2d)
        assert isinstance(model_transformed.module_conv2d_4, torch.nn.Conv2d)

        assert model_transformed.module_conv2d_4.padding == (model.padding_2, model.padding_2)
        assert model_transformed.module_conv2d_4.stride == (model.stride_2, model.stride_2)

        sim = QuantizationSimModel(model_transformed, input_tensor)

        def dummy_forward(model, args):
            model.eval()
            model(input_tensor)
        sim.compute_encodings(dummy_forward, None)
        sim.model(input_tensor)

    def test_fx_with_depthwise_conv(self):
        class ModelWithFunctionalDepthwiseConv(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """
            def __init__(self):
                super().__init__()
                self.w = torch.nn.parameter.Parameter(torch.randn(10, 1, 1, 1))
                self.groups = 10

            def forward(self, x):
                x = torch.nn.functional.conv2d(x, self.w, bias=None, groups=self.groups)
                return x

        input_shape = (1, 10, 10, 1)
        input_tensor = torch.randn(input_shape)
        model = ModelWithFunctionalDepthwiseConv().eval()
        model_transformed = prepare_model(model)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_conv2d, torch.nn.Conv2d)
        assert model_transformed.module_conv2d.groups == model.groups
        assert model_transformed.module_conv2d.in_channels == model.groups * model.w.shape[1]

        sim = QuantizationSimModel(model_transformed, input_tensor)

        def dummy_forward(model, args):
            model.eval()
            model(input_tensor)
        sim.compute_encodings(dummy_forward, None)
        sim.model(input_tensor)

    def test_fx_with_conv2d_weight_as_activation(self):
        class ModelWithFunctionalConv(torch.nn.Module):
            """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(10, 10, 1)

            def forward(self, x):
                weight = self.conv(x)
                x = torch.nn.functional.conv2d(x, weight)
                return x

        input_shape = (1, 10, 10, 1)
        input_tensor = torch.randn(input_shape)
        model = ModelWithFunctionalConv().eval()
        model_transformed = prepare_model(model)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

        assert isinstance(model_transformed.module_conv2d, aimet_modules.DynamicConv2d)

        sim = QuantizationSimModel(model_transformed, input_tensor)

        def dummy_forward(model, args):
            model.eval()
            model(input_tensor)
        sim.compute_encodings(dummy_forward, None)
        sim.model(input_tensor)

    def test_fx_with_elementwise_cat(self):
        """
        test torch fx with elementwise op - torch.cat
        """
        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ConcatModel().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_cat, aimet_modules.Concat)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_cat.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_subtract(self):
        """
        test torch fx with elementwise op - torch.subtract
        """
        class ModelWithSubtractOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithSubtractOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.subtract(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithSubtractOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_subtract, aimet_modules.Subtract)

    def test_fx_with_elementwise_mul(self):
        """
        test torch fx with elementwise op - torch.mul
        """
        class ModelWithMultiplyOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMultiplyOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.mul(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithMultiplyOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_mul, aimet_modules.Multiply)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_mul.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_div(self):
        """
        test torch fx with elementwise op - torch.div
        """
        class ModelWithDivideOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithDivideOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.div(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithDivideOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_div, aimet_modules.Divide)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_div.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_matmul(self):
        """
        test torch fx with elementwise op - torch.matmul
        """
        class ModelWithMatMulOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMatMulOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.matmul(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = (torch.randn(*input_shape), torch.randn(*input_shape))
        model = ModelWithMatMulOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_matmul, aimet_modules.MatMul)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_matmul.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_cat_default_dim(self):
        """
        test torch fx with elementwise op - torch.cat with default dim
        """
        class ModelWithCatOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithCatOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.cat((x1, x2))
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithCatOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_cat, aimet_modules.Concat)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_cat.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_cat_input_as_list_and_dim_as_kwargs(self):
        """
        test torch fx with elementwise op - torch.cat with input as list and dim as kwargs
        """
        class ModelWithCatOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithCatOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = torch.cat([x1, x2], dim=1)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithCatOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_cat, aimet_modules.Concat)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_cat.output_quantizers[0].enabled == True

    def verify_mean_op(self, model):
        """
        Utiliy function for verification of mean op
        """
        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape)]
        model = model().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_mean, aimet_modules.Mean)

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        assert quant_sim.model.module_mean.output_quantizers[0].enabled == True

    def test_fx_with_elementwise_mean_dim_as_args_keepdim_as_kwargs(self):
        """
        test torch fx with elementwise op - torch.mean with dim as args and keepdim as kwargs
        """
        class ModelWithMeanOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = x1.mean(1, keepdim=True)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_elementwise_mean_dim_as_kwargs_keepdim_as_kwargs(self):
        """
        test torch fx with elementwise op - torch.mean with dim as kwrgs and keepdim as kwargs
        """
        class ModelWithMeanOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = x1.mean(dim=1, keepdim=True)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_elementwise_mean_dim_as_args(self):
        """
        test torch fx with elementwise op - torch.mean with dim as args
        """
        class ModelWithMeanOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = x1.mean(1)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_elementwise_mean_dim_as_args_keepdim_as_args(self):
        """
        test torch fx with elementwise op - torch.mean with dim as args and keepdim as args
        """
        class ModelWithMeanOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = x1.mean(0, False)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_elementwise_mean(self):
        """
        test torch fx with elementwise op - torch.mean
        """
        class ModelWithMeanOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = x1.mean()
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_mean_op(self):
        """
        test torch fx with elementwise op - torch.mean
        """
        class ModelWithMeanOp(torch.nn.Module):

            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = torch.mean(x1)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_mean_dim_as_args_keepdim_as_args(self):
        """
        test torch fx with elementwise op - torch.mean with dim args and keepdim as kwargs
        """
        class ModelWithMeanOp(torch.nn.Module):

            def __init__(self):
                super(ModelWithMeanOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x = torch.mean(x1, (2,3), keepdim=True)
                return x

        self.verify_mean_op(ModelWithMeanOp)

    def test_fx_with_elementwise_scalar_add(self):
        """
        test torch fx with elementwise op - Scalar torch.add
        """
        class ModelWithScalarAddOp(torch.nn.Module):
            def __init__(self):
                super(ModelWithScalarAddOp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                _ = self.conv2(inputs[1])
                x = torch.add(x1, 5)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithScalarAddOp().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_add, aimet_modules.Add)

    def test_fx_with_interpolate(self):
        """
        test torch fx with interpolate functional
        """
        class ModelWithInterpolate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, x):
                x = self.conv(x)
                return torch.nn.functional.interpolate(x, scale_factor=2)

        input_shape = (1, 3, 32, 32)
        input_tensor = torch.randn(input_shape)
        model = ModelWithInterpolate().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor), model(input_tensor))

        assert isinstance(model_transformed.module_interpolate, aimet_modules.Interpolate)

    def test_fx_with_duplicate_conv(self):
        """
        test torch fx with reused/duplicate - torch.nn.Conv2d
        """
        class ModelWithDuplicateConv(torch.nn.Module):
            def __init__(self):
                super(ModelWithDuplicateConv, self).__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv(inputs[0])
                x2 = self.conv(inputs[1])
                x = torch.add(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithDuplicateConv().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

        assert isinstance(model_transformed.module_conv_1, torch.nn.Conv2d)
        assert model_transformed.module_conv_1.bias is None

        # Compare the weights
        conv_weight = model_transformed.conv.weight.clone()
        new_conv_weight = model_transformed.module_conv_1.weight.clone()
        assert np.array_equal(conv_weight.detach().cpu().numpy(),
                              new_conv_weight.detach().cpu().numpy())

        quant_sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        print(quant_sim)

    def test_fx_with_non_torch_function(self):
        """
        test torch fx with non torch function - len()
        Use torch.fx.wrap() API at the module-level scope
        """
        class ModelWithNonTorchFunction(torch.nn.Module):
            def __init__(self):
                super(ModelWithNonTorchFunction, self).__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv(inputs[0])
                return x / sqrt(len(x))

        input_shape = (1, 3, 8, 8)
        input_tensor = torch.randn(*input_shape)
        model = ModelWithNonTorchFunction().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(input_tensor),
                              model(input_tensor))

    def test_fx_with_custom_function(self):
        """
        test torch fx with custom function not to be traced -
        Use torch.fx.wrap() API at the module-level scope
        """
        class ModelWithCustomFunction(torch.nn.Module):
            def __init__(self):
                super(ModelWithCustomFunction, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = custom_function_not_to_be_traced(x1, x2)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithCustomFunction().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model(*input_tensor))

    def test_fx_with_static_control_flow(self):
        """
        test torch fx with model static control flow
        """
        class ModelWithBranch(torch.nn.Module):
            def __init__(self, branch):
                super(ModelWithBranch, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.relu = torch.nn.ReLU(inplace=True)
                self.branch = branch

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                if self.branch:
                    x = self.relu(x)
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model_with_branch_false = ModelWithBranch(branch=False).eval()
        model_transformed = prepare_model(model_with_branch_false)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model_with_branch_false(*input_tensor))

        model_with_branch_true = ModelWithBranch(branch=True).eval()
        model_transformed = prepare_model(model_with_branch_true)
        print(model_transformed)

        assert torch.allclose(model_transformed(*input_tensor),
                              model_with_branch_true(*input_tensor))

    def test_prepare_model_with_pytorch_transformer_layer(self):
        """
        Test that validates auto replacement of functional activation functions in
        PyTorch nn.Transformer layer with modules.
        :return:
        """

        src = torch.rand(10, 32, 512)
        dummy_input = torch.rand(10, 32, 512)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input, dummy_input)

        num_encoder_layers = 12

        # start with a vanilla PyTorch transformer layer
        transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=num_encoder_layers)
        transformer_model.eval()

        ops_with_missing_modules = connectedgraph_utils.get_ops_with_missing_modules(transformer_model, (src, src))
        ops_with_missing_modules = [op.name for op in ops_with_missing_modules]

        # first validate there are relu to be replaced
        r = re.compile("relu_*")
        find_relus = list(filter(r.match, ops_with_missing_modules))
        assert (find_relus)

        # auto replace functional activation with module for nn.Transformer layers
        prepare_pt_transformer_for_quantsim(transformer_model)
        ops_with_missing_modules = connectedgraph_utils.get_ops_with_missing_modules(transformer_model, (src, src))
        ops_with_missing_modules = [op.name for op in ops_with_missing_modules]

        # validate there are no activations with missing modules
        # check there are no Add
        r = re.compile("relu_*")
        find_relus = list(filter(r.match, ops_with_missing_modules))
        assert (not find_relus)

        # sanity check, in case there are any default gelu
        r = re.compile("gelu_*")
        find_gelus = list(filter(r.match, ops_with_missing_modules))
        assert (not find_gelus)

    def test_fx_with_interpolate_dynamic_inferred(self):
        """ test torch fx with interpolate functional with size dynamically inferred """
        class ModelWithInterpolate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.interpolate(x, size=(x.size(2),  x.size(3)), mode='bilinear', align_corners=True)
                x = torch.nn.functional.interpolate(x, align_corners=False, size=(x.size(2), x.size(3)), mode='bicubic')
                x = torch.nn.functional.interpolate(x, (x.size(2), x.size(3)), None, 'nearest', None, None)
                x = torch.nn.functional.interpolate(x, (x.size(2), x.size(3)), mode='bilinear')
                x = torch.nn.functional.interpolate(x, scale_factor=2)
                return x

        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(input_shape)
        model = ModelWithInterpolate().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        # Verify bit exact outputs.
        assert torch.equal(model_transformed(dummy_input), model(dummy_input))
        assert isinstance(model_transformed.module_interpolate, aimet_modules.Interpolate)
        assert isinstance(model_transformed.module_interpolate_1, aimet_modules.Interpolate)
        assert isinstance(model_transformed.module_interpolate_2, aimet_modules.Interpolate)
        assert isinstance(model_transformed.module_interpolate_3, aimet_modules.Interpolate)
        assert isinstance(model_transformed.module_interpolate_4, aimet_modules.Interpolate)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(model_transformed, dummy_input=dummy_input)
        sim.compute_encodings(evaluate, forward_pass_callback_args=dummy_input)
        sim.model(dummy_input)

        # Verify that activations encodings are correctly exported.
        with tempfile.TemporaryDirectory() as tempdir:
            sim.export(tempdir, filename_prefix='modified_model', dummy_input=dummy_input,
                       onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)))
            with open(os.path.join(tempdir, 'modified_model.encodings')) as json_file:
                encoding_data = json.load(json_file)

            # Total 7 encodings for activations.
            assert len(encoding_data["activation_encodings"]) == 7

    def test_fx_with_quantsim_export_and_encodings(self):
        """ test quantsim export and verify encodings are exported correctly for newly added modules """

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x2 = self.conv2(inputs[1])
                x = x1 + x2
                x = x + 1
                return x

        input_shape = (1, 3, 8, 8)
        input_tensor = (torch.randn(*input_shape), torch.randn(*input_shape))
        model = Model().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        assert torch.equal(model_transformed(*input_tensor),
                           model(*input_tensor))

        # Verify Quantization workflow.
        sim = QuantizationSimModel(model_transformed, dummy_input=input_tensor)
        sim.compute_encodings(evaluate, forward_pass_callback_args=input_tensor)

        with tempfile.TemporaryDirectory() as tempdir:
            sim.export(tempdir, filename_prefix='modified_model', dummy_input=input_tensor)
            with open(os.path.join(tempdir, 'modified_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
            # Total 6 encodings for activations. two inputs, two outputs of Convs and two outputs of Add modules.
            assert len(encoding_data["activation_encodings"]) == 6

    def test_model_with_exclusion(self):
        """ test model with exclusion list """
        class CustomModule(torch.nn.Module):
            @staticmethod
            def forward(x):
                return x * torch.nn.functional.softplus(x).sigmoid()

        class CustomModel(torch.nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(8)
                self.relu1 = torch.nn.ReLU(inplace=True)
                self.custom = CustomModule()

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.custom(x)
                return x

        # Create prepared_model without exclusion list.
        model = CustomModel().eval()
        prepared = prepare_model(model)
        assert hasattr(prepared, "module_softplus") and isinstance(getattr(prepared, "module_softplus"), torch.nn.Softplus)
        assert hasattr(prepared, "module_sigmoid") and isinstance(getattr(prepared, "module_sigmoid"), torch.nn.Sigmoid)
        assert hasattr(prepared, "module_mul") and isinstance(getattr(prepared, "module_mul"), aimet_modules.Multiply)

        # Creat prepared model with exclusion list.
        model = CustomModel().eval()
        prepared = prepare_model(model, modules_to_exclude=[model.custom])
        assert not hasattr(prepared, "module_softplus")
        assert not hasattr(prepared, "module_sigmoid")
        assert not hasattr(prepared, "module_mul")
        assert hasattr(prepared, "custom")

    def test_fx_with_max_pool2d_indices(self):
        """ test torch fx with max_pool2d """
        class ModelWithMaxPool2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = torch.nn.functional.max_pool2d(x, return_indices=False, kernel_size=2)
                x, indices = torch.nn.functional.max_pool2d(x, 2, return_indices=True)
                x, indices = torch.nn.functional.max_pool2d(x, return_indices=True, kernel_size=2)
                return x, indices

        input_shape = (1, 3, 64, 64)
        dummy_input = torch.randn(input_shape)
        model = ModelWithMaxPool2d().eval()
        model_transformed = prepare_model(model)
        print(model_transformed)

        # Compare output and indices.
        assert torch.equal(model_transformed(dummy_input)[0], model(dummy_input)[0])
        assert torch.equal(model_transformed(dummy_input)[1], model(dummy_input)[1])

        # Verify that the modules are added correctly
        assert isinstance(model_transformed.module_max_pool2d, aimet_modules.MaxPool2d)
        assert isinstance(model_transformed.module_max_pool2d_1, aimet_modules.MaxPool2d)
        assert isinstance(model_transformed.module_max_pool2d_with_indices, aimet_modules.MaxPool2d)
        assert isinstance(model_transformed.module_max_pool2d_with_indices_1, aimet_modules.MaxPool2d)

        # Verify Quantization workflow.
        sim = QuantizationSimModel(model_transformed, dummy_input=dummy_input)
        sim.compute_encodings(evaluate, forward_pass_callback_args=dummy_input)

        # Quantizer enabled for output and disabled for indices (integer values)
        assert sim.model.module_max_pool2d_with_indices.output_quantizers[0].enabled
        assert not sim.model.module_max_pool2d_with_indices.output_quantizers[1].enabled

    def test_find_functional_name_for_node(self):
        assert _find_functional_name_for_node("add_123") == "add"
        assert _find_functional_name_for_node("max_pool2d_with_indices_123") == "max_pool2d_with_indices"
        assert _find_functional_name_for_node("cat_123_1") == "cat"
        assert _find_functional_name_for_node("relu6_123") == "relu6"
        assert _find_functional_name_for_node("123_relu6_123") is None # Not a valid name.

    def test_fx_with_functional_batchnorm(self):
        """ test torch fx with function batchnorm """
        class ModelWithFunctionalBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)
                self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=2, stride=2, padding=2)
                self.rm = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=False)
                self.rv = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=False)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.nn.functional.batch_norm(x, running_mean=self.rm, running_var=self.rv)
                x = self.conv2(x)
                x = torch.nn.functional.batch_norm(x, running_mean=self.rm, running_var=self.rv, momentum=0.2, eps=1e-4)
                return x

        input_shape = (1, 3, 64, 64)
        dummy_input = torch.randn(input_shape)
        model = ModelWithFunctionalBN().eval()
        model_transformed = prepare_model(model)
        model(dummy_input)

        # Compare output.
        assert torch.equal(model_transformed(dummy_input), model(dummy_input))

        # Verify that the modules are added correctly
        assert isinstance(model_transformed.module_batch_norm, aimet_modules.BatchNorm)
        assert isinstance(model_transformed.module_batch_norm_1, aimet_modules.BatchNorm)

        # Verify Quantization workflow.
        sim = QuantizationSimModel(model_transformed, dummy_input=dummy_input)
        sim.compute_encodings(evaluate, forward_pass_callback_args=dummy_input)

        # Quantizer enabled for output
        assert sim.model.module_batch_norm.output_quantizers[0].enabled
        assert sim.model.module_batch_norm_1.output_quantizers[0].enabled

        # Apply Batchnorm folding
        fold_all_batch_norms(model_transformed, input_shape)

        # Compare output after bnf
        assert torch.equal(model_transformed(dummy_input)[0], model(dummy_input)[0])

    def test_replace_silu_with_custom_silu(self):
        """ test to verify replacement of silu with custom silu (sigmoid + mul) """
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, (3, 3))
                self.silu = torch.nn.SiLU()

            def forward(self, inputs):
                x = self.conv(inputs)
                x = self.silu(x)
                x = torch.nn.functional.silu(x, inplace=True)
                x = self.silu(x)
                return x

        dummy_input = torch.randn(1, 3, 8, 8)
        model = Model().eval()

        prepared_model = prepare_model(model)

        # Replace SiLU by CustomSiLU module
        assert isinstance(prepared_model.silu, aimet_modules.CustomSiLU)
        assert isinstance(prepared_model.module_silu_1, aimet_modules.CustomSiLU)
        assert isinstance(prepared_model.module_silu_2, aimet_modules.CustomSiLU)

        # Verify the outputs (won't be bit-exact).
        assert torch.allclose(model(dummy_input), prepared_model(dummy_input))

        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)

        # Check encodings are computed correctly for both (sigmoid + mul) for all three CustomSiLUs
        assert sim.model.silu.sigmoid.output_quantizers[0].encoding
        assert sim.model.silu.mul.output_quantizers[0].encoding
        assert sim.model.module_silu_1.sigmoid.output_quantizers[0].encoding
        assert sim.model.module_silu_1.mul.output_quantizers[0].encoding
        assert sim.model.module_silu_2.sigmoid.output_quantizers[0].encoding
        assert sim.model.module_silu_2.mul.output_quantizers[0].encoding

        # Verify Quantsim Export workflow.
        with tempfile.TemporaryDirectory() as tempdir:
            sim.export(tempdir, filename_prefix='modified_model', dummy_input=dummy_input)
            with open(os.path.join(tempdir,  'modified_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
            # Total 8 encodings for activations. 1 input, 1 output of Conv and 6 (2 * 3) for CustomSiLU modules.
            assert len(encoding_data["activation_encodings"]) == 8

    def test_fx_with_baddbmm(self):
        """ test torch fx with baddbmm """
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 10, (1, 1))

            def forward(self, *inputs):
                x = self.conv(inputs[0]).squeeze(0)
                x = x.baddbmm(inputs[1], inputs[2], beta=0.2, alpha=0.4)
                x = torch.baddbmm(x, inputs[1], inputs[2], beta=0.5, alpha=0.1)
                return x

        input_shape = (1, 3, 5, 5)
        dummy_input = (torch.randn(*input_shape), torch.randn(10, 5, 4), torch.randn(10, 4, 5))
        model = Model().eval()
        prepared_model = prepare_model(model)

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify bit-exact outputs.
        assert torch.equal(model(*dummy_input), prepared_model(*dummy_input))
        assert isinstance(prepared_model.module_baddbmm, aimet_modules.Baddbmm)
        assert isinstance(prepared_model.module_baddbmm_1, aimet_modules.Baddbmm)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(*dummy_input)
        assert sim.model.module_baddbmm.output_quantizers[0].encoding
        assert sim.model.module_baddbmm_1.output_quantizers[0].encoding

    def test_fx_with_addmm(self):
        """ test torch fx with addmm """
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 10, (1, 1))

            def forward(self, *inputs):
                x = self.conv(inputs[0]).squeeze()[0]
                x = x.addmm(inputs[1], inputs[2], beta=0.2, alpha=0.4)
                x = torch.addmm(x, inputs[1], inputs[2], beta=0.5)
                x = torch.addmm(x, inputs[1], inputs[2])
                return x

        input_shape = (1, 3, 3, 3)
        dummy_input = (torch.randn(*input_shape), torch.randn(3, 3), torch.rand(3, 3))
        model = Model().eval()
        prepared_model = prepare_model(model)

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify bit-exact outputs.
        assert torch.equal(model(*dummy_input), prepared_model(*dummy_input))
        assert isinstance(prepared_model.module_addmm, aimet_modules.Addmm)
        assert isinstance(prepared_model.module_addmm_1, aimet_modules.Addmm)
        assert isinstance(prepared_model.module_addmm_2, aimet_modules.Addmm)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(*dummy_input)
        assert sim.model.module_addmm.output_quantizers[0].encoding
        assert sim.model.module_addmm_1.output_quantizers[0].encoding
        assert sim.model.module_addmm_2.output_quantizers[0].encoding

    def test_fx_with_bmm(self):
        """ test torch fx with bmm """
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 10, (1, 1))

            def forward(self, *inputs):
                x = self.conv(inputs[0]).squeeze()
                x = x.bmm(inputs[1])
                x = torch.bmm(x, inputs[1])
                return x

        input_shape = (1, 3, 3, 3)
        dummy_input = (torch.randn(*input_shape), torch.randn(10, 3, 3))
        model = Model().eval()
        model(*dummy_input)
        prepared_model = prepare_model(model)
        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)
        #
        # Verify bit-exact outputs.
        assert torch.equal(model(*dummy_input), prepared_model(*dummy_input))
        assert isinstance(prepared_model.module_bmm, aimet_modules.Bmm)
        assert isinstance(prepared_model.module_bmm_1, aimet_modules.Bmm)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(*dummy_input)
        assert sim.model.module_bmm.output_quantizers[0].encoding
        assert sim.model.module_bmm_1.output_quantizers[0].encoding

    def test_fx_with_module_classes_to_exclude(self):
        """ test torch fx with module_classes_to_exclude """

        class Square(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @staticmethod
            def forward( x):
                return torch.square(x)
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 10, (1, 1))
                self.square1 = Square()
                self.square2 = Square()

            def forward(self, *inputs):
                x = self.square1(inputs[0])
                x = self.conv(x)
                x = self.square2(x)
                return x

        input_shape = (1, 3, 3, 3)
        dummy_input = (torch.randn(*input_shape), torch.randn(10, 3, 3))
        model = Model().eval()
        model(*dummy_input)
        prepared_model = prepare_model(model, module_classes_to_exclude=[Square])

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify bit-exact outputs.
        assert torch.equal(model(*dummy_input), prepared_model(*dummy_input))
        assert isinstance(prepared_model.square1, Square)
        assert isinstance(prepared_model.square2, Square)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(*dummy_input)
        assert sim.model.square1.output_quantizers[0].encoding
        assert sim.model.square2.output_quantizers[0].encoding

    def test_fx_with_cumsum(self):
        """ test torch fx with cumsum taking kwargs """

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 10, (1, 1))

            def forward(self, inputs):
                x = self.conv(inputs).squeeze()
                x = torch.cumsum(x, dim=1, dtype=torch.float32)
                return x

        input_shape = (1, 3, 3, 3)
        dummy_input = (torch.randn(*input_shape))
        model = Model().eval()
        prepared_model = prepare_model(model)

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify bit-exact outputs.
        assert torch.equal(model(dummy_input), prepared_model(dummy_input))
        assert isinstance(prepared_model.module_cumsum, aimet_modules.CumSum)

        # Verify with Quantization workflow.
        sim = QuantizationSimModel(prepared_model, dummy_input)
        sim.compute_encodings(evaluate, dummy_input)
        sim.model(dummy_input)
        assert sim.model.module_cumsum.output_quantizers[0].encoding

    def test_const_as_first_operand(self):
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super(SampleModel, self).__init__()

            def forward(self, x):
                x = torch.mul(x, 1.5)
                x = torch.mul(2, x)
                x = x * 3
                x = 4 * x
                x = torch.tensor(5.0) + x
                x = 6 + x
                x = 7.1 - x
                return x

        model = SampleModel()
        model = prepare_model(model)

        dummy_inp = torch.randn(1, 4)
        qsim_model = QuantizationSimModel(model, dummy_input=dummy_inp,
                                          quant_scheme="tf_enhanced",
                                          rounding_mode='nearest', default_output_bw=8,
                                          default_param_bw=8, in_place=False, config_file=None)
        qsim_model.compute_encodings(forward_pass_callback=evaluate, forward_pass_callback_args=dummy_inp)
        print(qsim_model)

        for wrapper_name, wrapper in qsim_model.quant_wrappers():
            # Verify first layer input and output quantizer enabled flag
            if wrapper_name == "module_mul":
                assert wrapper.input_quantizers[0].enabled == True
                assert wrapper.input_quantizers[1].enabled == False
                assert wrapper.output_quantizers[0].enabled == True
            else:
                # Verify rest of layer input and output quantizer enabled flag
                for inp_quant in wrapper.input_quantizers:
                    assert inp_quant.enabled == False
                for out_quant in wrapper.output_quantizers:
                    assert out_quant.enabled == True

    def test_prepare_custom_functional_conv(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.rand(1, 3, 16, 16, device=device)
        original_model = CustomFunctionalConv().to(device)

        with in_eval_mode(original_model):
            prepared_model = prepare_model(original_model)
            assert isinstance(prepared_model.module_conv2d, torch.nn.Conv2d)
            assert torch.allclose(original_model(dummy_input), prepared_model(dummy_input))
