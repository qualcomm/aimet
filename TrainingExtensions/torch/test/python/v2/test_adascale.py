# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Adascale tests"""

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset, DataLoader


from aimet_torch import QuantizationSimModel
from aimet_torch.experimental.adascale.adascale_optimizer import (
    AdaScale,
    AdaScaleModelConfig,
    adascale_model_config_dict,
    apply_adascale,
)
from aimet_torch.experimental.adascale.adascale_quantizer import (
    AdaScaleQuantizeDequantize,
    AdaScaleLinearQuantizeDequantize,
    AdaScaleConv2dQuantizeDequantize,
)
from aimet_torch.v2.nn import QuantizedLinear, QuantizedConv2d
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.utils import remove_all_quantizers, remove_activation_quantizers

from .models_ import test_models


@pytest.mark.parametrize(
    "ada_module_and_shape",
    [
        (AdaScaleLinearQuantizeDequantize, (1, 3, 224, 224), (1, 3, 1, 1)),
        (AdaScaleConv2dQuantizeDequantize, (10, 20, 4, 4), (10, 1, 1, 1)),
    ],
)
def test_adascale_compute_encodings(ada_module_and_shape):
    """
    Given:
    - Create QDQ module, store initial scale and create adascale equivalent with the QDQ module
    - Set Adascale params requires_grad to True
    When:
    - Train with random data
    - Save S2, S3
    Then:
    - S2, S3 Should not be zeros
    - Compare original scale with new scale
    """

    ada_module_type, weight_shape, qdq_shape = ada_module_and_shape
    torch.manual_seed(0)
    input_tensor = torch.rand(*weight_shape)

    torch.manual_seed(1)
    expected_tensor = torch.rand(*weight_shape)

    qdq = QuantizeDequantize(shape=qdq_shape, bitwidth=8, symmetric=True)

    with qdq.compute_encodings():
        _ = qdq(input_tensor)

    adascale_qdq = ada_module_type(qdq, weight_shape)
    assert torch.equal(adascale_qdq.min, qdq.min)
    assert torch.equal(adascale_qdq.max, qdq.max)
    assert torch.equal(qdq(input_tensor), adascale_qdq(input_tensor))

    adascale_qdq.eval()
    lwc_params, scale_params = adascale_qdq.get_adascale_trainable_parameters()
    adascale_params = lwc_params + scale_params
    for p in adascale_params:
        p.requires_grad = True

    prev_loss = None
    for epoch in range(5):
        optimizer = torch.optim.Adam(adascale_params)
        quant_out = adascale_qdq(input_tensor)
        loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
        assert prev_loss != loss
        prev_loss = loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    modified_q = adascale_qdq.get_qdq()
    adascale_out = adascale_qdq(input_tensor)
    input_with_adascale_params_folded = adascale_qdq.get_folded_weight(input_tensor)

    modified_out = modified_q(input_with_adascale_params_folded)

    assert torch.equal(adascale_qdq.get_max(), modified_q.get_max())
    assert torch.equal(adascale_qdq.get_min(), modified_q.get_min())
    assert torch.equal(adascale_qdq.get_scale(), modified_q.get_scale())
    assert torch.equal(adascale_qdq.get_offset(), modified_q.get_offset())

    assert torch.equal(modified_out, adascale_out)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestAdascale:
    @pytest.mark.parametrize(
        "model_and_shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks(), (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks(), (1, 64, 4, 4)),
        ],
    )
    def test_adascale_1(self, model_and_shape: tuple):
        """Test basic flow"""
        model, shape = model_and_shape
        batch_size = 1
        num_iterations = 1

        torch.manual_seed(0)
        dummy_input = torch.rand(shape)
        _ = model(dummy_input)

        data_set = CustomDataset(dummy_input)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        sim = QuantizationSimModel(model, dummy_input)

        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs
                ),
            },
        ):
            apply_adascale(sim, data_loader, None, num_iterations)

        for block in sim.model.blocks:
            for module in block.modules():
                if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                    assert type(module.param_quantizers["weight"]) == QuantizeDequantize
                    assert type(module.param_quantizers["weight"]) == QuantizeDequantize

    @pytest.mark.parametrize(
        "model_and_shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks(), (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks(), (1, 64, 4, 4)),
        ],
    )
    def test_adascale_2(self, model_and_shape):
        """validate QDQ is replaced correctly with AdascaleQDQ"""
        model, shape = model_and_shape
        dummy_input = torch.rand(shape)

        sim = QuantizationSimModel(model, dummy_input)
        sim.model.requires_grad_(False)
        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs
                ),
            },
        ):
            blocks = AdaScale._get_blocks(sim)
            assert len(blocks) == 5

            AdaScale._replace_with_adascale_weight_quantizers(blocks)

            for block in blocks:
                assert isinstance(
                    block.layer1.param_quantizers["weight"], AdaScaleQuantizeDequantize
                )
                assert isinstance(
                    block.layer2.param_quantizers["weight"], AdaScaleQuantizeDequantize
                )

                lwc_params, scale_params = AdaScale._get_adascale_trainable_params(
                    block
                )
                AdaScale._set_requires_grad(lwc_params + scale_params, True)

                for name, param in block.named_parameters():
                    if name in [
                        "layer1.param_quantizers.weight.beta",
                        "layer1.param_quantizers.weight.gamma",
                        "layer1.param_quantizers.weight.s2",
                        "layer1.param_quantizers.weight.s3",
                        "layer1.param_quantizers.weight.s4",
                        "layer2.param_quantizers.weight.beta",
                        "layer2.param_quantizers.weight.gamma",
                        "layer2.param_quantizers.weight.s2",
                        "layer2.param_quantizers.weight.s3",
                        "layer2.param_quantizers.weight.s4",
                    ]:
                        assert param.requires_grad, (
                            "Trainable param is not set to train mode"
                        )
                    else:
                        assert param.requires_grad is False, (
                            "Only adascale params are trainable"
                        )

    @pytest.mark.parametrize(
        "model_and_shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks(), (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks(), (1, 64, 4, 4)),
        ],
    )
    def test_adascale_3(self, model_and_shape):
        """test removing quantizers"""
        model, shape = model_and_shape
        dummy_input = torch.rand(shape)

        sim = QuantizationSimModel(model, dummy_input)
        sim.model.requires_grad_(False)
        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs
                ),
            },
        ):
            blocks = AdaScale._get_blocks(sim)
            AdaScale._replace_with_adascale_weight_quantizers(blocks)

            for block in blocks:
                with remove_all_quantizers(block):
                    for name, param in block.named_parameters():
                        assert name in [
                            "layer1.weight",
                            "layer1.bias",
                            "layer2.weight",
                            "layer2.bias",
                        ]

                lwc_params, scale_params = AdaScale._get_adascale_trainable_params(
                    block
                )
                AdaScale._set_requires_grad(lwc_params + scale_params, True)
                with remove_activation_quantizers(block):
                    for name, param in block.named_parameters():
                        if name in [
                            "layer1.weight",
                            "layer1.bias",
                            "layer2.weight",
                            "layer2.bias",
                            "layer1.param_quantizers.weight.min",
                            "layer1.param_quantizers.weight.max",
                            "layer2.param_quantizers.weight.min",
                            "layer2.param_quantizers.weight.max",
                        ]:
                            assert param.requires_grad == False
                        else:
                            assert param.requires_grad == True
            AdaScale._fold_weights_and_replace_with_qdq(blocks)

    @pytest.mark.parametrize(
        "model_and_shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks(), (200, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks(), (200, 64, 4, 4)),
        ],
    )
    def test_adascale_4(self, model_and_shape):
        """test training of adascale weights"""
        model, shape = model_and_shape

        batch_size = 16
        num_iterations = 130

        torch.manual_seed(0)
        dummy_input = torch.rand(shape)
        data_set = CustomDataset(dummy_input)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(lambda m, _: m(dummy_input), None)

        fp_output = model(dummy_input)
        quantized_output = sim.model(dummy_input)
        loss_before_opt = torch.nn.functional.mse_loss(fp_output, quantized_output)

        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs
                ),
            },
        ):
            apply_adascale(sim, data_loader, None, num_iterations)

        adascale_output = sim.model(dummy_input)
        loss_after_opt = torch.nn.functional.mse_loss(fp_output, adascale_output)
        assert (loss_before_opt - loss_after_opt) > 0

    def test_adascale_5(self):
        dummy_input = torch.rand(1, 3, 32, 64)
        model = test_models.ModelWithConsecutiveLinearBlocks()
        sim = QuantizationSimModel(model, dummy_input)
        lwc_params, scale_params = AdaScale._get_adascale_trainable_params(sim.model)
        assert not lwc_params + scale_params

        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs
                ),
            },
        ):
            adascale_blocks = AdaScale._get_blocks(sim)

            AdaScale._replace_with_adascale_weight_quantizers(adascale_blocks)
            for block in adascale_blocks:
                lwc_params, scale_params = AdaScale._get_adascale_trainable_params(
                    block
                )
                assert (
                    len(lwc_params + scale_params) == 8
                )  # two linear layers X [gamma, beta, s2, s3]
