# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Adascale tests"""

from unittest.mock import patch

import copy
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


class TestAdascaleQuantizer:
    def test_zero_point_shift(self):
        qdq = QuantizeDequantize(
            shape=(), bitwidth=4, symmetric=True, zero_point_shift=0.5
        )
        dummy_input = torch.tensor([-12.0, 6.0])
        with qdq.compute_encodings():
            _ = qdq(dummy_input)
        assert torch.equal(-qdq.min, qdq.max)

        dummy_input_2 = torch.tensor([-24.0, 12.0])
        adascale_qdq = AdaScaleLinearQuantizeDequantize(
            qdq, weight_shape=dummy_input.shape
        )
        assert adascale_qdq.zero_point_shift == 0.5
        assert torch.equal(-adascale_qdq.min, adascale_qdq.max)
        out = adascale_qdq(dummy_input_2)
        assert torch.equal(-out[0], out[1])

        new_qdq = adascale_qdq.get_qdq()
        assert new_qdq.zero_point_shift == 0.5
        assert torch.equal(-new_qdq.min, new_qdq.max)
        out = new_qdq(dummy_input_2)
        assert torch.equal(-out[0], out[1])


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

            for block in blocks:
                AdaScale._replace_with_adascale_weight_quantizers(block)

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
            for block in blocks:
                AdaScale._replace_with_adascale_weight_quantizers(block)

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

            AdaScale.fold_adascale_quantizers(sim.model)

    @pytest.mark.cuda()
    @pytest.mark.parametrize(
        "model_and_shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks(), (200, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks(), (200, 64, 4, 4)),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adascale_4(self, model_and_shape, dtype):
        """test training of adascale weights"""
        model, shape = model_and_shape
        model = model.to(dtype=dtype, device=torch.device("cuda"))

        batch_size = 16
        num_iterations = 130

        torch.manual_seed(0)
        dummy_input = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
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

            for block in adascale_blocks:
                AdaScale._replace_with_adascale_weight_quantizers(block)
            for block in adascale_blocks:
                lwc_params, scale_params = AdaScale._get_adascale_trainable_params(
                    block
                )
                assert (
                    len(lwc_params + scale_params) == 8
                )  # two linear layers X [gamma, beta, s2, s3]

    @pytest.mark.cuda()
    @pytest.mark.parametrize(
        "num_blocks,num_blocks_with_post_process", [(6, 3), (5, 0)]
    )
    def test_adascale_with_block_output_postprocessing(
        self, num_blocks, num_blocks_with_post_process
    ):
        dummy_input = torch.rand(200, 3, 32, 64)
        model = test_models.ModelWithLinearBlocks(
            num_blocks, num_blocks_with_post_process
        )
        sim = QuantizationSimModel(model, dummy_input, default_param_bw=4)
        sim.model.cuda()

        batch_size = 16
        num_iterations = 130

        data_set = CustomDataset(dummy_input)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears,
                    enable_caching_after_block=num_blocks_with_post_process,
                )
            },
        ):
            apply_adascale(sim, data_loader, None, num_iterations)

    def test_adascale_zero_point_shift(self):
        torch.manual_seed(0)
        dummy_input = torch.rand(200, 3, 32, 64)
        model = test_models.ModelWithConsecutiveLinearBlocks()
        sim = QuantizationSimModel(
            model, dummy_input, default_param_bw=4, config_file="default_config.json"
        )
        for module in sim.qmodules():
            if isinstance(module, torch.nn.Linear):
                module.param_quantizers["weight"].zero_point_shift = 0.5
        sim_copy = copy.deepcopy(sim)
        sim_copy.compute_encodings(lambda m: m(dummy_input))

        batch_size = 16
        num_iterations = 130

        data_set = CustomDataset(dummy_input)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        fp_output = model(dummy_input)
        quantized_output = sim_copy.model(dummy_input)
        loss_before_opt = torch.nn.functional.mse_loss(fp_output, quantized_output)
        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears
                )
            },
        ):
            apply_adascale(sim, data_loader, None, num_iterations)

        sim.compute_encodings(lambda m, _: m(dummy_input), None)
        adascale_output = sim.model(dummy_input)
        loss_after_opt = torch.nn.functional.mse_loss(fp_output, adascale_output)
        assert (loss_before_opt - loss_after_opt) > 0

        model = sim.get_original_model(sim.model, qdq_weights=True)
        found_linear = False
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                found_linear = True
                assert torch.allclose(
                    torch.abs(torch.min(module.weight)),
                    torch.max(module.weight),
                    atol=1e-7,
                )
        assert found_linear

    def test_block_level_adascale(self):
        dummy_input = torch.rand(1, 3, 32, 64)
        model = test_models.ModelWithConsecutiveLinearBlocks()
        sim = QuantizationSimModel(model, dummy_input)

        fp_inputs = [((torch.rand(1, 3, 32, 64),), {}) for _ in range(3)]
        qt_inputs = fp_inputs
        for block in sim.model.blocks:
            AdaScale.adascale_block(
                block, fp_inputs, qt_inputs=qt_inputs, num_iterations=10
            )

            with remove_all_quantizers(block), torch.no_grad():
                fp_inputs = [((block(*inp),), {}) for inp, _ in fp_inputs]

            with remove_activation_quantizers(block), torch.no_grad():
                qt_inputs = [((block(*inp),), {}) for inp, _ in qt_inputs]

        linear_layers = [
            mod for mod in sim.model.modules() if isinstance(mod, torch.nn.Linear)
        ]
        adascale_quantizers = [
            mod
            for mod in sim.model.modules()
            if isinstance(mod, AdaScaleQuantizeDequantize)
        ]

        assert len(linear_layers) == len(adascale_quantizers)

        AdaScale.fold_adascale_quantizers(sim.model)

        adascale_quantizers = [
            mod
            for mod in sim.model.modules()
            if isinstance(mod, AdaScaleQuantizeDequantize)
        ]

        assert len(adascale_quantizers) == 0
