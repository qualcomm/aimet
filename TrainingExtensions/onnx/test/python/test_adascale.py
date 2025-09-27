# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import numpy as np
import torch
from onnx import numpy_helper
import pytest
from aimet_common.utils import compute_psnr
from aimet_onnx.experimental.adascale.adascale_optimizer import (
    AdaScale,
    adascale_model_config_dict,
)

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    LiteWeightQuantizedLinear,
    AdaScaleWeightQdq,
    WeightQdq,
    get_adascale_trainable_params,
    replace_with_adascale_quantizers,
)


class ModelWithLinears(torch.nn.Module):
    def __init__(self):
        super(ModelWithLinears, self).__init__()

        self.layer1 = torch.nn.Linear(64, 32)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(32, 64)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class ModelWithConsecutiveLinearBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveLinearBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithLinears() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class TestAdascaleOnnx:
    def test_onnx_adascale_3(self):
        class TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # input_size is hardcoded to 10
                self.linear1 = torch.nn.Linear(10, 20)
                self.relu = torch.nn.ReLU()
                # hidden_size is hardcoded to 20, output_size is hardcoded to 5
                self.linear2 = torch.nn.Linear(20, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        model = TwoLayerModel()
        input_shape = (10, 10)
        input_tensor = torch.rand(*input_shape)
        orig_out = model(input_tensor).detach()

        model = add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        temp = model(input_tensor)

        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )

        for m in model.parameters():
            m.requires_grad = False

        for p in all_scale_parameters:
            p.requires_grad_(True)
        for p in all_beta_gamma_parameters:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(all_beta_gamma_parameters + all_scale_parameters)
        for epoch in range(5):
            quant_out = model(input_tensor)
            loss = torch.nn.functional.mse_loss(orig_out, quant_out)
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(name, "is not None. Sum=", param.grad.sum())
            # did_grad_update = any(param.grad is not None for param in all_beta_gamma_parameters)
            # print(f'Any grad present? {did_grad_update}')
            optimizer.zero_grad()

        # did_grad_update = any(param.grad is not None for param in all_beta_gamma_parameters)
        # print(f'Any grad present? {did_grad_update}')

        new_out = model(input_tensor)
        assert not torch.equal(new_out, orig_out)

    @pytest.mark.skip(reason="This test is temporarily disabled")
    def test_onnx_adascale_1(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        model_copy = copy.deepcopy(model)
        input_shape = (1, 3, 32, 64)
        torch.random.manual_seed(1)
        dummy_input = torch.rand(input_shape)
        out_1 = model(copy.deepcopy(dummy_input))

        add_qlinear_layers(model)
        out_2 = model(copy.deepcopy(dummy_input))

        # verify weights have not changed and the classes are swapped correctly
        for linear_block_1, linear_block_2 in zip(model.blocks, model_copy.blocks):
            assert torch.equal(
                linear_block_1.layer1.weight, linear_block_2.layer1.weight
            )
            assert torch.equal(
                linear_block_1.layer2.weight, linear_block_2.layer2.weight
            )

            assert isinstance(linear_block_1.layer1, LiteWeightQuantizedLinear)
            assert isinstance(linear_block_1.layer2, LiteWeightQuantizedLinear)

        # multiple calls show no change in model parameters (no attrs set to train mode)
        out_2_a = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_2, out_2_a)

        for linear_block in model.blocks:
            linear_block.layer1.param_quantizers["weight"] = None
            linear_block.layer2.param_quantizers["weight"] = None

        # with params removed, we should get the un-quantized output
        out_3 = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_3, out_1)

    def test_adascale_compute_encodings(self):
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

        weight_shape, qdq_shape = (1, 3, 224, 224), (1, 3, 1, 1)
        torch.manual_seed(0)
        input_tensor = torch.rand(*weight_shape)

        torch.manual_seed(1)
        expected_tensor = torch.rand(*weight_shape)

        qdq = WeightQdq(input_tensor, qdq_shape, 4)

        adascale_qdq = AdaScaleWeightQdq(input_tensor, qdq_shape, 4)
        assert torch.equal(adascale_qdq.min, qdq.min)
        assert torch.equal(adascale_qdq.max, qdq.max)
        assert torch.equal(qdq(input_tensor), adascale_qdq(input_tensor))

        adascale_qdq.eval()
        lwc_params, scale_params = adascale_qdq.get_adascale_trainable_parameters()
        adascale_params = lwc_params + scale_params
        for p in adascale_params:
            p.requires_grad = True

        orig_output = adascale_qdq(input_tensor)
        prev_loss = None
        optimizer = torch.optim.Adam(adascale_params)
        for epoch in range(5):
            quant_out = adascale_qdq(input_tensor)
            loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
            assert prev_loss != loss
            prev_loss = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        adascale_out = adascale_qdq(input_tensor)
        # verify training is changing the output
        assert not torch.equal(adascale_out, orig_output)

        # verify adascale_qdq can be converted to regular qdq
        input_with_adascale_params_folded = adascale_qdq.get_folded_weight(input_tensor)
        new_qdq = WeightQdq(input_tensor, qdq_shape, 4)
        new_qdq.set_range(adascale_qdq.get_min(), adascale_qdq.get_max())
        assert torch.equal(adascale_qdq.get_max(), new_qdq.get_max())
        assert torch.equal(adascale_qdq.get_min(), new_qdq.get_min())
        assert torch.equal(adascale_qdq.get_scale(), new_qdq.get_scale())
        assert torch.equal(adascale_qdq.get_offset(), new_qdq.get_offset())

        modified_out = new_qdq(input_with_adascale_params_folded)
        assert torch.equal(modified_out, adascale_out)

    def test_onnx_adascale_2(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )
        assert (
            len(all_beta_gamma_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(beta, gamma)
        assert (
            len(all_scale_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(s2, s3)


def test_adasclae_e2e(monkeypatch, small_model: bool = True):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX

    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2-0.5B"
    model_cls = Qwen_25_ONNX
    sim, config = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, small_model=small_model
    )

    onnx_weights_min_max = {}
    for initializer in sim.model.model.graph.initializer:
        weight_array = numpy_helper.to_array(initializer)
        onnx_weights_min_max[initializer.name] = {
            "min": float(np.min(weight_array)),
            "max": float(np.max(weight_array)),
        }
    adascale_model_config_dict["Qwen2Model"].model_config = config

    inputs = {
        "input_ids": np.random.randint(0, 100, size=(1, 16), dtype=np.int32),
        "attention_mask": np.random.randint(0, 100, size=(1, 1, 16, 32)).astype(
            np.float32
        ),
        "position_ids": np.arange(0, 16).reshape(1, 16).astype(np.int32),
        "past_key_0_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_value_0_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_key_1_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_value_1_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
    }

    AdaScale.apply_adascale(
        sim,
        [inputs],
        adascale_model_config_dict["Qwen2Model"],
        num_iterations=2,
    )

    for initializer in sim.model.model.graph.initializer:
        weight_array = numpy_helper.to_array(initializer)
        if initializer.name in [
            "onnx::MatMul_571",
            "onnx::MatMul_587",
            "onnx::MatMul_588",
            "onnx::MatMul_643",
            "onnx::MatMul_644",
            "onnx::MatMul_645",
            "onnx::MatMul_646",
            "onnx::MatMul_647",
            "onnx::MatMul_663",
            "onnx::MatMul_664",
            "onnx::MatMul_719",
            "onnx::MatMul_720",
            "onnx::MatMul_721",
            "onnx::MatMul_722",
        ]:
            assert onnx_weights_min_max[initializer.name]["min"] != float(
                np.min(weight_array)
            )
            assert onnx_weights_min_max[initializer.name]["max"] != float(
                np.max(weight_array)
            )
        else:
            assert onnx_weights_min_max[initializer.name]["min"] == float(
                np.min(weight_array)
            )
            assert onnx_weights_min_max[initializer.name]["max"] == float(
                np.max(weight_array)
            )

    assert len(sim.model.model.graph.output)
