# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Adascale tests"""

from pathlib import Path
import tempfile
from unittest.mock import patch
from copy import deepcopy

import copy
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaModel
from transformers import set_seed

from aimet_torch import QuantizationSimModel
from aimet_torch.adascale.adascale_optimizer import (
    AdaScale,
    AdaScaleModelConfig,
    adascale_model_config_dict,
    apply_adascale,
    PerBlockCheckpointManager,
    _mse_loss_fn,
)
from aimet_torch.adascale.adascale_quantizer import (
    AdaScaleQuantizeDequantize,
    AdaScaleLinearQuantizeDequantize,
    AdaScaleConv2dQuantizeDequantize,
)
from aimet_torch.v2.nn import QuantizedLinear, QuantizedConv2d
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.utils import remove_all_quantizers, remove_activation_quantizers

from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache

from .models_ import test_models


SEQUENCE_LENGTH = 32
CONTEXT_LENGTH = 64
VOCAB_SIZE = 128
HIDDEN_SIZE = 16
NUM_HIDDEN_LAYERS = 4
NUM_ATTN_HEADS = 2
NUM_ITERATIONS = 10

MODEL_CONFIGS = {
    "llama": {
        "config_class": LlamaConfig,
        "model_class": LlamaForCausalLM,
        "config_kwargs": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": HIDDEN_SIZE * 2,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_ATTN_HEADS,
            "max_position_embeddings": CONTEXT_LENGTH,
        },
    },
    "qwen2": {
        "config_class": Qwen2Config,
        "model_class": Qwen2ForCausalLM,
        "config_kwargs": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": HIDDEN_SIZE * 2,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_ATTN_HEADS,
            "max_position_embeddings": CONTEXT_LENGTH,
        },
    },
    "mistral": {
        "config_class": MistralConfig,
        "model_class": MistralForCausalLM,
        "config_kwargs": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": HIDDEN_SIZE * 2,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_ATTN_HEADS,
            "max_position_embeddings": CONTEXT_LENGTH,
        },
    },
    "phi3": {
        "config_class": Phi3Config,
        "model_class": Phi3ForCausalLM,
        "config_kwargs": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": HIDDEN_SIZE * 2,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_ATTN_HEADS,
            "max_position_embeddings": CONTEXT_LENGTH,
            "pad_token_id": None,
        },
    },
    "qwen3": {
        "config_class": Qwen3Config,
        "model_class": Qwen3ForCausalLM,
        "config_kwargs": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": HIDDEN_SIZE * 2,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_ATTN_HEADS,
            "max_position_embeddings": CONTEXT_LENGTH,
        },
    },
}

supported_modules: List = [QuantizedLinear, QuantizedConv2d]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=list(MODEL_CONFIGS.keys()))
def fxt_model_config(request):
    return MODEL_CONFIGS[request.param]


@pytest.fixture
def fxt_model(fxt_model_config):
    config = fxt_model_config["config_class"](**fxt_model_config["config_kwargs"])
    model = fxt_model_config["model_class"](config)
    model.eval()
    return model


@pytest.fixture
def fxt_dummy_input():
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQUENCE_LENGTH)),
        "attention_mask": torch.ones(1, SEQUENCE_LENGTH, dtype=torch.int),
    }


@pytest.fixture
def fxt_dataloader():
    class _Dataset(Dataset):
        def __init__(self, size=64):
            self._size = size

        def __getitem__(self, idx):
            # deterministic tokens
            ids = torch.full((SEQUENCE_LENGTH,), idx % VOCAB_SIZE, dtype=torch.int)
            mask = torch.full((SEQUENCE_LENGTH,), 0, dtype=torch.int)
            return ids, mask

        def __len__(self):
            return self._size

    return DataLoader(dataset=_Dataset(), batch_size=1, shuffle=False)


@pytest.fixture
def fxt_checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def fxt_quantsim_ready_model(fxt_model, fxt_dummy_input):
    traceable_model = ONNXExportableModuleWithCache(
        fxt_model, input_names=tuple(fxt_dummy_input.keys())
    )
    sim = QuantizationSimModel(
        model=traceable_model,
        dummy_input=tuple(fxt_dummy_input.values()),
        default_output_bw=16,
        default_param_bw=4,
    )
    return sim


@pytest.fixture
def fxt_block(fxt_quantsim_ready_model):
    return AdaScale._get_blocks(fxt_quantsim_ready_model)[0]


# ============================================================================
# Helpper Function
# ============================================================================


def get_quantsim_ready_model(model, dummy_input: dict):
    traceable_model = ONNXExportableModuleWithCache(
        model, input_names=tuple(dummy_input.keys())
    )
    sim = QuantizationSimModel(
        model=traceable_model,
        dummy_input=tuple(dummy_input.values()),
        default_output_bw=16,
        default_param_bw=4,
    )
    return sim


def has_adascale_quantizers(block: torch.nn.Module) -> bool:
    """Check if block already has AdaScale quantizers installed"""
    for module in block.modules():
        if isinstance(module, tuple(supported_modules)):
            weight_quantizer = module.param_quantizers["weight"]
            if isinstance(weight_quantizer, AdaScaleQuantizeDequantize):
                return True

    return False


def run_n_times_then_stop_non_return(func: callable, n: int):
    """Helper to simulate cancellation"""

    def f(*args, **kwargs):
        if f._count >= n:
            raise RuntimeError("cancel")

        func(*args, **kwargs)
        f._count += 1

    f._count = 0
    return f


def run_n_times_then_stop(func: callable, n: int):
    """Helper to simulate cancellation"""

    def wrapper(*args, **kwargs):
        if wrapper._count >= n:
            raise RuntimeError("cancel")

        output = func(*args, **kwargs)
        wrapper._count += 1
        return output

    wrapper._count = 0
    return wrapper


def count_adascale_quantizers(model):
    """Helper to count number of AdaScale quantizers in model"""
    count = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            if isinstance(
                module.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                count += 1
    return count


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
    @pytest.mark.parallel
    @pytest.mark.parametrize(
        "model_cls, shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks, (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks, (1, 64, 4, 4)),
        ],
    )
    def test_adascale_1(self, model_cls, shape):
        """Test basic flow"""
        model = model_cls()
        batch_size = 1

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
            apply_adascale(sim, data_loader, None, NUM_ITERATIONS)

        for block in sim.model.blocks:
            for module in block.modules():
                if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                    assert type(module.param_quantizers["weight"]) == QuantizeDequantize
                    assert type(module.param_quantizers["weight"]) == QuantizeDequantize

    @pytest.mark.parallel
    @pytest.mark.parametrize(
        "model_cls, shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks, (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks, (1, 64, 4, 4)),
        ],
    )
    def test_adascale_2(self, model_cls, shape):
        """validate QDQ is replaced correctly with AdascaleQDQ"""
        model = model_cls()
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

    @pytest.mark.parallel
    @pytest.mark.parametrize(
        "model_cls, shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks, (1, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks, (1, 64, 4, 4)),
        ],
    )
    def test_adascale_3(self, model_cls, shape):
        """test removing quantizers"""
        model = model_cls()
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
        "model_cls, shape",
        [
            (test_models.ModelWithConsecutiveLinearBlocks, (200, 3, 32, 64)),
            (test_models.ModelWithConsecutiveConv2dBlocks, (200, 64, 4, 4)),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adascale_4(self, model_cls, shape, dtype):
        """test training of adascale weights"""
        set_seed(0)
        num_iterations = NUM_ITERATIONS * 2
        model = model_cls()
        model = model.to(dtype=dtype, device=torch.device("cuda"))
        dummy_input = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
        data_set = CustomDataset(dummy_input)
        batch_size = dummy_input.shape[0] // num_iterations
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        sim = QuantizationSimModel(model, dummy_input, default_param_bw=4)
        sim.compute_encodings(lambda m: m(dummy_input))

        fp_output = model(dummy_input)
        quantized_output = sim.model(dummy_input)
        loss_before_opt = torch.nn.functional.mse_loss(fp_output, quantized_output)

        with patch.dict(
            adascale_model_config_dict,
            {
                test_models.ModelWithConsecutiveLinearBlocks: AdaScaleModelConfig(
                    test_models.ModelWithLinears,
                ),
                test_models.ModelWithConsecutiveConv2dBlocks: AdaScaleModelConfig(
                    test_models.ModelWithConvs,
                ),
            },
        ):
            apply_adascale(sim, data_loader, None, num_iterations)

        adascale_output = sim.model(dummy_input)
        loss_after_opt = torch.nn.functional.mse_loss(fp_output, adascale_output)
        assert loss_before_opt > loss_after_opt

    @pytest.mark.parallel
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
            apply_adascale(sim, data_loader, None, NUM_ITERATIONS)

    @pytest.mark.parallel
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
            apply_adascale(sim, data_loader, None, NUM_ITERATIONS)

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

    @pytest.mark.parallel
    def test_block_level_adascale(self):
        dummy_input = torch.rand(1, 3, 32, 64)
        model = test_models.ModelWithConsecutiveLinearBlocks()
        sim = QuantizationSimModel(model, dummy_input)

        fp_inputs = [((torch.rand(1, 3, 32, 64),), {}) for _ in range(3)]
        qt_inputs = fp_inputs
        for block in sim.model.blocks:
            AdaScale.adascale_block(
                block, fp_inputs, qt_inputs=qt_inputs, num_iterations=NUM_ITERATIONS
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

    @pytest.mark.parallel
    @pytest.mark.parametrize("seq_len", [8, 32, 2048])
    def test_mse_loss_fn(self, seq_len):
        """For p=2, the default loss equals plain MSE times dim 1's size."""
        torch.manual_seed(0)
        fp_out = torch.rand(4, seq_len, 16)  # [B, S, H]
        qt_out = torch.rand(4, seq_len, 16)

        lp = _mse_loss_fn(fp_out, qt_out)

        mse = torch.nn.functional.mse_loss(fp_out, qt_out)
        assert torch.allclose(lp, mse * seq_len)

    @pytest.mark.parallel
    def test_block_level_adascale_custom_loss_fn(self):
        """Test that adascale_block accepts and uses a custom loss function and
        passes the calibration input index (data_idx) cycling over the inputs."""
        dummy_input = torch.rand(1, 3, 32, 64)
        model = test_models.ModelWithConsecutiveLinearBlocks()
        sim = QuantizationSimModel(model, dummy_input)

        call_count = 0
        seen_indices = []

        def custom_loss_fn(fp_out, qt_out, data_idx):
            nonlocal call_count
            call_count += 1
            seen_indices.append(data_idx)
            return torch.nn.functional.l1_loss(fp_out, qt_out)

        num_inputs = 3
        fp_inputs = [((torch.rand(1, 3, 32, 64),), {}) for _ in range(num_inputs)]
        qt_inputs = fp_inputs

        block = sim.model.blocks[0]
        AdaScale.adascale_block(
            block,
            fp_inputs,
            qt_inputs=qt_inputs,
            num_iterations=NUM_ITERATIONS,
            loss_fn=custom_loss_fn,
        )

        assert call_count == NUM_ITERATIONS
        # data_idx cycles 0..num_inputs-1 across epochs
        assert seen_indices == [i % num_inputs for i in range(NUM_ITERATIONS)]

    @pytest.mark.parallel
    def test_block_level_adascale_early_stopping(self):
        """Integration test for the _EARLY_STOPPING flag using the real factory and
        _EarlyStopping."""
        from aimet_torch.adascale import adascale_optimizer as opt
        from aimet_torch.common.early_stopping import _EarlyStoppingConfig

        def make_block():
            torch.manual_seed(0)
            dummy_input = torch.rand(1, 3, 32, 64)
            model = test_models.ModelWithConsecutiveLinearBlocks()
            sim = QuantizationSimModel(model, dummy_input)
            return sim.model.blocks[0]

        fp_inputs = [((torch.rand(1, 3, 32, 64),), {}) for _ in range(3)]

        def make_counting_loss_fn(counter):
            def loss_fn(fp_out, qt_out, data_idx):
                counter[0] += 1
                return torch.nn.functional.mse_loss(fp_out, qt_out)

            return loss_fn

        # Early stopping ON
        cfg = _EarlyStoppingConfig(check_interval=1, rel_threshold=1e9, window=1)
        on_count = [0]
        with patch.object(opt, "_EARLY_STOPPING", cfg):
            AdaScale.adascale_block(
                make_block(),
                fp_inputs,
                qt_inputs=fp_inputs,
                num_iterations=NUM_ITERATIONS,
                loss_fn=make_counting_loss_fn(on_count),
            )
        assert 0 < on_count[0] < NUM_ITERATIONS

        # Early stopping OFF (default): the loop runs the full schedule.
        assert opt._EARLY_STOPPING is None
        off_count = [0]
        AdaScale.adascale_block(
            make_block(),
            fp_inputs,
            qt_inputs=fp_inputs,
            num_iterations=NUM_ITERATIONS,
            loss_fn=make_counting_loss_fn(off_count),
        )
        assert off_count[0] == NUM_ITERATIONS


@pytest.mark.parallel
class TestAdaScaleBasicFunctionality:
    """Test basic AdaScale functionality across all supported models"""

    def test_adascale_quantizer_initialization(self):
        """Test AdaScale quantizer can be created from standard quantizer"""
        from aimet_torch.v2.quantization.affine import QuantizeDequantize

        # Create standard quantizer
        qdq = QuantizeDequantize(
            shape=(64,),
            bitwidth=8,
            symmetric=True,
        )
        qdq.set_range(torch.tensor(-1.0), torch.tensor(1.0))

        # Create AdaScale quantizer
        weight_shape = torch.Size([64, 64])
        adascale_qdq = AdaScaleLinearQuantizeDequantize(qdq, weight_shape)

        assert hasattr(adascale_qdq, "beta")
        assert hasattr(adascale_qdq, "gamma")
        assert hasattr(adascale_qdq, "s2")
        assert hasattr(adascale_qdq, "s3")

        assert torch.allclose(adascale_qdq.beta, torch.zeros(64))
        assert torch.allclose(adascale_qdq.gamma, torch.zeros(64))

    def test_get_blocks(self, fxt_quantsim_ready_model):
        """Test _get_blocks correctly identifies decoder blocks for all model types"""
        blocks = AdaScale._get_blocks(fxt_quantsim_ready_model)

        assert len(blocks) == NUM_HIDDEN_LAYERS
        # Blocks should be of correct type
        assert all(hasattr(block, "self_attn") for block in blocks)
        assert all(hasattr(block, "mlp") for block in blocks)

    def test_replace_with_adascale_quantizers(self, fxt_quantsim_ready_model):
        """Test replacing standard quantizers with AdaScale quantizers"""
        # Initially should have standard quantizers
        model = fxt_quantsim_ready_model.model
        assert count_adascale_quantizers(model) == 0
        assert not has_adascale_quantizers(model)

        # Replace with AdaScale quantizers
        AdaScale._replace_with_adascale_weight_quantizers(model)

        # Now should have AdaScale quantizers
        assert count_adascale_quantizers(model) > 0
        assert has_adascale_quantizers(model)

    def test_extract_and_restore_adascale_params(self, fxt_block):
        """Test extracting and restoring AdaScale parameters"""
        AdaScale._replace_with_adascale_weight_quantizers(fxt_block)

        for module in fxt_block.modules():
            if isinstance(module, QuantizedLinear):
                weight_quantizer = module.param_quantizers["weight"]
                if isinstance(weight_quantizer, AdaScaleQuantizeDequantize):
                    with torch.no_grad():
                        for idx, param in enumerate(weight_quantizer.parameters()):
                            param.fill_(0.1 * (idx % 10 + 1))

        # Extract parameters
        expected_states = AdaScale.extract_adascale_params(fxt_block)
        assert len(expected_states) > 0

        # Corrupt all AdaScale params
        for module in fxt_block.modules():
            if isinstance(module, QuantizedLinear):
                quantizer = module.param_quantizers["weight"]
                if isinstance(quantizer, AdaScaleQuantizeDequantize):
                    with torch.no_grad():
                        for param in quantizer.parameters():
                            param.fill_(999.0)

        # Restore parameters
        AdaScale.restore_adascale_params(fxt_block, expected_states)

        # Verify all layers are correctly restored
        for name, module in fxt_block.named_modules():
            if name not in expected_states.keys():
                continue
            quantizer = module.param_quantizers["weight"]
            assert isinstance(quantizer, AdaScaleQuantizeDequantize)
            restored_state = quantizer.state_dict()
            for key, expected_tensor in expected_states[name].items():
                if key in ("extra_state", "_extra_state"):
                    continue
                assert torch.allclose(
                    restored_state[key].cpu(), expected_tensor.cpu()
                ), f"Mismatch after restore for layer {name}, key {key}"

    def test_fold_adascale_quantizers(self, fxt_block):
        """Test folding AdaScale parameters into weights"""
        AdaScale._replace_with_adascale_weight_quantizers(fxt_block)

        # Get original weight
        original_weight = None
        for module in fxt_block.modules():
            if isinstance(module, QuantizedLinear):
                original_weight = module.weight.data.clone()
                break

        # Fold quantizers
        AdaScale.fold_adascale_quantizers(fxt_block)

        assert count_adascale_quantizers(fxt_block) == 0

        # Check weight shape is preserved
        for module in fxt_block.modules():
            if isinstance(module, QuantizedLinear):
                assert module.weight.shape == original_weight.shape
                break


@pytest.mark.parallel
class TestCheckpointManagerFunctionality:
    def test_checkpoint_manager_initialization(self, fxt_checkpoint_dir):
        """Test checkpoint manager can be created"""
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)
        assert manager.checkpoint_dir.exists()
        assert manager.progress_file == Path(fxt_checkpoint_dir) / "progress.json"

    def test_save_and_load_completed_block(self, fxt_block, fxt_checkpoint_dir):
        """Test saving and loading completed block adascale params and subsequent folding"""
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)

        AdaScale._replace_with_adascale_weight_quantizers(fxt_block)

        for module in fxt_block.modules():
            if isinstance(module, tuple(supported_modules)):
                weight_quantizer = module.param_quantizers["weight"]
                if isinstance(weight_quantizer, AdaScaleQuantizeDequantize):
                    with torch.no_grad():
                        for idx, param in enumerate(weight_quantizer.parameters()):
                            param.fill_(0.1 * (idx % 10 + 1))

        original_adascale_params = AdaScale.extract_adascale_params(fxt_block)
        assert len(original_adascale_params) > 0, (
            "Block should have AdaScale quantizers"
        )

        # Save params
        manager.save_completed_block(
            block=fxt_block,
            block_idx=0,
        )

        checkpoint_file = Path(fxt_checkpoint_dir) / "checkpoint_block_0.safetensors"
        assert checkpoint_file.exists()

        # Corrupt the AdaScale params
        for module in fxt_block.modules():
            if isinstance(module, tuple(supported_modules)):
                if isinstance(
                    module.param_quantizers["weight"], AdaScaleQuantizeDequantize
                ):
                    with torch.no_grad():
                        for param in module.param_quantizers["weight"].parameters():
                            param.fill_(999.0)

        # Load AdaScale params
        manager.load_completed_block(fxt_block, block_idx=0)

        original_adascale_params_nested = AdaScale.unflatten_state_dict(
            original_adascale_params
        )
        for name, module in fxt_block.named_modules():
            if name in original_adascale_params_nested:
                weight_quantizer = module.param_quantizers["weight"]
                assert isinstance(weight_quantizer, AdaScaleQuantizeDequantize)
                restored_state = weight_quantizer.state_dict()
                for key, tensor in original_adascale_params_nested[name].items():
                    assert torch.allclose(restored_state[key].cpu(), tensor.cpu()), (
                        f"Restored adascale param mismatch for layer {name}, key {key}"
                    )

        # Verify progress was updated
        assert manager.is_block_completed(0)
        progress = manager.get_progress()
        assert 0 in progress["completed_blocks"]
        assert progress["current_block"] == 1

        # Verify no stale .pt checkpoint exists (format migrated to .safetensors)
        stale_pt_file = Path(fxt_checkpoint_dir) / "checkpoint_block_0.pt"
        assert not stale_pt_file.exists()

    def test_progress_tracking(self, fxt_checkpoint_dir):
        """Test progress tracking functionality"""
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)

        # Get initial progress (should be empty)
        progress = manager.get_progress()
        assert progress["total_blocks"] == 0

        # Update progress with metadata
        manager._update_progress(
            0, completed=False, total_blocks=5, config={"num_iterations": 100}
        )

        # Check progress file exists
        assert manager.progress_file.exists()

        # Get progress
        progress = manager.get_progress()
        assert progress["total_blocks"] == 5
        assert progress["config"]["num_iterations"] == 100

        # Update progress
        manager._update_progress(0, completed=True)
        progress = manager.get_progress()
        assert 0 in progress["completed_blocks"]
        assert progress["current_block"] == 1

    def test_is_block_completed(self, fxt_block, fxt_checkpoint_dir):
        """Test checking if block is completed"""
        AdaScale._replace_with_adascale_weight_quantizers(fxt_block)

        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)

        # Initially not completed
        assert not manager.is_block_completed(0)

        # Save block
        manager.save_completed_block(fxt_block, 0)

        # Now completed
        assert manager.is_block_completed(0)

    def test_get_resume_point(self, fxt_checkpoint_dir):
        """Test getting resume point"""
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)

        # Initially should start from beginning
        all_done, block_idx = manager.get_resume_point()
        assert not all_done
        assert block_idx == 0

        # Mark some blocks as completed
        manager._update_progress(0, completed=True, total_blocks=3)
        manager._update_progress(1, completed=True)

        # Should resume from block 2
        all_done, block_idx = manager.get_resume_point()
        assert not all_done
        assert block_idx == 2


@pytest.mark.parallel
class TestAdaScaleResumability:
    """Test AdaScale resumability features"""

    @pytest.mark.parametrize("n_times", [1, NUM_HIDDEN_LAYERS])
    def test_cancel_after_block_completes(
        self, n_times, fxt_quantsim_ready_model, fxt_dataloader, fxt_checkpoint_dir
    ):
        """
        Test cancellation after n blocks complete, including all blocks completion.
        This test is to test the case of saving and loading completed blocks.
        """
        # Simulate cancellation after n blocks
        with patch.object(
            AdaScale,
            "adascale_block",
            side_effect=run_n_times_then_stop(AdaScale.adascale_block, n=n_times),
        ):
            if n_times < NUM_HIDDEN_LAYERS:
                with pytest.raises(RuntimeError, match="cancel"):
                    AdaScale.apply_adascale(
                        fxt_quantsim_ready_model,
                        fxt_dataloader,
                        num_iterations=NUM_ITERATIONS,
                        checkpoint_dir=fxt_checkpoint_dir,
                    )
            else:
                # Should complete successfully if all blocks are done
                AdaScale.apply_adascale(
                    fxt_quantsim_ready_model,
                    fxt_dataloader,
                    num_iterations=NUM_ITERATIONS,
                    checkpoint_dir=fxt_checkpoint_dir,
                )

        # Verify checkpoint state and progress file
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)
        progress = manager.get_progress()

        assert progress["current_block"] == n_times

        completed_blocks = set(progress["completed_blocks"])
        expected_completed = set(range(n_times))
        assert completed_blocks == expected_completed

        for i in range(n_times):
            assert manager.is_block_completed(i), f"Block {i} should be completed"

        for i in range(n_times, NUM_HIDDEN_LAYERS):
            assert not manager.is_block_completed(i), (
                f"Block {i} should not be completed"
            )

        # If all blocks completed, verify optimization is done
        if n_times == NUM_HIDDEN_LAYERS:
            all_done, _ = manager.get_resume_point()
            assert all_done, "All blocks should be marked as done"

    def test_resume_reproducibility_and_determinism(
        self, fxt_model_config, fxt_dummy_input, fxt_dataloader, fxt_checkpoint_dir
    ):
        """
        Test that AdaScale optimization is deterministic and resume produces identical results.
        This test verifies:
        1. Same seed produces same optimization results (determinism)
        2. Interrupted + resumed optimization produces same final weights as uninterrupted run
        """

        # Set the same seed before adascale_block() to ensure determinism (gradient, initilization,...)
        def _set_seed_hook(func: callable):
            def f(*args, **kwargs):
                set_seed(432)
                func(*args, **kwargs)

            return f

        # =========================================================================
        # Part 1: Run complete optimization without interruption (baseline)
        # =========================================================================
        set_seed(432)
        config1 = fxt_model_config["config_class"](**fxt_model_config["config_kwargs"])
        model1 = fxt_model_config["model_class"](config1)
        model2 = deepcopy(model1)
        model1.eval()
        qsim1 = get_quantsim_ready_model(model1, fxt_dummy_input)
        with patch.object(
            AdaScale,
            "adascale_block",
            wraps=_set_seed_hook(AdaScale.adascale_block),
        ):
            AdaScale.apply_adascale(
                qsim1,
                fxt_dataloader,
                num_iterations=NUM_ITERATIONS,
                checkpoint_dir=None,
            )

        # Capture final folded weights from baseline
        baseline_weights = {}
        for block_idx, block in enumerate(AdaScale._get_blocks(qsim1)):
            baseline_weights[block_idx] = {
                name: module.weight.data.clone().cpu()
                for name, module in block.named_modules()
                if isinstance(module, QuantizedLinear)
                and isinstance(module.param_quantizers["weight"], QuantizeDequantize)
            }

        # =========================================================================
        # Part 2: Run with interruption and resume
        # =========================================================================
        set_seed(432)  # Same seed for reproducibility
        model2.eval()
        qsim2 = get_quantsim_ready_model(model2, fxt_dummy_input)

        n_times = int(NUM_ITERATIONS * 1.5)  # first block is completed
        with patch.object(
            AdaScale,
            "adascale_block",
            wraps=_set_seed_hook(AdaScale.adascale_block),
        ):
            with patch.object(
                torch.optim.Adam,
                "step",
                wraps=run_n_times_then_stop_non_return(
                    torch.optim.Adam.step, n=n_times
                ),
            ):
                with pytest.raises(RuntimeError, match="cancel"):
                    AdaScale.apply_adascale(
                        qsim2,
                        fxt_dataloader,
                        num_iterations=NUM_ITERATIONS,
                        checkpoint_dir=fxt_checkpoint_dir,
                    )

        # Verify interruption state
        manager = PerBlockCheckpointManager(fxt_checkpoint_dir)
        progress = manager.get_progress()
        assert progress["current_block"] == 1, "Should have completed first block"
        assert 0 in progress["completed_blocks"]

        # Resume and complete
        with patch.object(
            AdaScale,
            "adascale_block",
            wraps=_set_seed_hook(AdaScale.adascale_block),
        ):
            AdaScale.apply_adascale(
                qsim2,
                fxt_dataloader,
                num_iterations=NUM_ITERATIONS,
                checkpoint_dir=fxt_checkpoint_dir,
            )

        # Capture final folded weights from resumed run
        resumed_weights = {}
        for block_idx, block in enumerate(AdaScale._get_blocks(qsim2)):
            resumed_weights[block_idx] = {
                name: module.weight.data.clone().cpu()
                for name, module in block.named_modules()
                if isinstance(module, QuantizedLinear)
                and isinstance(module.param_quantizers["weight"], QuantizeDequantize)
            }

        # =========================================================================
        # Part 3: Verify reproducibility - compare baseline vs resumed
        # =========================================================================
        for block_idx in range(NUM_HIDDEN_LAYERS):
            for layer_name, baseline_weight in baseline_weights[block_idx].items():
                assert layer_name in resumed_weights[block_idx], (
                    f"Layer {layer_name} should exist in resumed block {block_idx}"
                )
                resumed_weight = resumed_weights[block_idx][layer_name]
                assert torch.allclose(baseline_weight, resumed_weight), (
                    f"Block {block_idx}, layer {layer_name} weight mismatch:\n"
                    f"Baseline: {baseline_weight.flatten()[:5]}\n"
                    f"Resumed:  {resumed_weight.flatten()[:5]}\n"
                    f"Max diff: {torch.max(torch.abs(baseline_weight - resumed_weight))}"
                )

    # (enable_caching_after_block, blocks_completed_before_interrupt)
    #  - (2, 1): resume point (block 1) is in the non-cached region and <= disable_caching, so it exercises the
    #            eager-restore + sampler start_block skip path.
    #  - (1, 2): resume point (block 2) is in the cached region, so it exercises the path where the eagerly-restored
    #            prefix is chained through internally by the sampler and only blocks from the resume point are yielded.
    @pytest.mark.parametrize(
        "enable_caching_after_block, blocks_completed", [(2, 1), (1, 2)]
    )
    def test_resume_reproducibility_with_disabled_caching(
        self,
        enable_caching_after_block,
        blocks_completed,
        fxt_dummy_input,
        fxt_dataloader,
        fxt_checkpoint_dir,
    ):
        """
        Resume must produce identical folded weights to an uninterrupted run for a hybrid model that has both a
        non-cached prefix (``enable_caching_after_block > 0``) and a cached tail. This pins correctness of both the
        new non-cached start_block skip and the cached-region re-propagation fallback.
        """

        def _set_seed_hook(func):
            def f(*args, **kwargs):
                set_seed(432)
                return func(*args, **kwargs)

            return f

        # Llama with disabled caching for the first `enable_caching_after_block` blocks.
        config = LlamaConfig(**MODEL_CONFIGS["llama"]["config_kwargs"])

        with patch.dict(
            adascale_model_config_dict,
            {
                LlamaModel: AdaScaleModelConfig(
                    block_type=adascale_model_config_dict[LlamaModel].block_type,
                    enable_caching_after_block=enable_caching_after_block,
                )
            },
        ):
            # ----- Baseline: uninterrupted -----
            set_seed(432)
            model1 = LlamaForCausalLM(config)
            model2 = deepcopy(model1)
            model1.eval()
            qsim1 = get_quantsim_ready_model(model1, fxt_dummy_input)
            with patch.object(
                AdaScale,
                "adascale_block",
                wraps=_set_seed_hook(AdaScale.adascale_block),
            ):
                AdaScale.apply_adascale(
                    qsim1,
                    fxt_dataloader,
                    num_iterations=NUM_ITERATIONS,
                    checkpoint_dir=None,
                )

            baseline_weights = {}
            for block_idx, block in enumerate(AdaScale._get_blocks(qsim1)):
                baseline_weights[block_idx] = {
                    name: module.weight.data.clone().cpu()
                    for name, module in block.named_modules()
                    if isinstance(module, QuantizedLinear)
                    and isinstance(
                        module.param_quantizers["weight"], QuantizeDequantize
                    )
                }

            # ----- Interrupt after `blocks_completed` blocks, then resume -----
            set_seed(432)
            model2.eval()
            qsim2 = get_quantsim_ready_model(model2, fxt_dummy_input)

            def _seed_then_run(func):
                wrapped = _set_seed_hook(func)

                def f(*args, **kwargs):
                    if f._count >= blocks_completed:
                        raise RuntimeError("cancel")
                    wrapped(*args, **kwargs)
                    f._count += 1

                f._count = 0
                return f

            with patch.object(
                AdaScale,
                "adascale_block",
                side_effect=_seed_then_run(AdaScale.adascale_block),
            ):
                with pytest.raises(RuntimeError, match="cancel"):
                    AdaScale.apply_adascale(
                        qsim2,
                        fxt_dataloader,
                        num_iterations=NUM_ITERATIONS,
                        checkpoint_dir=fxt_checkpoint_dir,
                    )

            manager = PerBlockCheckpointManager(fxt_checkpoint_dir)
            assert manager.get_progress()["completed_blocks"] == list(
                range(blocks_completed)
            )

            with patch.object(
                AdaScale,
                "adascale_block",
                wraps=_set_seed_hook(AdaScale.adascale_block),
            ):
                AdaScale.apply_adascale(
                    qsim2,
                    fxt_dataloader,
                    num_iterations=NUM_ITERATIONS,
                    checkpoint_dir=fxt_checkpoint_dir,
                )

            resumed_weights = {}
            for block_idx, block in enumerate(AdaScale._get_blocks(qsim2)):
                resumed_weights[block_idx] = {
                    name: module.weight.data.clone().cpu()
                    for name, module in block.named_modules()
                    if isinstance(module, QuantizedLinear)
                    and isinstance(
                        module.param_quantizers["weight"], QuantizeDequantize
                    )
                }

        for block_idx in range(NUM_HIDDEN_LAYERS):
            for layer_name, baseline_weight in baseline_weights[block_idx].items():
                resumed_weight = resumed_weights[block_idx][layer_name]
                assert torch.allclose(baseline_weight, resumed_weight), (
                    f"Block {block_idx}, layer {layer_name} weight mismatch after resume "
                    f"(enable_caching_after_block={enable_caching_after_block}, "
                    f"blocks_completed={blocks_completed}); "
                    f"max diff: {torch.max(torch.abs(baseline_weight - resumed_weight))}"
                )
