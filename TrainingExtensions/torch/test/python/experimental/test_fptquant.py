# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Test aimet-torch fptquant"""

import pytest
import torch
from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin
from aimet_torch.experimental.fptquant.fptquant_transforms import (
    GroupedHadamardTransformOp,
)
from aimet_torch.experimental.fptquant import fptquant_config
from aimet_torch.experimental.fptquant.fptquant_optimizer import FPTQuant
from aimet_torch.quantsim import QuantizationSimModel


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("size", [24, 64])
def test_grouped_hadamard_transform(size):
    linear = torch.nn.Linear(size, size, bias=False)
    transformed_linear = TransformationMixin.from_module(linear)
    transform = GroupedHadamardTransformOp(size)

    # Adding mergeable transform first since transforms are added as a stack
    transformed_linear.add_left_hand_transform(transform.get_inverted_op())
    transformed_linear.add_left_hand_transform(transform)

    assert len(transformed_linear.left_hand_transforms) == 2

    dummy_input = torch.randn(size, size)
    transformed_out = transformed_linear(dummy_input)
    orig_out = linear(dummy_input)
    assert torch.allclose(transformed_out, orig_out, atol=1e-6)

    transformed_linear.merge()
    assert len(transformed_linear.left_hand_transforms) == 1
    new_out = transformed_linear(dummy_input)
    assert torch.allclose(transformed_out, new_out, atol=1e-6)


@pytest.mark.skip("Enable this when quantized GroupedHadamardOp is supported")
@pytest.mark.parametrize("size", [24, 64])
def test_quantized_grouped_hadamard_transform(size):
    model = LinearModel(size)
    model.linear = TransformationMixin.from_module(model.linear)
    transform = GroupedHadamardTransformOp(size)

    # Adding mergeable transform first since transforms are added as a stack
    model.linear.add_left_hand_transform(transform.get_inverted_op())
    model.linear.add_left_hand_transform(transform)
    import pdb

    pdb.set_trace()

    dummy_input = torch.randn(size, size)
    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m: m(dummy_input))

    before_merge_out = qsim.model(dummy_input)

    assert (
        qsim.model.linear.left_hand_transforms[0].output_quantizers[0].get_min()
        is not None
    )
    assert qsim.model.linear.left_hand_transforms[1].output_quantizers[0] is None

    qsim.model.linear.merge()

    assert (
        qsim.model.linear.left_hand_transforms[0].output_quantizers[0].get_min()
        is not None
    )
    assert len(qsim.model.linear.left_hand_transforms) == 1

    after_merge_out = qsim.model(dummy_input)
    assert torch.allclose(before_merge_out, after_merge_out, atol=1e-6)


def test_insert_nonmergeable_down_project():
    class Block(torch.nn.Module):
        def __init__(self):
            super(Block, self).__init__()
            self.d_proj = torch.nn.Linear(24, 24, bias=False)

        def forward(self, x):
            return self.d_proj(x)

    class ModelWithBlocks(torch.nn.Module):
        def __init__(self):
            super(ModelWithBlocks, self).__init__()
            self.blocks = torch.nn.ModuleList()
            for _ in range(4):
                self.blocks.append(Block())

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    class MyBlockInterface(fptquant_config.BlockInterface):
        def __init__(self, block):
            self.block = block

        @property
        def down_proj(self):
            return self.block.d_proj

        @down_proj.setter
        def down_proj(self, value):
            self.block.d_proj = value

    class DummyConfig:
        def __init__(self):
            self.intermediate_size = 24

    model = ModelWithBlocks()
    fptquant_config.fptquant_model_config_dict[ModelWithBlocks] = (
        fptquant_config.FPTQuantConfig(Block, MyBlockInterface)
    )
    FPTQuant.insert_nonmergeable_down_project_transform(model, DummyConfig())

    num_transformed_linears = 0
    for module in model.modules():
        if isinstance(module, TransformationMixin):
            num_transformed_linears += 1
            assert isinstance(
                module.left_hand_transforms[0], GroupedHadamardTransformOp
            )
            assert not module.left_hand_transforms[0].mergeable
    assert num_transformed_linears == 4
