# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Test aimet-torch transforms"""

import pytest
import torch

from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch.experimental.transforms.transformed_layers import (
    TransformationMixin,
    TransformedLinear,
    QuantizedTransformedLinear,
)
from aimet_torch.experimental.transforms.transform_ops import (
    IdentityTransformOp,
    MatrixTransformOp,
)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def get_square_invertible_matrix(size):
    # Generate random orthogonal matrices U and V
    U = torch.linalg.qr(torch.randn(size, size))[0]
    V = torch.linalg.qr(torch.randn(size, size))[0]

    # Create a diagonal matrix with positive singular values
    singular_values = torch.rand(size) + 0.1  # Ensure non-zero
    S = torch.diag(singular_values)

    # Construct the invertible matrix
    matrix = U @ S @ V.T
    return matrix


def test_transformed_layer_conversion():
    # convert Linear to TransformedLinear
    # ensure that result computation is the same before, and with original module
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    assert isinstance(model.linear1, TransformedLinear)
    assert isinstance(model.linear2, TransformedLinear)
    assert isinstance(model.linear3, TransformedLinear)

    transformed_result = model(dummy_input)
    assert torch.allclose(orig_result, transformed_result)

    model.linear1 = model.linear1.get_original_module()
    model.linear2 = model.linear2.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    assert not isinstance(model.linear1, TransformedLinear)
    assert not isinstance(model.linear2, TransformedLinear)
    assert not isinstance(model.linear3, TransformedLinear)

    merged_result = model(dummy_input)
    assert torch.allclose(orig_result, merged_result)


def test_transformed_quantized_layer_conversion():
    # same as previous test but create quantsim first
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    quantized_model = QuantizationSimModel(model, dummy_input, in_place=True)
    remove_all_quantizers(quantized_model.model)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    assert isinstance(model.linear1, QuantizedTransformedLinear)
    assert isinstance(model.linear2, QuantizedTransformedLinear)
    assert isinstance(model.linear3, QuantizedTransformedLinear)

    transformed_result = model(dummy_input)
    assert torch.allclose(orig_result, transformed_result)

    model.linear1 = model.linear1.get_original_module()
    model.linear2 = model.linear2.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    assert not isinstance(model.linear1, QuantizedTransformedLinear)
    assert not isinstance(model.linear2, QuantizedTransformedLinear)
    assert not isinstance(model.linear3, QuantizedTransformedLinear)

    merged_result = model(dummy_input)
    assert torch.allclose(orig_result, merged_result)


@pytest.mark.skip("Issue with instantiating QuantSim on pre-transformed layers.")
def test_transformed_quantized_layer_conversion_2():
    model = LinearModel()

    dummy_input = torch.randn((10, 10))

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    quantized_model = QuantizationSimModel(model, dummy_input, in_place=True)

    assert isinstance(model.linear1, QuantizedTransformedLinear)
    assert isinstance(model.linear2, QuantizedTransformedLinear)
    assert isinstance(model.linear3, QuantizedTransformedLinear)


def test_transformed_layer_merge():
    # Ensure that results are the same with Indentity Transform added
    # before and after merge
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    model.linear1.add_right_hand_transform(IdentityTransformOp())
    model.linear2.add_right_hand_transform(IdentityTransformOp())
    model.linear3.add_right_hand_transform(IdentityTransformOp())

    transformed_result = model(dummy_input)
    assert torch.allclose(orig_result, transformed_result)

    model.linear1.merge()
    model.linear2.merge()
    model.linear3.merge()

    model.linear1 = model.linear1.get_original_module()
    model.linear2 = model.linear2.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    merged_result = model(dummy_input)
    assert torch.allclose(orig_result, merged_result)


@pytest.mark.skip("Skipping test for now")
def test_op_inverse():
    # Create a matrix transform op
    # Get its inverse
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    transform = MatrixTransformOp(torch.randn((10, 10)))
    model.linear1.add_right_hand_transform(transform)

    intermediate_result = model(dummy_input)
    assert not torch.allclose(intermediate_result, orig_result)

    model.linear2.add_left_hand_transform(transform.get_inverted_op())
    transformed_result = model(dummy_input)
    assert torch.allclose(transformed_result, orig_result, atol=1e-5)

    model.linear1.merge()
    model.linear2.merge()
    model.linear3.merge()

    model.linear1 = model.linear1.get_original_module()
    model.linear2 = model.linear2.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    merged_result = model(dummy_input)
    assert torch.allclose(merged_result, transformed_result, atol=1e-5)


def test_op_inverse_2():
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    transform = MatrixTransformOp(torch.randn((10, 10)))
    model.linear2.add_left_hand_transform(transform)
    model.linear2.left_hand_transforms[0].mergeable = False

    intermediate_result = model(dummy_input)
    assert not torch.allclose(intermediate_result, orig_result)

    model.linear1.add_right_hand_transform(transform.get_inverted_op())
    transformed_result = model(dummy_input)
    assert torch.allclose(transformed_result, orig_result, atol=1e-5)

    model.linear1.merge()
    model.linear2.merge()
    model.linear3.merge()

    merged_result = model(dummy_input)
    assert torch.allclose(merged_result, transformed_result, atol=1e-5)


@pytest.mark.skip("Skipping test for now")
def test_multiple_transforms():
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    transform = MatrixTransformOp(get_square_invertible_matrix(10))
    model.linear1.add_right_hand_transform(transform)
    model.linear2.add_left_hand_transform(transform.get_inverted_op())

    transform2 = MatrixTransformOp(get_square_invertible_matrix(10))
    model.linear1.add_right_hand_transform(transform2)
    model.linear2.add_left_hand_transform(transform2.get_inverted_op())

    transform3 = MatrixTransformOp(get_square_invertible_matrix(10))
    model.linear2.add_right_hand_transform(transform3)
    model.linear3.add_left_hand_transform(transform3.get_inverted_op())

    transformed_result = model(dummy_input)
    assert torch.allclose(transformed_result, orig_result, atol=1e-5)

    model.linear1.merge()
    model.linear2.merge()
    model.linear3.merge()

    model.linear1 = model.linear1.get_original_module()
    model.linear2 = model.linear2.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    merged_result = model(dummy_input)
    assert torch.allclose(merged_result, orig_result, atol=1e-5)


def test_non_mergeable_transforms():
    # Create a matrix transform op
    # Get its inverse
    model = LinearModel()

    dummy_input = torch.randn((10, 10))
    orig_result = model(dummy_input)

    model.linear1 = TransformationMixin.from_module(model.linear1)
    model.linear2 = TransformationMixin.from_module(model.linear2)
    model.linear3 = TransformationMixin.from_module(model.linear3)

    transform = MatrixTransformOp(get_square_invertible_matrix(10))
    model.linear1.add_right_hand_transform(transform)
    model.linear2.add_left_hand_transform(transform.get_inverted_op())

    model.linear2.left_hand_transforms[0].mergeable = False

    transformed_result = model(dummy_input)
    assert torch.allclose(transformed_result, orig_result, atol=1e-5)

    model.linear1.merge()
    model.linear2.merge()
    model.linear3.merge()

    model.linear1 = model.linear1.get_original_module()
    model.linear3 = model.linear3.get_original_module()

    assert len(model.linear2.left_hand_transforms) > 0

    merged_result = model(dummy_input)
    assert torch.allclose(merged_result, orig_result, atol=1e-5)
