# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import math
import itertools
import torch
import torch.nn.functional as F
import scipy

from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.experimental.transforms.transform_ops import (
    InvertibleTransformOp,
    MatrixTransformOp,
)


# pylint: disable=abstract-method
class ScaledRotateTransformOp(InvertibleTransformOp):
    def __init__(self, head_dim, num_attention_heads, num_key_value_heads):
        super().__init__(mergeable=True)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        k_proj_out_features = head_dim * num_key_value_heads

        self.rotation = torch.nn.Parameter(
            torch.randn(k_proj_out_features // 2), requires_grad=True
        )
        self.scale = torch.nn.Parameter(
            torch.randn((num_key_value_heads, head_dim // 2)), requires_grad=True
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + ScaledRotateTransformOp.rotate_half(x) * sin

    def get_rotation(self) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cos(self.rotation).view(self.scale.shape).repeat(1, 2)
        sin = torch.sin(self.rotation).view(self.scale.shape).repeat(1, 2)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated_scale = self.scale.repeat(1, 2)
        scaled_x = x.view(*x.shape[:-1], *repeated_scale.shape) * repeated_scale
        cos, sin = self.get_rotation()
        return self.rotate(scaled_x, cos, sin).view(x.shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        repeated_scale = self.scale.repeat_interleave(
            num_key_value_groups, dim=0
        ).repeat(1, 2)
        scaled_x = x.view(*x.shape[:-1], self.num_attention_heads, -1) / repeated_scale
        cos, sin = self.get_rotation()
        cos = cos.repeat_interleave(num_key_value_groups, dim=0)
        sin = sin.repeat_interleave(num_key_value_groups, dim=0)
        return self.rotate(scaled_x, cos, sin).view(x.shape)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight.T).T


class ScalingTransformOp(InvertibleTransformOp):
    def __init__(self, intermediate_size):
        super().__init__(True)
        self.scale = torch.nn.Parameter(
            torch.randn(intermediate_size), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight.T).T

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight)


class GroupedHadamardTransformOp(InvertibleTransformOp):
    def __init__(self, intermediate_size):
        super().__init__(True)
        num_two_factors = 0
        remaining_factor = intermediate_size
        while remaining_factor & 1 == 0:
            remaining_factor = remaining_factor >> 1
            num_two_factors += 1
        self.register_buffer(
            "hadamard",
            torch.tensor(
                scipy.linalg.hadamard(2**num_two_factors), dtype=torch.float32
            ),
        )
        self.group_size = 2**num_two_factors
        self.n_groups = remaining_factor
        self.scale = 1 / math.sqrt(2**num_two_factors)
        self.mergeable = False

    def forward(self, x):
        x_reshape = x.reshape(*x.shape[:-1], self.n_groups, self.group_size)
        return (F.linear(x_reshape, self.hadamard.to(x.device)) * self.scale).reshape(
            x.shape
        )

    def inverse(self, x):
        x_reshape = x.reshape(*x.shape[:-1], self.n_groups, self.group_size)
        return (F.linear(x_reshape, self.hadamard.to(x.device)) * self.scale).reshape(
            x.shape
        )

    def get_inverted_op(self):
        inverted_op = super().get_inverted_op()
        inverted_op.mergeable = True
        return inverted_op

    def left_hand_merge(self, weight):
        return self.forward(weight)


@QuantizationMixin.implements(GroupedHadamardTransformOp)
class QuantizedGroupedHadamardTransformOp(
    QuantizationMixin, GroupedHadamardTransformOp
):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):  # pylint: disable=arguments-differ
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


class MultiHeadValueTransformOp(InvertibleTransformOp):
    def __init__(self, head_dim, num_attention_heads, num_key_value_heads):
        super().__init__(mergeable=True)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.matrix_per_head = torch.nn.ModuleList(
            [
                MatrixTransformOp(torch.eye(head_dim)),
            ]
            * self.num_key_value_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(x, self.num_key_value_heads, dim=-1)
        results = [
            transform(slice) for transform, slice in zip(self.matrix_per_head, slices)
        ]
        return torch.cat(results, dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(x, self.num_attention_heads, dim=-1)
        results = [
            transform.inverse(slice)
            for transform, slice in zip(itertools.cycle(self.matrix_per_head), slices)
        ]
        return torch.cat(results, dim=-1)

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.data.dtype
        return (
            self.forward(
                torch.eye(
                    self.head_dim * self.num_attention_heads,
                    device=weight.data.device,
                    dtype=torch.float32,
                )
            )
            @ weight.T.to(dtype=torch.float32)
        ).T.to(dtype=orig_dtype)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.data.dtype
        return self.forward(weight.T.to(dtype=torch.float32)).T.to(orig_dtype)


class RotationTransformOp(InvertibleTransformOp):
    def __init__(self, matrix: torch.Tensor):
        super().__init__(mergeable=True)
        linear = torch.nn.Linear(*matrix.shape, bias=False)
        linear.weight.data = matrix
        self.rotation = torch.nn.utils.parametrizations.orthogonal(
            linear, orthogonal_map="cayley", use_trivialization=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return (x.to(dtype=torch.float32) @ self.rotation.weight.data).to(
            dtype=orig_dtype
        )

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return (x.to(dtype=torch.float32) @ self.rotation.weight.data.T).to(
            dtype=orig_dtype
        )

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.dtype
        return (
            self.forward(
                torch.eye(
                    self.rotation.weight.data.shape[-1],
                    device=weight.data.device,
                    dtype=torch.float32,
                )
            )
            @ weight.T.to(dtype=torch.float32)
        ).T.to(dtype=orig_dtype)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.dtype
        return self.forward(weight.T.to(dtype=torch.float32)).T.to(dtype=orig_dtype)
