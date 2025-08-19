# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from tqdm import tqdm
import itertools
import torch

from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin


class LocalTransformOptimizer:
    p: float = 4.0
    num_iterations: int = 200
    lr: float = 1e-2

    @staticmethod
    def compute_loss(weight: torch.Tensor) -> torch.Tensor:
        return torch.mean((weight.abs() ** LocalTransformOptimizer.p)) ** (
            1 / LocalTransformOptimizer.p
        )

    def __init__(self, transformed_layers: list[TransformationMixin]):
        self.layers = transformed_layers
        self.parameters = []
        for layer in self.layers:
            for transform in itertools.chain(
                layer.right_hand_transforms, layer.left_hand_transforms
            ):
                if transform.mergeable:
                    self.parameters.extend(list(transform.parameters()))
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr)

    # pylint: disable=protected-access
    def optimize(self):
        for _ in tqdm(range(self.num_iterations), desc="Locally optimizing transforms"):
            self.optimizer.zero_grad()
            loss = torch.stack(
                tuple(
                    self.compute_loss(layer._compute_merged_params()[0])
                    for layer in self.layers
                )
            ).sum(dim=0)
            loss.backward()
            self.optimizer.step()
