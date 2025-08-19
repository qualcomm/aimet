# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import abc
import functools

import torch


def is_mergeable_transform(module: torch.nn.Module) -> bool:
    return isinstance(module, TransformOp) and module.mergeable


class TransformOp(torch.nn.Module):
    def __init__(self, mergeable: bool = True):
        super().__init__()
        self.mergeable = mergeable

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class InvertibleTransformOp(TransformOp):
    @abc.abstractmethod
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @functools.lru_cache(maxsize=1)
    def _get_inverse_op_type(cls):
        attr_dict = cls.__dict__.copy()
        attr_dict["inverse"], attr_dict["forward"] = (
            attr_dict["forward"],
            attr_dict["inverse"],
        )

        return type(f"Inverse{cls.__name__}", (cls,), attr_dict)

    @functools.lru_cache(maxsize=1)
    def get_inverted_op(self) -> TransformOp:
        inverted_op = super().__new__(self._get_inverse_op_type())  # pylint: disable=no-value-for-parameter
        inverted_op.__dict__.update(self.__dict__)
        inverted_op._modules = self._modules.copy()  # pylint: disable=protected-access
        inverted_op._parameters = self._parameters.copy()  # pylint: disable=protected-access
        inverted_op._buffers = self._buffers.copy()  # pylint: disable=protected-access
        return inverted_op


class IdentityTransformOp(InvertibleTransformOp):
    def __init__(self):
        super().__init__(mergeable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return weight


class MatrixTransformOp(InvertibleTransformOp):
    def __init__(self, matrix):
        super().__init__(mergeable=True)
        self.matrix = torch.nn.Parameter(matrix, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.linalg.inv(self.matrix)

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return (
            self.forward(
                torch.eye(self.matrix.data.shape[-1], device=weight.data.device)
            )
            @ weight.T
        ).T

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight.T).T
