# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Hadamard utilities for SpinQuant - PyTorch wrapper"""

from aimet_torch.common.hadamard import get_hadamard_matrix as get_hadamard_matrix_numpy
import torch


def get_hadamard_matrix(size: int) -> torch.Tensor:
    """
    Get hadamard matrix with dimensions size x size as a PyTorch tensor.

    Hadamard matrices with size of powers of two are obtained via scipy.linalg.hadamard.
    For sizes of non powers of two, only sizes which can be decomposed into factor * 2^n for any n>=0 are supported,
    where factor is one of {12, 20, 28}.
    Such hadamard matrices are constructed by iteratively taking the Kronecker product of Hadamard size 2 ([1, 1], [1, -1]) starting with Hadamard size 'factor',
    doubling the size of the matrix each iteration until the matrix achieves size 'size'.

    :param size: Size of hadamard matrix to get
    :return: Hadamard matrix as PyTorch tensor with dtype float32
    """
    # Get the numpy array from the common implementation
    hadamard_numpy = get_hadamard_matrix_numpy(size)

    # Convert to PyTorch tensor
    return torch.from_numpy(hadamard_numpy)
