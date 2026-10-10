# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic in-place tensor transforms used by SpinQuant rotation passes.

These helpers are agnostic of which rotation (R1/R2/R3) is being applied — they
take an arbitrary ``R`` matrix and the storage metadata for an op and rotate
the corresponding initializer in-place. Pass-specific logic (which ops to
rotate, what hidden dim to use, validation) lives in
:mod:`aimet_onnx.spinquant.passes`.
"""

from aimet_onnx.spinquant.transforms.norm_fusion import (
    fuse_norm_layers_into_linears,
)
from aimet_onnx.spinquant.transforms.rotation_primitives import (
    apply_transform,
    block_diag_repeat,
    hadamard_rotation_matrix,
    insert_online_hadamard_node,
    left_multiply,
    right_multiply,
    rotate_gather_weight,
    rotate_linear_weight,
    is_online_rotation_op,
)

__all__ = [
    "apply_transform",
    "block_diag_repeat",
    "fuse_norm_layers_into_linears",
    "hadamard_rotation_matrix",
    "insert_online_hadamard_node",
    "left_multiply",
    "right_multiply",
    "rotate_gather_weight",
    "rotate_linear_weight",
]
