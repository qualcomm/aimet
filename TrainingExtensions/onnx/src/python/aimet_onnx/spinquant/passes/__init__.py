# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""SpinQuant rotation passes.

Each pass is a self-contained class that owns:

- which rotation matrix to construct (and at what dimension);
- which ops to rotate, with which storage / role;
- any pre-conditions to validate before mutating the model;
- any setup steps (e.g. norm fusion) that the rotation requires.

Add new rotations (R2, R3, ...) by subclassing :class:`RotationPass` in a new
module under this package and exporting them here.
"""

from aimet_onnx.spinquant.passes.base import (
    RotationPass,
    SpinquantContext,
)
from aimet_onnx.spinquant.passes.r1 import R1RotationPass
from aimet_onnx.spinquant.passes.r2 import R2RotationPass
from aimet_onnx.spinquant.passes.r3 import (
    R3RotationPass,
)

__all__ = [
    "R1RotationPass",
    "R2RotationPass",
    "R3RotationPass",
    "RotationPass",
    "SpinquantContext",
]
