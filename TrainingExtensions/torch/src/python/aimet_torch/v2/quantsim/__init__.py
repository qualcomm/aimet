# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.quantsim

All contents have been moved to aimet_torch.quantsim. This module re-exports everything
from the new location for backwards compatibility.
"""

from ...quantsim import (
    QuantizationSimModel,
    QuantizationSimModelOnnxExporter,
    QuantParams,
    load_encodings_to_sim,
)
