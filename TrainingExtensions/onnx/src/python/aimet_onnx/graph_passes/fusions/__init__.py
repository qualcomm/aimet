# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graph fusion passes"""

from .fusion import fuse_supergroups
from .fusion_registry import AIMET_SUPERGROUP_DOMAIN
from .ir_utils import inline_all_supergroups, is_fused_supergroup
from .layernorm import LayerNormFusion
from .matmul_add import MatmulAddFusion
from .rmsnorm import RMSNormFusion
from .masked_softmax import MaskedSoftmax
