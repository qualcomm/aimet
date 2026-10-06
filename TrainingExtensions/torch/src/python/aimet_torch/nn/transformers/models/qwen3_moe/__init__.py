# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from .modeling_qwen3_moe import QuantizedQwen3MoeRMSNorm

try:
    from .modeling_qwen3_moe import QuantizedQwen3MoeTopKRouter
except ImportError:
    # transformers<5.0.0 uses an nn.Linear gate instead of Qwen3MoeTopKRouter
    pass
