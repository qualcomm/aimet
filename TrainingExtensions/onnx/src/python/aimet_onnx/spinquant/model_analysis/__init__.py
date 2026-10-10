# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Rotation-specific model analysis for SpinQuant.

Holds the analysis each rotation pass needs beyond the technique-agnostic LLM
topology: R3 attention anchors (the ``onnx_ir`` insertion edges derived from the
KV cache), the VLM visual merger, and the R1 post-writing-norm precondition
check. Decoder block detection, role mapping, and fine-grained intra-block
structure (q/k/v/o, gate/up/down, dynamic MatMuls) live in ``llm_topology``;
import those directly from there. R2 reads V/O directly off the topology
(``block.v_proj`` / ``block.o_proj``) and needs no analysis here.
"""

from aimet_onnx.spinquant.model_analysis.attention_anchors import (
    BlockR3Anchors,
    find_r3_anchors,
)
from aimet_onnx.spinquant.model_analysis.norm_detection import (
    find_post_writing_norms,
)
from aimet_onnx.spinquant.model_analysis.visual_merger import (
    find_merger_linear2,
)

__all__ = [
    "BlockR3Anchors",
    "find_r3_anchors",
    "find_merger_linear2",
    "find_post_writing_norms",
]
