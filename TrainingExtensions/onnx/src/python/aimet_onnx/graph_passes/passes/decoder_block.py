# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

from aimet_common.connected_graph.operation import Op
from aimet_onnx.graph_passes.graph_pass import GraphPass
from aimet_onnx.graph_passes.passes.common_patterns import match_rms_norm_pattern
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.graph_passes.pass_registry import register_pass
from aimet_onnx.utils import ModelProto

from typing import Dict, List, Tuple


@register_pass("DecoderBlock")
class DecoderBlock(GraphPass):
    """
    Finds end points of Decoder blocks
    """

    def __init__(self):
        self.decoder_blocks: List[Tuple[str, str]] = []
        self.pattern_last_op: Op = None
        self.block_start_op = None
        self.intermediate_op = None

    def apply_on_op(self, op: Op, model: ModelProto, _: Dict[str, QcQuantizeOp]):
        if self.match_pattern(op, model):
            if not self.block_start_op:
                self.block_start_op = self.pattern_last_op
            elif not self.intermediate_op:
                self.intermediate_op = self.pattern_last_op
            else:
                self.decoder_blocks.append((self.block_start_op, self.pattern_last_op))
                self.block_start_op = self.pattern_last_op
                self.intermediate_op = None
            self.pattern_last_op = None

    # pylint: disable=too-many-branches, too-many-return-statements
    def match_pattern(self, op: Op, model: ModelProto):
        """
        Match RMSNorm pattern and collect ops to disable output quantizers
        """
        all_ops = match_rms_norm_pattern(op, model)
        if not all_ops:
            return False

        # Check if weights are present
        self.pattern_last_op = all_ops[0]

        return True
