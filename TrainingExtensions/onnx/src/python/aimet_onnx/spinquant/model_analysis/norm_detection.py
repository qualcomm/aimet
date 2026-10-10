# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R1-specific affine RMSNorm checks for SpinQuant.

Generic active-RMSNorm detection lives in ``llm_topology.norm_detection``.
This module holds only the SpinQuant R1 precondition check that reuses that
detection to find affine RMSNorms sitting between a writing layer and the
residual add.
"""

from typing import Iterable, List

import onnx_ir

from aimet_onnx.llm_topology import ir_analysis
from aimet_onnx.llm_topology.norm_detection import is_affine_rms_norm


def find_post_writing_norms(
    analysis_ir: onnx_ir.Model, writing_output_tensors: Iterable[str]
) -> List[str]:
    """Return names of affine RMSNorms immediately after writing layers.

    Used by R1 architecture compatibility checks: R1 absorption requires writing
    layers (o_proj, down_proj) to feed directly into the residual add, with no
    affine RMSNorm in between.

    :param analysis_ir: Analysis IR from :func:`~.ir_analysis.build_analysis_ir`.
        It must be that view rather than a faithful graph: a decomposed RMSNorm
        is only recognizable as one node once the supergroup fusion has run.
    :param writing_output_tensors: Output tensor names of the writing layers to
        check (a block's o_proj and down_proj).
    :return: List of norm names for detected post-writing norms.
    """
    node_by_output = ir_analysis.node_by_output_tensor(analysis_ir)

    found = []
    for tensor_name in writing_output_tensors:
        writing_node = node_by_output.get(tensor_name)
        if writing_node is None:
            continue
        for consumer in writing_node.successors():
            # A dtype hop between the writing layer and the norm is transparent.
            candidates = (
                consumer.successors() if consumer.op_type == "Cast" else [consumer]
            )
            for candidate in candidates:
                if is_affine_rms_norm(candidate):
                    found.append(ir_analysis.node_name(candidate))
                    break
    return found
