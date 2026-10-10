# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""PatchMerger output projection (linear_fc2) detection for VLM visual encoders."""

from typing import List

import onnx_ir

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.llm_topology.ir_analysis import is_weighted_linear

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def find_merger_linear2(ir_model: onnx_ir.Model) -> List[onnx_ir.Node]:
    """Find PatchMerger linear_fc2 nodes in a visual encoder ONNX graph.

    Identifies all weighted linear nodes that are leaves of the weighted-linear
    subgraph — i.e. have no downstream weighted linear consumers. These are the
    PatchMerger linear_fc2 layers that write into the language backbone residual
    stream and must always be rotated with R_L when the backbone is SpinQuant-rotated.

    NOTE:
        Assumes the PatchMerger linear_fc2 is the topological leaf of the weighted-linear
        subgraph — i.e. no downstream weighted linear follows it. This holds for Qwen2.5-VL
        and Qwen3-VL. Unknown architectures will be misdetected; an
        explicit override will be added as part of the general block-detection fallback.

    :param ir_model: IR model of visual.onnx.
    :return: List of merger_linear2 nodes in topological order.
    :raises ValueError: If no merger_linear2 nodes are found.
    """
    weighted_linears = [node for node in ir_model.graph if is_weighted_linear(node)]
    weighted_linear_set = set(weighted_linears)

    def _has_downstream_weighted_linear(node: onnx_ir.Node) -> bool:
        visited = set()
        stack = list(node.successors())
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in weighted_linear_set:
                return True
            stack.extend(current.successors())
        return False

    result = [
        node for node in weighted_linears if not _has_downstream_weighted_linear(node)
    ]

    if not result:
        raise ValueError(
            "merger_linear2 not detected: no leaf weighted linear op found in the ViT graph."
        )

    _logger.info(
        "Visual: merger_linear2=%s will be rotated with R_L.",
        [node.name for node in result],
    )
    return result
