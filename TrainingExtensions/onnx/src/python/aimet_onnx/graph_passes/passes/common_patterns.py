# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from aimet_common.connected_graph.operation import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.graph_passes.utils import (
    check_consecutive_ops,
    match_pow_2_pattern,
    match_a_div_b_pattern,
)


def match_rms_norm_pattern(op: Op, model: ModelProto):
    """Common pattern for RMSNormalization which can be re-used"""
    # Match Mul(x, x) or Pow(x, 2)
    match = match_pow_2_pattern(op, model)
    if not match or len(op.output_ops) != 1:
        return []

    # Sqrt(E(Pow(x, 2)) + ε)
    match, denominator_ops = check_consecutive_ops(
        op.output_ops[0],
        ["ReduceMean", "Add", "Sqrt"],
        validate_last_op_consumers=False,
    )
    if not match:
        return []

    all_ops = [op] + denominator_ops
    sqrt_op = all_ops[-1]

    if len(sqrt_op.output_ops) != 1:
        return []

    # Div pattern: Div(x, Sqrt(E(Pow(x, 2)) + ε))
    div_ops = match_a_div_b_pattern(op.inputs[0], sqrt_op.outputs[0], model)
    if not div_ops:
        return []

    all_ops = all_ops + div_ops
    return all_ops
