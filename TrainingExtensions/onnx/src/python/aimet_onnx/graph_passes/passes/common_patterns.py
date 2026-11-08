# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from typing import List, Optional, Tuple
import numpy as np
from aimet_onnx.common.connected_graph.operation import Op, Product
from aimet_onnx.utils import ModelProto, get_node_attribute

from aimet_onnx.graph_passes.utils import (
    _get_numpy_array,
    check_consecutive_ops,
    get_shape,
    is_constant_scalar,
    is_unidirectionally_broadcastable,
    match_pow_2_pattern,
    match_a_div_b_pattern,
)


def match_rms_norm_pattern(op: Op, model: ModelProto) -> Tuple[List[Op], bool]:
    """
    Common pattern for RMSNormalization which can be re-used

    Returns:
        Tuple[List[Op], bool]: The RMSNorm ops, followed by any Casts and the scale Mul
        if a scale Mul is matched (empty if no match), and whether the RMSNorm is
        elementwise affine, i.e. whether a scale Mul is matched
    """
    norm_ops = _match_rms_norm_without_scale(op, model)
    if not norm_ops:
        return [], False

    # Skip any Cast ops (e.g. from dtype casting in RMSNorm implementations like
    # Qwen2RMSNorm which casts to float32 for numerical stability then back).
    last_op = norm_ops[-1]
    cast_ops = []
    while len(last_op.output_ops) == 1 and last_op.output_ops[0].type == "Cast":
        cast_ops.append(last_op.output_ops[0])
        last_op = last_op.output_ops[0]
    if len(last_op.output_ops) != 1:
        return norm_ops, False

    consumer_op = last_op.output_ops[0]
    if _is_rms_norm_scale_mul(norm_ops, last_op.outputs[0], consumer_op, model):
        return norm_ops + cast_ops + [consumer_op], True
    return norm_ops, False


def _match_rms_norm_without_scale(op: Op, model: ModelProto) -> List[Op]:
    """
    Match a fused RMSNormalization, or the decomposed RMSNorm ops up to the normalized
    output, without the scale Mul
    """
    if op.type == "RMSNormalization":
        return [op]

    # Match Mul(x, x) or Pow(x, 2)
    match = match_pow_2_pattern(op, model)
    if not match or len(op.output_ops) != 1:
        return []

    # E(Pow(x, 2)) + ε
    match, denominator_ops = check_consecutive_ops(
        op.output_ops[0],
        ["ReduceMean", "Add"],
        validate_last_op_consumers=False,
    )
    if not match:
        return []

    all_ops = [op] + denominator_ops
    add_op = all_ops[-1]

    if len(add_op.output_ops) != 1:
        return []

    next_op = add_op.output_ops[0]

    # Sqrt form: Div(x, Sqrt(E(Pow(x, 2)) + ε))
    if next_op.type == "Sqrt" and len(next_op.output_ops) == 1:
        div_ops = match_a_div_b_pattern(op.inputs[0], next_op.outputs[0], model)
        if div_ops:
            return all_ops + [next_op] + div_ops

    # rsqrt form: Mul(x, Pow(E(Pow(x, 2)) + ε, -0.5))
    if (
        next_op.type == "Pow"
        and is_constant_scalar(model, next_op.inputs[1], -0.5)
        and len(next_op.output_ops) == 1
    ):
        mul_op = next_op.output_ops[0]
        if mul_op.type == "Mul" and op.inputs[0] in (
            mul_op.inputs[0],
            mul_op.inputs[1],
        ):
            return all_ops + [next_op, mul_op]

    return []


def _is_rms_norm_scale_mul(
    norm_ops: List[Op], norm_output: Product, consumer_op: Op, model: ModelProto
) -> bool:
    """
    Check if ``consumer_op`` is the scale Mul of an elementwise affine RMSNorm, i.e. it
    multiplies the normalized output by the scale (γ). As per the ONNX spec, the scale
    must be broadcastable to the normalized shape. Otherwise, e.g. for a Mul by an
    activation, it is a separate op that consumes the RMSNorm output. If the shapes are
    unknown, it is not taken as the scale Mul.
    """
    if consumer_op.type != "Mul":
        return False

    scale_inputs = [inp for inp in consumer_op.inputs if inp != norm_output]
    if not scale_inputs:
        return False

    norm_start_op = norm_ops[0]
    input_shape = get_shape(model, norm_start_op.inputs[0].name)
    # The shape of the input is unknown, or the input is a scalar that cannot be normalized
    if not input_shape:
        return False

    if norm_start_op.type == "RMSNormalization":
        # A fused RMSNormalization with a scale input already has its scale, so the Mul
        # that follows it is a separate op
        if len(norm_start_op.inputs) > 1:
            return False
        norm_axis = get_node_attribute(norm_start_op.get_module(), "axis")
        norm_axis = norm_axis if norm_axis is not None else -1
    else:
        norm_axis = _get_rms_norm_axis(norm_ops[1], len(input_shape), model)
        if norm_axis is None:
            return False

    if not -len(input_shape) <= norm_axis < len(input_shape):
        return False

    scale_shape = get_shape(model, scale_inputs[0].name)
    return is_unidirectionally_broadcastable(scale_shape, input_shape[norm_axis:])


def _get_rms_norm_axis(
    reduce_mean_op: Op, rank: int, model: ModelProto
) -> Optional[int]:
    """
    Returns the first normalization axis of RMSNorm from its ReduceMean, or None if the
    ReduceMean does not reduce over a single trailing range of axes [axis, ..., rank - 1]
    """
    if len(reduce_mean_op.inputs) > 1:
        if not reduce_mean_op.inputs[1].is_const:
            return None
        reduce_axes = _get_numpy_array(model, reduce_mean_op.inputs[1].name)
    else:
        reduce_axes = get_node_attribute(reduce_mean_op.get_module(), "axes")
        if reduce_axes is None:
            # ReduceMean without axes reduces over all axes
            return 0

    reduce_axes = sorted(axis % rank for axis in np.asarray(reduce_axes).flatten())
    if reduce_axes and reduce_axes == list(range(reduce_axes[0], rank)):
        return reduce_axes[0]
    return None
