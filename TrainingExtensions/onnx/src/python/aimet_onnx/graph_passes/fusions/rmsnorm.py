# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
RMSNormalization fusion pass for onnx-ir models
"""

import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from aimet_onnx.ir_utils import get_constant_singleton_value, get_upstream_cast_type
from aimet_onnx.graph_passes.utils import is_unidirectionally_broadcastable
from .fusion_registry import register_fusion, AIMET_SUPERGROUP_DOMAIN
from . import _patterns


AXIS_NAME = "axis"
EPS_NAME = "epsilon"
OUTPUT_TYPE_NAME = "output_type"
SUPPORTED_FLOAT_TYPES = (
    onnx_ir.DataType.FLOAT,
    onnx_ir.DataType.FLOAT16,
    onnx_ir.DataType.BFLOAT16,
)


def _dims(value: onnx_ir.Value) -> list | None:
    """
    Return the shape of ``value`` with symbolic dims as their names and unknown dims as
    None, or None if the shape is unknown
    """
    if value.shape is None:
        return None
    return [dim if isinstance(dim, int) else dim.value for dim in value.shape.dims]


@register_fusion("RMSNormalization", pattern_idx=0)
@register_fusion("RMSNormalization", pattern_idx=1)
@register_fusion("RMSNormalization", pattern_idx=2)
class NonAffineRMSNormFusion(rewriter.RewriteRuleClassBase):
    """
    NonAffineRMSNormalization(x) = x / Sqrt(E(x**2) + ε)

    Expected graph:
    Version 1: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
    Mul or Pow(x, 2)|
            |       |
        ReduceMean  |
            |       |
            Add     |
            |       |
            Sqrt    |
        1   |       |
        +-- Div     |
            |       |
            +---+---+
                Mul

    Version 2: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
            |       Mul or Pow(x, 2)
            |       |
            |       ReduceMean
            |       |
            |       Add
            |       |
            |       Sqrt
            |       |
            +---+---+
                Div

    Version 3: With x * rsqrt ( denominator )
                x
            +---+---+
            |       |
            |   Mul or Pow(x, 2)
            |       |
            |   ReduceMean
            |       |
            |       Add
            |       |
            |   Pow(·, -0.5)
            |       |
            +---+---+
                Mul
    """

    def __init__(self, *args, pattern_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if pattern_idx not in {0, 1, 2}:
            raise ValueError(f"Invalid value for div_impl: {pattern_idx}")
        self.pattern_idx = pattern_idx

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
    ):
        return _patterns.non_affine_rms_normalize(
            op, input_x, EPS_NAME, AXIS_NAME, self.pattern_idx
        )

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        match_result = pattern.MatchResult()
        axes: onnx_ir.Attr | onnx_ir.Value | None = kwargs.get(AXIS_NAME)
        epsilon: onnx_ir.Value | None = kwargs.get(EPS_NAME)

        axes_value = get_constant_singleton_value(axes)
        if axes is not None and axes_value is None:
            return match_result.fail(
                f"Axes attribute must be single element constant, got: {axes}"
            )

        eps = get_constant_singleton_value(epsilon)
        if eps is None or eps < 0:
            return match_result.fail(
                f"Epsilon must be a constant positive value, got {epsilon}"
            )

        return match_result

    # pylint: disable=arguments-differ
    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        axis = get_constant_singleton_value(kwargs.get(AXIS_NAME))
        axis = axis if axis is not None else -1
        output = op.RMSNormalization(
            input_x,
            axis=axis,
            epsilon=get_constant_singleton_value(kwargs.get(EPS_NAME)),
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )
        return output


@register_fusion("RMSNormalization", pattern_idx=0)
@register_fusion("RMSNormalization", pattern_idx=1)
class RMSNormFusion(pattern.RewriteRuleClassBase):
    """
    Expected graph:

                x
                |
        RMSNormalization(x)
                |
              Cast (Optional)
                |
               Mul

    Note:
        Input cast is not matched here intentionally for the following reasons

          - Optimization passes can result in input cast having multiple consumers, preventing fusion in some cases
          - In ORT <= 1.26, function overloads are not dispatched correctly, causing incorrect execution if patterns
            differ functionally

    """

    def __init__(self, *args, pattern_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if pattern_idx not in {0, 1}:
            raise ValueError(f"Invalid value for pattern_idx: {pattern_idx}")
        self.pattern_idx = pattern_idx

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
        scale: pattern.Var,
    ):
        return _patterns.rms_normalize(
            op,
            input_x,
            scale,
            EPS_NAME,
            AXIS_NAME,
            output_type=OUTPUT_TYPE_NAME,
            pattern_idx=self.pattern_idx,
        )

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        scale: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        match_result = pattern.MatchResult()
        axes: onnx_ir.Attr | None = kwargs.get(AXIS_NAME)
        epsilon: onnx_ir.Attr | None = kwargs.get(EPS_NAME)
        output_type: onnx_ir.Attr | None = kwargs.get(OUTPUT_TYPE_NAME)

        axes_value = get_constant_singleton_value(axes)
        if axes is not None and axes_value is None:
            return match_result.fail(
                f"Axes attribute must be single element constant, got: {axes}"
            )

        # The Mul is the RMSNorm scale Mul only if its other input is a valid scale as per
        # the ONNX spec, i.e. unidirectionally broadcastable to the normalized shape
        # ``input_shape[axis:]``. Otherwise, e.g. for a Mul by an activation, it consumes
        # the RMSNorm output. If the shapes are unknown, it is not taken as the scale Mul.
        axis = axes_value if axes_value is not None else -1
        input_shape = _dims(input_x)
        if input_shape is None or not -len(input_shape) <= axis < len(input_shape):
            return match_result.fail(
                f"Input shape {input_x.shape} is unknown or axis {axis} is out of range"
            )
        if not is_unidirectionally_broadcastable(_dims(scale), input_shape[axis:]):
            return match_result.fail(
                f"Scale shape {scale.shape} is not broadcastable to the normalized "
                f"shape of input shape {input_x.shape} from axis {axis}"
            )

        eps = get_constant_singleton_value(epsilon)
        if epsilon is None or eps < 0:
            return match_result.fail(
                f"Epsilon must be a constant positive value, got {epsilon}"
            )

        stash_type = get_upstream_cast_type(input_x)
        if stash_type is not None and stash_type not in SUPPORTED_FLOAT_TYPES:
            return match_result.fail(
                f"Only {SUPPORTED_FLOAT_TYPES} are supported for stash_type, got {stash_type}"
            )
        if (
            output_type
            and get_constant_singleton_value(output_type) not in SUPPORTED_FLOAT_TYPES
        ):
            return match_result.fail(
                f"Only {SUPPORTED_FLOAT_TYPES} are supported for output_type, got {output_type}"
            )
        if output_type is not None and stash_type is None:
            return match_result.fail(
                "Internal cast operation only supported if pattern is preceded by cast op"
            )

        return match_result

    # pylint: disable=arguments-differ
    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        scale: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        epsilon = get_constant_singleton_value(kwargs.get(EPS_NAME))
        axis = get_constant_singleton_value(kwargs.get(AXIS_NAME))
        axis = axis if axis is not None else -1
        attrs = {}
        stash_type = get_upstream_cast_type(input_x)
        output_type = kwargs.get(OUTPUT_TYPE_NAME)
        if stash_type is not None and output_type is not None:
            attrs["stash_type"] = stash_type
        output = op.RMSNormalization(
            input_x,
            scale,
            axis=axis,
            epsilon=epsilon,
            _domain=AIMET_SUPERGROUP_DOMAIN,
            **attrs,
        )
        return output
