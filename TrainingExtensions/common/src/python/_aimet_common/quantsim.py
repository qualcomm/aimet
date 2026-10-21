# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Common utility for Quantization"""

import os
import functools
from typing import TypeVar, Union, Tuple, Dict
import numpy as np
import torch
import ml_dtypes

from .defs import QuantScheme, QuantizationDataType, qtype, Float
from .quantsim_config.quantsim_config import QuantSimConfigurator
from . import libpymo

__all__ = [
    "ALLOW_EXPERIMENTAL",
    "VALID_ENCODING_VERSIONS",
    "_INT32_MINIMUM_SCALE",
    "_adjust_weight_scale_against_bias_overflow",
    "_adjust_weight_scale_against_export_dtype_underflow",
    "_adjust_weight_scale_against_scale_underflow",
    "_get_minimum_scale",
    "_is_bias_out_of_int32_range",
    "_max_representable_value",
    "calculate_delta_offset",
    "compute_min_max_given_delta_offset",
    "create_encoding_from_min_max_for_precision",
    "encoding_version",
    "extract_global_quantizer_args",
    "validate_quantsim_inputs",
]


# Defined below is a quantization encoding format version, which will follow XX.YY.ZZ versioning as described below,
#
#    XX = Major Revision
#    YY = Minor Revision
#    ZZ = Patching version
#
# Change in major revision should indicate substantial change to the format, updates to minor version indicates
# additional information element being added to encoding format and might require update to fully consume the encodings.
# The patching version shall be updated to indicate minor updates to quantization simulation e.g. bug fix etc.
encoding_version = os.getenv("AIMET_ENCODING_VERSION", "1.0.0")
ALLOW_EXPERIMENTAL = False
VALID_ENCODING_VERSIONS = (
    "0.6.1",
    "1.0.0",
    "2.0.0",
    "2.1.0",
)

if encoding_version not in VALID_ENCODING_VERSIONS:
    raise RuntimeError(
        "Invalid AIMET_ENCODING_VERSION variable."
        f"Expected one of {sorted(list(VALID_ENCODING_VERSIONS))}; got {encoding_version}"
    )


def gate_min_max(
    min_val: Union[float, np.ndarray], max_val: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Gates min and max encoding values to retain zero in the range representation.
    Rules : min at maximum can be zero, max at minimum can be zero and
    if max and min are equal, adds epsilon to maintain range.
    :param min_val: min encoding value
    :param max_val: max encoding value
    :return: gated min and max values
    """

    epsilon = 1e-5
    # For per channel quantization
    if isinstance(min_val, np.ndarray):
        gated_min = np.clip(min_val, None, 0.0)
        gated_max = np.clip(max_val, 0.0, None)
        gated_max = np.clip(gated_max, gated_min + epsilon, None)
    else:
        gated_min = min(min_val, 0.0)
        gated_max = max(max_val, 0.0)
        gated_max = max(gated_max, gated_min + epsilon)

    return gated_min, gated_max


def is_non_strict_symmetric(
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool,
    is_unsigned_symmetric: bool,
) -> bool:
    """
    Check whether non-strict symmetric encoding or not
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :param is_unsigned_symmetric: is_unsigned_symmetric flag
    :return: True if it satisfies non-strict symmetric else False
    """
    return (
        use_symmetric_encodings
        and not use_strict_symmetric
        and not is_unsigned_symmetric
    )


def create_encoding_from_min_max(
    min_val: float,
    max_val: float,
    bitwidth: int,
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool,
) -> libpymo.TfEncoding:
    """
    Returns a TfEncoding object with the provided min/max/bitwidth/symmetry

    :param min_val: Min value of the encoding
    :param max_val: Max value of the encoding
    :param bitwidth: Encoding bitwidth
    :param use_symmetric_encodings: If True, results in encoding with min = -max - delta
    :param use_strict_symmetric: If True, results in encoding with min = -max
    :return: libpymo.TfEncoding object
    """
    delta, offset = calculate_delta_offset(
        min_val, max_val, bitwidth, use_symmetric_encodings, use_strict_symmetric
    )

    encoding = libpymo.TfEncoding()
    encoding.bw = bitwidth
    encoding.min = min_val
    encoding.max = max_val
    encoding.delta = delta
    encoding.offset = offset
    # Note: need to recompute grid to account for offset rounding
    return recompute_grid_params(
        encoding, bitwidth, use_symmetric_encodings, use_strict_symmetric
    )


def create_encoding_from_min_max_for_precision(
    min_val: float,
    max_val: float,
    precision: qtype,
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool = False,
) -> libpymo.TfEncoding:
    """
    Returns a TfEncoding object spanning [min_val, max_val] at the given precision.

    Unlike :func:`create_encoding_from_min_max`, which takes a bitwidth and always builds
    an affine encoding, this takes a qtype and so can also build the single-scale
    encodings used by low-precision float formats. The two are kept separate for now so
    that existing bitwidth-based callers are unaffected; they can be merged once every
    caller speaks qtype.

    :param min_val: Min value of the encoding
    :param max_val: Max value of the encoding
    :param precision: qtype to quantize to
    :param use_symmetric_encodings: If True, results in encoding with min = -max - delta.
        Ignored for floating-point precisions, which are always symmetric.
    :param use_strict_symmetric: If True, results in encoding with min = -max.
        Ignored for floating-point precisions.
    :return: libpymo.TfEncoding object
    """
    dtype, bitwidth = precision.to_legacy_repr()

    if dtype == QuantizationDataType.float:
        return _create_float_encoding_from_min_max(min_val, max_val, precision)

    return create_encoding_from_min_max(
        min_val, max_val, bitwidth, use_symmetric_encodings, use_strict_symmetric
    )


def _max_representable_value(precision: "Float") -> float:
    """
    Returns the largest finite value representable by a floating-point qtype.

    Mirrors DlQuantization::QuantizationType::Float, which performs the same derivation
    for the quantization runtime. The two must agree, which
    test_python_max_representable_value_matches_runtime checks for every supported format.
    """
    exponent_bits = precision.exponent_bits
    mantissa_bits = precision.mantissa_bits
    mantissa_unit = 2.0**-mantissa_bits
    exponent_bias = 2 ** (exponent_bits - 1) - 1 + (1 if precision.unsigned_zero else 0)

    if precision.unsigned_zero:
        # fnuz formats spend no encoding on infinities or negative zero
        max_exponent = 2**exponent_bits - 1 - exponent_bias
        max_mantissa = 2.0 - mantissa_unit
    elif precision.finite:
        # Top exponent is usable except for the all-ones mantissa, which is NaN
        max_exponent = 2**exponent_bits - 1 - exponent_bias
        max_mantissa = 2.0 - 2.0 * mantissa_unit
    else:
        # IEEE-style: top exponent is reserved for inf/NaN
        max_exponent = 2**exponent_bits - 2 - exponent_bias
        max_mantissa = 2.0 - mantissa_unit

    return max_mantissa * 2.0**max_exponent


def _create_float_encoding_from_min_max(
    min_val: float,
    max_val: float,
    precision: qtype,
) -> libpymo.TfEncoding:
    """
    Builds a floating-point encoding covering [min_val, max_val].

    Unlike affine encodings, float encodings store a single symmetric scale in ``delta``
    and leave ``offset`` at zero; ``min``/``max`` describe the representable range that
    results from that scale.
    """
    max_representable_value = _max_representable_value(precision)

    amax = max(abs(min_val), abs(max_val))
    # The quantization kernels require a strictly positive scale, so degenerate ranges
    # fall back to the smallest representable step rather than 0.
    scale = max(amax / max_representable_value, float(np.finfo(np.float32).tiny))

    encoding = libpymo.TfEncoding()
    encoding.bw = precision.to_legacy_repr()[1]
    encoding.delta = scale
    encoding.offset = 0.0
    encoding.min = -scale * max_representable_value
    encoding.max = scale * max_representable_value
    return encoding


def calculate_delta_offset(
    min_val: Union[float, np.ndarray],
    max_val: Union[float, np.ndarray],
    bitwidth: int,
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool,
) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
    """
    Calculates delta and offset given min and max.

    Quantization policy:
    - Asymmetric quantization is applied if all channels have strictly non-negative ranges (i.e., np.all(min_val >= 0)).
    - Symmetric quantization is applied only if `use_symmetric_encodings=True` and
     at least one channel has a negative range (i.e., np.any(min_val < 0)).

    :param min_val: min encoding value
    :param max_val: max encoding value
    :param bitwidth: bitwidth used for quantization
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :return: delta and offset values computed
    """
    num_steps = 2**bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    min_val, max_val = gate_min_max(min_val, max_val)

    # Check if both delta and offset are scalars
    if np.isscalar(min_val) and np.isscalar(max_val):
        # Use only max val to compute delta in the case of signed symmetric
        if use_symmetric_encodings and min_val < 0:
            num_positive_steps = np.floor(num_steps / 2)
            delta = max_val / num_positive_steps
            offset = -num_positive_steps
            if not use_strict_symmetric:
                offset -= 1
        else:
            delta = (max_val - min_val) / num_steps
            offset = round(min_val / delta)
        return delta, offset

    # np.array case
    min_val = np.asarray(min_val, dtype=np.float32)
    max_val = np.asarray(max_val, dtype=np.float32)

    delta = np.empty_like(min_val, dtype=np.float32)
    offset = np.empty_like(min_val, dtype=np.int32)

    num_positive_steps = np.floor(num_steps / 2)

    apply_symmetric = use_symmetric_encodings and not np.all(min_val >= 0)
    if apply_symmetric:
        delta[:] = max_val / num_positive_steps
        offset[:] = -num_positive_steps
        if not use_strict_symmetric:
            offset[:] -= 1
    else:
        delta[:] = (max_val - min_val) / num_steps
        offset[:] = np.round(min_val / delta).astype(np.int32)

    return delta, offset


def compute_min_max_given_delta_offset(
    delta: Union[float, np.ndarray],
    offset: Union[int, np.ndarray],
    bitwidth: int,
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool,
) -> Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
    """
    Compute min and max given delta and offset.

    :param delta: Delta to compute with
    :param offset: Offset to compute with
    :param bitwidth: Bitwidth for finding number of steps
    :param use_symmetric_encodings: True if symmetric, False otherwise
    :param use_strict_symmetric: True if using strict symmetric, False otherwise
    :return: Tuple of computed min and max values
    """
    num_steps = 2**bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    # Check if both delta and offset are scalars
    is_scalar = np.isscalar(delta) and np.isscalar(offset)

    delta = np.asarray(delta, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)

    min_val = delta * offset
    max_val = (num_steps + offset) * delta

    # If inputs were scalars, return scalars
    if is_scalar:
        return float(min_val), float(max_val)

    return min_val, max_val


def recompute_grid_params(
    current_encoding: libpymo.TfEncoding,
    bitwidth: int,
    use_symmetric_encoding: bool,
    use_strict_symmetric: bool = False,
) -> libpymo.TfEncoding:
    """
    Recomputes the encoding grid params - min/max/offset and delta.

    :param current_encoding: Encoding associated with the quantizer as TfEncoding
    :param bitwidth: bit width configured for the quantizer
    :param use_symmetric_encoding: symmetric or asymmetric mode
    :param use_strict_symmetric: True if using strict symmetric, False otherwise
    :return: updated encoding params as libpymo.TfEncoding type.
    """

    MIN_RANGE = 0.01
    min_val = min(0.0, current_encoding.min)
    max_val = max(0.0, current_encoding.max, (min_val + MIN_RANGE))
    updated_encoding = libpymo.TfEncoding()

    # check mode used to recompute delta and offset
    if use_symmetric_encoding:
        num_positive_steps = (2 ** (bitwidth - 1)) - 1
        num_negative_steps = 2 ** (bitwidth - 1)
        delta = max(
            abs(max_val / num_positive_steps), abs(min_val / num_negative_steps)
        )
        offset = -(num_negative_steps - int(use_strict_symmetric))
        # recompute min/max values
        min_val = delta * offset
        max_val = delta * num_positive_steps

    else:
        num_steps = (2**bitwidth) - 1
        delta = (max_val - min_val) / num_steps
        # @todo check zero point representation related code
        offset = round(min_val / delta)
        # recompute min/max values
        min_val = delta * offset
        max_val = min_val + delta * num_steps

    updated_encoding.bw = bitwidth
    updated_encoding.min = min_val
    updated_encoding.max = max_val
    updated_encoding.delta = delta
    updated_encoding.offset = offset

    return updated_encoding


def validate_quantsim_inputs(
    quant_scheme: Union[str, QuantScheme],
    rounding_mode: str,
    default_output_bw: int,
    default_param_bw: int,
    data_type: QuantizationDataType = QuantizationDataType.int,
):
    """
    Perform sanity checks on inputs to QuantSim
    :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or 'percentile'
                         or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
                         or QuantScheme.post_training_percentile
    :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
    :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs
    :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
    :param data_type: Data type of the quantized values (int or float).
    """
    _validate_quant_scheme(quant_scheme)
    _validate_rounding_mode(rounding_mode)
    _validate_bitwidth(default_output_bw, default_param_bw, data_type)


def _validate_quant_scheme(quant_scheme: Union[str, QuantScheme]):
    if quant_scheme not in ("tf_enhanced", "tf", "percentile") and not isinstance(
        quant_scheme, QuantScheme
    ):
        raise ValueError(
            "Parameter quantization mode is not a valid selection. Valid selections are "
            "tf, tf_enhanced, percentile, QuantScheme.post_training_tf, "
            "QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_percentile"
        )


def _validate_rounding_mode(rounding_mode: str):
    if rounding_mode not in ("nearest", "stochastic"):
        raise ValueError(
            "Parameter round mode is not a valid selection. Valid selections are nearest or "
            "stochastic"
        )


def _validate_bitwidth(
    default_output_bw: int,
    default_param_bw: int,
    data_type: QuantizationDataType = QuantizationDataType.int,
):
    if default_param_bw < 2 or default_param_bw > 32:
        raise ValueError(
            "Default bitwidth for parameters must be between 2 and 32, not "
            + str(default_param_bw)
        )

    if default_output_bw < 4 or default_output_bw > 32:
        raise ValueError(
            "Activation bitwidth must be between 4 and 32, not "
            + str(default_output_bw)
        )

    if ALLOW_EXPERIMENTAL:
        if data_type == QuantizationDataType.float and default_output_bw not in [
            8,
            16,
            32,
        ]:
            raise ValueError(
                "float data_type can only be used when default_output_bw set to 8, 16 or 32, not "
                + str(default_output_bw)
            )

        if data_type == QuantizationDataType.float and default_param_bw not in [
            8,
            16,
            32,
        ]:
            raise ValueError(
                "float data_type can only be used when default_param_bw set to 8, 16 or 32, not "
                + str(default_param_bw)
            )

    else:
        if data_type == QuantizationDataType.float and default_output_bw not in [
            16,
            32,
        ]:
            raise ValueError(
                "float data_type can only be used when default_output_bw set to 16 or 32, not "
                + str(default_output_bw)
            )

        if data_type == QuantizationDataType.float and default_param_bw not in [16, 32]:
            raise ValueError(
                "float data_type can only be used when default_param_bw set to 16 or 32, not "
                + str(default_param_bw)
            )


def extract_global_quantizer_args(
    quant_scheme: Union[str, QuantScheme], quantsim_configurator: QuantSimConfigurator
) -> Dict:
    """
    Extracts quantizer arguments used to configure QuantSim
    :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or 'percentile'
                         or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
                         or QuantScheme.post_training_percentile
    :param quantsim_configurator: An instance of QuantSimConfigurator which has been populated either by config file
                                  or via function arguments.
    :return: A dictionary of quantizer arguments
    """
    quant_args = {}
    quantsim_config = quantsim_configurator.quantsim_configs
    param_dict = quantsim_config["defaults"]["params"]
    is_per_channel_quant = quantsim_config["defaults"].get(
        "per_channel_quantization", False
    )

    # Set per_channel_quantization=True if any of the op types have
    # per_channel_quantization set to True in the config file,
    # even if the default is False
    is_per_channel_quant |= any(
        op_config.get("per_channel_quantization", False)
        for _, op_config in quantsim_config["op_type"].items()
    )

    if (
        isinstance(quant_scheme, str)
        and quant_scheme == QuantScheme.training_range_learning_with_tf_init
        or quant_scheme == QuantScheme.training_range_learning_with_tf_init
    ):
        quant_scheme = QuantScheme.post_training_tf
    if (
        isinstance(quant_scheme, str)
        and quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init
    ) or quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init:
        quant_scheme = QuantScheme.post_training_tf_enhanced

    quant_args.update(
        {
            "quant_scheme": quant_scheme.name
            if isinstance(quant_scheme, QuantScheme)
            else quant_scheme,
            "param_bitwidth": quantsim_configurator.default_param_bw,
            "activation_bitwidth": quantsim_configurator.default_output_bw,
            "dtype": quantsim_configurator.default_param_data_type.name,
            "is_symmetric": param_dict["is_symmetric"]
            if "is_symmetric" in param_dict
            else is_per_channel_quant,
            "per_channel_quantization": is_per_channel_quant,
        }
    )

    return quant_args


@functools.lru_cache
def _get_minimum_scale(num_steps: int) -> float:
    """
    Return the minimum scale given the number of steps in the quantization grid.

    We define the minimum scale as the largest s <= float32.eps such that
    -0.005 <= s * min(x_int) <  s * max(x_int) <= 0.005

    Following this rule, the minimum scale in practice will be:

      | dtype | minimum scale |
      |-------|---------------|
      |  int4 |    1.19e-07   | (note: float32.eps = 1.19e-07)
      |  int8 |    1.19e-07   |
      | int16 |    1.19e-07   |
      | int32 |    2.33e-12   | (note: float64.eps = 2.22e-16)

    """
    fp32_eps = float(np.finfo(np.float32).eps)

    _MINIMUM_RANGE_TO_REPRESENT = (-0.005, 0.005)
    _min, _max = _MINIMUM_RANGE_TO_REPRESENT
    return min(fp32_eps, (_max - _min) / num_steps)


def _is_bias_out_of_int32_range(
    bias_float: Union[np.ndarray, float],
    bias_scale: np.ndarray,
    num_steps: int = 2**31,
) -> np.ndarray:
    """
    Checks if the quantized bias value is outside the signed int32 range (-2147483648 to 2147483647)

    NOTE: Directly computes the valid range for bias values in float-space to avoid division which can be sensitive.
    and allows to account for signed int32 range

    :param bias_float: Bias float values
    :param bias_scale: Bias scale
    :param num_steps: Maximum allowed quantized bias value (default is 2**31)
    :return: Boolean array indicating whether each bias value is out of range
    """
    # Ensures precision in calculations.
    bias_scale = bias_scale.astype(np.float64)
    bias_float = bias_float.astype(np.float64)
    min_value = bias_scale * -(num_steps + 1)
    max_value = bias_scale * num_steps
    return (bias_float > max_value) | (bias_float < min_value)


T_w = TypeVar("T_w", bound=Union[np.ndarray, float, torch.Tensor])


def _to_torch_tensor(value: Union[np.ndarray, float, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype == ml_dtypes.bfloat16:
            value = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
        else:
            value = torch.from_numpy(value)
    else:
        value = torch.tensor(value)

    return value


def _adjust_weight_scale_against_bias_overflow(
    bias_float: Union[np.ndarray, float, torch.Tensor],
    input_scale: Union[np.ndarray, float, torch.Tensor],
    weight_scale: T_w,
    num_steps: int = 2**31,
) -> T_w:
    """
    Adjusts weight scales to prevent bias overflow during INT16 quantization.

    Given, bias_scale = input_scale * weight_scale,
    If bias_float / bias_scale >= threshold, then:
        adjusted_weight_scale = bias_float / (threshold * input_scale)

    :param bias_float: Bias float values per output channel
    :param input_scale: Input scale applied to all input values
    :param weight_scale: np.ndarray or float, weight scale applied to weights
    :param num_steps: Maximum allowed quantized bias value (default threshold is 2**31)
    :return: adjusted weight scales
    """
    bias_float = _to_torch_tensor(bias_float)
    input_scale = _to_torch_tensor(input_scale)

    ret_type = type(weight_scale)
    ret_dtype = weight_scale.dtype if isinstance(weight_scale, torch.Tensor) else None

    weight_scale = _to_torch_tensor(weight_scale)
    ret_shape = weight_scale.shape

    if torch.any(input_scale == 0):
        raise ValueError("input_scale must be non-zero.")

    weight_scale = weight_scale.squeeze().to(torch.float64)
    input_scale = input_scale.squeeze().to(torch.float64)
    bias_float = bias_float.squeeze().to(torch.float64)

    if bias_float.shape != weight_scale.shape:
        if weight_scale.shape != ():
            raise RuntimeError(
                f"weight_scale must be either a scalar or have the same shape as bias_float. "
                f"Got weight_scale.shape={weight_scale.shape} and bias_float.shape={bias_float.shape}"
            )
        bias_float = bias_float.abs().amax()

    adjusted_weight_scale = torch.maximum(
        weight_scale,
        (bias_float / (num_steps * input_scale)).abs(),
    )
    adjusted_weight_scale = adjusted_weight_scale.to(ret_dtype).reshape(ret_shape)

    if issubclass(ret_type, float):
        return adjusted_weight_scale.item()
    elif issubclass(ret_type, np.ndarray):
        return adjusted_weight_scale.cpu().numpy().astype(np.float32)
    else:
        return adjusted_weight_scale


def _adjust_weight_scale_against_scale_underflow(
    input_scale: Union[np.ndarray, torch.Tensor],
    weight_scale: Union[np.ndarray, torch.Tensor],
    output_scale: Union[np.ndarray, torch.Tensor],
):
    """
    Cap weight scale such that
    requant_scale = input_scale * weight_scale / output_scale > 2**-24
    This is to work around a bug in HexNN where if requant_scale <= 2**-24,
    HexNN misinterprets the scale to be 2**(e+32) due to internal type casting bug.
    """
    ret_type = type(weight_scale)

    if isinstance(input_scale, np.ndarray):
        input_scale = torch.from_numpy(input_scale)
    if isinstance(weight_scale, np.ndarray):
        weight_scale = torch.from_numpy(weight_scale)
    if isinstance(input_scale, np.ndarray):
        output_scale = torch.from_numpy(output_scale)

    # Use 2**-23.5 as min_requant_scale to ensure requant_scale is strictly greater than 2**-24
    min_requant_scale = 2**-23.5
    min_weight_scale = (min_requant_scale * output_scale) / input_scale
    ret = torch.maximum(weight_scale, min_weight_scale)
    return ret.numpy() if issubclass(ret_type, np.ndarray) else ret


def _adjust_weight_scale_against_export_dtype_underflow(
    input_scale: Union[np.ndarray, torch.Tensor],
    weight_scale: T_w,
    floor: float,
) -> T_w:
    """
    Bump ``weight_scale`` so that ``bias_scale = input_scale * weight_scale >= floor``,
    per output channel.

    The int32 bias scale is exported as ``QuantizeLinear.y_scale`` /
    ``DequantizeLinear.x_scale``, whose ONNX dtype matches the surrounding
    activation dtype. When that dtype is fp16 (w8a16/w16a16), a bias_scale
    smaller than ``fp16.tiny`` (~6.1e-5) underflows to zero on cast and causes
    divide-by-zero during ``_quantize_const``.

    Bumping ``weight_scale`` (rather than clamping ``bias_scale`` directly)
    preserves the ``bias_scale = input_scale * weight_scale`` invariant that
    backends rely on to fuse ``input_q @ weight_q + bias_q`` without rescaling.

    Raises RuntimeError if ``input_scale`` itself is below ``floor`` — no
    weight_scale bump can produce a bias_scale that survives fp16 export
    when input_scale itself underflows.
    """
    ret_type = type(weight_scale)
    ret_dtype = weight_scale.dtype if isinstance(weight_scale, torch.Tensor) else None

    input_scale = _to_torch_tensor(input_scale)
    weight_scale = _to_torch_tensor(weight_scale)
    ret_shape = weight_scale.shape

    if torch.any(input_scale <= 0):
        raise RuntimeError("input_scale must be strictly positive.")
    if torch.any(input_scale < floor):
        raise RuntimeError(
            f"input_scale ({float(input_scale.min()):.3e}) is below the "
            f"export-dtype floor ({floor:.3e}); cannot derive an analytic "
            "bias_scale that round-trips through the export dtype. "
            "Please recalibrate the input encoding or increase activation precision."
        )

    required = torch.as_tensor(floor, dtype=torch.float64) / input_scale.to(
        torch.float64
    )
    adjusted = torch.maximum(weight_scale.to(torch.float64), required)
    adjusted = (
        adjusted.to(ret_dtype).reshape(ret_shape)
        if ret_dtype
        else adjusted.reshape(ret_shape)
    )

    if issubclass(ret_type, float):
        return adjusted.item()
    elif issubclass(ret_type, np.ndarray):
        return adjusted.cpu().numpy().astype(np.float32)
    else:
        return adjusted


_INT4_MINIMUM_SCALE = _get_minimum_scale(2**4 - 1)
_INT8_MINIMUM_SCALE = _get_minimum_scale(2**8 - 1)
_INT16_MINIMUM_SCALE = _get_minimum_scale(2**16 - 1)
_INT32_MINIMUM_SCALE = _get_minimum_scale(2**32 - 1)
