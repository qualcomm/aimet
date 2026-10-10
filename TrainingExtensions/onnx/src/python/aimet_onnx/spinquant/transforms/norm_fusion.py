# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""RMSNorm scale absorption: fuse gamma into downstream linear weights."""

from typing import List
import numpy as np

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import set_static_tensor, static_tensor

from aimet_onnx.llm_topology.ir_adapter import IrActiveNorm
from aimet_onnx.llm_topology.ir_analysis import get_weight_value

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def fuse_norm_layers_into_linears(active_norms: List[IrActiveNorm]):
    """Absorb RMSNorm gamma into downstream linear weights, then reset gamma to ones.

    For every affine RMSNorm in ``active_norms``, this function multiplies the
    scale weight (gamma) into the weights of its downstream linear
    (MatMul/Gemm/Conv) layers in-place, then resets gamma to ones.

    The transformation is numerically equivalent::

        gamma * RMSNorm(x) @ W  ==  RMSNorm(x) @ (diag(gamma) @ W)
                                ==  RMSNorm(x) @ W_fused

    where in ONNX convention W is [in_features, out_features], so::

        W_fused = gamma[:, None] * W

    After fusion, gamma is set to ones, making the norm a pure normalization
    with no learnable scale effect.

    :param active_norms: Active norms to fuse, resolved onto the IR model being
        mutated (see :func:`~.ir_adapter.resolve_active_norms`). Each entry
        carries the gamma tensor and the downstream linear nodes, and the tensors
        it names are what this function rewrites.
    """
    for active_norm in active_norms:
        scale_name = active_norm.scale_name
        downstream_linears = active_norm.downstream_linears

        if not downstream_linears:
            _logger.debug(
                "RMSNorm scale '%s': no downstream linear ops found, skipping.",
                scale_name,
            )
            continue

        scale = static_tensor(active_norm.scale).numpy()
        scale_dtype = scale.dtype
        scale_f64 = scale.astype(np.float64)  # promote for numerical precision

        for linear_node in downstream_linears:
            weight_value, is_transposed = get_weight_value(linear_node)
            weight = static_tensor(weight_value)
            if weight is None:
                _logger.warning(
                    "RMSNorm scale '%s': node '%s' has no static weight, skipping.",
                    scale_name,
                    linear_node.name,
                )
                continue
            W = weight.numpy()
            orig_dtype = W.dtype

            # Determine in_features based on storage layout (needed for tiling check below).
            if linear_node.op_type == "Conv":
                in_features = W.shape[1]  # [out, in, *k]
            elif is_transposed:
                in_features = W.shape[1]  # [out, in]
            else:
                in_features = W.shape[0]  # [in, out]

            # Repeat gamma when its length is smaller than in_features.
            scale_f64_effective = scale_f64
            if len(scale_f64) < in_features:
                if in_features % len(scale_f64) != 0:
                    raise ValueError(
                        f"RMSNorm scale '{scale_name}' length {len(scale_f64)} does not "
                        f"divide in_features={in_features} of op '{linear_node.name}'."
                    )
                tile_factor = in_features // len(scale_f64)
                scale_f64_effective = np.tile(scale_f64, tile_factor)
                _logger.debug(
                    "Repeating RMSNorm scale '%s' by %d for op '%s' "
                    "(gamma dim %d < in_features %d).",
                    scale_name,
                    tile_factor,
                    linear_node.name,
                    len(scale_f64),
                    in_features,
                )

            if linear_node.op_type == "Conv":
                # W shape: [out_channels, in_channels, *kernel]
                # gamma [in_channels] is absorbed along axis 1
                scale_broadcast = scale_f64_effective.reshape(
                    1, -1, *([1] * (W.ndim - 2))
                )
            elif is_transposed:
                # Gemm transB=1 or W -> Transpose -> MatMul: stored W is [out, in]
                # gamma [in_features] is absorbed along axis 1
                scale_broadcast = scale_f64_effective[None, :]
            else:
                # MatMul or Gemm transB=0: W shape [in_features, out_features]
                # gamma [in_features] is absorbed along axis 0
                scale_broadcast = scale_f64_effective[:, None]

            W_fused = (scale_broadcast * W.astype(np.float64)).astype(orig_dtype)
            set_static_tensor(weight_value, W_fused)
            _logger.debug(
                "Fused RMSNorm scale '%s' into weight '%s' of op '%s'.",
                scale_name,
                weight_value.name,
                linear_node.name,
            )

        # Reset gamma to ones so the norm no longer applies any scaling
        set_static_tensor(active_norm.scale, np.ones(scale.shape, dtype=scale_dtype))
