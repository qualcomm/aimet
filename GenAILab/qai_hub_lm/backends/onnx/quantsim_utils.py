# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for configuring AIMET-ONNX. Borrowed from AI Hub Models."""

import torch

from aimet_onnx.defs import QSpec
from aimet_onnx.quantsim import QuantizationSimModel

from GenAILab.bench.precision import (
    Granularity,
    WeightPrecision,
    float16,
    float32,
)


def weight_precision_to_qspec(precision: WeightPrecision) -> QSpec:
    """Convert a GenAILab weight precision into an AIMET-ONNX ``QSpec``."""
    if precision.is_float:
        return QSpec.per_tensor(precision.qtype)

    if precision.granularity == Granularity.PCQ:
        return QSpec.per_channel(precision.qtype)

    assert precision.block_size is not None
    if precision.granularity == Granularity.BQ:
        return QSpec.blockwise(
            precision.qtype,
            block_size=precision.block_size,
            symmetric=True,
        )

    if precision.granularity == Granularity.LPBQ:
        return QSpec.lpbq(
            precision.qtype,
            block_size=precision.block_size,
        )

    raise ValueError(f"Unsupported weight granularity: {precision.granularity}")


def quantize_embedding_weights(embedding: torch.nn.Module, n_bits: int):
    """Quantize embedding weights in-place using per-tensor asymmetric quantization.

    Derives scale and offset from the min/max of the weight tensor, quantizes
    to ``n_bits`` integers, then dequantizes back to a regular float tensor.
    The math runs in float32 so low-precision weights (e.g. bfloat16, which
    cannot represent integers above 256 exactly) still land on the integer
    grid; the result is cast back to the original dtype.
    """
    orig_dtype = embedding.weight.dtype
    w = embedding.weight.data.float()
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1
    w_min = w.min()
    w_max = w.max()
    scale = (w_max - w_min) / (qmax - qmin)
    offset = torch.round(w_min / scale) - qmin
    w_q = torch.clamp(torch.round(w / scale + offset), qmin, qmax)
    embedding.weight.data = ((w_q - offset) * scale).to(orig_dtype)


def get_ort_providers(
    device: torch.device,
) -> list[str | tuple[str, dict[str, int]]]:
    if device.type == "cuda":
        return (
            [
                ("CUDAExecutionProvider", {"device_id": device.index}),
                "CPUExecutionProvider",
            ]
            if device.index is not None
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return ["CPUExecutionProvider"]


class AttributePatch:
    def __init__(self, obj, attr_name, new_value):
        self.obj = obj
        self.attr_name = attr_name
        self.new_value = new_value

    class _NullAttribute:
        pass

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr_name, self._NullAttribute())
        setattr(self.obj, self.attr_name, self.new_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            delattr(self.obj, self.attr_name)
        except AttributeError:
            pass

        if not hasattr(self.obj, self.attr_name) and not isinstance(
            self.old_value, self._NullAttribute
        ):
            setattr(self.obj, self.attr_name, self.old_value)


def _remove_activation_quantizers(quantsim_model: QuantizationSimModel):
    for op_name, qc_op in quantsim_model.qc_quantize_op_dict.items():
        if op_name in quantsim_model.activation_names:
            qc_op.reset_encoding_stats()
            qc_op.enabled = False


def _disable_float_quantizers(sim: QuantizationSimModel):
    for quantizer in sim.qc_quantize_op_dict.values():
        if quantizer.precision() in (float16, float32):
            quantizer.enabled = False
