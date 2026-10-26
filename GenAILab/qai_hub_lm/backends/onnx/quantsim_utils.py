# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for configuring AIMET-ONNX. Borrowed from AI Hub Models."""

import copy
import os

import onnx
import torch

from aimet_onnx.defs import QSpec
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import OrtInferenceSession

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
    """
    w = embedding.weight.data
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1
    w_min = w.min()
    w_max = w.max()
    scale = (w_max - w_min) / (qmax - qmin)
    offset = torch.round(w_min / scale) - qmin
    w_q = torch.clamp(torch.round(w / scale + offset), qmin, qmax)
    embedding.weight.data = (w_q - offset) * scale


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


def has_bf16_io(model: onnx.ModelProto) -> bool:
    """Whether any graph input or output of ``model`` is bfloat16.

    ``QuantizationSimModel`` rejects such graphs outright, so they cannot be
    wrapped in a sim (see :class:`FloatOnlySim`).
    """
    return any(
        io.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for io in (*model.graph.input, *model.graph.output)
    )


class FloatOnlySim:
    """Float-only stand-in for a ``QuantizationSimModel``.

    Used for graphs that AIMET cannot wrap in a sim yet (bfloat16 graph I/O).
    It exposes the slice of the sim interface the GenAILab runner touches --
    ``model``, ``session``, ``providers``, ``_rebuild_session``, an empty
    ``qc_quantize_op_dict`` and ``export`` -- so the float model can be
    instantiated and evaluated through the normal generator and metrics path.
    With no quantizers, ``fp_mode`` and ``RemoveQuantization`` are no-ops.
    """

    def __init__(self, model: onnx.ModelProto, providers):
        self.model = _FloatModel(model)
        self.providers = providers
        self.qc_quantize_op_dict = {}
        self.param_names = []
        self.activation_names = []
        self.session = None
        self._rebuild_session()

    def _rebuild_session(self):
        # Release the old session (and its GPU memory) before building anew.
        self.session = None
        self.session = OrtInferenceSession(self.model.model, self.providers)

    @staticmethod
    def remove_quantizers(model: onnx.ModelProto) -> onnx.ModelProto:
        return model

    def export(self, path: str, filename_prefix: str, export_model: bool = True):
        """Write the float model; there are no encodings to export."""
        if export_model:
            # Saving with external data strips the tensors from the proto it is
            # given, which would break the live session; save a copy instead.
            onnx.save_model(
                copy.deepcopy(self.model.model),
                os.path.join(path, filename_prefix + ".onnx"),
                save_as_external_data=True,
                location=filename_prefix + ".data",
            )


class _FloatModel:
    """Minimal ``ONNXModel`` stand-in: just the ``.model`` proto attribute."""

    def __init__(self, model: onnx.ModelProto):
        self.model = model


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
