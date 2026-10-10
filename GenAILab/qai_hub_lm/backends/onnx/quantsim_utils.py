# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for configuring AIMET-ONNX. Borrowed from AI Hub Models."""

import torch
import onnx

from aimet_onnx.common.defs import qtype
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    set_grouped_blockwise_quantization_for_weights,
    set_blockwise_quantization_for_weights,
    set_lpbq_for_params,
)

from aimet_onnx.llm_configurator.llm_configurator import (
    _tie_quantizers_for_kv_cache,
)

from GenAILab.bench.precision import (
    Granularity,
    PrecisionConfig,
    WeightPrecision,
    float16,
    float32,
)
from GenAILab.bench.onnx.quant_recipes import _get_lm_head_node_names


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


def _resolve_kv_cache_quantization(
    quantsim_model: QuantizationSimModel, precision: qtype
) -> None:
    if precision in (float16, float32):
        # todo place KV cache quantizers in float mode
        pass
    else:
        kv_io_dict = {
            inp.name: inp.name.replace("in", "out")
            for inp in quantsim_model.model.model.graph.input
            if "past_key" in inp.name or "past_value" in inp.name
        }
        _tie_quantizers_for_kv_cache(quantsim_model, kv_io_dict)
        _set_tensors_to_output_n_bit_symmmetric(quantsim_model, precision.bits)


def _set_tensors_to_output_n_bit_symmmetric(
    quantsim_model: QuantizationSimModel, n: int = 8
):
    out_tensors = []
    out_tensors.extend(
        [
            t.name
            for t in quantsim_model.model.graph().input
            if "past_key" in t.name or "past_value" in t.name
        ]
    )
    out_tensors.extend(
        [
            t.name.replace("_updated", "")
            for t in quantsim_model.model.graph().output
            if "past_key" in t.name or "past_value" in t.name
        ]
    )
    for out_tensor in out_tensors:
        _set_tensor_to_n_bit_symmetric(quantsim_model, out_tensor, n)


def _set_tensor_to_n_bit_symmetric(
    quantsim_model: QuantizationSimModel, tensor_name: str, n: int = 8
):
    if tensor_name in quantsim_model.qc_quantize_op_dict:
        quantizer = quantsim_model.qc_quantize_op_dict[tensor_name]
        quantizer.set_bitwidth(n)
        quantizer.use_symmetric_encodings = True


def _set_lm_head_precision(
    quantsim_model: QuantizationSimModel, precision: WeightPrecision
):
    # todo: update placing LM head in LPBQ/BQ if specified
    # todo: update this to support weights in O, I format (like from torch.onnx.export)
    if precision.is_float:
        # FP lm_head — disable the lm_head weight quantizer.
        for weight in _get_lm_head_weights(quantsim_model.model.model):
            if weight.name in quantsim_model.qc_quantize_op_dict:
                quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
                quantizer.reset_encoding_stats()
                quantizer.enabled = False
        return
    if precision.granularity == Granularity.LPBQ:
        set_lpbq_for_params(
            sim=quantsim_model,
            bitwidth=precision.qtype.bits,
            block_size=precision.block_size,
            nodes_to_include=_get_lm_head_node_names(quantsim_model),
        )
    elif precision.granularity == Granularity.PCQ:
        for weight in _get_lm_head_weights(quantsim_model.model.model):
            quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
            quantizer.set_bitwidth(precision.qtype.bits)
            quantizer.quant_info.blockSize = 0
            quantizer.quant_info.blockAxis = -1
            quantizer.enable_per_channel_quantization()


def _get_lm_head_weights(quantsim_model: onnx.ModelProto):
    vocab_size = quantsim_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    for weight in quantsim_model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            for node in quantsim_model.graph.node:
                if node.op_type in ("Gemm", "MatMul", "Conv") and node.input[1] in {
                    weight.name,
                    weight.name + "_updated",
                    weight.name + "_qdq",
                }:
                    yield weight


def _remove_decoder_block_weight_quantizers(quantsim_model: QuantizationSimModel):
    """Disable weight quantizers on decoder-stack Gemm/MatMul/Conv ops.

    Used when ``precision.blocks.qtype`` is a floating-point type. lm_head
    weights and activation/KV quantizers are left untouched.
    """
    lm_head_nodes = set(_get_lm_head_node_names(quantsim_model))
    target_op_types = {"Gemm", "MatMul", "Conv"}
    weight_param_names: set[str] = set()
    for node in quantsim_model.model.model.graph.node:
        if node.op_type in target_op_types and node.name not in lm_head_nodes:
            if len(node.input) >= 2:
                weight_param_names.add(node.input[1])

    for op_name, qc_op in quantsim_model.qc_quantize_op_dict.items():
        if op_name in weight_param_names:
            qc_op.reset_encoding_stats()
            qc_op.enabled = False


def _apply_block_granularity_to_decoder_stack(
    quantsim_model: QuantizationSimModel, precision: PrecisionConfig
):
    """Apply block-level granularity (LPBQ/BQ) to weight quantizers if configured."""
    block_prec = precision.blocks["default"]
    if block_prec.is_float:
        # FP weights — nothing to configure here.
        return
    if block_prec.granularity == Granularity.LPBQ:
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim_model,
            op_types=("Gemm", "MatMul", "Conv"),
            bitwidth=block_prec.qtype.bits,
            decompressed_bw=8,
            block_size=block_prec.block_size,
            nodes_to_exclude=_get_lm_head_node_names(quantsim_model),
        )
    elif block_prec.granularity == Granularity.BQ:
        set_blockwise_quantization_for_weights(
            sim=quantsim_model,
            op_types=("Gemm", "MatMul", "Conv"),
            bitwidth=block_prec.qtype.bits,
            symmetric=True,
            block_size=block_prec.block_size,
            nodes_to_exclude=_get_lm_head_node_names(quantsim_model),
        )


def _remove_activation_quantizers(quantsim_model: QuantizationSimModel):
    for op_name, qc_op in quantsim_model.qc_quantize_op_dict.items():
        if op_name in quantsim_model.activation_names:
            qc_op.reset_encoding_stats()
            qc_op.enabled = False
