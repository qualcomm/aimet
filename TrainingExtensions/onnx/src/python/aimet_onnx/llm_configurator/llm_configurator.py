# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
import itertools
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    set_param_type,
)
from aimet_onnx.defs import QSpec, qtype
from aimet_onnx.llm_topology import LlmTopology
from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.qc_quantize_op import QcQuantizeOp

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

_KV_CACHE_COMBINE_OPS = ("Concat", "ScatterElements")


def _get_quantizer_no_split_slice(
    quantsim_model: QuantizationSimModel, tensor_name: str
) -> QcQuantizeOp:
    """
    Returns closest enabled quantizer to tensor traversing upwards only through invariant ops and no Split/Slice

    :param tensor_name: Name of tensor for which to find quantizer
    """
    quantizer = quantsim_model.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return quantizer

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product == None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
            )
        else:
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
            )

    producer = product.producer

    if producer == None:
        return None

    if (
        not _is_grid_preserving_op(producer.type, domain=producer.domain)
        or producer.type == "Slice"
        or producer.type == "Split"
        or producer.type == "SplitToSequence"
    ):
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_quantizer_no_split_slice(quantsim_model, upstream_tensor.name)


def _set_matmul_second_input_to_8b(quantsim_model: QuantizationSimModel):
    cg = quantsim_model.connected_graph

    for op in reversed(cg.ordered_ops):
        if op.type != "MatMul":
            continue

        upper_quantizer = quantsim_model._get_enabled_quantizer(op.inputs[1].name)  # pylint: disable=protected-access

        enabled_quantizer = _get_quantizer_no_split_slice(
            quantsim_model, op.inputs[1].name
        )

        if enabled_quantizer and enabled_quantizer.bitwidth <= 8:
            continue
        elif enabled_quantizer:
            enabled_quantizer.set_bitwidth(8)
            enabled_quantizer.use_symmetric_encodings = True
        elif upper_quantizer:
            if op.inputs[1].name in quantsim_model.qc_quantize_op_dict:
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
            else:
                quantsim_model._insert_quantizer(op.inputs[1].name, is_param=False)  # pylint: disable=protected-access
                quantsim_model._rebuild_session()  # pylint: disable=protected-access
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True


def _get_all_downstream_kv_cache_ops(
    sim: QuantizationSimModel, tensor_name: str
) -> set:
    product = sim.connected_graph.get_product(tensor_name)
    downstream = set()
    for consumer in product.consumers:
        if consumer.type in _KV_CACHE_COMBINE_OPS:
            downstream.add(consumer)
            downstream.update(
                _get_all_downstream_kv_cache_ops(sim, consumer.outputs[0].name)
            )
        elif _is_grid_preserving_op(consumer.type, domain=consumer.domain):
            downstream.update(
                _get_all_downstream_kv_cache_ops(sim, consumer.outputs[0].name)
            )

    return downstream


def _get_all_upstream_kv_cache_ops(sim: QuantizationSimModel, tensor_name: str) -> set:
    product = sim.connected_graph.get_product(tensor_name)
    upstream = set()
    producer = product.producer
    if producer and producer.type in _KV_CACHE_COMBINE_OPS:
        upstream.add(producer)
    elif producer and (
        _is_grid_preserving_op(producer.type, domain=producer.domain)
        or producer.type == "Cast"
    ):
        upstream.update(_get_all_upstream_kv_cache_ops(sim, producer.inputs[0].name))

    return upstream


def _tie_quantizers_for_kv_cache(
    quantsim_model: QuantizationSimModel, kv_io_map: dict[str, str]
) -> None:
    quantizer_mapping = dict()

    for input_name, output_name in kv_io_map.items():
        quantizer = quantsim_model._get_enabled_quantizer(output_name)  # pylint: disable=protected-access
        if not quantizer:
            logger.warning(
                "Warning: No valid quantizer found for output %s", output_name
            )
            continue

        quantizer_mapping[input_name] = quantizer

        ops_to_tie = _get_all_upstream_kv_cache_ops(
            quantsim_model, output_name
        ) | _get_all_downstream_kv_cache_ops(quantsim_model, input_name)
        for kv_cache_op in ops_to_tie:
            for tensor in kv_cache_op.inputs + kv_cache_op.outputs:
                qtzr_name = quantsim_model._get_enabled_quantizer_name(tensor.name)  # pylint: disable=protected-access
                if qtzr_name:
                    quantizer_mapping[qtzr_name] = quantizer

    quantsim_model.set_quantizers(quantizer_mapping)


def _set_lm_head_to_8b(quantsim_model: QuantizationSimModel, lm_head_tensor_name: str):
    quantizer = quantsim_model.qc_quantize_op_dict.get(lm_head_tensor_name, None)
    if quantizer == None:
        raise KeyError(
            f"Could not find quantizer for LM head tensor: {lm_head_tensor_name}"
        )
    quantizer.set_bitwidth(8)
    quantizer._enable_blockwise_quantization(0)  # pylint: disable=protected-access
    quantizer.enable_per_channel_quantization()


def _set_tensor_to_8_bit_symmetric(
    quantsim_model: QuantizationSimModel, tensor_name: str
):
    quantizer = quantsim_model._get_enabled_quantizer(tensor_name)  # pylint: disable=protected-access
    if quantizer:
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True
    else:
        logger.warning("Warning: No valid quantizer found for output %s", tensor_name)


def _set_tensors_to_output_8b_sym(
    quantsim_model: QuantizationSimModel, out_tensors: list[str]
):
    for out_tensor in out_tensors:
        _set_tensor_to_8_bit_symmetric(quantsim_model, out_tensor)


def _apply_int8_kv_cache_tying_and_lm_head(
    sim: QuantizationSimModel, kv_io_map: dict[str, str], lm_head_tensor_name: str
):
    # Setting kv_cache and some other layers to 8-bit
    kv_io_list = list(kv_io_map.keys()) + list(kv_io_map.values())
    _set_tensors_to_output_8b_sym(sim, kv_io_list)

    # Setting the LM head weights to 8-bit.
    _set_lm_head_to_8b(
        sim,
        lm_head_tensor_name,
    )

    # Tie kv_cache
    _tie_quantizers_for_kv_cache(sim, kv_io_map)

    # Setting Matmul second input to 8b
    _set_matmul_second_input_to_8b(sim)

    return sim


def _collect_all_projections(topology: LlmTopology):
    projections = []
    for block in topology.blocks:
        projections.extend(block.qkv.linears)
        projections.extend(block.o_proj)
        projections.extend(block.gate_up.linears)
        projections.extend(block.down_proj)

    return projections


def _enabled_precisions(sim: QuantizationSimModel) -> dict[str, qtype]:
    """Returns the precision of every enabled quantizer, keyed by tensor name."""
    return {
        name: quantizer.precision()
        for name, quantizer in sim.qc_quantize_op_dict.items()
        if quantizer.enabled
    }


def _warn_on_overridden_precision_request(
    original: dict[str, qtype], requested: dict[str, qtype], final: dict[str, qtype]
):
    """
    Logs a warning for tensors whose requested precision was overridden by backend constraints.
    """
    overridden = {
        name: (requested[name], final[name])
        for name in requested
        if original.get(name) != requested[name] and final[name] != requested[name]
    }

    if overridden:
        logger.warning(
            "Backend constraints overrode the requested precision of %d tensor(s): %s."
            "\nPass a precision supported by the target backend to avoid this warning.",
            len(overridden),
            ", ".join(
                f"{name} ({req} -> {got})" for name, (req, got) in overridden.items()
            ),
        )


def configure_llm(
    sim: QuantizationSimModel,
    topology: LlmTopology,
    *,
    kv_cache_type: Optional[qtype | str] = None,
    backbone_weight_type: Optional[qtype | str | QSpec] = None,
    lm_head_weight_type: Optional[qtype | str | QSpec] = None,
):
    """
    Configures LLM QuantSim model precisions based on transformer topology and ties kv-cache quantizers.

    Args:
        sim (QuantizationSimModel): QuantSim to configure
        topology (LlmTopology): Extracted LLM topology.
        kv_cache_type: Quantization precision for key/value cache tensors. If None, left unchanged.
        backbone_weight_type: Quantization precision for all weighted MatMul,
            Gemm, and Conv layers in the backbone except the LM head. If None,
            left unchanged.
        lm_head_weight_type: Quantization precision for lm head weight. If None, left unchanged.
    """
    if len(topology.past_key_input_names) != len(topology.past_key_output_names):
        raise RuntimeError(
            "topology contains different number of key cache inputs and outputs."
        )

    if len(topology.past_value_input_names) != len(topology.past_value_output_names):
        raise RuntimeError(
            "topology contains different number of value cache inputs and outputs."
        )

    kv_cache_io = {
        inp: out
        for inp, out in zip(
            topology.past_key_input_names, topology.past_key_output_names
        )
    }
    kv_cache_io.update(
        {
            inp: out
            for inp, out in zip(
                topology.past_value_input_names, topology.past_value_output_names
            )
        }
    )

    if kv_cache_io:
        logger.info("Tying KV cache quantizers")
        _tie_quantizers_for_kv_cache(sim, kv_cache_io)

    original_precisions = _enabled_precisions(sim)

    if kv_cache_type is not None:
        logger.info("Setting KV cache quantizers to %s", str(kv_cache_type))
        all_kv_cache = list(itertools.chain(kv_cache_io.keys(), kv_cache_io.values()))
        sim.set_tensor_precision(all_kv_cache, kv_cache_type, strict=False)

    lm_head_layers = topology.lm_head
    if lm_head_layers and lm_head_weight_type is not None:
        logger.info("Setting lm head precision to %s", str(lm_head_weight_type))
        set_param_type(sim, lm_head_weight_type, nodes_to_include=lm_head_layers)

    if backbone_weight_type is not None:
        logger.info("Setting backbone weight types to %s", str(backbone_weight_type))
        set_param_type(
            sim,
            backbone_weight_type,
            op_types=("Gemm", "MatMul", "Conv"),
            nodes_to_exclude=set(lm_head_layers),
            strict=False,
        )

    requested_precisions = _enabled_precisions(sim)
    sim._apply_exception_rules()  # pylint: disable=protected-access
    _warn_on_overridden_precision_request(
        original_precisions, requested_precisions, _enabled_precisions(sim)
    )
