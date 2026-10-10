# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Callable, ContextManager, Tuple, List, Dict, Collection

import torch
import onnx
import onnx_ir
from onnx.utils import Extractor
from onnx2torch import convert
from onnx2torch.onnx_graph import OnnxGraph

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.quantsim import calculate_delta_offset
from aimet_onnx.adascale.quantizer import QuantizedLinear, QuantizedConv2d
from aimet_onnx.adascale.onnx2torch_ext import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx import ir_utils
from aimet_onnx.graph_passes.fusions import inline_all_supergroups

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)

filter_op = ["MatMul", "Conv", "Gemm"]


def _get_onnx_subgraph(
    extractor: Extractor,
    block_input_output_names: Tuple[List[str], List[str]],
):
    """
    Given a onnx block end points get onnx subgraph
    """
    block_input_names, block_output_names = block_input_output_names
    try:
        block_fp32_model = extractor.extract_model(
            block_input_names,
            block_output_names,
        )
        return block_fp32_model
    except Exception:
        raise RuntimeError(  # pylint: disable=raise-missing-from
            f"Unable to extract onnx subgraph for given block input/output {block_input_output_names}"
        )


def _get_onnx_block_info(onnx_subgraph: onnx_ir.Model):
    """
    For an onnx subgraph get onnx param name from initializer list map
    """
    graph = onnx_subgraph.graph
    name_to_node_filtered = {
        n.name: n for n in graph.all_nodes() if n.op_type in filter_op
    }
    node_name_to_onnx_param = {}
    for node in name_to_node_filtered.values():
        # TODO remove using "bias" word search and add op specific logic instead
        if node.op_type == "Conv":
            node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = node.inputs[
                1
            ].name
        else:
            for edge in node.inputs:
                if (
                    edge.name in onnx_subgraph.graph.initializers
                    and "bias" not in edge.name
                ):
                    # Bias will not be updated so we donot need to keep track of bias
                    node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = (
                        edge.name
                    )
    return node_name_to_onnx_param


def required_extra_block_inputs(
    graph: onnx_ir.Graph,
    input_names: List[str],
    output_names: List[str],
) -> List[str]:
    """
    Return graph-input names the subgraph requires beyond ``input_names``.

    Walks back from ``output_names`` with ``input_names`` as a barrier; any
    producer-less, non-initializer value reached is an unbounded graph input.
    """

    name_to_value = onnx_ir.convenience.create_value_mapping(graph)
    declared = {name_to_value[n] for n in input_names if n in name_to_value}
    visited_values = set(declared)
    visited_nodes = set()
    stack = [name_to_value[n] for n in output_names if n in name_to_value]
    extras: List[str] = []
    seen = set(input_names)

    while stack:
        value = stack.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        producer = value.producer()
        if producer is None:
            if not value.is_initializer() and value.name and value.name not in seen:
                extras.append(value.name)
                seen.add(value.name)
            continue
        if producer in visited_nodes:
            continue
        visited_nodes.add(producer)
        for inp in producer.inputs:
            if inp is None or inp in visited_values:
                continue
            stack.append(inp)

    # Normalize order to graph input order
    return [inp.name for inp in graph.inputs if inp.name in extras]


@contextlib.contextmanager
def _retarget_fp16_casts_to_bf16(model: onnx_ir.Model):
    """
    Temporarily retarget every ``Cast(to=FLOAT16)`` node in ``model`` to
    ``BFLOAT16``, then restore it to ``FLOAT16`` on exit.

    onnx2torch bakes each Cast node's target dtype into a fixed attribute
    on the converted torch module. AdaScale later upcasts fp16 blocks to
    bf16 for training (fp16 underflows Adam's second moment); if the Cast
    modules still targeted fp16, that fp16 tensor could leak into
    downstream plain elementwise ops that ``torch.autocast(bfloat16)``
    doesn't cover. Retargeting the ONNX nodes before conversion makes
    onnx2torch bake in bf16 directly, so no fp16 tensor is ever produced.
    A no-op for models with no FLOAT16 Cast nodes (e.g. fp32 models).
    """
    casts = [
        node
        for node in model.graph.all_nodes()
        if node.op_type == "Cast"
        and node.attributes.get("to") is not None
        and node.attributes["to"].value == onnx.TensorProto.FLOAT16
    ]
    for node in casts:
        node.attributes["to"] = onnx_ir.AttrInt64("to", onnx.TensorProto.BFLOAT16)
    try:
        yield
    finally:
        for node in casts:
            node.attributes["to"] = onnx_ir.AttrInt64("to", onnx.TensorProto.FLOAT16)


def get_pt_block(
    model: onnx_ir.Model, block_input_output_names: Tuple[List[str], List[str]]
):
    """
    Given a onnx block end points get a pytorch block
    :param model: onnx.ModelProto
    :param block_input_output_names: input/output names for block end points
    """
    input_names, output_names = block_input_output_names

    subgraph = onnx_ir.convenience.extract(
        model.graph,
        input_names,
        output_names,
    )
    subgraph_model = onnx_ir.Model(
        subgraph, ir_version=model.ir_version, functions=list(model.functions.values())
    )
    ir_utils.remove_aimet_quantizers(subgraph_model)
    inline_all_supergroups(subgraph_model)
    onnx_ir.passes.common.TopologicalSortPass().call(subgraph_model)
    onnx_ir.external_data.load_to_model(subgraph_model)
    param_map = _get_onnx_block_info(subgraph_model)
    with _retarget_fp16_casts_to_bf16(subgraph_model):
        pt_block = convert(onnx_ir.to_proto(subgraph_model))
    return pt_block, param_map


def upcast_fp16_block_to_bf16(
    pytorch_block: torch.nn.Module,
    device: torch.device,
    *input_lists: List[List[torch.Tensor]],
) -> Tuple[
    torch.nn.Module, List[List[List[torch.Tensor]]], Callable[[], ContextManager[None]]
]:
    """
    Train fp16 blocks in bf16 instead: fp16 underflows Adam's second moment
    for tiny per-block gradients (e.g. AdaScale's scale/gamma/beta params).

    If ``pytorch_block`` (as returned by :func:`get_pt_block`) is fp16,
    upcasts it to bf16 and returns new nested lists with every fp16 tensor
    in ``input_lists`` likewise upcast (the originals are left untouched --
    use the returned lists), along with an autocast context-manager factory
    to wrap forward passes in. ``get_pt_block()`` already retargets the
    block's ``Cast(FLOAT16)`` ONNX nodes to ``BFLOAT16`` before onnx2torch
    conversion, so this only needs to upcast the weight/input tensors
    themselves.

    No-op for non-fp16 blocks: returns ``pytorch_block``/``input_lists``
    unchanged and a no-op context factory, so callers can use the same
    ``with autocast_ctx():`` pattern regardless of the block's dtype.

    :param pytorch_block: block to upcast, as returned by ``get_pt_block()``
    :param device: device the block will run on
    :param input_lists: any number of ``List[List[torch.Tensor]]`` batches
        (e.g. fp and quantized calibration inputs) to upcast alongside it
    :return: ``(pytorch_block, upcasted_input_lists, autocast_ctx)``, where
        ``autocast_ctx`` is a zero-arg callable returning a fresh context
        manager each call (safe to enter more than once)
    """
    first_param = next((p for p in pytorch_block.parameters()), None)
    is_fp16 = first_param is not None and first_param.dtype == torch.float16

    if not is_fp16:

        @contextlib.contextmanager
        def _noop_ctx():
            yield

        return pytorch_block, list(input_lists), _noop_ctx

    pytorch_block = pytorch_block.to(dtype=torch.bfloat16)

    def _to_bf16(inp_list):
        return [
            [t.to(dtype=torch.bfloat16) if t.dtype == torch.float16 else t for t in one]
            for one in inp_list
        ]

    upcasted_input_lists = [_to_bf16(inp_list) for inp_list in input_lists]

    device_type = device.type if hasattr(device, "type") else "cuda"

    @contextlib.contextmanager
    def _autocast_ctx():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield

    return pytorch_block, upcasted_input_lists, _autocast_ctx


def _get_tensor_consumers(tensor: onnx_ir.Value):
    consumers = set()
    for consumer, _ in tensor.uses():
        if consumer.op_type in ("Identity", "QcQuantizeOp"):
            consumers.update(_get_tensor_consumers(consumer.outputs[0]))
            continue
        consumers.add(consumer)
    return consumers


def _should_transpose_weight(module: torch.nn.Module, weight: onnx_ir.Value):
    if not isinstance(module, torch.nn.Linear):
        return False

    def _is_transposed_weight(node: onnx_ir.Node):
        if node.op_type not in ("MatMul", "Gemm"):
            return False
        if node.op_type == "MatMul":
            return True

        trans_b = node.attributes.get("transB", 0)
        if trans_b:
            trans_b = trans_b.as_int()

        return not trans_b

    consumers = _get_tensor_consumers(weight)

    if not any(_is_transposed_weight(node) for node in consumers):
        return False

    if not all(_is_transposed_weight(node) for node in consumers):
        raise RuntimeError(f"Conflicting uses of {weight} by consumers {consumers}")

    return True


def copy_pt_weights_to_onnx(
    pt_block: torch.fx.GraphModule,
    onnx_model: onnx_ir.Model,
    param_map: Collection[Dict[str, str]],
    quantizer_dict: Dict[str, QcQuantizeOp] = None,
):
    """
    Given a pt_block with adascale params computed, copy the params to onnx model
    :param pt_block: pytorch block with adascale weight quantizers
    :param onnx_model: onnx model before adascale
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    :param quantizer_dict: Optional quantizer dict; params whose quantizer is
        disabled are skipped (e.g. LoRA params during base-model AdaScale).
    """
    for name, module in pt_block.named_modules():
        if param_map.get(name) is None:
            continue
        if quantizer_dict is not None and not quantizer_dict[param_map[name]].enabled:
            continue
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            _folded = (
                module.param_quantizers["weight"]
                .get_folded_weight(module.weight)
                .detach()
                .cpu()
            )
            # numpy has no native bfloat16, so .numpy() on a bf16 tensor
            # raises TypeError. When AdaScale ran with a bf16 master (fp16
            # ONNX model case), the folded weight is bf16 -- promote to
            # fp32 before numpy; the .astype(onnx_dtype) below then casts
            # it to the ONNX initializer dtype (fp16) as usual.
            if _folded.dtype == torch.bfloat16:
                _folded = _folded.float()
            pytorch_weight = _folded.numpy()
        else:
            _raw = module.weight.detach().cpu()
            if _raw.dtype == torch.bfloat16:
                _raw = _raw.float()
            pytorch_weight = _raw.numpy()

        onnx_tensor_name = param_map[name]
        onnx_param_tensor = onnx_model.graph.initializers[onnx_tensor_name]
        if _should_transpose_weight(module, onnx_param_tensor):
            pytorch_weight = pytorch_weight.T
        if tuple(pytorch_weight.shape) != tuple(onnx_param_tensor.const_value.shape):
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.const_value.shape}"
            )
        # Preserve the original initializer dtype so downstream ONNX consumers
        # (e.g. the activation sampler's ORT session) keep type-consistent
        # MatMul inputs.
        onnx_dtype = onnx_param_tensor.const_value.dtype.numpy()
        if pytorch_weight.dtype != onnx_dtype:
            pytorch_weight = pytorch_weight.astype(onnx_dtype)
        onnx_param_tensor.const_value = onnx_ir.Tensor(pytorch_weight)
        _logger.info(
            "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
            name,
            onnx_tensor_name,
        )


def copy_pt_encodings_to_sim(
    pt_block: torch.fx.GraphModule,
    quantizer_dict: Dict[str, QcQuantizeOp],
    pt_weights_to_onnx_initializers: Collection[Dict[str, str]],
):
    """
    Given the PT block with adascale params computed, copy the encodings to sim
    :param pt_block: pytorch block with adascale weight quantizers
    :param quantizer_dict: Dictionary of quantizers
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    """
    for name, module in pt_block.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            onnx_param_name = pt_weights_to_onnx_initializers[name]
            #### TODO Check the modules
            # copy encodings over to onnx quantizers
            new_min = module.param_quantizers["weight"].get_min().detach().cpu().numpy()
            new_max = module.param_quantizers["weight"].get_max().detach().cpu().numpy()

            enc = quantizer_dict[onnx_param_name].get_encodings()
            if enc is None:
                # quantizer is disabled (e.g. LoRA params skipped during AdaScale) — skip
                continue
            if len(new_min) != len(enc) or len(new_max) != len(enc):
                raise RuntimeError(
                    "Encodings of the onnx quantizer and adascale quantizer have different lengths"
                )

            expected_bw = module.param_quantizers["weight"].bitwidth
            for i, encoding in enumerate(enc):
                delta, offset = calculate_delta_offset(
                    min_val=new_min[i],
                    max_val=new_max[i],
                    bitwidth=expected_bw,
                    use_symmetric_encodings=True,
                    use_strict_symmetric=False,
                )
                # TODO: #6393 calculate_delta_offset to return float
                encoding.delta = delta.item()
                encoding.offset = offset.item()
                encoding.min = new_min[i].item()
                encoding.max = new_max[i].item()
                # Catches bitwidth-override regressions before they silently corrupt delta/offset.
                assert encoding.bw == expected_bw, (
                    f"Encoding bitwidth mismatch for {onnx_param_name}: "
                    f"encoding.bw={encoding.bw} but AdaScale QDQ bitwidth={expected_bw}"
                )
            quantizer_dict[onnx_param_name].load_encodings(enc)
            quantizer_dict[onnx_param_name].freeze_encodings()
