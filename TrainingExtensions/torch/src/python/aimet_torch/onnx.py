# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Defines onnx export API"""

import copy
import contextlib
import io
import itertools
import os
from packaging import version
import traceback
from typing import Any, Mapping, Tuple, Union, Literal
from pathlib import Path
import math

import numpy as np
import onnx
import torch
from torch.onnx import _constants

import aimet_torch
from aimet_torch.utils import patch_attr
from aimet_torch.common.onnx._utils import (
    _add_onnx_qdq_nodes,
    _convert_version,
    _derive_data_movement_op_encodings,
    _derive_const_rescale_op_output_encodings,
    contains_tensor_type,
    _remove_onnx_qdq_nodes,
    to_array,
)

from .nn import QuantizationMixin
from .quantization import DequantizedTensor
from .quantization.base import EncodingBase
from .quantization.affine import AffineQuantizerBase, AffineEncoding
from .quantization.float import FloatQuantizeDequantize, FloatEncoding
from .quantization.float._finfo import _finfo
from .quantsim import QuantizationSimModel
from .experimental import onnx as _onnx
from .experimental.onnx._export import _get_all_constants

from onnx.external_data_helper import uses_external_data, set_external_data


_TORCH_VERSION = version.parse(torch.__version__)
_TORCH_DEFAULT_OPSET = _constants.ONNX_DEFAULT_OPSET
_TORCH_MIN_OPSET = _constants.ONNX_MIN_OPSET
_TORCH_MAX_OPSET = _constants.ONNX_MAX_OPSET

# Allow at least up to opset 21 to enable [u]int16 QDQ export
_AIMET_MAX_OPSET = max(_TORCH_MAX_OPSET, 25)


def _offload_tensor_to_external_data(tensor: onnx.TensorProto, base_dir: str):
    """
    Move a TensorProto's inline ``raw_data`` to an external file (in place), so it
    doesn't count toward protobuf's 2 GiB per-message ceiling (which large weights
    overflow on ``graph.initializer.append``, individually or cumulatively).

    set_external_data alone is not enough: raw_data must be written and cleared
    before append, not deferred to onnx.save, because the in-memory protobuf message
    accumulates all raw_data bytes and hits the ceiling during append itself.
    """
    if not tensor.HasField("raw_data"):
        return
    raw = bytes(tensor.raw_data)
    location = tensor.name.replace("/", "_")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(base_dir, location), "wb") as f:
        f.write(raw)
    set_external_data(tensor, location=location, offset=0, length=len(raw))
    tensor.ClearField("raw_data")


@torch.no_grad()
def export(
    model: Union[torch.nn.Module, QuantizationSimModel],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    *,
    export_int32_bias: bool = True,
    prequantize_constants: bool = False,
    force_activation_as: Literal["unsigned"] | Literal["signed"] | None = "unsigned",
    **kwargs,
):
    """
    Export :class:`QuantizationSimModel` to onnx model with
    onnx `QuantizeLinear`_ and `DequantizeLinear`_ embedded in the graph.

    This function takes the same set of arguments as `torch.onnx.export()`_

    Args:
        model: The model to be exported
        args: Same as `torch.onnx.export()`
        f: Same as `torch.onnx.export()`
        export_int32_bias (bool, optional):
            If true, generate and export int32 bias encoding on the fly (default: `True`).
        prequantize_constants (bool, optional):
            If True, quantized weights will be represented as quantized weight followed by DequantizeLinear.
            If False, they will be represented as floating point weight followed by
            QuantizeLinear and DequantizeLinear (default: `False`).
        force_activation_as (str, optional):
            Force representing quantized activations as signed or unsigned integers (default: `"unsigned"`)
        **kwargs: Same as `torch.onnx.export()`


    .. note::
        For robustness, onnx >=1.19 is highly recommended with this API,
        especially when exporting large models (>2GB).
        This is due to a known bug in onnx <1.19 version converter.
        For more information, see https://github.com/onnx/onnx/issues/6529

    .. _torch.onnx.export(): https://docs.pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
    .. _QuantizeLinear: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
    .. _DequantizeLinear: https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

    Examples:

        >>> aimet_torch.onnx.export(sim.model, x, f="model.onnx",
        ...                         input_names=["input"], output_names=["output"],
        ...                         opset_version=21, dynamo=False,
        ...                         export_int32_bias=True)
        ...
        >>> import onnxruntime as ort
        >>> options = ort.SessionOptions()
        >>> options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        >>> sess = ort.InferenceSession("model.onnx", sess_options=options)
        >>> onnx_output, = sess.run(None, {"input": x.detach().numpy()})
        >>> torch.nn.functional.cosine_similarity(torch.from_numpy(onnx_output), sim.model(x))
        tensor([1.0000, 0.9999, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
               grad_fn=<AliasBackward0>)

    .. image:: ../../images/conv_qdq.onnx.svg
        :align: center
    """
    if isinstance(model, QuantizationSimModel):
        model = model.model

    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"aimet_torch.export only supports torch.nn.Module or QuantizationSimModel; got {type(model)}"
        )

    base_dir = str(Path(str(f)).absolute().parent)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    _check_opset_version(kwargs)
    _check_unsupported_args(model, force_activation_as, kwargs)
    _check_non_standard_quantizer(model)

    target_version = kwargs.pop("opset_version", _TORCH_DEFAULT_OPSET)
    kwargs["opset_version"] = min(target_version, _TORCH_MAX_OPSET)

    _assert_minimum_required_opset(model, target_version)

    with contextlib.ExitStack() as stack:
        # Unfold all param quantizers to incorporate QuantizeLinear/DequantizeLinear
        # of those parameters in tracing time
        stack.enter_context(_temporarily_unfold_param_quantizers(model))

        if export_int32_bias:
            # Temoprarily instantiate int32 bias quantizers
            stack.enter_context(
                _concretize_int32_bias_quantizers(model, args, kwargs.get("kwargs"))
            )

        # Export quantize-dequantized weight
        # pylint: disable=protected-access
        if force_activation_as in ("unsigned", "signed"):
            signed = force_activation_as == "signed"
            stack.enter_context(
                _temporarily_convert_activation_to(model, signed=signed)
            )

        stack.enter_context(QuantizationSimModel._apply_qdq_to_model_parameters(model))

        # Remove [b]float16 quantizers
        stack.enter_context(_remove_fp16_quantizers(model))
        stack.enter_context(_remove_fp16_quantized_parameters(model))

        onnx_model, tensor_to_encoding_map = _to_onnx(
            model, args, f, **kwargs, propagate_int32_encodings=False
        )

    if _TORCH_MAX_OPSET < target_version:
        try:
            onnx_model = _convert_version(onnx_model, target_version)
        except Exception as e:  # pylint: disable=broad-exception-caught
            f = io.StringIO()
            traceback.print_exc(file=f)
            reason = _why_do_i_need_opset21(model)

            if reason:
                detail = (
                    f"torch.onnx.export only supports opset<={_TORCH_MAX_OPSET}, "
                    f"but onnx::QuantizeLinear requires opset>={target_version} for {reason}. "
                    "As a workaround, we tried to torch.onnx.export your model "
                    f"with opset={_TORCH_MAX_OPSET} and convert the onnx model to {target_version}, "
                    "but failed with the following error:\n\n"
                )
            else:
                detail = "\n\n"

            msg = (
                f"Failed to convert onnx model to {target_version} due to {type(e).__name__}. {detail}"
                "==============================================================\n"
                f"{f.getvalue()}"
                "==============================================================\n\n"
            )
            raise RuntimeError(msg) from e

    onnx_qdq_model = _to_onnx_qdq(
        onnx_model,
        tensor_to_encoding_map,
        prequantize_constants=prequantize_constants,
        base_dir=base_dir,
    )
    _remove_intermediate_identity_nodes(onnx_qdq_model, base_dir=base_dir)
    _remove_dangling_nodes_and_initializers(onnx_qdq_model)

    # Add metadata property to indicate the model is exported by AIMET and its version
    prop = onnx_qdq_model.metadata_props.add()
    prop.key = "producer"
    prop.value = f"aimet-torch {aimet_torch.__version__}"
    has_external_data = any(uses_external_data(i) for i in onnx_model.graph.initializer)
    onnx.save(
        onnx_qdq_model,
        f,
        save_as_external_data=has_external_data
        or onnx_qdq_model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF,
    )


def _why_do_i_need_opset21(model: torch.nn.Module) -> str:
    int4 = False
    int16 = False
    bq = False

    for qtzr in model.modules():
        if not isinstance(qtzr, AffineQuantizerBase):
            continue

        if qtzr.block_size is not None:
            bq = True

        if qtzr.bitwidth == 4:
            int4 = True

        if qtzr.bitwidth == 16:
            int16 = True

    reasons = []

    if int4 or int16:
        reasons.append("int4/int16 quantization")

    if bq:
        reasons.append("blockwise quantization")

    if not reasons:
        return ""  # This should never happen

    if len(reasons) == 1:
        return reasons[0]

    return f"{reasons[0]} and {reasons[1]}"


def _assert_minimum_required_opset(model: torch.nn.Module, target_opset: int):
    if target_opset < 21 and any(
        qtzr.block_size is not None
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
            f" got opset={target_opset}"
        )

    if target_opset < 21 and any(
        qtzr.bitwidth in (4, 16)
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with INT4/INT16 are only supported in opset >= 21;"
            f" got opset={target_opset}"
        )

    if target_opset < 13 and any(
        tuple(qtzr.shape)
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with per-channel are only supported in opset >= 13;"
            f" got opset={target_opset}"
        )

    if target_opset < 10:
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear are only supported in opset >= 10;"
            f" got opset={target_opset}"
        )


def _check_opset_version(kwargs):
    opset_version = kwargs.get("opset_version", _TORCH_DEFAULT_OPSET)

    if not (_TORCH_MIN_OPSET <= opset_version <= _AIMET_MAX_OPSET):
        raise ValueError(f"Unsupported ONNX opset version: {opset_version}")


def _check_unsupported_args(model, force_activation_as, kwargs):
    if _TORCH_VERSION >= version.parse("2.0.0") and isinstance(
        model,
        torch._dynamo.OptimizedModule,  # pylint: disable=protected-access
    ):
        if _TORCH_VERSION < version.parse("2.11.0.dev"):
            raise RuntimeError(
                "Exporting a torch.compile-d quantsim model is only supported in torch >= 2.11.0. "
                "For more information, see https://github.com/pytorch/pytorch/issues/171674"
            )

    if force_activation_as not in ("unsigned", "signed", None):
        raise ValueError(
            f"force_activation_as must be either 'unsigned', 'signed' or None; got {force_activation_as}"
        )

    dynamo = kwargs.get("dynamo", _TORCH_VERSION >= version.parse("2.9.0"))

    if dynamo and _TORCH_VERSION < version.parse("2.8.0"):
        raise RuntimeError("AIMET dynamo export is only supported in torch >= 2.8.0")

    export_params = kwargs.get("export_params", True)

    if not export_params:
        raise NotImplementedError("export_params=False is not supported yet")

    keep_initializers_as_inputs = kwargs.get("keep_initializers_as_inputs", False)

    if keep_initializers_as_inputs:
        raise NotImplementedError(
            "keep_initializers_as_inputs=True is not supported yet"
        )

    do_constant_folding = kwargs.get("do_constant_folding", True)

    if not do_constant_folding:
        raise NotImplementedError("do_constant_folding=False is not supported yet")

    export_modules_as_functions = kwargs.get("export_modules_as_functions", False)

    if export_modules_as_functions:
        raise RuntimeError("export_modules_as_functions=True is not supported")

    operator_export_type = kwargs.get(
        "operator_export_type", torch.onnx.OperatorExportTypes.ONNX
    )

    if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN:
        raise RuntimeError(
            "operator_export_type=OperatorExportTypes.ONNX_ATEN is not supported"
        )


def _check_non_standard_quantizer(model: torch.nn.Module):
    supported_bitwdiths = (2, 4, 8, 16, 32)

    for name, qtzr in model.named_modules():
        if not isinstance(qtzr, AffineQuantizerBase):
            continue

        if qtzr.bitwidth not in supported_bitwdiths:
            supported_bitwdiths = "/".join(str(b) for b in supported_bitwdiths)
            raise RuntimeError(
                f"torch.onnx.export only supports {supported_bitwdiths}-bit integers; "
                f"got '{name}' with bitwidth={qtzr.bitwidth}"
            )


def _is_aimet_qdq_node(node: onnx.NodeProto) -> bool:
    return (node.domain, node.op_type) in (
        ("aimet", "quantize_dequantize"),
        ("aimet", "QuantizeDequantize"),
        ("aimet", "FloatQuantizeDequantize"),
    )


def _duplicate_shared_qdq_inputs(
    onnx_model: onnx.ModelProto, base_dir: str | None
) -> dict[str, str]:
    """
    Duplicate input tensors associated with multiple QDQ nodes to avoid name collision
    For example,

    Before: One tensor associated with multiple encodings

        "conv1.weight" -+--------------> QDQ -> Conv
                        |
                        +--------------> QDQ -> Conv

    After:
      Case 1. Insert Identity nodes for each QDQ if QDQ nodes have different encodings

        "conv1.weight" -+--------------> QDQ -> Conv
                        |
                        +-> Identity --> QDQ -> Conv
                                      ↑
                              "conv1.weight_dup_0"

      Case 2. Otherwise, consolidate multiple QDQs into single QDQ

        "conv1.weight" -> QDQ -+--------------> Conv
                               |
                               +--------------> Conv
    """
    consumers: dict[str, list[onnx.NodeProto]] = {}
    aliases: dict[str, str] = {}
    qdq_nodes_with_shared_input: dict[str, list[onnx.NodeProto]] = {}

    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    for identity in onnx_model.graph.node:
        if identity.op_type == "Identity":
            output = identity.output[0]
            inp = aliases.get(identity.input[0], identity.input[0])
            aliases[output] = inp

    for node in onnx_model.graph.node:
        if not _is_aimet_qdq_node(node):
            continue
        input_name = aliases.get(node.input[0], node.input[0])
        qdq_nodes_with_shared_input.setdefault(input_name, []).append(node)

    aliases: dict[str, str] = {}
    producers: dict[str, onnx.NodeProto] = {}
    constants = _get_all_constants(onnx_model, consumers)

    for input_name, qdq_nodes in qdq_nodes_with_shared_input.items():
        for i, qdq_node in enumerate(qdq_nodes[1:]):
            if _encoding_equal(qdq_node, qdq_nodes[0], constants, base_dir):
                # If multiple QDQ nodes have the same encoding,
                # we can reuse the same QDQ's output for multiple consumers.
                for consumer in consumers.get(qdq_node.output[0], []):
                    for j, inp in enumerate(consumer.input):
                        if inp == qdq_node.output[0]:
                            consumer.input[j] = qdq_nodes[0].output[0]
            else:
                # Duplicate QDQ nodes for each usage
                # Create an Identity node to duplicate the tensor
                alias = f"{input_name}_dup_{i}"
                aliases[alias] = input_name
                identity_node = onnx.helper.make_node(
                    "Identity",
                    inputs=[input_name],
                    outputs=[alias],
                    name=f"Identity_{input_name}_dup_{i}",
                )
                qdq_node.input[0] = identity_node.output[0]
                producers[identity_node.output[0]] = identity_node

        other_consumers = [
            node
            for node in consumers.get(qdq_nodes[0].input[0], [])
            if not _is_aimet_qdq_node(node) and node.op_type != "Shape"
        ]

        # If there are non-QDQ consumers add Identity for qdq_nodes[0]
        if other_consumers:
            alias = f"{input_name}_dup_{len(qdq_nodes) - 1}"
            aliases[alias] = input_name
            identity_node = onnx.helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[alias],
                name=f"Identity_{alias}",
            )
            qdq_nodes[0].input[0] = identity_node.output[0]
            producers[identity_node.output[0]] = identity_node

    # Insert Identity nodes to the graph in topological order
    all_nodes = []
    for node in onnx_model.graph.node:
        if node.input and node.input[0] in producers:
            identity_node = producers[node.input[0]]
            all_nodes.append(identity_node)
        all_nodes.append(node)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(all_nodes)

    return aliases


def _remove_dangling_nodes_and_initializers(onnx_model: onnx.ModelProto):
    """
    Remove nodes and initializers that are not connected to any output.
    """
    producers: dict[str, onnx.NodeProto] = {
        out: node for node in onnx_model.graph.node for out in node.output
    }

    graph_outputs = {output.name for output in onnx_model.graph.output}
    queue = list(graph_outputs)
    reachable = set(queue)

    while queue:
        tensor_name = queue.pop()
        producer = producers.get(tensor_name, None)

        if not producer:
            continue

        for inp in producer.input:
            if inp not in reachable:
                reachable.add(inp)
                queue.append(inp)

    all_nodes = [
        node
        for node in onnx_model.graph.node
        if any(output in reachable for output in node.output)
    ]
    all_initializers = [
        initializer
        for initializer in onnx_model.graph.initializer
        if initializer.name in reachable
    ]

    onnx_model.graph.ClearField("node")
    onnx_model.graph.ClearField("initializer")
    onnx_model.graph.node.extend(all_nodes)
    onnx_model.graph.initializer.extend(all_initializers)


def _decouple_back_to_back_qdqs(onnx_model: onnx.ModelProto) -> dict[str, str]:
    """
    Insert Identity node between back-to-back QDQ nodes to avoid name collision
    when one tensor is associated with multiple encodings.
    For example,

    Before: One tensor associated with multiple encodings

        "conv1.weight" -> QDQ -------------> QDQ -> Conv
                       (float4)            (int8)

    After: Duplicate QDQ nodes for each usage

        "conv1.weight" -> QDQ -> Identity -> QDQ -> Conv
                       (float4)            (int8)
    """
    producers: dict[str, onnx.NodeProto] = {
        node.output[0]: node for node in onnx_model.graph.node
    }

    qdq_nodes = [
        node
        for node in onnx_model.graph.node
        if f"{node.domain}::{node.op_type}"
        in (
            "aimet::quantize_dequantize",
            "aimet::QuantizeDequantize",
            "aimet::FloatQuantizeDequantize",
        )
    ]
    new_identity_nodes: dict[str, onnx.NodeProto] = {}

    for qdq_node in qdq_nodes:
        producer = producers.get(qdq_node.input[0], None)

        if not producer:
            continue

        if f"{producer.domain}::{producer.op_type}" not in (
            "aimet::quantize_dequantize",
            "aimet::QuantizeDequantize",
            "aimet::FloatQuantizeDequantize",
        ):
            continue

        identity_node = onnx.helper.make_node(
            "Identity",
            inputs=[qdq_node.input[0]],
            outputs=[f"{qdq_node.input[0]}_alias"],
            name=f"Identity_{qdq_node.input[0]}_alias",
        )
        qdq_node.input[0] = identity_node.output[0]
        new_identity_nodes[identity_node.input[0]] = identity_node

    # Insert Identity nodes to the graph in topological order
    all_nodes = []
    for node in onnx_model.graph.node:
        all_nodes.append(node)
        if node.output[0] in new_identity_nodes:
            identity_node = new_identity_nodes[node.output[0]]
            all_nodes.append(identity_node)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(all_nodes)


def _remove_unsafe_barrier(onnx_model, tensor_to_encoding_map):
    """
    Remove _UnsafeBarrier nodes temporarily inserted by AIMET to suppress
    encoding propagation across supergroup boundaries and re-map output encodings
    of unsafe barrier nodes to those of their producer's outputs.
    """
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}

    for node in onnx_model.graph.node:
        producers[node.output[0]] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    for node in onnx_model.graph.node:
        if (node.domain, node.op_type) != ("aimet", "_UnsafeBarrier"):
            continue

        producer_output_tensor = node.input[0]
        producer = producers.get(node.input[0], None)

        for consumer in consumers.get(node.output[0], []):
            for i, inp in enumerate(consumer.input):
                if inp == node.output[0]:
                    consumer.input[i] = node.input[0]

        if producer:
            for output in onnx_model.graph.output:
                if output.name == node.output[0]:
                    for i, _ in enumerate(producer.output):
                        if producer.output[i] == node.input[0]:
                            producer.output[i] = producer_output_tensor = output.name

        # Re-map output encoding of unsafe_barrier op to its producer output tensor
        if node.output[0] in tensor_to_encoding_map:
            tensor_to_encoding_map[producer_output_tensor] = tensor_to_encoding_map.pop(
                node.output[0]
            )

    all_nodes = [
        node
        for node in onnx_model.graph.node
        if (node.domain, node.op_type) != ("aimet", "_UnsafeBarrier")
    ]
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(all_nodes)


def _remove_intermediate_identity_nodes(
    onnx_model: onnx.ModelProto, base_dir: str | None
):
    """
    Remove temporarily added Identity nodes between DequantizeLinear and QuantizeLinear nodes.

    Before:
        (weight) -> Q -> DQ -> Identity -> Q -> DQ -> MatMul

    After:
        (weight) ------------------------> Q -> DQ -> MatMul  (if back-to-back QDQs are redundant)

        (weight) -> Q -> DQ -------------> Q -> DQ -> MatMul  (otherwise)
    """
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}

    for node in onnx_model.graph.node:
        producers[node.output[0]] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    constants = _get_all_constants(onnx_model, consumers)

    for identity in onnx_model.graph.node:
        if identity.op_type != "Identity":
            continue

        producer = producers.get(identity.input[0], None)
        if not (producer and producer.op_type == "DequantizeLinear"):
            continue

        if len(consumers.get(identity.output[0], [])) != 1:
            continue

        (consumer,) = consumers[identity.output[0]]

        if consumer.op_type != "QuantizeLinear":
            continue

        # If previous Q and next Q doesn't share the same encoding:
        #                     ┌──────────┐
        # (weight) -> Q -> DQ ┘ Identity └ Q -> DQ -> ...
        prev_dq = producer
        next_q = consumer
        next_q.input[0] = prev_dq.output[0]

        # If previous Q and next Q share the same encoding:
        #          ┌─────────────────────┐
        # (weight) ┘ Q -> DQ -> Identity └ Q -> DQ -> ...
        prev_q = producers.get(prev_dq.input[0], None)
        if (
            prev_q
            and prev_q.op_type == "QuantizeLinear"
            and _onnx_encoding_equal(prev_q, next_q, constants, base_dir=base_dir)
        ):
            next_q.input[0] = prev_q.input[0]


def _onnx_encoding_equal(
    q_1: onnx.NodeProto,
    q_2: onnx.NodeProto,
    constants: dict[str, onnx.TensorProto],
    base_dir: str | None,
):
    for i, qdq in enumerate([q_1, q_2]):
        if qdq.op_type == "QuantizeLinear":
            continue

        raise ValueError(
            f"Expected input {i} to be QuantizeLinear or DequantizeLinear node; "
            f"got {qdq.domain}::{qdq.op_type}",
        )

    def get_attributes(qdq_node: onnx.NodeProto):
        axis = block_size = output_dtype = None

        for attr in qdq_node.attribute:
            if attr.name == "axis":
                axis = attr.i
            if attr.name == "block_size":
                block_size = attr.i
            if attr.name == "output_dtype":
                output_dtype = attr.i

        return axis, block_size, output_dtype

    def get_scale(qdq_node: onnx.NodeProto) -> np.ndarray:
        scale_proto = constants.get(qdq_node.input[1])
        if scale_proto is None:
            raise RuntimeError(
                f"Cannot find constant with name {qdq_node.input[1]} in onnx model"
            )
        return to_array(scale_proto, base_dir=base_dir)

    def get_zero_point(qdq_node: onnx.NodeProto) -> np.ndarray | None:
        if len(qdq_node.input) < 3:
            return None

        zp_proto = constants.get(qdq_node.input[2])

        if zp_proto is None:
            raise RuntimeError(
                f"Cannot find constant with name {qdq_node.input[2]} in onnx model"
            )

        zp = to_array(zp_proto, base_dir=base_dir)

        if np.all(zp == 0):
            zp = None

        return zp

    try:
        return bool(
            get_attributes(q_1) == get_attributes(q_2)
            and np.all(get_scale(q_1) == get_scale(q_2))
            and np.all(get_zero_point(q_1) == get_zero_point(q_2))
        )
    except RuntimeError:
        return False


def _encoding_equal(
    qdq_1: onnx.NodeProto,
    qdq_2: onnx.NodeProto,
    constants: dict[str, onnx.TensorProto],
    base_dir: str | None,
) -> bool:
    for i, qdq in enumerate([qdq_1, qdq_2]):
        if _is_aimet_qdq_node(qdq):
            continue

        raise ValueError(
            f"Expected input {i} to be one of aimet::quantize_dequantize, ",
            "aimet::QuantizeDequantize or aimet::FloatQuantizeDequantize node; "
            f"got {qdq.domain}::{qdq.op_type}",
        )

    def get_attributes(qdq_node: onnx.NodeProto):
        finfo = None
        qmin = None
        qmax = None
        block_size = None
        zero_point_shift = None

        for attr in qdq_node.attribute:
            if attr.name == "qmin":
                qmin = attr.i
            if attr.name == "qmax":
                qmax = attr.i
            if attr.name == "block_size":
                block_size = tuple(attr.ints) or None
            if attr.name == "zero_point_shift":
                zero_point_shift = attr.f
            if attr.name == "dtype":
                finfo = _finfo.from_onnx_dtype(attr.i)

        if finfo:
            dtype = finfo
        else:
            dtype = (qmin, qmax)

        return dtype, block_size, zero_point_shift

    def get_scale(qdq_node: onnx.NodeProto) -> np.ndarray:
        scale_proto = constants.get(qdq_node.input[1])
        if scale_proto is None:
            raise RuntimeError(
                f"Cannot find constant with name {qdq_node.input[1]} in onnx model"
            )
        return to_array(scale_proto, base_dir=base_dir)

    def get_offset(qdq_node: onnx.NodeProto) -> np.ndarray | None:
        if len(qdq_node.input) < 3:
            return None

        offset_proto = constants.get(qdq_node.input[2])

        if offset_proto is None:
            raise RuntimeError(
                f"Cannot find constant with name {qdq_node.input[2]} in onnx model"
            )

        offset = to_array(offset_proto, base_dir=base_dir)

        if np.all(offset == 0):
            offset = None

        return offset

    try:
        return bool(
            get_attributes(qdq_1) == get_attributes(qdq_2)
            and np.all(get_scale(qdq_1) == get_scale(qdq_2))
            and np.all(get_offset(qdq_1) == get_offset(qdq_2))
        )
    except RuntimeError:
        return False


def _remove_redundant_qdqs(onnx_model: onnx.ModelProto, base_dir):
    constants: dict[str, onnx.TensorProto]
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}

    for node in onnx_model.graph.node:
        producers[node.output[0]] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    constants = _get_all_constants(onnx_model, consumers)

    qdq_nodes = [
        node
        for node in onnx_model.graph.node
        if f"{node.domain}::{node.op_type}"
        in (
            "aimet::quantize_dequantize",
            "aimet::QuantizeDequantize",
            "aimet::FloatQuantizeDequantize",
        )
    ]

    for qdq_node in qdq_nodes:
        prev_qdq = producers.get(qdq_node.input[0], None)

        if not prev_qdq:
            continue

        if f"{prev_qdq.domain}::{prev_qdq.op_type}" not in (
            "aimet::quantize_dequantize",
            "aimet::QuantizeDequantize",
            "aimet::FloatQuantizeDequantize",
        ):
            continue

        if not _encoding_equal(qdq_node, prev_qdq, constants, base_dir):
            continue

        # Redirect consumers of redundant QDQ nodes to the previous QDQ nodes
        for consumer in consumers.get(qdq_node.output[0], []):
            for i, inp in enumerate(consumer.input):
                if inp == qdq_node.output[0]:
                    consumer.input[i] = qdq_node.input[0]


def _fold_linear_weight_transpose(onnx_model: onnx.ModelProto, base_dir):
    """
    Fold weight transpose of F.linear

    Before:
            W -------> QDQ -> Transpose -> ...
      (Cout x Cin)   (axis=0)
    After:
           W_t ------> QDQ -> ...
      (Cin x Cout)   (axis=1)
    """
    producers: dict[str, onnx.NodeProto] = {
        node.output[0]: node for node in onnx_model.graph.node
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    all_tensor_names = set()

    def _follow_identity_chain(name: str) -> str:
        producer = producers.get(name, None)
        while producer and producer.op_type == "Identity":
            name = producer.input[0]
            producer = producers.get(name, None)
        return name

    for node in onnx_model.graph.node:
        if node.op_type == "Identity":
            continue
        for inp in node.input:
            inp = _follow_identity_chain(inp)
            consumers.setdefault(inp, []).append(node)

        all_tensor_names.update(node.input)
        all_tensor_names.update(node.output)

    constants: dict[str, onnx.TensorProto] = _get_all_constants(onnx_model, consumers)
    for node in onnx_model.graph.node:
        if node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
    all_tensor_names.update(constants.keys())

    qdq_nodes = {
        node.name: node
        for node in onnx_model.graph.node
        if (node.domain, node.op_type)
        in (
            ("aimet", "quantize_dequantize"),
            ("aimet", "QuantizeDequantize"),
            ("aimet", "FloatQuantizeDequantize"),
        )
    }

    def get_constant(name: str) -> onnx.TensorProto | None:
        name = _follow_identity_chain(name)
        return constants.get(name, None)

    def get_new_name(base_name: str) -> str:
        i = 0
        new_name = f"{base_name}_{i}"
        while new_name in all_tensor_names:
            i += 1
            new_name = f"{base_name}_{i}"
        all_tensor_names.add(new_name)
        return new_name

    def transpose_2d(tensor: onnx.TensorProto) -> onnx.TensorProto:
        assert len(tensor.dims) <= 2
        transposed = onnx.numpy_helper.from_array(
            to_array(tensor, base_dir=base_dir).T,
            name=get_new_name(tensor.name),
        )
        transposed.dims[:] = [
            *reversed(tensor.dims),
            *(itertools.repeat(1, 2 - len(tensor.dims))),
        ]
        # `from_array` above always materializes the transposed weight inline as
        # raw_data, even when the source weight was stored as external data. Each
        # such re-inlined weight is then appended to the graph, and their combined
        # size can exceed protobuf's hard 2 GiB per-message ceiling -- crashing
        # graph.initializer.append with a DecodeError (e.g. a large multi-layer
        # backbone, where no single weight is near the limit but the sum is).
        # So if the source was external, offload the transposed copy back to
        # external data to keep the in-memory message small.
        if uses_external_data(tensor):
            _offload_tensor_to_external_data(transposed, base_dir)
        return transposed

    aliases = {}
    visited = set()

    for weight in list(constants.values()):
        if weight.name in visited:
            continue
        visited.add(weight.name)
        weight_consumers = consumers.get(weight.name)

        if not weight_consumers:
            continue

        qdq_transpose_pairs: list[tuple[onnx.NodeProto, onnx.NodeProto]] = []

        for qdq in weight_consumers:
            if not (
                # Weight should be consumed by QDQ
                qdq.name in qdq_nodes
                # Weight should be 1st input of QDQ
                and weight is get_constant(qdq.input[0])
                # Scale should be constant
                and get_constant(qdq.input[1])
                # Offset (if exists) should be constant
                and (
                    qdq.op_type == "FloatQuantizeDequantize"
                    or get_constant(qdq.input[2])
                )
            ):
                continue

            for transpose in consumers.get(qdq.output[0], []):
                if transpose.op_type != "Transpose":
                    continue

                perm = None
                for attr in transpose.attribute:
                    if attr.name == "perm":
                        perm = tuple(attr.ints)

                if perm and perm != (1, 0):
                    continue

                if any(
                    consumer.op_type
                    in (
                        "quantize_dequantize",
                        "QuantizeDequantize",
                        "FloatQuantizeDequantize",
                    )
                    for consumer in consumers.get(transpose.output[0], [])
                ):
                    continue

                qdq_transpose_pairs.append((qdq, transpose))

        if not qdq_transpose_pairs:
            continue

        weight_t = transpose_2d(weight)
        onnx_model.graph.initializer.append(weight_t)
        constants[weight_t.name] = weight_t
        aliases[weight_t.name] = weight.name

        replaced_nodes: dict[str, onnx.NodeProto] = {}
        for qdq, transpose in qdq_transpose_pairs:
            qdq_copy = copy.deepcopy(qdq)
            qdq_copy.input[0] = weight_t.name
            qdq_copy.output[0] = transpose.output[0]
            qdq_copy.name = f"{qdq.name}_copy"
            replaced_nodes[transpose.name] = qdq_copy

            scale = get_constant(qdq.input[1])
            offset = (
                get_constant(qdq.input[2])
                if qdq.op_type in ("quantize_dequantize", "QuantizeDequantize")
                else None
            )

            if math.prod(scale.dims) > 1:
                scale_t = transpose_2d(scale)
                onnx_model.graph.initializer.append(scale_t)
                qdq_copy.input[1] = scale_t.name
                constants[scale_t.name] = scale_t

            if offset is not None and math.prod(offset.dims) > 1:
                offset_t = transpose_2d(offset)
                onnx_model.graph.initializer.append(offset_t)
                qdq_copy.input[2] = offset_t.name
                constants[offset_t.name] = offset_t

            block_size = next(
                (attr for attr in qdq_copy.attribute if attr.name == "block_size"), None
            )
            if block_size and block_size.ints:
                # Transpose block_size in-place
                block_size.ints[:] = block_size.ints[::-1]

        all_nodes = {}
        for node in onnx_model.graph.node:
            node = replaced_nodes.get(node.name, node)
            all_nodes[node.name] = node

        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(all_nodes.values())

    return aliases


def _to_onnx(
    model: torch.nn.Module,
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    *,
    propagate_int32_encodings: bool,
    **kwargs,
) -> Tuple[onnx.ModelProto, dict]:
    # pylint: disable=protected-access
    base_dir = str(Path(str(f)).absolute().parent)
    _onnx.export(model, args, f, **kwargs)
    onnx_model = onnx.load(f, load_external_data=False)
    aliases = _fold_linear_weight_transpose(onnx_model, base_dir)
    aliases |= _duplicate_shared_qdq_inputs(onnx_model, base_dir)
    _remove_redundant_qdqs(onnx_model, base_dir)
    _decouple_back_to_back_qdqs(onnx_model)
    _remove_dangling_nodes_and_initializers(onnx_model)

    param_names = {
        f"{layer_name}.{param_name}" if layer_name else param_name
        for layer_name, layer in model.named_modules()
        if isinstance(layer, QuantizationMixin)
        for param_name, quantizer in layer.param_quantizers.items()
        if quantizer
    }

    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]] = {}

    for name, encoding in _onnx.remove_quantization_nodes_from_onnx_graph(
        onnx_model,
        base_dir=base_dir,
    ).items():
        is_param = name in param_names or aliases.get(name) in param_names
        tensor_to_encoding_map[name] = (encoding, is_param)

    for encoding in _remove_onnx_qdq_nodes(onnx_model, base_dir=base_dir):
        name = encoding.pop("name")
        is_param = name in param_names or aliases.get(name) in param_names
        encoding = AffineEncoding._from_qnn_encoding_dict(encoding, version="2.1.0")
        tensor_to_encoding_map[name] = (encoding, is_param)

    static_input_shapes = {
        name: tuple(const.dims)
        for name, const in _get_all_constants(onnx_model).items()
    }
    for name, (enc, _) in tensor_to_encoding_map.items():
        input_shape = static_input_shapes.get(name, None)
        if input_shape is not None:
            enc._hint_input_shape(input_shape)

    encoding_dict = {
        name: enc.to_qnn_encoding_dict("2.1.0")
        for name, (enc, _) in tensor_to_encoding_map.items()
    }
    derived_encodings = _derive_const_rescale_op_output_encodings(
        onnx_model, encoding_dict, base_dir
    )
    derived_encodings |= _derive_data_movement_op_encodings(
        onnx_model, encoding_dict | derived_encodings
    )

    if not propagate_int32_encodings:
        # When exporting to ONNX QDQ, int32 encodings cannot be propagated
        # because ONNX only supports int32 DequantizeLinear but not QuantizeLinear.
        derived_encodings = {
            name: enc
            for name, enc in derived_encodings.items()
            if enc["output_dtype"] != "int32"
        }

    tensor_to_encoding_map |= {
        name: (AffineEncoding._from_qnn_encoding_dict(encoding, version="2.1.0"), False)
        for name, encoding in derived_encodings.items()
    }

    _remove_unsafe_barrier(onnx_model, tensor_to_encoding_map)
    return onnx_model, tensor_to_encoding_map


@contextlib.contextmanager
def _concretize_int32_bias_quantizers(model, args, kwargs=None):
    if not isinstance(args, (tuple, list)):
        args = (args,)

    kwargs = kwargs or {}

    handles = []
    orig_bias_quantizers = {
        qmodule: qmodule.param_quantizers["bias"]
        for qmodule in model.modules()
        if isinstance(qmodule, QuantizationMixin)
        and "bias" in qmodule.param_quantizers
        and qmodule.bias is not None
    }

    try:
        for qmodule, qtzr in orig_bias_quantizers.items():
            if qtzr is not None:
                # Bias quantizer already exists.
                # This means the user created bias quantizer by him/herself
                # In this case, we honor the custom bias quantizer defined by the user
                continue

            if "weight" in qmodule.param_quantizers and isinstance(
                qmodule.param_quantizers["weight"], AffineQuantizerBase
            ):
                # pylint: disable=protected-access
                handle = qmodule.register_forward_hook(
                    type(qmodule)._create_int32_bias_quantizer
                )
                handles.append(handle)

        try:
            with contextlib.ExitStack() as stack:
                for qmodule in model.modules():
                    if not isinstance(qmodule, QuantizationMixin):
                        continue

                    # bias_scale will be derived as input_scale * weight_scale.
                    # Here, weight_scale is statically available in the weight quantizer,
                    # and we don't need to perform actual weight Q/DQ to capture weight_scale.
                    # Therefore, we temporarily set weight quantizers to "pass-through" mode
                    # to speed up export.
                    for qtzr in qmodule.param_quantizers.children():
                        stack.enter_context(patch_attr(qtzr, "forward", lambda _: _))

                model(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        yield
    finally:
        for qmodule, qtzr in orig_bias_quantizers.items():
            qmodule.param_quantizers["bias"] = qtzr


@contextlib.contextmanager
def _temporarily_unfold_param_quantizers(model: torch.nn.Module):
    # pylint: disable=protected-access
    """
    Temporarily re-instantiate param quantizers for ease of export
    """
    with contextlib.ExitStack() as stack:
        for qmodule in model.modules():
            if isinstance(qmodule, QuantizationMixin):
                stack.enter_context(qmodule._unfold_param_quantizers())
        yield


@contextlib.contextmanager
def _remove_fp16_quantized_parameters(model: torch.nn.Module):
    """
    Temporarily detach [b]float16 encodings from pre-quantized parameters
    """
    # pylint: disable=protected-access
    with contextlib.ExitStack() as stack:
        for qmodule in model.modules():
            if not isinstance(qmodule, QuantizationMixin):
                continue

            for name, param in qmodule.named_parameters(recurse=False):
                if (
                    isinstance(param, DequantizedTensor)
                    and isinstance(param.encoding, FloatEncoding)
                    and param.encoding._finfo.to_torch_dtype()
                    in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                ):
                    param = torch.nn.Parameter(param.as_subclass(torch.Tensor))
                    stack.enter_context(patch_attr(qmodule, name, param))

        yield


@contextlib.contextmanager
def _remove_fp16_quantizers(model: torch.nn.Module):
    """
    Temporarily remove [b]float16 quantizers for sim.onnx.export,
    as sim.onnx.export does NOT support exporting [b]float16 quantizers.
    """
    original_containers = {}

    try:
        for qmodule in model.modules():
            if not isinstance(qmodule, QuantizationMixin):
                continue

            for name, qtzr in qmodule.param_quantizers.items():
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.param_quantizers, name)] = qtzr
                    qmodule.param_quantizers[name] = None

            for i, qtzr in enumerate(qmodule.input_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.input_quantizers, i)] = qtzr
                    qmodule.input_quantizers[i] = None

            for i, qtzr in enumerate(qmodule.output_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.output_quantizers, i)] = qtzr
                    qmodule.output_quantizers[i] = None

        yield

    finally:
        for (container, key), qtzr in original_containers.items():
            container[key] = qtzr


def _to_onnx_qdq(
    onnx_model: onnx.ModelProto,
    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
    prequantize_constants: bool,
    base_dir: str,
) -> onnx.ModelProto:
    qnn_encodings = {
        name: encoding.to_qnn_encoding_dict("2.1.0")
        for name, (encoding, _) in tensor_to_encoding_map.items()
    }
    qnn_encodings = {
        name: encoding for name, encoding in qnn_encodings.items() if encoding
    }

    qdq_tensor_names = {
        fp_tensor_name: f"{fp_tensor_name}_qdq" for fp_tensor_name in qnn_encodings
    }

    onnx_opset_version = next(
        opset.version for opset in onnx_model.opset_import if opset.domain == ""
    )

    # TODO: Support exporting (b)float16 models
    if contains_tensor_type(
        onnx_model, (onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT16)
    ):
        raise RuntimeError(
            "Exporting to onnx QDQ is only supported for float32 models."
        )
    float_types = [np.float32 for _ in range(len(qnn_encodings))]

    # Add onnx QDQ nodes in batch
    _add_onnx_qdq_nodes(
        onnx_model,
        input_names=qnn_encodings.keys(),
        output_names=qdq_tensor_names.values(),
        node_name_prefixes=qnn_encodings.keys(),
        encodings=qnn_encodings.values(),
        float_types=float_types,
        onnx_opset=onnx_opset_version,
        prequantize_constants=prequantize_constants,
        base_dir=base_dir,
    )

    return onnx_model


def _check_float16_quantizers(module: torch.nn.Module):
    for qtzr in module.modules():
        if isinstance(qtzr, FloatQuantizeDequantize):
            if not qtzr.is_float16() and not qtzr.is_bfloat16():
                msg = " ".join(
                    [
                        "sim.onnx.export doesn't support exporting floating point encodings",
                        f"except [b]float16. Got {qtzr.bitwidth}-bit float encoding",
                    ]
                )
                raise RuntimeError(msg)


@torch.no_grad()
def _absorb_zero_point_shift(model: torch.nn.Module):
    """
    Absorb zero point shift to weights by promoting bitwidth from 2 to 4.

    NOTE: This function is only meant for internal testing purpose.
    """
    # pylint: disable=redefined-builtin
    for qmodule in model.modules():
        if not isinstance(qmodule, QuantizationMixin):
            continue

        for param_name, qtzr in qmodule.param_quantizers.items():
            if not isinstance(qtzr, AffineQuantizerBase):
                continue

            if not qtzr.is_initialized():
                continue

            if qtzr.zero_point_shift != 0.5:
                continue

            if not qtzr.symmetric:
                continue

            weight = getattr(qmodule, param_name)
            weight_qdq = qtzr(weight).dequantize()
            weight.copy_(weight_qdq)

            # weight_qdq ∈ {-1.5 * s,  -0.5 * s,  0.5 * s,  1.5 * s  }
            #            = {  -3 * s/2,  -1 * s/2,  1 * s/2,  3 * s/2}
            new_scale = qtzr.get_scale() / 2
            qtzr.bitwidth *= 2
            qtzr.zero_point_shift = 0.0
            min = new_scale * qtzr.qmin
            max = new_scale * qtzr.qmax
            qtzr.set_range(min, max)


@contextlib.contextmanager
def _temporarily_convert_activation_to(model: torch.nn.Module, signed: bool):
    """
    Temporarily convert all signed activation quantizers to unsigned ones for onnx export.

    TODO: This is a temporary workaround for QAIRT/HTP bug in handling signed activation encodings.
          Remove this once the bug is fixed.
          (https://github.qualcomm.com/qualcomm-ai/aimet/issues/5236)
    """
    target_activation_quantizers = [
        qtzr
        for module in model.modules()
        if isinstance(module, QuantizationMixin)
        for qtzr in itertools.chain(module.input_quantizers, module.output_quantizers)
        if isinstance(qtzr, AffineQuantizerBase) and qtzr.signed != signed
    ]

    try:
        for qtzr in target_activation_quantizers:
            qtzr.signed = signed
        yield
    finally:
        for qtzr in target_activation_quantizers:
            qtzr.signed = not signed
