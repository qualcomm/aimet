# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
import itertools
import contextlib
from typing import Tuple, Optional
from packaging.version import parse
import torch
from torch.export import ExportedProgram
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from ..onnx._export import _precompute_encodings
from ...utils import patch_attr
from ...quantization import QuantizedTensorBase
from ._utils import (
    _is_grid_preserving_op,
    _remove_dangling_nodes,
    _eval_node,
    _insert_placeholder,
    _is_multi_output_op,
    _refresh_output_specs,
)

__all__ = ["export"]


def export(mod: torch.nn.Module, *args, **kwargs) -> ExportedProgram:
    """
    Export :class:`QuantizationSimModel` to ExportedProgram with
    quantization ops embedded in the aten graph.

    This function takes set of same arguments as `torch.export.export()`_
    """
    if parse(torch.__version__) < parse("2.8.0"):
        raise RuntimeError(
            "Exporting to torch.export.ExportedProgram is only supported with torch>=2.8.0; "
            f" got torch=={torch.__version__}"
        )

    if isinstance(mod, torch._dynamo.OptimizedModule):
        if parse(torch.__version__) < parse("2.11.0.dev"):
            raise RuntimeError(
                "Exporting a torch.compile-d quantsim model is only supported in torch >= 2.11.0. "
                "For more information, see https://github.com/pytorch/pytorch/issues/171674"
            )

    from aimet_torch.nn import QuantizationMixin
    from aimet_torch.quantization.affine import AffineQuantizerBase

    quantizers = [
        qtzr for qtzr in mod.modules() if isinstance(qtzr, AffineQuantizerBase)
    ]

    if not quantizers and not any(
        isinstance(param_or_buffer, QuantizedTensorBase)
        for param_or_buffer in itertools.chain(mod.parameters(), mod.buffers())
    ):
        # No quantizer or quantized tensor. Export directly without extra processing
        return torch.export.export(mod, *args, **kwargs)

    #  If no quantizers are initialized, raise error
    if not any(qtzr.is_initialized() for qtzr in quantizers):
        raise RuntimeError(
            "Please ensure that the quantizers are initialized before exporting. "
            "You can do this by running a forward pass with representative data "
            "within QuantizationSimModel.compute_encodings() or "
            "under aimet_torch.nn.compute_encodings() context manager."
        )

    # If any qmodule is not dynamo traceable, raise error
    untraceable_modules = []

    for name, module in mod.named_modules():
        if not isinstance(module, QuantizationMixin):
            continue
        is_dynamo_traceable, reason = module._is_dynamo_traceable()
        if not is_dynamo_traceable:
            untraceable_modules.append((name, module, reason))

    if untraceable_modules:
        raise RuntimeError(
            "Following modules don't support dynamo tracing:\n"
            + "\n".join(
                [
                    f"- {name} (type: {type(module).__name__}): {reason}"
                    for name, module, reason in untraceable_modules
                ]
            )
        )

    # Pre-compute scale and offset to omit verbose
    # scale/offset derivation logic in the exported graph
    with _duplicate_shared_weights(mod), _precompute_encodings(mod), torch.no_grad():
        ep = torch.export.export(mod, *args, **kwargs)

    return _post_process(ep)


@_refresh_output_specs
def _post_process(ep: ExportedProgram) -> ExportedProgram:
    """
    Post-process exported program.

    1. Fold scale and zero_point into constants
    2. Insert missing quantize-dequantize pairs for grid-preserving ops
    3. Remove dangling nodes
    """

    # 1. Fold scale and zero_point into constants
    for node in ep.graph.nodes:
        if _is_qdq_op(node):
            _try_fold_scale_and_zp(node, ep)

    # 2. Insert missing quantize-dequantize pairs for grid-preserving ops
    for node in ep.graph.nodes:
        if _is_grid_preserving_op(node):
            _try_insert_output_qdq(ep, node)

    for node in reversed(ep.graph.nodes):
        if _is_grid_preserving_op(node):
            _try_insert_input_qdq(ep, node)

    # 3. Remove dangling nodes
    _remove_dangling_nodes(ep)
    return ep


@contextlib.contextmanager
def _duplicate_shared_weights(mod: torch.nn.Module):
    from aimet_torch.nn import QuantizationMixin

    shared_params = {
        name: param for name, param in mod.named_parameters(remove_duplicate=False)
    }
    for name, _ in mod.named_parameters(remove_duplicate=True):
        shared_params.pop(name, None)

    with contextlib.ExitStack() as stack:
        for full_param_name, param in shared_params.items():
            module_name, param_name = full_param_name.rsplit(".", 1)
            qmodule = mod.get_submodule(module_name)

            if not isinstance(qmodule, QuantizationMixin):
                continue

            if param_name not in qmodule.param_quantizers:
                continue

            param_qtzr = qmodule.param_quantizers[param_name]

            if not param_qtzr or not param_qtzr.is_initialized():
                continue

            stack.enter_context(
                patch_attr(qmodule, param_name, torch.nn.Parameter(param.clone()))
            )
        yield


def _try_insert_output_qdq(ep: ExportedProgram, node: torch.fx.Node):
    input_q, input_dq = _get_input_qdq(node)
    output_q, output_dq = _get_output_qdq(node)

    if not (
        input_q is not None
        and input_dq is not None
        and output_q is None
        and output_dq is None
    ):
        return

    qtype = input_q.args[5] if len(input_q.args) > 5 else input_q.kwargs["dtype"]

    if qtype == torch.int32:
        # Skip int32 encoding propagation for the convenience of ONNX export.
        # Currently, torch.onnx.export produces int32 QuantizeLinear
        # when exporting int32 torch.ops.quantized_decomposed.quantize_per_tensor,
        # which is technically invalid according to ONNX spec.
        # Therefore, disable int32 encoding propagation as a workaround to avoid
        # exporting int32 QuantizeLinear.
        # For more information, see https://github.com/pytorch/pytorch/issues/198691
        return

    if _is_multi_output_op(node):
        for user in list(node.users):
            if (
                "tensor_meta" in user.meta
                and user.meta["tensor_meta"].dtype.is_floating_point
            ):
                _insert_output_qdq(ep, user, input_q, input_dq)
    else:
        _insert_output_qdq(ep, node, input_q, input_dq)


def _insert_output_qdq(
    ep: ExportedProgram,
    node: torch.fx.Node,
    input_q: torch.fx.Node,
    input_dq: torch.fx.Node,
):
    qtype = input_q.args[5] if len(input_q.args) > 5 else input_q.kwargs["dtype"]
    with ep.graph.inserting_after(node):
        output_q = ep.graph.create_node(
            op=input_q.op,
            target=input_q.target,
            args=(node, *input_q.args[1:]),
            kwargs=input_q.kwargs.copy(),
            name=f"{input_q.name}_copy",
        )
        output_q.meta["val"] = node.meta["val"].to(qtype)
        output_q.meta["tensor_meta"] = _extract_tensor_metadata(output_q.meta["val"])

    with ep.graph.inserting_after(output_q):
        output_dq = ep.graph.create_node(
            op=input_dq.op,
            target=input_dq.target,
            args=(output_q, *input_dq.args[1:]),
            kwargs=input_dq.kwargs.copy(),
            name=f"{input_dq.name}_copy",
        )
        output_dq.meta.update(
            {
                "val": node.meta["val"].clone(),
                "tensor_meta": node.meta["tensor_meta"],
            }
        )

    node.replace_all_uses_with(output_dq)
    output_q.args = (node, *input_q.args[1:])


def _try_insert_input_qdq(ep: ExportedProgram, node: torch.fx.Node):
    if node.all_input_nodes:
        input = node.all_input_nodes[0]  # pylint: disable=redefined-builtin
    else:
        return

    if _is_multi_output_op(input):
        return

    input_q, input_dq = _get_input_qdq(node)
    output_q, output_dq = _get_output_qdq(node)

    if not (
        input
        and len(input.users) == 1
        and output_q is not None
        and output_dq is not None
        and input_q is None
        and input_dq is None
    ):
        return

    qtype = output_q.args[5] if len(output_q.args) > 5 else output_q.kwargs["dtype"]

    if qtype == torch.int32:
        # Skip int32 encoding propagation for the convenience of ONNX export.
        # Currently, torch.onnx.export produces int32 QuantizeLinear
        # when exporting int32 torch.ops.quantized_decomposed.quantize_per_tensor,
        # which is technically invalid according to ONNX spec.
        # Therefore, disable int32 encoding propagation as a workaround to avoid
        # exporting int32 QuantizeLinear.
        # For more information, see https://github.com/pytorch/pytorch/issues/198691
        return

    with ep.graph.inserting_after(input):
        input_q = ep.graph.create_node(
            op=output_q.op,
            target=output_q.target,
            args=(input, *output_q.args[1:]),
            kwargs=output_q.kwargs.copy(),
            name=f"{output_q.name}_copy",
        )
        input_q.meta["val"] = input.meta["val"].to(qtype)
        input_q.meta["tensor_meta"] = _extract_tensor_metadata(input_q.meta["val"])

    with ep.graph.inserting_after(input_q):
        input_dq = ep.graph.create_node(
            op=output_dq.op,
            target=output_dq.target,
            args=(input_q, *output_dq.args[1:]),
            kwargs=output_dq.kwargs.copy(),
            name=f"{output_dq.name}_copy",
        )
        input_dq.meta.update(
            {
                "val": input.meta["val"].clone(),
                "tensor_meta": node.meta["tensor_meta"],
            }
        )

    input.replace_all_uses_with(input_dq)
    input_q.args = (input, *input_q.args[1:])


def _get_output_qdq(
    node: torch.fx.Node,
) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    (q,) = node.users if len(node.users) == 1 else (None,)
    (dq,) = q.users if q and len(q.users) == 1 else (None,)

    if not (
        dq
        and isinstance(dq.target, torch._ops.OpOverload)
        and dq.target.overloadpacket
        == torch.ops.quantized_decomposed.dequantize_per_tensor
    ):
        dq = None
        q = None

    if not (
        q
        and isinstance(q.target, torch._ops.OpOverload)
        and q.target.overloadpacket
        == torch.ops.quantized_decomposed.quantize_per_tensor
    ):
        q = None

    return q, dq


def _get_input_qdq(
    node: torch.fx.Node,
) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    dq = node.all_input_nodes[0] if node.all_input_nodes else None
    q = dq.all_input_nodes[0] if dq and dq.all_input_nodes else None

    if not (
        dq
        and isinstance(dq.target, torch._ops.OpOverload)
        and dq.target.overloadpacket
        == torch.ops.quantized_decomposed.dequantize_per_tensor
    ):
        dq = None
        q = None

    if not (
        q
        and isinstance(q.target, torch._ops.OpOverload)
        and q.target.overloadpacket
        == torch.ops.quantized_decomposed.quantize_per_tensor
    ):
        q = None

    return q, dq


def _is_qdq_op(node: torch.fx.Node) -> bool:
    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    return (
        node.target.name().startswith("aten::fake_quantize")
        or node.target.name().startswith("quantized_decomposed::quantize")
        or node.target.name().startswith("quantized_decomposed::dequantize")
    )


def _try_fold_scale_and_zp(q_dq_node: torch.fx.Node, ep: ExportedProgram):
    if (
        len(q_dq_node.all_input_nodes) > 1
        and q_dq_node.all_input_nodes[1].op != "placeholder"
    ):
        scale: torch.Tensor = _eval_node(q_dq_node.all_input_nodes[1], ep)
        scale_placeholder: torch.fx.Node = _insert_placeholder(
            ep,
            val=scale,
            node_name=f"p_{q_dq_node.name}_scale",
            tensor_name=f"{q_dq_node.name}_scale",
            consumers=[q_dq_node],
        )
        q_dq_node.replace_input_with(q_dq_node.all_input_nodes[1], scale_placeholder)

    if (
        len(q_dq_node.all_input_nodes) > 2
        and q_dq_node.all_input_nodes[2].op != "placeholder"
    ):
        zero_point: torch.Tensor = _eval_node(q_dq_node.all_input_nodes[2], ep)
        zero_point_placeholder: torch.fx.Node = _insert_placeholder(
            ep,
            val=zero_point,
            node_name=f"p_{q_dq_node.name}_zero_point",
            tensor_name=f"{q_dq_node.name}_zero_point",
            consumers=[q_dq_node],
        )
        q_dq_node.replace_input_with(
            q_dq_node.all_input_nodes[2], zero_point_placeholder
        )
