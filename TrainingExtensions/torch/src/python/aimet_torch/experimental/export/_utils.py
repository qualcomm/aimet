# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
from contextlib import ExitStack, contextmanager
from typing import Any, Iterable, Callable
import operator

import torch
import torch.fx.node
from torch.export import ExportedProgram
from torch._export.utils import _detect_fake_mode_from_gm
from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._subclasses.fake_tensor import FakeTensorMode


_GRID_PRESERVING_OPS = (
    torch.ops.aten.alias,
    torch.ops.aten.alias_copy,
    torch.ops.aten.as_strided,
    torch.ops.aten.as_strided_copy,
    torch.ops.aten.amax,
    torch.ops.aten.amin,
    torch.ops.aten.clone,
    torch.ops.aten.contiguous,
    # torch.ops.aten.copy(input, src) returns src, not input. In other words,
    # aten.copy is grid-preserving with respect to 2nd input, not 1st.
    # Temporarily exclude aten.copy since aimet_torch.export can currently only
    # handle grid-preserving ops w.r.t 1st input
    # torch.ops.aten.copy,
    # torch.ops.aten.copy_,
    torch.ops.aten.detach,
    torch.ops.aten.diag,
    torch.ops.aten.diag_embed,
    torch.ops.aten.diagonal,
    torch.ops.aten.diagonal_backward,
    torch.ops.aten.diagonal_copy,
    torch.ops.aten.dropout,
    torch.ops.aten.dropout_,
    torch.ops.aten.embedding,
    torch.ops.aten.expand,
    torch.ops.aten.expand_copy,
    torch.ops.aten.flatten,
    torch.ops.aten.gather,
    torch.ops.aten.item,
    torch.ops.aten.kthvalue,
    torch.ops.aten.masked_select,
    torch.ops.aten.max_pool1d,
    torch.ops.aten.max_pool2d,
    torch.ops.aten.max_pool2d_with_indices,
    torch.ops.aten.max_pool3d,
    torch.ops.aten.max_pool3d_with_indices,
    torch.ops.aten.max,
    torch.ops.aten.min,
    torch.ops.aten.narrow,
    torch.ops.aten.narrow_copy,
    torch.ops.aten.native_dropout,
    torch.ops.aten.nonzero,
    torch.ops.aten.pad,
    torch.ops.aten.permute,
    torch.ops.aten.permute_copy,
    torch.ops.aten.reflection_pad1d,
    torch.ops.aten.reflection_pad2d,
    torch.ops.aten.reflection_pad3d,
    torch.ops.aten.relu,
    torch.ops.aten.relu_,
    torch.ops.aten.repeat,
    torch.ops.aten.repeat_interleave,
    torch.ops.aten.replication_pad1d,
    torch.ops.aten.replication_pad2d,
    torch.ops.aten.replication_pad3d,
    torch.ops.aten.reshape,
    torch.ops.aten._reshape_alias,
    torch.ops.aten._reshape_copy,
    torch.ops.aten.rot90,
    torch.ops.aten.select,
    torch.ops.aten.slice,
    torch.ops.aten.split,
    torch.ops.aten.split_with_sizes,
    torch.ops.aten.squeeze,
    torch.ops.aten.squeeze_copy,
    torch.ops.aten.t,
    torch.ops.aten.t_,
    torch.ops.aten.t_copy,
    torch.ops.aten.take,
    torch.ops.aten.tile,
    torch.ops.aten.topk,
    torch.ops.aten.transpose,
    torch.ops.aten.transpose_,
    torch.ops.aten.transpose_copy,
    torch.ops.aten.unfold,
    torch.ops.aten.unfold_backward,
    torch.ops.aten.unfold_copy,
    torch.ops.aten._unsafe_view,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.unsqueeze_,
    torch.ops.aten.unsqueeze_copy,
    torch.ops.aten.view,
    torch.ops.aten.view_copy,
    torch.ops.aten.zeros_like,
)


def _is_grid_preserving_op(node: torch.fx.Node) -> bool:
    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    return node.target.overloadpacket in _GRID_PRESERVING_OPS


def _remove_dangling_nodes(ep: ExportedProgram):
    output_node = ep.graph.output_node()
    visited: set[torch.fx.Node] = set()
    stack = [output_node]

    # Reverse-DFS from output node
    while stack:
        node = stack.pop(-1)
        if node in visited:
            continue
        visited.add(node)
        stack += node.all_input_nodes

    # Mark all visited nodes as non-dangling node
    dangling_nodes = set(ep.graph.nodes) - visited

    node_name_to_input_spec = {
        spec.arg.name: spec for spec in ep.graph_signature.input_specs
    }

    def is_user_input(node: torch.fx.Node) -> bool:
        input_spec = node_name_to_input_spec.get(node.name)
        return bool(input_spec) and input_spec.kind not in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        )

    # Remove dangling nodes from graph except for user inputs
    for node in reversed(list(ep.graph.nodes)):
        if node in dangling_nodes and not is_user_input(node):
            ep.graph.erase_node(node)

    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()

    # Clean up graph_signature and state_dict
    ep.graph_signature.input_specs = [
        node_name_to_input_spec[input_node.name]
        for input_node in ep.graph.find_nodes(op="placeholder", sort=False)
    ]
    all_targets: set[str | None] = set(
        input_spec.target for input_spec in ep.graph_signature.input_specs
    )

    for dangling_key in ep.state_dict.keys() - all_targets:
        del ep.state_dict[dangling_key]

    for dangling_key in ep.constants.keys() - all_targets:
        del ep.constants[dangling_key]


def _insert_placeholder(
    ep: ExportedProgram,
    val: torch.Tensor,
    node_name: str,
    tensor_name: str,
    consumers: Iterable[torch.fx.Node],
):
    with ExitStack() as stack:
        for consumer in consumers:
            stack.enter_context(ep.graph.inserting_before(consumer))

        node = ep.graph.create_node(
            op="placeholder",
            target=node_name,
            name=node_name,
        )

    fake_mode = _detect_fake_mode_from_gm(ep.graph_module) or FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    fake_tensor = converter.from_real_tensor(fake_mode, val)
    node.meta.update(
        {
            "val": fake_tensor,
            "tensor_meta": _extract_tensor_metadata(fake_tensor),
        }
    )

    i = InputSpec(
        kind=InputKind.BUFFER,
        arg=TensorArgument(name=node_name),
        target=tensor_name,
        persistent=True,
    )
    ep.graph_signature.input_specs.append(i)
    ep.state_dict.update({tensor_name: val})

    return node


def _eval_node(
    arg: torch.fx.node.Argument,
    ep: ExportedProgram,
) -> Any:
    input_specs = {spec.arg.name: spec for spec in ep.graph_signature.input_specs}
    params_and_constants = ep.state_dict | ep.constants

    def _do_eval(arg: torch.fx.node.Argument):
        if not isinstance(arg, torch.fx.Node):
            return arg

        node = arg

        if node.op == "placeholder":
            input_spec = input_specs[node.name]
            param_or_const_name = input_spec.target
            if param_or_const_name not in params_and_constants:
                raise RuntimeError(
                    "Couldn't find parameter, buffer, or constant "
                    f"with name {param_or_const_name} of node {node.name}"
                )
            return params_and_constants[param_or_const_name]

        if not callable(node.target):
            raise RuntimeError(
                f"Internal error occurred. Expected node {node.name} (op: {node.op}) "
                f"to be callable, but got node.target of type {type(node.target)}"
            )

        args = tuple(_do_eval(arg) for arg in node.args)
        kwargs = {key: _do_eval(val) for key, val in node.kwargs.items()}

        return node.target(*args, **kwargs)

    return _do_eval(arg)


def _is_multi_output_op(node: torch.fx.Node) -> bool:
    return (
        all(user.target is operator.getitem for user in node.users)
        and len(node.users) >= 1
    )


@contextmanager
def _refresh_output_specs_ctx(ep: ExportedProgram):
    original_output_names = [
        node.name for node in ep.graph.output_node().all_input_nodes
    ]

    yield

    # Edge case: if any new QDQ nodes were added before the output nodes,
    # we need to update the output names in the graph signature accordingly
    # Example:
    #                         (q/dq inserted by
    #                       encoding propagation)
    #         reshape -----------> q -------> dq -----------------> (output)
    #           ↑                             ↑
    #  ep.graph_signature is           ep.graph_signature should be
    #  still pointing to this          updated to point to this dq node
    #  as graph output                 as graph output
    new_output_names = {
        old_name: node.name
        for old_name, node in zip(
            original_output_names,
            ep.graph.output_node().all_input_nodes,
        )
    }
    for spec in ep.graph_signature.output_specs:
        old_output_name = spec.arg.name
        new_output_name = new_output_names.get(old_output_name, old_output_name)
        spec.arg.name = new_output_name


def _refresh_output_specs(fn: Callable):
    def wrapper(*args, **kwargs):
        ep, *args = args
        with _refresh_output_specs_ctx(ep):
            return fn(ep, *args, **kwargs)

    return wrapper


def get_example_inputs(op: torch._ops.OpOverload) -> tuple[tuple, dict]:
    """
    Get example inputs for a given operator.
    """

    from ._example_inputs import _example_inputs_registry

    input_factory = _example_inputs_registry.get(op, None)

    if not input_factory:
        raise RuntimeError(
            f"Example inputs for {op} not found. "
            "Please register example inputs using the @register_example_inputs decorator."
        )

    return input_factory()


def _is_torch_ao_qdq_node(node: torch.fx.Node) -> bool:
    return _is_torch_ao_q_node(node) or _is_torch_ao_dq_node(node)


def _is_torch_ao_q_node(node: torch.fx.Node) -> bool:
    return node.target in _torch_ao_q_ops


def _is_torch_ao_dq_node(node: torch.fx.Node) -> bool:
    return node.target in _torch_ao_dq_ops


_torch_ao_q_ops = (
    *(
        getattr(torch.ops.quantized_decomposed.quantize_per_tensor, overload_name)
        for overload_name in torch.ops.quantized_decomposed.quantize_per_tensor.overloads()
    ),
    *(
        getattr(torch.ops.quantized_decomposed.quantize_per_channel, overload_name)
        for overload_name in torch.ops.quantized_decomposed.quantize_per_channel.overloads()
    ),
)
_torch_ao_dq_ops = (
    *(
        getattr(torch.ops.quantized_decomposed.dequantize_per_tensor, overload_name)
        for overload_name in torch.ops.quantized_decomposed.dequantize_per_tensor.overloads()
    ),
    *(
        getattr(torch.ops.quantized_decomposed.dequantize_per_channel, overload_name)
        for overload_name in torch.ops.quantized_decomposed.dequantize_per_channel.overloads()
    ),
)
