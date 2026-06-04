# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

import onnx_ir
import onnx
import numpy as np

from .fusion_registry import AIMET_SUPERGROUP_DOMAIN


def get_constant_singleton_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> float | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    numpy_value = get_constant_or_attribute_value(value)

    if numpy_value is None or numpy_value.size != 1:
        return None

    return numpy_value.flatten()[0].item()


def get_constant_as_array(value: onnx_ir.Value | None) -> np.ndarray | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    if value is None:
        return None

    const_value = onnx_ir.convenience.get_const_tensor(value)
    if const_value is None:
        return None

    return const_value.numpy()


def get_constant_or_attribute_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> None | np.ndarray:
    """Get the constant value from an ONNX IR Value or Attr, if it exists."""
    if value is None:
        return None
    if isinstance(value, onnx_ir.Value):
        return get_constant_as_array(value)
    if isinstance(value, onnx_ir.Attr):
        return np.asarray(value.value)
    raise RuntimeError(f"Received unexpected type for value: {type(value)}")


def _sort_functions_hierarchically(model: onnx_ir.Model) -> None:
    """Sort model functions from outermost to innermost to prevent mangling of names during inlining."""
    # pylint: disable=protected-access
    sorted_funcs = {}

    def node_has_impl(node: onnx_ir.Node) -> bool:
        return not is_fused_supergroup(node) or node.op_identifier() in sorted_funcs

    while True:
        runnable_functions = {
            fid: func
            for fid, func in model.functions.items()
            if fid not in sorted_funcs
            and all(node_has_impl(n) for n in func.graph.all_nodes())
        }
        if not runnable_functions:
            break
        sorted_funcs.update(runnable_functions)

    if not sorted_funcs.keys() == model.functions.keys():
        raise RuntimeError(
            f"Cycle detected among supergroup functions: {set(model.functions.keys()) - set(sorted_funcs.keys())}"
        )

    # Reverse ordering to prevent name mangling while unrolling
    model._functions = dict(reversed(list(sorted_funcs.items())))


def inline_all_supergroups(model: onnx_ir.Model) -> None:
    """Inline all aimet supergroup functions, restoring original node and value names."""
    supergroup_functions = {
        func for func in model.functions.values() if is_fused_supergroup(func)
    }
    if not supergroup_functions:
        return

    _sort_functions_hierarchically(model)
    onnx_ir.passes.common.InlinePass(lambda f: f in supergroup_functions).call(model)
    supergroup_opsets = [
        opset
        for opset in model.graph.opset_imports
        if opset.startswith(AIMET_SUPERGROUP_DOMAIN)
    ]
    for name in supergroup_opsets:
        model.graph.opset_imports.pop(name)


def unique_name(base: str, existing: set[str]) -> str:
    """Generate a unique name based on the provided base that does not exist in the existing set."""
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def fold_overload_into_op_domain(model: onnx_ir.Model):
    """
    Works around bug in onnxruntime dispatch to overloaded functions by replacing
    function.domain with "{function.domain}.{function.overload}" for all supergroup ops
    """
    new_functions = {}
    for function in model.functions.values():
        if function.domain == AIMET_SUPERGROUP_DOMAIN and function.overload:
            function.domain = f"{function.domain}.{function.overload}"
        new_functions[function.identifier()] = function

    for node in model.graph.all_nodes():
        if node.domain != AIMET_SUPERGROUP_DOMAIN:
            continue
        if not node.overload:
            continue
        node.domain = f"{node.domain}.{node.overload}"

    model._functions = new_functions  # pylint:disable = protected-access
    opsets = set(f.domain for f in model.functions.values() if is_fused_supergroup(f))
    for opset in opsets:
        model.opset_imports[opset] = 1


def is_fused_supergroup(
    node: onnx_ir.Node | onnx_ir.Function | onnx.NodeProto | onnx.FunctionProto,
) -> bool:
    """Return True if ``node`` represents an AIMET supergroup op."""
    return node.domain.startswith(AIMET_SUPERGROUP_DOMAIN)
