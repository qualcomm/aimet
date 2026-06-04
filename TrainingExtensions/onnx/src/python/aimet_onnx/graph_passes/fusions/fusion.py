# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX supergroup fusion implementation"""

from collections import defaultdict
import onnx_ir
from onnxscript.rewriter import pattern
from .fusion_registry import FUSION_PASS_REGISTRY
from . import ir_utils


def fuse_supergroups(
    model: onnx_ir.Model, patterns: list[str], *, verbose: int | None = None
) -> onnx_ir.Model:
    """
    Fuses specified patterns in the ONNX model into supergroups.

    This function identifies fusable decomposed operators (e.g., decomposed LayerNormalization)
    in the ONNX model and replaces them with onnx_ir.Functions representing the fused operator.

    Args:
        model: The ONNX model to process.
        patterns: List of pattern names to fuse (available patterns: {"LayerNormalization"}).

    Returns:
        onnx_ir.Model: The modified model with each matched pattern represented as an onnx_ir Function.

    Example:
        >>> model = onnx_ir.load("decomposed_model.onnx")
        >>> fused = fuse_supergroups(model, patterns=['LayerNormalization'])
    """
    unknown_patterns = [p for p in patterns if p not in FUSION_PASS_REGISTRY]
    if unknown_patterns:
        raise ValueError(
            f"Graph pass requested but not found: {unknown_patterns}. "
            f"Available patterns: {list(FUSION_PASS_REGISTRY.passes.keys())}"
        )

    if not patterns:
        return model

    # Create rewrite rule set to match specified patterns into onnx functions
    fusion_rules = [
        pattern for name in patterns for pattern in FUSION_PASS_REGISTRY[name]
    ]
    rule_set = pattern.RewriteRuleSet(fusion_rules)

    # Apply the rewrite rules to the model
    count = rule_set.apply_to_model(model, verbose=verbose)
    if count:
        # Note: ORT shape inference cannot handle nested functions, unroll anything nested
        _inline_nested_functions(model)
        onnx_ir.passes.common.RemoveUnusedNodesPass().call(model)
        _rename_supergroup_nodes(model)
        # Note: ORT does not correctly dispatch using function.overload, workaround by fusing to domain
        ir_utils.fold_overload_into_op_domain(model)

    return model


def _inline_nested_functions(model: onnx_ir.Model):
    """Inline supergroup functions that are called inside other supergroups."""
    # Collect all supergroup functions that are called from another function
    nested_supergroups = set(
        node.op_identifier()
        for func in model.functions.values()
        for node in func.graph.all_nodes()
        if ir_utils.is_fused_supergroup(node)
    )
    # Note: To get around name mangling of nested functions by InlinePass, sort functions hierarchically (outermost first)
    ir_utils._sort_functions_hierarchically(model)  # pylint: disable=protected-access
    onnx_ir.passes.common.InlinePass(
        lambda f: f.identifier() in nested_supergroups
    ).call(model)


def _rename_supergroup_nodes(model: onnx_ir.Model):
    """Rename supergroup nodes to have more meaningful names derived from their function body, if possible."""
    node_names = {node.name for node in model.graph.all_nodes()}
    supergroup_nodes = [
        node for node in model.graph.all_nodes() if ir_utils.is_fused_supergroup(node)
    ]
    root_to_nodes = defaultdict(list)

    # Find root name for each supergroup node
    for node in supergroup_nodes:
        func = model.functions.get(node.op_identifier())
        if not func:
            continue
        root = _derive_function_name_from_nodes(func)
        if not root or root in node_names:
            continue

        root_to_nodes[root].append(node)

    for root, nodes in root_to_nodes.items():
        for node in nodes:
            # If multiple nodes share the same root, append a unique suffix for each
            new_name = root if len(nodes) == 1 else f"{root}/{node.op_type}"
            new_name = ir_utils.unique_name(new_name, node_names)
            node_names.discard(node.name)
            node_names.add(new_name)
            node.name = new_name


def _derive_function_name_from_nodes(func: onnx_ir.Function) -> str | None:
    nodes = [
        node
        for node in func.all_nodes()
        if node.op_type not in ("Constant", "Identity")
    ]
    node_roots = set(
        node.name.rpartition("/")[0] for node in nodes if node.name and "/" in node.name
    )
    if node_roots and len(node_roots) == 1:
        return node_roots.pop()
    return None
