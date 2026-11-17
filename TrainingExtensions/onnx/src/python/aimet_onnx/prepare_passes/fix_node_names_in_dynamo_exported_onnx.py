# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Optional, Set, Union

import onnx
import onnx_ir


_SEQUENTIAL_LIKE_CLASSES = frozenset(
    {
        "torch.nn.modules.container.Sequential",
        "torch.nn.modules.container.ModuleList",
    }
)


def _parse_namespace(namespace: str):
    """Parse namespace string into list of (dotted_name, class_type) pairs.

    Namespace format: ': ClassName/attr_name: ClassName/attr_name: ClassName/...'
    """
    pairs = []
    for entry in namespace.split("/"):
        parts = entry.split(": ", 1)
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def _get_name_from_metadata(node: Union[onnx.NodeProto, onnx_ir.Node]) -> Optional[str]:
    """Derive a torchscript-style node name from dynamo metadata_props.

    Parses the 'namespace' field which contains the module hierarchy with class
    types. For Sequential-like containers, TorchScript uses a redundant naming
    convention where children are named '{parent_attr}.{index}' (e.g.,
    'features.0' instead of just '0'). This function reconstructs that pattern.
    """
    if isinstance(node.metadata_props, dict):
        namespace = node.metadata_props.get("namespace")
    else:
        namespace = next(
            (prop.value for prop in node.metadata_props if prop.key == "namespace"),
            None,
        )

    if namespace is None:
        return None

    pairs = _parse_namespace(namespace)
    if len(pairs) < 2:
        return None

    # Skip root (first) and FX node name (last).
    # Build path segments from module hierarchy.
    segments = []
    for i in range(1, len(pairs) - 1):
        dotted_name = pairs[i][0]
        parent_dotted_name = pairs[i - 1][0]
        parent_class = pairs[i - 1][1]

        # Compute relative name within parent
        if parent_dotted_name:
            relative = dotted_name.removeprefix(parent_dotted_name + ".")
        else:
            relative = dotted_name

        # For Sequential-like containers, TorchScript prefixes child segments
        # with the container's own attribute name: block.0 instead of just 0
        if parent_class in _SEQUENTIAL_LIKE_CLASSES and segments:
            segment = f"{segments[-1]}.{relative}"
        else:
            segment = relative

        segments.append(segment)

    if not segments:
        return f"/{node.op_type}"

    return "/" + "/".join(segments) + "/" + node.op_type


def _get_name_from_initializers(
    node: onnx.NodeProto, initializer_names: Set[str]
) -> Optional[str]:
    """Derive a torchscript-style node name from weight initializer input names.

    For a node consuming 'layer1.conv1.weight', strips the parameter suffix
    to get module path 'layer1.conv1' and constructs '/layer1/conv1/Conv'.
    """
    param_suffixes = (".weight", ".bias")
    for input_name in node.input:
        if input_name not in initializer_names:
            continue
        for suffix in param_suffixes:
            if input_name.endswith(suffix):
                module_dotted_name = input_name[: -len(suffix)]
                path = module_dotted_name.replace(".", "/")
                return f"/{path}/{node.op_type}"
    return None


def _make_unique(name: str, seen: Dict[str, int]) -> str:
    """Ensure name is unique by appending a counter suffix if needed."""
    if name not in seen:
        seen[name] = 0
        return name
    seen[name] += 1
    unique = f"{name}_{seen[name]}"
    while unique in seen:
        seen[name] += 1
        unique = f"{name}_{seen[name]}"
    seen[unique] = 0
    return unique


def fix_node_names_pass(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix node names in a dynamo-exported ONNX model.

    Renames nodes to match the torchscript-exported naming convention:
        /{module_path}/{OpType}
    e.g. /layer1/conv1/Conv, /layer1/relu1/Relu

    Also renames output tensors to '{node_name}_output_{index}' and updates
    all downstream references (node inputs and graph outputs) accordingly.

    Uses two strategies in priority order:
    1. Parse 'namespace' metadata (set by dynamo exporter)
    2. Derive module path from weight initializer input names
    If neither works, the original node name is kept.
    """
    initializer_names = {init.name for init in model.graph.initializer}
    graph_output_names = {o.name for o in model.graph.output}
    seen: Dict[str, int] = {}
    # Maps old output tensor name -> new output tensor name
    output_rename_map: Dict[str, str] = {}

    for node in model.graph.node:
        new_name = _get_name_from_metadata(node)
        if new_name is None:
            new_name = _get_name_from_initializers(node, initializer_names)
        if new_name is None:
            new_name = node.name if node.name else f"/{node.op_type}"

        node.name = _make_unique(new_name, seen)

        # Rename output tensors: {node_name}_output_{index}
        # Preserve model output names so external callers aren't broken.
        for i, old_output in enumerate(node.output):
            # Skip empty outputs because there is no valid tensor name to rename.
            if not old_output:
                continue

            # Preserve graph output names so the model's external interface stays stable.
            if old_output in graph_output_names:
                continue

            new_output = f"{node.name}_output_{i}"
            output_rename_map[old_output] = new_output
            node.output[i] = new_output

    # Update all downstream input references
    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in output_rename_map:
                node.input[i] = output_rename_map[input_name]

    # Update intermediate tensor type/shape metadata
    for vi in model.graph.value_info:
        if vi.name in output_rename_map:
            vi.name = output_rename_map[vi.name]

    return model
