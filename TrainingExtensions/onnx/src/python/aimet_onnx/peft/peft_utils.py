# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graph surgery for LoRA: target resolution, branch insertion and weight loading."""

import dataclasses
import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnx_ir
from safetensors import safe_open

from aimet_onnx.peft.config import (
    WEIGHTS_FILE,
    AdapterLayout,
    RankSlice,
    checkpoint_key,
)
from aimet_onnx.ir_utils import get_weight_value, static_tensor
from aimet_onnx.common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

_WEIGHT_OPS = ("MatMul", "Gemm", "Conv")

_SAFETENSORS_DTYPES = {
    "F32": onnx_ir.DataType.FLOAT,
    "F16": onnx_ir.DataType.FLOAT16,
    "F64": onnx_ir.DataType.DOUBLE,
}

# Name fragments. Every name below is built from these, and the finder reads them back.
_A_WEIGHT, _B_WEIGHT, _SCALE = ".lora_A.weight", ".lora_B.weight", ".lora_scale"
_MUL, _ADD = "/lora/Mul", "/lora/Add"
_A_NODE, _B_NODE = "/lora_A/{}", "/lora_B/{}"  # formatted with MatMul or Conv
# The one scale input every branch shares after ``pull_branch_weights_to_inputs``
LORA_SCALE_INPUT = "lora_scale"


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttachPoint:
    """A base layer that gets a LoRA branch, and where the branch merges back."""

    # dotted torch path in the graph, e.g. model.layers.0.self_attn.q_proj
    module_path: str
    # the same layer's path in the adapter checkpoints
    checkpoint_path: str
    node: onnx_ir.Node  # the base MatMul / Gemm / Conv
    weight: onnx_ir.Value
    kind: Literal["linear", "conv"]
    in_features: int  # Linear in-features, or Conv Cin
    out_features: int  # Linear out-features, or Conv Cout
    # The branch output is added to this. For a linear with a bias it is the bias Add's output, so
    # the MatMul + Add bias pattern stays intact for QuantSim.
    base_out: onnx_ir.Value


@dataclass(frozen=True)
class BranchWeights:
    """
    The values of one branch's three weights, in the base weight's dtype.

    These are what ``LoraBranch.a_weight`` / ``b_weight`` / ``scale_weight`` name. For a linear
    layer with in/out features ``d``/``m`` and total rank ``R`` (all adapters' ranks added up):

    - ``a_weight``: ``[d, R]``, every adapter's ``lora_A`` side by side
    - ``b_weight``: ``[R, m]``, every adapter's ``lora_B`` stacked
    - ``scale_weight``: ``[R]``, ``adapter_scale * alpha / r`` for the adapter that owns each rank

    Conv weights have the same layout with the kernel dimensions of the base layer added.
    """

    a_weight: np.ndarray
    b_weight: np.ndarray
    scale_weight: np.ndarray


@dataclass(frozen=True)
class LoraBranch:
    """
    The names of everything in one LoRA branch. Example values are for ``layers.0.q_proj``.

    The branch is added next to the base layer and merged back into its output::

        x -> a_proj -> scale -> b_proj -> merge -> merged_output
        base layer ------------------------^  (base_layer_output)
    """

    module_path: str  # layers.0.q_proj
    node_prefix: (
        str  # /layers.0/q_proj, shared by the base layer's and the branch's node names
    )
    kind: Literal["linear", "conv"]
    a_weight: str  # layers.0.q_proj.lora_A.weight: [in, R] for linear
    b_weight: str  # layers.0.q_proj.lora_B.weight: [R, out] for linear
    scale_weight: str  # layers.0.q_proj.lora_scale: one multiplier per rank
    a_proj_node: str  # /layers.0/q_proj/lora_A/MatMul: x @ A
    scale_node: str  # /layers.0/q_proj/lora/Mul: multiplies by the scale
    b_proj_node: str  # /layers.0/q_proj/lora_B/MatMul: @ B
    merge_node: str  # /layers.0/q_proj/lora/Add: base layer output + branch output
    a_proj_output: str  # the output tensor of a_proj_node
    scaled_output: str  # the output tensor of scale_node
    b_proj_output: str  # the output tensor of b_proj_node, the LoRA delta
    merged_output: (
        str  # the output tensor of merge_node, what the rest of the model reads
    )
    base_layer_output: str  # the output tensor of the base layer (or its bias Add)

    def created_names(self) -> list[str]:
        """The names ``insert_branches`` creates; ``base_layer_output`` is only new if renamed."""
        names = [self.a_weight, self.b_weight, self.scale_weight]
        names += [self.a_proj_node, self.scale_node, self.b_proj_node, self.merge_node]
        names += [self.a_proj_output, self.scaled_output, self.b_proj_output]
        names.append(self.merged_output)
        return names


@dataclass(frozen=True)
class LoraTensorNames:
    """
    The tensors that only the LoRA branches have, and where each branch merges into the base.

    Built by ``get_lora_tensor_names``; the ``adapter_utils`` helpers take it.
    """

    params: tuple[
        str, ...
    ]  # every branch's A and B weights (initializers or graph inputs)
    # every branch's a_proj, scaled and b_proj outputs. Not the merged output, which is on the
    # base path (see ``merges``).
    activations: tuple[str, ...]
    # every branch's per-rank scale initializer, or the one shared ``lora_scale`` input
    scales: tuple[str, ...]
    # (base layer output, merged output) per branch, e.g. to make the merged output share the
    # base layer's output quantizer
    merges: tuple[tuple[str, str], ...]

    @property
    def quantizer_names(self) -> tuple[str, ...]:
        """The LoRA tensors that are quantized: params and activations, not the scales."""
        return self.params + self.activations


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------


def resolve_targets(
    graph: onnx_ir.Graph, module_paths: Iterable[str]
) -> dict[str, AttachPoint]:
    """
    Find the base layer for each module path.

    Paths match exactly. If none of them does, one shared prefix difference between the adapter's
    paths and the graph's (e.g. a wrapper module around the model that adds ``model.``) is
    accepted when it explains the match for every path that has one; see ``_find_prefix_shift``.
    Paths with no match keep their own name and resolve as missing. The graph is indexed once.
    Ambiguous paths only raise when targeted.

    Paths that match no layer are left out with a warning, as torch peft does, e.g. a 28-layer
    adapter on a 2-layer base. If none match, it raises: the model is the wrong one.

    :return: The graph's module path -> attachment point. ``AttachPoint.checkpoint_path`` is the
        adapter's path for the layer, which differs only when a prefix shift was applied.
    :raises ValueError: If no path matches, a path is ambiguous, some paths match exactly and
        others only through a prefix, or a node can't take a LoRA branch
    """
    module_paths = list(module_paths)
    index = _index_weight_nodes(graph)

    graph_paths = {path: path for path in module_paths}
    if not any(path in index for path in module_paths):
        graph_paths = _find_prefix_shift(module_paths, index)
    else:
        _check_not_mixed_prefixes(module_paths, index)

    output_ids = {id(v) for v in graph.outputs}
    points = {}
    skipped = []
    for path in module_paths:
        graph_path = graph_paths[path]
        nodes = index.get(graph_path)
        if not nodes:
            skipped.append(path)
            continue
        if len(nodes) > 1:
            raise ValueError(
                f"Module path {graph_path!r} is ambiguous; it matches nodes "
                f"{[n.name for n in nodes]}"
            )
        # Exact matching and a single prefix shift are one-to-one, so graph_path is new here
        points[graph_path] = _make_attach_point(graph_path, path, nodes[0], output_ids)
    if skipped:
        _report_skipped(graph, skipped, index, found_any=bool(points))
    return points


def build_branch_weights(
    layout: AdapterLayout,
    points: Mapping[str, AttachPoint],
    ckpt_dirs: Mapping[str, str | Path],
    adapter_scales: Mapping[str, float],
) -> dict[str, BranchWeights]:
    """
    Build each branch's weights from the checkpoints, without touching a graph.

    Each adapter's ``lora_A`` and ``lora_B`` are copied into its rank slice of ``a_weight`` and
    ``b_weight``; ``scale_weight`` is filled from the layout.

    The checkpoint's dtype must equal the base weight's dtype (no casting) and its shapes must
    match the layer.

    :param layout: Slices keyed by the graph's module path, like ``points``
    :return: Graph module path -> the branch's weights
    :raises ValueError: On a dtype or shape mismatch
    """
    result: dict[str, BranchWeights] = {}
    slices_by_adapter: dict[str, list[tuple[str, RankSlice]]] = {
        name: [] for name in layout.adapters
    }
    for path, slices in layout.slices_by_path.items():
        point = points[path]
        dtype = static_tensor(point.weight).dtype.numpy()
        a_shape, b_shape, scale_shape = _weight_shapes(point, layout.rank(path))
        scale = layout.scale_vector(path, adapter_scales).astype(dtype)
        result[path] = BranchWeights(
            np.zeros(a_shape, dtype),
            np.zeros(b_shape, dtype),
            scale.reshape(scale_shape),
        )
        for rank_slice in slices:
            slices_by_adapter[rank_slice.adapter].append((path, rank_slice))

    for name, targets in slices_by_adapter.items():
        with safe_open(
            str(Path(ckpt_dirs[name]) / WEIGHTS_FILE), framework="np"
        ) as ckpt:
            for path, rank_slice in targets:
                point = points[path]
                dtype = static_tensor(point.weight).dtype
                for which, dest in (
                    ("A", result[path].a_weight),
                    ("B", result[path].b_weight),
                ):
                    key = checkpoint_key(point.checkpoint_path, which)
                    ckpt_dtype = ckpt.get_slice(key).get_dtype()
                    if _SAFETENSORS_DTYPES.get(ckpt_dtype) != dtype:
                        raise ValueError(
                            f"Adapter {name!r}: {key} is {ckpt_dtype} but the base weight is "
                            f"{dtype.name}; re-save the adapter in the base model's dtype "
                            "(e.g. model.half() before save_pretrained)"
                        )
                    _write_slice(
                        name,
                        key,
                        which,
                        point.kind,
                        ckpt.get_tensor(key),
                        dest,
                        rank_slice,
                    )
    return result


def insert_branches(
    graph: onnx_ir.Graph,
    points: Mapping[str, AttachPoint],
    weights: Mapping[str, BranchWeights],
) -> None:
    """
    Add ``x -> A -> Mul(s) -> B -> Add(base_out, .)`` at every point, in place.

    Existing nodes, initializers and tensors keep their names; only a base output that is also a
    graph output is renamed (the Add takes the graph output name). The LoRA tensors are named
    after each point's ``module_path``.

    :param points: Graph module path -> attachment point
    :param weights: Graph module path -> the values of the branch's three weights
    :raises ValueError: If a name we would create already exists in the graph
    """
    # every name that a new node, tensor or initializer must not reuse
    taken = {v.name for v in graph.inputs} | {v.name for v in graph.outputs}
    taken |= set(graph.initializers)
    for node in graph:
        taken.add(node.name)
        taken.update(v.name for v in node.outputs)
    output_ids = {id(v) for v in graph.outputs}
    for path, point in points.items():
        names = branch_names(
            point.module_path,
            _node_prefix(point.node),
            point.kind,
            point.base_out.name,
            at_graph_output=id(point.base_out) in output_ids,
        )
        created = names.created_names()
        if names.base_layer_output != point.base_out.name:
            # the Add reuses the graph output's name, and the base output gets a new one
            created.remove(names.merged_output)
            created.append(names.base_layer_output)
        clash = sorted(set(created) & taken)
        if clash:
            raise ValueError(f"{path!r}: names already exist in the graph: {clash}")
        taken.update(created)
        dtype = static_tensor(point.weight).dtype
        a, b, s = (
            _initializer(graph, name, array, dtype)
            for name, array in (
                (names.a_weight, weights[path].a_weight),
                (names.b_weight, weights[path].b_weight),
                (names.scale_weight, weights[path].scale_weight),
            )
        )
        _insert_branch(graph, point, names, a, b, s)


def pull_branch_weights_to_inputs(model: onnx.ModelProto) -> list[LoraBranch]:
    """
    Make every branch's ``A`` and ``B`` initializers graph inputs, in place.

    The inputs keep the initializers' names, shapes and dtypes. Every branch's scale initializer
    is removed and its ``Mul`` reads one ``[1]`` graph input, ``LORA_SCALE_INPUT``, instead. That is
    only exact if every scale holds one value, as for a model built from one adapter.

    :param model: Model built by ``insert_branches``
    :return: The branches, as before the change
    :raises ValueError: If the model has no branches, their weights are not initializers, or the
        scales hold more than one value or dtype
    """
    graph_inputs = {i.name for i in model.graph.input}
    if LORA_SCALE_INPUT in graph_inputs:
        raise ValueError(
            f"The model's LoRA weights are already inputs ({LORA_SCALE_INPUT!r})"
        )
    branches = _find_lora_branches(model)
    if not branches:
        raise ValueError(
            "The model has no LoRA branches; build it with load_lora_adapters"
        )

    initializers = {i.name: i for i in model.graph.initializer}
    scale_values, scale_dtypes = set(), set()
    for branch in branches:
        scale = initializers[branch.scale_weight]
        scale_values.update(onnx.numpy_helper.to_array(scale).ravel().tolist())
        scale_dtypes.add(scale.data_type)
    if len(scale_values) != 1 or len(scale_dtypes) != 1:
        raise ValueError(
            "The LoRA scales hold several values; one lora_scale input can't replace them. "
            "Build the model from one adapter, without adapter_scales or pad_rank"
        )

    pulled = {n for b in branches for n in (b.a_weight, b.b_weight, b.scale_weight)}
    for branch in branches:
        for name in (branch.a_weight, branch.b_weight):
            weight = initializers[name]
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(name, weight.data_type, weight.dims)
            )
    model.graph.input.append(
        onnx.helper.make_tensor_value_info(LORA_SCALE_INPUT, scale_dtypes.pop(), (1,))
    )
    kept = [i for i in model.graph.initializer if i.name not in pulled]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    scale_nodes = {b.scale_node: b.scale_weight for b in branches}
    for node in model.graph.node:
        if node.name in scale_nodes:
            node.input[list(node.input).index(scale_nodes[node.name])] = (
                LORA_SCALE_INPUT
            )
    return branches


def get_lora_tensor_names(model: onnx.ModelProto) -> LoraTensorNames:
    """
    The LoRA tensors of a model built by ``load_lora_adapters``, in branch (Add node) order.

    Call it on the model ``load_lora_adapters`` returned, not on a sim's graph: QuantSim inserts
    quantize nodes between the branch tensors. The names are the same in the sim, so the result
    can be passed to the ``adapter_utils`` helpers with it.

    :param model: Model built by ``load_lora_adapters``
    :raises ValueError: If a branch is missing a node, tensor or initializer, or has a stray scale
    """
    branches = _find_lora_branches(model)
    return LoraTensorNames(
        params=tuple(n for b in branches for n in (b.a_weight, b.b_weight)),
        activations=tuple(
            n
            for b in branches
            for n in (b.a_proj_output, b.scaled_output, b.b_proj_output)
        ),
        # one entry per scale tensor: branches with LoRA inputs share one
        scales=tuple(dict.fromkeys(b.scale_weight for b in branches)),
        merges=tuple((b.base_layer_output, b.merged_output) for b in branches),
    )


def _find_lora_branches(model: onnx.ModelProto) -> list[LoraBranch]:
    """
    List the LoRA branches in a model built by ``load_lora_adapters``, in Add node order.

    Names are matched, not graph structure: every ``<node prefix>/lora/Add`` node marks a branch
    and all other names come from ``branch_names``. In a model built by
    ``load_lora_adapters_as_inputs`` the weights are graph inputs and every branch's scale is the
    ``lora_scale`` input.

    :param model: Model built by ``load_lora_adapters`` or ``load_lora_adapters_as_inputs``
    :raises ValueError: If a branch is missing a node, tensor or initializer, or has a stray scale
    """
    nodes = {n.name: n for n in model.graph.node}
    initializers = {i.name for i in model.graph.initializer}
    inputs = {i.name for i in model.graph.input}
    shared_scale = LORA_SCALE_INPUT if LORA_SCALE_INPUT in inputs else None
    branches = []
    for add in model.graph.node:
        if add.name.endswith(_ADD):
            branches.append(
                _find_branch(add, nodes, initializers | inputs, shared_scale)
            )

    stray = {i for i in initializers if i.endswith(_SCALE)} - {
        b.scale_weight for b in branches
    }
    if stray:
        raise ValueError(
            f"LoRA scale {sorted(stray)[0]!r} has no '<node prefix>{_ADD}' node"
        )
    return branches


def branch_names(
    module_path: str,
    node_prefix: str,
    kind: Literal["linear", "conv"],
    base_layer_output: str,
    at_graph_output: bool = False,
    scale_weight: str | None = None,
) -> LoraBranch:
    """
    The names of the branch for the layer at ``node_prefix``, merged into ``base_layer_output``.

    The merged output is ``<merge node>_output_0``, unless ``base_layer_output`` is a graph
    output: then the merge takes over that name (so the model's outputs keep theirs) and the base
    layer's output is renamed.

    :param scale_weight: The scale's name, if the branch shares one (``LORA_SCALE_INPUT``) instead
        of having ``<module path>.lora_scale``
    """
    op = "MatMul" if kind == "linear" else "Conv"
    a_proj_node = node_prefix + _A_NODE.format(op)
    scale_node = node_prefix + _MUL
    b_proj_node = node_prefix + _B_NODE.format(op)
    merge_node = node_prefix + _ADD
    a_weight, b_weight = lora_weight_names(module_path)
    return LoraBranch(
        module_path=module_path,
        node_prefix=node_prefix,
        kind=kind,
        a_weight=a_weight,
        b_weight=b_weight,
        scale_weight=scale_weight or module_path + _SCALE,
        a_proj_node=a_proj_node,
        scale_node=scale_node,
        b_proj_node=b_proj_node,
        merge_node=merge_node,
        a_proj_output=f"{a_proj_node}_output_0",
        scaled_output=f"{scale_node}_output_0",
        b_proj_output=f"{b_proj_node}_output_0",
        merged_output=base_layer_output
        if at_graph_output
        else f"{merge_node}_output_0",
        base_layer_output=(
            f"{node_prefix}/lora/base_output_0"
            if at_graph_output
            else base_layer_output
        ),
    )


def lora_weight_names(module_path: str) -> tuple[str, str]:
    """The names of the ``A`` and ``B`` weights of the branch at ``module_path``."""
    return module_path + _A_WEIGHT, module_path + _B_WEIGHT


def check_no_branches(graph: onnx_ir.Graph) -> None:
    """Raise if the graph already has LoRA branches, so that every LoRA name is ours."""
    node_suffixes = (_MUL, _ADD)
    node_suffixes += tuple(
        fragment.format(op)
        for fragment in (_A_NODE, _B_NODE)
        for op in ("MatMul", "Conv")
    )
    tensor_suffixes = (_A_WEIGHT, _B_WEIGHT, _SCALE)
    found = [n.name for n in graph if (n.name or "").endswith(node_suffixes)]
    found += [name for name in graph.initializers if name.endswith(tensor_suffixes)]
    # a load_lora_adapters_as_inputs model has its LoRA tensors as graph inputs
    found += [
        v.name
        for v in graph.inputs
        if v.name.endswith(tensor_suffixes) or v.name == LORA_SCALE_INPUT
    ]
    if found:
        raise ValueError(
            f"The model already contains LoRA branches (e.g. {found[0]!r}); "
            "pass the base model without LoRA"
        )


# ---------------------------------------------------------------------------------------------
# Helpers of resolve_targets: module paths from node names
# ---------------------------------------------------------------------------------------------


def _node_prefix(node: onnx_ir.Node) -> str:
    """The node name minus its ``/<OpType>`` or ``/<OpType>_<n>`` suffix: ``/layers.0/q_proj``."""
    return re.sub(rf"/{node.op_type}(_\d+)?$", "", node.name or "")


def _module_path_of_node_prefix(node_prefix: str) -> str:
    """
    Torch module path for a TorchScript-style node prefix.

    Drops the leading ``/``, then removes the redundant parent segment TorchScript puts in front
    of Sequential/ModuleList children (``/features/features.0`` -> ``features.0``).
    """
    segments = node_prefix.lstrip("/").split("/")
    out: list[str] = []
    for seg in segments:
        if out and seg.startswith(out[-1] + "."):
            out.pop()
        out.append(seg)
    return ".".join(out)


# ---------------------------------------------------------------------------------------------
# Helpers of resolve_targets: matching adapter paths to graph paths
# ---------------------------------------------------------------------------------------------


def _index_weight_nodes(graph: onnx_ir.Graph) -> dict[str, list[onnx_ir.Node]]:
    """Graph module path -> the MatMul/Gemm/Conv nodes with a static weight at that path."""
    index: dict[str, list[onnx_ir.Node]] = {}
    for node in graph:
        if node.domain in ("", "ai.onnx") and node.op_type in _WEIGHT_OPS:
            weight, _ = get_weight_value(node)
            if weight is not None:
                index.setdefault(
                    _module_path_of_node_prefix(_node_prefix(node)), []
                ).append(node)
    return index


@dataclass(frozen=True)
class PrefixMatch:
    """An adapter path that matches a graph path once one side's extra leading prefix is ignored."""

    path: str  # the adapter's module path
    graph_path: str  # the graph module path it matches
    side: Literal["graph", "adapter"]  # whose paths carry the extra prefix
    prefix: str  # that prefix, including the trailing dot


def _prefix_matches(
    module_paths: Iterable[str], index: dict[str, list]
) -> list[PrefixMatch]:
    """Every way each adapter path matches a graph path by ignoring a leading prefix."""
    # Index every proper dotted suffix of the graph paths once, so each adapter path is a lookup
    graph_by_suffix: dict[str, list[tuple[str, str]]] = {}
    for graph_path in index:
        parts = graph_path.split(".")
        for i in range(1, len(parts)):
            suffix = ".".join(parts[i:])
            prefix = ".".join(parts[:i]) + "."
            graph_by_suffix.setdefault(suffix, []).append((prefix, graph_path))

    matches = []
    for path in module_paths:
        for prefix, graph_path in graph_by_suffix.get(path, []):
            matches.append(PrefixMatch(path, graph_path, "graph", prefix))
        parts = path.split(".")
        for i in range(1, len(parts)):
            graph_path = ".".join(parts[i:])
            if graph_path in index:
                prefix = ".".join(parts[:i]) + "."
                matches.append(PrefixMatch(path, graph_path, "adapter", prefix))
    return matches


def _check_not_mixed_prefixes(module_paths: list[str], index: dict[str, list]) -> None:
    """
    Raise if some adapter paths match the graph exactly and others only through a prefix.

    That means two paths name the same layer differently (e.g. ``layers.0.q_proj`` from one
    adapter and ``model.layers.0.q_proj`` from another), usually because the adapters were
    trained on different base models.
    """
    exact = {path for path in module_paths if path in index}
    if not exact:
        return
    inexact = [path for path in module_paths if path not in exact]
    for match in _prefix_matches(inexact, index):
        if match.graph_path in exact:
            raise ValueError(
                f"Module paths {match.path!r} and {match.graph_path!r} refer to the same layer "
                "through different prefixes; the adapters use different module names"
            )


def _find_prefix_shift(
    module_paths: list[str], index: dict[str, list]
) -> dict[str, str]:
    """
    Map adapter paths to graph paths when they differ by one shared leading prefix.

    Nothing matched exactly, so look for graph paths ``<prefix>.<adapter path>`` (the graph has an
    extra prefix) or adapter paths ``<prefix>.<graph path>`` (the adapter has one). The shift is
    only used if every path that matches at all does so through the same prefix on the same side
    and has exactly one match. Paths with no match keep their own name and resolve as missing.

    :raises ValueError: If the paths match through different prefixes, or one path is ambiguous
    """
    matches = _prefix_matches(module_paths, index)

    by_path: dict[str, list[PrefixMatch]] = {}
    for match in matches:
        by_path.setdefault(match.path, []).append(match)
    for path, found in by_path.items():
        if len(found) > 1:
            shifts = sorted((m.side, m.prefix) for m in found)
            raise ValueError(
                f"Module path {path!r} matches several graph paths when a prefix is ignored: "
                f"{shifts}"
            )

    shifts = {(m.side, m.prefix) for m in matches}
    if len(shifts) > 1:
        raise ValueError(
            "The adapter's module paths match the graph's through different prefixes: "
            f"{sorted(shifts)}. Module names in the graph must match the adapter's."
        )
    if shifts:
        side, prefix = next(iter(shifts))
        _logger.warning(
            "The %s has an extra leading %r on its module paths; matching the adapter with it",
            side,
            prefix,
        )
    matched = {m.path: m.graph_path for m in matches}
    return {path: matched.get(path, path) for path in module_paths}


def _report_skipped(
    graph: onnx_ir.Graph, skipped: list[str], index: dict[str, list], found_any: bool
) -> None:
    """Warn about the skipped paths, or raise if no path matched at all (a wrong model)."""
    if not found_any:
        raise ValueError(_missing_message(graph, skipped[0], index))
    hint = ""
    matches = _prefix_matches(skipped, index)
    if matches:
        hint = (
            f" They match when the {matches[0].side}'s paths carry a leading "
            f"{matches[0].prefix!r}, which was not applied because other paths already match "
            "exactly"
        )
    _logger.warning(
        "Skipping %d adapter layer(s) not in the base model, e.g. %s.%s",
        len(skipped),
        skipped[0],
        hint,
    )


def _missing_message(graph: onnx_ir.Graph, path: str, index: dict[str, list]) -> str:
    """The error for a path with no match: close matches, plus a hint for unfixed dynamo names."""
    msg = f"No MatMul/Gemm/Conv with a static weight matches module path {path!r}."
    near = difflib.get_close_matches(path, list(index), n=5)
    if near:
        msg += f" Closest paths: {near}."
    if any(
        node.op_type in _WEIGHT_OPS
        and "namespace" in node.metadata_props
        and not (node.name or "").startswith("/")
        and get_weight_value(node)[0] is not None
        for node in graph
    ):
        msg += (
            " The graph has unfixed dynamo node names; call fix_node_names_pass first."
        )
    return msg


# ---------------------------------------------------------------------------------------------
# Helpers of resolve_targets: describing the layer a branch attaches to
# ---------------------------------------------------------------------------------------------


def _make_attach_point(
    path: str, checkpoint_path: str, node: onnx_ir.Node, output_ids: set[int]
) -> AttachPoint:
    """Check the node can take a branch and describe it as an ``AttachPoint``."""
    weight, transposed = get_weight_value(node)
    shape = tuple(static_tensor(weight).shape)

    if node.op_type == "Conv":
        _check_conv(path, node, shape)
        kind = "conv"
        in_features, out_features = shape[1], shape[0]
        base_out = node.outputs[0]
    else:
        if node.op_type == "Gemm" and node.attributes.get_int("transA", 0) != 0:
            raise ValueError(f"{path!r}: Gemm with transA=1 is not supported")
        if len(shape) != 2:
            raise ValueError(f"{path!r}: expected a 2-D weight, got shape {shape}")
        kind = "linear"
        out_features, in_features = shape if transposed else shape[::-1]
        bias_add_output = None
        if node.op_type == "MatMul":
            bias_add_output = _bias_add_output(node, out_features, output_ids)
        base_out = bias_add_output or node.outputs[0]

    return AttachPoint(
        module_path=path,
        checkpoint_path=checkpoint_path,
        node=node,
        weight=weight,
        kind=kind,
        in_features=in_features,
        out_features=out_features,
        base_out=base_out,
    )


def _check_conv(path: str, node: onnx_ir.Node, shape: tuple[int, ...]) -> None:
    """Raise if the Conv is one a single A/B branch can't reproduce (grouped, dilated, auto_pad)."""
    if len(shape) < 3:
        raise ValueError(
            f"{path!r}: expected a Conv weight [Cout, Cin, *kernel], got {shape}"
        )
    if node.attributes.get_int("group", 1) != 1:
        raise ValueError(f"{path!r}: grouped Conv is not supported")
    dilations = node.attributes.get("dilations")
    if dilations is not None and any(v != 1 for v in dilations.as_ints()):
        raise ValueError(f"{path!r}: dilated Conv is not supported")
    auto_pad = node.attributes.get("auto_pad")
    if auto_pad is not None and auto_pad.as_string() != "NOTSET":
        raise ValueError(
            f"{path!r}: Conv with auto_pad={auto_pad.as_string()!r} is not supported"
        )


def _bias_add_output(
    node: onnx_ir.Node, m: int, output_ids: set[int]
) -> onnx_ir.Value | None:
    """The output of ``MatMul -> Add(bias)`` if that Add is this projection's bias, else None."""
    out = node.outputs[0]
    uses = out.uses()
    if len(uses) != 1 or id(out) in output_ids:
        return None
    add = uses[0].node
    if add.op_type != "Add" or add.domain not in ("", "ai.onnx"):
        return None
    if _node_prefix(add) != _node_prefix(node):
        return None
    other = add.inputs[1 - uses[0].idx]
    tensor = static_tensor(other)
    if tensor is None or tuple(tensor.shape) != (m,):
        return None
    return add.outputs[0]


# ---------------------------------------------------------------------------------------------
# Helpers of build_branch_weights
# ---------------------------------------------------------------------------------------------


def _weight_shapes(point: AttachPoint, rank: int):
    """Shapes of ``(a_weight, b_weight, scale_weight)`` for total rank ``rank``."""
    if point.kind == "linear":
        return (point.in_features, rank), (rank, point.out_features), (rank,)
    kernel = tuple(static_tensor(point.weight).shape[2:])
    ones = (1,) * len(kernel)
    return (
        (rank, point.in_features, *kernel),
        (point.out_features, rank, *ones),
        (1, rank, *ones),
    )


def _write_slice(
    adapter: str,
    key: str,
    which: Literal["A", "B"],
    kind: Literal["linear", "conv"],
    tensor: np.ndarray,
    dest: np.ndarray,
    rank_slice: RankSlice,
) -> None:
    """Copy one checkpoint tensor into its rank slice of ``dest``, in the branch's layout."""
    ranks = slice(rank_slice.offset, rank_slice.offset + rank_slice.r)
    if kind == "conv":
        # peft A [r, Cin, *k] -> A[ranks]; peft B [Cout, r, 1, ...] -> B[:, ranks]
        view = dest[ranks] if which == "A" else dest[:, ranks]
        value = tensor
    else:
        # peft A [r, d] -> A[:, ranks]; peft B [m, r] -> B[ranks, :]
        view = dest[:, ranks] if which == "A" else dest[ranks, :]
        value = tensor.T
    if view.shape != value.shape:
        raise ValueError(
            f"Adapter {adapter!r}: {key} has shape {tuple(tensor.shape)}, which doesn't fit the "
            f"base layer (expected {tuple(view.shape if kind == 'conv' else view.shape[::-1])})"
        )
    view[...] = value


# ---------------------------------------------------------------------------------------------
# Helpers of insert_branches
# ---------------------------------------------------------------------------------------------


def _initializer(
    graph: onnx_ir.Graph, name: str, array: np.ndarray, dtype: onnx_ir.DataType
) -> onnx_ir.Value:
    """Register ``array`` as a graph initializer called ``name`` and return its value."""
    value = onnx_ir.Value(
        name=name,
        shape=onnx_ir.Shape(array.shape),
        type=onnx_ir.TensorType(dtype),
        const_value=onnx_ir.tensor(array, dtype=dtype, name=name),
    )
    graph.register_initializer(value)
    return value


def _insert_branch(
    graph: onnx_ir.Graph,
    point: AttachPoint,
    names: LoraBranch,
    a: onnx_ir.Value,
    b: onnx_ir.Value,
    s: onnx_ir.Value,
) -> None:
    """Insert one branch at ``point`` that reads ``a``, ``s`` and ``b``, and rewire the merge."""
    dtype = static_tensor(point.weight).dtype
    x = point.node.inputs[0]
    op = "Conv" if point.kind == "conv" else "MatMul"
    a_attrs = {}
    if point.kind == "conv":
        a_attrs = {
            k: point.node.attributes[k].as_ints()
            for k in ("strides", "pads")
            if k in point.node.attributes
        }
    proj_a = onnx_ir.node(op, [x, a], a_attrs, name=names.a_proj_node)
    mul = onnx_ir.node("Mul", [proj_a.outputs[0], s], name=names.scale_node)
    proj_b = onnx_ir.node(op, [mul.outputs[0], b], name=names.b_proj_node)
    for node, out in zip(
        (proj_a, mul, proj_b),
        (names.a_proj_output, names.scaled_output, names.b_proj_output),
    ):
        node.outputs[0].name = out
        node.outputs[0].type = onnx_ir.TensorType(dtype)

    base_out = point.base_out
    add_out = onnx_ir.Value(type=base_out.type, shape=base_out.shape)
    base_out.replace_all_uses_with(add_out, replace_graph_outputs=True)
    base_out.name, add_out.name = names.base_layer_output, names.merged_output
    add = onnx_ir.node(
        "Add", [base_out, proj_b.outputs[0]], outputs=[add_out], name=names.merge_node
    )
    graph.insert_after(base_out.producer(), [proj_a, mul, proj_b, add])


# ---------------------------------------------------------------------------------------------
# Helpers of _find_lora_branches
# ---------------------------------------------------------------------------------------------


def _find_branch(add, nodes, tensors, shared_scale=None) -> LoraBranch:
    """Build the ``LoraBranch`` for one ``<node prefix>/lora/Add`` node; check its names exist.

    :param tensors: Names of the graph's initializers and inputs
    :param shared_scale: The scale every branch uses, if they share one
    """
    node_prefix = add.name[: -len(_ADD)]
    path = _module_path_of_node_prefix(node_prefix)
    kind = "conv" if node_prefix + _A_NODE.format("Conv") in nodes else "linear"
    # the Add's output and input[0] are read from the graph: they differ when at a graph output
    names = branch_names(
        path, node_prefix, kind, add.input[0], scale_weight=shared_scale
    )
    names = dataclasses.replace(names, merged_output=add.output[0])

    missing_nodes = [
        n
        for n in (names.a_proj_node, names.scale_node, names.b_proj_node)
        if n not in nodes
    ]
    if missing_nodes:
        raise ValueError(f"LoRA branch {path!r}: node {missing_nodes[0]!r} not found")
    missing_tensors = [
        i
        for i in (names.a_weight, names.b_weight, names.scale_weight)
        if i not in tensors
    ]
    if missing_tensors:
        raise ValueError(
            f"LoRA branch {path!r}: initializer or input {missing_tensors[0]!r} not found"
        )
    expected_outputs = {
        names.a_proj_node: names.a_proj_output,
        names.scale_node: names.scaled_output,
        names.b_proj_node: names.b_proj_output,
    }
    for node_name, output in expected_outputs.items():
        if list(nodes[node_name].output) != [output]:
            raise ValueError(
                f"LoRA branch {path!r}: {node_name!r} must output {output!r}"
            )
    if add.input[1] != names.b_proj_output:
        raise ValueError(
            f"LoRA branch {path!r}: Add input[1] must be {names.b_proj_output!r}"
        )
    return names
