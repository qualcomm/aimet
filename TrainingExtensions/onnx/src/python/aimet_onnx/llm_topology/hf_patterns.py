# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Find an LLM's named layers by their HuggingFace module paths.

For a known HuggingFace ``model_type`` the module paths of a decoder stack are
fixed by the modeling code, so every topology field that corresponds to a named
module (the attention and MLP projections, the decoder norms, ``embed_tokens``,
``lm_head``) can be looked up by name instead of inferred from graph structure.
The decoder layer index is read off the same path (``layers.<N>``), which is
what assigns each match to a block.

Module paths from node names
----------------------------
Both supported exporters name a node ``/<module path>/<op>``, where each
``/``-segment is one module attribute and a container index stays attached to
its container (``layers.0``, ``heads.0.1``):

* **torchscript** — ``/model/layers.0/self_attn/q_proj/MatMul``.
* **dynamo** — names nodes ``node_linear_N``, but records the module hierarchy in
  node metadata. Run
  :func:`~aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx.fix_node_names_pass`
  first to restore ``/<module path>/<op>`` names; unfixed dynamo exports are
  rejected with instructions.

:func:`module_path_of` joins the segments with ``.``, giving the spelling
``nn.Module.named_modules()`` uses, which is what the patterns are written in.

Failing loudly
--------------
Node names follow the module *call* stack rather than module ownership, and a
wrong ``model_type`` can look plausible, so matching is strict and reports every
problem instead of guessing:

* All decoder layers must sit under one prefix. A second ``layers.<N>`` stack
  (e.g. a VLM vision tower) is an error, not extra blocks.
* Every weighted linear and RMSNorm inside a decoder layer must be named by the
  table (or listed as ``ignored``). This catches a wrong ``model_type`` (e.g.
  Phi-3's fused ``qkv_proj`` or Gemma's extra norms under the ``llama`` table)
  and a module called again from its
  parent, which torchscript names ``<parent>/<module>`` without the module's
  own attribute path.
* A node outside the decoder stack whose module is named like a table entry
  (e.g. ``down_proj`` at the root, left by a torchscript export of a module
  called directly rather than through its parents) is an error.
* Under a decoder prefix, every other node must sit under that prefix or be
  ``lm_head``. This catches a VLM vision tower or projector exported alongside
  the language backbone, whatever its modules are named (Qwen3-VL's vision
  blocks are ``blocks.<N>`` with ``attn.qkv`` / ``linear_fc1``).
* Layer indices must be contiguous, and each field must match exactly one node
  per layer. Under dynamo, a module called twice has two nodes.

One quirk is allowed for rather than rejected: an HF VLM looks up token
embeddings from its outer model (``get_input_embeddings()(input_ids)``), so
torchscript names a ``<X>.language_model.embed_tokens`` call ``<X>.embed_tokens``.
For a decoder under ``<X>.language_model``, ``embed_tokens`` matches at either.

One case name matching cannot see: torchscript drops a ``ModuleDict`` level, so
``mlp.proj.gate_proj`` exports as ``mlp/gate_proj``. Standard HF decoders have
no ``ModuleDict`` on these paths; the structural cross-checks in
:func:`~.topology.analyze_llm_topology` are the safety net for the rest.

.. note::
   Per-head split (SHA) exports, whose projections are named e.g.
   ``q_proj.heads.0``, are not supported yet.
"""

import re
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import onnx_ir

from aimet_onnx.graph_passes.fusions import is_fused_supergroup
from aimet_onnx.ir_utils import is_static

from aimet_onnx.llm_topology import ir_analysis
from aimet_onnx.llm_topology.layer_roles import LinearRole
from aimet_onnx.llm_topology.topology_types import BlockKind

#: Offending nodes listed per problem in an error message.
_MAX_LISTED = 5

#: De-duplication suffix an exporter appends to a repeated module name.
_DEDUP_SUFFIX = re.compile(r"_\d+$")

#: Name the torch dynamo exporter gives a node: ``node_<op>`` or ``node_<op>_<N>``
#: (e.g. ``node_linear_3``, ``node_MatMul_2``), with no module path in it.
_DYNAMO_NODE_NAME = re.compile(r"^node_[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$")


class ModuleKind(Enum):
    """What a matchable node computes. A table entry only matches its own kind."""

    LINEAR = "linear"
    NORM = "norm"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class ModuleNode:
    """A graph node that may correspond to a named HF module.

    :param node_name: Name of the node in the graph.
    :param module_path: Dotted path of the module the node was exported from.
    :param kind: What the node computes.
    """

    node_name: str
    module_path: str
    kind: ModuleKind


@dataclass(frozen=True)
class HfModelPatterns:
    """Module paths of the named layers of one HF architecture.

    Per-layer paths are relative to one decoder layer
    (``<decoder prefix>.<decoder_layers>.<N>``); ``embed_tokens`` and
    ``final_norm`` are relative to the decoder prefix; ``lm_head`` is matched
    under any prefix. The decoder prefix itself is not part of the table — it is
    whatever precedes ``<decoder_layers>.<N>`` (e.g. ``model``,
    ``model.language_model``), and must be the same for every match.

    Every field defaults to the Llama module name, so an entry spells out only
    where an architecture differs. An architecture with a fused projection sets
    ``qkv_proj`` (``gate_up_proj``), which then replaces ``q_proj`` / ``k_proj`` /
    ``v_proj`` (``gate_proj`` / ``up_proj``) under the corresponding ``FUSED_*``
    role, and the split fields are then not used. Overriding both a fused field
    and any of its split fields is contradictory, and raises.

    :param q_proj: Per-layer query projection. Unused when ``qkv_proj`` is set.
    :param k_proj: Per-layer key projection. Unused when ``qkv_proj`` is set.
    :param v_proj: Per-layer value projection. Unused when ``qkv_proj`` is set.
    :param o_proj: Per-layer attention output projection.
    :param gate_proj: Per-layer MLP gate projection. Unused when ``gate_up_proj``
        is set.
    :param up_proj: Per-layer MLP up projection. Unused when ``gate_up_proj`` is
        set.
    :param down_proj: Per-layer MLP down projection.
    :param qkv_proj: Per-layer fused query/key/value projection, if fused.
    :param gate_up_proj: Per-layer fused MLP gate/up projection, if fused.
    :param input_norm: Per-layer pre-attention norm.
    :param post_attention_norm: Per-layer pre-MLP norm.
    :param ignored: Per-layer modules that are known to exist but belong to no
        topology field (e.g. Qwen3's ``self_attn.q_norm``).
    :param decoder_layers: Attribute holding the decoder layer ``ModuleList``.
    :param embed_tokens: Token embedding.
    :param final_norm: Norm after the last decoder layer.
    :param lm_head: Vocabulary projection.
    :param block_kind: What every decoder layer computes. For
        :attr:`~.topology_types.BlockKind.MAMBA` the attention and MLP fields are
        unused, and a layer is one ``input_norm`` (the block's only norm) feeding
        ``mixer_in_proj``, with ``mixer_out_proj`` writing the residual;
        ``post_attention_norm`` is unused.
    :param mixer_in_proj: Per-layer Mamba mixer input projection.
    :param mixer_out_proj: Per-layer Mamba mixer output projection.
    :param language_model: Attribute a VLM keeps its language backbone under
        (``<outer>.language_model``). When the decoder stack sits under it, the
        outer model calls ``embed_tokens`` itself, so an exporter may name that
        node after the outer model; ``embed_tokens`` is then also accepted one
        level up.
    :raises ValueError: If a fused projection is set together with an override
        of any of the split projections it replaces.
    """

    q_proj: str = "self_attn.q_proj"
    k_proj: str = "self_attn.k_proj"
    v_proj: str = "self_attn.v_proj"
    o_proj: str = "self_attn.o_proj"
    gate_proj: str = "mlp.gate_proj"
    up_proj: str = "mlp.up_proj"
    down_proj: str = "mlp.down_proj"
    qkv_proj: Optional[str] = None
    gate_up_proj: Optional[str] = None
    input_norm: str = "input_layernorm"
    post_attention_norm: str = "post_attention_layernorm"
    ignored: Tuple[str, ...] = ()
    decoder_layers: str = "layers"
    embed_tokens: str = "embed_tokens"
    final_norm: str = "norm"
    lm_head: str = "lm_head"
    language_model: str = "language_model"
    block_kind: BlockKind = BlockKind.ATTENTION
    mixer_in_proj: str = "mixer.in_proj"
    mixer_out_proj: str = "mixer.out_proj"

    def __post_init__(self):
        self._check_fused_excludes_split("qkv_proj", ("q_proj", "k_proj", "v_proj"))
        self._check_fused_excludes_split("gate_up_proj", ("gate_proj", "up_proj"))

    def _check_fused_excludes_split(self, fused: str, split: Tuple[str, ...]):
        """Reject a fused projection set together with an override of its split ones.

        Read-only: the split fields keep their declared defaults; a set fused field
        simply takes precedence over them in :attr:`linears`.
        """
        if getattr(self, fused) is None:
            return
        defaults = {f.name: f.default for f in dataclasses.fields(self)}
        overridden = [name for name in split if getattr(self, name) != defaults[name]]
        if overridden:
            raise ValueError(
                f"HfModelPatterns: {fused} replaces {', '.join(split)}, but "
                f"{', '.join(overridden)} {'is' if len(overridden) == 1 else 'are'} "
                "also overridden. Set one or the other."
            )

    @property
    def norms(self) -> Dict[str, str]:
        """Per-layer norm paths, by :class:`BlockMatch` field."""
        if self.block_kind is BlockKind.MAMBA:
            return {"input_norm": self.input_norm}
        return {
            "input_norm": self.input_norm,
            "post_attention_norm": self.post_attention_norm,
        }

    @property
    def linears(self) -> Dict[LinearRole, str]:
        """Per-layer projection paths, by role."""
        if self.block_kind is BlockKind.MAMBA:
            return {
                LinearRole.MIXER_IN_PROJ: self.mixer_in_proj,
                LinearRole.MIXER_OUT_PROJ: self.mixer_out_proj,
            }
        if self.qkv_proj is not None:
            attention = {LinearRole.FUSED_QKV: self.qkv_proj}
        else:
            attention = {
                LinearRole.Q_PROJ: self.q_proj,
                LinearRole.K_PROJ: self.k_proj,
                LinearRole.V_PROJ: self.v_proj,
            }
        if self.gate_up_proj is not None:
            mlp = {LinearRole.FUSED_GATE_UP: self.gate_up_proj}
        else:
            mlp = {
                LinearRole.GATE_PROJ: self.gate_proj,
                LinearRole.UP_PROJ: self.up_proj,
            }
        return {
            **attention,
            LinearRole.O_PROJ: self.o_proj,
            **mlp,
            LinearRole.DOWN_PROJ: self.down_proj,
        }


@dataclass
class BlockMatch:
    """Named layers of one decoder layer, by node name.

    :param layer_id: Decoder layer index read off the module path.
    :param linears: Node(s) of each projection role, in topological order.
    :param input_norm: The block's first norm: pre-attention, or a Mamba block's
        only norm.
    :param post_attention_norm: The pre-MLP norm node; ``None`` for a Mamba block.
    """

    layer_id: int
    linears: Dict[LinearRole, List[str]]
    input_norm: str
    post_attention_norm: Optional[str] = None


@dataclass
class NamedLayerMatch:
    """Every named layer of a model, validated.

    :param decoder_prefix: Module path the decoder layers sit under (``""`` at
        the root).
    :param blocks: One entry per decoder layer, by layer index.
    :param embed_tokens: Token-embedding ``Gather`` node, if exported.
    :param final_norm: Final norm node, if exported.
    :param lm_head: Vocabulary projection node, if exported.
    """

    decoder_prefix: str
    blocks: List[BlockMatch] = field(default_factory=list)
    embed_tokens: Optional[str] = None
    final_norm: Optional[str] = None
    lm_head: Optional[str] = None


class NamedLayerMatchError(ValueError):
    """The model's layers could not be matched by name.

    :param problems: Every problem found, one sentence each.
    """

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__(
            "Could not identify the model's layers by HuggingFace module name:\n"
            + "\n".join(f"  - {problem}" for problem in self.problems)
        )


_LLAMA_PATTERNS = HfModelPatterns()

_QWEN3_PATTERNS = HfModelPatterns(
    # Per-head norms on Q and K, applied before RoPE.
    ignored=("self_attn.q_norm", "self_attn.k_norm"),
)

_GEMMA2_PATTERNS = HfModelPatterns(
    # Gemma norms both sides of each sub-block. The norms *before* attention and
    # the MLP are the ones the topology wants; HF's post_attention_layernorm here
    # is the attention-output norm, ahead of the residual Add.
    post_attention_norm="pre_feedforward_layernorm",
    ignored=("post_attention_layernorm", "post_feedforward_layernorm"),
)

_GEMMA3_PATTERNS = HfModelPatterns(
    post_attention_norm=_GEMMA2_PATTERNS.post_attention_norm,
    ignored=(*_QWEN3_PATTERNS.ignored, *_GEMMA2_PATTERNS.ignored),
)

_PHI3_PATTERNS = HfModelPatterns(
    qkv_proj="self_attn.qkv_proj",
    gate_up_proj="mlp.gate_up_proj",
)

_MAMBA2_PATTERNS = HfModelPatterns(
    block_kind=BlockKind.MAMBA,
    # Mamba2ForCausalLM keeps its stack under ``backbone`` and names the embedding
    # and final norm ``embeddings`` / ``norm_f``. Each layer's only decoder norm is
    # ``norm``; the mixer's own gated RMSNorm is internal to the scan, and the
    # depthwise ``conv1d`` is a weighted Conv that belongs to no topology field.
    input_norm="norm",
    embed_tokens="embeddings",
    final_norm="norm_f",
    ignored=("mixer.conv1d", "mixer.norm"),
)

#: Built-in patterns, keyed by HF ``PretrainedConfig.model_type``. For a VLM, key
#: on the text config's ``model_type``.
_HF_MODEL_PATTERNS: Dict[str, HfModelPatterns] = {
    "gemma2": _GEMMA2_PATTERNS,
    # Gemma 3 text models, and the text backbone of Gemma 3 VLMs.
    "gemma3_text": _GEMMA3_PATTERNS,
    "llama": _LLAMA_PATTERNS,
    # Mamba2 in transformers format (Mamba2ForCausalLM), e.g. Mamba-Codestral.
    # Original mamba_ssm checkpoints (e.g. state-spaces/mamba2-*) must be
    # converted to this format first.
    "mamba2": _MAMBA2_PATTERNS,
    # Phi-3, Phi-3.5 and Phi-4 (incl. -mini) text models.
    "phi3": _PHI3_PATTERNS,
    # Qwen2 and Qwen2.5. The q/k/v bias is a separate Add, which is not matched.
    "qwen2": _LLAMA_PATTERNS,
    "qwen2_5_vl_text": _LLAMA_PATTERNS,
    "qwen3": _QWEN3_PATTERNS,
    "qwen3_vl_text": _QWEN3_PATTERNS,
}


def get_hf_model_patterns(model_type: str) -> HfModelPatterns:
    """Return the built-in patterns for HF ``model_type``.

    :raises ValueError: If ``model_type`` has no built-in patterns.
    """
    patterns = _HF_MODEL_PATTERNS.get(model_type)
    if patterns is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported model types: "
            f"{sorted(_HF_MODEL_PATTERNS)}."
        )
    return patterns


def module_path_of(node: onnx_ir.Node) -> str:
    """Return the dotted module path ``node`` was exported from.

    A regular node is named ``/<module path>/<op>``, so its last segment is
    dropped. A fused supergroup node (e.g. an ``RMSNormalization``) is named
    after the module path itself, or ``/<module path>/<op type>`` when several
    supergroups share a module (see ``fuse_supergroups``).

    ``/model/layers.0/self_attn/q_proj/MatMul`` -> ``model.layers.0.self_attn.q_proj``
    """
    segments = (node.name or "").strip("/").split("/")
    if not is_fused_supergroup(node) or segments[-1] == node.op_type:
        segments = segments[:-1]
    return ".".join(segments)


def module_nodes_of(ir_model: onnx_ir.Model) -> List[ModuleNode]:
    """Return every node of ``ir_model`` that a table entry could name.

    Only nodes whose op fits some :class:`ModuleKind` are returned: a weighted
    linear, a fused RMSNorm, or an embedding-table ``Gather``. This drops the
    other ops a module exports, e.g. the bias ``Add`` under ``q_proj``.
    """
    nodes = []
    for node in ir_model.graph:
        if not node.name:
            continue
        if ir_analysis.is_weighted_linear(node) or _is_shared_weight_linear(node):
            kind = ModuleKind.LINEAR
        elif ir_analysis.is_rms_norm(node):
            kind = ModuleKind.NORM
        elif ir_analysis.is_embedding_table_gather(node):
            kind = ModuleKind.EMBEDDING
        else:
            continue
        nodes.append(ModuleNode(node.name, module_path_of(node), kind))
    return nodes


def _is_shared_weight_linear(node: onnx_ir.Node) -> bool:
    """Return True for a linear whose weight is a static tensor behind ``Identity`` ops.

    torchscript exports every call of a module after the first with its weight
    routed through an ``Identity``. Such a node is not a weighted linear to the
    rest of the analysis, but it is still a module call that must be accounted
    for, or a repeated call would go unnoticed.
    """
    if node.op_type not in ir_analysis.LINEAR_TYPES:
        return False
    if len(node.inputs) <= ir_analysis.WEIGHT_INDEX:
        return False
    value = node.inputs[ir_analysis.WEIGHT_INDEX]
    hops = 0
    while value is not None and not is_static(value):
        producer = value.producer()
        if producer is None or producer.op_type != "Identity":
            return False
        value, hops = producer.inputs[0], hops + 1
    return value is not None and hops > 0


def match_named_layers(
    ir_model: onnx_ir.Model, patterns: HfModelPatterns
) -> NamedLayerMatch:
    """Find and validate every layer of ``ir_model`` that ``patterns`` names.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param patterns: Patterns for the model's architecture.
    :raises NamedLayerMatchError: If the graph still carries dynamo node names,
        or on any problem listed in the module docstring.
    """
    nodes = module_nodes_of(ir_model)
    if any(_DYNAMO_NODE_NAME.match(node.node_name) for node in nodes):
        raise NamedLayerMatchError(
            [
                "this looks like a torch dynamo export, whose nodes are named "
                "'node_<op>_<N>' rather than after their modules. Restore "
                "module-path names before analyzing:\n\n"
                "    from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx "
                "import fix_node_names_pass\n"
                "    model = fix_node_names_pass(model)\n"
            ]
        )
    return match_module_nodes(nodes, patterns)


def match_module_nodes(
    nodes: Iterable[ModuleNode], patterns: HfModelPatterns
) -> NamedLayerMatch:
    """Match ``nodes`` against ``patterns`` and validate the result.

    Graph-free core of :func:`match_named_layers`, so the patterns can also be
    checked against a torch model's ``named_modules()``.

    :param nodes: Candidate nodes, in topological order.
    :param patterns: Patterns for the model's architecture.
    :raises NamedLayerMatchError: On any problem listed in the module docstring.
    """
    nodes = list(nodes)
    layer_pattern = re.compile(
        rf"^(?:(?P<prefix>.+)\.)?{re.escape(patterns.decoder_layers)}\.(?P<layer>\d+)\."
        r"(?P<rest>.+)$"
    )
    per_layer_fields = {
        ModuleKind.LINEAR: {path: role for role, path in patterns.linears.items()},
        ModuleKind.NORM: {path: key for key, path in patterns.norms.items()},
    }

    # decoder prefix -> layer id -> field -> node names
    stacks: Dict[str, Dict[int, Dict[object, List[str]]]] = {}
    # Every prefix with a ``<decoder_layers>.<N>`` child, whether or not any node
    # under it is named by the table: a second stack (e.g. a VLM vision tower) whose
    # module names this model_type does not know must still count as a stack.
    all_stacks: Dict[str, set] = {}
    # (decoder prefix, node) for nodes inside a decoder layer that no field names
    unrecognized: List[Tuple[str, ModuleNode]] = []
    outside: List[ModuleNode] = []
    for node in nodes:
        match = layer_pattern.match(node.module_path)
        if match is None:
            outside.append(node)
            continue
        prefix = match["prefix"] or ""
        all_stacks.setdefault(prefix, set()).add(int(match["layer"]))
        key = per_layer_fields.get(node.kind, {}).get(match["rest"])
        if key is not None:
            layer = stacks.setdefault(prefix, {}).setdefault(int(match["layer"]), {})
            layer.setdefault(key, []).append(node.node_name)
        elif match["rest"] not in patterns.ignored:
            unrecognized.append((prefix, node))

    if not stacks:
        examples = [n.module_path for n in nodes if n.kind is ModuleKind.LINEAR]
        raise NamedLayerMatchError(
            [
                "no decoder layer was found. Check that model_type matches the "
                "exported architecture and that node names carry module paths "
                "(e.g. '/model/layers.0/self_attn/q_proj/MatMul'). Module paths of "
                f"the first weighted linears: {examples[:_MAX_LISTED]}."
            ]
        )
    if len(all_stacks) > 1:
        raise NamedLayerMatchError(
            [
                f"found {len(all_stacks)} stacks of '{patterns.decoder_layers}.<N>' "
                "layers, expected one: "
                + ", ".join(
                    f"'{prefix}' (layers {sorted(layer_ids)})"
                    for prefix, layer_ids in all_stacks.items()
                )
                + ". Export the language backbone on its own."
            ]
        )
    decoder_prefix, layers = next(iter(stacks.items()))

    problems: List[str] = []
    in_decoder = [node for _, node in unrecognized]
    if in_decoder:
        problems.append(
            "these weighted linears / norms inside decoder layers are not modules "
            "of this model_type. Either model_type does not match the model, or a "
            "module was called from outside its own forward (node names follow the "
            "call stack): " + _listed(in_decoder)
        )

    result = NamedLayerMatch(decoder_prefix=decoder_prefix)
    problems += _match_model_level(outside, patterns, result)
    problems += _validate_layers(layers, patterns)
    if problems:
        raise NamedLayerMatchError(problems)

    for layer_id in sorted(layers):
        fields = layers[layer_id]
        result.blocks.append(
            BlockMatch(
                layer_id=layer_id,
                linears={role: fields[role] for role in patterns.linears},
                input_norm=fields["input_norm"][0],
                post_attention_norm=fields.get("post_attention_norm", [None])[0],
            )
        )
    return result


def _match_model_level(
    outside: List[ModuleNode], patterns: HfModelPatterns, result: NamedLayerMatch
) -> List[str]:
    """Fill the model-level fields of ``result`` from nodes outside the decoder layers.

    :return: Problems found.
    """
    prefix = result.decoder_prefix

    def under_decoder(path: str) -> str:
        return f"{prefix}.{path}" if prefix else path

    embed_paths = {under_decoder(patterns.embed_tokens)}
    # A VLM's outer model calls its language backbone's embed_tokens directly.
    outer, _, leaf = prefix.rpartition(".")
    if leaf == patterns.language_model:
        embed_paths.add(
            f"{outer}.{patterns.embed_tokens}" if outer else patterns.embed_tokens
        )

    model_level = {
        "embed_tokens": (ModuleKind.EMBEDDING, embed_paths),
        "final_norm": (ModuleKind.NORM, {under_decoder(patterns.final_norm)}),
    }
    matches: Dict[str, List[str]] = {name: [] for name in (*model_level, "lm_head")}
    leftovers: List[ModuleNode] = []
    for node in outside:
        for name, (kind, paths) in model_level.items():
            if node.kind is kind and node.module_path in paths:
                matches[name].append(node.node_name)
                break
        else:
            if node.kind is ModuleKind.LINEAR and (
                node.module_path == patterns.lm_head
                or node.module_path.endswith("." + patterns.lm_head)
            ):
                matches["lm_head"].append(node.node_name)
            else:
                leftovers.append(node)

    problems = []
    for name, names in matches.items():
        if len(names) > 1:
            problems.append(f"expected at most one {name}, matched {names}.")
        setattr(result, name, names[0] if names else None)

    lookalikes = [
        node
        for node in leftovers
        if _DEDUP_SUFFIX.sub("", node.module_path.rsplit(".", 1)[-1])
        in _leaf_names(patterns, node.kind)
    ]
    if lookalikes:
        stack = under_decoder(f"{patterns.decoder_layers}.<N>")
        problems.append(
            f"these nodes are named like modules of this model_type but sit outside "
            f"the decoder stack '{stack}' (e.g. a module called directly rather "
            f"than through its parents, or a second model in the graph): "
            + _listed(lookalikes)
        )

    # At the root, everything is under the decoder prefix.
    foreign = [
        node
        for node in leftovers
        if prefix
        and node not in lookalikes
        and not node.module_path.startswith(prefix + ".")
    ]
    if foreign:
        problems.append(
            f"these nodes sit outside the language backbone '{prefix}' (e.g. a VLM "
            f"vision tower or projector). Export the language backbone on its own: "
            + _listed(foreign)
        )
    return problems


def _validate_layers(
    layers: Dict[int, Dict[object, List[str]]], patterns: HfModelPatterns
) -> List[str]:
    """Check that layer ids are contiguous and every field matched exactly one node.

    :return: Problems found.
    """
    problems = []
    ids = sorted(layers)
    if ids != list(range(ids[0], ids[-1] + 1)):
        problems.append(f"decoder layer indices are not contiguous: {ids}.")

    expected = {
        **{role: role.value for role in patterns.linears},
        **patterns.norms,
    }
    for layer_id in ids:
        for key, label in expected.items():
            names = layers[layer_id].get(key, [])
            if len(names) != 1:
                problems.append(
                    f"layer {layer_id}: expected exactly one {label}, matched {names}."
                )
    return problems


def _leaf_names(patterns: HfModelPatterns, kind: ModuleKind) -> set:
    """Last path component of every table entry of ``kind``."""
    paths = {
        ModuleKind.LINEAR: [*patterns.linears.values(), patterns.lm_head],
        ModuleKind.NORM: [*patterns.norms.values(), patterns.final_norm],
        ModuleKind.EMBEDDING: [patterns.embed_tokens],
    }[kind]
    return {path.rsplit(".", 1)[-1] for path in paths}


def _listed(nodes: List[ModuleNode]) -> str:
    """Render up to :data:`_MAX_LISTED` of ``nodes`` for an error message."""
    shown = ", ".join(
        f"'{node.module_path}' (node '{node.node_name}')"
        for node in nodes[:_MAX_LISTED]
    )
    more = len(nodes) - _MAX_LISTED
    return shown + (f" and {more} more." if more > 0 else ".")


__all__ = [
    "BlockMatch",
    "HfModelPatterns",
    "ModuleKind",
    "ModuleNode",
    "NamedLayerMatch",
    "NamedLayerMatchError",
    "get_hf_model_patterns",
    "match_module_nodes",
    "match_named_layers",
    "module_nodes_of",
    "module_path_of",
]
