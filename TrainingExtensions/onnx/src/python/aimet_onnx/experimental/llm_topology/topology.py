# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology.

Describes the structure of an ONNX decoder-stack model at two levels:

* *block level* — where each decoder block starts/ends on the residual stream; and
* *intra-block level* — the individual weighted projections
  (q/k/v/o, gate/up/down) and the two dynamic (non-weighted) attention MatMuls
  (Q·Kᵀ and softmax·V) inside each block, plus the model-level embed_tokens and
  lm_head.

:func:`analyze_llm_topology` finds every named layer by its HuggingFace module
path for a given ``model_type`` (see :mod:`hf_patterns`), derives the unnamed
parts (residual tensors, dynamic MatMuls) from the graph around them, and
cross-checks the names against the graph structure.

.. note::
   :func:`analyze_llm_topology_by_norm_count` (built on :func:`get_llm_topology`)
   is the legacy path, which infers blocks by counting active norms (from
   :func:`get_decoder_block_boundaries`) and splits roles by :mod:`layer_roles`.
   It is kept until its callers migrate and will be removed.

Technique-agnostic: it describes a decoder stack without knowing about any
specific quantization technique. Everything here works on the analysis IR (see
:mod:`ir_analysis`) and reports results by ONNX name (see
:mod:`topology_types`), so a topology needs no graph object to stay meaningful;
techniques that must mutate raw ``NodeProto`` edges (e.g. SpinQuant R3) derive
their insertion anchors from this topology separately.

A consumer that goes on to *rewrite* the graph re-attaches an
:class:`onnx_ir.Model` with :func:`~.ir_adapter.resolve_topology`, which hands
back the same structure carrying ``onnx_ir.Node`` / ``onnx_ir.Value`` handles.
"""

import re
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import static_tensor
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology import ir_analysis
from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries_in_ir,
    headless_block_end,
)
from aimet_onnx.experimental.llm_topology.hf_patterns import (
    BlockMatch,
    NamedLayerMatch,
    get_hf_model_patterns,
    match_named_layers,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    get_active_norm,
    find_active_norms_in_ir,
)
from aimet_onnx.experimental.llm_topology.topology_types import (
    BlockTopology,
    LinearGroup,
    LlmTopology,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


# KV-cache names vary across exporters. Matches patterns:
# ``past_{key,value}_<layer>[_in]`` and ``past_{k,v}_<layer>[_in]`` inputs
# ``past_{key,value}_<layer>[_out]`` and ``past_{k,v}_<layer>[_out]`` outputs
# ``past_key_values.<layer>.{key,value}`` (HF input style)
# ``present_{key,value}_<layer>`` or ``present.<layer>.{key,value}`` (HF outputs)
_PAST_KEY_INPUT_NAME_PATTERN = re.compile(
    r"^(?:past_(?:key|k)_\d+(?:_in)?|past_key_values\.\d+\.key)$"
)
_PAST_VALUE_INPUT_NAME_PATTERN = re.compile(
    r"^(?:past_(?:value|v)_\d+(?:_in)?|past_key_values\.\d+\.value)$"
)
_PAST_KEY_OUTPUT_NAME_PATTERN = re.compile(
    r"^(?:past_(?:key|k)_\d+(?:_out)?|present_key_\d+|present\.\d+\.key)$"
)
_PAST_VALUE_OUTPUT_NAME_PATTERN = re.compile(
    r"^(?:past_(?:value|v)_\d+(?:_out)?|present_value_\d+|present\.\d+\.value)$"
)


def get_llm_topology(
    ir_model: onnx_ir.Model,
    block_boundaries: List[Tuple[str, str]],
    active_norms: Optional[List[ActiveNorm]] = None,
    active_norms_per_block: int = 2,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    topo_index: Optional[Dict[onnx_ir.Node, int]] = None,
) -> LlmTopology:
    """Build the LLM topology from pre-computed block boundaries.

    Per-block. Read groups are the weighted linears downstream of each active
    norm; write groups are found by walking back from the residual output
    tensor to the weighted linear(s) that feed it; the read groups are then
    split into fine-grained roles by module name, and the dynamic attention
    MatMuls are located by graph topology:

    * ``qkv``             — ``downstream_linears`` of the first active norm in
      the block (input norm), as a :class:`LinearGroup` split into
      ``q_proj`` / ``k_proj`` / ``v_proj`` (or ``FUSED_QKV``) by
      :func:`classify_linear_role`.
    * ``o_proj``          — weighted linear(s) that write the attention residual
      output (the post-attention norm input).
    * ``gate_up``         — ``downstream_linears`` of the second active norm in
      the block (post-attention norm), as a :class:`LinearGroup` split into
      ``gate_proj`` / ``up_proj`` (or ``FUSED_GATE_UP``).
    * ``down_proj``       — weighted linear(s) that write the block residual
      output (the block-end tensor).
    * ``qk_matmul`` / ``attn_v_matmul`` — the two dynamic (non-weighted)
      attention MatMuls between the ``qkv`` group and ``o_proj`` (see
      :func:`_find_attention_matmuls`).

    Model-level roles:

    * ``lm_head``         — downstream linears of active norms at or after the
      last block boundary (outside all decoder blocks).
    * ``embed_tokens``    — Gather ops with a static-weight input that appear
      before the first block boundary.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param block_boundaries: List of ``(start_tensor, end_tensor)`` residual-stream
        tensor names, as returned by :func:`get_decoder_block_boundaries`.
    :param active_norms: Active norms in topological order. Recomputed via
        :func:`find_active_norms_in_ir` when not supplied; pass a precomputed
        value to avoid a redundant graph scan.
    :param active_norms_per_block: Expected number of active norms per decoder
        block. Must match the value used in :func:`get_decoder_block_boundaries`.
        Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
    :param role_patterns: Optional override of the default module-name → role
        table used to split the read groups (see :func:`classify_linear_role`).
    :param topo_index: Precomputed node → topological index map.
    :return: LlmTopology with block and backbone roles populated.
        ``hidden_size`` and ``head_dim`` are left ``None`` — use
        :func:`analyze_llm_topology` to also infer those.
    """
    if topo_index is None:
        topo_index = ir_analysis.topological_index(ir_model)
    if active_norms is None:
        active_norms = find_active_norms_in_ir(ir_model, topo_index)
    boundary_topo = ir_analysis.tensor_to_first_consumer_index(ir_model, topo_index)
    node_by_output = ir_analysis.node_by_output_tensor(ir_model)
    node_by_name = ir_analysis.node_by_name(ir_model)

    result = LlmTopology(active_norms=active_norms)

    for block_idx, (start_tensor, end_tensor) in enumerate(block_boundaries):
        start_topo = boundary_topo[start_tensor]
        end_topo = boundary_topo[end_tensor]

        # Active norms whose norm node falls in [start_topo, end_topo).
        # index 0 = input_norm (pre-attention),
        # index 1 = post_attn_norm (pre-MLP).
        block_active_norms = [
            active_norm
            for active_norm in active_norms
            if start_topo <= active_norm.topo_index < end_topo
        ]
        if len(block_active_norms) != active_norms_per_block:
            raise ValueError(
                f"Block {block_idx}: expected exactly {active_norms_per_block} active "
                f"norm(s) in topo range [{start_topo}, {end_topo}), "
                f"found {len(block_active_norms)}. "
                f"Ensure active_norms_per_block={active_norms_per_block} matches the "
                "value used in get_decoder_block_boundaries."
            )

        input_norm = block_active_norms[0]
        post_attn_norm = block_active_norms[1]

        qkv = LinearGroup.classify(input_norm.downstream_linears, role_patterns)
        gate_up = LinearGroup.classify(post_attn_norm.downstream_linears, role_patterns)

        intermediate_tensor = post_attn_norm.input_tensor
        o_proj_candidates = _find_nearest_upstream_linears(
            intermediate_tensor, start_tensor, node_by_output, topo_index
        )
        if not o_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no attention residual writer (o_proj) found "
                f"for residual output '{intermediate_tensor}'."
            )

        down_proj_candidates = _find_nearest_upstream_linears(
            end_tensor, intermediate_tensor, node_by_output, topo_index
        )
        if not down_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no MLP residual writer (down_proj) found "
                f"for residual output '{end_tensor}'."
            )

        qk_matmul, attn_v_matmul = _find_attention_matmuls(
            [node_by_name[name] for name in qkv.linears],
            o_proj_candidates,
            topo_index,
        )

        block = BlockTopology(
            qkv=qkv,
            o_proj=ir_analysis.node_names(o_proj_candidates),
            gate_up=gate_up,
            down_proj=ir_analysis.node_names(down_proj_candidates),
            qk_matmul=ir_analysis.node_names(qk_matmul),
            attn_v_matmul=ir_analysis.node_names(attn_v_matmul),
            residual_input=start_tensor,
            residual_output=end_tensor,
        )
        result.blocks.append(block)
        _logger.debug(
            "Block %d: q=%s k=%s v=%s (fused_qkv=%s) o_proj=%s  gate=%s up=%s "
            "(fused_gate_up=%s) down_proj=%s  qk_matmul=%s attn_v_matmul=%s",
            block_idx,
            qkv.role(LinearRole.Q_PROJ),
            qkv.role(LinearRole.K_PROJ),
            qkv.role(LinearRole.V_PROJ),
            qkv.role(LinearRole.FUSED_QKV),
            block.o_proj,
            gate_up.role(LinearRole.GATE_PROJ),
            gate_up.role(LinearRole.UP_PROJ),
            gate_up.role(LinearRole.FUSED_GATE_UP),
            block.down_proj,
            block.qk_matmul,
            block.attn_v_matmul,
        )

    block_role_counts = [
        (
            len(b.qkv.linears),
            len(b.o_proj),
            len(b.gate_up.linears),
            len(b.down_proj),
        )
        for b in result.blocks
    ]
    if len(set(block_role_counts)) > 1:
        _logger.warning(
            "Inconsistent role shapes across %d decoder blocks — downstream algorithms "
            "may not apply correctly. Per-block shapes "
            "(n_qkv, n_o_proj, n_gate_up, n_down_proj): %s",
            len(result.blocks),
            block_role_counts,
        )

    last_end_topo = boundary_topo[block_boundaries[-1][1]]
    result.lm_head = [
        linear
        for active_norm in active_norms
        if active_norm.topo_index >= last_end_topo
        for linear in active_norm.downstream_linears
    ]
    if not result.lm_head:
        _logger.debug(
            "lm_head not detected: no active norm found after the last block boundary."
        )
    else:
        _logger.debug("lm_head: %s", result.lm_head)

    first_start_topo = boundary_topo[block_boundaries[0][0]]
    result.embed_tokens = ir_analysis.node_names(
        [
            node
            for node in ir_model.graph
            if topo_index[node] < first_start_topo
            and ir_analysis.is_embedding_table_gather(node)
        ]
    )
    if not result.embed_tokens:
        _logger.info(
            "Backbone: embed_tokens not detected, no Gather op with a static weight found before "
            "the first block boundary. This is expected for VLM backbones exported with "
            "use_inputs_embeds=True. Rotate embedding.pth separately."
        )
    _logger.debug("embed_tokens: %s", result.embed_tokens)

    result.past_key_input_names = _collect_matching_names_in_order(
        ir_model.graph.inputs, _PAST_KEY_INPUT_NAME_PATTERN
    )
    result.past_value_input_names = _collect_matching_names_in_order(
        ir_model.graph.inputs, _PAST_VALUE_INPUT_NAME_PATTERN
    )
    result.past_key_output_names = _collect_matching_names_in_order(
        ir_model.graph.outputs, _PAST_KEY_OUTPUT_NAME_PATTERN
    )
    result.past_value_output_names = _collect_matching_names_in_order(
        ir_model.graph.outputs, _PAST_VALUE_OUTPUT_NAME_PATTERN
    )
    _logger.debug("past_key inputs: %s", result.past_key_input_names)
    _logger.debug("past_value inputs: %s", result.past_value_input_names)
    _logger.debug("past_key outputs: %s", result.past_key_output_names)
    _logger.debug("past_value outputs: %s", result.past_value_output_names)

    _logger.info(
        "Backbone: %d block(s), embed_tokens=%s, lm_head=%s.",
        len(result.blocks),
        result.embed_tokens,
        result.lm_head,
    )

    return result


def analyze_llm_topology(
    model: ModelProto,
    model_type: str,
    *,
    ir_model: Optional[onnx_ir.Model] = None,
) -> LlmTopology:
    """Analyze ``model`` end-to-end and return a name-based :class:`LlmTopology`.

    Every field that corresponds to a named HuggingFace module is found by name
    (see :mod:`~.hf_patterns`): the per-block projections and norms,
    ``embed_tokens``, the final norm and ``lm_head``. Blocks are the decoder
    layers those names belong to. The remaining fields are derived from the
    graph around the named layers:

    * ``residual_input`` — the tensor entering the block's input norm;
      ``residual_output`` — the next block's ``residual_input``, or for the last
      block the tensor entering the final norm (the final residual ``Add``'s
      output for a headless backbone).
    * ``qk_matmul`` / ``attn_v_matmul`` — the dynamic MatMuls between the named
      q/k/v and o_proj (best-effort: empty when attention is a fused op).
    * KV-cache names, ``hidden_size`` and ``head_dim`` — as before.

    The named layers are then cross-checked against the graph, so a name that
    points at the wrong node is an error rather than a wrong topology: each
    block's q/k/v (gate/up) must be exactly the weighted linears its input
    (post-attention) norm feeds, o_proj and down_proj must be the linears
    writing the residual stream, and the final norm must feed exactly lm_head.

    :param model: ONNX ModelProto to analyze. May be a float export or a
        ``QuantizationSimModel`` graph. Not mutated. A dynamo export must first
        go through
        :func:`~aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx.fix_node_names_pass`.
    :param model_type: HuggingFace ``PretrainedConfig.model_type`` of the model
        (for a VLM, of its text config), e.g. ``"llama"``.
    :param ir_model: Pre-built *analysis* IR for ``model`` — as returned by
        :func:`~.ir_analysis.build_analysis_ir`, i.e. quantizer-stripped and
        RMSNorm-fused. Built here when ``None``. Pass one only to avoid a second
        ``from_proto`` of a large model when the caller already holds it; a
        faithful (unfused) IR will not detect norms and must not be passed.
    :return: LlmTopology with every field populated. ``head_dim`` is ``None``
        when the export exposes no ``past_value`` graph input to derive it from.
    :raises ValueError: If ``model_type`` is unsupported, if the layers cannot
        be identified by name (:class:`~.hf_patterns.NamedLayerMatchError`), or
        if the named layers disagree with the graph.
    """
    patterns = get_hf_model_patterns(model_type)
    if ir_model is None:
        ir_model = ir_analysis.build_analysis_ir(model)
    topo_index = ir_analysis.topological_index(ir_model)
    match = match_named_layers(ir_model, patterns)
    topology = _build_named_topology(ir_model, match, topo_index)
    topology.hidden_size = _infer_hidden_size(ir_model, topology)

    # head_dim requires a KV-cache 'past_value' graph input; tolerate its
    # absence (prefill-only / R1-only flows do not need it).
    try:
        topology.head_dim = _infer_head_dim(model)
    except ValueError:
        topology.head_dim = None

    _logger.info(
        "Backbone (%s): %d block(s), embed_tokens=%s, lm_head=%s.",
        model_type,
        len(topology.blocks),
        topology.embed_tokens,
        topology.lm_head,
    )
    return topology


def analyze_llm_topology_by_norm_count(
    model: ModelProto,
    active_norms_per_block: int = 2,
    expected_num_blocks: Optional[int] = None,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    ir_model: Optional[onnx_ir.Model] = None,
) -> LlmTopology:
    """Analyze ``model`` the legacy way: decoder blocks by counting active norms.

    The behavior (and signature) :func:`analyze_llm_topology` had before it took a
    ``model_type``: detect every active RMSNorm, group them ``active_norms_per_block``
    at a time into decoder blocks, split each block's linears into roles with the
    generic :func:`~.layer_roles.classify_linear_role` table, and infer
    ``hidden_size`` / ``head_dim``. Needs no ``model_type``, so it also runs on
    architectures without a built-in name table.

    .. warning::
       Temporary. Kept for callers that have not migrated to
       :func:`analyze_llm_topology`, and removed together with
       :func:`get_llm_topology`.

    :param model: ONNX ModelProto to analyze. Not mutated.
    :param active_norms_per_block: Active norms per decoder block (see
        :func:`~.block_boundaries.get_decoder_block_boundaries`). Defaults to 2.
    :param expected_num_blocks: If given, validated against the detected count.
    :param role_patterns: Optional module-name → role override (see
        :func:`~.layer_roles.classify_linear_role`).
    :param ir_model: Pre-built analysis IR for ``model``, as for
        :func:`analyze_llm_topology`.
    :return: LlmTopology with block/backbone roles, ``active_norms``,
        ``hidden_size`` and ``head_dim`` populated. ``head_dim`` is ``None`` when
        the export exposes no ``past_value`` graph input to derive it from.
    """
    if ir_model is None:
        ir_model = ir_analysis.build_analysis_ir(model)
    topo_index = ir_analysis.topological_index(ir_model)

    active_norms = find_active_norms_in_ir(ir_model, topo_index)
    boundaries = get_decoder_block_boundaries_in_ir(
        ir_model,
        active_norms=active_norms,
        expected_num_blocks=expected_num_blocks,
        active_norms_per_block=active_norms_per_block,
        topo_index=topo_index,
    )
    topology = get_llm_topology(
        ir_model,
        boundaries,
        active_norms=active_norms,
        active_norms_per_block=active_norms_per_block,
        role_patterns=role_patterns,
        topo_index=topo_index,
    )

    topology.hidden_size = _infer_hidden_size(ir_model, topology)

    # head_dim requires a KV-cache 'past_value' graph input; tolerate its
    # absence (prefill-only / R1-only flows do not need it).
    try:
        topology.head_dim = _infer_head_dim(model)
    except ValueError:
        topology.head_dim = None

    return topology


def _build_named_topology(
    ir_model: onnx_ir.Model,
    match: NamedLayerMatch,
    topo_index: Dict[onnx_ir.Node, int],
) -> LlmTopology:
    """Build an :class:`LlmTopology` from validated named layers.

    :raises ValueError: Listing every place the named layers disagree with the
        graph.
    """
    node_by_name = ir_analysis.node_by_name(ir_model)
    node_by_output = ir_analysis.node_by_output_tensor(ir_model)
    problems: List[str] = []

    def active_norm(name: str) -> ActiveNorm:
        norm = get_active_norm(node_by_name[name], topo_index)
        # A gamma shared between norms (identical tensors de-duplicated by the
        # exporter) is rejected rather than followed: SpinQuant's norm fusion resets
        # each gamma to ones in place, so with one shared tensor the first reset
        # would silently change every other norm. Trained models never share gammas.
        if norm is None:
            raise ValueError(
                f"Norm '{name}' has no static gamma (scale) input of its own. "
                "Either the norm is not affine, or the exporter de-duplicated "
                "identical gamma tensors (e.g. behind an Identity op), as happens "
                "for an untrained model whose gammas are all 1.0."
            )
        return norm

    input_norms = [active_norm(block.input_norm) for block in match.blocks]
    post_attention_norms = [
        active_norm(block.post_attention_norm) for block in match.blocks
    ]
    final_norm = active_norm(match.final_norm) if match.final_norm else None

    # The last block ends where the final norm reads the residual stream, as every
    # other block ends at the next one's input norm. A headless backbone's final
    # norm feeds no linear and may be absent altogether, so there the last block
    # ends on the final residual Add instead: SpinQuant un-rotates the stream by
    # rewiring every consumer of that tensor, the backbone output included.
    residual_starts = [norm.input_tensor for norm in input_norms]
    if final_norm is not None and final_norm.downstream_linears:
        residual_starts.append(final_norm.input_tensor)
    else:
        residual_starts.append(
            headless_block_end(ir_model, residual_starts[-1], topo_index)
        )

    result = LlmTopology()
    for i, block in enumerate(match.blocks):
        input_norm, post_attention_norm = input_norms[i], post_attention_norms[i]
        label = f"Block {i} (layer {block.layer_id})"

        qkv = _named_group(block, _QKV_ROLES, topo_index, node_by_name)
        gate_up = _named_group(block, _GATE_UP_ROLES, topo_index, node_by_name)
        problems += _compare(
            f"{label}: q/k/v",
            qkv.linears,
            "input norm consumers",
            input_norm.downstream_linears,
        )
        problems += _compare(
            f"{label}: gate/up",
            gate_up.linears,
            "post-attention norm consumers",
            post_attention_norm.downstream_linears,
        )

        o_proj = block.linears[LinearRole.O_PROJ]
        down_proj = block.linears[LinearRole.DOWN_PROJ]
        attn_writers = _find_nearest_upstream_linears(
            post_attention_norm.input_tensor,
            input_norm.input_tensor,
            node_by_output,
            topo_index,
        )
        mlp_writers = _find_nearest_upstream_linears(
            residual_starts[i + 1],
            post_attention_norm.input_tensor,
            node_by_output,
            topo_index,
        )
        problems += _compare(
            f"{label}: o_proj",
            o_proj,
            "attention residual writers",
            ir_analysis.node_names(attn_writers),
        )
        problems += _compare(
            f"{label}: down_proj",
            down_proj,
            "MLP residual writers",
            ir_analysis.node_names(mlp_writers),
        )

        qk_matmul, attn_v_matmul = _find_attention_matmuls(
            [node_by_name[name] for name in qkv.linears],
            [node_by_name[name] for name in o_proj],
            topo_index,
        )
        if not qk_matmul or not attn_v_matmul:
            _logger.debug(
                "%s: dynamic attention MatMuls not found (qk=%s, attn_v=%s); "
                "attention may be a fused op.",
                label,
                qk_matmul,
                attn_v_matmul,
            )

        result.blocks.append(
            BlockTopology(
                qkv=qkv,
                o_proj=list(o_proj),
                gate_up=gate_up,
                down_proj=list(down_proj),
                qk_matmul=ir_analysis.node_names(qk_matmul),
                attn_v_matmul=ir_analysis.node_names(attn_v_matmul),
                residual_input=residual_starts[i],
                residual_output=residual_starts[i + 1],
            )
        )

    result.lm_head = [match.lm_head] if match.lm_head else []
    result.embed_tokens = [match.embed_tokens] if match.embed_tokens else []
    if final_norm is not None:
        problems += _compare(
            "lm_head",
            result.lm_head,
            "final norm consumers",
            final_norm.downstream_linears,
        )
    if problems:
        raise ValueError(
            "The layers identified by name disagree with the graph structure:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    # Same membership rule as find_active_norms_in_ir: a norm is active iff it
    # feeds a weighted linear, which leaves out a headless backbone's final norm.
    result.active_norms = sorted(
        (
            norm
            for norm in (*input_norms, *post_attention_norms, final_norm)
            if norm is not None and norm.downstream_linears
        ),
        key=lambda norm: norm.topo_index,
    )

    result.past_key_input_names = _collect_matching_names_in_order(
        ir_model.graph.inputs, _PAST_KEY_INPUT_NAME_PATTERN
    )
    result.past_value_input_names = _collect_matching_names_in_order(
        ir_model.graph.inputs, _PAST_VALUE_INPUT_NAME_PATTERN
    )
    result.past_key_output_names = _collect_matching_names_in_order(
        ir_model.graph.outputs, _PAST_KEY_OUTPUT_NAME_PATTERN
    )
    result.past_value_output_names = _collect_matching_names_in_order(
        ir_model.graph.outputs, _PAST_VALUE_OUTPUT_NAME_PATTERN
    )
    return result


#: Roles of the attention read group, and of the MLP read group.
_QKV_ROLES = (LinearRole.Q_PROJ, LinearRole.K_PROJ, LinearRole.V_PROJ)
_GATE_UP_ROLES = (LinearRole.GATE_PROJ, LinearRole.UP_PROJ)


def _named_group(
    block: BlockMatch,
    roles: Tuple[LinearRole, ...],
    topo_index: Dict[onnx_ir.Node, int],
    node_by_name: Dict[str, onnx_ir.Node],
) -> LinearGroup:
    """Build a read group from the named linears of ``roles``, in topological order."""
    by_role: Dict[LinearRole, List[str]] = {role: [] for role in LinearRole}
    for role in roles:
        by_role[role] = list(block.linears[role])
    linears = sorted(
        (name for role in roles for name in block.linears[role]),
        key=lambda name: topo_index[node_by_name[name]],
    )
    return LinearGroup(linears=linears, by_role=by_role)


def _compare(
    named_label: str,
    named: List[str],
    graph_label: str,
    from_graph: List[str],
) -> List[str]:
    """Return a problem if the ``named`` nodes are not the ``from_graph`` nodes."""
    if set(named) == set(from_graph):
        return []
    return [
        f"{named_label} by name {sorted(named)} != {graph_label} in the graph "
        f"{sorted(from_graph)}."
    ]


def _collect_matching_names_in_order(
    values: Iterable[onnx_ir.Value], pattern: Pattern
) -> List[str]:
    """Return names matching ``pattern`` in graph declaration order."""
    return [value.name for value in values if value.name and pattern.search(value.name)]


def _find_attention_matmuls(
    qkv_linears: List[onnx_ir.Node],
    o_proj: List[onnx_ir.Node],
    topo_index: Dict[onnx_ir.Node, int],
) -> Tuple[List[onnx_ir.Node], List[onnx_ir.Node]]:
    """Return ``(qk_matmul_nodes, attn_v_matmul_nodes)`` for a decoder block.

    Attention computes ``softmax(Q @ Kᵀ / scale) @ V``. Both MatMuls are
    *dynamic* — both inputs are activations, so neither has a static weight.
    Walking forward from the QKV projections toward O (not crossing ``o_proj``
    or any other weighted linear), we collect every dynamic MatMul and every
    Softmax. A dynamic MatMul that *feeds* a Softmax is Q·Kᵀ; one that
    *consumes* a Softmax output is softmax·V.

    Per-head split (SHA) exports emit one of each per head, so both lists may
    hold multiple nodes. Returns empty lists when the pattern is absent (e.g. an
    export that fuses attention into a single op with no explicit MatMuls) —
    dynamic-MatMul identification is best-effort and not required by every
    consumer.

    :param qkv_linears: The block's Q/K/V projection nodes (walk start).
    :param o_proj: The block's attention-output projection node(s) (walk fence).
    :param topo_index: Node → topological index map, used to order the results.
    :return: Two lists of dynamic MatMul nodes: Q·Kᵀ and softmax·V.
    """
    o_proj_nodes = set(o_proj)
    dynamic_matmuls: List[onnx_ir.Node] = []
    visited = set()
    queue = [successor for linear in qkv_linears for successor in linear.successors()]
    while queue:
        node = queue.pop()
        if node in visited or node in o_proj_nodes:
            continue
        visited.add(node)
        # Do not cross other weighted linears — the attention path holds only
        # dynamic MatMuls between the QKV projections and O.
        if ir_analysis.is_weighted_linear(node):
            continue
        if ir_analysis.is_dynamic_matmul(node):
            dynamic_matmuls.append(node)
        queue.extend(node.successors())

    qk_matmul = [m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=True)]
    attn_v_matmul = [
        m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=False)
    ]
    return (
        ir_analysis.sorted_by_topology(qk_matmul, topo_index),
        ir_analysis.sorted_by_topology(attn_v_matmul, topo_index),
    )


def _matmul_touches_softmax(matmul: onnx_ir.Node, forward: bool) -> bool:
    """Return True if a Softmax is reachable from ``matmul`` in the given direction.

    Walks ``forward`` (through consumers) or backward (through input producers)
    from ``matmul``, stopping at the next MatMul boundary. A Softmax reached
    before hitting another MatMul means ``matmul`` feeds (forward) or consumes
    (backward) that Softmax — i.e. it is Q·Kᵀ or softmax·V respectively.
    """
    visited = set()
    queue = list(matmul.successors() if forward else matmul.predecessors())
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op_type in ir_analysis.SOFTMAX_TYPES:
            return True
        # Stop at any other MatMul so a head's Q·Kᵀ is not linked to the next
        # head's Softmax through a shared downstream op.
        if node.op_type == "MatMul":
            continue
        queue.extend(node.successors() if forward else node.predecessors())
    return False


def _find_nearest_upstream_linears(
    target_tensor: str,
    boundary_tensor: str,
    node_by_output: Dict[str, onnx_ir.Node],
    topo_index: Dict[onnx_ir.Node, int],
) -> List[onnx_ir.Node]:
    """Nearest weighted linears feeding target_tensor, via a backward walk.

    Walks backward from the node producing target_tensor and collects the first
    weighted linear (MatMul/Gemm/Conv with a static weight) on each path,
    stopping there; any other op type is crossed transparently.

    NOTE: The walk is fenced at boundary_tensor's producer: that node and anything
    earlier are skipped, so the walk does not cross into the previous block.

    :param target_tensor: Tensor whose upstream linears are wanted (walk start).
    :param boundary_tensor: Upstream edge; its producer and earlier nodes are the
        lower fence.
    :param node_by_output: Map of output tensor name -> producing node.
    :param topo_index: Map of node -> topological index.
    :return: The nearest weighted linear on each backward path, in topological order.
    """
    start = node_by_output.get(target_tensor)
    if start is None:
        return []

    fence_node = node_by_output.get(boundary_tensor)
    lo = topo_index[fence_node] if fence_node is not None else -1

    linears = []
    seen = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        if topo_index.get(node, -1) <= lo:
            continue
        if ir_analysis.is_weighted_linear(node):
            linears.append(node)
            continue  # this linear shadows everything upstream of it
        queue.extend(node.predecessors())
    return ir_analysis.sorted_by_topology(linears, topo_index)


def _infer_hidden_size(ir_model: onnx_ir.Model, role_map: LlmTopology) -> int:
    """Infer the model hidden size from embed_tokens, lm_head, or q/k/v_proj weights.

    Tries ``embed_tokens`` first (Gather table ``[vocab, hidden]``, last dim = hidden).
    Falls back to ``lm_head``, then to each block's ``qkv`` group, for backbones
    exported with ``use_inputs_embeds=True`` that have no Gather op.

    Takes the analysis IR rather than a ``ModelProto`` so the weight layout is
    derived by the one implementation that already knows it,
    :func:`~.ir_analysis.get_weight_value` — the topology itself only carries node
    names. Shapes are read off the static tensor without materializing it; an
    lm_head table can be hundreds of megabytes.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param role_map: Topology produced by :func:`get_llm_topology`.
    :return: The hidden dimension size.
    """
    node_by_name = ir_analysis.node_by_name(ir_model)

    for embed_name in role_map.embed_tokens:
        # Only the data input (a [vocab, hidden] table) yields hidden_size; other
        # static inputs (e.g. axis attributes, indices) are not embedding tables.
        node = node_by_name.get(embed_name)
        table = static_tensor(node.inputs[0]) if node else None
        if table is not None and len(table.shape) >= 2:
            return int(table.shape[-1])

    # Gemm transB=1 stores W [vocab, hidden] -> hidden = shape[-1].
    # MatMul stores W [hidden, vocab]        -> hidden = shape[0].
    # Conv 1x1 stores W [vocab, hidden, 1, 1] -> hidden = shape[1].
    for linear_name in [
        *role_map.lm_head,
        *(name for block in role_map.blocks for name in block.qkv.linears),
    ]:
        node = node_by_name.get(linear_name)
        if node is None:
            continue
        weight, is_transposed = ir_analysis.get_weight_value(node)
        if weight is None:
            continue
        shape = static_tensor(weight).shape
        if node.op_type == "Conv":
            return int(shape[1])  # [out_ch, in_ch, *k]: in_ch = hidden
        return int(shape[-1] if is_transposed else shape[0])

    raise ValueError(
        "Cannot infer hidden_size: no embed_tokens, lm_head or qkv_proj static weight found in role_map"
    )


def _infer_head_dim(model: ModelProto) -> int:
    """Infer per-head dimension from a ``past_value`` graph input's last axis.

    HF/optimum LLM exports include ``past_value_*`` (or ``past_key_values.*.value``)
    inputs whose final dimension is ``head_dim`` regardless of the surrounding
    layout (``[B, num_kv_heads, past_seq, head_dim]`` or
    ``[B, past_seq, num_kv_heads, head_dim]``). This avoids having to derive
    ``head_dim`` from ``hidden_size / num_heads``, which is wrong for models
    that decouple the two (e.g. Gemma3 fixes ``head_dim=256`` independent of
    hidden size).

    :param model: ONNX ModelProto whose graph inputs are scanned.
    :return: ``head_dim`` read from the last dim of the first matching input.
    :raises ValueError: If no ``past_value`` input exists, or if its last dim
        is not a static positive integer.
    """
    for inp in model.graph.input:
        if not _PAST_VALUE_INPUT_NAME_PATTERN.search(inp.name):
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 0:
            continue
        last = dims[-1]
        # Must be a statically-known positive int. Symbolic dims (dim_param set,
        # or dim_value == 0) cannot be used to derive head_dim.
        if last.HasField("dim_value") and last.dim_value > 0:
            head_dim = last.dim_value
            _logger.info(
                "Derived head_dim=%d from graph input '%s' (last dim of shape %s).",
                head_dim,
                inp.name,
                [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims],
            )
            return head_dim

    raise ValueError(
        "Cannot infer head_dim: no graph input matching 'past_value' with a "
        "static positive last dimension was found."
    )


__all__ = [
    "analyze_llm_topology",
    "analyze_llm_topology_by_norm_count",
    "get_llm_topology",
]
