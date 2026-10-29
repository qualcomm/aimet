# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ExportableQSA adaptation for Qwen4-Exp models.

Replaces ``Qwen4ExpTextQSAIndexer.forward`` with a vectorized equivalent that
torch.jit.trace, dynamo and quantsim can capture. The stock forward loops over
every (batch, query) pair in Python, calling ``k_layernorm`` once per query and
sizing tensors from ``nonzero``, and builds its mask with a bool-valued
``scatter`` that torch.jit.trace cannot record.

The rewrite relies on the visible keys of every query being a prefix, in key
order, of the keys visible to any query. This holds for causal masks with
padding, with or without a KV cache. Under it a key's rank among the visible
keys is the same for every query, so the compressed key blocks are shared and
only the number of complete blocks differs per query. Blocks are pooled once,
scored for all queries in one matmul, and selected with a fixed-k topk in which
blocks a query cannot see are scored ``-inf``. Outside tracing the prefix
property is checked and a mask that breaks it raises.

Attention then reads only the keys some query selected. Each query selects at
most ``budget + ratio - 1`` keys (its top blocks plus a partial tail block), so
a call with ``S`` queries needs at most ``min(KV, S * (budget + ratio - 1))``
of them. Keys and values are gathered to that fixed width, and the mask with
them, before the attention matmuls. Attention sums over keys, so their order
does not matter, and gathered keys no query selected stay masked. At decode
(``S = 1``) this cuts attention to the token budget; for long prefills the
width reaches ``KV`` and the gather is skipped where shapes are static.
"""

from __future__ import annotations

import math
import types

import torch
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


def exportable_qsa_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    """Drop-in replacement for ``Qwen4ExpTextQSAIndexer.forward``."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch_size, seq_length, _ = hidden_states.shape
    ratio = self.compress_ratio
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = (
        full_cos[:, -seq_length:, :],
        full_sin[:, -seq_length:, :],
    )

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [
            self.index_n_heads * self.index_head_dim,
            self.index_kv_heads * self.index_head_dim,
        ],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    # [B, S, KV]
    visible = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )[:, 0]
    kv_length = visible.shape[-1]
    num_blocks = kv_length // ratio

    # Rank of each key among the keys visible to any query
    any_visible = visible.any(dim=1)
    rank = (torch.cumsum(any_visible.to(torch.int64), dim=-1) - 1).clamp(min=0)
    num_visible = visible.sum(dim=-1, keepdim=True)

    if not (torch.jit.is_tracing() or torch.compiler.is_compiling()):
        if not torch.equal(
            visible, any_visible.unsqueeze(1) & (rank.unsqueeze(1) < num_visible)
        ):
            raise ValueError(
                "ExportableQSA requires the keys visible to each query to be a prefix "
                "of the keys visible to any query, as with causal masks with padding."
            )

    # Key positions in rank order: visible keys first, then the rest
    kv_positions = torch.arange(kv_length, device=visible.device)
    order = torch.argsort(
        kv_positions + kv_length * (~any_visible).to(torch.int64), dim=-1
    )
    block_positions = order[:, : num_blocks * ratio]

    key_groups = torch.gather(
        raw_keys, 1, block_positions.unsqueeze(-1).expand(-1, -1, raw_keys.shape[-1])
    ).view(batch_size, num_blocks, ratio, self.index_head_dim)
    pooled_keys = key_groups.float().mean(dim=2).to(raw_keys.dtype)
    pooled_keys = self.k_layernorm(pooled_keys)

    group_starts = block_positions[:, ::ratio]
    full_cos = full_cos.expand(batch_size, -1, -1)
    full_sin = full_sin.expand(batch_size, -1, -1)
    rope_index = group_starts.unsqueeze(-1).expand(-1, -1, full_cos.shape[-1])
    block_keys = apply_rotary_pos_emb(
        pooled_keys.unsqueeze(2),
        cos=torch.gather(full_cos, 1, rope_index),
        sin=torch.gather(full_sin, 1, rope_index),
        unsqueeze_dim=2,
    ).squeeze(2)

    # [B, S, H, NB] -> [B, S, NB]
    scores = torch.matmul(q.float(), block_keys.float().transpose(-1, -2).unsqueeze(1))
    scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)

    # Each query selects among its complete blocks only
    num_complete = num_visible // ratio
    block_ids = torch.arange(num_blocks, device=visible.device)
    scores = scores.masked_fill(block_ids >= num_complete, float("-inf"))
    # sym_min keeps the extent symbolic under dynamo; jit.trace passes sizes as tensors
    if isinstance(num_blocks, torch.SymInt):
        topk = torch.sym_min(self.block_topk, num_blocks)
    else:
        topk = min(self.block_topk, num_blocks)
    top_blocks = scores.topk(topk, dim=-1).indices
    # One extra False slot for keys outside any complete block
    selected_blocks = torch.zeros(
        (batch_size, seq_length, num_blocks + 1),
        dtype=torch.bool,
        device=visible.device,
    ).scatter(-1, top_blocks, top_blocks < num_complete)

    # Tokens of the selected blocks, plus the tail that does not fill a block
    token_block = (
        (rank // ratio).clamp(max=num_blocks).unsqueeze(1).expand(-1, seq_length, -1)
    )
    in_selected_block = torch.gather(selected_blocks, -1, token_block)
    in_tail = rank.unsqueeze(1) >= num_complete * ratio
    selected_token_mask = (visible & (in_selected_block | in_tail)).unsqueeze(1)

    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(
            selected_token_mask, attention_mask.new_zeros(()), min_dtype
        )

    return selected_token_mask


def _gather_selected_keys(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    max_selected: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather keys, values and mask down to the keys any query can attend to.

    :param key_states: ``(B, H_kv, KV, D)``, before ``repeat_kv``
    :param value_states: ``(B, H_kv, KV, D_v)``
    :param attention_mask: ``(B, 1, S, KV)``, combined causal and indexer mask
    :param max_selected: Upper bound on the keys one query attends to
    :return: The three inputs with ``KV`` reduced to the gather width
    """
    batch_size, num_heads, kv_length, _ = key_states.shape
    seq_length = attention_mask.shape[2]
    if isinstance(kv_length, torch.SymInt) or isinstance(seq_length, torch.SymInt):
        width = torch.sym_min(kv_length, seq_length * max_selected)
    else:
        width = min(kv_length, seq_length * max_selected)
        if width == kv_length:
            return key_states, value_states, attention_mask

    attended = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )
    # Every key some query attends to ranks above every key none does
    any_attended = attended.any(dim=2)[:, 0].to(torch.float32)
    kv_index = any_attended.topk(width, dim=-1).indices  # (B, W)

    # index_select over flattened rows exports as Gather rather than GatherElements
    row_offsets = (
        torch.arange(batch_size * num_heads, device=kv_index.device) * kv_length
    ).view(batch_size, num_heads, 1)
    rows = (kv_index.unsqueeze(1) + row_offsets).reshape(-1)

    def gather_kv(states: torch.Tensor) -> torch.Tensor:
        head_dim = states.shape[-1]
        flat = states.reshape(batch_size * num_heads * kv_length, head_dim)
        return flat.index_select(0, rows).view(batch_size, num_heads, width, head_dim)

    mask_offsets = (torch.arange(batch_size, device=kv_index.device) * kv_length).view(
        batch_size, 1
    )
    mask_rows = (kv_index + mask_offsets).reshape(-1)
    flat_mask = (
        attention_mask[:, 0].transpose(1, 2).reshape(batch_size * kv_length, seq_length)
    )
    gathered_mask = (
        flat_mask.index_select(0, mask_rows)
        .view(batch_size, width, seq_length)
        .transpose(1, 2)
        .unsqueeze(1)
    )
    return gather_kv(key_states), gather_kv(value_states), gathered_mask


def exportable_qsa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``Qwen4ExpTextAttention.forward`` with keys gathered to the indexer's selection."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    selected_token_mask = self.indexer(
        hidden_states, position_embeddings, attention_mask, past_key_values
    )
    if attention_mask.is_floating_point():
        attention_mask = attention_mask + selected_token_mask
    else:
        attention_mask = attention_mask & selected_token_mask

    position_embeddings = (
        x[:, -hidden_states.shape[1] :, :] for x in position_embeddings
    )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    key_states, value_states, attention_mask = _gather_selected_keys(
        key_states,
        value_states,
        attention_mask[..., : key_states.shape[-2]],
        self.indexer.token_budget + self.indexer.compress_ratio - 1,
    )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _patch_qsa_instances(model: PreTrainedModel) -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    for module in model.modules():
        if isinstance(module, modeling_qwen4_exp.Qwen4ExpTextQSAIndexer):
            module.forward = types.MethodType(exportable_qsa_indexer_forward, module)
        elif isinstance(module, modeling_qwen4_exp.Qwen4ExpTextAttention):
            module.forward = types.MethodType(exportable_qsa_attention_forward, module)


class Qwen4ExpExportableQSAAdaptation:
    """Makes the Qwen4-Exp QSA indexer traceable and exportable."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)
        _patch_qsa_instances(model)
        return model


_QSA_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")


def register_adaptations() -> None:
    """Register ExportableQSA for every Qwen4-Exp model type.

    Also called by tests, which run under a fixture that wipes the registry.
    """
    for model_type in _QSA_MODEL_TYPES:
        YAMLConfigParser.register_adaptation(
            "ExportableQSA", model_type=model_type, required_for_export=True
        )(Qwen4ExpExportableQSAAdaptation)


register_adaptations()


# ---------------------------------------------------------------------------
# Hardening-time graph passes (specification -- not implemented here)
# ---------------------------------------------------------------------------
#
# One export serves prefill and decode: the query length S is symbolic. Fixing
# it ("hardening", e.g. S = 128 and S = 1) makes parts of the QSA graph
# provably dead or provably trivial, but only per specialization, so the
# removal has to happen after hardening rather than in the forward (branching
# on S would specialize the axis). Both specializations still come from one
# export and one set of encodings.
#
# The constraints of the ExportableLinearAttention passes apply here too: run
# after quantsim export (renaming or deleting nodes changes the tensor names
# encodings are keyed on), and keep the rewritten region unquantized -- the
# index arithmetic below (``Range``, ``CumSum``, ``Less``, ``Where``, ``TopK``
# indices, ``Gather`` index inputs) carries no activations and must not carry
# QDQ, or constant propagation stops at the first QDQ pair.
#
# Facts the passes rest on, each checked on the Generator's real masks:
#
#   (A) If the cache holds at most ``budget`` keys (CL <= budget), every
#       complete block is selected, so the indexer's mask equals the visible
#       mask: the indexer is the identity.
#   (B) The keys visible to any query are [one contiguous cached run] ++ [the
#       current chunk's real tokens], with the chunk's left padding between
#       them -- not one contiguous suffix of the KV axis.
#   (D) A query selects at most ``budget + ratio - 1`` keys.
#
# ===========================================================================
# Pass 0 -- Harden the symbolic dims                            PREREQUISITE
# ===========================================================================
#
# As for ExportableLinearAttention: evaluate every boundary ``dim_param`` with
# S bound (the gather width appears as ``Min(KV, S * (budget + ratio - 1))``),
# strip stale symbolic shapes, re-run shape inference.
#
# ===========================================================================
# Pass 1 -- Constant-fold                                       PREREQUISITE
# ===========================================================================
#
# ``ORT_ENABLE_BASIC`` folds the shape arithmetic: the block count, the TopK
# ``k`` (``Min(block_topk, KV // ratio)``) and the gather width all become
# literals.
#
# ===========================================================================
# Pass 2 -- Drop the full-width attention gather               PREFILL
# ===========================================================================
#
# ``_gather_selected_keys`` returns its inputs untouched when the width equals
# KV, but only when shapes are static at trace time; the dynamic export keeps
# the gather. Once S is fixed, a width that folded to KV (measured: S = 128 at
# CL = 4096, budget 2048, ratio 4) makes it a no-op that still costs a TopK
# over KV and three Gathers (K, V, mask).
#
# Rewrite: when the gathered width equals the KV extent, replace the gathered
# K, V and mask by their sources and delete the TopK, its Range/offset
# arithmetic and the three Gathers. Attention sums over keys, so the order
# the TopK would have imposed does not matter.
#
# How to find it: the TopK whose input is the reduced ``any_attended`` row
# (B, KV) and whose ``k`` folded to KV, feeding index arithmetic into exactly
# three Gathers. Do not match on names. Safe as a no-op at decode, where the
# width (2051 in the measurement above) is below KV.
#
# ===========================================================================
# Pass 3 -- Delete the indexer when CL <= budget                ANY S
# ===========================================================================
#
# By (A), the selected-token mask equals the visible mask, so
# ``attention_mask + selected_token_mask`` (float) adds zero to every visible
# position and only re-masks positions that are already masked. Replace the
# combined mask by ``attention_mask`` and delete the indexer subgraph: the
# indexer q/k projection, both layernorms, block pooling, block RoPE, the
# scoring matmul and both TopKs. The indexer keys are still written to the
# cache, so the cache layout does not change.
#
# Not bit-exact. Already-masked positions lose the extra ``finfo.min``: under
# the Generator's -100 mask their softmax weight goes from 0 back to ~1e-43,
# far below anything a quantized graph resolves. (Where the base mask is
# itself ``finfo.min``, the sum is ``-inf``, which is one more reason the pass
# should drop it rather than keep it.)
#
# Guard: CL // ratio <= budget // ratio, read off the folded block count and
# TopK ``k``. Configs with CL > budget keep the indexer.
#
# ===========================================================================
# Pass 4 -- Replace the rank argsort                           DO NOT
# ===========================================================================
#
# The rank order (visible keys first) is computed with an argsort over KV,
# which exports as a TopK of k = KV. It is tempting to replace it with a
# rotation (a Gather with a computed offset), but by (B) the visible keys are
# two runs separated by padding, not one suffix, so a rotation is wrong for
# 50 of the 134 Generator calls measured. A sound rewrite needs the two run
# boundaries: concat(range(cached run), range(current chunk's real tokens)).
# That is two Ranges and a Concat instead of a TopK over KV; worth it only if
# the TopK shows up on target.
