# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ExportablePLE adaptation for Qwen4-Exp's per-layer embeddings (PLE).

A PLE layer looks up hashed n-gram embeddings of the current and previous
``ngram_size - 1`` tokens, gates them against every residual stream, and adds
a dilated depthwise short conv over the result. The lookup cannot run on
target: it hashes token ids with int64 multiply / xor / modulo and reads tables
of tens of millions of rows. So the host computes the embeddings, in
:class:`Qwen4ExpGenerator`, from the unpadded token stream, and feeds each PLE
layer's as a graph input ``ple_embeds_{i}``. The graph keeps the projections,
norms, gate and conv, and carries the conv window as the ``ple_conv_state_{i}``
cache state.

Two padding effects of the stock forward are avoided by construction:

- The stock model turns padded positions into ``eos`` before hashing, and the
  hash restarts its n-grams at every ``eos``, so left-padding each chunk cuts
  the n-grams at chunk boundaries. Hashing the unpadded stream on the host
  matches the unpadded stock model.
- The short conv mixes adjacent columns, so the generator's left-padding would
  sit between the cached window and the real tokens. The forward rolls the
  real tokens next to the window, convolves, and rolls back, exactly as
  ExportableLinearAttention does for the GatedDeltaNet conv; here the conv is
  dilated, so the window is ``(kernel - 1) * ngram_size`` columns long.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.utils.layer_cache import PLE_CONV_STATE_IDX
from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
    _real_token_mask,
    _roll_indices,
)


def exportable_ple_layer_forward(
    self,
    hidden_states: torch.Tensor,
    ple_embeds: torch.Tensor,
    past_key_values,
    conv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Qwen4ExpTextPLELayer.forward`` over host-computed n-gram embeddings.

    :param ple_embeds: ``(B, S, ple_embed_dim)`` n-gram embeddings of the current
        tokens, right-aligned like ``hidden_states``. Arrives in the stock
        forward's ``input_ids`` slot (see :func:`_route_ple_embeds`).
    """
    if not ple_embeds.is_floating_point():
        raise ValueError(
            "ExportablePLE expects this layer's host-computed n-gram embeddings, "
            "not token ids; Qwen4ExpGenerator provides them as `ple_embeds_{i}`."
        )
    batch_size, seq_length, _ = hidden_states.shape
    real = _real_token_mask(conv_mask, batch_size, seq_length, hidden_states)

    key_normed = self.norm_key(self.key_proj(ple_embeds)).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    value = self.value_proj(ple_embeds)
    query_normed = self.norm_query(hidden_states).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
        self.hidden_size
    )
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = self.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    # Padded positions must not reach the conv window or the residual
    real_3d = real.unsqueeze(-1)
    gated_value = gated_value * real_3d
    gated_value_normed = gated_value_normed * real_3d

    window = self.short_conv_state_len
    cache_layer = past_key_values.layers[self.layer_idx]
    conv_state = cache_layer.conv_states[PLE_CONV_STATE_IDX]  # (B, C, window)
    if conv_state is None:  # a trace without cache inputs starts from an empty window
        conv_state = hidden_states.new_zeros(batch_size, gated_value.shape[-1], window)

    # Roll the real (right-aligned) tokens to the front so the cached window sits
    # immediately before them, convolve, then roll the outputs back
    n_real = real.sum(dim=-1).to(torch.int64)
    npad = seq_length - n_real
    x = gated_value_normed.transpose(1, 2)  # (B, C, S)
    fwd_idx = _roll_indices(npad, seq_length, x.device)
    back_idx = _roll_indices(n_real, seq_length, x.device)
    x = torch.gather(x, -1, fwd_idx.unsqueeze(1).expand_as(x))
    buf = torch.cat([conv_state.to(x.dtype), x], dim=-1)  # (B, C, window + S)

    # New window = the last `window` real columns, which start at n_real in buf
    cols = torch.arange(window, device=buf.device).unsqueeze(0) + n_real.unsqueeze(1)
    new_state = torch.gather(
        buf, -1, cols.unsqueeze(1).expand(batch_size, buf.shape[1], window)
    )
    cache_layer.conv_states[PLE_CONV_STATE_IDX] = new_state

    # buf already holds `window` columns of history, so a valid (unpadded) conv
    # yields exactly S outputs, one per column of x
    conv_out = F.silu(self.conv1d(buf))
    conv_out = torch.gather(conv_out, -1, back_idx.unsqueeze(1).expand_as(conv_out))
    return gated_value + conv_out.transpose(1, 2) * real_3d


def _route_ple_embeds(forward):
    """Wrap a stock ``Qwen4ExpTextDecoderLayer.forward`` to take ``ple_embeds``.

    ``ple_embeds`` maps a decoder layer index to its n-gram embeddings. The stock
    forward hands its PLE layer ``ple_input_ids`` positionally, so this layer's
    embeddings are passed in that slot, which the exportable PLE forward reads.
    """

    def routed(*args, ple_embeds=None, ple_input_ids=None, **kwargs):
        layer = forward.__self__
        if layer.ple is not None and ple_embeds is not None:
            ple_input_ids = ple_embeds[layer.ple.layer_idx]
        return forward(*args, ple_input_ids=ple_input_ids, **kwargs)

    return routed


def _patch_ple_instances(model: PreTrainedModel) -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    for module in model.modules():
        if isinstance(module, modeling_qwen4_exp.Qwen4ExpTextPLELayer):
            module.forward = types.MethodType(exportable_ple_layer_forward, module)
        elif isinstance(module, modeling_qwen4_exp.Qwen4ExpTextDecoderLayer):
            module.forward = _route_ple_embeds(module.forward)


class Qwen4ExpExportablePLEAdaptation:
    """Moves the PLE n-gram lookup to the host and makes the rest exportable."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)
        _patch_ple_instances(model)
        return model


_PLE_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")


def register_adaptations() -> None:
    """Register ExportablePLE for every Qwen4-Exp model type.

    Also called by tests, which run under a fixture that wipes the registry.
    """
    for model_type in _PLE_MODEL_TYPES:
        YAMLConfigParser.register_adaptation(
            "ExportablePLE", model_type=model_type, required_for_export=True
        )(Qwen4ExpExportablePLEAdaptation)


register_adaptations()
