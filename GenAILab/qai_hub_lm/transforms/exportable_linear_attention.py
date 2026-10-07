# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ExportableLinearAttention adaptation for Qwen 3.5 models.

Monkey-patches the gated delta rule functions on each GatedDeltaNet layer
instance to use a single unified code path suitable for all export backends
(torch.export, torch.jit.trace, eager).

The unified function replaces both:
  - chunk_gated_delta_rule: prefill path (any seq_len, chunked internally)
  - recurrent_gated_delta_rule: generation path (seq_len == 1)

Within each chunk the math follows QNN-cores' formulation: cumsum via
lower-triangular matmul, and a product-form Newton refinement for the
triangular solve (~11 MatMul ONNX ops per chunk).  The intra-chunk math is
batched across the chunk axis, so it is already sequence-length independent.

Chunk-size handling
-------------------
``chunk_size`` is a *cap*, not a fixed width: the chunk extent used is
``min(seq_len, chunk_size)``, derived inside the graph via ``torch.sym_min`` so
it stays symbolic.  One exported graph therefore hardens into a prefill graph
(chunk 64) and a decode graph (chunk 1) by fixing the sequence length alone --
the same knob already hardened before compilation -- with one set of quantsim
encodings covering both.

This matters because the intra-chunk triangular solve is pure waste at decode.
At ``seq_len == 1`` the padded rows are structurally zero and
``strict_lower_tri`` zeros the diagonal, so ``attn`` is identically zero and
``(I - A)^-1`` is *exactly* the identity -- at any chunk size.  A fixed
``chunk_size=64`` decode graph spends 11 matmuls over 64x64 matrices per head
per layer computing that constant.  Deriving the extent from the sequence length
collapses them to 1x1: measured on Qwen3.5-0.8B (3 linear-attention layers,
16 heads, head dim 128), the Scan body drops from 201,326,592 MACs per iteration
to 2,371,584 -- 85x less work per decoded token.

Do NOT write this as ``chunk = 1 if seq_len == 1 else 64``.  Branching on a
symbolic shape makes the exporter resolve the guard by *specializing* the
sequence axis, exactly as documented for ``pad_size`` below.  ``sym_min`` emits a
symbolic ``Min`` instead, which survives into the ONNX graph and folds away once
the sequence length is fixed.

The cap must stay <= 64: ``_solve_triangular``'s banded seed plus four
product-form squarings spans ``A^0..A^15`` against a nilpotent ``A`` with
``A^chunk == 0``, which is exact for chunk <= 64 and silently inexact beyond it.

Sequence-length handling
------------------------
The inter-chunk recurrence is a ``scan`` (``torch._higher_order_ops.scan``),
which the dynamo ONNX exporter lowers to a single ONNX ``Scan`` node.  The
chunk count is therefore the scan's trip count -- a runtime property of the
scanned input's leading dimension -- not a compile-time constant, so one
graph serves every sequence length and the graph no longer grows with it.

Any sequence length works, including ones that are not a multiple of
``chunk_size``: inputs are padded up to a multiple and the result sliced back
down.  Both operations stay in the graph under a dynamic sequence dim -- do
not make them conditional on ``pad_size``, since branching on a symbolic
value makes the exporter specialize the axis to exact multiples and quietly
reject every other length (``seq_len=1`` decode included).

Variable *real* lengths within a batch are still carried by
``attention_mask`` (a data input), which zeros the key/value/decay
contributions of padded positions.  Without this, left-padded inputs would
let padded-token garbage pollute the recurrent state; with it, masked
positions are inert and results match the unmasked variable-length reference
to machine precision.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

# Private path: ``scan`` has no public alias as of torch 2.11.  Switch to the
# public name once one exists.
from torch._higher_order_ops.scan import scan
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.utils.compat import linear_attention_states


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


# ---------------------------------------------------------------------------
# Triangular solve: (I - A)^{-1} via product-form Newton refinement
# ---------------------------------------------------------------------------


def _solve_triangular(attn, chunk_size, order=4):
    """Approximate (I - A)^{-1} via banded Taylor + product-form refinement.

    Uses an order-``order`` banded initial approximation M0, then refines with
    M0 @ (I+E0)(I+E0^2)(I+E0^4)(I+E0^8) where E0 = I - (I-A)@M0.
    Produces ~11 MatMul ONNX ops and survives INT16 quantization.

    ``attn`` is strictly lower triangular (the diagonal is zero), so ``I - A``
    is unit lower triangular and invertible; the banded Taylor seed plus four
    product-form refinement steps converge for the chunk sizes used here.
    """
    I = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    mask_acc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=attn.dtype, device=attn.device),
        diagonal=-order,
    )

    M_mat = I - attn
    acc = I + attn
    power = attn
    for _ in range(2, order + 1):
        power = torch.matmul(power, attn)
        acc = acc + power
    M0 = mask_acc * acc

    E0 = I - torch.matmul(M_mat, M0)
    E1 = torch.matmul(E0, E0)
    E2 = torch.matmul(E1, E1)
    E3 = torch.matmul(E2, E2)
    return torch.matmul(
        torch.matmul(
            torch.matmul(
                torch.matmul(M0, I + E0),
                I + E1,
            ),
            I + E2,
        ),
        I + E3,
    )


# ---------------------------------------------------------------------------
# Unified exportable gated delta rule
# ---------------------------------------------------------------------------


def exportable_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    attention_mask=None,
    **kwargs,
):
    """Unified gated delta rule for both prefill and decode.

    For prefill (seq_len > 1): processes each chunk with QNN-cores-style
    matrix ops, the inter-chunk recurrence carried by a ``scan``.
    For decode (seq_len == 1): the chunk extent collapses to 1, so the matrix
    ops degenerate to vector ops and the scan runs one iteration.

    Sequence length may be dynamic, and need not be a multiple of
    ``chunk_size`` -- see the module docstring.

    :param chunk_size: Upper bound on the chunk extent, not a fixed width. The
        extent actually used is ``min(seq_len, chunk_size)``, kept symbolic so a
        single graph serves prefill and decode. Must be <= 64; see the module
        docstring on the solve's convergence bound.

    :param attention_mask: Optional ``[batch, seq_len]`` mask (1 = real token,
        0 = padding).  Padded positions have their key/value/decay zeroed so
        they contribute nothing to the recurrent state or intra-chunk
        attention.  Required for correctness when inputs are left-padded
        within a batch of mixed real lengths.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Derive the chunk extent from the sequence length: ``chunk_size`` is a cap,
    # not a fixed width. ``sym_min`` keeps this symbolic -- see the module
    # docstring for why a ``if seq_len == 1`` branch cannot work here.
    chunk_size = torch.sym_min(seq_len, chunk_size)

    # Zero out padded positions before chunking. Masking key/value/g makes a
    # padded token's contribution to the recurrent state and to intra-chunk
    # attention exactly zero, so a fixed-length (padded) graph matches the
    # variable-length reference regardless of left/right padding.
    if attention_mask is not None:
        mask = attention_mask[:, None, :].to(torch.float32)  # [B, 1, S]
        key = key * mask.unsqueeze(-1)
        value = value * mask.unsqueeze(-1)
        g = g * mask

    # Pad to a multiple of chunk_size. Kept unconditional on purpose: under a
    # dynamic sequence dim ``pad_size`` is symbolic, and branching on it (``if
    # pad_size:``) forces the exporter to *specialize* -- it resolves the branch
    # by constraining the sequence length to exact multiples of chunk_size, which
    # silently makes the graph reject every other length (decode at seq_len=1
    # included). An unconditional pad keeps the axis genuinely free at the cost
    # of a few Pad nodes.
    # With a derived chunk extent both operands are symbolic (``Mod(S, Min(64,
    # S))``), which exports fine and keeps every length working: for
    # ``seq_len <= chunk_size`` the extent equals the length so the pad is zero,
    # and above it the pad rounds up to the cap as before. No divisibility
    # requirement is imposed on the sequence length.
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    query = query * (k_head_dim**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape into chunks: [B, H, num_chunks, chunk_size, ...]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Precompute masks (shared across all chunks)
    tril_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=query.device)
    )
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)
    strict_lower_tri = tril_mask - eye

    # Per-chunk cumsum via lower-triangular matmul
    g_cum = (tril_mask @ g.unsqueeze(-1)).squeeze(-1)
    decay_mask = (
        (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tril_mask
    ).exp() * tril_mask

    # Intra-chunk triangular solve: (I - A)^{-1}
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower_tri
    attn = _solve_triangular(attn, chunk_size)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    # State initialization
    state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            device=value.device,
            dtype=value.dtype,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    # Inter-chunk recurrence, as a native scan. ``scan`` iterates dim 0, so the
    # chunk axis moves to the front; the recurrent state is the carry and the
    # per-chunk outputs are stacked. Lowers to one ONNX ``Scan`` whose body is
    # ``_chunk_step``, keeping the graph independent of sequence length.
    xs = [x.movedim(2, 0) for x in (query, key, value, k_cumdecay, decay_mask, g_cum)]

    def _chunk_step(state, xs_i):
        q_i, k_i, v_i, kc_i, dm_i, g_i = xs_i

        attn_i = (q_i @ k_i.transpose(-1, -2) * dm_i) * tril_mask
        v_new = v_i - kc_i @ state
        o_i = (q_i * g_i.unsqueeze(-1).exp()) @ state + attn_i @ v_new

        next_state = (
            state * g_i[:, :, -1, None, None].exp()
            + (k_i * (g_i[:, :, -1, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2)
            @ v_new
        )
        return next_state, o_i

    state, core_attn_out = scan(_chunk_step, state, xs)
    core_attn_out = core_attn_out.movedim(
        0, 2
    )  # [n_chunks, B, H, C, Dv] -> [B, H, ...]

    if not output_final_state:
        state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, state


# ---------------------------------------------------------------------------
# Exportable GatedDeltaNet.forward
# ---------------------------------------------------------------------------


def _roll_indices(npad, seq_len, device):
    """Gather indices that roll a [..., seq_len] tensor left by ``npad`` columns.

    ``npad`` is a per-batch int64 tensor of shape ``(B,)``. Implemented as an
    index arithmetic expression so it exports to a single ONNX
    ``GatherElements`` (no Loop / data-dependent control flow), unlike
    ``torch.roll`` with a tensor shift.

    The wrap-around is a conditional subtraction rather than ``% seq_len``: with
    a dynamic sequence dim the divisor is symbolic, and ONNX's
    ``aten_remainder_scalar`` translation calls ``int()`` on it, failing with
    "int() argument must be ... not 'SymbolicTensor'".  Since ``base`` is at most
    ``seq_len - 1`` and ``npad`` at most ``seq_len``, the sum is always below
    ``2 * seq_len``, so subtracting ``seq_len`` once where it overflows is
    exactly equivalent to the modulo.
    """
    import torch

    base = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    idx = base + npad.unsqueeze(1)  # (B, S), < 2 * seq_len
    return torch.where(idx < seq_len, idx, idx - seq_len)


def exportable_gated_delta_net_forward(
    self, hidden_states, cache_params=None, attention_mask=None, **kwargs
):
    """Single-graph GatedDeltaNet forward for export.

    The stock forward gates its conv / recurrent-state handling on
    ``use_precomputed_states = cache_params.has_previous_state(layer_idx)``.
    At trace time the cache is freshly built, so that is False and the graph
    bakes in the from-scratch prefill path: the incoming ``conv_state`` and
    ``recurrent_state`` graph inputs become dead code and are ignored at
    inference, so decode has no memory and collapses after the first token.

    This replacement always threads the cached states (conv + recurrent) as
    live inputs, with no data-dependent branch, so a single static graph serves
    both prefill and decode:
      - delta rule: pass ``initial_state=recurrent_state`` unconditionally, and
        let ``attention_mask`` zero the padded positions (the kernel handles
        left-padding exactly).
      - conv: the generator left-pads inputs to a fixed width with the real
        tokens right-aligned, but the depthwise causal conv mixes *adjacent*
        columns, so the cached ``conv_state`` history must sit immediately
        before the first real token. We roll the real tokens to the front
        (via a gather), prepend ``conv_state`` there, convolve, capture the new
        conv window, then roll the outputs back. Padding is never between the
        history and the real tokens, so the conv sees the correct context.
    """
    import torch
    import torch.nn.functional as F

    # No ``apply_mask_to_padding_states`` here: it expects a 2D mask, and the
    # padding is handled below via ``mask2d`` (conv roll + kernel masking).
    # Before transformers 5.15 it was a no-op on our 4D mask (it gated on
    # ``mask.shape[1] > 1``); from 5.15 it multiplies unconditionally and the
    # (B, S, H) * (B, 1, 1, S, KV) broadcast fails.
    batch_size, seq_len, _ = hidden_states.shape

    conv_state, recurrent_state = linear_attention_states(
        cache_params.layers[self.layer_idx]
    )

    # Recover a 2D real-token mask [B, S] (1 = real) from whatever form the
    # model hands down. The generator feeds a 4D additive causal mask
    # (B, 1, S, KV); query row i is a real token iff its own diagonal key
    # (kv index q_offset + i, q_offset = KV - S) is unmasked (== 0). A plain 2D
    # mask is used directly; ``None`` means all-real.
    if attention_mask is None:
        mask2d = torch.ones(
            batch_size, seq_len, dtype=hidden_states.dtype, device=hidden_states.device
        )
    elif attention_mask.dim() == 2:
        mask2d = attention_mask.to(hidden_states.dtype)
    else:
        mask4d = attention_mask
        kv_len = mask4d.shape[-1]
        q_offset = kv_len - seq_len
        diag_idx = torch.arange(seq_len, device=mask4d.device) + q_offset  # (S,)
        diag = (
            mask4d[:, 0]
            .gather(-1, diag_idx.view(1, seq_len, 1).expand(batch_size, seq_len, 1))
            .squeeze(-1)
        )  # (B, S)
        mask2d = (diag >= 0).to(hidden_states.dtype)
    n_real = mask2d.sum(dim=-1).to(torch.int64)  # (B,)
    npad = seq_len - n_real

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Roll the real tokens (right-aligned) to the front so the cached conv_state
    # sits immediately before them, convolve over [conv_state | real | pad],
    # then roll the per-position outputs back to their original alignment.
    K = self.conv_kernel_size
    fwd_idx = _roll_indices(npad, seq_len, mixed_qkv.device)  # roll left by npad
    back_idx = _roll_indices(
        seq_len - npad, seq_len, mixed_qkv.device
    )  # roll right by npad
    mixed_rolled = torch.gather(
        mixed_qkv, -1, fwd_idx.unsqueeze(1).expand_as(mixed_qkv)
    )

    buf = torch.cat([conv_state, mixed_rolled], dim=-1)  # (B, conv_dim, K + S)
    # New conv window = the last K real columns. In ``buf`` the real region is
    # cols [K : K + n_real], so the trailing K of it start at ``n_real``. Gather
    # those K columns per batch (tensor index → exportable, no dynamic slice).
    cs_cols = torch.arange(K, device=buf.device).unsqueeze(0) + n_real.unsqueeze(
        1
    )  # (B, K)
    new_conv_state = torch.gather(
        buf, -1, cs_cols.unsqueeze(1).expand(buf.shape[0], buf.shape[1], K)
    )
    cache_params.update_conv_state(new_conv_state, self.layer_idx)

    # conv1d has built-in left padding (K-1); slice to the input width then drop
    # the K leading conv_state columns, leaving S outputs aligned to mixed_rolled.
    conv_out = F.silu(self.conv1d(buf)[:, :, : buf.shape[-1]])[:, :, K:]
    mixed_qkv = torch.gather(conv_out, -1, back_idx.unsqueeze(1).expand_as(conv_out))

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        attention_mask=mask2d,
    )
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Adaptation registration
# ---------------------------------------------------------------------------


def _passthrough_linear_attn_mask(self, attention_mask, past_key_values):
    """Always hand the linear-attention layers the real mask.

    The stock ``_update_linear_attn_mask`` nulls the mask when it is all-ones or
    when the cache has prior state. During export the sample input is all-real,
    so the mask is nulled and the exportable forward bakes in npad=0 — breaking
    decode, where padded positions must be located from the mask. Passing the
    mask through unconditionally keeps that information live in the graph.

    This is the transformers <=5.12.1 hook (the model exposed an instance
    method). In 5.13.0 the equivalent logic moved to a module-level function;
    see ``_passthrough_recurrent_attn_mask``.
    """
    return attention_mask


def _passthrough_recurrent_attn_mask(
    config=None,
    inputs_embeds=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Always hand the linear-attention layers the real mask (transformers >=5.13.0).

    Patches ``qwen3_5_modeling.create_recurrent_attention_mask``, which nulls the
    mask when it is not a 2D padding mask (``ndim != 2``), on cached forwards, or
    when all-ones. The generator feeds a 4D mask carrying the (left-)padding, so
    the stock function drops it and the exportable forward bakes in ``npad=0``,
    corrupting decode. Passing it through keeps the padding info live.

    The signature mirrors the stock call (invoked by keyword with ``config``,
    ``inputs_embeds``, ``attention_mask``, ``past_key_values``, ``position_ids``);
    ``**kwargs`` absorbs the args we don't use.
    """
    return attention_mask


#: Largest chunk extent ``_solve_triangular`` inverts exactly. Its banded seed
#: plus four product-form squarings spans ``A^0..A^15`` against a nilpotent ``A``
#: with ``A^chunk == 0``; beyond this the solve degrades silently, so the cap is
#: enforced rather than documented.
MAX_CHUNK_SIZE = 64


def _patch_gated_delta_net_instances(
    model: PreTrainedModel, chunk_size: int = 64
) -> None:
    """Walk all modules and replace the gated delta rule + forward on GatedDeltaNet instances."""
    import types

    if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be in [1, {MAX_CHUNK_SIZE}], got {chunk_size}. It "
            "caps the chunk extent; above the cap the triangular solve stops "
            "being exact (see the module docstring)."
        )

    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    # Qwen3.5-MoE's GatedDeltaNet is Qwen3.5's apart from the class name, so the
    # same exportable forward applies to both.
    gdn_classes = (
        modeling_qwen3_5.Qwen3_5GatedDeltaNet,
        modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet,
    )

    for module in model.modules():
        if isinstance(module, gdn_classes):
            # One cap serves both entry points: the extent is derived as
            # ``min(seq_len, chunk_size)``, so the recurrent path no longer
            # needs its own hardcoded chunk_size=1 -- a seq_len=1 call reaches
            # the same extent through the same graph.
            rule = functools.partial(exportable_gated_delta_rule, chunk_size=chunk_size)
            module.chunk_gated_delta_rule = rule
            module.recurrent_gated_delta_rule = rule
            module.forward = types.MethodType(
                exportable_gated_delta_net_forward, module
            )

    # Keep the linear-attention mask live (the stock model nulls it). The hook
    # differs by transformers version (transition landed in 5.13.0):
    #  - <=5.12.1: an instance method ``_update_linear_attn_mask`` on the model.
    #  - >=5.13.0: a module-level ``create_recurrent_attention_mask`` imported
    #    into the modeling namespace and called from ``Qwen3_5Model.forward``.
    patched_any = False
    for module in model.modules():
        if hasattr(module, "_update_linear_attn_mask"):
            module._update_linear_attn_mask = types.MethodType(
                _passthrough_linear_attn_mask, module
            )
            patched_any = True

    # Patch the mask factory in both modeling namespaces: Qwen3.5-MoE has its own
    # copy of ``create_recurrent_attention_mask``, and patching only Qwen3.5's
    # would leave the MoE model's mask nulled (silently corrupting decode).
    for namespace in (modeling_qwen3_5, modeling_qwen3_5_moe):
        if hasattr(namespace, "create_recurrent_attention_mask"):
            namespace.create_recurrent_attention_mask = _passthrough_recurrent_attn_mask
            patched_any = True

    if not patched_any:
        raise RuntimeError(
            "ExportableLinearAttention could not find a linear-attention mask hook "
            "to patch (neither `_update_linear_attn_mask` nor "
            "`create_recurrent_attention_mask`). The transformers Qwen3.5 masking "
            "API has likely changed; decode will silently corrupt without this patch."
        )


class Qwen3_5ExportableLinearAttentionAdaptation:
    """ExportableLinearAttention adaptation for Qwen 3.5 models.

    Replaces both gated delta rule functions with a single unified
    implementation that works under all export backends without graph
    explosion.

    Configurable via YAML adaptation kwargs:
        chunk_size (int): Upper bound on the chunk extent for the triangular
            solve, not a fixed width -- the extent used is
            ``min(seq_len, chunk_size)``, so one exported graph hardens into
            prefill and decode graphs by fixing the sequence length alone.
            Defaults to 64, which is also the maximum (``MAX_CHUNK_SIZE``).
    """

    chunk_size = 64

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)
        _patch_gated_delta_net_instances(model, chunk_size=cls.chunk_size)
        return model


# ---------------------------------------------------------------------------
# Hardening-time graph passes (specification -- not implemented here)
# ---------------------------------------------------------------------------
#
# "Hardening" is the step that turns the one exported graph into a compilable
# artifact by fixing its symbolic sequence length: prefill at e.g. 128, decode
# at 1. Because the chunk extent is derived as ``min(seq_len, chunk_size)``,
# fixing the sequence length also fixes the chunk extent, and a great deal of
# the graph becomes provably dead in the decode specialization specifically.
#
# None of this can be done in the kernel. Removing the dead work requires
# knowing the sequence length, and branching on it inside the graph would
# specialize the axis and destroy the single-graph property (see the module
# docstring). So the removal has to happen *after* hardening, per
# specialization. That does not weaken the single-graph paradigm: both
# specializations still come from one export and one set of quantsim encodings.
#
# Two global constraints apply to every pass below.
#
#   * Run them AFTER quantsim export. They delete and rewrite nodes; running
#     them earlier changes the tensor names the encodings are keyed on. Deleted
#     tensors leave orphaned entries in the encodings file, which is harmless
#     for name-keyed lookup -- prune them only if some tool validates that every
#     entry maps to a live tensor.
#   * The rewritten region must be UNQUANTIZED. A QDQ pair or QcQuantizeOp
#     sitting on a chain stops constant propagation dead, so the
#     ``op_outputs_to_ignore`` extension for the mask-construction ops
#     (``Range``, ``Equal``, ``GreaterOrEqual``, ``Min``, ``Not``, ``And``,
#     ``Cast``) is a precondition for passes 1-3, not a cosmetic cleanup.
#
# Node counts quoted below were measured on Qwen/Qwen3.5-0.8B truncated to 4
# layers (3 ``linear_attention`` + 1 ``full_attention``), CL=4096, vocab
# shrunk to 4096 so a hardened copy fits under protobuf's 2GB ceiling. They
# are structural measurements: op counts and shapes, not numerics.
#
# ===========================================================================
# Pass 0 -- Harden the symbolic dims                            PREREQUISITE
# ===========================================================================
#
# Everything else depends on this producing genuinely static shapes.
#
# A name-matching substitution is NOT sufficient. ``make_dim_param_fixed(graph,
# "s50", 128)`` rewrites only dims whose ``dim_param`` is literally ``"s50"``.
# The chunk extent appears as the *derived* string ``Min(64, s50)``, and the
# KV-cache length as ``4096 - s50`` -- which these exports already carry on the
# graph boundary today, so whatever hardens the sequence length must already
# cope with derived expressions. After a name-only substitution the Scan body
# still reads ``Min(64, s50)`` and nothing downstream is static.
#
# What works, in order:
#
#   1. For every boundary ``dim_param``, EVALUATE the expression with the
#      sequence symbol bound (handles ``4096 - s50`` and ``Min(64, s50)``
#      alike) and write the result as a ``dim_value``.
#   2. Strip stale symbolic shape metadata RECURSIVELY, including each
#      subgraph's own ``input``/``output`` declarations -- not just
#      ``value_info``. A Scan body carries the chunk extent on its own
#      boundary, and no amount of outer-graph substitution reaches it.
#      Clearing ``value_info`` alone leaves ``Min(64, s50)`` in place.
#   3. Re-run shape inference, then constant-fold (pass 1).
#
# Note that ONNX ``shape_inference`` alone does not re-derive Scan-body dims
# (it leaves ``unk__NN``); ORT's optimiser does, and yields fully static
# shapes. Verify by asserting that no ``dim_param`` survives anywhere in the
# graph, subgraphs included.
#
# ===========================================================================
# Pass 1 -- Constant-fold                                       PREREQUISITE
# ===========================================================================
#
# Ordinary constant folding, which ``ORT_ENABLE_BASIC`` already performs. It
# collapses the shape-derived mask construction: ``Min``, ``Range``, ``Trilu``
# and ``Equal`` all disappear, and ``tril_mask``, ``eye`` and
# ``strict_lower_tri`` become literal initializers. Measured: 625 -> 582 nodes
# (prefill), 625 -> 571 (decode).
#
# Do NOT use ``ORT_ENABLE_EXTENDED`` or ``ORT_ENABLE_ALL`` to produce a
# deployment artifact. They rewrite 6 Scan-body MatMuls into ``FusedMatMul``
# in the ``com.microsoft`` domain -- ORT-specific ops a non-ORT backend cannot
# consume. They remove no arithmetic (the MAC count is unchanged; the ops are
# absorbed Transposes), so the only thing they buy here is a trap.
#
# ===========================================================================
# Pass 2 -- Collapse the dead intra-chunk solve                 DECODE ONLY
# ===========================================================================
#
# The largest node-count win, and the one nothing off-the-shelf does.
#
# WHY IT IS DEAD. At chunk extent 1, ``strict_lower_tri = tril(1,1) - eye(1)``
# is the 1x1 zero matrix, so ``attn = -((k_beta @ key.T) * decay_mask) *
# strict_lower_tri`` is identically zero whatever the data. ``_solve_triangular``
# then evaluates, exactly:
#
#     M_mat = I - 0 = I        acc = I + 0 = I       power = 0, 0 @ 0 = 0
#     M0    = mask_acc * I = I  E0 = I - I @ I = 0   E1 = E2 = E3 = 0
#     return  I @ (I+0) @ (I+0) @ (I+0) @ (I+0)  =  I
#
# Eleven matmuls to produce the identity, plus the two that consume it
# (``value = inv @ v_beta`` and ``k_cumdecay = inv @ (...)``). Thirteen matmuls
# and ~14 elementwise ops per layer computing nothing, for 44 MACs of real
# arithmetic -- roughly 39 of the 77 top-level MatMuls in the decode graph.
#
# WHY FOLDING CANNOT DO IT. Folding evaluates nodes whose inputs are all
# constant. ``attn``'s other operands are data, so ``Mul(data, [[0.0]])`` is
# not foldable and the folder correctly stops there. Confirmed: every one of
# these matmuls survives ``ORT_ENABLE_ALL``.
#
# THE REWRITE. Two algebraic identities, which hold regardless of the unknown
# operand's value:
#
#     (a)  Mul(x, 0)     -> 0
#     (b)  MatMul(I, x)  -> x
#
# Only (a) needs the insight. Once it fires, ``attn`` is a constant and
# ORDINARY CONSTANT FOLDING CASCADES THROUGH THE WHOLE SOLVE unaided --
# ``I - 0``, ``I + 0``, ``0 @ 0``, the Taylor loop, all four squarings, the
# product chain -- until the solve's output is the literal identity. (b) then
# removes the two matmuls that consume it, which folding cannot touch because
# ``v_beta`` is data. So the pass is: apply (a), re-fold, apply (b), re-fold.
#
# HOW TO FIND IT. After pass 1 the decode graph contains an all-zero 1x1
# initializer feeding one ``Mul`` per linear-attention layer, each of whose
# output feeds ``[MatMul, MatMul, MatMul, Add, Sub]`` -- the entry to the
# Newton chain. In the reference export those were ``sub_143`` (value
# ``[[0.0]]``) feeding ``mul_886``, ``mul_1635`` and ``mul_2384``. Do not match
# on names, which are export-order dependent: scan for ``Mul`` nodes with an
# all-zero constant input. The prefill graph has no such initializer -- at
# extent 64 the mask is a real strictly-lower-triangular matrix -- which is
# why this pass is decode-only and must be safe to run as a no-op on prefill.
#
# For (b), verify the constant really is an identity matrix before rewriting,
# and check the batch dims broadcast compatibly. At extent 1 it is the 1x1
# ``[[1.0]]``, i.e. a scalar multiply by one.
#
# SOUNDNESS. ``0 * x = 0`` is FALSE in IEEE-754 when x is Inf or NaN
# (``0 * Inf = NaN``). This is a fast-math rewrite, and that is very likely why
# ORT declines to do it in general. It is sound here for a specific reason:
# quantization makes finiteness a graph invariant, because every activation on
# this path carries an encoding that clamps it to a finite range, so Inf/NaN
# cannot reach the Mul. State that precondition explicitly in the pass -- it is
# the difference between a sound local rewrite and a footgun someone later
# points at a float graph.
#
# OPTIONAL EXTENSION. ``Sub(x, x) -> 0`` additionally collapses ``decay_mask``,
# which at extent 1 is ``exp(g_cum - g_cum) * 1 = 1``, removing two more Muls
# in the Scan body. It requires recognising the *same tensor* on both inputs
# (both are unsqueezes of ``g_cum``), and carries the same NaN caveat.
#
# ===========================================================================
# Pass 3 -- Drop zero-width Pad                                 DECODE ONLY
# ===========================================================================
#
# At extent ``min(S, chunk_size)`` the pad is zero whenever ``S <= chunk_size``,
# so at decode all 15 ``Pad`` nodes are no-ops with literal zero pad amounts.
# They survive ``ORT_ENABLE_ALL`` unchanged.
#
# Rewrite: a ``Pad`` whose ``pads`` input is a constant of all zeros becomes
# ``Identity`` and is removed. Guard on the pads being *constant* -- in the
# prefill specialization they are non-zero, and pre-hardening they are computed.
#
# ===========================================================================
# Pass 4 -- Drop full-extent Slice                              DECODE ONLY
# ===========================================================================
#
# The kernel's closing ``core_attn_out[:, :, :seq_len]`` undoes the pad. When
# the pad was zero the slice spans the whole axis and is a no-op. ORT's
# EXTENDED level removes some (22 -> 12) but leaves others, and EXTENDED is
# unusable for deployment (pass 1).
#
# Rewrite: a ``Slice`` whose ``starts``/``ends``/``steps`` are constants
# covering the full extent of a now-static axis becomes ``Identity``. This
# needs pass 0 to have made the axis static -- against a symbolic dim the
# comparison cannot be made.
#
# ===========================================================================
# Pass 5 -- Inline a trip-count-1 Scan                          DECODE ONLY
# ===========================================================================
#
# Plausibly the biggest *real* win, and certainly the one least likely to come
# for free from a backend.
#
# At decode there is exactly one chunk, so each of the three ``Scan`` nodes
# runs a single iteration. The loop machinery is then pure overhead: the carry
# is marshalled in and out for one pass, and the subgraph boundary blocks
# fusion between the body and the surrounding graph. ``Scan=3`` survives every
# ORT level.
#
# Rewrite: when a ``Scan``'s scan axis has static extent 1, splice the body into
# the parent graph -- squeeze the scanned inputs along the scan axis, wire them
# to the body's inputs, rename body nodes to avoid collisions, unsqueeze the
# scan outputs back, and connect the final carry directly. Assert the extent is
# 1 rather than assuming it; at prefill it is ``S / chunk_size``.
#
# This is worth doing regardless of whether QNN supports Scan well: if support
# is weak it is a correctness/performance necessity, and if it is good it still
# removes a fusion barrier. It is also the pass most likely to be reusable
# outside this model.
#
# ===========================================================================
# Appendix A -- The conv roll is NOT a graph rewrite
# ===========================================================================
#
# ``exportable_gated_delta_net_forward`` rolls the real tokens to the front so
# the cached ``conv_state`` sits adjacent to them, and rolls the outputs back:
# 10 ``GatherElements`` plus index arithmetic across the linear layers. At
# decode ``npad = 1 - n_real`` is always 0, because the single token being
# decoded is real by construction -- so both rolls are the identity.
#
# But ``npad`` is derived from the attention mask, i.e. from DATA. No folder can
# prove it, and no algebraic identity applies. Removing these requires
# *asserting* "at seq_len == 1 the query token is always real", which narrows
# the graph's contract rather than simplifying its algebra. Keep it in a
# separate category from passes 2-5, and if it is ever done, make the assertion
# explicit and checkable at runtime rather than silent.
#
# ===========================================================================
# Appendix B -- Measured baseline (what to expect, and what ORT will not do)
# ===========================================================================
#
#   hardened, both specializations : 625 nodes
#       {Scan:3, MatMul:80, Pad:15, Slice:22, Min:1, Range:4, Trilu:2, Equal:1}
#
#   after pass 1 (ORT BASIC):
#       prefill : 582 nodes  Scan body 72 nodes, 15 MatMul, 201,326,592 MACs/iter
#       decode  : 571 nodes  Scan body 69 nodes, 15 MatMul,   2,371,584 MACs/iter
#
# The 85x MAC ratio is the payoff of the derived chunk extent itself, already
# realised. The Scan body's 15 MatMuls are 5 per layer -- the genuine
# recurrence (``q@k.T``, ``attn@v_new``, ``kc@state``, ``q'@state``,
# ``k.T@v_new``) -- so the ARITHMETIC is essentially at its floor and passes
# 2-5 buy node count, not MACs. The dead solve is intra-chunk and therefore
# lives in the TOP-LEVEL graph, not the Scan body; that is where pass 2 acts.
#
# Whether ~80 fewer nodes matters is a dispatch-overhead question that only an
# on-target profile answers. Profile before building passes 2-4. Also check
# first whether QAIRT's own folding already implements ``Mul(x, 0) -> 0``: it
# does constant folding and some algebraic simplification, and if that rule is
# in there, pass 2 comes free.
#
# ===========================================================================
# Appendix C -- Ordering, verification, and where this code belongs
# ===========================================================================
#
# Order: 0 (harden) -> 1 (fold) -> 2 (solve) -> 1 -> 3, 4 (no-op Pad/Slice)
# -> 5 (inline Scan) -> 1. Re-fold after 2 and 5, since both expose new
# constant regions.
#
# Verification, per pass: run the graph in ORT at that specialization's
# sequence length before and after, on the same inputs, and require bitwise or
# near-bitwise agreement (these are all supposed to be semantics-preserving at
# that shape). Then assert the structural property the pass claims -- no
# ``dim_param`` anywhere after 0, no all-zero-mask ``Mul`` after 2, no
# zero-width ``Pad`` after 3, no ``Scan`` after 5. A pass that changes numerics
# is a bug, not a tradeoff.
#
# These passes do NOT belong in this file when implemented. This module is a
# model transform that runs before export; the passes operate on a hardened
# ONNX artifact after quantsim export. They belong in the export/deploy layer
# -- alongside whatever performs pass 0 today -- and each needs its own tests
# on a small fixture graph. They are specified here only because this is where
# the reasoning about *why* they are safe lives.
#
# Not graph passes, but the other half of the decode story, recorded in
# ``exportable_linear_attention_dynamic_chunk.md``: merging the two ``@ state``
# matmuls (which read the same state and are memory-bound GEMVs at decode) into
# one, and hardening an extra AR-N specialization for speculative decode, where
# ``min(S, chunk_size) == S`` gives one chunk and amortises the state traffic.


#: Model types whose decoder stacks contain GatedDeltaNet layers. Qwen3.5-MoE is
#: registered under both its top-level (VLM) and text-config model types so the
#: adaptation resolves whichever one the config reports.
_LINEAR_ATTENTION_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text")


def register_adaptations() -> None:
    """Register ExportableLinearAttention for every linear-attention model type.

    Called at import; also callable by tests, which run under an autouse fixture
    that wipes ``adaptation_lookup`` before each test.
    """
    for model_type in _LINEAR_ATTENTION_MODEL_TYPES:
        YAMLConfigParser.register_adaptation(
            "ExportableLinearAttention",
            model_type=model_type,
            required_for_export=True,
        )(Qwen3_5ExportableLinearAttentionAdaptation)


register_adaptations()
