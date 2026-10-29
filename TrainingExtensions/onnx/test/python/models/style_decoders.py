# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tiny hand-built decoder / ViT / VLM fixtures for topology and SpinQuant tests.

These are minimal, architecture-flavored ``nn.Module`` s exported to ONNX to
exercise decoder-stack topology detection (``llm_topology``) and the SpinQuant
rotation passes. They are shared by ``test_llm_topology.py`` (pure topology) and
``test_spinquant.py`` (rotation), so both suites see identical graphs.

Naming mirrors real HF architectures at a coarse level:

* :class:`LlamaStyleDecoder` — dense pre-norm, unfused q/k/v and gate/up.
* :class:`Qwen3StyleDecoder` — internal q_norm/k_norm (excluded from active norms).
* :class:`Phi3StyleDecoder`  — fused ``qkv_proj`` / ``gate_up_proj``.
* :class:`Gemma3StyleDecoder` — post-writing norms after o_proj / down_proj.
* :class:`Gemma4StyleDecoder` — later attention reuses an earlier block's K/V.
* :class:`ViTEncoder` / :class:`LayerNormViTEncoder` — vision encoders + PatchMerger.
* :class:`VLMBackbone` — decoder consuming ``inputs_embeds`` (no embed Gather).
"""

import io

import numpy as np
import onnx
import onnx_ir
import torch
import torch.nn as nn
from onnx import load_model

from aimet_onnx.graph_passes.fusions import fuse_supergroups

from .test_models import RMSNorm

# ---------------------------------------------------------------------------
# Shared dimensions. ``_H == _NUM_HEADS * _HEAD_DIM`` must hold.
# ---------------------------------------------------------------------------
_NORM_KW = dict(mul_for_pow=False, mul_rsqrt_pattern="mul_rsqrt")
_H, _I = 8, 16  # hidden dim, intermediate dim
_NUM_HEADS, _HEAD_DIM = 2, 4  # _H == _NUM_HEADS * _HEAD_DIM
_VOCAB = 16
_B, _SEQ = 1, 4  # batch, sequence length
_GEMMA4_PLE_DIM = 4
_GEMMA4_SHARED_I = 2 * _I

#: ``model_type`` to analyze hand-built decoder fixtures with. Their modules carry no
#: HuggingFace names, so it must be one that ``analyze_llm_topology`` analyzes by
#: active norms rather than by name; which one is otherwise irrelevant.
STRUCTURAL_MODEL_TYPE = "llama"

# ViT dims.
_VIT_D, _VIT_I = 8, 12  # ViT hidden size, intermediate dim
_VIT_S_SQ = 4
_VIT_N = 8
_VIT_D_L = _H  # language hidden size; must equal backbone _H for VLM integration tests


# ---------------------------------------------------------------------------
# Export helpers.
# ---------------------------------------------------------------------------
def _export_to_onnx(
    module: nn.Module,
    dummy_input: torch.Tensor,
    opset: int = 17,
    do_constant_folding: bool = True,
):
    """Export via the legacy torchscript exporter.

    ``dynamo=False`` is deliberate and not currently parameterizable: the dynamo
    exporter renames nodes to ``node_linear_N``, which defeats the name-based
    ``classify_linear_role`` matching these fixtures assert on (q/k/v roles come
    back empty). Dynamo is covered where names are not asserted — see the
    ``("torchscript", "dynamo")`` parametrization over real HF configs in
    ``test_llm_topology_integration.py``.
    """
    buf = io.BytesIO()
    torch.onnx.export(
        module.eval(),
        dummy_input,
        buf,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=do_constant_folding,
        dynamo=False,
    )
    buf.seek(0)
    return load_model(buf)


def _attach_past_value_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Attach a dangling ``past_value_0`` graph input whose last dim is ``_HEAD_DIM``.

    The input is unused by any node in the graph; it exists solely so that
    ``_infer_head_dim`` can derive ``head_dim`` from a real export-style input,
    matching the HF/optimum convention where ``past_value_*`` tensors carry the
    KV cache. Static shape ``[1, 1, 1, _HEAD_DIM]`` is the smallest valid
    rank-4 KV-cache layout that gives the helper a static last dim to read.
    """
    past_value = onnx.helper.make_tensor_value_info(
        name="past_value_0",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[1, 1, 1, _HEAD_DIM],
    )
    model.graph.input.append(past_value)
    return model


def _export_decoder(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_B, _SEQ, _H)
    return _attach_past_value_input(_export_to_onnx(module, x))


def _export_decoder_with_ids(
    module: nn.Module, add_value_input=True
) -> onnx.ModelProto:
    token_ids = torch.randint(0, _VOCAB, (_B, _SEQ))
    model = _export_to_onnx(module, token_ids)
    if add_value_input:
        model = _attach_past_value_input(model)
    return model


def _export_gemma4_shared_kv_decoder(module: nn.Module) -> onnx.ModelProto:
    """Export a Gemma4-style decoder with per-source-layer K/V-cache inputs."""
    token_ids = torch.randint(0, _VOCAB, (_B, _SEQ))
    past_key_0 = torch.randn(_B, _SEQ, _H)
    past_value_0 = torch.randn(_B, _SEQ, _H)
    past_key_1 = torch.randn(_B, _SEQ, _H)
    past_value_1 = torch.randn(_B, _SEQ, _H)
    per_layer_inputs = torch.randn(_B, _SEQ, 4, _GEMMA4_PLE_DIM)
    buf = io.BytesIO()
    torch.onnx.export(
        module.eval(),
        (
            token_ids,
            past_key_0,
            past_value_0,
            past_key_1,
            past_value_1,
            per_layer_inputs,
        ),
        buf,
        input_names=[
            "input",
            "past_key_0_in",
            "past_value_0_in",
            "past_key_1_in",
            "past_value_1_in",
            "per_layer_inputs",
        ],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    buf.seek(0)
    return load_model(buf)


def _export_vit(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_VIT_N, _VIT_D)
    return _export_to_onnx(module, x)


def _export_vlm_backbone(module: nn.Module) -> onnx.ModelProto:
    inputs_embeds = torch.randn(_B, _SEQ, _H)
    return _export_to_onnx(module, inputs_embeds)


def _fuse_rms_norms(model: onnx.ModelProto) -> onnx.ModelProto:
    """Coalesce decomposed RMSNorm patterns into ``RMSNormalization`` supergroup ops.

    QuantizationSimModel applies this fusion before constructing its ConnectedGraph,
    so the spinquant analyzers see fused ops in production. Tests that build a bare
    ConnectedGraph need this shim to exercise the same branches.
    """
    ir_model = onnx_ir.from_proto(model)
    fused = fuse_supergroups(ir_model, patterns=["RMSNormalization"])
    return onnx_ir.to_proto(fused)


def _mha(q, k, v, num_heads, head_dim):
    """Standard multi-head attention supporting cached K/V sequences."""
    batch_size, query_length, _ = q.shape
    key_value_length = k.shape[1]
    q = q.reshape(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, key_value_length, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, key_value_length, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    y = torch.matmul(attn, v)
    return y.transpose(1, 2).reshape(batch_size, query_length, num_heads * head_dim)


# ---------------------------------------------------------------------------
# Decoder blocks and full decoders.
# ---------------------------------------------------------------------------
class _LlamaBlock(nn.Module):
    """Simplified LLaMA/Qwen2 decoder block: 2 active norms per block.

    input_norm feeds q/k/v projections.
    post_attn_norm feeds gate/up projections.

    :param bias: When True, enables bias on the writing layers (o_proj, down_proj).
        Reading layers (q, k, v, gate, up) always have bias=False.
    """

    def __init__(self, bias: bool = False):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q = nn.Linear(_H, _H, bias=False)
        self.k = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=bias)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=bias)

    def forward(self, x):
        h = self.input_norm(x)
        y = _mha(self.q(h), self.k(h), self.v(h), _NUM_HEADS, _HEAD_DIM)
        x = x + self.o(y)
        h2 = self.post_attn_norm(x)
        return x + self.down(self.gate(h2) * self.up(h2))


class LlamaStyleDecoder(nn.Module):
    """2-block LLaMA decoder + embed_tokens + final norm + lm_head: 5 active norms total.

    :param bias: When True, enables bias on the writing layers (o_proj, down_proj) in each block.
    """

    def __init__(self, bias: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _LlamaBlock(bias=bias)
        self.block1 = _LlamaBlock(bias=bias)
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Qwen3Block(nn.Module):
    """Qwen3-style decoder block: 4 norms per block, only 2 active (q_norm / k_norm are internal and their output not fed directly into a weight MatMul/Gemm/Conv ).

    "Qwen3-style" refers to *norm placement* only, not attention layout: this
    fixture is plain MHA, whereas real Qwen3 uses GQA. Head layout is irrelevant
    to the norm-detection branches this fixture exercises, and ``_H ==
    _NUM_HEADS * _HEAD_DIM`` with 2 heads leaves no room for distinct KV heads.
    Real GQA coverage lives in ``test_spinquant.py``, which parametrizes
    ``_export_qwen3_causal_lm_with_kv_cache(num_kv_heads=...)`` over MHA/GQA.
    """

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q_proj = nn.Linear(_H, _H, bias=False)
        self.k_proj = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.q_norm = RMSNorm(_H, **_NORM_KW)
        self.k_norm = RMSNorm(_H, **_NORM_KW)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        h = self.input_norm(x)
        q = self.q_norm(self.q_proj(h))
        k = self.k_norm(self.k_proj(h))
        v = self.v(h)
        y = _mha(q, k, v, _NUM_HEADS, _HEAD_DIM)
        x = x + self.o(y)
        h2 = self.post_attn_norm(x)
        return x + self.down(self.gate(h2) * self.up(h2))


class Qwen3StyleDecoder(nn.Module):
    """2-block Qwen3 decoder + embed_tokens + final norm + lm_head: 5 active norms total."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Qwen3Block()
        self.block1 = _Qwen3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Phi3Block(nn.Module):
    """Phi3-style decoder block: fused qkv_proj (single linear) and fused gate_up_proj (single linear)."""

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.qkv_proj = nn.Linear(_H, 3 * _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate_up_proj = nn.Linear(_H, 2 * _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        h = self.input_norm(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        attn = self.o(q + k + v)
        x = x + attn
        h2 = self.post_attn_norm(x)
        gate, up = self.gate_up_proj(h2).chunk(2, dim=-1)
        return x + self.down(gate * up)


class Phi3StyleDecoder(nn.Module):
    """2-block Phi3 decoder + embed_tokens + final norm + lm_head: 5 active norms total."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Phi3Block()
        self.block1 = _Phi3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Gemma3Block(nn.Module):
    """Gemma3-style decoder block: post-writing norms after o_proj and down_proj."""

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q = nn.Linear(_H, _H, bias=False)
        self.k = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.pre_ffn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)
        self.post_ffn_norm = RMSNorm(_H, **_NORM_KW)

    def forward(self, x):
        h = self.input_norm(x)
        attn = self.o(self.q(h) + self.k(h) + self.v(h))
        x = x + self.post_attn_norm(attn)
        h2 = self.pre_ffn_norm(x)
        ffn = self.down(self.gate(h2) * self.up(h2))
        return x + self.post_ffn_norm(ffn)


class Gemma3StyleDecoder(nn.Module):
    """2-block Gemma3 decoder"""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Gemma3Block()
        self.block1 = _Gemma3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Gemma4Block(nn.Module):
    """Decoder block that either computes or reuses K/V states."""

    def __init__(self, shared_kv: bool):
        super().__init__()
        self.shared_kv = shared_kv
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q_proj = nn.Linear(_H, _H, bias=False)
        self.q_norm = RMSNorm(_HEAD_DIM, **_NORM_KW)
        if not shared_kv:
            self.k_proj = nn.Linear(_H, _H, bias=False)
            self.k_norm = RMSNorm(_HEAD_DIM, **_NORM_KW)
            self.v_proj = nn.Linear(_H, _H, bias=False)
            self.v_norm = RMSNorm(_HEAD_DIM, elementwise_affine=False, **_NORM_KW)
        self.o_proj = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.pre_ffn_norm = RMSNorm(_H, **_NORM_KW)
        intermediate_size = _GEMMA4_SHARED_I if shared_kv else _I
        self.gate_proj = nn.Linear(_H, intermediate_size, bias=False)
        self.up_proj = nn.Linear(_H, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, _H, bias=False)
        self.post_ffn_norm = RMSNorm(_H, **_NORM_KW)
        self.per_layer_input_gate = nn.Linear(_H, _GEMMA4_PLE_DIM, bias=False)
        self.per_layer_projection = nn.Linear(_GEMMA4_PLE_DIM, _H, bias=False)
        self.post_per_layer_input_norm = RMSNorm(_H, **_NORM_KW)

    def forward(self, x, key, value, per_layer_input):
        h = self.input_norm(x)
        q = self.q_proj(h)
        q = self.q_norm(q.reshape(*q.shape[:-1], _NUM_HEADS, _HEAD_DIM)).reshape_as(q)
        if not self.shared_kv:
            projected_key = self.k_proj(h)
            projected_key = self.k_norm(
                projected_key.reshape(*projected_key.shape[:-1], _NUM_HEADS, _HEAD_DIM)
            ).reshape_as(projected_key)
            key = torch.cat((key, projected_key), dim=1)
            projected_value = self.v_proj(h)
            projected_value = self.v_norm(
                projected_value.reshape(
                    *projected_value.shape[:-1], _NUM_HEADS, _HEAD_DIM
                )
            ).reshape_as(projected_value)
            value = torch.cat((value, projected_value), dim=1)
        y = _mha(q, key, value, _NUM_HEADS, _HEAD_DIM)
        x = x + self.post_attn_norm(self.o_proj(y))
        h2 = self.pre_ffn_norm(x)
        gate = torch.nn.functional.gelu(self.gate_proj(h2), approximate="tanh")
        y = self.down_proj(gate * self.up_proj(h2))
        x = x + self.post_ffn_norm(y)
        residual = x
        h3 = torch.nn.functional.gelu(self.per_layer_input_gate(x), approximate="tanh")
        h3 = h3 * per_layer_input
        h3 = self.per_layer_projection(h3)
        x = residual + self.post_per_layer_input_norm(h3)
        return x, key, value


class Gemma4StyleDecoder(nn.Module):
    """Four-block decoder with Gemma4-style cross-block K/V sharing.

    Blocks 0 and 1 are the final non-shared layers of two attention types. Each
    consumes its own cache and exposes full-length K/V. Blocks 2 and 3 are shared
    layers of those respective types, so block 2 reuses block 0's K/V and block
    3 reuses block 1's K/V. Every block also applies Gemma4's gated
    ``per_layer_input`` projection after its feed-forward sublayer.
    """

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Gemma4Block(shared_kv=False)
        self.block1 = _Gemma4Block(shared_kv=False)
        self.block2 = _Gemma4Block(shared_kv=True)
        self.block3 = _Gemma4Block(shared_kv=True)
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(
        self,
        token_ids,
        past_key_0_in,
        past_value_0_in,
        past_key_1_in,
        past_value_1_in,
        per_layer_inputs,
    ):
        x = self.embed_tokens(token_ids)
        x, key_0, value_0 = self.block0(
            x, past_key_0_in, past_value_0_in, per_layer_inputs[:, :, 0, :]
        )
        x, key_1, value_1 = self.block1(
            x, past_key_1_in, past_value_1_in, per_layer_inputs[:, :, 1, :]
        )
        x, _, _ = self.block2(x, key_0, value_0, per_layer_inputs[:, :, 2, :])
        x, _, _ = self.block3(x, key_1, value_1, per_layer_inputs[:, :, 3, :])
        return self.lm_head(self.norm(x))


class VLMBackbone(nn.Module):
    """Backbone for VLM: takes inputs_embeds, no embed_tokens Gather."""

    def __init__(self):
        super().__init__()
        self.block0 = _LlamaBlock()
        self.block1 = _LlamaBlock()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, inputs_embeds):
        x = self.block0(inputs_embeds)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# ViT encoders (used by both PatchMerger detection and R1 merger rotation).
# ---------------------------------------------------------------------------
class _ViTBlock(nn.Module):
    """Minimal ViT transformer block."""

    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(_VIT_D, **_NORM_KW)
        self.qkv = nn.Linear(_VIT_D, _VIT_D, bias=False)  # reading — no bias rotation
        self.proj = nn.Linear(
            _VIT_D, _VIT_D, bias=True
        )  # writing — bias rotated with R_V
        self.norm2 = RMSNorm(_VIT_D, **_NORM_KW)
        self.gate_proj = nn.Linear(
            _VIT_D, _VIT_I, bias=False
        )  # reading — no bias rotation
        self.up_proj = nn.Linear(
            _VIT_D, _VIT_I, bias=False
        )  # reading — no bias rotation
        self.down_proj = nn.Linear(
            _VIT_I, _VIT_D, bias=True
        )  # writing — bias rotated with R_V

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.proj(self.qkv(h))
        h2 = self.norm2(x)
        return x + self.down_proj(self.gate_proj(h2) * self.up_proj(h2))


class _ViTMerger(nn.Module):
    """PatchMerger: ln_q -> view(-1, s²·d_V) -> linear1 -> GELU -> linear2."""

    def __init__(self):
        super().__init__()
        self.ln_q = RMSNorm(_VIT_D, **_NORM_KW)
        self.linear1 = nn.Linear(
            _VIT_D * _VIT_S_SQ, _VIT_D * _VIT_S_SQ, bias=False
        )  # reading — no bias rotation
        self.linear2 = nn.Linear(
            _VIT_D * _VIT_S_SQ, _VIT_D_L, bias=True
        )  # writing — bias rotated with R_L

    def forward(self, x):
        x = self.ln_q(x)
        x = x.reshape(-1, _VIT_D * _VIT_S_SQ)
        x = torch.nn.functional.gelu(self.linear1(x))
        return self.linear2(x)


class ViTEncoder(nn.Module):
    """ViT encoder: Conv patch_embed -> 2 ViT blocks -> PatchMerger."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv1d(_VIT_D, _VIT_D, kernel_size=1, bias=False)
        self.block0 = _ViTBlock()
        self.block1 = _ViTBlock()
        self.merger = _ViTMerger()

    def forward(self, x):
        h = self.patch_embed(x.T.unsqueeze(0)).squeeze(0).T
        h = self.block0(h)
        h = self.block1(h)
        return self.merger(h)


class _LayerNormViTBlock(nn.Module):
    """Qwen3-VL style ViT block using LayerNorm."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(_VIT_D)
        self.qkv = nn.Linear(_VIT_D, _VIT_D, bias=False)
        self.proj = nn.Linear(_VIT_D, _VIT_D, bias=False)
        self.norm2 = nn.LayerNorm(_VIT_D)
        self.gate_proj = nn.Linear(_VIT_D, _VIT_I, bias=False)
        self.up_proj = nn.Linear(_VIT_D, _VIT_I, bias=False)
        self.down_proj = nn.Linear(_VIT_I, _VIT_D, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.proj(self.qkv(h))
        h2 = self.norm2(x)
        return x + self.down_proj(self.gate_proj(h2) * self.up_proj(h2))


class LayerNormViTEncoder(nn.Module):
    """Qwen3-VL style ViT encoder."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv1d(_VIT_D, _VIT_D, kernel_size=1, bias=False)
        self.block0 = _LayerNormViTBlock()
        self.block1 = _LayerNormViTBlock()
        self.linear2 = nn.Linear(_VIT_D, _VIT_D_L, bias=False)

    def forward(self, x):
        h = self.patch_embed(x.T.unsqueeze(0)).squeeze(0).T
        h = self.block0(h)
        h = self.block1(h)
        return self.linear2(h)
