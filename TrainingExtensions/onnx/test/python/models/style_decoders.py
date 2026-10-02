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
    """Standard multi-head attention: split into heads, softmax(QK^T/sqrt(d))V, merge.

    :param q: [B, S, hidden]
    :param k: [B, S, hidden]
    :param v: [B, S, hidden]
    :return: [B, S, hidden]
    """
    B, S, _ = q.shape
    q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
    k = k.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    y = torch.matmul(attn, v)  # [B, H, S, D]
    return y.transpose(1, 2).reshape(B, S, num_heads * head_dim)


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
