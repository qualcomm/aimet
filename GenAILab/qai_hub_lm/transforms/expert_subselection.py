# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Expert subselection: cap the experts one prefill call may light up.

On target, a MoE layer runs a fixed number of expert slots. The tokens of one
prefill call together route to far more than ``k`` experts, so the deployed
model first keeps the ``S`` experts with the highest peak routing probability
over the call's tokens, and routes each token among those only::

    probs = softmax(logits)                  # [B, T, E]
    score = probs.amax(dim=1)                # [B, E]: max over this call's tokens
    keep  = one_hot(topk(score, S))          # [B, E]
    topk(probs * keep, k) -> renormalize     # the usual per-token routing

This changes outputs, so it is its own adaptation rather than a mode of the
output-equivalent ``ExportableMoE`` adaptation.

Two properties the rest of the pipeline relies on:

* **Exact at decode.** At ``T == 1`` the score is the token's own probabilities,
  so its top ``k`` lie inside its top ``S`` (``S > k``) and the mask only zeroes
  values top-k never picks. One graph therefore serves prefill and decode;
  hardening deletes the mask chain from the decode graph.
* **Sequence length changes accuracy.** For ``T > 1`` the kept set depends on
  which tokens share a call, so a model calibrated or evaluated at one prefill
  length is not exactly the model the device runs at another. Nothing enforces
  a match: quantizing at a longer length than the device prefills at is a
  deliberate speed/memory trade-off (open item A1 in the design doc).

The mask is plain tensor ops inside the router's forward, so aimet-torch (which
quantizes modules) never quantizes it: it carries routing decisions.

Design: ``exportable_moe_expert_subselection.md``. The graph passes that turn
the export into efficient prefill and decode graphs are specified at the end of
this file.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_moe import modeling_qwen3_moe

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.transforms.exportable_moe import (
    _MOE_MODEL_TYPES,
    ExportableMoEMixin,
)

#: MoE blocks whose router this adaptation replaces.
_MOE_BLOCKS = (
    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock,
    modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock,
)


class SubselectedTopKRouter(nn.Module):
    """Drop-in for a Qwen MoE ``TopKRouter`` that routes among a per-call subset.

    Takes the per-sequence ``[B, T, H]`` view -- the stock router receives the
    flattened ``[B*T, H]`` and cannot tell sequences apart -- and returns the
    stock ``(router_logits, router_scores, router_indices)``, flattened to
    ``[B*T, ...]``.

    The projection is an ``nn.Linear`` leaf, so quantsim quantizes it without a
    bespoke definition and routing reads the quantized logits, as the exported
    graph does on target.
    """

    def __init__(self, router: nn.Module, num_selected_experts: int):
        super().__init__()
        self.top_k = router.top_k
        self.num_experts = router.num_experts
        self.hidden_dim = router.hidden_dim
        # Qwen3.5-MoE's router renormalizes unconditionally and has no flag.
        self.norm_topk_prob = getattr(router, "norm_topk_prob", True)
        if not self.top_k < num_selected_experts <= self.num_experts:
            raise ValueError(
                f"num_selected_experts must be in ({self.top_k}, {self.num_experts}] "
                f"(greater than num_experts_per_tok, at most num_experts), got "
                f"{num_selected_experts}. At S <= k a decode token could be forced "
                "onto zero-weight experts, and decode would no longer match the "
                "stock router."
            )

        # Shared, not copied: the stock router is discarded.
        self.proj = nn.Linear(
            self.hidden_dim, self.num_experts, bias=False, device="meta"
        )
        self.proj.weight = router.weight
        self.num_selected_experts = num_selected_experts

    def forward(self, hidden_states: torch.Tensor):
        router_logits = self.proj(hidden_states)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        # Keep the S experts with the highest peak probability over this call's
        # tokens. Per sequence ([B, T, E] -> [B, 1, E]), never across the batch.
        score = router_probs.amax(dim=1, keepdim=True)
        kept = torch.topk(score, self.num_selected_experts, dim=-1).indices
        mask = torch.zeros_like(score).scatter(-1, kept, 1.0)
        masked_probs = router_probs * mask
        top_value, top_index = torch.topk(masked_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        top_value = top_value.to(router_logits.dtype)
        return (
            router_logits.reshape(-1, self.num_experts),
            top_value.reshape(-1, self.top_k),
            top_index.reshape(-1, self.top_k),
        )


def _subselected_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """The stock ``SparseMoeBlock.forward``, except routing sees ``[B, T, H]``."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    _, routing_weights, selected_experts = self.gate(hidden_states)
    expert_output = self.experts(
        hidden_states_reshaped, selected_experts, routing_weights
    )
    if hasattr(self, "shared_expert"):
        # Qwen3.5-MoE: an always-active shared expert behind a sigmoid gate.
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def apply_expert_subselection(model: nn.Module, num_selected_experts: int) -> list[str]:
    """Give every MoE block a :class:`SubselectedTopKRouter`.

    :return: qualified names of the blocks adapted.
    :raises RuntimeError: if the model has no MoE block this module knows --
        silently doing nothing would evaluate the unconstrained model.
    """
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, _MOE_BLOCKS)
    ]
    if not targets:
        raise RuntimeError(
            "ExpertSubselection found no MoE block (looked for "
            f"{[cls.__name__ for cls in _MOE_BLOCKS]})."
        )
    for _, block in targets:
        block.gate = SubselectedTopKRouter(block.gate, num_selected_experts)
        block.forward = types.MethodType(_subselected_moe_block_forward, block)
    return [name for name, _ in targets]


# ---------------------------------------------------------------------------
# Adaptation registration
# ---------------------------------------------------------------------------


class ExpertSubselectionAdaptation:
    """Caps per-call expert selection at ``num_selected_experts``.

    ``num_selected_experts`` (S) has no default: it is a property of the
    deployment. Not ``required_for_export``, since it changes outputs; export
    still needs the ``ExportableMoE`` adaptation, which is.
    """

    num_selected_experts: int | None = None

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        if cls.num_selected_experts is None:
            raise ValueError(
                "ExpertSubselection needs `num_selected_experts`, e.g.\n"
                "  adaptations:\n"
                "    - ExportableMoE\n"
                "    - ExpertSubselection:\n"
                "        num_selected_experts: 32"
            )
        # Checked on the class, not the model, so list order does not matter.
        if not issubclass(cls, ExportableMoEMixin):
            raise ValueError(
                "ExpertSubselection needs the ExportableMoE adaptation alongside it: "
                "without it the experts cannot be quantized or exported. Add "
                "`- ExportableMoE` to model.adaptations."
            )
        model = super().instantiate_model(*args, **kwargs)
        apply_expert_subselection(model, cls.num_selected_experts)
        return model


def register_adaptations() -> None:
    """Register ExpertSubselection for every MoE model type.

    Also called by tests, which run under a fixture that wipes the registry.
    """
    for model_type in _MOE_MODEL_TYPES:
        YAMLConfigParser.register_adaptation(
            "ExpertSubselection", model_type=model_type
        )(ExpertSubselectionAdaptation)


register_adaptations()


# ---------------------------------------------------------------------------
# Hardening-time graph passes (specification -- not implemented here)
# ---------------------------------------------------------------------------
#
# The export is one graph with a symbolic sequence length T. To land on target
# it is hardened twice -- prefill at the on-target chunk length, decode at T=1
# -- and each copy is rewritten into the form the compiler's MoE lowering
# ("dynamic weight switching") matches. The compiler does the expensive part
# itself: it replaces the E static expert branches with a few *expert slots*,
# each running whichever expert is selected (``ElementWiseMux`` over the
# experts' weights) -- S slots in prefill, k in decode.
#
# So these passes do not build slots. They only hand the compiler a graph that
# is (a) numerically identical to the calibrated one and (b) in its "original
# graph" form:
#
#   decode (AR1)    Softmax -> TopK(k) -> ReduceSum/Div -> ScatterElements
#                   -> per expert e: Gather[e] weight, Linear/act/Linear/Linear,
#                      Mul(weight) -> cascade Add from 0
#   prefill (AR-N)  the same, with the S-expert pre-selection ahead of TopK(k),
#                   and each expert's contribution gated by
#                   Where(Greater(ReduceSum(Equal(topk_indices, e)), 0), y, 0)
#
# What we export differs from that in five ways; each pass removes one.
#
#   1. T is symbolic, in the outer graph and inside every If body.     pass 0
#   2. Decode still carries the subselection chain, dead at T=1.       pass 2
#   3. Indices pass through a dropped-token sentinel guard.            pass 3
#   4. Each expert is an ``If`` predicated on its routing weight,
#      with a ``force_all`` switch and an input mask.                  passes 4-5
#   5. The router's op spelling differs (only matters if the
#      compiler's matcher is op-exact).                                pass 6
#
# Measurements below are on the toy Qwen3-MoE block from the tests (E=8, k=2,
# S=3, predicated, T symbolic): 90 top-level nodes, 8 ``If``, one sequence
# symbol. They are structural (op counts and shapes), not numerics.
#
# ===========================================================================
# Global constraints
# ===========================================================================
#
#   * Run the passes AFTER quantsim export, on the exported float graph plus
#     its encodings file (measured: the aimet-onnx export holds no
#     QcQuantizeOp, so nothing blocks folding). Deleted tensors leave orphaned
#     encodings, harmless for name-keyed lookup. Tensors a pass ADDS must get
#     the encoding of the tensor they stand in for; every such addition below
#     is grid-preserving, so copying is exact.
#   * Find targets by module scope, not by op-type pattern matching. Measured:
#     aimet-onnx's export keeps every node name and its
#     ``pkg.torch.onnx.name_scopes`` metadata (128/128 on the dense block), so
#     ``gate`` and ``experts`` are reliable addresses.
#     ``[unverified]`` for nodes inside If bodies.
#   * Anything a pass deletes must carry no output quantizer of its own, or
#     deleting it changes numerics. That is a CALIBRATION-TIME precondition --
#     hardening cannot retrofit it -- and today it mostly does not hold:
#
# Preconditions: quantizer placement in the ONNX sim
# ---------------------------------------------------------------------------
#
# Default aimet-onnx placement on the dense block, measured:
#
#   router      MatMul (logits) ON, Softmax ON; subselect chain: ReduceMax and
#               TopK off, ConstantOfShape, ScatterElements and the masking Mul
#               ON
#   selector    Less/Where/Cast -, sentinel Mul ON, ConstantOfShape ON,
#               ScatterElements ON (the dense [T, E] routing weights)
#   per expert  Slice off, Abs ON, Sign ON, Max(force_all) ON, input-mask Mul
#               ON, Gemm/Sigmoid/Mul ON, contribution Mul ON, accumulator Add ON
#
#   P0  The subselect chain (ConstantOfShape, ScatterElements, masking Mul)
#       -> off. Pass 2 deletes it from decode; its values are exact 0/1 masks
#       and re-quantized probabilities, so this costs nothing at calibration.
#   P1  Abs, Sign, Max (the predicate/mask chain) -> off. Passes 4-5 delete
#       them; quantizing a 0/1 mask is meaningless anyway.
#   P2  The sentinel Mul (weights * in_range) -> off. Pass 3 deletes it; it
#       multiplies by exactly 1, so with no quantizer of its own it is exact.
#   P3  The input-mask Mul -> off. Then each expert Gemm reads the block
#       input's encoding times an exact 0/1 mask, which is what it reads on
#       device once pass 5 removes the mask.
#   P4  Accumulator Adds -> only the final sum quantized. A partial sum over
#       experts 0..e has no counterpart once the compiler regroups experts into
#       slots, so no rewrite can preserve its encoding. aimet-torch already
#       leaves these unquantized (a plain ``+`` in ``PredicatedExperts``, not a
#       module), so this also aligns the ONNX sim with the torch sim. How the
#       compiler encodes the Adds it rebuilds is open (A7 in the design doc).
#
#   Kept quantized and carried 1:1: router logits, routing weights (inherited
#   by each ``Gather[e]``, pass 5), expert Gemm/activation outputs and the
#   per-expert contribution Mul. Whether the device keeps these per expert or
#   per slot is open (A2) -- no pass can fix that.
#
# Status: none of P0-P4 is implemented. They are the first thing to implement,
# as one mechanism, and they change calibration for every MoE ONNX run.
#
# ===========================================================================
# Pass 0 -- Harden T                                            PREREQUISITE
# ===========================================================================
#
# Identical to linear attention's pass 0 (``exportable_linear_attention.py``),
# and should be the same code. Measured: the one sequence symbol appears on the
# outer graph AND on every If body's declarations, so the substitution must be
# recursive; the full model adds derived dims (``CL - T``) that need evaluating,
# not name-matching. ``k`` and ``S`` are Python ints at export, so both TopK
# ``K`` inputs are already constants.
#
# ===========================================================================
# Pass 1 -- Constant-fold                                       PREREQUISITE
# ===========================================================================
#
# ``ORT_ENABLE_BASIC``, with the same warning as linear attention: not
# EXTENDED/ALL, which emit ``com.microsoft`` ops. Folds the shape-derived zeros
# -- the accumulator start (Shape -> Expand), the dense-weight base (Concat ->
# ConstantOfShape) and the subselection mask's ConstantOfShape. Re-run after
# passes 2-5, which each expose more.
#
# ===========================================================================
# Pass 2 -- Delete the subselection                             DECODE ONLY
# ===========================================================================
#
# WHY IT IS DEAD. At T=1 the max over T is the token's own probabilities, so
# the mask marks that token's top S. With S > k its top k lie inside them, so
# ``TopK(probs * mask, k) == TopK(probs, k)``: the mask zeroes only entries
# top-k never reaches. Both TopKs break ties the same way (by index), so this
# holds on ties too. Tested bit-identical in torch
# (``test_expert_subselection.py::TestDecodeIsExact``).
#
# Deleting it is for speed, not numerics: left in, the decode graph looks like
# the AR-N pattern and would likely be given S slots instead of k.
#
# WHAT. Inside the ``gate`` scope, the chain is the Softmax output's ReduceMax
# (the router's only one) -> TopK(S) -> ConstantOfShape -> ScatterElements,
# plus its one consumer, the masking Mul. Replace uses of that Mul's output
# with its other input (the Softmax output), delete the Mul, then
# dead-code-eliminate the chain.
#
# PRECONDITIONS, asserted: the router input's sequence dim is 1 after pass 0;
# the subselection TopK's ``K`` is greater than the per-token TopK's (S > k).
#
# VERIFY. ORT bit-exact at T=1; no ReduceMax left in ``gate``; one TopK per
# router. The router is now the AR1 pattern's.
#
# ===========================================================================
# Pass 3 -- Drop the dropped-token sentinel guard               BOTH
# ===========================================================================
#
# ``PredicatedExperts._selection_weights`` guards index == E, qwen4_exp's
# dropped-token sentinel: Less(idx, E) -> Where(idx, 0) and Cast -> Mul(weights,
# in_range). Measured: 6 nodes per layer (Less, Shape, Expand, Where, Cast, Mul).
#
# For Qwen3-/Qwen3.5-MoE the indices come from a TopK over the expert axis, so
# ``idx < E`` always and ``in_range`` is all ones: Where -> idx, Mul -> weights.
# Constant folding cannot see that (it is data-dependent); the pass proves it
# structurally by checking the index producer is that TopK. Precondition P2.
#
# VERIFY. ORT bit-exact; TopK indices feed ScatterElements directly, as in the
# pattern.
#
# ===========================================================================
# Pass 4 -- Inline the expert Ifs                               BOTH
# ===========================================================================
#
# Replace each ``If`` with its then-branch, unconditionally. Exact: when the
# predicate is false the expert's weight column and input mask are all zero,
# so the then-branch computes ``acc + expert(0) * 0 = acc`` -- what the
# else-branch (an Identity) returns.
#
# Measured: bodies take no inputs (they capture outer names) and their
# internal names are unique (``*_true_graph_0``), so inlining is a move, and it
# keeps the tensor names that encodings inside bodies will be keyed on. The
# then-branch is the same op sequence as the ``dense`` export per expert
# (measured: Mul, Gemm, Sigmoid, Mul, Gemm, Mul, Gemm, Mul, Add), so after this
# pass the graph IS the dense form. That is also how to build and test passes
# 5-6 before aimet-onnx can quantize inside If bodies: export ``dense``, skip
# this pass.
#
# The predicate's tail (Reshape -> ReduceMax -> Cast) loses its only consumer
# and goes to DCE; Abs/Sign/Max still feed the input mask until pass 5.
#
# ===========================================================================
# Pass 5 -- Rewrite each expert into the pattern's branch       BOTH (differs)
# ===========================================================================
#
# Per expert e, the dense form is (measured):
#
#   w_e = Slice(W, e:e+1);  m_e = Max(Sign(Abs(w_e)), force_all)
#   x_e = Mul(x, m_e) -> Gemm_gate, Gemm_up -> SiLU * up -> Gemm_down
#   acc = Add(acc, Mul(down, w_e))
#
#   a. Remove the input mask: Gemm_gate and Gemm_up read x. Delete the Mul,
#      Max, Sign, Abs and the ``force_all`` initializer (P1, P3). Exact:
#      unrouted rows now produce expert(x) instead of expert(0), still times
#      w_e = 0. Caveat: exact only while every token's expert output is finite
#      (``inf * 0 = NaN``). True in fp32; assert it in fp16 verification.
#      ``force_all`` never changes outputs (contract 2 in ``exportable_moe``),
#      so its value in the artifact does not matter.
#   b. Weight via ``Gather[e]``: replace ``Slice(W, e:e+1)`` with
#      ``Gather(W, [e], axis=-1)``. A shape-(1,) index keeps the [T, 1] shape,
#      so nothing else changes. The new tensor inherits W's encoding.
#   c. PREFILL ONLY -- gate the contribution:
#        c_e  = Greater(ReduceSum(Cast(Equal(Reshape(topk_idx, [T*k]), e))), 0)
#        acc  = Add(acc, Where(c_e, Mul(down, w_e), 0))
#      Value-identity: c_e false means column e of W is all zero, so the
#      contribution is already 0. It exists so the compiler can slot the
#      expert, and it is derived from the indices, not the weights, because the
#      compiler drives both the weight Mux and this Equal from the same slot-ID
#      Gather. Where's output inherits the contribution's encoding (0 is exactly
#      representable). Decode gets no Where; the AR1 pattern has none.
#   d. Keep the cascade Add from 0 (folded by pass 1): it is the pattern's
#      original aggregation. P4 applies.
#
# Qwen3.5-MoE's shared expert, its sigmoid gate and the final Add sit outside
# the routed experts and are left alone. Whether the compiler matches a block
# that has them is open (A6).
#
# VERIFY. ORT bit-exact at the specialization's T; no ``If``; no Abs/Sign/Max
# in the ``experts`` scope; per expert exactly one Gather with constant index e
# feeding the contribution Mul; prefill has E Where nodes, decode none.
#
# ===========================================================================
# Pass 6 -- Canonicalize the router spelling                    CONDITIONAL
# ===========================================================================
#
# Only if the compiler's matcher is op-exact (open, A8). The page's AR-N router
# builds the mask with Mul(*0)/Add(+1)/ScatterElements where we have
# ConstantOfShape/ScatterElements, and normalizes via GatherElements ->
# ReduceSum(axis=2) -> Div where we divide TopK.values by their ReduceSum. Both
# pairs are numerically identical.
#
# Scoring with ReduceMax where the page shows ReduceSum is NOT a spelling
# difference: they pick different experts. That needs the compiler to accept
# max (A4); no pass can bridge it.
#
# ===========================================================================
# Appendix -- Ordering, verification, and where this code belongs
# ===========================================================================
#
# Order:
#   prefill  0 -> 1 -> 3 -> 4 -> 5 -> 1 (-> 6)
#   decode   0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 1 (-> 6)
#
# Verification, per pass: ORT at that specialization's T, before and after, on
# the same inputs, bit-exact -- on the float graph, and on a QDQ graph built
# from the encodings file (exact given P1-P4). Then the structural asserts
# listed with each pass. A pass that changes numerics is a bug.
#
# Scale, derived from the toy's per-expert counts (not measured): Qwen3-30B-A3B
# (E=128, 48 MoE layers) exports 6144 ``If`` nodes, each behind a 7-node
# predicate chain (Slice, Abs, Sign, Max, Reshape, ReduceMax, Cast). Passes 4-5
# delete the Ifs, the input-mask Muls and 6 of those 7; the Slice becomes the
# ``Gather[e]``.
#
# These passes do NOT belong in this file when implemented. They operate on an
# exported artifact, so they belong in the export/deploy layer, sharing pass 0
# and pass 1 with linear attention's (nothing in-tree implements either yet).
# Passes 3-5 apply to any predicated MoE export, not only a subselected one.
# Each needs a fixture-graph test; the 90-node toy block above is small enough.
# They are specified here because this is where the reasoning about *why* they
# are exact lives.
