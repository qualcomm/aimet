# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R3 online Hadamard rotation pass.

R3 = H / sqrt(head_dim) is an online Hadamard inserted as a ``MatMul`` node
on each decoder block's Q-side and K-side, immediately upstream of the
attention QK^T MatMul. Because ``H @ H^T = I``,
``(Q @ H) @ (K @ H)^T = Q @ K^T`` — the two inserted rotations cancel
algebraically in float, so attention output is unchanged.

K-side placement: when the export has KV-cache inputs, R3 is inserted on the
current-K branch *before* the past-key ``Concat``. This is the paper-correct
placement: K values entering the cache (and so quantized for cache storage) are
already rotated, which is what motivates R3 in the first place. The model's
``present_key`` output now carries rotated K, and the runtime feeds it back
as ``past_key`` next step — the cache convention is self-consistent across
autoregressive steps.

Unlike R1 / R2, R3 introduces new ops and new initializers in the graph
rather than mutating existing weights. R3 is independent of R1 / R2 — composes
freely.

R3 inserts plain float ``MatMul`` Hadamards. It runs on the float ONNX model
before any ``QuantizationSimModel`` is built, so the sim — created afterward on
the rotated graph — wraps the inserted MatMuls in activation quantizers like
any other op, and ``compute_encodings`` calibrates them on the rotated values.

Limitations (this iteration):

* Requires a KV-cache-style export: the model must expose one ``past_key_*``
  graph input per decoder block. Each is consumed by exactly one ``Concat``
  (MHA) or num_heads concats (SHA) whose other input is the post-RoPE current K.
  The Concat output reaches the QK^T MatMul through pass-through ops only
  (Transpose / Reshape / Cast / Identity).
* Prefill-only exports without KV-cache inputs are not supported.

Note on ordering: R3 is the last pass in the pipeline. The IR nodes and values in
``ctx.backbone_topology`` stay valid across R3's insertions — an IR graph is
mutable, so splicing a MatMul onto an edge does not invalidate the handles either
side of it — but the *topology* no longer describes the graph exactly: the Q/K
edges it reported now run through a Hadamard. Passes that reason about those
edges should run before R3.
"""

from typing import List

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.spinquant.model_analysis import (
    BlockR3Anchors,
    find_r3_anchors,
)
from aimet_onnx.spinquant.passes.base import (
    RotationPass,
    SpinquantContext,
)
from aimet_onnx.spinquant.transforms import (
    insert_online_hadamard_node,
    hadamard_rotation_matrix,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


class R3RotationPass(RotationPass):
    """R3 online Hadamard rotation on Q and K paths into the QK^T MatMul.

    ``head_dim`` is read from :attr:`SpinquantContext.backbone_head_dim`,
    which the orchestrator derives from a ``past_value`` graph input. An
    export without KV-cache inputs cannot run R3.
    """

    @property
    def name(self) -> str:
        return "R3"

    def validate(self, ctx: SpinquantContext) -> None:
        """Verify the model exposes one past_key_* graph input per decoder block,
        each consumed by exactly one Concat reaching exactly one MatMul forward."""
        _require_head_dim(ctx)
        # find_r3_anchors performs all the structural validation and raises
        # with a precise per-block error if anything looks wrong.
        _get_or_build_anchor_cache(ctx)

    def apply(self, ctx: SpinquantContext) -> None:
        """Insert one online Hadamard MatMul on the Q path and one on the K path per block.

        * Q-side: the Hadamard is spliced on the edge feeding QK^T.
        * K-side: the Hadamard is spliced on the current-K edge feeding the
          past-key Concat, so the cache stores rotated K.

        The inserted MatMuls are plain float ops; the ``QuantizationSimModel``
        built afterward on the rotated graph wraps them in quantizers.
        """
        head_dim = _require_head_dim(ctx)
        anchors = _get_or_build_anchor_cache(ctx)
        ir_model = ctx.backbone_ir
        _logger.info(
            "Backbone: Applying R3 online Hadamard rotation per attention block "
            "(head_dim=%d, blocks=%d).",
            head_dim,
            len(anchors),
        )

        for block_idx, anchor in enumerate(anchors):
            self._rotate_q_side(ir_model, anchor, head_dim, block_idx)
            self._rotate_k_side(ir_model, anchor, head_dim, block_idx)
            _logger.debug(
                "R3 cache %s: inserted Q-side rotation before %s at input %s "
                "and K-side rotation before %s.",
                anchor.past_key_input_name,
                [node.name for node in anchor.qk_matmul_nodes],
                anchor.q_input_indices,
                [node.name for node in anchor.k_consumers],
            )

    @staticmethod
    def _rotate_q_side(ir_model, anchor, head_dim, block_idx) -> None:
        """Insert ``... -> R3 -> QK^T`` on the Q path."""
        h_mat = hadamard_rotation_matrix(head_dim)
        for idx, node in enumerate(anchor.qk_matmul_nodes):
            name_prefix = f"spinquant_block{block_idx}_q"
            if idx:
                name_prefix += f"_{idx}"
            insert_online_hadamard_node(
                ir_model,
                target_value=anchor.q_input_values[idx],
                consumer_nodes=[node],
                H=h_mat,
                name_prefix=f"{name_prefix}_R3",
            )

    @staticmethod
    def _rotate_k_side(ir_model, anchor, head_dim, block_idx) -> None:
        """Insert ``... -> R3 -> Concat`` on the current-K path.

        R3 is spliced on the current-K edge feeding the past-key Concat so K
        values entering the cache are already rotated.
        """
        h_mat = hadamard_rotation_matrix(head_dim)
        insert_online_hadamard_node(
            ir_model,
            target_value=anchor.k_input_value,
            consumer_nodes=anchor.k_consumers,
            H=h_mat,
            name_prefix=f"spinquant_block{block_idx}_k_R3",
        )


def _require_head_dim(ctx: SpinquantContext) -> int:
    """Return ``ctx.backbone_head_dim`` or raise if it could not be derived."""
    head_dim = ctx.backbone_head_dim
    if head_dim is None:
        raise ValueError(
            "R3 rotation: head_dim could not be derived from the backbone "
            "model. The export must expose a 'past_value' graph input whose "
            "last dimension is a static positive integer (HF/optimum LLM "
            "exports satisfy this by default)."
        )
    return head_dim


_ANCHOR_CACHE_KEY = "_r3_anchor_cache"


def _get_or_build_anchor_cache(ctx: SpinquantContext) -> List[BlockR3Anchors]:
    """Compute R3 anchors once per ctx and cache them on the ctx instance."""
    cached = getattr(ctx, _ANCHOR_CACHE_KEY, None)
    if cached is not None:
        return cached
    anchors = find_r3_anchors(ctx.backbone_topology, ctx.backbone_ir)
    object.__setattr__(ctx, _ANCHOR_CACHE_KEY, anchors)
    return anchors
