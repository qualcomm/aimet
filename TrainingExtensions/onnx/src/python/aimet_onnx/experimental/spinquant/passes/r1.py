# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Hadamard rotation pass.

R1 = H / sqrt(hidden_size) rotates the backbone residual stream
(``h_rot = h @ R1``). Each weight that reads from or writes to the residual
stream absorbs the rotation so the model output is unchanged.

R1 also rotates:

* the external embedding tensor (for VLM backbones exported with
  ``use_inputs_embeds=True``);
* the PatchMerger ``linear_fc2`` in the visual encoder (R_L), since its outputs
  must land in the rotated backbone residual stream space.

Reading vs writing layers:

* reading layer: takes input from the residual stream
  (qkv, gate_up, lm_head)
* writing layer: adds output back to the residual stream
  (o_proj, down_proj, embed_tokens)
"""

from typing import List

import numpy as np
import torch
from onnx import numpy_helper

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils

from aimet_onnx.experimental.llm_topology.topology import LlmTopology
from aimet_onnx.experimental.llm_topology.weight_utils import get_weight_product
from aimet_onnx.experimental.spinquant.model_analysis import (
    find_post_writing_norms,
)
from aimet_onnx.experimental.spinquant.passes.base import (
    RotationPass,
    SpinquantContext,
)
from aimet_onnx.experimental.spinquant.transforms import (
    fuse_norm_layers_into_linears,
    hadamard_rotation_matrix,
    insert_online_hadamard_node,
    rotate_gather_weight,
    rotate_linear_weight,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


class R1RotationPass(RotationPass):
    """R1 residual-stream Hadamard rotation."""

    @property
    def name(self) -> str:
        return "R1"

    def validate(self, ctx: SpinquantContext) -> None:
        """Verify R1 architectural pre-conditions and that all weights exist with the right shape."""
        _validate_backbone_weights(
            ctx.backbone_model,
            ctx.backbone_topology,
            ctx.backbone_hidden_size,
        )
        if ctx.visual_model is not None:
            assert ctx.visual_merger_linear2 is not None
            _validate_merger_linear2(
                ctx.visual_model,
                ctx.visual_merger_linear2,
                ctx.backbone_hidden_size,
            )

    def apply(self, ctx: SpinquantContext) -> None:
        """Fuse norms, rotate backbone, optionally rotate external embedding and merger_linear2."""
        R1 = hadamard_rotation_matrix(ctx.backbone_hidden_size)

        # R1 absorbs RMSNorm scale into the downstream linears it reads from,
        # so norm fusion is part of R1 (not of every rotation).
        fuse_norm_layers_into_linears(ctx.backbone_model, ctx.backbone_active_norms)

        _logger.info(
            "Backbone: Applying R1 Hadamard rotation with backbone_hidden_size=%d.",
            ctx.backbone_hidden_size,
        )

        # Note: If external and internal embeddings are missing, add online rotation
        rotate_embeddings_online = not (
            ctx.backbone_topology.embed_tokens or ctx.embedding is not None
        )

        _rotate_backbone(
            ctx.backbone_model,
            ctx.backbone_topology,
            R1,
            rotate_embeddings_online=rotate_embeddings_online,
        )

        if ctx.embedding is not None:
            _rotate_external_embedding(ctx.embedding, R1)

        # If online embedding rotation is already present, don't rotate visual model
        if ctx.visual_model is not None and not rotate_embeddings_online:
            assert ctx.visual_merger_linear2 is not None
            _logger.info(
                "Visual: Applying R1 Hadamard rotation to merger_linear2 with backbone_hidden_size=%d.",
                ctx.backbone_hidden_size,
            )
            _rotate_merger_linear2(ctx.visual_model, ctx.visual_merger_linear2, R1)


def _rotate_backbone(
    model: ModelProto,
    role_map: LlmTopology,
    R1: np.ndarray,
    *,
    rotate_embeddings_online: bool = False,
) -> None:
    """Rotate every weight in ``role_map`` with R1 in-place."""
    if rotate_embeddings_online:
        _insert_embedding_hadamard_rotation(model, role_map, R1)
    else:
        for op in role_map.embed_tokens:
            rotate_gather_weight(model, op, R1)

    if role_map.lm_head:
        for op in role_map.lm_head:
            rotate_linear_weight(model, op, R1, is_writing=False)
    else:
        # Headless backbone: no lm_head to absorb R1's inverse.
        _insert_final_hadamard_rotation(model, role_map, R1)

    for block_idx, block in enumerate(role_map.blocks):
        _logger.debug("Applying R1 to block %d.", block_idx)
        for op in block.qkv.ops:
            rotate_linear_weight(model, op, R1, is_writing=False)
        for op in block.o_proj:
            rotate_linear_weight(model, op, R1, is_writing=True)
        for op in block.gate_up.ops:
            rotate_linear_weight(model, op, R1, is_writing=False)
        for op in block.down_proj:
            rotate_linear_weight(model, op, R1, is_writing=True)


def _insert_embedding_hadamard_rotation(
    model: ModelProto, role_map: LlmTopology, R1: np.ndarray
) -> None:
    """Place an online Hadamard onto the input embeddings."""
    embedding_tensor = _find_embedding_tensor(role_map)

    consumer_nodes = [op.get_module() for op in embedding_tensor.consumers]
    if not consumer_nodes:
        raise RuntimeError(
            f"Embedding tensor '{embedding_tensor}' has no consumers to rewire; "
            f"cannot rotate the residual stream."
        )
    _logger.info(
        "Backbone: Placing online R1 Hadamard rotation at tensor '%s'.",
        embedding_tensor,
    )
    insert_online_hadamard_node(
        model,
        target_tensor_name=embedding_tensor.name,
        consumer_nodes=consumer_nodes,
        H=R1,
        name_prefix=f"spinquant_{embedding_tensor.name}_R1",
    )


def _find_embedding_tensor(role_map: LlmTopology) -> str:
    """
    Return the input-embedding tensor, skipping the input norm's leading Cast.

    TODO: Move this logic into LLM topology
    """
    embedding_tensor = role_map.blocks[0].residual_input
    if embedding_tensor is None:
        raise RuntimeError("Failed to find embedding tensor")

    # Propagate through data movement ops and casts
    while embedding_tensor.producer is not None:
        if not (
            _is_grid_preserving_op(embedding_tensor.producer.type)
            or embedding_tensor.producer.type == "Cast"
        ):
            break
        embedding_tensor = embedding_tensor.producer.inputs[0]
    return embedding_tensor


def _insert_final_hadamard_rotation(
    model: ModelProto, role_map: LlmTopology, R1: np.ndarray
) -> None:
    """Splice an online Hadamard onto the last residual add to un-rotate the stream.

    Used for headless backbones with no lm_head to absorb R1's inverse. All
    consumers of the residual add are rewired through the inserted MatMul.
    """
    residual_product = role_map.blocks[-1].residual_output
    consumer_nodes = [op.get_module() for op in residual_product.consumers]
    if not consumer_nodes:
        raise RuntimeError(
            f"Residual tensor '{residual_product.name}' has no consumers to rewire; "
            f"cannot un-rotate the residual stream."
        )
    insert_online_hadamard_node(
        model,
        target_tensor_name=residual_product.name,
        consumer_nodes=consumer_nodes,
        H=R1.T,
        name_prefix=f"spinquant_{residual_product.name}_R1",
    )


def _rotate_external_embedding(embedding: torch.Tensor, R1: np.ndarray) -> None:
    """Rotate an external embedding tensor in-place: ``W <- W @ R1``."""
    original_device = embedding.device
    W = embedding.detach().cpu().numpy()
    W_rot = (W @ R1).astype(W.dtype)
    embedding.data.copy_(torch.from_numpy(W_rot).to(original_device))
    _logger.info(
        "Backbone: Rotated external embedding weight with R_L (shape %s).", W.shape
    )


def _rotate_merger_linear2(
    model: ModelProto, merger_linear2: List[Op], R_L: np.ndarray
) -> None:
    """Rotate PatchMerger ``linear_fc2`` weights with R_L in-place."""
    for op in merger_linear2:
        rotate_linear_weight(model, op, R_L, is_writing=True)


def _validate_backbone_weights(
    model: ModelProto, role_map: LlmTopology, hidden_size: int
) -> None:
    """Verify R1 architectural compatibility and that every weight in ``role_map``
    exists with the correct shape.

    R1 absorption requires writing layers (o_proj, down_proj) to feed directly
    into the residual add. An affine RMSNorm between the writing layer and the
    residual add breaks that property.
    """
    post_writing_norms = find_post_writing_norms(model, role_map)
    if post_writing_norms:
        raise ValueError(
            f"R1 rotation absorption requires writing layers (o_proj, down_proj) to feed "
            f"directly into the residual add. Detected {len(post_writing_norms)} affine "
            f"RMSNorm(s) between writing layers and the residual add "
            f"- R1 absorption is not feasible for this architecture."
        )

    for op in role_map.embed_tokens:
        found = False
        for inp in op.inputs:
            if inp.is_parm or inp.is_const:
                found = True
                tensor = ParamUtils.get_param_by_name(model, inp.name)
                if tensor is None:
                    raise RuntimeError(
                        f"embed_tokens op '{op.name}': weight '{inp.name}' not found in "
                        f"initializers. Cannot apply R1 rotation."
                    )
                shape = numpy_helper.to_array(tensor).shape
                if shape[-1] != hidden_size:
                    raise RuntimeError(
                        f"embed_tokens op '{op.name}': weight '{inp.name}' has shape {shape}, "
                        f"but shape[-1]={shape[-1]} != hidden_size={hidden_size}."
                    )
        if not found:
            raise RuntimeError(
                f"embed_tokens op '{op.name}': no static weight input found. "
                f"Can't apply R1 rotation."
            )

    linear_ops_with_role = [(op, False) for op in role_map.lm_head]
    for block in role_map.blocks:
        linear_ops_with_role += [(op, False) for op in block.qkv.ops]
        linear_ops_with_role += [(op, True) for op in block.o_proj]
        linear_ops_with_role += [(op, False) for op in block.gate_up.ops]
        linear_ops_with_role += [(op, True) for op in block.down_proj]

    for op, is_writing in linear_ops_with_role:
        weight_inp, is_transposed = get_weight_product(op)
        if weight_inp is None:
            raise RuntimeError(
                f"Op '{op.name}': no static weight found. Can't apply R1 rotation."
            )
        tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
        if tensor is None:
            raise RuntimeError(
                f"Op '{op.name}': weight '{weight_inp.name}' not found in initilizers. "
                f"Can't apply R1 rotation."
            )
        shape = numpy_helper.to_array(tensor).shape

        if op.type == "Conv" or is_transposed:  # [out, in]
            rotated_axis = 0 if is_writing else 1
        else:  # [in, out]
            rotated_axis = -1 if is_writing else 0
        if shape[rotated_axis] != hidden_size:
            raise RuntimeError(
                f"Op '{op.name}': weight shape {shape}, axis {rotated_axis} "
                f"= {shape[rotated_axis]}, expected hidden_size={hidden_size}."
            )


def _validate_merger_linear2(
    model: ModelProto, merger_linear2: List[Op], backbone_hidden_size: int
) -> None:
    """Verify that every merger_linear2 weight exists and writes into ``backbone_hidden_size``."""
    for op in merger_linear2:
        weight_inp, is_transposed = get_weight_product(op)
        if weight_inp is None:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': no static weight found. Cannot apply R_L rotation."
            )
        tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
        if tensor is None:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': weight '{weight_inp.name}' not found in initializers."
            )
        shape = numpy_helper.to_array(tensor).shape
        # writing layer: output axis = backbone_hidden_size
        # Conv/transposed [out, in]: axis 0; MatMul [in, out]: axis -1
        rotated_axis = 0 if (op.type == "Conv" or is_transposed) else -1
        if shape[rotated_axis] != backbone_hidden_size:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': weight shape {shape}, axis {rotated_axis} "
                f"= {shape[rotated_axis]}, expected backbone_hidden_size={backbone_hidden_size}."
            )
