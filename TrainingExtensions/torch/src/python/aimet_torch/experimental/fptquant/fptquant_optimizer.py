# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import copy
import torch
from typing import Type
from types import NoneType
from tqdm import tqdm

from transformers import PretrainedConfig

from aimet_torch.v2.nn.true_quant import QuantizationMixin

from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix
from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin

from .fptquant_config import (
    fptquant_model_config_dict,
    FPTQuantConfig,
    BlockInterface,
    _get_block_dtype,
)
from .fptquant_transforms import (
    ScaledRotateTransformOp,
    ScalingTransformOp,
    GroupedHadamardTransformOp,
    MultiHeadValueTransformOp,
    RotationTransformOp,
)
from .fptquant_local_transform_optimizer import LocalTransformOptimizer


class FPTQuant:
    @staticmethod
    def insert_fpt_quant_transforms(model: torch.nn.Module, config: PretrainedConfig):
        if model.model.embed_tokens.weight is model.lm_head.weight:
            raise RuntimeError(
                "FPTQuant requires embed_tokens and lm_head weights to be untied. Ensure that "
                "model.config.tie_word_embeddings or a similar relevant setting is set to False for the model."
            )

        FPTQuant._fuse_norm_layer_into_linears(model.model.norm, [model.lm_head])

        joint_residual_transform = FPTQuant._get_residual_transform(config).to(
            device=model.device, dtype=torch.float32
        )
        model.model.embed_tokens = TransformationMixin.from_module(
            model.model.embed_tokens
        )
        model.model.embed_tokens.add_right_hand_transform(joint_residual_transform)

        model.lm_head = TransformationMixin.from_module(model.lm_head)
        model.lm_head.add_left_hand_transform(
            joint_residual_transform.get_inverted_op()
        )

        layers_to_optimize = [model.model.embed_tokens, model.lm_head]

        for block_interface in tqdm(
            FPTQuant._get_blocks(model), desc="Block transforms inserted"
        ):
            FPTQuant._fuse_norm_layer_into_linears(
                block_interface.input_norm,
                [
                    block_interface.q_proj,
                    block_interface.k_proj,
                    block_interface.v_proj,
                ],
            )
            FPTQuant._fuse_norm_layer_into_linears(
                block_interface.post_attention_norm,
                [block_interface.up_proj, block_interface.gate_proj],
            )

            block_interface.q_proj = TransformationMixin.from_module(
                block_interface.q_proj
            )
            block_interface.k_proj = TransformationMixin.from_module(
                block_interface.k_proj
            )
            block_interface.v_proj = TransformationMixin.from_module(
                block_interface.v_proj
            )
            block_interface.o_proj = TransformationMixin.from_module(
                block_interface.o_proj
            )
            block_interface.up_proj = TransformationMixin.from_module(
                block_interface.up_proj
            )
            block_interface.down_proj = TransformationMixin.from_module(
                block_interface.down_proj
            )
            block_interface.gate_proj = TransformationMixin.from_module(
                block_interface.gate_proj
            )

            FPTQuant._apply_prerope_transform(block_interface, config, model.device)
            FPTQuant._apply_value_transform(block_interface, config, model.device)
            FPTQuant._apply_up_down_scaling_transform(
                block_interface, config, model.device
            )
            FPTQuant._apply_residual_transform(
                block_interface, joint_residual_transform
            )
            FPTQuant._apply_down_projection_transform(
                block_interface, config, model.device
            )

            layers_to_optimize.extend(
                [
                    block_interface.q_proj,
                    block_interface.k_proj,
                    block_interface.v_proj,
                    block_interface.o_proj,
                    block_interface.up_proj,
                    block_interface.down_proj,
                    block_interface.gate_proj,
                ]
            )

        transform_optimizer = LocalTransformOptimizer(layers_to_optimize)
        transform_optimizer.optimize()

    @staticmethod
    def merge_fpt_quant_transforms(model: torch.nn.Module):
        model.model.embed_tokens = (
            FPTQuant._merge_transforms_and_recover_original_layer(
                model.model.embed_tokens
            )
        )
        model.lm_head = FPTQuant._merge_transforms_and_recover_original_layer(
            model.lm_head
        )

        for block_interface in FPTQuant._get_blocks(model):
            block_interface.q_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.q_proj
                )
            )
            block_interface.k_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.k_proj
                )
            )
            block_interface.v_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.v_proj
                )
            )
            block_interface.o_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.o_proj
                )
            )
            block_interface.up_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.up_proj
                )
            )
            block_interface.down_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.down_proj
                )
            )
            block_interface.gate_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.gate_proj
                )
            )

    @staticmethod
    def insert_nonmergeable_down_project_transform(
        model: torch.nn.Module, config: PretrainedConfig
    ):
        for block_interface in tqdm(
            FPTQuant._get_blocks(model), desc="Block transforms inserted"
        ):
            block_interface.down_proj = TransformationMixin.from_module(
                block_interface.down_proj
            )
            transform = GroupedHadamardTransformOp(config.intermediate_size)
            block_interface.down_proj.add_left_hand_transform(transform)

    @staticmethod
    def _merge_transforms_and_recover_original_layer(layer: torch.nn.Module):
        if not isinstance(layer, TransformationMixin):
            return layer  # Do nothing if it is not a transformed layer

        layer.merge()
        if len(layer.right_hand_transforms) == len(layer.left_hand_transforms) == 0:
            return TransformationMixin.get_original_module(layer)
        return layer

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        for module in model.modules():
            for target in fptquant_model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(model: torch.nn.Module) -> list[BlockInterface]:
        target_type = FPTQuant._screen_for_target_type(model)
        config = fptquant_model_config_dict.get(target_type, FPTQuantConfig())
        target_modules = []
        if config.block_type is not None:
            target_modules = [
                config.block_interface(m)
                for m in model.modules()
                if isinstance(m, config.block_type)
            ]
        return target_modules

    @staticmethod
    def _fuse_norm_layer_into_linears(
        norm: torch.nn.Module, linears: list[torch.nn.Linear]
    ):
        """Helper function to merge RMS Norm weights into linear layer"""
        for linear in linears:
            W = linear.weight.data
            dtype = linear.weight.dtype
            linear.weight.data = (W.double() * norm.weight.data.double()).to(
                dtype=dtype
            )
            if hasattr(norm, "bias") and linear.bias is not None:
                linear.bias.data = (
                    linear.bias.data.double() + (W.double() @ norm.bias.data.double())
                ).to(dtype=dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)

    @staticmethod
    def _apply_prerope_transform(
        block: BlockInterface, config: PretrainedConfig, device: torch.device
    ):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        transform = ScaledRotateTransformOp(
            head_dim, config.num_attention_heads, config.num_key_value_heads
        ).to(device=device, dtype=_get_block_dtype(block))

        block.q_proj.add_right_hand_transform(transform.get_inverted_op())
        block.k_proj.add_right_hand_transform(transform)

    @staticmethod
    def _apply_value_transform(
        block: BlockInterface, config: PretrainedConfig, device: torch.device
    ):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        transform = MultiHeadValueTransformOp(
            head_dim, config.num_attention_heads, config.num_key_value_heads
        ).to(device=device, dtype=torch.float32)

        block.v_proj.add_right_hand_transform(transform)
        block.o_proj.add_left_hand_transform(transform.get_inverted_op())

    @staticmethod
    def _apply_up_down_scaling_transform(
        block: BlockInterface, config: PretrainedConfig, device: torch.device
    ):
        transform = ScalingTransformOp(config.intermediate_size).to(
            device=device, dtype=_get_block_dtype(block)
        )

        block.up_proj.add_right_hand_transform(transform)
        block.down_proj.add_left_hand_transform(transform.get_inverted_op())

    @staticmethod
    def _get_residual_transform(config: PretrainedConfig) -> RotationTransformOp:
        return RotationTransformOp(matrix=get_hadamard_matrix(config.hidden_size))

    @staticmethod
    def _apply_residual_transform(
        block: BlockInterface, transform: RotationTransformOp
    ):
        block.q_proj.add_left_hand_transform(transform.get_inverted_op())
        block.k_proj.add_left_hand_transform(transform.get_inverted_op())
        block.v_proj.add_left_hand_transform(transform.get_inverted_op())
        block.o_proj.add_right_hand_transform(transform)

        block.gate_proj.add_left_hand_transform(transform.get_inverted_op())
        block.up_proj.add_left_hand_transform(transform.get_inverted_op())
        block.down_proj.add_right_hand_transform(transform)

    @staticmethod
    def _apply_down_projection_transform(
        block: BlockInterface, config: PretrainedConfig, device: torch.device
    ):
        transform = GroupedHadamardTransformOp(config.intermediate_size).to(
            device=device, dtype=_get_block_dtype(block)
        )
        block.down_proj.add_left_hand_transform(transform.get_inverted_op())
        block.down_proj.add_left_hand_transform(transform)

        # todo: make this more general
        if isinstance(block.down_proj.left_hand_transforms[0], QuantizationMixin):
            # if the transform is being inserted into a Quantized model, need to make sure that quantizer settings
            # are updated appropriately
            block.down_proj.left_hand_transforms[0].output_quantizers.append(
                copy.deepcopy(block.down_proj.output_quantizers[0])
            )
