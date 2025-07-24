# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Optimizer for Spinquant"""

from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

RMSNORM_LINEAR_PAIRS = "RMSNORM_LINEAR_PAIRS"
R1_FUSION_PAIRS = "R1_FUSION_PAIRS"


def _default_r1_fusion_func(llm_model):
    """Default R1 fusion function"""
    r1_direction_pairs = []
    for layer in llm_model.model.layers:
        r1_direction_pairs.extend(
            [
                (layer.self_attn.q_proj, True),
                (layer.self_attn.k_proj, True),
                (layer.self_attn.v_proj, True),
                (layer.self_attn.o_proj, False),
                (layer.mlp.gate_proj, True),
                (layer.mlp.up_proj, True),
                (layer.mlp.down_proj, False),
            ]
        )
    r1_direction_pairs.extend(
        [(llm_model.model.embed_tokens, False), (llm_model.lm_head, True)]
    )
    return r1_direction_pairs


def _default_rmsnorm_linear_pairs_func(llm_model):
    """Default RMSNorm Linear pairs function"""
    rmsnorm_linear_pairs = []
    for layer in llm_model.model.layers:
        rmsnorm_linear_pairs.extend(
            [
                (
                    layer.input_layernorm,
                    [
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                    ],
                )
            ]
        )
        rmsnorm_linear_pairs.extend(
            [
                (
                    layer.post_attention_layernorm,
                    [
                        layer.mlp.gate_proj,
                        layer.mlp.up_proj,
                    ],
                )
            ]
        )
    rmsnorm_linear_pairs.extend([(llm_model.model.norm, [llm_model.lm_head])])
    return rmsnorm_linear_pairs


# Dictionary of supported modules and associated information for RMSNORM Linear fusion pairs as well as R1 fusion pairs.
SUPPORTED_MODULE_DICT = {
    LlamaForCausalLM: {
        RMSNORM_LINEAR_PAIRS: _default_rmsnorm_linear_pairs_func,
        R1_FUSION_PAIRS: _default_r1_fusion_func,
    },
    Qwen2ForCausalLM: {
        RMSNORM_LINEAR_PAIRS: _default_rmsnorm_linear_pairs_func,
        R1_FUSION_PAIRS: _default_r1_fusion_func,
    },
    MistralForCausalLM: {
        RMSNORM_LINEAR_PAIRS: _default_rmsnorm_linear_pairs_func,
        R1_FUSION_PAIRS: _default_r1_fusion_func,
    },
}


def apply_spinquant(model: torch.nn.Module):
    """
    Apply SpinQuant to the model, modifying weights in place. https://arxiv.org/pdf/2405.16406
    Currently only R1 rotations without optimization are supported.
    The model is updated in place.

    :param model: The model to apply SpinQuant to.
    """
    # Ensure that any user registered module types are fully registered
    for supported_module_type, module_info in SUPPORTED_MODULE_DICT.items():
        if module_info.get(RMSNORM_LINEAR_PAIRS) is None:
            raise RuntimeError(
                f"{RMSNORM_LINEAR_PAIRS} info missing for module type {supported_module_type.__name__}"
            )
        if module_info.get(R1_FUSION_PAIRS) is None:
            raise RuntimeError(
                f"{R1_FUSION_PAIRS} info missing for module type {supported_module_type.__name__}"
            )

    found_module = False
    for module in model.modules():
        if isinstance(module, tuple(SUPPORTED_MODULE_DICT.keys())):
            if module.model.embed_tokens.weight is module.lm_head.weight:
                raise RuntimeError(
                    "SpinQuant requires embed_tokens and lm_head weights to be untied. Ensure that model.config.tie_word_embeddings or a similar relevant "
                    "setting is set to False for the model."
                )

            found_module = True
            _identify_and_fuse_rmsnorms_into_linears(module)

            _fuse_r1_rotations(module)

    if not found_module:
        _logger.warning(
            "SpinQuant optimizer did not find any modules to apply SpinQuant for in the model."
        )


def _identify_and_fuse_rmsnorms_into_linears(llm_model: torch.nn.Module):
    for rmsnorm, linears in SUPPORTED_MODULE_DICT[type(llm_model)][
        RMSNORM_LINEAR_PAIRS
    ](llm_model):
        _fuse_rmsnorm_into_linear(rmsnorm, linears)


def _fuse_rmsnorm_into_linear(rmsnorm, linear_layers):
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        W = linear.weight.data
        linear.weight.data = (W * rmsnorm.weight).to(linear_dtype)

        if hasattr(rmsnorm, "bias"):
            if linear.bias is not None:
                linear.bias.data = linear.bias.data.double() + torch.matmul(
                    W, rmsnorm.bias.double()
                )
                linear.bias.data = linear.bias.data.to(linear_dtype)

    rmsnorm.weight.data = torch.ones_like(rmsnorm.weight.data)


def _fuse_r1_rotations(llm_model: torch.nn.Module):
    modules_and_fuse_directions = SUPPORTED_MODULE_DICT[type(llm_model)][
        R1_FUSION_PAIRS
    ](llm_model)
    if modules_and_fuse_directions:
        hidden_size = modules_and_fuse_directions[0][0].weight.shape[1]
        had_matrix = get_hadamard_matrix(hidden_size).to(
            modules_and_fuse_directions[0][0].weight.device
        ) / torch.sqrt(torch.tensor(hidden_size))

        for module, fuse_before in modules_and_fuse_directions:
            _fuse_r1_rotation(module, fuse_before, had_matrix)


def _fuse_r1_rotation(module, fuse_before, had_matrix):
    with torch.no_grad():
        if isinstance(module, torch.nn.Linear):
            if fuse_before:
                module.weight.copy_(module.weight @ had_matrix.T)
            else:
                module.weight.copy_((module.weight.T @ had_matrix.T).T)
                if module.bias is not None:
                    module.bias.copy_((module.bias.T @ had_matrix.T).T)
        elif isinstance(module, torch.nn.Embedding):
            if not fuse_before:
                module.weight.copy_(module.weight @ had_matrix.T)
            else:
                raise RuntimeError("Embedding module is expected to fuse after only")
