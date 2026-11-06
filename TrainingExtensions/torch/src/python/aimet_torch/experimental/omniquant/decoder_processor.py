# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Process transformer models to get decoder list and LET pair modules for supporting models only"""

from transformers import LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)
from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralDecoderLayer,
    MistralForCausalLM,
)

from aimet_torch import QuantizationSimModel
from typing import List
import torch

from .defs import _LetPair

LlamaModelGroup = (LlamaModel, LlamaForCausalLM)
QwenModelGroup = (Qwen2Model, Qwen2ForCausalLM)
MistralModelGroup = (MistralModel, MistralForCausalLM)

model_to_block_mapping = {
    LlamaModel: LlamaDecoderLayer,
    Qwen2Model: Qwen2DecoderLayer,
    MistralModel: MistralDecoderLayer,
}


def get_transformer_processor(qsim: QuantizationSimModel):
    """Return transformer_processor based on model class family."""
    for module in qsim.model.modules():
        if isinstance(module, (LlamaModelGroup, QwenModelGroup, MistralModelGroup)):
            return TransformerProcessor(qsim.model)

    def _get_supporting_model_class():
        """Helping function to pretty print supporting model classes."""
        model_class_str = ""
        for model_class in LlamaModelGroup:
            model_class_str += model_class.__name__
            model_class_str += ", "
        return model_class_str[:-2]

    raise ValueError(
        f"AIMET Omniquant only support class: {_get_supporting_model_class()} from transformer package,\
but got class {qsim.model.__class__}"
    )


class TransformerProcessor:
    def __init__(self, model):
        self._screen_for_target_type(model)

    def _screen_for_target_type(self, model):
        for module in model.modules():
            for target in model_to_block_mapping:
                if isinstance(module, target):
                    self.target_type = target
                    return

        assert False, "No targets found in provided model"

    def get_decoder_list(self, model):
        """helper to get all the blocks in the model represented by model_to_block_mapping"""
        target_type = model_to_block_mapping.get(self.target_type)
        target_modules = []
        if target_type is not None:
            target_modules = [
                m for m in model.model.modules() if isinstance(m, target_type)
            ]
        return target_modules

    def get_let_module_pair(self, decoder_block) -> List:
        """Method to get a list of let module pairs in a decoder_block."""
        if isinstance(
            decoder_block, (LlamaDecoderLayer, Qwen2DecoderLayer, MistralDecoderLayer)
        ):
            input_layernorm = decoder_block.get_submodule("input_layernorm")
            q_proj = decoder_block.get_submodule("self_attn.q_proj")
            k_proj = decoder_block.get_submodule("self_attn.k_proj")
            v_proj = decoder_block.get_submodule("self_attn.v_proj")
            o_proj = decoder_block.get_submodule("self_attn.o_proj")
            gate_proj = decoder_block.get_submodule("mlp.gate_proj")
            up_proj = decoder_block.get_submodule("mlp.up_proj")
            down_proj = decoder_block.get_submodule("mlp.down_proj")
            output_layernorm = decoder_block.get_submodule("post_attention_layernorm")
            return [
                _LetPair([input_layernorm], [q_proj, k_proj, v_proj]),
                _LetPair([v_proj], [o_proj]),
                _LetPair([output_layernorm], [gate_proj, up_proj]),
                _LetPair([up_proj], [down_proj]),
            ]

    @classmethod
    def init_let_params(
        cls, let_pair_list: List[_LetPair], num_repeats, num_key_value_heads=1
    ):
        """Register let params to LET pairs."""
        for _let_pair in let_pair_list:
            prev_modules, foll_modules = _let_pair.prev, _let_pair.follow
            prev_out_ch = prev_modules[0].weight.shape[0]
            prev_scale = torch.nn.Parameter(torch.ones(prev_out_ch))

            # Currently only one module is expected in prev_list
            assert len(prev_modules) == 1
            prev_modules[0].param_quantizers["weight"].register_let_params(
                prev_scale=prev_scale
            )
            if "bias" in prev_modules[0].param_quantizers:
                prev_modules[0].param_quantizers["bias"].register_let_params(
                    prev_scale=prev_scale
                )

            for module in foll_modules:
                foll_in_ch = module.weight.shape[1]

                # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
                # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
                # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
                if prev_out_ch != foll_in_ch:
                    assert foll_in_ch // prev_out_ch == num_repeats
                    nr = num_repeats
                else:
                    nr = 1
                module.param_quantizers["weight"].register_let_params(
                    foll_scale=prev_scale,
                    num_repeats=nr,
                    num_key_value_heads=num_key_value_heads if nr > 1 else 1,
                )
