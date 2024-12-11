# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Process transformer models to get decoder list and LET pair modules for supporting models only """
from abc import ABC, abstractmethod
from transformers import LlamaModel, LlamaForCausalLM
from typing import List
import torch

from .defs import _LetPair

LlamaModelGroup = (LlamaModel, LlamaForCausalLM)

def get_transformer_processor(model):
    """ Return transformer_processor based on model class family. """
    if isinstance(model, LlamaModelGroup):
        return LlamaProcessor

    def _get_supporting_model_class():
        """ Helping function to pretty print supporting model classes. """
        model_class_str = ""
        for model_class in LlamaModelGroup:
            model_class_str += model_class.__name__
            model_class_str += ", "
        return model_class_str[:-2]

    raise ValueError(f"AIMET Omniquant only support class: {_get_supporting_model_class()} from transformer package,\
but got class {model.__class__}")

# pylint: disable=unnecessary-pass
class TransformerProcessor(ABC):
    """ Abstract class for transformer processors. """
    transformer_block_list_path = ""

    @classmethod
    @abstractmethod
    def get_decoder_list(cls, model):
        """ Method to get decoder module list. """
        pass

    @classmethod
    @abstractmethod
    def get_let_module_pair(cls, decoder_block):
        """ Method to get a list of let module pairs in a decoder_block. """
        pass

    @classmethod
    def init_let_params(cls, let_pair_list: List[_LetPair]):
        """ Register let params to LET pairs. """
        for _let_pair in let_pair_list:
            prev_modules, foll_modules = _let_pair.prev, _let_pair.follow
            prev_out_ch = prev_modules[0].weight.shape[0]
            prev_scale = torch.nn.Parameter(torch.ones(prev_out_ch))

            for _module in prev_modules:
                _module.register_let_params(prev_scale = prev_scale)

            for _module in foll_modules:
                _module.register_let_params(foll_scale = prev_scale)

class LlamaProcessor(TransformerProcessor):
    """
    Transformer Procesor for LlamaModelGroup = (LlamaModel, LlamaForCausalLM)
    LlamaModel has transformer_block_list_path = "layers"
    LlamaForCausalLM has transformer_block_list_path = "model.layers"
    """
    transformer_block_list_path = "layers" # Used for get_block_inputs

    @classmethod
    def get_decoder_list(cls, model)->torch.nn.ModuleList:
        """ Method to get decoder module list. """
        if isinstance(model, LlamaForCausalLM):
            model = getattr(model, "model", model)
            cls.transformer_block_list_path = "model.layers"

        transformer_block_list = model.get_submodule("layers")

        assert isinstance(transformer_block_list, torch.nn.ModuleList), \
            f"transformer_block_list: {transformer_block_list} is not a ModuleList"
        return transformer_block_list

    @classmethod
    def get_let_module_pair(cls, decoder_block) -> List:
        """ Method to get a list of let module pairs in a decoder_block. """
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
            _LetPair([up_proj], [down_proj])
            ]
