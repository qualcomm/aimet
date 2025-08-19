# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from dataclasses import dataclass
from typing import Type
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer


class BlockInterface:
    def __init__(self, block):
        self.block = block

    @property
    def q_proj(self):
        return self.block.self_attn.q_proj

    @q_proj.setter
    def q_proj(self, value):
        self.block.self_attn.q_proj = value

    @property
    def k_proj(self):
        return self.block.self_attn.k_proj

    @k_proj.setter
    def k_proj(self, value):
        self.block.self_attn.k_proj = value

    @property
    def v_proj(self):
        return self.block.self_attn.v_proj

    @v_proj.setter
    def v_proj(self, value):
        self.block.self_attn.v_proj = value

    @property
    def o_proj(self):
        return self.block.self_attn.o_proj

    @o_proj.setter
    def o_proj(self, value):
        self.block.self_attn.o_proj = value

    @property
    def up_proj(self):
        return self.block.mlp.up_proj

    @up_proj.setter
    def up_proj(self, value):
        self.block.mlp.up_proj = value

    @property
    def down_proj(self):
        return self.block.mlp.down_proj

    @down_proj.setter
    def down_proj(self, value):
        self.block.mlp.down_proj = value

    @property
    def gate_proj(self):
        return self.block.mlp.gate_proj

    @gate_proj.setter
    def gate_proj(self, value):
        self.block.mlp.gate_proj = value

    @property
    def input_norm(self):
        return self.block.input_layernorm

    @input_norm.setter
    def input_norm(self, value):
        self.block.input_layernorm = value

    @property
    def post_attention_norm(self):
        return self.block.post_attention_layernorm

    @post_attention_norm.setter
    def post_attention_norm(self, value):
        self.block.post_attention_layernorm = value


def _get_block_dtype(block: BlockInterface):
    return block.q_proj.weight.data.dtype


# Same as default, so don't need to do anything
class LlamaBlockInterface(BlockInterface):
    pass


# Same as default, so don't need to do anything
class Qwen2BlockInterface(BlockInterface):
    pass


@dataclass
class FPTQuantConfig:
    block_type: Type = None  # block types to use in a given model
    block_interface: Type = None  # interface class describing block layout


fptquant_model_config_dict = {
    LlamaModel: FPTQuantConfig(
        block_type=LlamaDecoderLayer, block_interface=LlamaBlockInterface
    ),
    Qwen2Model: FPTQuantConfig(
        block_type=Qwen2DecoderLayer, block_interface=Qwen2BlockInterface
    ),
}
