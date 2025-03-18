# /usr/bin/env python
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
""" Test Omniquant functions. (Not include LET modules.) """
import contextlib
import torch
import pytest

from aimet_torch.omniquant import decoder_processor

@contextlib.contextmanager
def add_custom_model_class_to_support_model_group(model_class, target_group_name):
    """
    Add model_class Class to decoder_processor.target_group_name.
    """
    target_group = getattr(decoder_processor, target_group_name)
    new_target_group = list(target_group)
    new_target_group.append(model_class)
    setattr(decoder_processor, target_group_name, tuple(new_target_group))

    yield

    setattr(decoder_processor, target_group_name, target_group)

class FakeLlamaModel(torch.nn.Module):
    def __init__(self, layer_num, seq_len, head_num, emb_dim):
        super().__init__()
        assert emb_dim % head_num == 0, "emb_dim need to be dividable by head_num."
        self.layers = torch.nn.ModuleList([FakeDecoderBlcok(seq_len, head_num, emb_dim) for _ in range(layer_num)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FakeDecoderBlcok(torch.nn.Module):
    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(emb_dim)
        self.self_attn = FakeSelfAttn(seq_len, head_num, emb_dim)
        self.mlp = FakeMlp(emb_dim)
        self.post_attention_layernorm = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.mlp(x)
        x = self.post_attention_layernorm(x)
        return x

class FakeSelfAttn(torch.nn.Module):
    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.head_num = head_num
        self.emb_dim = emb_dim
        self.q_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.k_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.v_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.o_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        head_dim = self.emb_dim//self.head_num
        q = self.q_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 1, 3)
        qk = torch.matmul(q, k)
        qkv = torch.matmul(qk, v).permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.head_num*head_dim)
        out = self.o_proj(qkv)

        return out

class FakeMlp(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.up_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.down_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        y0 = self.gate_proj(x)
        y1 = self.up_proj(x)
        y1 = self.down_proj(y1)
        return y0 + y1

class TestOmniquant:
    def test_get_transformer_processor(self):
        """ Test get_transformer_processor returns correct TransformerProcessor and raise error. """

        layer_num = 5
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        fake_llama_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        with torch.no_grad():
            # Make sure model is runnable.
            fake_llama_model(dummy_input)

        with add_custom_model_class_to_support_model_group(FakeLlamaModel, "LlamaModelGroup"):
            llama_processor = decoder_processor.get_transformer_processor(fake_llama_model)
            assert llama_processor.__name__ == "LlamaProcessor"

            decoder_list = llama_processor.get_decoder_list(fake_llama_model)
            assert len(decoder_list) == layer_num

            for _decoder_block in decoder_list:
                let_module_pair = llama_processor.get_let_module_pair(_decoder_block)
                assert len(let_module_pair) == 4 # Llama Model Group should have 4 let pairs.

        with pytest.raises(ValueError):
            decoder_processor.get_transformer_processor(fake_llama_model)
