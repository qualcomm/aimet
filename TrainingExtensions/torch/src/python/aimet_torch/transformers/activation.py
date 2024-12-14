# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  From PyTorch:
#
#  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
#  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
#  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
#  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
#  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
#  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
#  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
#  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
#  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
#  From Caffe2:
#
#  Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
#  All contributions by Facebook:
#  Copyright (c) 2016 Facebook Inc.
#
#  All contributions by Google:
#  Copyright (c) 2015 Google Inc.
#  All rights reserved.
#
#  All contributions by Yangqing Jia:
#  Copyright (c) 2015 Yangqing Jia
#  All rights reserved.
#
#  All contributions by Kakao Brain:
#  Copyright 2019-2020 Kakao Brain
#
#  All contributions by Cruise LLC:
#  Copyright (c) 2022 Cruise LLC.
#  All rights reserved.
#
#  All contributions from Caffe:
#  Copyright(c) 2013, 2014, 2015, the respective contributors
#  All rights reserved.
#
#  All other contributions:
#  Copyright(c) 2015, 2016 the respective contributors
#  All rights reserved.
#
#  Caffe2 uses a copyright model similar to Caffe: each contributor holds
#  copyright over their contributions to Caffe2. The project versioning records
#  all such contribution and copyright details. If a contributor wants to further
#  mark their specific copyright on a particular contribution, they should
#  indicate their copyright solely in the commit message of the change when it is
#  committed.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#     and IDIAP Research Institute nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" This file contains a modified version of PyTorch's quantizable MultiHeadAttn unit """
# --------------------------------------------------------------------------------------------------------
# Reference : https://github.com/pytorch/pytorch/blob/master/torch/nn/quantizable/modules/activation.py#L11
# Above PyTorch code is used as base implementation of this MHA unit, along with addition updates listed below :
# 1) Exclude quant/dequant operations
# 2) Update functionals to torch modules (softmax, add, matmul)
# 3) Add an explicit mask_add op
# 4) updates bmm op to matmuls
# ----------------------------------------------------------------------------------------------------------

# pylint check enabled

from typing import Optional, Tuple, Union
import warnings
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as nnF
import aimet_torch._base.nn.modules.custom as aimet_modules

# pylint: disable=too-many-arguments
class QuantizableMultiheadAttention(nn.MultiheadAttention):
    """ quantizable defn of MHA """
    _FLOAT_MODULE = nn.MultiheadAttention

    r"""Quantizable implementation of the MultiheadAttention.

    Note::
        Please, refer to :class:`~torch.nn.MultiheadAttention` for more
        information

    Allows the model to jointly attend to information from different
    representation subspaces.
    See reference: Attention Is All You Need

    The original MHA module is not quantizable.
    This reimplements it by explicitly instantiating the linear layers.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> import torch.nn.quantizable as nnqa
        >>> multihead_attn = nnqa.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    Note::
        Please, follow the quantization flow to convert the quantizable MHA.
    """
    __constants__ = ['batch_first']
    # pylint: disable=too-many-arguments
    # pylint: disable=arguments-differ
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0., bias: bool = True,
                 add_bias_kv: bool = False, add_zero_attn: bool = False,
                 kdim: int = None, vdim: int = None, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, num_heads, dropout,
                                                            bias, add_bias_kv,
                                                            add_zero_attn, kdim, vdim, batch_first, **factory_kwargs)
        self.linear_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.linear_K = nn.Linear(self.kdim, self.embed_dim, bias=bias, **factory_kwargs)
        self.linear_V = nn.Linear(self.vdim, self.embed_dim, bias=bias, **factory_kwargs)
        # for the type: ignore, see https://github.com/pytorch/pytorch/issues/58969
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)  # type: ignore[assignment]

        self.div = aimet_modules.Divide()
        self.matmul_1 = aimet_modules.MatMul()
        self.matmul_2 = aimet_modules.MatMul()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.mask_add = aimet_modules.Add()

    def _get_name(self):
        return 'QuantizableMultiheadAttention'

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Note::
        Please, refer to :func:`~torch.nn.MultiheadAttention.forward` for more
        information

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
          heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
          effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: If ``average_attn_weights=True``, returns attention weights averaged
          across heads of shape :math:`(N, L, S)`, where N is the batch size, L is the target sequence length,
          S is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(N, num_heads, L, S)`.
        """
        return self._forward_impl(query, key, value, key_padding_mask,
                                  need_weights, attn_mask, average_attn_weights)
    # pylint: disable = too-many-locals
    # pylint: disable = too-many-branches
    # pylint: disable = too-many-statements
    def _forward_impl(self,
                      query: Tensor,
                      key: Tensor,
                      value: Tensor,
                      key_padding_mask: Optional[Tensor] = None,
                      need_weights: bool = True,
                      attn_mask: Optional[Tensor] = None,
                      average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        # This version will not deal with the static key/value pairs.
        # Keeping it here for future changes.
        #
        # TODO: This method has some duplicate lines with the
        # `torch.nn.functional.multi_head_attention`. Will need to refactor.
        static_k = None
        static_v = None

        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        tgt_len, bsz, embed_dim_to_check = query.size()
        assert self.embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        k = self.linear_K(key)
        v = self.linear_V(value)
        scaling = float(head_dim) ** 0.5
        q = self.div(self.linear_Q(query), scaling)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
        if self.bias_k is not None and self.bias_v is not None:
            if static_k is None and static_v is None:

                # Explicitly assert that bias_k and bias_v are not None
                # in a way that TorchScript can understand.
                bias_k = self.bias_k
                assert bias_k is not None
                bias_v = self.bias_v
                assert bias_v is not None

                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = nnF.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = nnF.pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * self.num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * self.num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k_zeros = torch.zeros((k.size(0), 1) + k.size()[2:])
            k = torch.cat([k, k_zeros], dim=1)
            v_zeros = torch.zeros((v.size(0), 1) + k.size()[2:])
            v = torch.cat([v, v_zeros], dim=1)

            if attn_mask is not None:
                attn_mask = nnF.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = nnF.pad(key_padding_mask, (0, 1))

        attn_output_weights = self.matmul_1(q, k.transpose(1, 2))

        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                self.mask_add(attn_output_weights, attn_mask)

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = nnF.dropout(attn_output_weights, p=self.dropout, training=self.training)

        # attn_output = torch.bmm(attn_output_weights, v)
        attn_output = self.matmul_2(attn_output_weights, v)

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        if self.batch_first:
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        # for the type: ignore[has-type], see https://github.com/pytorch/pytorch/issues/58969
        attn_output = self.out_proj(attn_output)  # type: ignore[has-type]

        # pylint: disable = no-else-return
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


def create_quantizable_multihead_attention(module: torch.nn.MultiheadAttention) -> QuantizableMultiheadAttention:
    """
    Create QuantizableMultiheadAttention using existing torch.nn.MultiheadAttention module
    :param module: Existing torch.nn.MultiheadAttention module
    :return: Newly created QuantizableMultiheadAttention module
    """
    # inspect MHA if bias is required.
    bias = module.in_proj_bias is not None

    # if bias k/v parameter exist set quantizable MHA to create 3 separate bias tensors as expected.
    add_bias_kv = module.bias_k is not None and module.bias_v is not None

    q_MHA = QuantizableMultiheadAttention(embed_dim=module.embed_dim, num_heads=module.num_heads,
                                          dropout=module.dropout, bias=bias, add_bias_kv=add_bias_kv,
                                          add_zero_attn=module.add_zero_attn, kdim=module.kdim, vdim=module.vdim,
                                          batch_first=module.batch_first)

    # copy over weight and bias tensors
    with torch.no_grad():
        if module.in_proj_weight is not None:
            weights_q, weights_k, weights_v = torch.chunk(module.in_proj_weight.data, 3, dim=0)
        else:
            weights_q = module.q_proj_weight.data
            weights_k = module.k_proj_weight.data
            weights_v = module.v_proj_weight.data
        q_MHA.linear_Q.weight.copy_(weights_q)
        q_MHA.linear_K.weight.copy_(weights_k)
        q_MHA.linear_V.weight.copy_(weights_v)

        q_MHA.out_proj.weight.copy_(module.out_proj.weight.data)

        if bias:
            bias_q, bias_k, bias_v = torch.chunk(module.in_proj_bias.data, 3, dim=0)
            if add_bias_kv:
                bias_k = q_MHA.linear_K.bias.copy_(module.bias_k.data)
                bias_v = q_MHA.linear_V.bias.copy_(module.bias_v.data)
            q_MHA.linear_K.bias.copy_(bias_k)
            q_MHA.linear_V.bias.copy_(bias_v)
            q_MHA.linear_Q.bias.copy_(bias_q)

            q_MHA.out_proj.bias.copy_(module.out_proj.bias.data)

    return q_MHA

class QuantizableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
       QuantizableTransformerEncoderLayer replaces add operations in TransformerEncoderLayer with elementwise add operations
       """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nnF.relu, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first)
        self.norm_first = norm_first
        self.add1 = aimet_modules.Add()
        self.add2 = aimet_modules.Add()

    # pylint: disable=unused-argument
    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # pylint: disable = too-many-branches
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        # pylint: disable = protected-access
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif self.norm1.eps != self.norm2.eps:
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )

        x = src
        if self.norm_first:
            x = self.add1(x, self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = self.add2(x, self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(self.add1(x, self._sa_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(self.add2(x, self._ff_block(x)))

        return x


class QuantizableTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    QuantizableTransformerDecoderLayer replaces add operations in TransformerDecoderLayer with elementwise add operations
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nnF.relu, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first)
        self.norm_first = norm_first
        self.add1 = aimet_modules.Add()
        self.add2 = aimet_modules.Add()
        self.add3 = aimet_modules.Add()

    # pylint: disable=unused-argument
    # pylint: disable=arguments-differ
    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                # Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward compatibility.
            memory_is_causal: If specified, applies a causal mask as ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect hints
                 can result in incorrect execution, including forward and backward compatibility.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = self.add1(x, self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask))
            x = self.add2(x, self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask))
            x = self.add3(x, self._ff_block(self.norm3(x)))
        else:
            x = self.norm1(self.add1(x, self._sa_block(x, tgt_mask, tgt_key_padding_mask)))
            x = self.norm2(self.add2(x, self._mha_block(x, memory, memory_mask, memory_key_padding_mask)))
            x = self.norm3(self.add3(x, self._ff_block(x)))

        return x

def copy_params_helper(src_module: Union[torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer],
                       dest_module: Union[QuantizableTransformerEncoderLayer, QuantizableTransformerDecoderLayer]):
    """
    Copy params in torch enc/dec modules to equivalent quantizable enc/dec modules
    :param src_module: source module of type torch.nn.TransformerEncoderLayer or torch.nn.TransformerDecoderLayer
    :param dest_module: dest module of type QuantizableTransformerEncoderLayer, QuantizableTransformerDecoderLayer
    """
    if isinstance(src_module, torch.nn.TransformerEncoderLayer):
        assert isinstance(dest_module, QuantizableTransformerEncoderLayer)

    if isinstance(src_module, torch.nn.TransformerDecoderLayer):
        assert isinstance(dest_module, QuantizableTransformerDecoderLayer)

    with torch.no_grad():
        # copy params of all the layers in transformerEncoderLayer to quantizable_encoder
        enc_layers = {}
        for layer_name, layer in src_module.named_children():
            enc_layers[layer_name] = layer

        q_enc_layers = {}
        for layer_name, layer in dest_module.named_children():
            q_enc_layers[layer_name] = layer

        for layer_name in enc_layers:
            for param_name, _ in enc_layers[layer_name].named_parameters():
                q_enc_layers[layer_name].get_parameter(param_name).data.copy_(
                    enc_layers[layer_name].get_parameter(param_name).data)


def create_quantizable_transformer_encoder_layer(
        transformerEncoderLayer: torch.nn.TransformerEncoderLayer) -> QuantizableTransformerEncoderLayer:
    """
    Create QuantizableTransformerEncoderLayer using existing torch.nn.TransformerEncoderLayer module
    :param transformerEncoderLayer: Existing torch.nn.TransformerEncoderLayer module
    :return: Newly created QuantizableTransformerEncoderLayer module
    """
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(transformerEncoderLayer.activation, (torch.nn.modules.activation.ReLU, torch.nn.functional.relu)):
        activation = 'relu'
    elif isinstance(transformerEncoderLayer.activation, (torch.nn.modules.activation.GELU, torch.nn.functional.gelu)):
        activation = 'gelu'
    else:
        assert False

    quantizable_encoder = QuantizableTransformerEncoderLayer(d_model=transformerEncoderLayer.linear1.in_features,
                                                             nhead=transformerEncoderLayer.self_attn.num_heads,
                                                             dim_feedforward=transformerEncoderLayer.linear1.out_features,
                                                             dropout=transformerEncoderLayer.dropout.p,
                                                             activation=activation,
                                                             layer_norm_eps=transformerEncoderLayer.norm1.eps,
                                                             batch_first=transformerEncoderLayer.self_attn.batch_first,
                                                             norm_first=transformerEncoderLayer.norm_first)

    copy_params_helper(src_module=transformerEncoderLayer, dest_module=quantizable_encoder)
    return quantizable_encoder


def create_quantizable_transformer_decoder_layer(
        transformerDecoderLayer: torch.nn.TransformerDecoderLayer) -> QuantizableTransformerDecoderLayer:
    """
    Create QuantizableTransformerDecoderLayer using existing torch.nn.TransformerDecoderLayer module
    :param transformerDecoderLayer: Existing torch.nn.TransformerDecoderLayer module
    :return: Newly created QuantizableTransformerDecoderLayer module
    """
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(transformerDecoderLayer.activation, (torch.nn.modules.activation.ReLU, torch.nn.functional.relu)):
        activation = 'relu'
    elif isinstance(transformerDecoderLayer.activation, (torch.nn.modules.activation.GELU, torch.nn.functional.gelu)):
        activation = 'gelu'
    else:
        assert False

    quantizable_decoder = QuantizableTransformerDecoderLayer(d_model=transformerDecoderLayer.linear1.in_features,
                                                             nhead=transformerDecoderLayer.self_attn.num_heads,
                                                             dim_feedforward=transformerDecoderLayer.linear1.out_features,
                                                             dropout=transformerDecoderLayer.dropout.p,
                                                             activation=activation,
                                                             layer_norm_eps=transformerDecoderLayer.norm1.eps,
                                                             batch_first=transformerDecoderLayer.self_attn.batch_first,
                                                             norm_first=transformerDecoderLayer.norm_first)

    copy_params_helper(src_module=transformerDecoderLayer, dest_module=quantizable_decoder)
    return quantizable_decoder
