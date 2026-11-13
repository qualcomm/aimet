# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Omniquant quantizers to run Omniquant Optimizations."""

import torch
from typing import Optional

from aimet_torch.quantization import DequantizedTensor
from aimet_torch.quantization.affine import QuantizeDequantize


# pylint: disable=no-member
class OmqQuantizeDequantize:
    """Specialized class to apply Omniquant LET scale and optional LWC clipping on weight using QDQ"""

    def __init__(self, qdq: Optional[QuantizeDequantize] = None):
        """
        Creates the Omniquant Quantizer object.

        :param qdq: Optional. If qdq is a QuantizeDequantize object, OmqQuantizeDequantize will init QuantizeDequantize
                    to provide qdq feature.
        """
        if isinstance(qdq, QuantizeDequantize):
            assert qdq.symmetric is True, "Only symmetric quantization is supported"
            super().__init__(
                qdq.shape,
                qdq.bitwidth,
                qdq.symmetric,
                qdq.encoding_analyzer,
                qdq.block_size,
            )
            self.set_range(qdq.min, qdq.max)
            self.run_qdq = True
            self.min.requires_grad = True
            self.max.requires_grad = True

        else:
            super().__init__()
            self.run_qdq = False
            self._allow_overwrite = False

        self.prev_scale = None
        self.foll_scale = None
        self._cached_prev_scale = None
        self._cached_foll_scale = None
        self.enabled = True
        self.num_repeats = 1
        self.num_key_value_heads = 1

    def is_initialized(self):
        """Set to True for Non QuantizeDequantize Object to prevent from init."""
        return True

    def register_let_params(
        self, prev_scale=None, foll_scale=None, num_repeats=1, num_key_value_heads=1
    ):
        """Set prev_scale and foll_scale to LET pairs."""
        if prev_scale is not None:
            self.prev_scale = prev_scale
        if foll_scale is not None:
            self.foll_scale = foll_scale

        # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
        # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
        # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
        self.num_repeats = num_repeats
        self.num_key_value_heads = num_key_value_heads

    def fold_let_params(self, module, key):
        """Fold let scale in this quantizer to module.weight."""
        self._cache_train_scale_to_numpy()
        self._fold(module, key)
        self._reset_let_params()

    def _fold(self, module, key):
        """Impl for fold scale to model."""
        param = getattr(module, key)
        param = self._update_param(param)
        setattr(module, key, torch.nn.Parameter(param))

    def _cache_train_scale_to_numpy(self):
        """Cache trained scale to numpy tensor."""
        self._cached_prev_scale = (
            self.prev_scale.data.cpu().numpy() if self.prev_scale is not None else None
        )
        self._cached_foll_scale = (
            self.foll_scale.data.cpu().numpy() if self.foll_scale is not None else None
        )

    def _reset_let_params(self):
        """Set LET modules prev_scale/foll_scale to None"""
        self.prev_scale = None
        self.foll_scale = None

    def get_qdq(self) -> QuantizeDequantize:
        """
        Return the Quantized QDQ object for the sim object to be restored to original condition.
        """
        q = QuantizeDequantize(
            self.shape,
            self.bitwidth,
            self.symmetric,
            self.encoding_analyzer,
            self.block_size,
        )
        q.set_range(self.get_min(), self.get_max())
        return q

    def _update_param(self, weight):
        """Apply prev and foll scale to weight."""
        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            # Not to reshape scale if weight is 1d vector. e.g. Norm layers.
            if len(weight.shape) != 1:
                prev_scale = prev_scale.reshape(-1, 1)
            weight = weight / prev_scale

        if self.foll_scale is not None:
            # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
            # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
            # Ex pair: self_attn.v_proj and self_attn.o_prj  for llama in gqa
            # Repeat whole KV heads, preserving the channel order within each head.
            foll_scale = torch.repeat_interleave(
                self.foll_scale.reshape(self.num_key_value_heads, -1),
                dim=0,
                repeats=self.num_repeats,
            ).flatten()
            weight = weight * foll_scale

        return weight

    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """
        Performs QDQ on the input tensor based on the learnt parameters: prev and foll by using the
        parameters min and max tensors

        :param input: Input tensor to be QDQ
        :return: Dequantized tensor after applying AdaScale QDQ
        """
        input = self._update_param(input)

        if self.run_qdq:
            return super().forward(input)
        else:
            return input


class OmqGemmaWeightQuantizer(torch.nn.Module):
    """Specialized class to apply Omniquant LET scale on QuantizedGemma3RMSNorm"""

    def __init__(self):
        """
        Creates the Omniquant quantizer to forward weight scaling that doesn't require qdq. Usually for normalize layer.

        :param module: Parent torch.nn.Module for quantizer.
        :param key: weight or bias to quantizer
        :param qdq: If got QuantizeDequantize object will init QuantizeDequantize to enable LWC weight clipping.
                    If got None, it won't init QuantizeDequantize and quantizer will perform LET scale only.
        """
        super().__init__()
        self.enabled = True
        self._allow_overwrite = False

        self.prev_scale = None
        self.foll_scale = None

        self._cached_prev_scale = None
        self._cached_foll_scale = None

        self.num_repeats = 1

    def is_initialized(self):
        return True

    def register_let_params(self, prev_scale=None, foll_scale=None, num_repeats=1):
        """Set prev_scale and foll_scale to LET pairs."""
        if prev_scale is not None:
            self.prev_scale = prev_scale
        if foll_scale is not None:
            self.foll_scale = foll_scale

        # For some pairs prev_layer out channel != foll_layer in channel. In such cases assert that
        # foll_scale has the correct shape in let_modules. We will repeat the prev_scale num_repeats times to match the dimension.
        # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
        self.num_repeats = num_repeats

    def fold_let_params(self, module, key):
        """Fold let scale in this quantizer to module.weight."""
        self._fold(module, key)
        self._cache_train_scale_to_numpy()
        self._reset_let_params()

    def _fold(self, module, key):
        """Impl for fold scale to model."""
        weight = getattr(module, key)
        weight = self._update_param(weight)
        setattr(module, key, torch.nn.Parameter(weight))

    def _cache_train_scale_to_numpy(self):
        """Cache trained scale to numpy tensor."""
        self._cached_prev_scale = (
            self.prev_scale.data.cpu().numpy() if self.prev_scale is not None else None
        )
        self._cached_foll_scale = (
            self.foll_scale.data.cpu().numpy() if self.foll_scale is not None else None
        )

    def _reset_let_params(self):
        """Set LET modules prev_scale/foll_scale to None"""
        self.prev_scale = None
        self.foll_scale = None

    def _update_param(self, weight):
        if self.prev_scale is not None:
            prev_scale = self.prev_scale
            weight = (weight / prev_scale) + (1 / prev_scale) - 1

        return weight

    def forward(self, weight: torch.Tensor):
        """
        Performs QDQ on weight.

        :param weight: Input tensor to be QDQ
        :return: Tensor after apply Omniquant scales.
        """
        weight = self._update_param(weight)
        return weight
