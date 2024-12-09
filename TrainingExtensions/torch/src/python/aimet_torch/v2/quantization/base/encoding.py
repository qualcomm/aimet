# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
""" Base encoding definition """
import abc
import copy

import torch


__all__ = ['EncodingBase']


class EncodingBase(abc.ABC):
    """
    Quantizer encoding base class
    """
    @property
    @abc.abstractmethod
    def bitwidth(self) -> int:
        """
        Returns the bitwidth of the quantized representation
        """

    @property
    @abc.abstractmethod
    def mapping(self) -> str:
        """
        Returns the type of mapping function of this encoding object
        """

    @property
    @abc.abstractmethod
    def granularity(self) -> str:
        """
        Returns the granularity of this encoding
        """

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Returns new encoding with changed dtype and device without changing current encoding
        as `torch.Tensor.to`.
        """

    @abc.abstractmethod
    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input with the encoding

        :param input: Tensor to be quantized
        :return: Quantized tensor
        """

    @abc.abstractmethod
    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the input with the encoding

        :param input: Tensor to be dequantized
        :return: Dequantized tensor
        """

    def _detach(self) -> "EncodingBase":
        """
        Returns a new encoding object with all tensors attributes detached from the current graph
        """
        self_copy = copy.copy(self)
        for name, item in self_copy.__dict__.items():
            if isinstance(item, torch.Tensor):
                setattr(self_copy, name, item.detach())
        return self_copy

    def _clone(self) -> "EncodingBase":
        """
        Returns a new encoding object all tensor attributes cloned
        """
        self_copy = copy.copy(self)
        for name, item in self_copy.__dict__.items():
            if isinstance(item, torch.Tensor):
                setattr(self_copy, name, item.clone())
        return self_copy

    def to_qnn_encoding_dict(self, encoding_version=None):
        """
        Converts encoding object into QNN encoding dictionary
        """
        raise NotImplementedError

    @classmethod
    def from_qnn_encoding_dict(cls, encoding_dict, version=None) -> "EncodingBase":
        """
        Create an encoding object from a QNN encoding dictionary
        """
        raise NotImplementedError
