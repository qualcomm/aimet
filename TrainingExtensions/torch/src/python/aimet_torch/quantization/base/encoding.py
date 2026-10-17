# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Base encoding definition"""

import abc
import copy
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aimet_torch.quantization.base import QuantizerBase


__all__ = ["EncodingBase"]


class EncodingBase(abc.ABC):
    """
    Quantizer encoding base class
    """

    scale: torch.Tensor
    block_size: Optional[tuple[int, ...]]
    producer: Optional["QuantizerBase"]
    _input_shape_hint: Optional[tuple[int, ...]]

    def __init__(self):
        self._input_shape_hint = None

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

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> "EncodingBase":
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

    @abc.abstractmethod
    def quantize_dequantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Quantize-dequantize the input with the encoding

        :param input: Tensor to be dequantized
        :return: Quantize-dequantized tensor
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
    def _from_qnn_encoding_dict(cls, encoding_dict, version=None) -> "EncodingBase":
        """
        Create an encoding object from a QNN encoding dictionary
        """
        raise NotImplementedError

    @property
    def granularity(self) -> str:
        """
        Returns the granularity of the quantizer encoding
        """
        if self.scale.dim() == 0:
            return "pertensor"
        if self.block_size is not None:
            return "blockwise"
        non_singleton_dims = tuple(dim for dim in self.scale.shape if dim > 1)
        if len(non_singleton_dims) <= 1:
            return "perchannel"
        return "unknown"

    def _hint_input_shape(self, input_shape: tuple[int, ...]) -> "EncodingBase":
        """
        Hint shape of the input tensor to be quantized
        to concretize the scale shape and block size of the encoding
        This is internal API only for ONNX export

        Args:
            input_shape (tuple[int, ...]): Shape of the input tensor to be quantized
        """
        self._input_shape_hint = input_shape
        return self

    def _safe_get_channel_and_block_axis(self) -> tuple[int | None, int | None]:
        """
        Returns the channel axis and block axis based on the input tensor shape.
        This is internal API only for ONNX export
        """
        if self._input_shape_hint is None:
            raise RuntimeError(
                "Could not safely infer channel and block axes "
                "because input shape hint is not provided. "
                "Call `_hint_input_shape` before calling this function."
            )

        from aimet_torch.quantization._utils import concretize_block_size

        concrete_block_size = concretize_block_size(
            self._input_shape_hint,
            self.scale.shape,
            self.block_size or (),
        )

        channel_axis_candidates = []
        block_axis_candidates = []

        for axis, (n_blocks, blk_size) in enumerate(
            zip(self.scale.shape, concrete_block_size)
        ):
            if n_blocks > 1:
                if blk_size > 1:
                    block_axis_candidates.append(axis)
                else:
                    channel_axis_candidates.append(axis)

        if len(block_axis_candidates) > 1:
            raise RuntimeError(
                f"Multiple non-trivial block axes found: {block_axis_candidates}. "
                f"Cannot determine a unique block axis."
            )

        (block_axis,) = block_axis_candidates or (None,)

        if not channel_axis_candidates and block_axis is not None:
            # A singleton dimension can still be the channel axis if its block size is 1.
            channel_axis_candidates = [
                axis
                for axis, blk_size in enumerate(concrete_block_size)
                if blk_size == 1
            ]

        if len(channel_axis_candidates) > 1:
            raise RuntimeError(
                f"Multiple non-trivial channel axes found: {channel_axis_candidates}. "
                f"Cannot determine a unique channel axis."
            )

        (channel_axis,) = channel_axis_candidates or (None,)

        if channel_axis is None and block_axis is not None:
            raise RuntimeError(
                f"Block axis {block_axis} found without a corresponding channel axis. "
                f"Cannot determine a unique channel axis."
            )

        if channel_axis is not None:
            # Convert to negative axis
            channel_axis = channel_axis - self.scale.dim()

        if block_axis is not None:
            # Convert to negative axis
            block_axis = block_axis - self.scale.dim()

        return channel_axis, block_axis
