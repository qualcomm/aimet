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
""" Affine encoding definition """

from typing import Tuple, Optional, Dict, Any, overload, Union, List
from itertools import chain, repeat
import torch
from torch._C._nn import _parse_to as parse_to_args

from aimet_common.defs import EncodingType
from aimet_torch.v2.utils import docstring
from aimet_torch.v2.quantization.base import EncodingBase
from aimet_torch.v2.quantization.affine.backends import quantize, dequantize, _derive_qmin_qmax
from ._utils import _GridMixin, _register_signature # pylint: disable=import-error


__all__ = ["AffineEncoding", "VectorEncoding", "GroupedBlockEncoding"]


class AffineEncoding(EncodingBase, _GridMixin):
    """
    Encoding object for affine quantization
    """
    _init_signatures = []

    @overload
    @_register_signature(_init_signatures)
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, qmin: int, qmax: int, symmetry=False,
                 block_size: Optional[Tuple[int, ...]] = None):
        ...

    @overload
    @_register_signature(_init_signatures)
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, bitwidth: int, signed=False, symmetry=False,
                 block_size: Optional[Tuple[int, ...]] = None):
        ...

    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, *args, **kwargs): # pylint: disable=too-many-locals
        self._scale = scale
        self._offset = offset
        full_args = (scale, offset, *args)

        # Pad positional args with None's such that len(args) == 4
        args = tuple(chain(args, repeat(None, 4 - len(args))))
        arg0 = kwargs.pop('qmin', kwargs.pop('bitwidth', args[0]))
        arg1 = kwargs.pop('qmax', kwargs.pop('signed', args[1]))
        symmetry = kwargs.pop('symmetry', args[2])
        if symmetry is None:
            symmetry = False
        block_size = kwargs.pop('block_size', args[3])

        if arg1 is None or isinstance(arg1, bool):
            # (arg0, arg1) == (bitwidth, signed)
            bitwidth, signed = arg0, bool(arg1)
            if (bitwidth is None) or (signed is None):
                raise self._arg_parsing_error(full_args, kwargs)
            qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=signed)
        else:
            # (arg0, arg1) == (qmin, qmax)
            qmin, qmax = arg0, arg1
            if (qmin is None) or (qmax is None):
                raise self._arg_parsing_error(full_args, kwargs)

        assert qmin is not None
        assert qmax is not None

        if kwargs:
            cls = type(self).__qualname__
            unexpected_keys = ', '.join(kwargs.keys())
            raise TypeError(f"{cls}.__init__ got unexpected keyword argument: {unexpected_keys}")

        if qmin >= qmax:
            raise ValueError(f"qmax should be strictly larger than qmin. Got qmax={qmax}, qmin={qmin}")

        self.qmin = qmin
        self.qmax = qmax
        self._symmetry = symmetry
        self._block_size = block_size

    @property
    def mapping(self) -> str:
        """
        Returns the mapping method for this encoding
        """
        return "affine"

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

    @property
    def scale(self) -> torch.Tensor:
        """
        Returns the scale of the quantizer encoding
        """
        return self._scale

    @property
    def offset(self) -> torch.Tensor:
        """
        Returns the offset of the quantizer encoding
        """
        return self._offset

    @property
    def num_steps(self) -> int:
        """
        Returns the number of steps of the quantizer encoding
        """
        return self.qmax - self.qmin

    @property
    def min(self) -> torch.Tensor:
        """
        Returns the min value of the quantizer encoding
        """
        return (self.offset + self.qmin) * self.scale

    @property
    def max(self) -> torch.Tensor:
        """
        Returns the max value of the quantizer encoding
        """
        return (self.offset + self.qmax) * self.scale

    @property
    def symmetry(self) -> bool:
        """
        Returns the symmetry mode of the quantizer encoding
        """
        return self._symmetry

    @property
    @docstring(_GridMixin._get_bitwidth.__doc__)
    def bitwidth(self) -> int: # pylint: disable=missing-function-docstring
        return self._get_bitwidth()

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        self._set_bitwidth(bitwidth)

    @property
    @docstring(_GridMixin._get_signed.__doc__)
    def signed(self) -> bool: # pylint: disable=missing-function-docstring
        return self._get_signed()

    @signed.setter
    def signed(self, signed: bool):
        self._set_signed(signed)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the quantizer encoding
        """
        if 0 <= self.qmin < self.qmax < 256:
            return torch.uint8

        if -128 <= self.qmin < self.qmax < 128:
            return torch.int8

        if -32768 <= self.qmin < self.qmax < 32768:
            return torch.int16

        return torch.int32

    @property
    def block_size(self) -> Optional[Tuple[int, ...]]:
        """
        Returns the block sizes of the quantizer encoding
        """
        return self._block_size

    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Behaves similar to torch.Tensor.to
        """
        to_args = parse_to_args(*args, **kwargs)
        device, dtype, _, _ = to_args
        dtype = dtype if dtype else self._scale.dtype
        device = device if device else self._scale.device
        if dtype is self._scale.dtype and device is self._scale.device:
            return self

        if not dtype.is_floating_point:
            raise RuntimeError(f"Cannot change encoding data dtype to {dtype}, "
                               "only floating point data types are supported")

        scale = self._scale.to(dtype=dtype, device=device)
        offset = self._offset.to(dtype=dtype, device=device)
        properties = self._get_additional_properties()
        return type(self)(scale, offset, self.qmin, self.qmax, self._symmetry, **properties)

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset
        qmin = self.qmin
        qmax = self.qmax
        block_size = self.block_size

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        return quantize(input.as_subclass(torch.Tensor),
                        scale.to(input.dtype).as_subclass(torch.Tensor),
                        offset.to(input.dtype).as_subclass(torch.Tensor),
                        qmin, qmax, block_size=block_size)

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset
        block_size = self.block_size

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        return dequantize(input.as_subclass(torch.Tensor),
                          scale.to(input.dtype).as_subclass(torch.Tensor),
                          offset.to(input.dtype).as_subclass(torch.Tensor),
                          block_size=block_size)

    def _to_legacy_format(self):
        min = self.min.flatten()
        max = self.max.flatten()
        scale = self.scale.flatten()

        # Legacy behavior is to shift offset by qmin
        offset = self.offset.flatten() + self.qmin

        return [
            {'min': float(min_), 'max': float(max_),
             'scale': float(scale_), 'offset': int(offset_),
             'bitwidth': self.bitwidth, 'dtype': 'int', 'is_symmetric': str(self.symmetry)}
            for min_, max_, scale_, offset_ in zip(min, max, scale, offset)
        ]

    # pylint: disable=no-self-use
    def _get_additional_properties(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _get_block_size(block_size: Tuple):
        assert len(block_size) >= 2
        for dim_block_size in block_size:
            if dim_block_size != 1:
                return dim_block_size
        return block_size[1]

    def to_qnn_encoding_dict(self, encoding_version=None) -> Union[List, Dict]:
        """
        Converts encoding object into QNN encoding
        """
        if encoding_version == '0.6.1':
            return self._to_legacy_format()
        if encoding_version == '1.0.0':
            encoding_dict = {'dtype': 'INT',
                             'bw': self.bitwidth,
                             'is_sym': self.symmetry,
                             'scale': self.scale.flatten().tolist()}

            # Compute signed offset if necessary
            offset = self.offset
            if self.signed:
                offset = offset - 2 ** (self.bitwidth - 1)
            encoding_dict['offset'] = offset.to(torch.int).flatten().tolist()

            assert self.granularity != 'unknown'
            if self.granularity == 'pertensor':
                encoding_dict['enc_type'] = EncodingType.PER_TENSOR.name
            elif self.granularity == 'perchannel':
                encoding_dict['enc_type'] = EncodingType.PER_CHANNEL.name
            else:
                encoding_dict['enc_type'] = EncodingType.PER_BLOCK.name
                encoding_dict['block_size'] = self._get_block_size(self.block_size)
                if encoding_dict['block_size'] == -1:
                    raise NotImplementedError('Exporting encodings to 1.0.0 format with block size -1 is not '
                                              'supported yet. Export using sim.export() instead.')
            return encoding_dict

        raise AssertionError(f'Export encoding version {encoding_version} not supported.')

class VectorEncoding(AffineEncoding):
    """
    Encoding object for vector quantization
    """
    def __init__(
        self,
        scale: torch.Tensor,
        offset: torch.Tensor,
        bitwidth: int,
        signed=False,
        symmetry=False,
        block_size: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__(scale, offset, bitwidth, signed, symmetry, block_size)
        self.rows_per_block = kwargs["rows_per_block"]
        self.cols_per_block = kwargs["cols_per_block"]
        self.vector_dim = kwargs["vector_dim"]
        self.vector_stride = kwargs["vector_stride"]
        self.index_bw = kwargs["index_bw"]

    def _to_legacy_format(self):
        encoding = super()._to_legacy_format()
        for i, _ in enumerate(encoding):
            encoding[i].update(
                rows_per_block=self.rows_per_block,
                cols_per_block=self.cols_per_block,
                vector_dim=self.vector_dim,
                vector_stride=self.vector_stride,
                index_bw=self.index_bw,
            )
        return encoding

    def _get_additional_properties(self) -> Dict[str, Any]:
        return {
            "rows_per_block": self.rows_per_block,
            "cols_per_block": self.cols_per_block,
            "vector_dim": self.vector_dim,
            "vector_stride": self.vector_stride,
            "index_bw": self.index_bw,
        }

    def to_qnn_encoding_dict(self, encoding_version=None):
        encodings = super().to_qnn_encoding_dict(encoding_version)
        if encoding_version == '1.0.0':
            encodings.update(
                rows_per_block=self.rows_per_block,
                cols_per_block=self.cols_per_block,
                vector_dim=self.vector_dim,
                vector_stride=self.vector_stride,
                index_bw=self.index_bw)
            encodings['enc_type'] = EncodingType.VECTOR.name
        return encodings

# pylint: disable=too-many-arguments
class GroupedBlockEncoding(AffineEncoding):
    """
    Encoding object for grouped block quantization
    """
    def __init__(
        self,
        scale: torch.Tensor,
        offset: torch.Tensor,
        bitwidth: int,
        signed=False,
        symmetry=False,
        block_size: Optional[Tuple[int, ...]] = None,
        block_grouping: Optional[Tuple[int, ...]] = None,
        decompressed_bw: Optional[int] = None,
        per_channel_scale: Optional[torch.Tensor] = None,
        per_block_int_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(scale, offset, bitwidth, signed, symmetry, block_size, **kwargs)
        self.block_grouping = block_grouping
        self.decompressed_bw = decompressed_bw
        self.per_channel_scale = per_channel_scale
        self.per_block_int_scale = per_block_int_scale

    def to_qnn_encoding_dict(self, encoding_version=None) -> Union[List, Dict]:
        """
        Converts encoding object into QNN encoding
        """
        encoding_dict = super().to_qnn_encoding_dict(encoding_version)

        # Version 0.6.1 currently used for save_encodings_to_json
        if all(group_size == 1 for group_size in self.block_grouping) or encoding_version == '0.6.1':
            # Equivalent to AffineEncoding
            pass
        else:
            encoding_dict['bw'] = self.decompressed_bw
            encoding_dict['compressed_bw'] = self.bitwidth
            encoding_dict['scale'] = self.per_channel_scale.flatten().tolist()
            encoding_dict['offset'] = \
                [-2 ** (self.decompressed_bw - 1) for _ in encoding_dict['scale']]
            encoding_dict['enc_type'] = EncodingType.LPBQ.name
            encoding_dict['per_block_int_scale'] = self.per_block_int_scale.flatten().tolist()
        return encoding_dict

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
