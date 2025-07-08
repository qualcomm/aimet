# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Custom QcQuantizeOp to quantize weights and activations using ONNXRuntime"""

# pylint: disable=too-many-lines
from __future__ import (
    annotations,
)  # Needed to typehint private class _EncodingMismatchInfo
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Tuple
import numpy as np

from aimet_common import libpymo
from aimet_common.defs import (
    QuantScheme,
    MAP_QUANT_SCHEME_TO_PYMO,
    QuantizationDataType,
    EncodingType,
)
from aimet_common import libquant_info
from aimet_common.utils import deprecated
from aimet_common.quantsim import calculate_delta_offset, create_encoding_from_min_max
from aimet_onnx import lpbq_utils


OpMode = libpymo.TensorQuantizerOpMode


class TensorQuantizerParams:
    """
    Per channel quantization parameters
    """

    def __init__(self, tensor_shape, channel_axis: int = None, block_axis: int = None):
        """

        :param tensor_shape: Shape of the input tensor
        :param channel_axis: Axis along which per channel quantization is performed
        :param block_axis: Axis along which blockwise quantization is performed
        """
        self.tensor_shape = tensor_shape
        self.channel_axis = channel_axis
        self.block_axis = block_axis


# pylint: disable=too-many-public-methods
class QcQuantizeOp:
    """A custom quantization operation to perform using ONNXRuntime"""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        quant_info: libquant_info.QcQuantizeInfo,
        quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
        rounding_mode: str = "nearest",
        op_mode: Union[OpMode, None] = None,
        bitwidth: int = 8,
        use_symmetric_encodings: bool = False,
        tensor_quantizer_params: Union[TensorQuantizerParams, None] = None,
    ):
        """
        Args:
            quant_info: libquant_info.QcQuantizeInfo object holding quantization parameters passed to the C++ op
            quant_scheme: Quantization scheme (e.g. QuantScheme.post_training_tf)
            rounding_mode: Rounding mode (e.g. nearest)
            op_mode: QcQuantizeOp mode (e.g. update_stats)
            bitwidth: Quantization bitwidth
            use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
            tensor_quantizer_params: Parameters like number of output channels, axis if per channel quantization is performed
        """
        self.quant_info = quant_info
        self._tensor_quantizer = libpymo.BlockTensorQuantizer(
            [], bitwidth, MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
        )
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self._is_encoding_frozen = False
        self.op_mode = op_mode
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = True
        self.data_type = QuantizationDataType.int
        self.tensor_quantizer_params = tensor_quantizer_params
        self._encoding_min_max_fixed_vals = None

    def is_encoding_frozen(self) -> bool:
        """Returns is_encoding_frozen var"""
        return self._is_encoding_frozen

    def freeze_encodings(self):
        """Sets encodings to frozen"""
        self._is_encoding_frozen = True

    def enable_per_channel_quantization(self, enable: bool = True):
        """
        Enables per channel quantization for qc_quantize_op
        """
        self.quant_info.usePerChannelMode = enable

        if enable:
            assert self.tensor_quantizer_params is not None
            assert self.tensor_quantizer_params.channel_axis is not None
            channel_axis = self.tensor_quantizer_params.channel_axis
            self.quant_info.channelAxis = (
                channel_axis
                if channel_axis >= 0
                else channel_axis + len(self.tensor_quantizer_params.tensor_shape)
            )

        self._tensor_quantizer = self._build_tensor_quantizer()

    def _enable_blockwise_quantization(self, block_size):
        assert self.tensor_quantizer_params is not None
        tensor_shape = self.tensor_quantizer_params.tensor_shape
        block_axis = self.tensor_quantizer_params.block_axis
        channel_axis = self.tensor_quantizer_params.channel_axis
        assert channel_axis is not None
        assert block_axis is not None
        assert block_axis != channel_axis

        if block_size != 0:
            if tensor_shape[block_axis] % block_size != 0:
                raise ValueError(
                    f"Input shape {tensor_shape} not divisible by block size {block_size} at axis {block_axis}"
                )

        self.quant_info.usePerChannelMode = True
        self.quant_info.channelAxis = (
            channel_axis if channel_axis >= 0 else channel_axis + len(tensor_shape)
        )
        self.quant_info.blockAxis = (
            block_axis if block_axis >= 0 else block_axis + len(tensor_shape)
        )
        self.quant_info.blockSize = block_size
        self._tensor_quantizer = self._build_tensor_quantizer()

    @property
    def data_type(self) -> QuantizationDataType:
        """
        Returns the data type for quantization

        :return: Quantization data type
        """
        return (
            QuantizationDataType.int
            if self.quant_info.isIntDataType
            else QuantizationDataType.float
        )

    @data_type.setter
    def data_type(self, data_type: QuantizationDataType):
        """
        Sets the quantization data type field in the op and sets isIntDataType inside quantizer_info to true or false
        based on the data type

        :param data_type: Quantization data type
        """
        if not self._is_encoding_frozen:
            self.quant_info.isIntDataType = data_type == QuantizationDataType.int

    def _build_tensor_quantizer(self):
        shape = ()
        if self.quant_info.usePerChannelMode:
            assert self.tensor_quantizer_params is not None

            input_shape = self.tensor_quantizer_params.tensor_shape
            channel_axis = self.quant_info.channelAxis
            block_axis = self.quant_info.blockAxis
            block_size = self.quant_info.blockSize

            shape = tuple(
                input_dim
                if axis == channel_axis
                else input_dim // block_size
                if axis == block_axis and block_size > 0
                else 1
                for axis, input_dim in enumerate(input_shape)
            )

        quantizer = libpymo.BlockTensorQuantizer(
            shape, self.bitwidth, MAP_QUANT_SCHEME_TO_PYMO[self.quant_scheme]
        )
        quantizer.setUnsignedSymmetric(self.use_unsigned_symmetric)
        quantizer.setStrictSymmetric(self.use_strict_symmetric)

        return quantizer

    @property
    def _tensor_quantizer(self):
        return self.quant_info.tensorQuantizerRef

    @_tensor_quantizer.setter
    def _tensor_quantizer(self, tensor_quantizer):
        self.quant_info.tensorQuantizerRef = tensor_quantizer

    @property
    def bitwidth(self):
        """Get the bitwidth of the quantizer"""
        return self.quant_info.tensorQuantizerRef.bitwidth

    @bitwidth.setter
    def bitwidth(self, bitwidth):
        self._tensor_quantizer.bitwidth = bitwidth

    @property
    def enabled(self) -> bool:
        """
        If False, quant_info.OpMode will be overriden with OpMode.passThrough to prevent quantization
        :return: True if the quantizer is to be utilized, False otherwise
        """
        return self.quant_info.enabled

    @enabled.setter
    def enabled(self, enable: bool):
        """
        Set the value of enabled to be accessed by the C++ op
        :param enable: True if the op is to be utilized, False will override the OpMode with passThrough
        """
        self.quant_info.enabled = enable

    @property
    def use_symmetric_encodings(self) -> bool:
        """
        Reads useSymmetricEncoding from the node's QcQuantizeInfo object
        :return: True if the node is to use symmetric encodings
        """
        return self.quant_info.useSymmetricEncoding

    @use_symmetric_encodings.setter
    def use_symmetric_encodings(self, use_symmetric_encodings: bool):
        """
        Sets the useSymmetricEncoding attribute of the nodes QcQuantizeInfo object
        :param use_symmetric_encodings: True if the node is to use symmetric encodings
        """
        if not self._is_encoding_frozen:
            self.quant_info.useSymmetricEncoding = use_symmetric_encodings

    @property
    def use_strict_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if strict symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._reset_encodings()

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._reset_encodings()

    def get_encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Reads the encodings object from the node's QcQuantizeInfo

        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        if not self.is_initialized() or self.data_type == QuantizationDataType.float:
            return None
        return self._tensor_quantizer.getEncodings()

    @property
    @deprecated(f"Use {get_encodings.__qualname__} instead")
    def encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """Deprecated. Use :meth:`get_encodings` to set the quantizer encodings.

        Reads the encodings object from the node's QcQuantizeInfo

        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        return self.get_encodings()

    def _get_scale(self) -> Optional[np.ndarray]:
        encodings = self.get_encodings()
        if encodings is None:
            return None
        return np.array([enc.delta for enc in encodings], dtype=np.float32).reshape(
            self._encoding_shape()
        )

    def _get_offset(self) -> Optional[np.ndarray]:
        encodings = self.get_encodings()
        if encodings is None:
            return None
        return np.array([enc.offset for enc in encodings], dtype=np.float32).reshape(
            self._encoding_shape()
        )

    def _encoding_shape(self):
        """
        Returns expected shape of y_scale and y_zero_point as defined in onnx::QuantizeLinear
        EXCEPT this function allows slightly more flexible shapes in blockwise encodings
        """
        if not self.quant_info.usePerChannelMode:
            return ()

        assert self.tensor_quantizer_params is not None

        input_shape = self.tensor_quantizer_params.tensor_shape
        channel_axis = self.quant_info.channelAxis
        block_axis = self.quant_info.blockAxis
        block_size = self.quant_info.blockSize

        if block_size > 0:
            # NOTE
            # Given input of shape (i_0, i_1, i_2, i_3, ...), block_size=B, and block_axis=1
            # onnx::QuantizeLinear only allows y_scale of shape (i_0, i_1/B, i_2, i_3, ...).
            # In practice, this enforces the input to be a 2D matrix of shape (i_0, i_1),
            # meaning only Gemm but not Conv can be blockwise quantized.
            # This is a deal-breaking limitation for us.
            # For internal purposes, we carefully deviate from onnx definition and allow
            # blockwise scale of shape (i_0, i_1/B, 1, 1, ...)
            return tuple(
                input_dim
                if axis == channel_axis
                else input_dim // block_size
                if axis == block_axis
                else 1
                for axis, input_dim in enumerate(input_shape)
            )

        return (input_shape[channel_axis],)

    def update_quantizer_and_load_encodings(
        self,
        encoding: List[libpymo.TfEncoding],
        is_symmetric: Optional[bool],
        is_strict_symmetric: Optional[bool],
        is_unsigned_symmetric: Optional[bool],
        data_type: QuantizationDataType,
    ):
        """
        Update quantizer settings and load pre-existing encodings to quantizer which can be used during
        quantize-dequantize.

        :param encoding: The libpymo.TfEncoding object to be used by the C++ op
        :param is_symmetric: True if encoding is symmetric, False otherwise
        :param is_strict_symmetric: True if encoding is strict symmetric, False otherwise
        :param is_unsigned_symmetric: True if encoding is unsigned symmetric, False otherwise
        :param data_type: Data type of encoding
        """
        self.enabled = True
        self.bitwidth = encoding[0].bw
        self.data_type = data_type
        if self.data_type == QuantizationDataType.int:
            assert self.use_symmetric_encodings is not None
            assert self.use_strict_symmetric is not None
            assert self.use_unsigned_symmetric is not None

            self.use_symmetric_encodings = is_symmetric
            if self.use_symmetric_encodings:
                self.use_strict_symmetric = is_strict_symmetric
            # is_unsigned_symmetric is a special case since the flag could be enabled but the encoding can be signed
            # if the observed tensor had negative values.
            # To err on the side of caution, only set self.use_unsigned_symmetric if we know for sure that the encodings
            # were unsigned.
            if self.use_symmetric_encodings and is_unsigned_symmetric:
                self.use_unsigned_symmetric = is_unsigned_symmetric

            self.load_encodings(encoding)

    def _load_encodings_dict(self, encoding_dict: dict, allow_overwrite: bool = True):
        self.bitwidth = encoding_dict["bw"]
        data_type = (
            QuantizationDataType.int
            if encoding_dict["dtype"] == QuantizationDataType.int.name.upper()
            else QuantizationDataType.float
        )
        self.data_type = data_type

        if data_type == QuantizationDataType.float:
            return

        if encoding_dict["enc_type"] == EncodingType.LPBQ.name:
            raise AssertionError(
                f"Loading LPBQ encodings for tensor name {encoding_dict['name']} into a non-LPBQ quantizer"
                f" is not yet supported. Ensure QuantizationSimModel is set with proper quantizers before "
                f"loading."
            )
        if encoding_dict["enc_type"] == EncodingType.PER_TENSOR.name:
            self.enable_per_channel_quantization(False)
        elif encoding_dict["enc_type"] == EncodingType.PER_CHANNEL.name:
            self.enable_per_channel_quantization()
        elif encoding_dict["enc_type"] == EncodingType.PER_BLOCK.name:
            self._enable_blockwise_quantization(encoding_dict["block_size"])
        else:
            raise RuntimeError(
                f"Cannot load encodings for unknown encoding type {encoding_dict['enc_type']}"
            )

        is_symmetric, is_strict_symmetric, is_unsigned_symmetric = (
            _get_symmetric_properties(encoding_dict)
        )
        self.use_symmetric_encodings = is_symmetric
        if self.use_symmetric_encodings:
            self.use_strict_symmetric = is_strict_symmetric
        # is_unsigned_symmetric is a special case since the flag could be enabled but the encoding can be signed
        # if the observed tensor had negative values.
        # To err on the side of caution, only set self.use_unsigned_symmetric if we know for sure that the encodings
        # were unsigned.
        if self.use_symmetric_encodings and is_unsigned_symmetric:
            self.use_unsigned_symmetric = is_unsigned_symmetric

        libpymo_encodings = []
        scales = encoding_dict["scale"]
        offsets = encoding_dict["offset"]
        for idx, scale in enumerate(scales):
            enc = libpymo.TfEncoding()
            enc.bw = encoding_dict["bw"]
            enc_min = scale * offsets[idx]
            enc_max = scale * (2 ** encoding_dict["bw"] - 1 + offsets[idx])
            enc.delta, enc.max, enc.min, enc.offset = (
                scale,
                enc_max,
                enc_min,
                offsets[idx],
            )
            libpymo_encodings.append(enc)

        self.load_encodings(libpymo_encodings)

        if not allow_overwrite:
            self.freeze_encodings()

        self.enabled = True

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        """
        Load pre-existing encodings to the quantizer and sets the op-mode to quantize-dequantize

        :param encoding: The list of libpymo.TfEncoding objects to be used by the C++ op
        """
        assert isinstance(encoding, (list, tuple))
        if self.data_type == QuantizationDataType.float:
            raise RuntimeError(
                f"{type(self).load_encodings.__qualname__} is not supported for floating-point quantizers."
            )
        self._tensor_quantizer.setEncodings(encoding)
        self.op_mode = OpMode.quantizeDequantize

    @encodings.setter
    @deprecated(f"Use {load_encodings.__qualname__} instead.")
    def encodings(self, encoding: Union[List[libpymo.TfEncoding], None]):
        """Deprecated. Use :meth:`load_encodings` to set the quantizer encodings.

        Stores encoding in self._encoding to prevent deletion and sets self.quant_info.encoding to point to encoding.
        If encoding is None, creates an empty encoding to prevent seg faults
        :param encoding: The libpymo.TfEncoding object to be used by the C++ op
        """
        if encoding is None:
            self._reset_encodings()

        else:
            self.load_encodings(encoding)

    @property
    def op_mode(self) -> OpMode:
        """
        Reads the OpMode from the node's quant_info object
        :return: The node's current mode of operation
        """
        return self.quant_info.opMode

    @op_mode.setter
    def op_mode(self, op_mode: OpMode):
        """
        Sets the opMode field in the node's quant_info
        :param op_mode: The OpMode to be used
        """
        self.quant_info.opMode = op_mode

    def reset_encoding_stats(self):
        """
        reset the stats of tensor quantizer
        """
        if not self._is_encoding_frozen:
            self._tensor_quantizer.resetEncodingStats()
            self._reset_encodings()

    def _reset_encodings(self):
        """
        Resets the quantizer's encodings
        """
        if self.is_encoding_frozen():
            return

        self._tensor_quantizer.isEncodingValid = False

    def set_bitwidth(self, bitwidth: int):
        """
        Set bitwidth for quantization
        """
        if not self._is_encoding_frozen and bitwidth != self.bitwidth:
            self.bitwidth = bitwidth
            self._reset_encodings()

    def set_quant_scheme(self, quant_scheme: QuantScheme):
        """
        Set QcQuantizeOp as given quant scheme
        """
        self.quant_scheme = quant_scheme
        self._tensor_quantizer = self._build_tensor_quantizer()
        self.reset_encoding_stats()

    def compute_encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Compute and return encodings of each tensor quantizer
        """
        if self._is_encoding_frozen:
            return None

        if not self.enabled:
            return None

        if self._encoding_min_max_fixed_vals is None:
            encodings = self._tensor_quantizer.computeEncodings(
                self.use_symmetric_encodings
            )
        else:
            min_val, max_val = self._encoding_min_max_fixed_vals
            encodings = [
                create_encoding_from_min_max(
                    min_val,
                    max_val,
                    self.bitwidth,
                    self.use_symmetric_encodings,
                    self.use_strict_symmetric,
                )
                for _ in range(int(np.prod(self._encoding_shape())))
            ]
        self.load_encodings(encodings)
        return encodings

    def get_stats_histogram(self) -> List[List]:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Get histogram of statistics. Returns list of buckets where each bucket is
        tuple of two values - the float value representing the left edge of the
        bucket and a PDF of the values in this bucket relative to all the values
        seen across all buckets.

        :return: List of buckets where each bucket is (xLeft, PDF).
        """
        if self.quant_scheme != QuantScheme.post_training_tf_enhanced:
            raise RuntimeError(
                "get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced."
            )

        if not self.get_encodings():
            raise RuntimeError(
                "get_stats_histogram() can be invoked only when encoding is computed."
            )

        return self._tensor_quantizer.getStatsHistogram()

    def is_initialized(self) -> bool:
        """
        Returns True if all quantizers have been initialized, False otherwise
        """
        if self.data_type == QuantizationDataType.float:
            # Fp16 quantizers do not need to be initialized
            return True

        return self._tensor_quantizer.isEncodingValid

    def export_encodings(self, encoding_version: str = "0.6.1"):
        """
        Exports the quantizer's encodings in the selected format.

        :param encoding_version: Version string indicated the encoding export format.
        """
        if encoding_version == "0.6.1":
            return self._export_legacy_encodings()

        if encoding_version == "1.0.0":
            return self._export_1_0_0_encodings()

        if encoding_version == "2.0.0":
            return self._export_2_0_0_encodings()

        raise RuntimeError(f"Unsupported encoding export version: {encoding_version}")

    def _export_legacy_encodings(self) -> Union[List, None]:
        """
        Create encoding dictionary from encoding object

        :return: List of encoding dictionaries in 0.6.1 encoding format
        """
        if not self.enabled or not self.is_initialized():
            return None

        if self.data_type == QuantizationDataType.float:
            return [{"bitwidth": self.bitwidth, "dtype": "float"}]

        if self.data_type == QuantizationDataType.int:
            encodings = []
            for encoding in self.get_encodings():
                enc_dict = {
                    "min": encoding.min,
                    "max": encoding.max,
                    "scale": encoding.delta,
                    "offset": int(encoding.offset),
                    "bitwidth": encoding.bw,
                    "is_symmetric": str(self.use_symmetric_encodings),
                    "dtype": "int",
                }
                encodings.append(enc_dict)
            return encodings

        raise RuntimeError(f"Exporting data type {self.data_type} not supported")

    def _encoding_type(self):
        if (
            not self.quant_info.usePerChannelMode
            or self.data_type == QuantizationDataType.float
        ):
            return EncodingType.PER_TENSOR
        if not self.quant_info.blockSize:
            return EncodingType.PER_CHANNEL
        return EncodingType.PER_BLOCK

    def _export_1_0_0_encodings(self) -> Optional[Dict]:
        """
        Exports the quantizer's encodings in the "1.0.0" encoding format
        """
        if not self.enabled or not self.is_initialized():
            return None

        enc_dict = {
            "enc_type": self._encoding_type().name,
            "dtype": "INT" if self.data_type == QuantizationDataType.int else "FLOAT",
            "bw": self.bitwidth,
        }

        if self.data_type == QuantizationDataType.int:
            enc_dict["is_sym"] = self.use_symmetric_encodings
            encodings = self.get_encodings()
            enc_dict["scale"] = [enc.delta for enc in encodings]
            enc_dict["offset"] = [enc.offset for enc in encodings]
            if self.quant_info.blockSize > 0:
                enc_dict["block_size"] = self.quant_info.blockSize

        return enc_dict

    def _export_2_0_0_encodings(self) -> Optional[Dict]:  # pylint: disable=too-many-branches
        if (
            not self.enabled
            or not self.is_initialized()
            or (self.data_type == QuantizationDataType.float and self.bitwidth >= 16)
        ):
            return None

        encodings = self.get_encodings()

        if encodings is None:
            # This means one of the three:
            #   1. This quantizer not enabled
            #   2. This quantizer not initialized
            #   3. This quantizer is a floating point quantizer
            # In any case, this corresponds to no-encoding in encoding_version 2.0.0
            return None

        signed = self.use_symmetric_encodings
        bw = encodings[0].bw

        output_dtype = f"int{bw}" if signed else f"uint{bw}"

        y_scale = np.array([e.delta for e in encodings])
        offset = np.array([e.offset for e in encodings])

        # NOTE: AIMET TfEncoding offset is defined in a bit quirky way
        #
        #                    (AIMET)
        #                +-  -offset                    ... uint4, uint8, uint16, uint32
        # y_zero_point = |
        #    (ONNX)      +-  -offset - 2 ** (bits - 1)  ... int4, int8, int16, int32
        #                    (AIMET)
        if signed:
            y_zero_point = -offset - 2 ** (bw - 1)
        else:
            y_zero_point = -offset

        if self.quant_info.usePerChannelMode and self.tensor_quantizer_params:
            channel_axis = self.tensor_quantizer_params.channel_axis
            block_axis = self.tensor_quantizer_params.block_axis
            block_size = self.quant_info.blockSize or None
        else:
            channel_axis = None
            block_axis = None
            block_size = None

        if block_size is not None:
            axis = block_axis
        elif channel_axis is not None:
            axis = channel_axis
        else:
            axis = None
            assert y_scale.size == 1
            assert y_zero_point.size == 1

        y_scale = y_scale.reshape(self._encoding_shape())
        y_zero_point = y_zero_point.reshape(self._encoding_shape()).astype(np.int64)

        y_scale = y_scale.tolist()
        y_zero_point = None if np.all(y_zero_point == 0) else y_zero_point.tolist()

        ret = {
            "output_dtype": output_dtype,
            "y_scale": y_scale,
        }
        if y_zero_point is not None:
            ret.update({"y_zero_point": y_zero_point})
        if axis is not None:
            ret.update({"axis": axis})
        if block_size is not None:
            ret.update({"block_size": block_size})

        return ret

    def update_encoding_stats(self, tensor: np.ndarray):
        """
        Update the stats for computing encodings.

        :param tensor: Tensor to use for updating the encodings stats
        """
        self._tensor_quantizer.updateStats(tensor)

    def clip_and_recompute_encodings(self, clamp_val: float) -> bool:
        """
        Clips min and max values and recomputes the encodings

        :param clamp_val: Clamping value
        :return: A boolean value telling whether clipping was performed
        """
        encodings = self.get_encodings()
        is_clipped = False

        if (not encodings) or (not self.enabled) or self._is_encoding_frozen:
            return None

        for encoding in encodings:
            e_min = encoding.min
            e_max = encoding.max
            if e_min < -clamp_val or e_max > clamp_val:
                tensor = np.clip(np.array([e_min, e_max]), -clamp_val, clamp_val)
                delta, offset = calculate_delta_offset(
                    min_val=tensor[0],
                    max_val=tensor[1],
                    bitwidth=self.bitwidth,
                    use_symmetric_encodings=self.use_symmetric_encodings,
                    use_strict_symmetric=self.use_strict_symmetric,
                )
                encoding.min = tensor[0]
                encoding.max = tensor[1]
                encoding.delta = delta
                encoding.offset = offset

                is_clipped = True

        self.load_encodings(encodings)

        return is_clipped

    def set_fixed_encoding_range(self, fixed_range: Tuple[float, float]):
        """
        Set the min/max values to be used when computing encodings

        :param fixed_range: Tuple of (min, max) value to use in-place of observer statistics when computing encodings
        """
        self._encoding_min_max_fixed_vals = fixed_range

    def _fill_mismatching_encoding_settings_info(
        self,
        encoding_dict: Optional[dict],
        encoding_mismatch_info: _EncodingMismatchInfo,
    ):
        # Match enabled state
        if self.enabled and encoding_dict is None:
            encoding_mismatch_info.enabled_mismatch = (self.enabled, False)
        if not self.enabled and encoding_dict is not None:
            encoding_mismatch_info.enabled_mismatch = (self.enabled, True)
            return  # Other mismatch info is irrelevant

        if encoding_dict is not None:
            is_symmetric, is_strict_symmetric, is_unsigned_symmetric = (
                _get_symmetric_properties(encoding_dict)
            )

            if self._encoding_type().name != encoding_dict["enc_type"]:
                encoding_mismatch_info.enc_type_mismatch = (
                    self._encoding_type(),
                    encoding_dict["enc_type"],
                )
            else:
                if self.bitwidth != encoding_dict["bw"]:
                    encoding_mismatch_info.bitwidth_mismatch = (
                        self.bitwidth,
                        encoding_dict["bw"],
                    )
            if self.data_type.name.upper() != encoding_dict["dtype"]:
                encoding_mismatch_info.dtype_mismatch = (
                    self.data_type.name,
                    encoding_dict["dtype"],
                )
            if self.data_type == QuantizationDataType.int:
                if self.use_symmetric_encodings != is_symmetric:
                    encoding_mismatch_info.is_symmetric_mismatch = (
                        self.use_symmetric_encodings,
                        is_symmetric,
                    )
                if self.use_strict_symmetric != is_strict_symmetric:
                    encoding_mismatch_info.is_strict_symmetric_mismatch = (
                        self.use_strict_symmetric,
                        is_strict_symmetric,
                    )

                # Unsigned symmetric is a special case because even if the setting is true, the encodings may appear to be
                # signed symmetric if any observed tensor values were < 0.
                # In this case, only mark a mismatch if quantizer was set to signed symmetric but an unsigned symmetric
                # encoding was seen.
                if (
                    self.use_unsigned_symmetric != is_unsigned_symmetric
                    and not self.use_unsigned_symmetric
                ):
                    encoding_mismatch_info.is_unsigned_symmetric_mismatch = (
                        self.use_unsigned_symmetric,
                        is_unsigned_symmetric,
                    )


class GroupedBlockQuantizeDequantize(QcQuantizeOp):
    """Class for performing Grouped Block Quantize Dequantize"""

    def __init__(
        self,
        quant_info: libquant_info.QcQuantizeInfo,
        bitwidth: int,
        decompressed_bw: int,
        block_size: int,
        quant_scheme: QuantScheme,
        op_mode: OpMode,
        tensor_quantizer_params: TensorQuantizerParams,
    ):
        if (
            block_size
            and tensor_quantizer_params.tensor_shape[tensor_quantizer_params.block_axis]
            % block_size
            != 0
        ):
            raise ValueError(
                f"Input shape {tensor_quantizer_params.tensor_shape} is not divisible by block size "
                f"{block_size} at axis {tensor_quantizer_params.block_axis}"
            )
        super().__init__(
            quant_info=quant_info,
            quant_scheme=quant_scheme,
            op_mode=op_mode,
            bitwidth=bitwidth,
            use_symmetric_encodings=True,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        self.decompressed_bw = decompressed_bw
        self._enable_blockwise_quantization(block_size)
        self.data_type = QuantizationDataType.int

    def _get_per_channel_scale(self) -> Optional[np.ndarray]:
        scale = self._get_scale()
        if scale is None:
            return None

        decompressed_bw = self.decompressed_bw
        compressed_bw = self.bitwidth
        _, per_channel_scale = lpbq_utils.grouped_dynamic_quantize(
            scale, self._block_grouping(), decompressed_bw - compressed_bw
        )
        return per_channel_scale

    def _block_grouping(self):
        grouping = [1 for _ in range(len(self._encoding_shape()))]
        if self.quant_info.blockSize > 0 and self.quant_info.blockAxis >= 0:
            grouping[self.quant_info.blockAxis] = -1

        return grouping

    def _load_encodings_dict(self, encoding_dict: dict, allow_overwrite: bool = True):
        # pylint: disable=too-many-locals
        data_type = (
            QuantizationDataType.int
            if encoding_dict["dtype"] == QuantizationDataType.int.name.upper()
            else QuantizationDataType.float
        )
        if data_type == QuantizationDataType.float:
            raise AssertionError(
                f"Loading float encodings for tensor name {encoding_dict['name']} into a GroupedBlock quantizer is not yet "
                f"supported. Ensure QuantizationSimModel is set with proper quantizers before loading."
            )

        if encoding_dict["enc_type"] != EncodingType.LPBQ.name:
            raise AssertionError(
                f"Loading non-LPBQ encodings for tensor name {encoding_dict['name']} into an LPBQ quantizer is not yet supported."
                f" Ensure QuantizationSimModel is set with proper quantizers before loading."
            )

        is_symmetric, is_strict_symmetric, is_unsigned_symmetric = (
            _get_symmetric_properties(encoding_dict)
        )
        self.use_symmetric_encodings = is_symmetric
        if self.use_symmetric_encodings:
            self.use_strict_symmetric = is_strict_symmetric
        # is_unsigned_symmetric is a special case since the flag could be enabled but the encoding can be signed
        # if the observed tensor had negative values.
        # To err on the side of caution, only set self.use_unsigned_symmetric if we know for sure that the encodings
        # were unsigned.
        if self.use_symmetric_encodings and is_unsigned_symmetric:
            self.use_unsigned_symmetric = is_unsigned_symmetric

        self.data_type = data_type
        self.decompressed_bw = encoding_dict["bw"]
        self.bitwidth = encoding_dict["compressed_bw"]
        if self.quant_info.blockSize != encoding_dict["block_size"]:
            self._enable_blockwise_quantization(encoding_dict["block_size"])

        libpymo_encodings = []
        encoding_shape = self._encoding_shape()
        channel_axis = self.quant_info.channelAxis
        block_axis = self.quant_info.blockAxis

        if channel_axis < block_axis:
            per_block_int_scales_np = np.array(
                encoding_dict["per_block_int_scale"]
            ).reshape(encoding_shape[channel_axis], -1)
            per_channel_scales_np = np.array(encoding_dict["scale"]).reshape(
                encoding_shape[channel_axis], 1
            )
        else:
            per_block_int_scales_np = np.array(
                encoding_dict["per_block_int_scale"]
            ).reshape(-1, encoding_shape[channel_axis])
            per_channel_scales_np = np.array(encoding_dict["scale"]).reshape(
                1, encoding_shape[channel_axis]
            )
        per_block_scales_np = per_channel_scales_np * per_block_int_scales_np
        per_block_scales = per_block_scales_np.reshape(-1).tolist()
        per_block_offsets = [-(2 ** (encoding_dict["compressed_bw"] - 1))] * len(
            per_block_scales
        )

        for idx, scale in enumerate(per_block_scales):
            enc = libpymo.TfEncoding()
            enc.bw = encoding_dict["compressed_bw"]
            enc_min = scale * per_block_offsets[idx]
            enc_max = scale * (
                2 ** encoding_dict["compressed_bw"] - 1 + per_block_offsets[idx]
            )
            enc.delta, enc.max, enc.min, enc.offset = (
                scale,
                enc_max,
                enc_min,
                per_block_offsets[idx],
            )
            libpymo_encodings.append(enc)

        self.load_encodings(libpymo_encodings)

        if not allow_overwrite:
            self.freeze_encodings()

        self.enabled = True

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        encoding = lpbq_utils.compress_encoding_scales(
            encoding,
            self._encoding_shape(),
            self._block_grouping(),
            scale_bitwidth=self.decompressed_bw - self.bitwidth,
        )
        super().load_encodings(encoding)

    def _export_legacy_encodings(self) -> Union[List, None]:
        raise NotImplementedError(
            f"0.6.1 encoding format is not supported for {type(self).__qualname__}. Please export "
            f"using 1.0.0 format instead."
        )

    def _encoding_type(self):
        encoding_type = super()._encoding_type()
        if encoding_type == EncodingType.PER_BLOCK:
            return EncodingType.LPBQ
        return encoding_type

    def _export_1_0_0_encodings(self) -> Optional[Dict]:
        encodings = super()._export_1_0_0_encodings()
        if not encodings:
            return None
        if "block_size" not in encodings:
            return encodings

        encodings["compressed_bw"] = self.bitwidth
        encodings["bw"] = self.decompressed_bw
        scale, _ = lpbq_utils.encodings_to_scale_offset_arrays(
            self.get_encodings(), self._encoding_shape()
        )
        compressed_bw = self.bitwidth
        decompressed_bw = self.decompressed_bw
        per_block_int_scale, per_channel_scale = lpbq_utils.grouped_dynamic_quantize(
            scale, self._block_grouping(), decompressed_bw - compressed_bw
        )
        encodings["per_block_int_scale"] = (
            per_block_int_scale.astype(np.uint32).flatten().tolist()
        )
        encodings["scale"] = per_channel_scale.flatten().tolist()
        encodings["offset"] = [
            -(2 ** (self.decompressed_bw - 1)) for _ in encodings["scale"]
        ]

        return encodings

    def _export_2_0_0_encodings(self) -> Optional[Dict]:
        encodings = super()._export_2_0_0_encodings()

        if encodings is None:
            return None

        output_dtype = encodings.pop("output_dtype")
        y_zero_point = encodings.pop("y_zero_point", None)

        if y_zero_point is not None and np.any(np.array(y_zero_point) != 0):
            raise RuntimeError(
                f"LPBQ only supports symmetric quantization; got non-zero y_zero_point {y_zero_point}"
            )

        compressed_bw = self.bitwidth
        decompressed_bw = self.decompressed_bw
        y_scale = np.array(encodings.pop("y_scale"))
        per_block_int_scale, per_channel_scale = lpbq_utils.grouped_dynamic_quantize(
            y_scale, self._block_grouping(), decompressed_bw - compressed_bw
        )
        per_channel_scale = per_channel_scale.squeeze(
            tuple(range(1, per_channel_scale.ndim, 2))
        )
        assert per_block_int_scale.ndim == per_channel_scale.ndim

        return {
            "per_block_int_scale": per_block_int_scale.astype(np.uint32).tolist(),
            "per_channel_float_scale": per_channel_scale.tolist(),
            **encodings,
            "output_dtype": f"int{compressed_bw}"
            if output_dtype.startswith("int")
            else f"uint{compressed_bw}",
        }

    def _fill_mismatching_encoding_settings_info(
        self,
        encoding_dict: Optional[dict],
        encoding_mismatch_info: _EncodingMismatchInfo,
    ):
        super()._fill_mismatching_encoding_settings_info(
            encoding_dict, encoding_mismatch_info
        )
        encoding_mismatch_info.bitwidth_mismatch = None
        if self.bitwidth != encoding_dict["compressed_bw"]:
            encoding_mismatch_info.bitwidth_mismatch = (
                self.bitwidth,
                encoding_dict["compressed_bw"],
            )
        if self.decompressed_bw != encoding_dict["bw"]:
            # Possibly overwriting above bitwidth mismatch, but leaving as is to simplify mismatch info instead of adding LPBQ specific field.
            encoding_mismatch_info.bitwidth_mismatch = (
                self.decompressed_bw,
                encoding_dict["bw"],
            )


def _get_symmetric_properties(
    encodings: Dict,
) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """
    Return symmetric properties of the given encodings. If encodings are float, return None for each.

    :param encodings: Encodings to get symmetric properties for
    :return: Tuple of is_symmetric, is_strict_symmetric, and is_unsigned symmetric properties
    """
    # TODO: Move this function into proper encodings module
    if encodings["dtype"] == QuantizationDataType.float.name.upper():
        return None, None, None

    is_symmetric = encodings["is_sym"] is True

    is_strict_symmetric = False
    if is_symmetric and encodings["offset"][0] == -(2 ** (encodings["bw"] - 1)) + 1:
        is_strict_symmetric = True

    # Note: Even if the original quantizer had is_unsigned_symmetric set to True, if any observed values were negative,
    # the resulting encodings will look signed. This logic can only perform a best effort check to return True only if
    # any encoding showed unsigned symmetric properties.
    is_unsigned_symmetric = is_symmetric and encodings["offset"][0] == 0

    return is_symmetric, is_strict_symmetric, is_unsigned_symmetric


# TODO: Move this class into proper encodings module
@dataclass
class _EncodingMismatchInfo:
    """
    Dataclass tracking information about mismatched quantizer vs. encoding settings.
    """

    quantizer_name: str
    enabled_mismatch: Optional[Tuple] = None
    dtype_mismatch: Optional[Tuple] = None
    bitwidth_mismatch: Optional[Tuple] = None
    is_symmetric_mismatch: Optional[Tuple] = None
    is_strict_symmetric_mismatch: Optional[Tuple] = None
    is_unsigned_symmetric_mismatch: Optional[Tuple] = None
    enc_type_mismatch: Optional[Tuple] = None

    def has_mismatch(self) -> bool:
        """
        Returns True if there is a mismatched setting.

        :return: True if there is a mismatched setting, False otherwise
        """
        return (
            self.enabled_mismatch is not None
            or self.dtype_mismatch is not None
            or self.bitwidth_mismatch is not None
            or self.is_symmetric_mismatch is not None
            or self.is_strict_symmetric_mismatch is not None
            or self.is_unsigned_symmetric_mismatch is not None
            or self.enc_type_mismatch is not None
        )
