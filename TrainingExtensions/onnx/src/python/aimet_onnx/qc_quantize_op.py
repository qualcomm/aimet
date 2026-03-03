# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Custom QcQuantizeOp to quantize weights and activations using ONNXRuntime"""

# pylint: disable=too-many-lines
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Tuple
import numpy as np

from aimet_onnx.common import libpymo
from aimet_onnx.common.defs import (
    QuantScheme,
    MAP_QUANT_SCHEME_TO_PYMO,
    QuantizationDataType,
    EncodingType,
)
from aimet_onnx.common import libquant_info
from aimet_onnx.common.utils import deprecated
from aimet_onnx.common.quantsim import (
    calculate_delta_offset,
    create_encoding_from_min_max,
)
from aimet_onnx import lpbq_utils
from ._encoding import EncodingBase, LPBQEncoding


OpMode = libpymo.TensorQuantizerOpMode


@dataclass
class TensorQuantizerParams:
    """
    Per channel quantization parameters

    Args:
      tensor_shape Shape of the input tensor
      channel_axis Axis along which per channel quantization is performed
      block_axis Axis along which blockwise quantization is performed
    """

    tensor_shape: Tuple[int, ...]
    channel_axis: Optional[int] = None
    block_axis: Optional[int] = None


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

    def _copy(self) -> QcQuantizeOp:
        # pylint: disable=protected-access
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.tensorQuantizerRef = self.quant_info.tensorQuantizerRef
        quant_info.opMode = self.quant_info.opMode
        quant_info.useSymmetricEncoding = self.quant_info.useSymmetricEncoding
        quant_info.enabled = self.quant_info.enabled
        quant_info.isIntDataType = self.quant_info.isIntDataType
        quant_info.usePerChannelMode = self.quant_info.usePerChannelMode
        quant_info.channelAxis = self.quant_info.channelAxis
        quant_info.blockAxis = self.quant_info.blockAxis
        quant_info.blockSize = self.quant_info.blockSize
        quant_info.name = self.quant_info.name
        tensor_quantizer_params = copy.deepcopy(self.tensor_quantizer_params)

        qc_quantize_op = copy.copy(self)
        qc_quantize_op.quant_info = quant_info
        qc_quantize_op.tensor_quantizer_params = tensor_quantizer_params
        qc_quantize_op._tensor_quantizer = qc_quantize_op._build_tensor_quantizer()

        return qc_quantize_op

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
        if self.quant_info.usePerChannelMode == enable:
            return

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

        if block_size == self.quant_info.blockSize:
            return

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
        quantizer.setZeroPointShift(self._tensor_quantizer.getZeroPointShift())

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
        if self.tensor_quantizer_params:
            input_shape = self.tensor_quantizer_params.tensor_shape
            default_channel_axis = self.tensor_quantizer_params.channel_axis
            default_block_axis = self.tensor_quantizer_params.block_axis
        else:
            input_shape = None
            default_channel_axis = None
            default_block_axis = None

        e = EncodingBase.from_qnn_encoding_dict(
            encoding_dict,
            input_shape=input_shape,
            default_channel_axis=default_channel_axis,
            default_block_axis=default_block_axis,
        )

        if isinstance(e, LPBQEncoding) and not isinstance(
            self, GroupedBlockQuantizeDequantize
        ):
            name = encoding_dict.get("name", None)
            if name:
                msg = (
                    f"Loading LPBQ encodings for tensor name {name} "
                    "into a non-LPBQ quantizer is not supported"
                )
            else:
                msg = "Loading LPBQ encodings into a non-LPBQ quantizer"

            raise AssertionError(
                msg
                + " Ensure QuantizationSimModel is set with proper quantizers before loading."
            )

        e.load_to(self)

        if not allow_overwrite:
            self.freeze_encodings()

        self.enabled = True

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        """
        Load pre-existing encodings to the quantizer and sets the op-mode to quantize-dequantize

        :param encoding: The list of libpymo.TfEncoding objects to be used by the C++ op
        """
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

        encodings = self._tensor_quantizer.computeEncodings(
            self.use_symmetric_encodings
        )

        if self._encoding_min_max_fixed_vals:
            min_val, max_val = self._encoding_min_max_fixed_vals

            encodings = [
                create_encoding_from_min_max(
                    e.min if min_val is None else min_val,
                    e.max if max_val is None else max_val,
                    self.bitwidth,
                    self.use_symmetric_encodings,
                    self.use_strict_symmetric,
                )
                for e in encodings
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
        e = EncodingBase.from_quantizer(self)

        if e:
            return e.to_qnn_encoding_dict(encoding_version)

        return None

    def _encoding_type(self):
        if (
            not self.quant_info.usePerChannelMode
            or self.data_type == QuantizationDataType.float
        ):
            return EncodingType.PER_TENSOR
        if not self.quant_info.blockSize:
            return EncodingType.PER_CHANNEL
        return EncodingType.PER_BLOCK

    def update_encoding_stats(self, tensor: np.ndarray):
        """
        Update the stats for computing encodings.

        :param tensor: Tensor to use for updating the encodings stats
        """
        self._tensor_quantizer.updateStats(tensor)

    def quantize_dequantize(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Convert an input tensor from float to quantized int using the computed encodings and back to float.

        :param input_tensor: Input tensor
        :return: quantized-dequantized tensor
        """
        input_tensor = np.ascontiguousarray(input_tensor, dtype=input_tensor.dtype)
        output_tensor = self._tensor_quantizer.quantizeDequantize(input_tensor)
        if output_tensor.shape != input_tensor.shape:
            raise ValueError("Output tensor shape mismatch after quantize-dequantize.")

        return output_tensor

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

    def set_fixed_encoding_range(
        self, fixed_range: Tuple[Optional[float], Optional[float]]
    ):
        """
        Set the min/max values to be used when computing encodings

        :param fixed_range: Tuple of (min, max) value to use in-place of observer statistics when computing encodings
        """
        self._encoding_min_max_fixed_vals = fixed_range

    def set_zero_point_shift(self, zero_point_shift: float = 0.5):
        """
        Set the zero point shift value for the quantizer

        :param zero_point_shift: Zero point shift value
        """
        if not self._is_encoding_frozen:
            self._tensor_quantizer.setZeroPointShift(zero_point_shift)

    def get_zero_point_shift(self) -> float:
        """
        Get the zero point shift value for the quantizer

        :return: Zero point shift value
        """
        return self._tensor_quantizer.getZeroPointShift()

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

    def _merge_constraints(self, other: "QcQuantizeOp") -> None:
        """
        Merge configuration with other QcQuantizeOp.
        """
        # pylint: disable=protected-access
        if self.quant_info.usePerChannelMode != other.quant_info.usePerChannelMode:
            raise RuntimeError("Can't merge per-tensor and per-channel quantizer")

        if (
            self.quant_info.usePerChannelMode
            and self.tensor_quantizer_params
            and other.quant_info.usePerChannelMode
            and other.tensor_quantizer_params
        ):
            if (
                self.tensor_quantizer_params.channel_axis
                != other.tensor_quantizer_params.channel_axis
            ):
                raise RuntimeError(
                    "Can't merge quantizers with different channel axes: "
                    f"{self.tensor_quantizer_params.channel_axis} vs "
                    f"{other.tensor_quantizer_params.channel_axis}"
                )

            if (
                self.tensor_quantizer_params.block_axis
                != other.tensor_quantizer_params.block_axis
            ):
                raise RuntimeError(
                    "Can't merge quantizers with different block axes: "
                    f"{self.tensor_quantizer_params.block_axis} vs "
                    f"{other.tensor_quantizer_params.block_axis}"
                )

            if self.quant_info.blockSize != other.quant_info.blockSize:
                raise RuntimeError(
                    "Can't merge quantizers with different block sizes: "
                    f"{self.quant_info.blockSize} vs {other.quant_info.blockSize}"
                )

        if (
            self._encoding_min_max_fixed_vals
            and other._encoding_min_max_fixed_vals
            and self._encoding_min_max_fixed_vals != other._encoding_min_max_fixed_vals
        ):
            raise RuntimeError(
                "Can't merge quantizers with different fixed ranges: "
                f"{self._encoding_min_max_fixed_vals} vs "
                f"{other._encoding_min_max_fixed_vals}"
            )

        self.bitwidth = min(self.bitwidth, other.bitwidth)
        self.use_symmetric_encodings |= other.use_symmetric_encodings
        self.use_unsigned_symmetric |= other.use_unsigned_symmetric

        fixed_range = (
            self._encoding_min_max_fixed_vals or other._encoding_min_max_fixed_vals
        )

        if self.use_symmetric_encodings and fixed_range:
            _min, _max = fixed_range
            absmax = max(abs(_min), abs(_max))
            fixed_range = (-absmax, absmax)

        self.set_fixed_encoding_range(fixed_range)


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

    def export_encodings(self, encoding_version: str = "0.6.1"):
        """
        Exports the quantizer's encodings in the selected format.

        :param encoding_version: Version string indicated the encoding export format.
        """
        if self._tensor_quantizer.getZeroPointShift() != 0.0:
            raise NotImplementedError(
                "Exporting encodings with shifted zero point is not supported"
            )

        return super().export_encodings(encoding_version)

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

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        block_grouping = self._block_grouping()

        if any(dim != 1 for dim in block_grouping):
            encoding = lpbq_utils.compress_encoding_scales(
                encoding,
                self._encoding_shape(),
                block_grouping,
                scale_bitwidth=self.decompressed_bw - self.bitwidth,
            )
        super().load_encodings(encoding)

    def _encoding_type(self):
        encoding_type = super()._encoding_type()
        if encoding_type == EncodingType.PER_BLOCK:
            return EncodingType.LPBQ
        return encoding_type

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

    def _merge_constraints(self, other: "QcQuantizeOp") -> None:
        """
        Merge configuration with other QcQuantizeOp.
        """
        if not isinstance(other, GroupedBlockQuantizeDequantize):
            raise RuntimeError("Can't merge regular quantizer into LPBQ quantizer")

        if self.bitwidth != other.bitwidth:
            raise RuntimeError(
                "Can't merge LPBQ quantizers with different bitwidths: "
                f"{self.bitwidth} vs {other.bitwidth}"
            )

        if self.decompressed_bw != other.decompressed_bw:
            raise RuntimeError(
                "Can't merge LPBQ quantizers with different decompressed bitwidths: "
                f"{self.decompressed_bw} vs {other.decompressed_bw}"
            )

        return super()._merge_constraints(other)


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
