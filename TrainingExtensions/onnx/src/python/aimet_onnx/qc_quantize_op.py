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
""" Custom QcQuantizeOp to quantize weights and activations using ONNXRuntime """

from typing import Union, List, Optional, Dict, Tuple
import numpy as np

import aimet_common.libpymo as libpymo
from aimet_common.libpymo import TensorQuantizerOpMode
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantizationDataType, EncodingType
from aimet_common import libquant_info
from aimet_common.utils import deprecated
from aimet_common.quantsim import calculate_delta_offset, create_encoding_from_min_max
from aimet_onnx import lpbq_utils


OpMode = TensorQuantizerOpMode


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
    """ A custom quantization operation to perform using ONNXRuntime """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, quant_info: libquant_info.QcQuantizeInfo,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest',
                 op_mode: Union[OpMode, None] = None,
                 bitwidth: int = 8, use_symmetric_encodings: bool = False,
                 tensor_quantizer_params: Union[TensorQuantizerParams, None] = None):
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
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self._is_encoding_frozen = False
        self._tensor_quantizer = None
        self._set_tensor_quantizer([self._build_tensor_quantizer()])
        self.op_mode = op_mode
        self.bitwidth = bitwidth
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = True
        self._data_type = QuantizationDataType.int
        self.tensor_quantizer_params = tensor_quantizer_params
        self._reset_encodings()
        self._encoding_min_max_fixed_vals = None

    def is_encoding_frozen(self) -> bool:
        """ Returns is_encoding_frozen var """
        return self._is_encoding_frozen

    def freeze_encodings(self):
        """ Sets encodings to frozen """
        self._is_encoding_frozen = True

    def enable_per_channel_quantization(self):
        """
        Enables per channel quantization for qc_quantize_op
        """
        assert self.tensor_quantizer_params is not None
        assert self.tensor_quantizer_params.channel_axis is not None
        channel_axis = self.tensor_quantizer_params.channel_axis
        self.quant_info.usePerChannelMode = True
        num_channels = self.tensor_quantizer_params.tensor_shape[channel_axis]
        self._create_tensor_quantizers(num_channels)
        self.quant_info.channelAxis = channel_axis if channel_axis >= 0 else channel_axis + len(self.tensor_quantizer_params.tensor_shape)

    def _create_tensor_quantizers(self, num: int):
        tensor_quantizers = []
        for _ in range(num):
            tensor_quantizer = self._build_tensor_quantizer()
            tensor_quantizer.setStrictSymmetric(self.use_strict_symmetric)
            tensor_quantizer.setUnsignedSymmetric(self.use_unsigned_symmetric)
            tensor_quantizer.isEncodingValid = False
            tensor_quantizers.append(tensor_quantizer)

        self._set_tensor_quantizer(tensor_quantizers)

        self.reset_encoding_stats()

    def _enable_blockwise_quantization(self, block_size):
        assert self.tensor_quantizer_params is not None
        tensor_shape = self.tensor_quantizer_params.tensor_shape
        block_axis = self.tensor_quantizer_params.block_axis
        channel_axis = self.tensor_quantizer_params.channel_axis
        assert channel_axis is not None
        assert block_axis is not None
        assert block_axis != channel_axis

        num_quantizers = tensor_shape[channel_axis]

        if block_size != 0:
            if tensor_shape[block_axis] % block_size != 0:
                raise ValueError(
                    f"Input shape {tensor_shape} not divisible by block size {block_size} at axis {block_axis}")
            num_quantizers *= tensor_shape[block_axis] // block_size

        self._create_tensor_quantizers(num_quantizers)
        self.quant_info.usePerChannelMode = True
        self.quant_info.channelAxis = channel_axis if channel_axis >= 0 else channel_axis + len(tensor_shape)
        self.quant_info.blockAxis = block_axis if block_axis >= 0 else block_axis + len(tensor_shape)
        self.quant_info.blockSize = block_size

    @property
    def data_type(self) -> QuantizationDataType:
        """
        Returns the data type for quantization

        :return: Quantization data type
        """
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: QuantizationDataType):
        """
        Sets the quantization data type field in the op and sets isIntDataType inside quantizer_info to true or false
        based on the data type

        :param data_type: Quantization data type
        """
        if not self._is_encoding_frozen:
            self._data_type = data_type
            self.quant_info.isIntDataType = False
            if data_type == QuantizationDataType.int:
                self.quant_info.isIntDataType = True

    def _build_tensor_quantizer(self):
        return libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self.quant_scheme],
                                       MAP_ROUND_MODE_TO_PYMO[self.rounding_mode])

    def _set_tensor_quantizer(self, tensor_quantizers: List[libpymo.TensorQuantizer]):
        """
        Stores tensor_quantizer in self._tensor_quantizer and passes a pointer to the object
        to the C++ op's QcQuantInfo object
        :param tensor_quantizers: The list of libpymo.TensorQuantizer objects to give to the C++ op
        """
        self._tensor_quantizer = tensor_quantizers
        self.quant_info.tensorQuantizerRef = [libpymo.PtrToInt64(tensor_quantizer) for tensor_quantizer in tensor_quantizers]

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
        return self._tensor_quantizer[0].getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._reset_encodings()

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer[0].getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._reset_encodings()

    def get_encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Reads the encodings object from the node's QcQuantizeInfo

        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        if not self.is_initialized() or self.data_type == QuantizationDataType.float:
            return None
        return self.quant_info.encoding

    @property
    @deprecated(f"Use {get_encodings.__qualname__} instead")
    def encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """Deprecated. Use :meth:`get_encodings` to set the quantizer encodings.

        Reads the encodings object from the node's QcQuantizeInfo

        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        return self.get_encodings()

    def update_quantizer_and_load_encodings(self, encoding: List[libpymo.TfEncoding], is_symmetric: Optional[bool],
                                            is_strict_symmetric: Optional[bool], is_unsigned_symmetric: Optional[bool],
                                            data_type: QuantizationDataType):
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

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        """
        Load pre-existing encodings to the quantizer and sets the op-mode to quantize-dequantize

        :param encoding: The list of libpymo.TfEncoding objects to be used by the C++ op
        """
        assert isinstance(encoding, (list, tuple))
        assert len(encoding) == len(self._tensor_quantizer)
        if self.data_type == QuantizationDataType.float:
            raise RuntimeError(f"{type(self).load_encodings.__qualname__} is not supported for floating-point quantizers.")
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.isEncodingValid = True
        self.op_mode = OpMode.quantizeDequantize
        # pylint: disable=attribute-defined-outside-init
        self._encoding = encoding
        self.quant_info.encoding = encoding

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
            for tensor_quantizer in self._tensor_quantizer:
                tensor_quantizer.resetEncodingStats()
            self._reset_encodings()

    def _reset_encodings(self):
        """
        Resets the quantizer's encodings
        """
        if self.is_encoding_frozen():
            return

        encodings = []
        for tensor_quantizer in self._tensor_quantizer:
            encoding = libpymo.TfEncoding()
            encoding.bw = self.bitwidth
            encodings.append(encoding)
            tensor_quantizer.isEncodingValid = False

        self._encoding = encodings
        self.quant_info.encoding = encodings

    def set_bitwidth(self, bitwidth: int):
        """
        Set bitwidth for quantization
        """
        if not self._is_encoding_frozen:
            self.bitwidth = bitwidth
            self._reset_encodings()

    def set_quant_scheme(self, quant_scheme: QuantScheme):
        """
        Set QcQuantizeOp as given quant scheme
        """
        self.quant_scheme = quant_scheme
        if self.quant_info.usePerChannelMode:
            self.enable_per_channel_quantization()
        else:
            self._set_tensor_quantizer([self._build_tensor_quantizer()])
        self.reset_encoding_stats()

    def compute_encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Compute and return encodings of each tensor quantizer
        """
        if self._is_encoding_frozen:
            return None

        if not self.enabled:
            return None

        encodings = []
        for tensor_quantizer in self._tensor_quantizer:
            if self._encoding_min_max_fixed_vals is None:
                encodings.append(tensor_quantizer.computeEncoding(self.bitwidth, self.use_symmetric_encodings))
            else:
                min_val, max_val = self._encoding_min_max_fixed_vals
                encodings.append(create_encoding_from_min_max(min_val, max_val, self.bitwidth, self.use_symmetric_encodings,
                                                              self.use_strict_symmetric))

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
            raise RuntimeError("get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced.")

        if not self.get_encodings():
            raise RuntimeError("get_stats_histogram() can be invoked only when encoding is computed.")

        histogram = []
        for tensor_quantizer in self._tensor_quantizer:
            histogram.append(tensor_quantizer.getStatsHistogram())

        return histogram

    def is_initialized(self) -> bool:
        """
        Returns True if all quantizers have been initialized, False otherwise
        """
        if self.data_type == QuantizationDataType.float:
            # Fp16 quantizers do not need to be initialized
            return True

        if all(tq.isEncodingValid for tq in self._tensor_quantizer):
            return True

        return False

    def export_encodings(self, encoding_version: str = "0.6.1"):
        """
        Exports the quantizer's encodings in the selected format.

        :param encoding_version: Version string indicated the encoding export format.
        """
        if encoding_version == '0.6.1':
            return self._export_legacy_encodings()

        if encoding_version == "1.0.0":
            return self._export_1_0_0_encodings()

        raise RuntimeError(f"Unsupported encoding export version: {encoding_version}")

    def _export_legacy_encodings(self) -> Union[List, None]:
        """
        Create encoding dictionary from encoding object

        :return: List of encoding dictionaries in 0.6.1 encoding format
        """
        if not self.enabled or not self.is_initialized():
            return None

        if self.data_type == QuantizationDataType.float:
            return [{'bitwidth': self.bitwidth, 'dtype': "float"}]

        if self.data_type == QuantizationDataType.int:
            encodings = []
            for encoding in self.get_encodings():
                enc_dict = dict(min=encoding.min,
                                max=encoding.max,
                                scale=encoding.delta,
                                offset=int(encoding.offset),
                                bitwidth=encoding.bw,
                                is_symmetric=str(self.use_symmetric_encodings),
                                dtype="int")
                encodings.append(enc_dict)
            return encodings

        raise RuntimeError(f"Exporting data type {self.data_type} not supported")

    def _encoding_type(self):
        if not self.quant_info.usePerChannelMode:
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

        enc_dict = dict(enc_type=self._encoding_type().name,
                        dtype="INT" if self.data_type == QuantizationDataType.int else "FLOAT",
                        bw=self.bitwidth,
                        )

        if self.data_type == QuantizationDataType.int:
            enc_dict["is_sym"] = self.use_symmetric_encodings
            encodings = self.get_encodings()
            enc_dict["scale"] = [enc.delta for enc in encodings]
            enc_dict["offset"] = [enc.offset for enc in encodings]
            if self.quant_info.blockSize > 0:
                enc_dict["block_size"] = self.quant_info.blockSize

        return enc_dict

    def update_encoding_stats(self, tensor: np.ndarray):
        """
        Update the stats for computing encodings.

        :param tensor: Tensor to use for updating the encodings stats
        """
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.updateStats(tensor, False)

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
                delta, offset = calculate_delta_offset(min_val=tensor[0], max_val=tensor[1], bitwidth=self.bitwidth,
                                                       use_symmetric_encodings=self.use_symmetric_encodings,
                                                       use_strict_symmetric=self.use_strict_symmetric)
                encoding.min = tensor[0]
                encoding.max = tensor[1]
                encoding.delta = delta
                encoding.offset = offset

                is_clipped = True

        return is_clipped

    def set_fixed_encoding_range(self, fixed_range: Tuple[float, float]):
        """
        Set the min/max values to be used when computing encodings

        :param fixed_range: Tuple of (min, max) value to use in-place of observer statistics when computing encodings
        """
        self._encoding_min_max_fixed_vals = fixed_range


class GroupedBlockQuantizeDequantize(QcQuantizeOp):
    """ Class for performing Grouped Block Quantize Dequantize """

    def __init__(self,
                 quant_info: libquant_info.QcQuantizeInfo,
                 bitwidth: int,
                 decompressed_bw: int,
                 block_size: int,
                 quant_scheme: QuantScheme,
                 op_mode: OpMode,
                 tensor_quantizer_params: TensorQuantizerParams):
        if block_size and tensor_quantizer_params.tensor_shape[tensor_quantizer_params.block_axis] % block_size != 0:
            raise ValueError(f"Input shape {tensor_quantizer_params.tensor_shape} is not divisible by block size "
                             f"{block_size} at axis {tensor_quantizer_params.block_axis}")
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

    def _encoding_shape(self):
        tensor_shape = self.tensor_quantizer_params.tensor_shape
        shape = [1 for _ in range(len(tensor_shape))]

        if not self.quant_info.usePerChannelMode:
            return shape

        ch_axis, block_axis = self.quant_info.channelAxis, self.quant_info.blockAxis
        assert ch_axis >= 0

        shape[ch_axis] = tensor_shape[ch_axis]

        if self.quant_info.blockSize > 0:
            assert block_axis >= 0
            shape[block_axis] = tensor_shape[block_axis] // self.quant_info.blockSize

        return shape

    def _block_grouping(self):
        grouping = [1 for _ in range(len(self._encoding_shape()))]
        if self.quant_info.blockSize > 0 and self.quant_info.blockAxis >= 0:
            grouping[self.quant_info.blockAxis] = -1

        return grouping

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        encoding = lpbq_utils.compress_encoding_scales(encoding, self._encoding_shape(), self._block_grouping(),
                                                        scale_bitwidth=self.decompressed_bw - self.bitwidth)
        super().load_encodings(encoding)

    def _export_legacy_encodings(self) -> Union[List, None]:
        raise NotImplementedError(f"0.6.1 encoding format is not supported for {type(self).__qualname__}. Please export "
                                  f"using 1.0.0 format instead.")

    def _encoding_type(self):
        encoding_type = super()._encoding_type()
        if encoding_type == EncodingType.PER_BLOCK:
            return EncodingType.LPBQ
        return encoding_type

    def _export_1_0_0_encodings(self) -> Optional[Dict]:
        encodings = super()._export_1_0_0_encodings()
        if not encodings:
            return None
        if "block_size" not in encodings.keys():
            return encodings

        encodings["compressed_bw"] = self.bitwidth
        encodings["bw"] = self.decompressed_bw
        scale, _ = lpbq_utils.encodings_to_scale_offset_arrays(self.get_encodings(), self._encoding_shape())
        int_scale, per_block_scale = lpbq_utils.grouped_dynamic_quantize(scale, self._block_grouping(), self.decompressed_bw - self.bitwidth)
        encodings['per_block_int_scale'] = int_scale.flatten().tolist()
        encodings['scale'] = per_block_scale.flatten().tolist()
        encodings["offset"] = [-2 ** (self.decompressed_bw - 1) for _ in encodings['scale']]

        return encodings
