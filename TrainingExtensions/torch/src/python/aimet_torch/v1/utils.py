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
# pylint: disable = too-many-lines
""" v1-specific utils """

from typing import Dict, Union

import numpy as np
import torch

from aimet_common.utils import AimetLogger, log_with_error_and_assert_if_false
from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_QUANT_SCHEME_TO_PYMO
import aimet_common.libpymo as libpymo
from aimet_torch.v1.tensor_quantizer import TensorQuantizer, StaticGridPerChannelQuantizer, StaticGridPerTensorQuantizer # pylint:disable = cyclic-import

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def compute_encoding_for_given_bitwidth(data: np.ndarray, bitwidth: int, quant_scheme: QuantScheme,
                                        is_symmetric: bool, data_type: QuantizationDataType) -> Dict:
    """
    Return encoding dictionary for given bitwidth
    :param data: Numpy data
    :param bitwidth: bitwidth (4-31) to use for quantizing data
    :param quant_scheme: Quantization scheme
    :param is_symmetric: True if symmetric encodings is used, False otherwise
    :return: Encoding Dictionary
    """
    # Create Encodings Analyzer and collect statistical data to compute encodings
    # Since the data is numpy array and on CPU memory, useCuda is False
    encoding_analyzer = libpymo.EncodingAnalyzerForPython(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme])
    encoding_analyzer.updateStats(data, False)

    encoding, is_encoding_valid = encoding_analyzer.computeEncoding(bitwidth, is_symmetric, False, False)

    if is_encoding_valid:
        return {'min': encoding.min,
                'max': encoding.max,
                'scale': encoding.delta,
                'offset': encoding.offset,
                'bitwidth': encoding.bw,
                'is_symmetric': str(is_symmetric),
                'dtype': 'int' if data_type == QuantizationDataType.int else 'float'}

    return {}


def compute_partial_encoding(quantizer: TensorQuantizer, encoding_dict: Dict) -> Dict:
    """
    Generates the full encoding from partially provided encoding.

    :param quantizer:  Quantizer object for which the encoding needs to be computed.
    :param encoding_dict: Partial Encoding
    :return: Full encoding
    """

    encoding = libpymo.TfEncoding()
    encoding.bw = encoding_dict.get('bitwidth')
    encoding.max = encoding_dict.get('max', 0)
    encoding.min = encoding_dict.get('min', 0)
    encoding.delta = encoding_dict.get('scale', 0)
    encoding.offset = encoding_dict.get('offset', 0)

    if not (encoding.max == 0 and encoding.min == 0) and encoding.delta != 0:
        return encoding_dict

    partial_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF, quantizer.round_mode)
    partial_quantizer.computePartialEncoding(encoding.bw, encoding, quantizer.use_symmetric_encodings,
                                             quantizer.use_unsigned_symmetric, quantizer.use_strict_symmetric)

    encoding_dict['max'] = encoding.max
    encoding_dict['min'] = encoding.min
    encoding_dict['scale'] = encoding.delta
    encoding_dict['offset'] = encoding.offset
    encoding_dict['is_symmetric'] = 'True' if quantizer.use_symmetric_encodings else 'False'

    return encoding_dict


def create_encoding_dict(encoding: libpymo.TfEncoding, quantizer, propagate_encodings: bool) -> Union[Dict, None]:
    """
    Create encoding dictionary from encoding object
    :param encoding: Encoding of the quantizer
    :param quantizer: Tensor Quantizer
    :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
            multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
            ops.
    :return: Encoding Dictionary
    """
    data_type, bitwidth = quantizer.data_type, quantizer.bitwidth

    if data_type == QuantizationDataType.float:
        enc_dict = {'bitwidth': bitwidth, 'dtype': "float"}
    else:
        if encoding:
            if propagate_encodings:
                # Shortened encodings will be filled into a layer that only exists due to expansion of PyTorch ops
                # into multiple ONNX ops so that it's necessarily to use the same bitwidth and type
                enc_dict = {'bitwidth': encoding.bw, 'dtype': "int"}
            else:
                encoding_min, encoding_max, bw, scale, offset = encoding.min, encoding.max, encoding.bw, \
                                                                encoding.delta, encoding.offset
                is_symmetric = quantizer.use_symmetric_encodings

                enc_dict = {'min': encoding_min, 'max': encoding_max, 'scale': scale, 'offset': int(offset),
                            'bitwidth': bw, 'is_symmetric': str(is_symmetric), 'dtype': "int"}
        else:
            enc_dict = None
    return enc_dict


def create_encoding_from_dict(encoding_dict: dict) -> libpymo.TfEncoding:
    """
    Create encoding object from encoding dictionary
    :param encoding_dict: Dictionary containing encodings
    :return: Encoding object, is_symmetric
    """
    encoding = libpymo.TfEncoding()
    encoding.bw = encoding_dict.get('bitwidth')
    encoding.max = encoding_dict.get('max')
    encoding.min = encoding_dict.get('min')
    encoding.delta = encoding_dict.get('scale')
    encoding.offset = encoding_dict.get('offset')
    log_with_error_and_assert_if_false(encoding_dict.get('is_symmetric') in ['True', 'False'],
                                       logger,
                                       f'Unexpected value for is_symmetric: {encoding_dict.get("is_symmetric")}')
    return encoding


def get_per_channel_quantizer_from_per_tensor(quantizer: TensorQuantizer, original_module: torch.nn.Module):
    """ Get PerChannel Quantizer with same settings as given PerTensor Quantizer """
    channel_axis = 0
    if isinstance(original_module, (torch.nn.ConvTranspose1d,
                          torch.nn.ConvTranspose2d,
                          torch.nn.ConvTranspose3d)):
        if len(original_module.weight.shape) > 1:
            channel_axis = 1

    num_channels = original_module.weight.shape[channel_axis]
    use_strict_symmetric = quantizer.use_strict_symmetric
    use_unsigned_symmetric = quantizer.use_unsigned_symmetric
    quantizer = StaticGridPerChannelQuantizer(quantizer.bitwidth, quantizer.round_mode,
                                              quantizer.quant_scheme,
                                              quantizer.use_symmetric_encodings,
                                              num_channels=num_channels,
                                              enabled_by_default=quantizer.enabled,
                                              ch_axis=channel_axis,
                                              data_type=quantizer.data_type)
    quantizer.use_strict_symmetric = use_strict_symmetric
    quantizer.use_unsigned_symmetric = use_unsigned_symmetric
    return quantizer


def get_per_tensor_quantizer_from_per_channel(quantizer: TensorQuantizer):
    """ Get PerTensor Quantizer with same settings as given PerChannel Quantizer """
    use_strict_symmetric = quantizer.use_strict_symmetric
    use_unsigned_symmetric = quantizer.use_unsigned_symmetric
    quantizer = StaticGridPerTensorQuantizer(quantizer.bitwidth, quantizer.round_mode,
                                             quantizer.quant_scheme,
                                             quantizer.use_symmetric_encodings,
                                             enabled_by_default=quantizer.enabled,
                                             data_type=quantizer.data_type)
    quantizer.use_strict_symmetric = use_strict_symmetric
    quantizer.use_unsigned_symmetric = use_unsigned_symmetric
    return quantizer


def _validate_is_symmetric_flag(quantizer: TensorQuantizer, encoding_dict: Dict, strict: bool):
    """
    sub utility of 'validate_is_symmetric_flag'
    """
    if 'is_symmetric' in encoding_dict:
        is_symmetric = encoding_dict['is_symmetric'] == 'True'
        if quantizer.use_symmetric_encodings != is_symmetric:
            # If not strict, raise a warning and override the quantizer
            # setting with provided 'is_symmetric' flag from encoding_dict
            if not strict:
                logger.warning("Using Provided 'is_symmetric' flag in encodings (set to %s) "
                               "which doesn't match with quantizer setting (set to %s), to "
                               "compute partial encodings", is_symmetric, quantizer.use_symmetric_encodings)
            else:
                raise AssertionError("Provided 'is_symmetric' flag in encodings (set to %s) doesn't match with "
                                     "quantizer setting (set to %s)" % (is_symmetric, quantizer.use_symmetric_encodings))
    else:
        raise AttributeError("Provided encoding doesn't have 'is_symmetric' flag")


def validate_is_symmetric_flag(quantizer: TensorQuantizer, encoding_dict: Dict, strict: bool = True):
    """
    Validate 'is_symmetric' flag from encoding_dict with quantizer.use_symmetric_encodings and set the later accordingly
    :param quantizer: Quantizer for which use_symmetric_encodings needs to be validated and set
    :param encoding_dict: encoding_dict from external overrides
    :param strict: flag to decide whether to raise an error or soft warning
    :return:
    """
    if not (encoding_dict.get('max', 0) == 0 and encoding_dict.get('min', 0) == 0) and encoding_dict.get('delta', 0) != 0:
        # In case of full encoding, error out when quantizer setting doesn't match with provided 'is_symmetric' flag
        _validate_is_symmetric_flag(quantizer, encoding_dict, strict=True)

    # In case of partial encodings, use is_symmetric from encodings provided to compute full encoding
    _validate_is_symmetric_flag(quantizer, encoding_dict, strict=strict)
