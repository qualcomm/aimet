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
""" Export utilities for QuantizationSimModel """

import json
import os
from typing import Dict, List, Tuple

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantizationDataType, EncodingType
from aimet_torch.utils import is_vector_encoding


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# TODO: implement direct 1.0.0 export for v1 quantsim so this function can be removed.
def _export_to_1_0_0(path: str,
                     filename_prefix: str,
                     tensor_to_activation_encodings: Dict[str, List],
                     tensor_to_param_encodings: Dict[str, List],
                     tensor_to_quantizer_map: Dict,
                     excluded_layer_names: List[str],
                     quantizer_args: Dict):
    """
    Export encodings using format version 1.0.0.

    :param path: Path to save encodings
    :param filename_prefix: Filename to save encodings with
    :param tensor_to_activation_encodings: Dictionary of activation encodings which maps onnx attribute to encodings
    :param tensor_to_param_encodings: Dictionary of param encodings
    :param tensor_to_quantizer_map: Dictionary mapping tensor names to quantizers
    :param excluded_layer_names: List of names of layers that have been excluded from quantization
    :param quantizer_args: Arguments to top leve quantsim
    """
    activation_encodings = _get_activation_encodings(tensor_to_activation_encodings, tensor_to_quantizer_map)
    param_encodings = _get_param_encodings(tensor_to_param_encodings, tensor_to_quantizer_map)

    encoding_file = {'version': '1.0.0',
                     'activation_encodings': activation_encodings,
                     'param_encodings': param_encodings,
                     'excluded_layers': excluded_layer_names}
    if quantizer_args:
        encoding_file['quantizer_args'] = quantizer_args

    logger.info("Layers excluded from quantization: %s", excluded_layer_names)

    # export weight encodings to output json file
    encoding_file_path = os.path.join(path, filename_prefix + '.encodings')

    with open(encoding_file_path, 'w') as encoding_fp_json:
        json.dump(encoding_file, encoding_fp_json, sort_keys=True, indent=4)


def _get_activation_encodings(tensor_to_activation_encodings: Dict[str, List], tensor_to_quantizer_map: Dict):
    activation_encodings = []
    for tensor, encodings in tensor_to_activation_encodings.items():
        assert tensor in tensor_to_quantizer_map
        if isinstance(encodings, dict) and 'enc_type' in encodings:
            # Already in 1_0_0 format. Simply need to add the name as a key. This will occur if v2 quantsim is exported
            # with encoding_version set to 1.0.0. If v1 quantsim is used, translation from 0.6.1 format to 1.0.0 format
            # is still needed.
            encodings['name'] = tensor
            encoding_dict = encodings
        else:
            assert encodings[0]['dtype'] in {'int', 'float'}
            encoding = encodings[0]
            encoding_dict = {'name': tensor,
                             'dtype': encoding['dtype'].upper(),
                             'enc_type': EncodingType.PER_TENSOR.name,
                             'bw': encoding['bitwidth']}
            if encoding_dict['dtype'] == QuantizationDataType.int.name.upper():
                encoding_dict['is_sym'] = encoding['is_symmetric'] == 'True'
                encoding_dict['scale'] = [encoding['scale']]
                encoding_dict['offset'] = [encoding['offset']]
        activation_encodings.append(encoding_dict)
    return activation_encodings


def _get_param_encodings(tensor_to_param_encodings: Dict[str, List], tensor_to_quantizer_map: Dict):
        # pylint: disable=import-outside-toplevel, cyclic-import
    from aimet_torch.v2.quantization.affine import AffineQuantizerBase

    param_encodings = []
    for tensor, encodings in tensor_to_param_encodings.items():
        assert tensor in tensor_to_quantizer_map
        assert encodings
        if isinstance(encodings, dict) and 'enc_type' in encodings:
            # Already in 1_0_0 format. Simply need to add the name as a key. This will occur if v2 quantsim is exported
            # with encoding_version set to 1.0.0. If v1 quantsim is used, translation from 0.6.1 format to 1.0.0 format
            # is still needed.
            encodings['name'] = tensor
            encoding_dict = encodings
        else:
            assert encodings[0]['dtype'] in {'int', 'float'}
            quantizer = tensor_to_quantizer_map[tensor]
            encoding_dict = {'name': tensor,
                             'dtype': encodings[0]['dtype'].upper(),
                             'bw': encodings[0]['bitwidth']}
            if encoding_dict['dtype'] == QuantizationDataType.float.name.upper():
                encoding_dict['enc_type'] = EncodingType.PER_TENSOR.name
            else:
                encoding_dict['is_sym'] = encodings[0]['is_symmetric'] == 'True'
                encoding_dict['scale'] = [encoding['scale'] for encoding in encodings]
                encoding_dict['offset'] = [encoding['offset'] for encoding in encodings]
                if isinstance(quantizer, AffineQuantizerBase):
                    _handle_v2_quantizer(encoding_dict, encodings, quantizer)
                elif is_vector_encoding(encodings):
                    assert quantizer is None, "Quantizer should be None if encoding is from vector quantization"
                    _handle_vector_encoding(encoding_dict, encodings)
                else:
                    if len(encodings) > 1:
                        encoding_dict['enc_type'] = EncodingType.PER_CHANNEL.name
                    else:
                        encoding_dict['enc_type'] = EncodingType.PER_TENSOR.name
        param_encodings.append(encoding_dict)
    return param_encodings


def _handle_v2_quantizer(encoding_dict: Dict, encodings: List[Dict], quantizer):
    from aimet_torch.v2.quantization.affine import GroupedBlockQuantizeDequantize # pylint: disable=import-outside-toplevel

    # TODO: enhance these checks for robustness in detecting per channel vs. per block vs. grouped block
    if not quantizer.block_size:
        if len(encodings) > 1:
            encoding_dict['enc_type'] = EncodingType.PER_CHANNEL.name
        else:
            encoding_dict['enc_type'] = EncodingType.PER_TENSOR.name
    else:
        if isinstance(quantizer, GroupedBlockQuantizeDequantize):
            if all(group_size == 1 for group_size in quantizer.block_grouping):
                encoding_dict['enc_type'] = EncodingType.PER_BLOCK.name
            else:
                encoding_dict['enc_type'] = EncodingType.LPBQ.name
                encoding_dict['compressed_bw'] = quantizer.bitwidth
                encoding_dict['bw'] = quantizer.decompressed_bw
                encoding_dict['scale'] = quantizer.get_per_channel_scale().flatten().tolist()
                encoding_dict['offset'] = \
                    [-2 ** (quantizer.decompressed_bw - 1) for _ in encoding_dict['scale']]
                encoding_dict['per_block_int_scale'] = quantizer.get_per_block_integer_scale().flatten().tolist()
        else:
            encoding_dict['enc_type'] = EncodingType.PER_BLOCK.name
        encoding_dict['block_size'] = _get_block_size(quantizer.block_size)

def _get_block_size(block_size: Tuple):
    assert len(block_size) >= 2
    for dim_block_size in block_size:
        if dim_block_size != 1:
            return dim_block_size
    return block_size[1]


def _handle_vector_encoding(encoding_dict: Dict, encodings: List[Dict]):
    """
    Update encoding dictionary if encodings are from Vector Quantization

    :param encoding_dict: Dictionary to store parameter encoding
    :param encodings: List of encoding dictionary
    """
    encoding = encodings[0]

    encoding_dict["enc_type"] = EncodingType.VECTOR.name
    encoding_dict["rows_per_block"] = encoding["rows_per_block"]
    encoding_dict["cols_per_block"] = encoding["cols_per_block"]
    encoding_dict["vector_dim"] = encoding["vector_dim"]
    encoding_dict["vector_stride"] = encoding["vector_stride"]
    encoding_dict["index_bw"] = encoding["index_bw"]
