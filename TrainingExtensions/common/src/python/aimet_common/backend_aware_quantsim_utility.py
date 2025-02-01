#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for backend aware quantization """

# pylint: disable=import-error, no-name-in-module
import logging
from typing import List, Dict
from dataclasses import dataclass
import json
from aimet_torch.translation_mapping import op_to_weight_index_map, backend_datatype_to_aimet_map
from aimet_common.defs import QuantizationDataType
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys
from aimet_common.libpymo import ModelOpDefParser


logger = logging.getLogger()
default_index_to_set_bitwidth_dtype = 0

@dataclass
class SupportedBackendInfo:
    """
    Class which holds backend constraint information
    """
    activation_constraints: List[Dict]
    weights_constraints: List[Dict]

@dataclass
class QuantsimInfo:
    """
    Class which holds info. about quantsim object
    """
    activation_bitwidth: int
    param_bitwidth: int
    data_type: QuantizationDataType


def merge_constraints_from_xmls(op_name: str, supported_backend_info_list: List[SupportedBackendInfo], merged_op_and_supported_backend_info_map: Dict[str, List]):

    """
    Merges backend constraints in supported_backend_info to merged_op_and_supported_backend_info_map

    :param op_name: op_name in model
    :param supported_backend_info_list: List of backend constraints to be merged in merged_op_and_supported_backend_info_map
    :param merged_op_and_supported_backend_info_map: Map which holds info. about op to it's backend constraints
    """
    if op_name in merged_op_and_supported_backend_info_map:
        merged_op_and_supported_backend_info_map[op_name].extend(supported_backend_info_list)
    else:
        merged_op_and_supported_backend_info_map[op_name] = supported_backend_info_list


def get_activation_constraints(parser: ModelOpDefParser, op_name_in_opdef: str, op_index: int) -> List[Dict]:
    """
    Returns activation constraints for the op using parser
    :param parser: ModelOpDefParser object
    :param op_name_in_opdef: Op name whose activation constraints needs to be fetched
    :param op_index: Op's occurrence index in backend XML
    :return: List of activation constraints for the op in {bitwidth, dtype} Dict form
    """

    activation_constraints = []
    try:
        output_datatype_constraints_size = parser.get_size_list(op_name_in_opdef)[op_index]['output_size']
        for output_index in range(output_datatype_constraints_size):
            datatype_constraints = parser.get_output_datatype_list(op_name_in_opdef, output_index)[op_index]
            for datatype in datatype_constraints:
                if backend_datatype_to_aimet_map[datatype] not in activation_constraints:
                    activation_constraints.append(backend_datatype_to_aimet_map[datatype])
    # pylint: disable=bare-except
    except:
        #Parser API will throw appropriate error message if not able to get output datattypes
        pass

    return activation_constraints

def get_weight_constraints(parser: ModelOpDefParser, op_name_in_opdef: str, op_index: int) -> List[Dict]:
    """
    Returns weight constraints for the op using parser
    :param parser: ModelOpDefParser object
    :param op_name_in_opdef: Op name whose weight constraints needs to be fetched
    :param op_index: Op's occurrence index in backend XML
    :return: List of weight constraints for the op in {bitwidth, dtype} Dict form
    """
    weight_constraints = []
    if op_name_in_opdef in op_to_weight_index_map:
        try:
            datatype_constraints = parser.get_input_datatype_list(op_name_in_opdef, op_to_weight_index_map[op_name_in_opdef])[op_index]
            for datatype in datatype_constraints:
                if backend_datatype_to_aimet_map[datatype] not in weight_constraints:
                    weight_constraints.append(backend_datatype_to_aimet_map[datatype])
        # pylint: disable=bare-except
        except:
            #Parser API will throw appropriate error message if not able to get input datatypes
            pass
    return weight_constraints


def get_backend_info(master_opdef_path: str, backend_opdef_paths: List[str]) -> Dict[str, List[SupportedBackendInfo]]:

    """
    Returns backend constraints

    :param master_opdef_path: Master Op. Def. file path
    :param backend_opdef_paths: List of Backend Op. Def. file path
    :return: Dict of op_name and it's constraints
    """
    merged_opname_supported_backend_info_map = {}
    for backend_opdef_path in backend_opdef_paths:
        parser = ModelOpDefParser(master_opdef_path, backend_opdef_path)
        supported_ops_in_backend = parser.get_supported_ops_in_backend()
        op_and_supported_backend_info_map = {}

        for op_name_in_opdef in supported_ops_in_backend:
            num_of_valid_combinations_of_op_in_xml = len(parser.get_size_list(op_name_in_opdef))

            for op_index in range(num_of_valid_combinations_of_op_in_xml):

                activation_constraints = get_activation_constraints(parser, op_name_in_opdef, op_index)
                weight_constraints = get_weight_constraints(parser, op_name_in_opdef, op_index)

                supported_backend_info = SupportedBackendInfo(activation_constraints, weight_constraints)

                if op_name_in_opdef not in op_and_supported_backend_info_map:
                    op_and_supported_backend_info_map[op_name_in_opdef] = []

                op_and_supported_backend_info_map[op_name_in_opdef].append(supported_backend_info)

            if op_name_in_opdef in op_and_supported_backend_info_map:
                merge_constraints_from_xmls(op_name_in_opdef, op_and_supported_backend_info_map[op_name_in_opdef], merged_opname_supported_backend_info_map)

    return merged_opname_supported_backend_info_map

def get_constraint_accrording_to_json_config(constraint: Dict) -> Dict:
    """
    Returns supported kernel constraint according to JSON file
    :param constraint: Activation or Param constraint
    :return: Dict format of suported kernel
    """
    constraint_according_json = {}
    constraint_according_json[ConfigDictKeys.BITWIDTH] = constraint[ConfigDictKeys.BITWIDTH]
    if constraint[ConfigDictKeys.DTYPE] == QuantizationDataType.int:
        constraint_according_json[ConfigDictKeys.DTYPE] = "int"
    elif constraint[ConfigDictKeys.DTYPE] == QuantizationDataType.float:
        constraint_according_json[ConfigDictKeys.DTYPE] = "float"
    return constraint_according_json

# pylint: disable=too-many-branches
def get_supported_kernels_from_backend_info(supported_backend_info_list: List[SupportedBackendInfo]) -> List[Dict]:
    """
    Returns supported for JSON config file from backend constraints

    :param supported_backend_info_list: List of supported backend info. object
    :return: List of supported kernels
    """
    supported_kernels = []
    # pylint: disable=too-many-nested-blocks
    for supported_backend_info in supported_backend_info_list:
        for activation_constraint in supported_backend_info.activation_constraints:
            if activation_constraint[ConfigDictKeys.BITWIDTH] != 64:
                if supported_backend_info.weights_constraints:
                    for weight_constraint in supported_backend_info.weights_constraints:

                        if weight_constraint[ConfigDictKeys.BITWIDTH] != 64 and \
                           (weight_constraint[ConfigDictKeys.DTYPE] == activation_constraint[ConfigDictKeys.DTYPE]):
                            json_act_constraint = get_constraint_accrording_to_json_config(activation_constraint)
                            json_param_constraint = get_constraint_accrording_to_json_config(weight_constraint)

                            supported_kernel = {ConfigDictKeys.ACTIVATION: json_act_constraint,
                                                ConfigDictKeys.PARAM: json_param_constraint}

                            dtype_already_exist = False
                            for kernel in supported_kernels:
                                if json_act_constraint[ConfigDictKeys.BITWIDTH] == kernel[ConfigDictKeys.ACTIVATION][ConfigDictKeys.BITWIDTH] and \
                                   json_act_constraint[ConfigDictKeys.DTYPE] == kernel[ConfigDictKeys.ACTIVATION][ConfigDictKeys.DTYPE] and \
                                   json_param_constraint[ConfigDictKeys.BITWIDTH] == kernel[ConfigDictKeys.PARAM][ConfigDictKeys.BITWIDTH] and \
                                   json_param_constraint[ConfigDictKeys.DTYPE] == kernel[ConfigDictKeys.PARAM][ConfigDictKeys.DTYPE]:

                                    dtype_already_exist = True
                                    break

                            if not dtype_already_exist:
                                supported_kernels.append(supported_kernel)
                else:
                    json_act_constraint = get_constraint_accrording_to_json_config(activation_constraint)
                    supported_kernel = {ConfigDictKeys.ACTIVATION: json_act_constraint}
                    dtype_already_exist = False
                    for kernel in supported_kernels:
                        if json_act_constraint[ConfigDictKeys.BITWIDTH] == kernel[ConfigDictKeys.ACTIVATION][ConfigDictKeys.BITWIDTH] and \
                           json_act_constraint[ConfigDictKeys.DTYPE] == kernel[ConfigDictKeys.ACTIVATION][ConfigDictKeys.DTYPE]:

                            dtype_already_exist = True
                            break

                    if not dtype_already_exist:
                        supported_kernels.append(supported_kernel)

    return supported_kernels

def populate_supported_kernels_in_json_config(master_opdef_file_path: str,
                                              backend_opdef_file_paths: List[str],
                                              json_config_file_path: str):
    """
    Populate supported kernels per op basis in JSON config file

    :param master_opdef_file_path: Master opdef file path
    :param backend_opdef_file_paths: Backend opdef file paths
    :param json_config_file_path: Config file in which supported kernels will be populated
    """
    supported_kernels_dict = get_backend_info(master_opdef_file_path, backend_opdef_file_paths)

    with open(json_config_file_path) as file:
        quantsim_config = json.load(file)

    for backend_op_type, info in supported_kernels_dict.items():
        supported_kernels = get_supported_kernels_from_backend_info(info)
        if supported_kernels:
            if backend_op_type not in quantsim_config[ConfigDictKeys.OP_TYPE]:
                quantsim_config[ConfigDictKeys.OP_TYPE][backend_op_type] = {}

            quantsim_config[ConfigDictKeys.OP_TYPE][backend_op_type][ConfigDictKeys.SUPPORTED_KERNELS] = supported_kernels

    with open(json_config_file_path, 'w') as file:
        json.dump(quantsim_config, file, indent=4)
