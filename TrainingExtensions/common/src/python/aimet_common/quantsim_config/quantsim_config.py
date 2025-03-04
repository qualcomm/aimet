# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Utilities for parsing and applying quantsim configurations from json config file """

from abc import ABC, abstractmethod
import os
from typing import Dict, List, Optional
from aimet_common.defs import QuantizationDataType, QuantDtypeBwInfo
from aimet_common.connected_graph.operation import Op
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.quantsim_config.json_config_importer import JsonConfigImporter, ConfigDictKeys, DefaultsType, \
    ParamType, OpTypeType, SupergroupType, ConfigType, OpType
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# --------------------------------------------------------------------------------------------------------------------
# Overriding AIMET QuantSim data type and bit-width using supported_kernels specified in target driven config file.
# --------------------------------------------------------------------------------------------------------------------
# supported_kernels can be specified at default as well as op level in a given target specific config file
# Example rule in the target specific config file is as below:
# "supported_kernels": [
#                     {
#                         "activation": {
#                             "bitwidth": 16,
#                             "dtype": "int"
#                         },
#                         "param": {
#                             "bitwidth": 16,
#                             "dtype": "int"
#                         }
#                     },
#                     {
#                         "activation": {
#                             "bitwidth": 16,
#                             "dtype": "float"
#                         },
#                         "param": {
#                             "bitwidth": 16,
#                             "dtype": "float"
#                         }
#                     }
#                 ]
# supported_kernels includes data type and bit-width options for activation and param quantization
# applied together as a pair. In above rule act and param can be set to [int16, int16] OR [FP16, FP16]
# supported_kernels can be used to enforce target driven data type and bit-width during AIMET Quantsim
# by setting: ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
#
# AIMET Quantsim is created with specific defaults for data type/ bit-width using:
# default_data_type default_output_bw and default_param_bw arguments as below :
# sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
#                            config_file='./data/quantsim_config.json',
#                            dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
#                            default_data_type=QuantizationDataType.int,
#                            default_output_bw=8, default_param_bw=8)
# Rules for override :
# (i) If a given QuantSim default data type and bit-width is found at either the default or op-level
# supported_kernels list : override shall NOT be applied.
# (ii) AIMET supports overrides ONLY when a lower precision kernel is unavailable.
# For example :
# a) QuantSim default set to int 8, op level supported_kernels only has FP 16 available  --> override supported
# b) QuantSim default set to int 8, op level supported_kernels only has int 4 available  --> override NOT supported
#
# --------------------------------------------------------------------------------------------------------------------

# Flag to enforce target configs for data type and bit-width for params and activation.
ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX = 0


class SupergroupConfigCallback(ABC):
    """ Class acting as a callback for when supergroups are found """
    @abstractmethod
    def __call__(self, _, op_list: List[Op]):
        """ Callback logic """


class OnnxConnectedGraphTypeMapper:
    """
    Class maintaining dictionaries for two way mapping from onnx types to connected graph types
    """
    def __init__(self, type_pairs: List[List[List[str]]]):
        self._onnx_to_conn_graph_dict = {}
        self._conn_graph_to_onnx_dict = {}
        for onnx_types, conn_graph_types in type_pairs:
            for onnx_type in onnx_types:
                self._onnx_to_conn_graph_dict[onnx_type] = conn_graph_types
            for conn_graph_type in conn_graph_types:
                self._conn_graph_to_onnx_dict[conn_graph_type] = onnx_types

    def get_conn_graph_type_from_onnx_type(self, onnx_type: str):
        """
        Return connected graph type corresponding to onnx type
        :param onnx_type: Onnx type to find corresponding connected graph type
        :return: Connected graph type corresponding to onnx_type
        """
        return self._onnx_to_conn_graph_dict.get(onnx_type)

    def get_onnx_type_from_conn_graph_type(self, conn_graph_type: str):
        """
        Return onnx type corresponding to connected graph type
        :param conn_graph_type: Connected graph type to find corresponding onnx type
        :return: Onnx type corresponding to conn_graph_type
        """
        return self._conn_graph_to_onnx_dict.get(conn_graph_type)


_config_file_aliases = {
    "default": "default_config_per_channel.json",
    "htp_v66": "htp_quantsim_config_v66.json",
    "htp_v68": "htp_quantsim_config_v68.json",
    "htp_v69": "htp_quantsim_config_v69.json",
    "htp_v73": "htp_quantsim_config_v73.json",
    "htp_v75": "htp_quantsim_config_v75.json",
    "htp_v79": "htp_quantsim_config_v79.json",
    "htp_v81": "htp_quantsim_config_v81.json",
}

def _get_config_file(alias_or_filename: Optional[str]) -> str:
    if alias_or_filename is None:
        alias_or_filename = "default"

    if os.path.exists(alias_or_filename):
        return os.path.abspath(alias_or_filename)

    config_file = _config_file_aliases.get(alias_or_filename, alias_or_filename)
    config_file = os.path.join(os.path.dirname(__file__), config_file)

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found: {alias_or_filename}. " +
            "Expected a file path or an alias to the standard config files.\n" +
            "Config file aliases consist of:\n" +
            "\n".join([
                f"  * \"{key}\": {value}"
                for key, value in _config_file_aliases.items()
            ])
        )

    return config_file


class QuantSimConfigurator(ABC):
    """ Class for parsing and applying quantsim configurations from json config file """

    def __init__(self,
                 config_file: Optional[str],
                 default_data_type: QuantizationDataType,
                 default_output_bw: int,
                 default_param_bw: int):
        config_file = _get_config_file(config_file)

        self._quantsim_configs = JsonConfigImporter.import_json_config_file(config_file)
        self._supported_kernels = {}
        self._default_data_type = default_data_type
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw

    def _set_quantsim_configs(self):
        """
        Apply quantsim configurations to the given model
        """
        self._set_default_configs(self._quantsim_configs[ConfigDictKeys.DEFAULTS])
        self._set_param_configs(self._quantsim_configs[ConfigDictKeys.PARAMS])
        self._set_op_type_configs(self._quantsim_configs[ConfigDictKeys.OP_TYPE])
        self._set_supergroup_configs(self._quantsim_configs[ConfigDictKeys.SUPERGROUPS])
        self._set_model_input_configs(self._quantsim_configs[ConfigDictKeys.MODEL_INPUT])
        self._set_model_output_configs(self._quantsim_configs[ConfigDictKeys.MODEL_OUTPUT])

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def _get_hw_version(self) -> str:
        """
        Return the hardware version from the config file if present, else return the default value
        """
        return self._quantsim_configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.HW_VERSION, 'default')

    @abstractmethod
    def _generate_and_apply_op_instance_specific_config(self):
        """
        Generate op instance specific configurations - currently supported_kernels and per_channel_quantization fields
        This function uses op specific supported_kernels (if absent use defaults), op specific per_channel_quantization
        fields (if absent use default per_channel_quantization) and generate op instance specific config
        :return: {op_instance_name, op_specific_config}
        """

    # pylint: disable=no-member
    def get_module_names(self):
        """
        :return: List of unique module names present in wrapper dict
        """
        modules = set([])
        for module in self._module_to_quantsim_wrapper_dict.keys():
            modules.add(module.__class__.__name__)
        return list(modules)

    def _parse_supported_kernels(self) -> Dict:
        """
        Parse the layer overrides present in default and op_type section of the config file
        :return: Dict of the supported_kernels parsed from the config file
        """

        supported_kernels = {}
        if ConfigDictKeys.SUPPORTED_KERNELS in self._quantsim_configs[ConfigDictKeys.DEFAULTS]:
            default_supported_kernels = self._quantsim_configs[ConfigDictKeys.DEFAULTS][ConfigDictKeys.SUPPORTED_KERNELS]
            if default_supported_kernels:
                supported_kernels[ConfigDictKeys.DEFAULTS] = default_supported_kernels
            else:
                # TODO make this an assert
                logger.debug("supported_kernels is a required argument in the defaults section")
        else:
            supported_kernels[ConfigDictKeys.DEFAULTS] = {}

        op_type_configs = self._quantsim_configs[ConfigDictKeys.OP_TYPE]
        for op_type in op_type_configs:
            if ConfigDictKeys.SUPPORTED_KERNELS in op_type_configs[op_type]:
                op_type_supported_kernels = op_type_configs[op_type][ConfigDictKeys.SUPPORTED_KERNELS]
                if op_type_supported_kernels:
                    supported_kernels[op_type] = op_type_supported_kernels

        return supported_kernels

    def _parse_per_channel_quantization(self) -> Dict:
        """
        Parse the per_channel_quantization fields in "defaults" and "op_type" sections of the config file
        If "defaults" entry does not exist, assume it to be False
        :return: Dict of per_channel_quantization entries in the config file
        """
        per_channel_quantization = {ConfigDictKeys.DEFAULTS: self._quantsim_configs[ConfigDictKeys.DEFAULTS].get(
            ConfigDictKeys.PER_CHANNEL_QUANTIZATION, False)}

        for op_type in self._quantsim_configs[ConfigDictKeys.OP_TYPE].keys():
            if ConfigDictKeys.PER_CHANNEL_QUANTIZATION in self._quantsim_configs[ConfigDictKeys.OP_TYPE][op_type]:
                per_channel_quantization[op_type] = \
                    self._quantsim_configs[ConfigDictKeys.OP_TYPE][op_type][ConfigDictKeys.PER_CHANNEL_QUANTIZATION]

        return per_channel_quantization

    def check_correctness_of_dtype_bw_rules(self, quantsim_dtype_bw_info: QuantDtypeBwInfo):
        """
        Validates correctness of data type and bitwidth rules specified using config file supported_kernels option.
        :param quantsim_dtype_bw_info: data type (int or float) as QuantizationDataType and act/param bit-width info.
        :return:
        """
        # validation rules:
        # AIMET supports overrides ONLY when a lower precision kernel is unavailable.
        # for example :
        # 1) (default) int 8, but only FP16 kernel is available for a given op type --> override supported
        # 2) (default) int 8, but only int 4 kernel is available is available for a given op type --> override not supported

        default_config = self._quantsim_configs[ConfigDictKeys.DEFAULTS]
        default_valid = False
        op_level_valid = False

        # user has provided default supported kernel options
        if ConfigDictKeys.SUPPORTED_KERNELS in default_config:
            default_supported_kernels = default_config[ConfigDictKeys.SUPPORTED_KERNELS]
            # quantsim dtype/bw found in default supported kernels
            if current_config_in_supported_kernels(quantsim_dtype_bw_info, default_supported_kernels) and \
                    is_current_config_same_as_override_option(quantsim_dtype_bw_info, default_supported_kernels):
                default_valid = True
                # default level override is not required
                logger.info("Quantsim config found in default supported kernels, "
                            "skipping default level dtype and bitwidth override")

            else:
                # override is required, first validate the override option
                # if valid, update default dtype, bw to be used to validate op level overrides.
                if is_override_dtype_bw_valid(get_override_from_supported_kernels(default_supported_kernels),
                                              quantsim_dtype_bw_info):
                    default_valid = True
                    quantsim_dtype_bw_info = get_override_from_supported_kernels(default_supported_kernels)

                else:
                    error_msg = (f'Default supported_kernels override check failed, one way to rectify is to include \n'
                                 f'default quantsim candidate: {quantsim_dtype_bw_info}\n'
                                 f'in supported_kernels list under default section of target specific config file \n')
                    logger.error(error_msg)
                    raise NotImplementedError(error_msg)
        else:
            # user has not provided default supported_kernels, log quantsim defaults treated as default target kernel support
            default_valid = True
            logger.info(' Default supported_kernels not specified in given target specific config file.\n'
                        ' Using default quantsim candidate: %s as default target support', quantsim_dtype_bw_info)

        # in either case, validate op level override options
        if self._quantsim_configs[ConfigDictKeys.OP_TYPE]:
            op_level_valid = validate_all_op_level_dtype_bw_overrides(self._quantsim_configs[ConfigDictKeys.OP_TYPE],
                                                                      quantsim_dtype_bw_info)

        return default_valid and op_level_valid

    def _override_default_dtype_bw_act_param(self, default_configs: Dict):
        """
        Applies dtype/bw overrides specified as default supported kernels in config file in place of quantsim defaults.
        (first level of specificity in configuration file).
        :param default_configs: Default configurations for quantizers.
        :return:
        """

        if is_current_config_same_as_override_option(QuantDtypeBwInfo(self._default_data_type, self._default_output_bw,
                                                                      self._default_data_type, self._default_param_bw),
                                                     default_configs[ConfigDictKeys.SUPPORTED_KERNELS]):
            logger.info('Default quantsim config is same as override option at override index {%s}. Skipping override',
                        DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX)
            return

        config_file_default_act_bw_dtype_config = \
            default_configs[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][
                ConfigDictKeys.ACTIVATION]
        config_file_default_param_bw_dtype_config = \
            default_configs[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][ConfigDictKeys.PARAM]

        # keep the updated with defaults, to be used later to apply op level config
        self._default_data_type = config_file_default_act_bw_dtype_config[ConfigDictKeys.DTYPE]
        self._default_output_bw = config_file_default_act_bw_dtype_config[ConfigDictKeys.BITWIDTH]
        self._default_param_bw = config_file_default_param_bw_dtype_config[ConfigDictKeys.BITWIDTH]

        logger.info('Target default supported kernel at override index {%s} is different from quantsim defaults.'
                    ' Enforcing target kernel config for act bw = {%s} and dtype = {%s}',
                    DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX,
                    self._default_output_bw,
                    self._default_data_type)

        self._override_default_act_bw_dtype(self._default_data_type, self._default_output_bw)

        logger.info('Target default supported kernel at override index {%s} is different from quantsim defaults.'
                    ' Enforcing target kernel config for param bw = {%s} and dtype = {%s}',
                    DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX,
                    self._default_param_bw,
                    self._default_data_type)

        # override param config defaults.
        self._override_default_param_bw_dtype(self._default_data_type, self._default_param_bw)

    def _apply_overrides_for_op(self, op_config: OpType, quantizer_data):
        """
        Applies op level overrides based on supported_kernels specified for a given module.
        :param op_config: Op level configs from config file.
        :param quantizer_data: Op level quantizer that needs to be updated.
        :return:
        """
        # ---------------------------------- [ op level overrides ] ---------------------------------------- #
        # AIMET applies op level override only when a lower precision kernel is not supported for a given op.
        # For example, if quantsim is created with a default act: int8, param: int8 quantization
        # And, if some op has only FP16 kernel support on a given target, this can be specified under
        # op level config using supported_kernels option.
        # It must be noted that, AIMET shall not perform conversions ops (such as int8 -> FP16) that is done on-target,
        # as there is no noise to be simulated while going from lower precision to higher precision.
        # This means, if an op supports only higher precision kernel, there are two ways it is handled during override :
        # a. op has params : param quantizers are updated to higher precision (ex: FP 16 in this case),
        # No override applied to in/out tensor quantizers.
        # b. op has no params : no override.
        # --------------------------------------------------------------------------------------------------- #

        quant_dtype_bw_info = QuantDtypeBwInfo(self._default_data_type, self._default_output_bw, self._default_data_type,
                                               self._default_param_bw)
        if not current_config_in_supported_kernels(quant_dtype_bw_info, op_config[ConfigDictKeys.SUPPORTED_KERNELS]):
            act_bw = op_config[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][
                ConfigDictKeys.ACTIVATION][ConfigDictKeys.BITWIDTH]
            act_dtype = op_config[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][
                ConfigDictKeys.ACTIVATION][ConfigDictKeys.DTYPE]
            if ConfigDictKeys.PARAM in op_config[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX]:
                param_bw = op_config[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][
                    ConfigDictKeys.PARAM][ConfigDictKeys.BITWIDTH]
                param_dtype = op_config[ConfigDictKeys.SUPPORTED_KERNELS][DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][
                    ConfigDictKeys.PARAM][ConfigDictKeys.DTYPE]
                self._override_param_bw_dtype(quantizer_data, param_dtype, param_bw)
                logger.info('Enforcing target kernel config for param bw/dtype = %s/%s',
                            param_bw, param_dtype)
            logger.info('Enforcing target kernel config for activation bw/dtype = %s/%s',
                        act_bw, act_dtype)
            self._override_act_bw_dtype(quantizer_data, act_dtype, act_bw)


    @property
    def quantsim_configs(self):
        """ Returns quant sim config"""
        return self._quantsim_configs

    @property
    def default_output_bw(self):
        """ Returns output bit-width"""
        return self._default_output_bw

    @property
    def default_param_bw(self):
        """ Returns parameter bit-width"""
        return self._default_param_bw

    @property
    def default_data_type(self):
        """ Returns default datatype"""
        return self._default_data_type

    @abstractmethod
    def _override_param_bw_dtype(self, quantizer_data, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for param quantizers of given data
        :param quantizer_data: object containing which param override will be applied to
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """

    @abstractmethod
    def _override_act_bw_dtype(self, quantizer_data, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for activation quantizers of given data
        :param quantizer_data: object containing which activation override will be applied to
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """

    @abstractmethod
    def _override_default_act_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bw default config for input/output quantizers.
        :param data_type: data type as QuantizationDataType
        :param bitwidth: bitwidth to be configured
        :return:
        """

    @abstractmethod
    def _override_default_param_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for param quantizers
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """

    @abstractmethod
    def _set_default_configs(self, default_configs: DefaultsType):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """

    @abstractmethod
    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """

    @abstractmethod
    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """

    def _build_supergroup_patterns(self, supergroup_config: SupergroupType, callback: SupergroupConfigCallback,
                                   onnx_conn_graph_type_mapper: OnnxConnectedGraphTypeMapper) \
            -> List[PatternType]:
        """
        Create a list holding pattern types corresponding to sequences specified in the supergroup config
        :param supergroup_config: Quantsim wrapper configurations for supergroup ops
        :return: List of PatternTypes holding supergroup ops and callback for when the supergroup is found
        """
        op_list = supergroup_config[ConfigDictKeys.OP_LIST]
        list_of_permutations = _build_list_of_permutations(op_list, onnx_conn_graph_type_mapper)
        return self._build_list_of_pattern(list_of_permutations, callback)

    @staticmethod
    def _build_list_of_pattern(list_of_op_names: List[List[str]], callback: SupergroupConfigCallback) -> \
            List[PatternType]:
        """
        Builds list of patterns given a list of op names
        """
        list_of_patterns = []
        for op_names in list_of_op_names:
            list_of_patterns.append(PatternType(pattern=op_names, action=callback))
        return list_of_patterns

    @abstractmethod
    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """

    @abstractmethod
    def _set_model_input_configs(self, model_input_configs: ConfigType):
        """
        Set model input specific configurations (fifth level of specificity in configuration file)
        :param model_input_configs: Configuration for model inputs
        """

    @abstractmethod
    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """

    def _op_type_default_override_supported_kernel_lookup(self, op_type: str, bw: int, dtype: QuantizationDataType) \
             -> bool:
        """
        Check if bw and dtype are provided as the default supported kernel override for the given op type
        :param op_type: Op type to check override for
        :param bw: Bitwidth to check in supported kernel default override
        :param dtype: Data type to check in supported kernel default override
        :return: True if bw, dtype is the default supported kernel override, False otherwise
        """
        if op_type in self._quantsim_configs[ConfigDictKeys.OP_TYPE] and \
            ConfigDictKeys.SUPPORTED_KERNELS in self._quantsim_configs[ConfigDictKeys.OP_TYPE][op_type]:
            op_supported_kernels = self._quantsim_configs[ConfigDictKeys.OP_TYPE][op_type][
                ConfigDictKeys.SUPPORTED_KERNELS]
            if op_supported_kernels[DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][ConfigDictKeys.ACTIVATION][
                    ConfigDictKeys.BITWIDTH] == bw and \
                op_supported_kernels[DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][ConfigDictKeys.ACTIVATION][
                    ConfigDictKeys.DTYPE] == dtype:
                return True
        return False


def _build_list_of_permutations(op_list: List[str], onnx_conn_graph_type_mapper: OnnxConnectedGraphTypeMapper) \
        -> List[List[str]]:
    """
    Given a list of onnx op types, where each onnx op type could potentially map to multiple connected graph types,
    create a list of all permutations of lists of connected graph types that would satisfy the same ordering as the
    original onnx op type list.
    For example, for an onnx op type "o1" that maps to two connected graph types "c1_1" and
    "c1_2", and an onnx op type "o2" that maps to two connected graph types "c2_1" and "c2_2", all permutations of
    ["o1", "o2"] would lead to ["c1_1", "c2_1"], ["c1_1", "c2_2"], ["c1_2", "c2_1"], and ["c1_2", "c2_2"].
    :param op_list: List of onnx op types
    :param onnx_conn_graph_type_mapper: Class that provides utilities for mapping onnx op types to connected graph types
    :return: List of permutations of connected graph op types satisfying the ordering specified by op_list onnx types
    """
    # base case, return list of lists of connected graph ops corresponding to the only op in the list
    if len(op_list) == 1:
        permutations_of_op_list = []
        conn_graph_types_of_current_op = onnx_conn_graph_type_mapper.get_conn_graph_type_from_onnx_type(op_list[0])
        for op in conn_graph_types_of_current_op:
            permutations_of_op_list.append([op])
        return permutations_of_op_list

    permutations_of_op_list = []
    permutations_of_succeeding_ops = _build_list_of_permutations(op_list[1:], onnx_conn_graph_type_mapper)
    conn_graph_types_of_current_op = onnx_conn_graph_type_mapper.get_conn_graph_type_from_onnx_type(op_list[0])
    for op in conn_graph_types_of_current_op:
        for permutation in permutations_of_succeeding_ops:
            new_permutation = [op] + permutation
            permutations_of_op_list.append(new_permutation)
    return permutations_of_op_list


def get_setting_type(setting_name: str) -> str:
    """
    Return a string corresponding to the type of setting that is specified by setting_name.
    :param setting_name: Name of the setting to change
    :return: String corresponding to the type of setting that is specified by setting_name.
    """

    if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
        return ConfigDictKeys.IS_QUANTIZED
    if setting_name == ConfigDictKeys.IS_SYMMETRIC:
        return ConfigDictKeys.IS_SYMMETRIC
    if setting_name == ConfigDictKeys.ENCODING_CONSTRAINTS:
        return ConfigDictKeys.ENCODING_CONSTRAINTS
    error_msg = f'Unrecognized quantizer setter name {setting_name}'
    logger.error(error_msg)
    raise AssertionError(error_msg)


def get_all_ops_in_neighborhood(op: Op, direction: str, neighborhood=None, split_type: str = "Split"):
    """
    Given an op and a direction, populate neighborhood dictionary with all ops adjacent to that op, and which direction
    they are adjacent in.  If a neighboring op has other connections in the same direction as the op we began with, ops
    connecting to the neighboring op in those other connections will also be part of the same neighborhood.
    :param op: Op to find neighboring ops from
    :param direction: Direction to search for neighboring ops (will be 'input' or 'output')
    :param neighborhood: Dictionary mapping neighboring ops to the direction which they connect to op.
    :param split_type: Type string of ConnectedGraph Split. By default, its value will be "Split"
    """
    if neighborhood is None:
        neighborhood = {}
    neighborhood[op] = direction
    if direction == 'input' and op.inputs:
        input_products = [inp for inp in op.inputs if inp.is_inter_module()]
        input_ops = [inp.producer for inp in input_products]
        for input_op in input_ops:
            if input_op not in neighborhood:
                neighborhood[input_op] = 'output'
                if input_op.type == split_type:
                    # Neighborhood ops include input of split, as well as all other consumers of split
                    get_all_ops_in_neighborhood(input_op, 'input', neighborhood, split_type)
                    get_all_ops_in_neighborhood(input_op, 'output', neighborhood, split_type)
    elif op.outputs:
        output_ops = op.output_ops
        for output_op in output_ops:
            if output_op not in neighborhood:
                neighborhood[output_op] = 'input'
                if output_op.type == split_type:
                    # Neighborhood ops include all consumers of split
                    get_all_ops_in_neighborhood(output_op, 'output', neighborhood, split_type)
    return neighborhood


def current_config_in_supported_kernels(current_dtype_bw: QuantDtypeBwInfo, supported_kernels: List) -> bool:
    """
    Checks if given bw/dtype config is in (act, param) in supported kernels provided.
    :param current_dtype_bw : current data type and bitwidths for act and param as QuantDtypeBwInfo.
    :param supported_kernels: supported kernels (Default level in config file).
    :return: True, if current config is part of the supported Kernels, False otherwise.
    """

    for supported_kernel_config in supported_kernels:
        # retrieve one set of act/param kernel config support
        act_config = supported_kernel_config[ConfigDictKeys.ACTIVATION]
        param_config = None
        if ConfigDictKeys.PARAM in supported_kernel_config:
            param_config = supported_kernel_config[ConfigDictKeys.PARAM]

        # we need to compare combination of act/param with default user provided config.
        # Because a given kernel support is valid only as a combination.
        if act_config[ConfigDictKeys.DTYPE] == current_dtype_bw.act_dtype and \
            act_config[ConfigDictKeys.BITWIDTH] == current_dtype_bw.act_bw:
            if param_config:
                if param_config[ConfigDictKeys.DTYPE] == current_dtype_bw.param_dtype and \
                    param_config[ConfigDictKeys.BITWIDTH] == current_dtype_bw.param_bw:
                    return True
            else:
                return True

    return False


def is_current_config_same_as_override_option(current_candidate: QuantDtypeBwInfo, supported_kernels: List) -> bool:
    """
   Checks if given bw/dtype config is in (act, param) is same as supported kernel provided as an
   option at DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX.
   :param current_candidate : current candidate as QuantDtypeBwInfo.
   :param supported_kernels: supported kernels (Default level in config file).
   :return: True, if current config is supported Kernel at index specified by , False otherwise.
   """

    override_candidate = get_override_from_supported_kernels(supported_kernels)

    # we need to compare combination of act/param with default user provided config.
    # Because a given kernel support is valid only as a combination.
    return override_candidate == current_candidate


def get_override_from_supported_kernels(supported_kernels: Dict) -> QuantDtypeBwInfo:
    """
    extracts the first option from list of supported kernels configured as QuantDtypeBwInfo.
    :param supported_kernels: Dictionary of supported kernels at default level.
    :return:
    """

    assert supported_kernels

    config_file_default_act_bw_dtype_config = supported_kernels[DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][ConfigDictKeys.ACTIVATION]
    config_file_default_param_bw_dtype_config = None
    if ConfigDictKeys.PARAM in supported_kernels[DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX]:
        config_file_default_param_bw_dtype_config = supported_kernels[DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX][ConfigDictKeys.PARAM]

    override_act_dtype = config_file_default_act_bw_dtype_config[ConfigDictKeys.DTYPE]
    override_act_bw = config_file_default_act_bw_dtype_config[ConfigDictKeys.BITWIDTH]
    if  config_file_default_param_bw_dtype_config:
        override_param_dtype = config_file_default_param_bw_dtype_config[ConfigDictKeys.DTYPE]
        override_param_bw = config_file_default_param_bw_dtype_config[ConfigDictKeys.BITWIDTH]
        return QuantDtypeBwInfo(override_act_dtype, override_act_bw, override_param_dtype, override_param_bw)

    return QuantDtypeBwInfo(override_act_dtype, override_act_bw)


def is_override_dtype_bw_valid(override_dtype_bw_info: QuantDtypeBwInfo, quantsim_dtype_bw_info: QuantDtypeBwInfo) -> bool:
    """
    check if override dtype bw is valid given quantsim default dtype and bw.
    :param override_dtype_bw_info: override data type, bitwidth info as QuantDtypeBwInfo.
    :param quantsim_dtype_bw_info: quantsim default data type, bitwidth info as QuantDtypeBwInfo.
    :return: bool, True if override option is valid, False otherwise.
    """

    # Rule : When an Op does NOT have lower precision kernel support, supported_kernels based override can be applied =>
    # quantsim default dtype/bw should be lower precision compared to override.
    # case (i) if both are int or both are float dtype, compare bitwidths.
    # ex : {quantsim default = int16, override = int8}  or {quantsim default = int8, override = int4} are not supported
    # case (ii) if quantsim default is float => override is not float, then it fails to satisfy criteria because:
    # quantsim defaults are higher precision compared to overrides . (ex : quantsim default = Fp16 > override = int)

    # pylint: disable=too-many-boolean-expressions
    if override_dtype_bw_info.param_dtype and override_dtype_bw_info.param_bw:
        if (quantsim_dtype_bw_info.param_dtype == override_dtype_bw_info.param_dtype and
                quantsim_dtype_bw_info.act_dtype == override_dtype_bw_info.act_dtype and
                (quantsim_dtype_bw_info.act_bw > override_dtype_bw_info.act_bw or
                 quantsim_dtype_bw_info.param_bw > override_dtype_bw_info.param_bw)) or (
                     quantsim_dtype_bw_info.act_dtype == QuantizationDataType.float
                     and quantsim_dtype_bw_info.param_dtype == QuantizationDataType.float):
            logger.info(' Target specfic op level override is lower precision than quantsim info.  \n,'
                        ' (please check both quantsim defaults and default supported_kernels in config file specified at override index {%s}) \n'
                        ' quantsim is configured with %s and supported_kernels override configured as %s \n',
                        DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX, quantsim_dtype_bw_info, override_dtype_bw_info)

    return True


def validate_all_op_level_dtype_bw_overrides(op_configs: OpTypeType, default_candidate: QuantDtypeBwInfo):
    """
    Checks if given op level supported_kernel is supported (across all op types).
    :param op_configs: Op level config information (Level 3 spec in target config file).
    :param default_candidate: default candidate
    :return: bool, indicating valid or not.
    """

    for op_name, op_config in op_configs.items():
        if ConfigDictKeys.SUPPORTED_KERNELS in op_config:
            op_level_supported_kernels = op_config[ConfigDictKeys.SUPPORTED_KERNELS]

            # if current quantsim config or default level supported kernel is in op level supported kernels
            # no override required at op level.
            if current_config_in_supported_kernels(default_candidate,
                                                   op_level_supported_kernels):
                logger.info(" Default option found in op level supported kernels list,  skip "
                            "op level override needed for op {%s} \n", op_name)
            else:
                # If there are multiple options - we always override with DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX
                # in supported_kernels, check if the override option dtype and bitwidth is valid.
                # option specified at DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX of default supported_kernels
                # will be applied during override.
                override_dtype_bw_info = get_override_from_supported_kernels(op_level_supported_kernels)
                if not is_override_dtype_bw_valid(override_dtype_bw_info, default_candidate):
                    error_msg = (f'Op level supported_kernels override check failed for op {op_name} \n'
                                 f'Op level override only with higher precision kernel is supported \n'
                                 f'(please check both quantsim defaults and default supported_kernels in config file '
                                 f'specified at override index {DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX})\n'
                                 f'One way to rectify this is to specify lower precision data type and bit-width as defaults'
                                 f'\nex : (act_bw = {override_dtype_bw_info.act_bw}, param_bw = '
                                 f'{override_dtype_bw_info.param_bw}, act_data_type = {override_dtype_bw_info.act_dtype},'
                                 f'param_data_type = {override_dtype_bw_info.param_dtype}) and '
                                 f'use op level supported_kernels override \n'
                                 f'for this op to indicate higher precision kernel that is supported on given target \n'
                                 f'ex: (act_bw = {default_candidate.act_bw}, param_bw ='
                                 f'{override_dtype_bw_info.param_bw} , act_data_type = {override_dtype_bw_info.act_dtype},'
                                 f'param_data_type = {override_dtype_bw_info.param_dtype}) \n')

                    logger.info(error_msg)
                    raise NotImplementedError(error_msg)
    return True

#TODO change to QuantDtypeBwInfo
def reformat_supported_kernels(supported_kernels: Dict):
    """
    reformat the supported kernels dict to match the internal representation of it ->
    ((activation bitwidth, activation data type), (param bitwidth, param data type))
    :param supported_kernels: Dict with module name and its candidates
    """

    ret_dict = {}

    for op_name, op_supported_kernels in supported_kernels.items():
        candidates = []
        for supported_kernel in op_supported_kernels:
            activation_info = (supported_kernel['activation']['bitwidth'], supported_kernel['activation']['dtype'])
            candidate = activation_info
            if 'param' in supported_kernel:
                param_info = (supported_kernel['param']['bitwidth'], supported_kernel['param']['dtype'])
                candidate = (activation_info, param_info)

            candidates.append(candidate)
        ret_dict[op_name] = candidates

    return ret_dict
