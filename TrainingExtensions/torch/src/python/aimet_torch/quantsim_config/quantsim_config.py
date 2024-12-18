# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import copy
from abc import abstractmethod
from typing import Dict, List, Tuple, Set, Union
import torch
from torch.nn.modules import ConvTranspose2d

from aimet_common.utils import AimetLogger, log_with_error_and_assert_if_false
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.connected_graph.operation import Op
from aimet_common.defs import QuantizationDataType, QuantDtypeBwInfo
from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.connected_graph import connectedgraph_utils as cg_utils
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys, ConfigType, SupergroupType, OpType, \
    ParamType, DefaultsType, OpTypeType, ConfigDictType
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator, \
    get_all_ops_in_neighborhood, get_setting_type, ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG, reformat_supported_kernels
from aimet_common.quantsim_config.quantsim_config import SupergroupConfigCallback as AimetCommonSupergroupConfigCallback
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.onnx_utils import map_torch_types_to_onnx, pytorch_functional_name_to_onnx_dict
from aimet_torch.translation_mapping import aimet_op_to_backend_op_name_map
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper, LazyQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME = {
    "bias": "bias",
    "sp_conv_3d.bias": "bias",
    "weight": "weight"
}

TensorQuantizersTupleType = Tuple[List[LazyQuantizer], List[LazyQuantizer], List[LazyQuantizer],
                                  List[LazyQuantizer]]


class SupergroupConfigCallback(AimetCommonSupergroupConfigCallback):
    """ Class acting as a callback for when supergroups are found """

    def __init__(self, module_to_quantsim_wrapper_dict: Dict[torch.nn.Module, LazyQuantizeWrapper]):
        self._module_to_quantsim_wrapper_dict = module_to_quantsim_wrapper_dict

    def __call__(self, _, op_list: List[Op]):
        # Turn off input and output quantizations for all ops (only turn off output quantization for first op, and only
        # turn off input quantization for last op)
        # Assumes op list is at least of length two
        for index, op in enumerate(op_list):
            if _is_elementwise_functional(op):
                # if op is an elementwise op, it does not have wrappers.  Thus nothing needs to be done, since previous
                # and subsequent ops with wrappers will be set correctly anyway.
                continue
            if op.get_module() is None:
                continue
            if index == 0:
                # turn off only output quantization of first op in the list
                first_quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op_list[0].get_module()]
                # pylint: disable=protected-access
                for quantizer in first_quantsim_wrapper.output_quantizers:
                    quantizer.enabled = False
            elif index == len(op_list) - 1:
                # turn off only input quantization of last op in the list
                last_quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op_list[-1].get_module()]

                for quantizer in last_quantsim_wrapper.input_quantizers:
                    quantizer.enabled = False
            else:
                quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]

                for quantizer in quantsim_wrapper.input_quantizers + quantsim_wrapper.output_quantizers:
                    quantizer.enabled = False


# pylint: disable=too-many-arguments
class QuantSimConfigurator(AimetCommonQuantSimConfigurator):
    """ Class for parsing and applying quantsim configurations from json config file """
    def __init__(self, model, connected_graph: ConnectedGraph, config_file: str, quantsim_output_bw: int,
                 quantsim_param_bw: int, quantsim_data_type: QuantizationDataType):
        super().__init__(config_file, quantsim_data_type, quantsim_output_bw, quantsim_param_bw)
        _report_unsupported_ops(self._quantsim_configs)
        self._conn_graph = connected_graph
        self._module_to_quantsim_wrapper_dict = _create_module_to_quantsim_wrapper_dict(model)
        self._named_modules_to_tensor_quantizers_dict = self._create_named_modules_to_tensor_quantizers_dict()
        self._elementwise_op_to_tensor_quantizers_dict = self._create_elementwise_op_to_tensor_quantizers_dict()
        self._disable_all_quantizers()
        self._supported_kernels = self._parse_supported_kernels()
        if ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG:
            if self.check_correctness_of_dtype_bw_rules(QuantDtypeBwInfo(self._default_data_type, self._default_output_bw,
                                                                         self._default_data_type, self._default_param_bw)):
                logger.info("Supported Kernel check for valid dtype and bitwidth overrides completed")
        self._set_quantsim_configs()
        self._generate_and_apply_op_instance_specific_config()

    def _create_named_modules_to_tensor_quantizers_dict(self) -> Dict[torch.nn.Module, TensorQuantizersTupleType]:
        """
        For every named module in the graph, associate it with a tuple containing 4 lists:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
        :return: Dictionary mapping op to tuple of lists of tensor quantizers to change
        """
        module_to_tensor_quantizers_dict = {}
        module_to_cg_op_dict = {}
        for op in self._conn_graph.get_all_ops().values():
            module = op.get_module()
            if module:
                module_to_cg_op_dict[module] = op
        for module, quantsim_wrapper in self._module_to_quantsim_wrapper_dict.items():
            # Attempt to find op in connected graph corresponding to this module
            op = module_to_cg_op_dict.get(module)
            if op:
                input_true_list = self._get_tensor_quantizers_for_input_true_setting(op)
                output_true_list = self._get_tensor_quantizers_for_output_true_setting(op)
                input_false_list = self._get_tensor_quantizers_for_input_false_setting(op)
                output_false_list = self._get_tensor_quantizers_for_output_false_setting(op)
            else:
                input_true_list = quantsim_wrapper.input_quantizers
                output_true_list = quantsim_wrapper.output_quantizers
                input_false_list = quantsim_wrapper.input_quantizers
                output_false_list = quantsim_wrapper.output_quantizers
            module_to_tensor_quantizers_dict[module] = (input_true_list, output_true_list, input_false_list,
                                                        output_false_list)
        return module_to_tensor_quantizers_dict

    def _create_elementwise_op_to_tensor_quantizers_dict(self) -> Dict[Op, TensorQuantizersTupleType]:
        """
        For every elementwise op in the graph, associate it with a tuple containing 4 lists:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
        :return: Dictionary mapping op to tuple of lists of tensor quantizers to change
        """
        module_to_tensor_quantizers_dict = {}
        # Extract only ops in the model which correspond to elementwise ops
        elementwise_ops = [op for op in self._conn_graph.get_all_ops().values() if _is_elementwise_functional(op)]
        for op in elementwise_ops:
            input_true_list = self._get_tensor_quantizers_for_input_true_setting(op)
            output_true_list = self._get_tensor_quantizers_for_output_true_setting(op)
            input_false_list = self._get_tensor_quantizers_for_input_false_setting(op)
            output_false_list = self._get_tensor_quantizers_for_output_false_setting(op)

            module_to_tensor_quantizers_dict[op] = (input_true_list, output_true_list, input_false_list,
                                                    output_false_list)
        return module_to_tensor_quantizers_dict

    def _get_tensor_quantizers_for_input_true_setting(self, op: Op) -> List[LazyQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have input quantization
        enabled.
        :param op: Op to enable input quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have input
        quantization enabled.
        """
        tensor_quantizers_for_input_true = []
        if op.get_module() is not None:
            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
            tensor_quantizers_for_input_true += qc_quantize_wrapper.input_quantizers
        else:
            queue = [op]
            while queue:
                current_op = queue.pop()
                if current_op.inputs:
                    input_ops = [inp.producer for inp in current_op.inputs if not inp.is_model_input and
                                 not inp.is_const]
                    for input_op in input_ops:
                        if input_op.get_module() is not None and input_op.get_module() in \
                                self._module_to_quantsim_wrapper_dict:
                            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[input_op.get_module()]
                            tensor_quantizers_for_input_true.append(qc_quantize_wrapper.output_quantizers[0])
                        elif input_op.type == CG_SPLIT:
                            queue.append(input_op)
        return tensor_quantizers_for_input_true

    def _get_tensor_quantizers_for_output_true_setting(self, op: Op) -> List[LazyQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have output quantization
        enabled.
        :param op: Op to enable output quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have output
        quantization enabled.
        """
        tensor_quantizers_for_output_true = []
        if op.get_module() is not None:
            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
            tensor_quantizers_for_output_true += qc_quantize_wrapper.output_quantizers
        else:
            queue = [op]
            while queue:
                current_op = queue.pop()
                if current_op.output:
                    output_ops = [consumer for consumer in current_op.output.consumers]  # pylint: disable=unnecessary-comprehension
                    for output_op in output_ops:
                        if output_op.get_module() is not None and output_op.get_module() in \
                                self._module_to_quantsim_wrapper_dict:
                            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[output_op.get_module()]
                            tensor_quantizers_for_output_true += qc_quantize_wrapper.input_quantizers
                        elif output_op.type == CG_SPLIT:
                            queue.append(output_op)
        return tensor_quantizers_for_output_true

    def _get_tensor_quantizers_for_input_false_setting(self, op: Op) -> List[LazyQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have input quantization
        disabled.
        :param op: Op to disable input quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have input
        quantization disabled.
        """
        tensor_quantizers_for_input_false = []
        neighboring_input_ops = get_all_ops_in_neighborhood(op, 'input', split_type=CG_SPLIT)
        for input_op in neighboring_input_ops:
            if input_op.type != CG_SPLIT and input_op.get_module() is not None and input_op.get_module() in \
                    self._module_to_quantsim_wrapper_dict:
                qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[input_op.get_module()]
                if neighboring_input_ops[input_op] == 'input':
                    tensor_quantizers_for_input_false += qc_quantize_wrapper.input_quantizers
                else:
                    tensor_quantizers_for_input_false += qc_quantize_wrapper.output_quantizers
        return tensor_quantizers_for_input_false

    def _get_tensor_quantizers_for_output_false_setting(self, op: Op) -> List[LazyQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have output quantization
        disabled.
        :param op: Op to disable output quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have output
        quantization disabled.
        """
        tensor_quantizers_for_output_false = []
        neighboring_output_ops = get_all_ops_in_neighborhood(op, 'output', split_type=CG_SPLIT)
        for output_op in neighboring_output_ops:
            if output_op.type != CG_SPLIT and output_op.get_module() is not None and output_op.get_module() in \
                    self._module_to_quantsim_wrapper_dict:
                qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[output_op.get_module()]
                if neighboring_output_ops[output_op] == 'input':
                    tensor_quantizers_for_output_false += qc_quantize_wrapper.input_quantizers
                else:
                    tensor_quantizers_for_output_false += qc_quantize_wrapper.output_quantizers
        return tensor_quantizers_for_output_false

    def _disable_all_quantizers(self):
        """
        Set all tensor quantizers to enabled False and use_symmetric_encodings False
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for quantizer in quantsim_wrapper.input_quantizers + quantsim_wrapper.output_quantizers:
                quantizer.enabled = False
                quantizer.use_symmetric_encodings = False
            for param_quantizer in quantsim_wrapper.param_quantizers.values():
                param_quantizer.enabled = False
                param_quantizer.use_symmetric_encodings = False

    def _set_default_configs(self, default_configs: DefaultsType):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """
        self._set_default_configs_for_ops(default_configs[ConfigDictKeys.OPS])
        self._set_default_configs_for_params(default_configs[ConfigDictKeys.PARAMS])

        # Bitwidth and dtype config can be specified at default level or op level in a config file.
        # Rules for override :
        # 1) ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is set to True [AND]
        # 2) Config file has a list of supported_kernels specified in the defaults
        # 2) Quantsim configured default config is not in the supported kernels specified in config file's
        # (defaults supported_kernels list).
        if ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG and ConfigDictKeys.SUPPORTED_KERNELS in default_configs:
            self._override_default_dtype_bw_act_param(default_configs)

        if ConfigDictKeys.STRICT_SYMMETRIC in default_configs:
            self._set_strict_symmetric(default_configs[ConfigDictKeys.STRICT_SYMMETRIC])

        if ConfigDictKeys.UNSIGNED_SYMMETRIC in default_configs:
            self._set_unsigned_symmetric(default_configs[ConfigDictKeys.UNSIGNED_SYMMETRIC])


    def _set_default_configs_for_ops(self, default_op_configs: ConfigType):
        """
        Set default configurations for all ops in the model.
        :param default_op_configs: Default configurations for ops
        """
        # Set configs for all ops
        modified_tensor_quantizers = {}
        # Set configs for all named modules
        for input_output_tensor_quantizers in self._named_modules_to_tensor_quantizers_dict.values():
            self._set_config_for_module(input_output_tensor_quantizers, default_op_configs, modified_tensor_quantizers)
        # Set configs for all elementwise ops
        for input_output_tensor_quantizers in self._elementwise_op_to_tensor_quantizers_dict.values():
            self._set_config_for_module(input_output_tensor_quantizers, default_op_configs, modified_tensor_quantizers)

    def _set_default_configs_for_params(self, default_param_configs: ConfigType):
        """
        Set default configurations for all params in the model.
        :param default_param_configs: Default configurations for params
        """
        # Set configs for all params
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            if quantsim_wrapper.param_quantizers:
                for param_quantizer in quantsim_wrapper.param_quantizers.values():
                    _set_config_for_param(param_quantizer, default_param_configs)

    def _set_strict_symmetric(self, strict_symmetric: bool):
        """
        Set strict symmetric configuration for all quantizers in the model.
        :param strict_symmetric: True or False setting for using strict symmetric mode
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for input_quantizer in quantsim_wrapper.input_quantizers:
                input_quantizer.use_strict_symmetric = strict_symmetric
            for output_quantizer in quantsim_wrapper.output_quantizers:
                output_quantizer.use_strict_symmetric = strict_symmetric
            for param_quantizer in quantsim_wrapper.param_quantizers.values():
                param_quantizer.use_strict_symmetric = strict_symmetric

    def _set_unsigned_symmetric(self, unsigned_symmetric: bool):
        """
        Set unsigned symmetric configuration for all quantizers in the model.
        :param unsigned_symmetric: True or False setting for using unsigned symmetric mode
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for input_quantizer in quantsim_wrapper.input_quantizers:
                input_quantizer.use_unsigned_symmetric = unsigned_symmetric
            for output_quantizer in quantsim_wrapper.output_quantizers:
                output_quantizer.use_unsigned_symmetric = unsigned_symmetric
            for param_quantizer in quantsim_wrapper.param_quantizers.values():
                param_quantizer.use_unsigned_symmetric = unsigned_symmetric


    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for param_name in quantsim_wrapper.param_quantizers.keys():
                quantsim_param_name = MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in param_configs:
                    param_config = param_configs[quantsim_param_name]
                    _set_config_for_param(quantsim_wrapper.param_quantizers[param_name], param_config)

    def _merge_op_types_info(self, op_configs: OpTypeType):
        """
        Merges op type info. from ONNX op type to backend op types

        :param op_configs: Dictionary containing configurations for ops of certain types
        """
        merged_backend_types = []

        for module, _ in self._named_modules_to_tensor_quantizers_dict.items():

            onnx_types = map_torch_types_to_onnx.get(type(module))
            backend_type = aimet_op_to_backend_op_name_map.get(type(module))

            if onnx_types and backend_type in op_configs and backend_type not in merged_backend_types:
                backend_type_op_config = op_configs[backend_type]
                for onnx_type in onnx_types:
                    if onnx_type in op_configs and backend_type != onnx_type:
                        onnx_op_config = op_configs[onnx_type]
                        op_configs[backend_type] = merge_op_type_config(backend_type_op_config, onnx_op_config)
                        merged_backend_types.append(backend_type)

    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """
        modified_tensor_quantizers = {}

        self._merge_op_types_info(op_configs)

        # Set op type configs for named modules
        for module, input_output_tensor_quantizers in self._named_modules_to_tensor_quantizers_dict.items():
            onnx_types = map_torch_types_to_onnx.get(type(module))
            backend_type = aimet_op_to_backend_op_name_map.get(type(module))

            if backend_type in op_configs:
                self._set_config_for_module(input_output_tensor_quantizers, op_configs[backend_type],
                                            modified_tensor_quantizers, module)
            else:
                if not onnx_types:
                    continue
                for onnx_type in onnx_types:
                    if onnx_type in op_configs:
                        op_config = op_configs[onnx_type]
                        self._set_config_for_module(input_output_tensor_quantizers, op_config, modified_tensor_quantizers,
                                                    module)

        # Set op type configs for elementwise ops
        for op, input_output_tensor_quantizers in self._elementwise_op_to_tensor_quantizers_dict.items():
            if op.type in op_configs:
                op_config = op_configs[op.type]
                self._set_config_for_module(input_output_tensor_quantizers, op_config, modified_tensor_quantizers)

    def _set_config_for_module(self, input_output_tensor_quantizers: TensorQuantizersTupleType, op_config: OpType,
                               modified_tensor_quantizers: Dict[LazyQuantizer, Set], module: torch.nn.Module = None):
        """
        Set configurations for a specific op
        :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
            - List of tensor quantizers to change if op's input quantizer setting is set to True
            - List of tensor quantizers to change if op's output quantizer setting is set to True
            - List of tensor quantizers to change if op's input quantizer setting is set to False
            - List of tensor quantizers to change if op's output quantizer setting is set to False
        :param op_config: Configuration for the op
        :param modified_tensor_quantizers: Dictionary of tensor quantizers mapping to set of settings that have been
            changed for that tensor quantizer already.
        :param module: Module to set config of (will be None for elementwise ops)
        """
        if ConfigDictKeys.IS_INPUT_QUANTIZED in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_INPUT_QUANTIZED,
                                      op_config[ConfigDictKeys.IS_INPUT_QUANTIZED], modified_tensor_quantizers)
        if ConfigDictKeys.IS_OUTPUT_QUANTIZED in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_OUTPUT_QUANTIZED,
                                      op_config[ConfigDictKeys.IS_OUTPUT_QUANTIZED], modified_tensor_quantizers)
        if ConfigDictKeys.IS_SYMMETRIC in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_SYMMETRIC,
                                      op_config[ConfigDictKeys.IS_SYMMETRIC], modified_tensor_quantizers)
        if ConfigDictKeys.ENCODING_CONSTRAINTS in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.ENCODING_CONSTRAINTS,
                                      op_config[ConfigDictKeys.ENCODING_CONSTRAINTS], modified_tensor_quantizers)

        # Will only see this in the op_type section, not default
        if ConfigDictKeys.PARAMS in op_config:
            if module is None:
                logger.error('No module provided to set params for')
                raise AssertionError('No module provided to set params for')
            quantsim_wrapper = self._module_to_quantsim_wrapper_dict[module]
            for param_name in quantsim_wrapper.param_quantizers.keys():
                quantsim_param_name = MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in op_config[ConfigDictKeys.PARAMS]:
                    param_config = op_config[ConfigDictKeys.PARAMS][quantsim_param_name]
                    _set_config_for_param(quantsim_wrapper.param_quantizers[param_name], param_config)

        # override op level supported kernel config if it is enforced
        if ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG and ConfigDictKeys.SUPPORTED_KERNELS in op_config:
            if module is None:
                logger.error('No module provided to set params for')
                raise AssertionError('No module provided to set params for')
            quantsim_wrapper = self._module_to_quantsim_wrapper_dict[module]
            self._apply_overrides_for_op(op_config, quantsim_wrapper)

    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """
        def find_scale_foldable_bns(cg):
            """
            Find batchnorms that can be folded to scale
            """
            conv_bn_pairs = []

            def handler(_, op_list):
                conv, bn = op_list
                conv_module = conv.get_module()
                # Transposed depthwise convolutions are not supported for batchnorm folding
                if isinstance(conv_module, ConvTranspose2d) and conv_module.groups != 1:
                    return
                conv_bn_pairs.append((conv, bn))

            patterns_with_callbacks = []
            conv_types = ['Conv1d', 'Conv', 'ConvTranspose']
            linear_types = ['Gemm']

            for op_type in conv_types + linear_types:
                patterns_with_callbacks.append(PatternType(pattern=[op_type, 'BatchNormalization'],
                                                           action=handler))

            # create graph searcher instance with connected graph and patterns to search
            graph_searcher = GraphSearcher(cg, patterns_with_callbacks)
            graph_searcher.find_all_patterns_in_graph_apply_actions()
            return conv_bn_pairs

        conv_bn_pairs = find_scale_foldable_bns(self._conn_graph)
        foldable_bns = [bn for _, bn in conv_bn_pairs]

        patterns_with_callbacks = []
        for supergroup_config in supergroups_configs:
            callback = SupergroupConfigCallback(self._module_to_quantsim_wrapper_dict)
            op_list = supergroup_config[ConfigDictKeys.OP_LIST]
            patterns_with_callbacks.append(PatternType(pattern=op_list, action=callback))

        if patterns_with_callbacks:
            graph_searcher = GraphSearcher(self._conn_graph, patterns_with_callbacks)
            graph_searcher.find_all_patterns_in_graph_apply_actions(ignore=foldable_bns)

        def fuse_config(conv: Op, bn: Op):
            """
            Fuse configs of conv and bn

            If conv output quantizer is enabled, disable it and enable output quantizer of BN instead
            so that we can fold batch norm to conv.
            """
            if conv.get_module() not in self._module_to_quantsim_wrapper_dict:
                return

            if bn.get_module() not in self._module_to_quantsim_wrapper_dict:
                return

            conv_wrapper = self._module_to_quantsim_wrapper_dict[conv.get_module()]
            bn_wrapper = self._module_to_quantsim_wrapper_dict[bn.get_module()]

            for quantizer in bn_wrapper.input_quantizers:
                quantizer.enabled = False

            for quantizer in bn_wrapper.param_quantizers.values():
                quantizer.enabled = False

            for conv_quantizer, bn_quantizer in zip(conv_wrapper.output_quantizers,
                                                    bn_wrapper.output_quantizers):
                bn_quantizer.enabled = conv_quantizer.enabled
                conv_quantizer.enabled = False

        for conv, bn in conv_bn_pairs:
            fuse_config(conv, bn)

    def _set_model_input_configs(self, model_input_configs: ConfigType):
        """
        Set model input specific configurations (fifth level of specificity in configuration file).

        :param model_input_configs: Configuration for model inputs
        """

        if ConfigDictKeys.IS_INPUT_QUANTIZED in model_input_configs:
            # Find all ops which are either model input ops or have constant inputs as inputs
            input_ops = cg_utils.get_all_input_ops(self._conn_graph)
            ops_to_set = list(set(input_ops).union(cg_utils.get_all_ops_with_constant_inputs(self._conn_graph)))

            for op in ops_to_set:
                if op.get_module() in self._module_to_quantsim_wrapper_dict:
                    input_and_constant_input_indices = [idx for idx, inp in enumerate(op.inputs) if
                                                        inp.is_model_input or inp.is_const]
                    curr_module = self._module_to_quantsim_wrapper_dict[op.get_module()]
                    for idx, quantizer in enumerate(curr_module.input_quantizers):
                        if idx in input_and_constant_input_indices:
                            quantizer.enabled = True

    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """
        output_ops = cg_utils.get_all_output_ops(self._conn_graph)
        for op in output_ops:
            if op.get_module() in self._named_modules_to_tensor_quantizers_dict:
                modified_tensor_quantizers = {}
                self._set_config_for_module(self._named_modules_to_tensor_quantizers_dict[op.get_module()],
                                            model_output_configs, modified_tensor_quantizers)

    # -----------------------------------[ override support begin] --------------------------------------------- #

    def _override_default_param_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for param quantizers
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """
        # Set configs for all params
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            self._override_param_bw_dtype(quantsim_wrapper, data_type, bitwidth)

    def _override_default_act_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bw default config for input/output quantizers.
        :param data_type: data type as QuantizationDataType
        :param bitwidth: bitwidth to be configured
        :return:
        """

        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for input_quantizer in quantsim_wrapper.input_quantizers:
                input_quantizer.data_type = data_type
                input_quantizer.bitwidth = bitwidth
            for output_quantizer in quantsim_wrapper.output_quantizers:
                output_quantizer.data_type = data_type
                output_quantizer.bitwidth = bitwidth

    # pylint: disable=arguments-differ
    def _override_param_bw_dtype(self, quantize_wrapper: LazyQuantizeWrapper, data_type: QuantizationDataType = None,
                                 bitwidth: int = None):
        """
        overrides data type and bitwidth default config for param quantizers of given quantsim wrapper.
        :param quantize_wrapper : Quantize wrapper that to which param override will be applied to.
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """
        # Set configs for all params
        if quantize_wrapper.param_quantizers and data_type and bitwidth:
            for param_quantizer in quantize_wrapper.param_quantizers.values():
                param_quantizer.data_type = data_type
                param_quantizer.bitwidth = bitwidth

    # pylint: disable=arguments-differ
    def _override_act_bw_dtype(self, quantize_wrapper: LazyQuantizeWrapper, data_type: QuantizationDataType,
                               bitwidth: int):
        """
        Override activation bw and dtype for activation quantizers of the quantsim wrapper if applicable.
        :param quantize_wrapper : Quantize wrapper that to which activation override will be applied to.
        :param data_type: data type as QuantizationDataType
        :param bitwidth: bitwidth
        """
        # For a standalone kernel supporting fp16, only its parameter quantizer needs to be changed to fp16 because
        # requantization will happen on the input and output of the kernel (int8 -> fp16 for input and fp16 -> int8 for
        # output), causing lower precision encodings to be necessary for input and output.
        # In the case of back to back kernels supporting fp16, no requantization will happen in between, so that
        # quantizer can be set to fp16.
        if data_type == QuantizationDataType.float and bitwidth == 16:
            # pylint: disable=protected-access
            conn_graph_op = self._conn_graph._module_to_op_dict[quantize_wrapper._module_to_wrap]

            # Checking if input quantizer(s) should be set to fp16. Only set to fp16 if all input ops are also only
            # fp16 supported.
            input_ops = conn_graph_op.input_ops
            all_input_ops_fp16 = True
            for input_op in input_ops:
                if not self._op_type_default_override_supported_kernel_lookup(input_op.type, bitwidth, data_type):
                    all_input_ops_fp16 = False
                    break
            if all_input_ops_fp16:
                for input_quantizer in quantize_wrapper.input_quantizers:
                    input_quantizer.bitwidth = bitwidth
                    input_quantizer.data_type = data_type

            # Checking if output quantizer(s) should be set to fp16. Only set to fp16 if all output ops are also only
            # fp16 supported.
            output_ops = conn_graph_op.output_ops
            all_output_ops_fp16 = True
            for output_op in output_ops:
                if not self._op_type_default_override_supported_kernel_lookup(output_op.type, bitwidth, data_type):
                    all_output_ops_fp16 = False
                    break
            if all_output_ops_fp16:
                for output_quantizer in quantize_wrapper.output_quantizers:
                    output_quantizer.bitwidth = bitwidth
                    output_quantizer.data_type = data_type
        else:
            for quantizer in quantize_wrapper.input_quantizers + quantize_wrapper.output_quantizers:
                quantizer.bitwidth = bitwidth
                quantizer.data_type = data_type

    # -----------------------------------[ override support end] --------------------------------------------- #

    def _generate_and_apply_op_instance_specific_config(self):
        """
        Generate op instance specific configurations - currently supported_kernels and per_channel_quantization fields
        This function uses op specific supported_kernels (if absent use defaults), op specific per_channel_quantization
        fields (if absent use default per_channel_quantization) and generate op instance specific config
        """
        per_channel_quantization = self._parse_per_channel_quantization()
        hw_version = self._get_hw_version()
        supported_kernels = reformat_supported_kernels(self._supported_kernels)
        config_generator = config_generator_factory(hw_version, supported_kernels, per_channel_quantization)

        for op in self._conn_graph.ordered_ops:
            if op.get_module() in self._module_to_quantsim_wrapper_dict:
                wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
                wrapper.supported_kernels, per_channel_quantization = config_generator.generate(op.get_module(),
                                                                                                op.type)
                if per_channel_quantization:
                    wrapper.enable_per_channel_quantization()

def merge_op_type_config(backend_type_op_config: Dict, onnx_type_op_config: Dict) -> Dict:
    """
    Merges op_type_info from onnx_type_op_config to backend_type_op_config
    except supported kernels

    :param backend_type_op_config: Op. config generated from backend
    :param onnx_type_op_config:  ONNX op type config
    :return: Merged op type config
    """
    backend_type_op_config_supported_kernels = backend_type_op_config.get(ConfigDictKeys.SUPPORTED_KERNELS)
    backend_type_op_config = copy.deepcopy(onnx_type_op_config)

    if backend_type_op_config_supported_kernels:
        backend_type_op_config[ConfigDictKeys.SUPPORTED_KERNELS] = backend_type_op_config_supported_kernels

    return backend_type_op_config

def config_generator_factory(hw_version, supported_kernels, per_channel_quantization):
    """
    factory to select the config generator based on the hw_version
    :param hw_version: hw_version field from the config file
    :param supported_kernels: aggregated supported_kernels fields from the config file
    :param per_channel_quantization: aggregated per_channel_quantization fields from the config file
    :return: Config Generator object
    """

    config_generator = DefaultOpInstanceConfigGenerator(supported_kernels, per_channel_quantization)
    logger.info('Selecting DefaultOpInstanceConfigGenerator to compute the specialized config. hw_version:%s',
                hw_version)
    return config_generator


def _create_module_to_quantsim_wrapper_dict(model: torch.nn.Module) -> Dict[torch.nn.Module, LazyQuantizeWrapper]:
    """
    Create a dictionary mapping modules in the model to corresponding quantsim wrappers
    :param model: Pytorch model with quantsim wrappers in place
    :return: Dictionary mapping modules in the model to corresponding quantsim wrappers
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    from aimet_torch._base.quantsim import _QuantizedModuleProtocol
    from aimet_torch.v2.nn import BaseQuantizationMixin
    module_to_quantsim_wrapper_dict = {}
    for _, module in model.named_modules():
        if isinstance(module, BaseQuantizationMixin):
            continue
        if isinstance(module, (_QuantizedModuleProtocol, LazyQuantizeWrapper)):
            module_to_quantsim_wrapper_dict[module._module_to_wrap] = module      # pylint: disable=protected-access
    return module_to_quantsim_wrapper_dict


def _set_config_for_param(param_quantizer: LazyQuantizer, param_config: ConfigType):
    """
    Set configurations for a specific param tensor quantizer
    :param param_quantizer: Tensor quantizer to set configurations for
    :param param_config: Configuration for the tensor quantizer
    """
    if ConfigDictKeys.IS_QUANTIZED in param_config:
        param_quantizer.enabled = param_config[ConfigDictKeys.IS_QUANTIZED]
    if ConfigDictKeys.IS_SYMMETRIC in param_config:
        param_quantizer.use_symmetric_encodings = \
            param_config[ConfigDictKeys.IS_SYMMETRIC]


def _modify_tensor_quantizers(input_output_tensor_quantizers: TensorQuantizersTupleType, setting_name: str,
                              quantizer_setting: Union[dict, bool],
                              modified_tensor_quantizers: Dict[LazyQuantizer, Set]):
    """
    Modify the appropriate tensor quantizers for the given quantizer setting.  If a tensor quantizer has already been
    modified, compare the old setting with the new setting and assert if the settings conflict.
    :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
    :param setting_name: String representing the setting to be modified
    :param quantizer_setting: Boolean representing the new setting value
    :param modified_tensor_quantizers: Dictionary of tensor quantizers mapping to set of settings that have been changed
        for that tensor quantizer already.
    """
    setting_type = get_setting_type(setting_name)

    tensor_quantizers_to_modify = _get_tensor_quantizers_to_modify(input_output_tensor_quantizers, setting_name,
                                                                   quantizer_setting)
    for tensor_quantizer in tensor_quantizers_to_modify:
        if tensor_quantizer in modified_tensor_quantizers and \
                setting_type in modified_tensor_quantizers[tensor_quantizer]:
            # Tensor quantizer's setting has already been modified
            if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                current_setting = tensor_quantizer.enabled
            elif setting_name == ConfigDictKeys.IS_SYMMETRIC:
                current_setting = tensor_quantizer.use_symmetric_encodings
            else:
                current_setting = {ConfigDictKeys.MIN: tensor_quantizer.encoding_min_max_fixed_vals[0],
                                   ConfigDictKeys.MAX: tensor_quantizer.encoding_min_max_fixed_vals[1]}
            log_with_error_and_assert_if_false(current_setting == quantizer_setting,
                                               logger,
                                               f'Conflicting tensor quantizer settings for {setting_name}')
        else:
            if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                tensor_quantizer.enabled = quantizer_setting
            elif setting_name == ConfigDictKeys.IS_SYMMETRIC:
                tensor_quantizer.use_symmetric_encodings = quantizer_setting
            elif setting_name == ConfigDictKeys.ENCODING_CONSTRAINTS:
                tensor_quantizer.encoding_min_max_fixed_vals = (quantizer_setting[ConfigDictKeys.MIN],
                                                                quantizer_setting[ConfigDictKeys.MAX])
            if tensor_quantizer not in modified_tensor_quantizers:
                modified_tensor_quantizers[tensor_quantizer] = {setting_type}
            else:
                modified_tensor_quantizers[tensor_quantizer].add(setting_type)


def _get_tensor_quantizers_to_modify(input_output_tensor_quantizers: TensorQuantizersTupleType, setting_name: str,
                                     quantizer_setting: bool) -> List[LazyQuantizer]:
    """
    Given the tuple containing lists of tensor quantizers to modify under different situations, identify a list of
    tensor quantizers to modify given the specific setting name and quantizer setting.
    :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
    :param setting_name: String representing the setting to be modified
    :param quantizer_setting: Boolean representing the new setting value
    :return: List of tensor quantizers to modify given the specific setting name and quantizer setting.
    """
    input_true_list, output_true_list, input_false_list, output_false_list = input_output_tensor_quantizers

    if setting_name == ConfigDictKeys.IS_INPUT_QUANTIZED and quantizer_setting:
        return input_true_list
    if setting_name == ConfigDictKeys.IS_OUTPUT_QUANTIZED and quantizer_setting:
        return output_true_list
    if setting_name == ConfigDictKeys.IS_INPUT_QUANTIZED and not quantizer_setting:
        return input_false_list
    if setting_name == ConfigDictKeys.IS_OUTPUT_QUANTIZED and not quantizer_setting:
        return output_false_list
    if setting_name == ConfigDictKeys.IS_SYMMETRIC:
        # Will modify all input and output quantizers in the False case
        return input_false_list + output_false_list
    if setting_name == ConfigDictKeys.ENCODING_CONSTRAINTS:
        return output_true_list
    error_msg = f'Encountered unrecognized case for setting name {setting_name}, setting value {quantizer_setting}'
    logger.error(error_msg)
    raise AssertionError(error_msg)


def _report_unsupported_ops(quantsim_config: ConfigDictType):
    """ Log unsupported op types found in the config """

    # Look for unsupported ops in op_type section
    op_type_configs = quantsim_config[ConfigDictKeys.OP_TYPE]
    for op in op_type_configs.keys():
        found_op = False
        for op_list in map_torch_types_to_onnx.values():
            if op in op_list:
                found_op = True
                break
        # Need the below for elementwise ops which will use the type_mapper instead of map_torch_types_to_onnx
        found_op = found_op or (op in pytorch_functional_name_to_onnx_dict.values())
        if not found_op:
            logger.info('Unsupported op type %s', op)

    # Look for unsupported ops in supergroups section
    supergroups = quantsim_config[ConfigDictKeys.SUPERGROUPS]
    for supergroup in supergroups:
        for op in supergroup[ConfigDictKeys.OP_LIST]:
            known_onnx_types = []
            for val in map_torch_types_to_onnx.values():
                known_onnx_types.extend(val)
            if op not in known_onnx_types:
                error_msg = f'Unsupported op type {op}'
                logger.error(error_msg)
                # Raising an error here since an unrecognized op will cause supergroup graph matching to fail
                raise AssertionError(error_msg)


def _is_elementwise_functional(op: Op) -> bool:
    """
    Check if op is a functional elementwise op.
    :param op: Operation to check whether it is functional elementwise op
    :return: True if op is functional elementwise, False otherwise
    """
    return op.type in ['Add', 'Mul', 'Concat', 'Div'] and op.get_module() is None


class OpInstanceConfigGenerator:
    """
    Class to specify op instance specific rules and generate the updated config
    """

    def __init__(self, op_type_supported_kernels: dict, op_type_pcq: dict):
        """
        :param op_type_supported_kernels: supported_kernels fields from the config file(specific op types + default)
        :param op_type_pcq: per_channel_quantization(pcq) fields from the config file(specific op types + default)
        """
        self.op_type_supported_kernels = op_type_supported_kernels
        self.op_type_pcq = op_type_pcq
        assert ConfigDictKeys.DEFAULTS in self.op_type_supported_kernels
        assert ConfigDictKeys.DEFAULTS in self.op_type_pcq

    @abstractmethod
    def generate(self, module: torch.nn.Module, op_type: str) -> dict:
        """ generate the config for the given op """

    def _generate_pcq(self, module: torch.nn.Module) -> bool:
        """
        Helper function to generate the pcq field
        :param module: torch op instance to generate the pcq value to
        :return: pcq value for the op type

        Steps:
        1. Generate onnx_types for the given module
        2. Check if the above onnx_types exist in the config file for pcq
        3. If any entry is present, use it else use the default value
        """
        pcq = False
        onnx_types = map_torch_types_to_onnx.get(type(module), [])
        onnx_types_in_config = set(onnx_types).intersection(self.op_type_pcq)

        if onnx_types_in_config:
            for onnx_type in onnx_types_in_config:
                if self.op_type_pcq[onnx_type]:
                    pcq = True
                    break
        elif self.op_type_pcq[ConfigDictKeys.DEFAULTS]:
            pcq = True

        return pcq


class DefaultOpInstanceConfigGenerator(OpInstanceConfigGenerator):
    """
    Default implementation of OpInstanceConfigGenerator
    """

    def generate(self, module: torch.nn.Module, op_type: str) -> Tuple[dict, bool]:
        """
        :param module: module to generate the specialized config
        :param op_type: Type str retrieved from CG op
        :return: supported_kernels and per_channel_quantization fields
        """
        supported_kernels = []
        backend_type = aimet_op_to_backend_op_name_map.get(type(module))

        if backend_type is not None:
            supported_kernels = self.op_type_supported_kernels.get(backend_type)

        if not supported_kernels:
            supported_kernels = self.op_type_supported_kernels.get(op_type,
                                                                   self.op_type_supported_kernels[ConfigDictKeys.DEFAULTS])
        pcq = self._generate_pcq(module)

        return supported_kernels, pcq
