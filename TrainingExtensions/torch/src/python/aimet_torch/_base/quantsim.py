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
# pylint: disable=too-many-lines
""" QuantizationSimModel interface """
from abc import ABC, abstractmethod
from itertools import chain
import os
import copy
from collections import OrderedDict, defaultdict
import json
import warnings
import pickle
from typing import (
    Callable,
    List,
    Union,
    Dict,
    Optional,
    runtime_checkable,
    Protocol,
    Mapping,
    TYPE_CHECKING,
    Tuple,
    Type,
    Iterable,
)

import torch
import onnx
from packaging import version  # pylint: disable=wrong-import-order
from safetensors.numpy import save_file as save_safetensor_file

from aimet_common.utils import AimetLogger, save_json_yaml, log_with_error_and_assert_if_false
from aimet_common.defs import QuantScheme, QuantizationDataType, SupportedKernelsAction, QuantDtypeBwInfo
from aimet_common.quantsim import validate_quantsim_inputs, extract_global_quantizer_args, VALID_ENCODING_VERSIONS
from aimet_common.quant_utils import get_conv_accum_bounds
from aimet_common.utils import deprecated, _red
from aimet_common import quantsim

from aimet_torch import torchscript_utils, utils, onnx_utils
from aimet_torch.meta.connectedgraph import ConnectedGraph, Op
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper
from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_torch._base.nn.modules.custom import MatMul
from aimet_torch.onnx_utils import (
    OnnxSaver,
    OnnxExportApiArgs,
    CustomMarker
)
from aimet_torch.experimental.v2.quantsim.export_utils import _export_to_1_0_0

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from aimet_torch.v2.quantization.base.encoding import EncodingBase


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SKIP_TORCH_ENCODINGS_EXPORT = False

SUPPORTED_KERNELS_ACTION = SupportedKernelsAction.warn_on_error

unquantizable_modules = (
    torch.nn.Identity,
)


class QuantParams:
    """
    Data type to hold quantization related params.
    """

    def __init__(self,
                 weight_bw: int = 8,
                 act_bw: int = 8,
                 round_mode: str = 'nearest',
                 quant_scheme: Union[QuantScheme, str] = QuantScheme.post_training_tf_enhanced,
                 config_file: str = None):
        """
        Constructor

        :param weight_bw: Weight bitwidth (4-31) to use for quantizing layer weights. Default = 8
        :param act_bw: Activation bitwidth(4-31) to use for quantizing layer activations. Default = 8
        :param round_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param config_file: Path to Configuration file for model quantizers
        """

        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.round_mode = round_mode
        self.quant_scheme = quant_scheme
        self.config_file = config_file


@runtime_checkable
class _QuantizerProtocol(Protocol):
    def get_encodings(self) -> Optional["EncodingBase"]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """

    def set_encodings(self, encoding: "EncodingBase"):
        """
        Set the quantizer's encodings
        """


@runtime_checkable
class _QuantizedModuleProtocol(Protocol):
    """
    Defines the minimum interface requirements for exporting encodings from a module.
    """
    input_quantizers: List[_QuantizerProtocol]
    output_quantizers: List[_QuantizerProtocol]
    param_quantizers: Dict[str, _QuantizerProtocol]

    def export_input_encodings(self, encoding_version: str) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """

    def export_output_encodings(self, encoding_version: str) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """

    def export_param_encodings(self, encoding_version: str) -> Dict[str, List[Dict]]:
        """
        Returns a dict of {param name: param encodings}, with each encoding represented as a List of Dicts
        """

    def import_input_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import input encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_output_encodings(self,
                                encodings: Mapping[str, Mapping],
                                strict: bool,
                                partial: bool,
                                requires_grad: Optional[bool],
                                allow_overwrite: bool):
        """
        Import output encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_param_encodings(self,
                               encodings: Mapping[str, Mapping],
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Import parameter encodings represented in below format:
        {
            'param_name_0': [dict, dict, ...],
            'param_name_1': [dict, dict, ...],
            ...
        }
        """

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns the floating point version of quantized module
        """


ExportableQuantModule = _QuantizedModuleProtocol


class _QuantizationSimModelInterface(ABC):
    model: torch.nn.Module

    @abstractmethod
    def compute_encodings(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def export(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def load_encodings(self, *args, **kwargs): # pylint: disable=missing-function-docstring
        ...

    @abstractmethod
    def named_qmodules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        """Generator that yields all quantized modules in the model and their names
        """

    def qmodules(self) -> Iterable[torch.nn.Module]:
        """Generator that yields all quantized modules in the model
        """
        yield from (module for _, module in self.named_qmodules())


class _QuantizationSimModelBase(_QuantizationSimModelInterface):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals, too-many-public-methods
    def __init__(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                 quant_scheme: Union[str, QuantScheme] = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest', default_output_bw: int = 8, default_param_bw: int = 8,
                 in_place: bool = False, config_file: str = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int):

        """
        Constructor for QuantizationSimModel.

        :param model: Model to add simulation ops to
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param quant_scheme: Quantization scheme. The Quantization scheme is used to compute the Quantization encodings.
                             There are multiple schemes available. Please refer the QuantScheme enum definition.
        :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing all layer inputs and outputs
                unless otherwise specified in the config file.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing all layer parameters
                unless otherwise specified in the config file.
        :param in_place: If True, then the given 'model' is modified in-place to add quant-sim nodes.
                Only suggested use of this option is when the user wants to avoid creating a copy of the model
        :param config_file: Path to Configuration file for model quantizers
        :param default_data_type: Default data type to use for quantizing all inputs, outputs and parameters.
                                 unless otherwise specified in the config file.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 or 32 and default_param_bw=16 or 32.
        """
        # Perform sanity checks on inputs
        validate_quantsim_inputs(quant_scheme, rounding_mode, default_output_bw, default_param_bw,
                                 default_data_type)
        # save some parameters
        if in_place:
            self.model = model
        else:
            self.model = copy.deepcopy(model)

        try:
            self.connected_graph = ConnectedGraph(self.model, dummy_input)
        except (torch.jit.TracingCheckError, AssertionError):
            self.connected_graph = None

        if isinstance(quant_scheme, str):
            if quant_scheme == 'tf':
                quant_scheme = QuantScheme.post_training_tf
            elif quant_scheme == 'tf_enhanced':
                quant_scheme = QuantScheme.post_training_tf_enhanced
            elif quant_scheme == 'percentile':
                quant_scheme = QuantScheme.post_training_percentile
        self._quant_scheme = quant_scheme
        self._rounding_mode = rounding_mode
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw
        self._config_file = config_file
        self._is_conditional = False
        self._module_marker_map = {}
        self._percentile_value = 100 # default percentile value
        self._excluded_layer_names = []

        # Add quantization layers
        inout_tensor_shapes = utils.get_inout_tensor_shape_per_module(self.model, dummy_input)
        num_inout_tensors = {
            module: (len(input_tensor_shapes), len(output_tensor_shapes))
            for module, (input_tensor_shapes, output_tensor_shapes)
            in inout_tensor_shapes.items()
        }
        inout_tensors_dtypes_for_cast_ops = utils.get_inout_tensors_dtypes_for_cast_modules(self.model, dummy_input)

        self._add_quantization_wrappers(self.model, num_inout_tensors, default_data_type)
        self._set_tensor_quantizers_for_consts(inout_tensor_shapes)

        # Disable bias quantization
        self.exclude_param_from_quantization("bias")

        quantsim_configurator = self.configure_quantization_ops(config_file, default_output_bw, default_param_bw,
                                                                default_data_type)

        self.quant_args = extract_global_quantizer_args(quant_scheme, quantsim_configurator)

        self._enable_output_quantizers_for_specific_cast_ops(inout_tensors_dtypes_for_cast_ops)

        # pylint: disable=protected-access
        self._hw_version = quantsim_configurator._get_hw_version()
        self._supported_kernels = quantsim_configurator.get_supported_kernels()
        self._validate_supported_kernels_for_quantizers(SUPPORTED_KERNELS_ACTION)

        self._apply_exception_rules()

        # Initialize real wrappers using collected information
        self._realize_quant_wrappers_in_model(self.model)

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    @abstractmethod
    def _realize_quant_wrappers_in_model(self, model: torch.nn.Module):
        """
        Prepare QuantSim for compute encodings. Resets encodings for each quantizable layer and sets mode to Analysis.
        Realize quant wrappers using collected information in LazyQuantWrapper.

        :param model: model containing modules wrapped with LazyQuantWrapper
        """

    @abstractmethod
    def _add_quantization_wrappers(self, module, num_inout_tensors, default_data_type: QuantizationDataType):
        ...

    def _set_tensor_quantizers_for_consts(self, inout_tensor_shape_dict: Dict):
        """
        Identify and set is_const for tensor quantizers which correspond to constant inputs in the model.
        """

        if self.connected_graph is None:
            return

        for qmodule in self.qmodules():
            if not isinstance(qmodule, (_QuantizedModuleProtocol, LazyQuantizeWrapper)):
                continue

            # Only handling QcQuantWrappers and not QcQuantizeRecurrents
            # pylint: disable=protected-access
            conn_graph_op = self.connected_graph._module_to_op_dict.get(qmodule.get_original_module())
            if conn_graph_op is None:
                continue

            input_tensor_shape_list = inout_tensor_shape_dict.get(qmodule.get_original_module())

            for idx, (input_quantizer, inp) in enumerate(zip(qmodule.input_quantizers, conn_graph_op.inputs)):
                input_quantizer.is_const = inp.is_const
                input_quantizer.is_parm = inp.is_parm
                input_quantizer.is_singleton = (input_tensor_shape_list is not None \
                                                and input_tensor_shape_list[0][idx] is not None \
                                                and input_tensor_shape_list[0][idx].numel() == 1)

    def exclude_param_from_quantization(self, param_name_to_exclude: str):
        """
        Excludes all parameters matching 'param_name' from quantization
        :param param_name_to_exclude: Name of the parameter to exclude
        :return: None
        """
        for qmodule in self.qmodules():
            try:
                qtzr = qmodule.param_quantizers[param_name_to_exclude]
            except KeyError:
                qtzr = None

            if qtzr:
                qmodule.param_quantizers[param_name_to_exclude].enabled = False

    def configure_quantization_ops(self, config_file: str, default_output_bw: int, default_param_bw: int,
                                   default_data_type: QuantizationDataType) -> QuantSimConfigurator:
        """
        Configure inserted quantize ops using config file and fill in all the supported kernels
        :param config_file: Configuration file to use
        :param default_output_bw: default bitwidth for activations
        :param default_param_bw: default bitwidth for params
        :param default_data_type: default data type
        :return: QuantSimConfigurator object
        """
        if self.connected_graph is None:
            error_msg = ('A connected graph failed to be built.\n'
                         'Unable to proceed with automatically configuring quantization ops using the config file.\n'
                         'Please configure quantization ops manually by redefining '
                         'QuantizationSimModel.configure_quantization_ops()')
            logger.error(error_msg)
            raise AssertionError(error_msg)
        return QuantSimConfigurator(self.model, self.connected_graph, config_file, default_output_bw,
                                    default_param_bw, default_data_type)

    def _enable_output_quantizers_for_specific_cast_ops(self, inout_tensors_dtypes: Dict[torch.nn.Module, Tuple[torch.dtype, torch.dtype]]):
        """
        Enable output quantizer for Cast Ops where datatype of input tensor is int/bool
        and data type of output tensor is float.
        """
        # pylint: disable=protected-access
        model_prefix = self.connected_graph._model_name + '.'
        torch_int_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, torch.uint8}
        torch_float_dtypes = {torch.float16, torch.float32, torch.float64}

        for module, inout_dtypes in inout_tensors_dtypes.items():
            input_tensor_dtype = inout_dtypes[0]
            output_tensor_dtype = inout_dtypes[1]
            # pylint: disable=protected-access
            module_name = self.connected_graph._module_to_name[module].split(model_prefix)[-1]

            if input_tensor_dtype != output_tensor_dtype and input_tensor_dtype in torch_int_dtypes and output_tensor_dtype in torch_float_dtypes:
                logger.info("Enabling output quantizer for module %s", module_name)
                wrapped_module = getattr(self.model, module_name)
                for output_quantizer in wrapped_module.output_quantizers:
                    setattr(output_quantizer, 'enabled', True)

    def _validate_supported_kernels_for_quantizers(self, action: SupportedKernelsAction):
        """
        Validate supported kernels for all the Quantizers in the QuantSimModel
        :param action: The action to be performed when incorrect candidate is set in a quantizer
        """

        def apply_act_param_rules(curr_candidate: QuantDtypeBwInfo, allowed_supported_kernels: List[QuantDtypeBwInfo], module_name):
            """
            helper function to validate both activation and param against the supported_kernels passed
            :param curr_candidate: candidate of interest
            :param allowed_supported_kernels: List of supported kernels for the given module
            :param module_name: name of the module
            """
            if action != SupportedKernelsAction.allow_error:
                for k in allowed_supported_kernels:
                    if curr_candidate == k:
                        return

                if action == SupportedKernelsAction.warn_on_error:
                    logger.warning("candidate:%s is not under the supported_kernels for the module %s", curr_candidate,
                                   module_name)

                if action == SupportedKernelsAction.assert_on_error:
                    error_msg = f'candidate: {curr_candidate} is not under the supported_kernels for the module {module_name}'
                    raise RuntimeError(error_msg)

        def apply_act_rules(act: Tuple[int, QuantizationDataType], allowed_supported_kernels: List[QuantDtypeBwInfo], module_name):
            """
            helper function to validate both activation only against the supported_kernels passed
            :param act: act of the candidate to be validated
            :param allowed_supported_kernels: List of supported kernels for the given module
            :param module_name: name of the module
            """
            if action != SupportedKernelsAction.allow_error:
                for k in allowed_supported_kernels:
                    if k.is_same_activation(act[1], act[0]):
                        return

                if action == SupportedKernelsAction.warn_on_error:
                    logger.warning("activation:%s is not under the supported_kernels for the module %s", act, module_name)

                if action == SupportedKernelsAction.assert_on_error:
                    error_msg = f'activation: {act} is not under the supported_kernels for the module {module_name}'
                    raise RuntimeError(error_msg)

        # retrieve all the act and param quantizer candidates, and validate them against supported_kernels
        for name, module in self.named_qmodules():
            if getattr(module, 'supported_kernels', False):
                supported_kernels = []
                for supported_kernel in module.supported_kernels:
                    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
                    # TODO modify this once reformat_supported_kernels generates of type QuantDtypeBwInfo
                    if isinstance(supported_kernel[1], tuple):
                        supported_kernels.append(
                            QuantDtypeBwInfo(supported_kernel[0][1], supported_kernel[0][0],
                                             supported_kernel[1][1], supported_kernel[1][0]))
                    else:
                        supported_kernels.append(
                            QuantDtypeBwInfo(supported_kernel[1], supported_kernel[0]))
                act_candidates = []
                param_candidate = ()
                for quantizer in module.input_quantizers + module.output_quantizers:
                    act_candidates.append((quantizer.bitwidth, quantizer.data_type))

                if 'weight' in module.param_quantizers:
                    param_candidate = (module.param_quantizers['weight'].bitwidth,
                                       module.param_quantizers['weight'].data_type)

                if param_candidate:
                    # we need to check weights against all the activations
                    for act_candidate in set(act_candidates):
                        apply_act_param_rules(QuantDtypeBwInfo(act_candidate[1], act_candidate[0], param_candidate[1],
                                                               param_candidate[0]), supported_kernels, name)
                else:
                    for candidate in set(act_candidates):
                        apply_act_rules(candidate, supported_kernels, name)


    # pylint: disable=protected-access, too-many-branches, too-many-locals, import-outside-toplevel
    def _apply_exception_rules(self):
        """
        Apply exception rules to specific op. For example, a rule can override high bitwidth to Embedding module
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        from aimet_torch.v2.nn import BaseQuantizationMixin

        for wrapper in self.qmodules():
            if isinstance(wrapper, BaseQuantizationMixin):
                continue

            original_module = wrapper.get_original_module()

            if isinstance(original_module, torch.nn.Embedding):
                if self._hw_version not in {'V73', 'V75', 'V79'}:
                    continue
                weight_quantizer = wrapper.param_quantizers['weight']
                output_quantizer = wrapper.output_quantizers[0]

                weight_quantizer.bitwidth = output_quantizer.bitwidth
                weight_quantizer.use_symmetric_encodings = output_quantizer.use_symmetric_encodings

            elif isinstance(original_module, torch.nn.GroupNorm):
                if self._hw_version not in {'V73', 'V75', 'V79'}:
                    continue
                if 'weight' in wrapper.param_quantizers:
                    output_quantizer = wrapper.output_quantizers[0]
                    for _, param_quantizer in wrapper.param_quantizers.items():
                        param_quantizer.bitwidth = output_quantizer.bitwidth
                        param_quantizer.use_symmetric_encodings = output_quantizer.use_symmetric_encodings

            elif isinstance(original_module, MatMul):
                # Skip unused modules
                if original_module not in self.connected_graph._module_to_op_dict.keys():
                    continue

                first_input_quantizer, second_input_quantizer = wrapper.input_quantizers

                op = self.connected_graph._module_to_op_dict[original_module]
                first_input_op = op.inputs[0].producer if (not first_input_quantizer.enabled) else None
                second_input_op = op.inputs[1].producer if (not second_input_quantizer.enabled) else None

                target_quantizer_for_first_input = self._get_target_quantizer(first_input_quantizer, first_input_op)
                target_quantizer_for_second_input = self._get_target_quantizer(second_input_quantizer, second_input_op)

                # We don't need to apply exception rule when both first and second inputs are FP quantization
                if (
                    target_quantizer_for_first_input
                    and target_quantizer_for_first_input.data_type == QuantizationDataType.float
                    and target_quantizer_for_second_input
                    and target_quantizer_for_second_input.data_type == QuantizationDataType.float
                ):
                    continue

                # According to opdef for Matmul in HTP:
                # 16bit Weight(second input for dynamic MatMul) must have 16bit Activation(first input for dynamic MatMul).
                # 16bit Activation and 16bit Weight require minimum arch V73.
                # 16bit Weight must be symmetric quantized.

                # Below are the possible combinations for MatMul with 8/16 bitwidth:
                # If version is V73 and higher: {input0->8, input1->8 symm/asymm} {input0->16 , input1->8 symm/asymm} {input0->16, input1->16 symmetric}
                # If version is lesser than V73: {input0->8, input1->8 symmetric} {input0->16, input1->8 symmetric}

                if self._hw_version in {'V66', 'V68', 'V69'}:
                    if target_quantizer_for_second_input is None:
                        logger.warning("The target quantizer for second input could not be found. MatMul exception rule does not apply for layer: %s. "
                                       "If you haven't used model preparer, consider using it.", str(original_module))
                    else:
                        target_quantizer_for_second_input.use_symmetric_encodings = True
                        target_quantizer_for_second_input.bitwidth = 8
                elif self._hw_version in {'V73', 'V75', 'V79'}:
                    if target_quantizer_for_first_input is None or target_quantizer_for_second_input is None:
                        logger.warning("The target quantizers could not be found. MatMul exception rule does not apply for layer: %s. "
                                       "If you haven't used model preparer, consider using it.", str(original_module))
                    elif target_quantizer_for_second_input.bitwidth == 16:
                        target_quantizer_for_second_input.use_symmetric_encodings = True
                        target_quantizer_for_first_input.bitwidth = 16

    def _get_target_quantizer(self, input_quantizer: _QuantizerProtocol, input_op: Op) -> _QuantizerProtocol:
        """
        Returns input quantizer if enabled otherwise returns closest enabled parent output quantizer.

        :param input_quantizer: Input quantizer
        :param input_op: Input Op
        :return: Target quantizer
        """
        target_quantizer = None
        if input_quantizer.enabled:
            target_quantizer = input_quantizer
        elif input_op:
            closest_producer_wrapper = self._get_closest_producer_wrapper(input_op)
            if closest_producer_wrapper:
                target_quantizer = closest_producer_wrapper.output_quantizers[0] \
                        if closest_producer_wrapper.output_quantizers[0] else closest_producer_wrapper.input_quantizers[0]
            else:
                logger.warning("The closest wrapper could not be found. MatMul exception rule does not apply. "
                               "If you haven't used model preparer, consider using it.")
        return target_quantizer


    def _get_closest_producer_wrapper(self, op: Op) -> Optional[_QuantizedModuleProtocol]:
        """
        Find the closest producer QcQuantizeWrapper and return it

        :param op: Target operation
        :return: QcQuantizerWrapper if exists else None
        """
        wrapper = self._get_qmodule(op)
        if wrapper:
            if wrapper.output_quantizers[0].enabled or wrapper.input_quantizers[0].enabled:
                return wrapper

            if len(op.input_ops) == 1:
                return self._get_closest_producer_wrapper(op.input_ops[0])

            logger.warning("A wrapper of %s with output quantization disabled has no input or more than one input "
                           "exists. It's ambiguous to find the nearest producer in this case", str(op.get_module()))
            return None

        if not op.input_ops:
            logger.warning("No input exists for navigation for traversal, it's not possible to find the closest producer")
            return None

        if len(op.input_ops) > 1:
            logger.warning("Multiple input ops exist, traversal to find closest producer is performed based on the "
                           "first input")

        return self._get_closest_producer_wrapper(op.input_ops[0])

    def _get_qmodule(self, op: Op) -> Optional[_QuantizedModuleProtocol]:
        orig_module = op.get_module()
        if not orig_module:
            return None

        full_name = self.connected_graph._module_to_name[orig_module] # pylint: disable=protected-access
        _, *module_names = full_name.split('.')

        if not module_names:
            return None

        module_name = '.'.join(module_names)
        return utils.get_named_module(self.model, module_name)

    def export(self, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple], # pylint: disable=arguments-differ
               onnx_export_args: Optional[Union[OnnxExportApiArgs, Dict]] = None, propagate_encodings: bool = False,
               export_to_torchscript: bool = False, use_embedded_encodings: bool = False, export_model: bool = True,
               filename_prefix_encodings: str = None):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved:

        1. The sim-model is exported to a regular PyTorch model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)
        3. Optionally, An equivalent model in ONNX format is exported. In addition, nodes in the ONNX model are named
           the same as the corresponding PyTorch module names. This helps with matching ONNX node to their quant
           encoding from #2.

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph. It is required for the dummy_input to
                be placed on CPU.
        :param onnx_export_args: Optional export argument with onnx specific overrides provided as a dictionary or
            OnnxExportApiArgs object. If not provided, defaults to "opset_version" = None, "input_names" = None,
            "output_names" = None, and for torch version < 1.10.0, "enable_onnx_checker" = False.
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops. Defaults to False.
        :param export_to_torchscript: If True, export to torchscript. Export to onnx otherwise. Defaults to False.
        :param use_embedded_encodings: If True, another onnx model embedded with fakequant nodes will be exported
        :param export_model: If True, then ONNX model is exported. When False, only encodings are exported. User should
                            disable (False) this flag only if the corresponding ONNX model already exists in the path
                            specified
        :param filename_prefix_encodings: File name prefix to be used when saving encodings.
                                          If None, then user defaults to filename_prefix value
        """
        if quantsim.encoding_version == '0.6.1':
            msg = _red("Encoding version 0.6.1 will be deprecated in a future release, with version 1.0.0 becoming "
                       "the default. If your code depends on parsing the exported encodings file, ensure that it is "
                       "updated to be able to parse 1.0.0 format.\n"
                       "To swap the encoding version to 1.0.0, run the following lines prior to calling quantsim "
                       "export:\n\n"
                       "from aimet_common import quantsim\n"
                       "quantsim.encoding_version = '1.0.0'")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        if not filename_prefix_encodings:
            filename_prefix_encodings = filename_prefix

        if quantsim.encoding_version not in VALID_ENCODING_VERSIONS:
            raise NotImplementedError(f'Encoding version {quantsim.encoding_version} not in set of valid encoding '
                                      f'versions {VALID_ENCODING_VERSIONS}.')
        # save the quantized model and encodings
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # Create a version of the model without any quantization ops
        model_to_export = self.get_original_model(self.model, qdq_weights=True)

        torch.save(model_to_export, model_path)

        if onnx_export_args is None:
            onnx_export_args = {'opset_version': None,
                                'input_names': None,
                                'output_names': None}
            if version.parse(torch.__version__) < version.parse("1.10.0") and isinstance(onnx_export_args, dict):
                onnx_export_args['enable_onnx_checker'] = False
        log_with_error_and_assert_if_false(isinstance(onnx_export_args, (OnnxExportApiArgs, dict)),
                                           logger,
                                           f'unsupported opt_args type={type(onnx_export_args)}')

        if use_embedded_encodings:
            self.save_model_with_embedded_quantization_nodes(self.model, path, filename_prefix, dummy_input,
                                                             onnx_export_args, export_to_torchscript, self._is_conditional)
        else:
            if export_to_torchscript:
                self.export_torch_script_model_and_encodings(path, filename_prefix, filename_prefix_encodings,
                                                             model_to_export, self.model,
                                                             dummy_input,
                                                             self._excluded_layer_names)
            else:
                self.export_onnx_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                                     dummy_input, onnx_export_args, propagate_encodings,
                                                     self._module_marker_map, self._is_conditional,
                                                     self._excluded_layer_names, quantizer_args=self.quant_args,
                                                     export_model=export_model,
                                                     filename_prefix_encodings=filename_prefix_encodings)

    # pylint: disable=missing-function-docstring
    @classmethod
    @abstractmethod
    def save_model_with_embedded_quantization_nodes(cls, sim_model, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
                                                    onnx_export_args: Optional[Union[OnnxExportApiArgs, Dict]] = None,
                                                    export_to_torchscript: bool = False, is_conditional: bool = False):
        ...

    @classmethod
    def export_torch_script_model_and_encodings(cls, path: str, filename_prefix: str,
                                                filename_prefix_encodings: str,
                                                original_model: torch.nn.Module,
                                                sim_model: torch.nn.Module,
                                                dummy_input: Union[torch.Tensor, Tuple],
                                                excluded_layer_names: List = None):
        """
        This method exports a torchscript mode and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param filename_prefix_encodings: File name prefix for encodings. Can be same as filename_prefix
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :return: None
        """
        # Create torchscript model and obtain node to i/o tensor name map
        ts_path = os.path.join(path, filename_prefix + '.torchscript.pth')
        with utils.in_eval_mode(original_model), torch.no_grad():
            torchscript_utils.create_torch_script_model(ts_path, original_model, dummy_input)

            trace = torch.jit.load(ts_path)
            torch_script_node_io_tensor_map, valid_param_set = \
                torchscript_utils.get_node_to_io_tensor_names_map(original_model, trace, dummy_input)

        # Export encodings
        cls._export_encodings_to_files(sim_model, path, filename_prefix_encodings,
                                       torch_script_node_io_tensor_map, valid_param_set,
                                       excluded_layer_names, propagate_encodings=False)

    @classmethod
    def export_onnx_model_and_encodings(cls, path: str, filename_prefix: str, original_model: torch.nn.Module,
                                        sim_model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                                        onnx_export_args: Union[OnnxExportApiArgs, dict], propagate_encodings: bool,
                                        module_marker_map: Dict[torch.nn.Module, torch.Tensor] = None,
                                        is_conditional: bool = False, excluded_layer_names: List = None,
                                        quantizer_args: Dict = None, export_model: bool = True,
                                        filename_prefix_encodings: str = None):
        """
        This method exports a onnx model and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param onnx_export_args: Additional onnx export args including export api overrides
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
               multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
               ops.
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param is_conditional: True if model is conditional, False otherwise
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :param export_model: If True, then ONNX model is exported. When False, only encodings are exported. User should
                            disable (False) this flag only if the corresponding ONNX model already exists in the path
                            specified
        :param filename_prefix_encodings: File name prefix to be used when saving encodings.
                                          If None, then user defaults to filename_prefix value
        :return: None

        """
        # pylint: disable=too-many-locals
        if not filename_prefix_encodings:
            filename_prefix_encodings = filename_prefix
        onnx_path = os.path.join(path, filename_prefix + '.onnx')
        if export_model:
            OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_path, original_model, dummy_input, is_conditional,
                                                                 module_marker_map, onnx_export_args)

        assert os.path.exists(onnx_path), 'The onnx model does not exist in the location specified. Please re-run export' \
                                          'with export_model flag as True or check path/file_name'
        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, valid_param_set = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        # Export encodings
        cls._export_encodings_to_files(sim_model, path, filename_prefix_encodings,
                                       onnx_node_to_io_tensor_map, valid_param_set,
                                       excluded_layer_names, propagate_encodings,
                                       quantizer_args=quantizer_args)

    def export_weights_to_safetensors(self, path: str, filename_prefix: str):
        """
        Exports the updated weights in the safetensors format

        :param path: Path to save file
        :param filename_prefix: Filename to use for saved file
        """

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # Save state dict in safetensors file
        unwrapped_model = self.get_original_model(self.model)
        data = unwrapped_model.state_dict()
        data = {k: to_numpy(v) for k, v in data.items()}
        metadata = self.model.mpp_meta if hasattr(self.model, 'mpp_meta') else {}

        file_path = os.path.join(path, filename_prefix + '.safetensors')
        save_safetensor_file(data, file_path, metadata)

    def save_encodings_to_json(self, path: str, filename_prefix: str):
        """
        Save encodings in the model to json.

        :param path: Path to save file
        :param filename_prefix: Filename to use for saved file
        """
        activation_encodings, param_encodings = self.get_activation_param_encodings()
        encodings_dict = {'activation_encodings': activation_encodings, 'param_encodings': param_encodings}
        with open(os.path.join(path, filename_prefix + '.json'), 'w') as encoding_json:
            json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

    def get_activation_param_encodings(self):
        """
        Get activation and param encodings from sim.model.

        :return: Tuple of activation and param encodings dictionaries mapping torch module names to encodings
        """
        activation_encodings = OrderedDict()
        param_encodings = OrderedDict()

        for module_name, module in self.model.named_modules():
            if not isinstance(module, _QuantizedModuleProtocol):
                continue

            activation_encodings[module_name] = defaultdict(OrderedDict)

            for i, encoding in enumerate(module.export_input_encodings(encoding_version='0.6.1')):
                if not encoding:
                    continue
                if len(encoding) == 1:
                    encoding = encoding[0]
                activation_encodings[module_name]['input'][i] = encoding

            for i, encoding in enumerate(module.export_output_encodings(encoding_version='0.6.1')):
                if not encoding:
                    continue
                if len(encoding) == 1:
                    encoding = encoding[0]
                activation_encodings[module_name]['output'][i] = encoding

            if not activation_encodings[module_name]:
                del activation_encodings[module_name]

            for param_name, encoding in module.export_param_encodings(encoding_version='0.6.1').items():
                if not encoding:
                    continue
                param_encodings[f'{module_name}.{param_name}'] = encoding

        return activation_encodings, param_encodings

    @classmethod
    def get_original_model(cls, model: torch.nn.Module, qdq_weights: bool = False) -> torch.nn.Module:
        """
        This function returns the model with all quantization wrappers removed.

        :param model: The input model with quantization wrappers.
        :param qdq_weights: Whether to replace weights inside model by qdq weights.
        :return: Model without quantization wrappers.
        """
        original_model = copy.deepcopy(model)
        if qdq_weights:
            cls._apply_qdq_to_model_parameters(original_model)
        # pylint: disable=unnecessary-comprehension
        all_modules_in_original_model = [module for module in original_model.modules()]
        cls._remove_quantization_wrappers(original_model, all_modules_in_original_model)
        return original_model

    @classmethod
    @abstractmethod
    def _remove_quantization_wrappers(cls, starting_module, list_of_modules_to_exclude):
        ...

    @classmethod
    @abstractmethod
    def _apply_qdq_to_model_parameters(cls, model: torch.nn.Module):
        ...

    _quantized_modules: Tuple[Type, ...]

    def exclude_layers_from_quantization(self, layers_to_exclude: List[torch.nn.Module]):
        """
        Excludes certain layers from being quantized-dequantized by the simulator
        :param layers_to_exclude: List of torch layers to exclude
        :return: None
        """
        # Save the excluded layer names. Do not save the modules since the wrapper removal depends on
        # reference count to automatically remove the layers.
        module_to_name_dict = utils.get_module_to_name_dict(self.model)
        quant_layers_to_exclude = []
        for layer in layers_to_exclude:
            for module in layer.modules():
                if isinstance(module, self._quantized_modules):
                    quant_layers_to_exclude.append(module)
                    excluded_module_name = module_to_name_dict.get(module)
                    self._excluded_layer_names.append(excluded_module_name)

        self._remove_quantization_wrappers(self.model, quant_layers_to_exclude)

    @staticmethod
    def _get_torch_encodings_for_missing_layers(layer: _QuantizedModuleProtocol, layer_name: str,# pylint: disable=too-many-branches
                                                missing_activation_encodings_torch: Dict,
                                                missing_param_encodings: Dict,
                                                valid_param_set: set):
        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting torch encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param missing_activation_encodings_torch: dictionary of activation encodings which maps pytorch names to encodings
        :param missing_param_encodings: dictionary of param encodings
        :param valid_param_set: a set of valid param input names in model
        """
        if isinstance(layer, _QuantizedModuleProtocol):
            # --------------------------------------
            # Update encodings for Input activations
            # --------------------------------------
            input_encodings = layer.export_input_encodings(encoding_version='0.6.1')
            # skip layer if it has no input encodings.
            if all(encoding is None for encoding in input_encodings):
                return

            for index, encoding in enumerate(input_encodings):
                if encoding is not None:
                    if layer_name not in missing_activation_encodings_torch:
                        missing_activation_encodings_torch[layer_name] = {}
                    if 'input' not in missing_activation_encodings_torch[layer_name]:
                        missing_activation_encodings_torch[layer_name]['input'] = {}
                    # Store encodings for a particular index so that they can be used to check if a quantizer was
                    # enabled or not
                    missing_activation_encodings_torch[layer_name]['input'][index] = encoding[0]

            # ---------------------------------------
            # Update encodings for output activations
            # ---------------------------------------
            output_encodings = layer.export_output_encodings(encoding_version='0.6.1')
            for index, encoding in enumerate(output_encodings):
                if encoding is not None:
                    if layer_name not in missing_activation_encodings_torch:
                        missing_activation_encodings_torch[layer_name] = {}
                    if 'output' not in missing_activation_encodings_torch[layer_name]:
                        missing_activation_encodings_torch[layer_name]['output'] = {}
                    missing_activation_encodings_torch[layer_name]['output'][index] = encoding[0]

            # ---------------------------
            # Update encodings for Params
            # ---------------------------
            for orig_param_name, param_encoding in layer.export_param_encodings(encoding_version='0.6.1').items():
                param_name = layer_name + '.' + orig_param_name
                if param_encoding is None:
                    continue
                if param_name not in valid_param_set:
                    logger.error('Param tensor {%s} not found in valid param set', param_name)
                    continue
                missing_param_encodings[param_name] = param_encoding

    # pylint: disable=too-many-branches
    @classmethod
    def _export_encodings_to_files(cls, sim_model: torch.nn.Module, path: str, filename_prefix: str,
                                   op_to_io_tensor_map: Dict, valid_param_set: set, excluded_layer_names,
                                   propagate_encodings: bool, quantizer_args: Dict = None):
        """
        Save the quantized model weight encodings

        :param sim_model: Quantsim model to export encodings for
        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param op_to_io_tensor_map: Dictionary of layer to I/O tensor mapping from onnx or torch script model
        :param valid_param_set: a set of valid param input names in model
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :param quantizer_args
        """

        # pylint: disable=too-many-locals

        # Create a dictionary to export to JSON
        activation_encodings_onnx = {}
        activation_encodings_torch = {}
        missing_activation_encodings_torch = {}
        param_encodings = {}
        missing_param_encodings = {}
        layers_to_onnx_op_names = onnx_utils.get_layers_in_io_tensor_map(op_to_io_tensor_map)
        tensor_to_consumer_map = onnx_utils.get_tensor_to_consumer_map(op_to_io_tensor_map)
        layer_names_not_found = []
        tensor_to_quantizer_map = {}

        for layer_name, layer in sim_model.named_modules():
            if not isinstance(layer, cls._quantized_modules):
                continue
            # TODO: specifically call out dropout layers here since they are specifically switched out during export.
            # These ops should eventually be reworked as part of math invariant ops to ignore quantization altogether.
            # pylint: disable=protected-access
            if isinstance(layer, _QuantizedModuleProtocol) and isinstance(layer.get_original_module(), utils.DROPOUT_TYPES):
                continue

            if layer_name not in layers_to_onnx_op_names.keys():
                layer_names_not_found.append(layer_name)
                # Some layers like transpose etc. may get removed after onnx export due to internal onnx optimization.
                # An error will be thrown if the exported encodings(without missing layers's encoding) are loaded back
                # to the sim model because the layers will be present in the sim model whereas the
                # corresponding encodings will not be present in the encoding file.
                cls._get_torch_encodings_for_missing_layers(layer, layer_name,
                                                            missing_activation_encodings_torch,
                                                            missing_param_encodings,valid_param_set)
            else:
                cls._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings_onnx,
                                                     activation_encodings_torch,
                                                     param_encodings, op_to_io_tensor_map,
                                                     valid_param_set, propagate_encodings,
                                                     tensor_to_consumer_map, layers_to_onnx_op_names,
                                                     tensor_to_quantizer_map)

        if layer_names_not_found:
            logger.warning("The following layers were not found in the exported onnx model. Encodings for these layers"
                           " will not appear in the exported encodings file, however it will continue to"
                           " exist in torch encoding file:\n"
                           "%s\n"
                           "This can be due to several reasons:\n"
                           "\t- The layer is set to quantize with float datatype, but was not exercised in compute "
                           "encodings. Not an issue if the layer is not meant to be run.\n"
                           "\t- The layer has valid encodings but was not seen while exporting to onnx using the dummy "
                           "input provided in sim.export(). Ensure that the dummy input covers all layers.",
                           layer_names_not_found)

        if quantsim.encoding_version == '0.6.1':
            encodings_dict_onnx = {'version': quantsim.encoding_version,
                                   'activation_encodings': activation_encodings_onnx,
                                   'param_encodings': param_encodings,
                                   'excluded_layers': excluded_layer_names}

            if quantizer_args:
                encodings_dict_onnx.update({'quantizer_args': quantizer_args})

            logger.info("Layers excluded from quantization: %s", excluded_layer_names)

            # export weight encodings to output json file
            encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
            save_json_yaml(encoding_file_path, encodings_dict_onnx)
        else:
            _export_to_1_0_0(path, filename_prefix, activation_encodings_onnx, param_encodings, tensor_to_quantizer_map,
                             excluded_layer_names, quantizer_args)

        logger.warning(_red("Quantsim export will stop exporting encodings for saving and loading in a future AIMET "
                       "release.\nTo export encodings for saving and loading, use QuantizationSimModel's "
                       "save_encodings_to_json() utility instead."))
        if not SKIP_TORCH_ENCODINGS_EXPORT:
            # Export torch.encodings used for saving/loading common to 0.6.1 and 1.0.0 versions (uses 0.6.1 version).
            if quantsim.encoding_version == '0.6.1':
                param_encodings.update(missing_param_encodings)
                param_encodings_torch = param_encodings
            else:
                param_encodings_torch = {}
                for module_name, module in sim_model.named_modules():
                    if not isinstance(module, _QuantizedModuleProtocol):
                        continue
                    for param_name, encoding in module.export_param_encodings(encoding_version='0.6.1').items():
                        if not encoding:
                            continue
                        param_encodings_torch[f'{module_name}.{param_name}'] = encoding

            activation_encodings_torch.update(missing_activation_encodings_torch)
            encodings_dict_pytorch = {'version': quantsim.encoding_version,
                                      'activation_encodings': activation_encodings_torch,
                                      'param_encodings': param_encodings_torch,
                                      'excluded_layers': excluded_layer_names}

            if quantizer_args:
                encodings_dict_pytorch.update({'quantizer_args': quantizer_args})

            encoding_file_path_pytorch = os.path.join(path, filename_prefix + '_torch' + '.encodings')
            save_json_yaml(encoding_file_path_pytorch, encodings_dict_pytorch)

    @staticmethod
    def _update_param_encodings_dict_for_layer(layer: _QuantizedModuleProtocol, layer_name: str, param_encodings: Dict,
                                               valid_param_set: set, tensor_to_quantizer_map: Dict):
        """
        :param layer: layer as torch.nn.Module
        :param layer_name : Name of the layer
        :param param_encodings: dictionary of param encodings
        :param valid_param_set: a set of valid param input names in model
        """

        for orig_param_name, param_encoding in layer.export_param_encodings(quantsim.encoding_version).items():
            param_name = layer_name + '.' + orig_param_name
            if param_encoding is None:
                continue
            if param_name not in valid_param_set:
                logger.error('Param tensor {%s} not found in valid param set', param_name)
                continue
            param_encodings[param_name] = param_encoding
            tensor_to_quantizer_map[param_name] = layer.param_quantizers[orig_param_name]

    @classmethod
    def _update_encoding_dicts_for_layer(cls, layer: _QuantizedModuleProtocol, layer_name: str, activation_encodings_onnx: Dict,
                                         activation_encodings_torch: Dict, param_encodings: Dict,
                                         op_to_io_tensor_map: Dict, valid_param_set: set, propagate_encodings: bool,
                                         tensor_to_consumer_map: Dict[str, str],
                                         layers_to_onnx_op_names: Dict[str, str],
                                         tensor_to_quantizer_map: Dict):
        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param activation_encodings_onnx: dictionary of activation encodings which maps onnx attribute to encodings
        :param activation_encodings_torch: dictionary of activation encodings which maps pytorch names to encodings
        :param param_encodings: dictionary of param encodings
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param valid_param_set: a set of valid param input names in model
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        """

        if isinstance(layer, _QuantizedModuleProtocol):

            # --------------------------------------
            # Update encodings for Input activations
            # --------------------------------------
            cls._update_encoding_dict_for_input_activations(layer, layer_name, op_to_io_tensor_map,
                                                            activation_encodings_onnx,
                                                            activation_encodings_torch,
                                                            layers_to_onnx_op_names,
                                                            tensor_to_quantizer_map)
            # ---------------------------------------
            # Update encodings for output activations
            # ---------------------------------------
            cls._update_encoding_dict_for_output_activations(layer, layer_name,
                                                             op_to_io_tensor_map,
                                                             activation_encodings_onnx,
                                                             activation_encodings_torch,
                                                             propagate_encodings,
                                                             tensor_to_consumer_map,
                                                             layers_to_onnx_op_names,
                                                             tensor_to_quantizer_map)
            # ---------------------------
            # Update encodings for Params
            # ---------------------------
            cls._update_param_encodings_dict_for_layer(layer, layer_name, param_encodings,
                                                       valid_param_set, tensor_to_quantizer_map)


    @staticmethod
    def find_op_names_for_layer(layer_name: str, op_to_io_tensor_map: Dict,
                                tensor_to_consumer_map: Optional[Dict[str, str]],
                                layers_to_onnx_op_names: Optional[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """
        This function returns the last ONNX op and the list of ONNX Ops that were mapped from a PyTorch Op.

        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: tuple(end op names, all op names)
        """
        if version.parse(torch.__version__) < version.parse("1.13.0") or not onnx_utils.EXPORT_TO_ONNX_DIRECT:
            op_names = [key for key in op_to_io_tensor_map if (key.startswith(layer_name) and layer_name+'#' in key)
                        or key == layer_name]
            if len(op_names) == 1:
                return op_names, op_names

            end_op_names = [op_name for op_name in op_names if op_name.endswith('.end')]
            return end_op_names, op_names

        assert tensor_to_consumer_map is not None
        assert layers_to_onnx_op_names is not None
        # Get all ops which correspond to the current PyTorch layer being processed.
        op_names = layers_to_onnx_op_names.get(layer_name, [])
        op_name_set = set(op_names)

        end_op_names = []
        end_op_names_set = set()
        for op_name in op_names:
            # Loop through outputs of each op and check whether the output leads to an op not in
            for output in op_to_io_tensor_map[op_name].outputs:
                assert output in tensor_to_consumer_map.keys()
                if not tensor_to_consumer_map[output]:
                    if op_name not in end_op_names_set:
                        # output has no consumers, and can either be a model output or an unused op output.
                        # List it as an end_op_name all the same.
                        end_op_names.append(op_name)
                        end_op_names_set.add(op_name)
                else:
                    for consumer in tensor_to_consumer_map[output]:
                        if consumer not in op_name_set and op_name not in end_op_names_set:
                            end_op_names.append(op_name)
                            end_op_names_set.add(op_name)

        return end_op_names, op_names

    @classmethod
    def _update_encoding_dict_for_output_activations(cls, layer: _QuantizedModuleProtocol, layer_name: str, op_to_io_tensor_map: Dict,
                                                     activation_encodings_onnx: Dict, activation_encodings_torch: Dict,
                                                     propagate_encodings: bool, tensor_to_consumer_map: Dict[str, str],
                                                     layers_to_onnx_op_names: Dict[str, str],
                                                     tensor_to_quantizer_map: Dict):
        # pylint: disable=too-many-locals
        output_tensors, propagate_tensors = cls._get_layer_activation_tensors(layer_name,
                                                                              op_to_io_tensor_map,
                                                                              tensor_to_consumer_map,
                                                                              layers_to_onnx_op_names)
        output_encodings = layer.export_output_encodings(quantsim.encoding_version)

        if len(output_tensors) != len(output_encodings):
            logger.warning("number of output quantizers: %d available for layer: %s "
                           "doesn't match with number of output tensors: %d", len(output_encodings), layer_name,
                           len(output_tensors))

        legacy_output_encodings = None
        if not SKIP_TORCH_ENCODINGS_EXPORT:
            if quantsim.encoding_version == '0.6.1':
                legacy_output_encodings = output_encodings
            else:
                legacy_output_encodings = layer.export_output_encodings(encoding_version='0.6.1')
        for index, (output_tensor, encoding) in enumerate(zip(output_tensors, output_encodings)):
            if encoding is not None:
                activation_encodings_onnx[output_tensor] = encoding
                tensor_to_quantizer_map[output_tensor] = layer.output_quantizers[index]
                if not SKIP_TORCH_ENCODINGS_EXPORT and legacy_output_encodings[index] is not None:
                    legacy_encoding = legacy_output_encodings[index]
                    if layer_name not in activation_encodings_torch:
                        activation_encodings_torch[layer_name] = {}
                    if 'output' not in activation_encodings_torch[layer_name]:
                        activation_encodings_torch[layer_name]['output'] = {}
                    activation_encodings_torch[layer_name]['output'][index] = legacy_encoding[0]

        if propagate_encodings:
            valid_encodings = [enc for enc in output_encodings if enc is not None]
            if valid_encodings:
                encoding = valid_encodings[0]
                for activation_tensor in propagate_tensors:
                    activation_encodings_onnx[activation_tensor] = utils.get_propagated_encoding_dict(encoding)


    @classmethod
    def _update_encoding_dict_for_input_activations(cls, layer: _QuantizedModuleProtocol, layer_name: str, op_to_io_tensor_map: Dict,
                                                    activation_encodings_onnx: Dict, activation_encodings_torch: Dict,
                                                    layers_to_onnx_op_names: Dict[str, str],
                                                    tensor_to_quantizer_map: Dict):
        input_encodings = layer.export_input_encodings(quantsim.encoding_version)
        # skip layer if it has no input encodings.
        if all(encoding is None for encoding in input_encodings):
            return

        input_tensors = cls._get_layer_input_tensors(layer, layer_name, op_to_io_tensor_map,
                                                     layers_to_onnx_op_names)

        if len(input_tensors) != len(input_encodings):
            logger.warning("number of input quantizers: %d available for layer: %s "
                           "doesn't match with number of input tensors: %d", len(input_encodings), layer_name,
                           len(input_tensors))

        legacy_input_encodings = None
        if not SKIP_TORCH_ENCODINGS_EXPORT:
            if quantsim.encoding_version == '0.6.1':
                legacy_input_encodings = input_encodings
            else:
                legacy_input_encodings = layer.export_input_encodings(encoding_version='0.6.1')
        for index, (input_tensor, encoding) in enumerate(zip(input_tensors, input_encodings)):
            if encoding is not None:
                activation_encodings_onnx[input_tensor] = encoding
                # TODO: Modify this so quantsim does not make assumptions about the length of input_quantizers
                tensor_to_quantizer_map[input_tensor] = layer.input_quantizers[min(index, len(layer.input_quantizers) - 1)]
                if not SKIP_TORCH_ENCODINGS_EXPORT and legacy_input_encodings[index] is not None:
                    legacy_encoding = legacy_input_encodings[index]
                    # Check if layer exists in the pytorch encoding dictionary
                    if layer_name not in activation_encodings_torch:
                        activation_encodings_torch[layer_name] = {}
                    if 'input' not in activation_encodings_torch[layer_name]:
                        activation_encodings_torch[layer_name]['input'] = {}
                    # Store encodings for a particular index so that they can be used to check if a quantizer was
                    # enabled or not
                    activation_encodings_torch[layer_name]['input'][index] = legacy_encoding[0]

    @classmethod
    def _get_layer_input_tensors(cls, layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                 layers_to_onnx_op_names: Dict[str, str] = None) -> List[str]:
        """
        This function returns the list of input tensor names mapped from a PyTorch Op.

        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: list of input tensor names.
        """

        param_inputs = [layer_name + '.' + param_name for param_name, _ in layer.named_parameters()]
        for idx, param_input in enumerate(param_inputs):
            param_inputs[idx] = ''.join(param_input.split('._module_to_wrap'))
        if version.parse(torch.__version__) < version.parse("1.13.0") or not onnx_utils.EXPORT_TO_ONNX_DIRECT:
            start_op_names = [key for key in op_to_io_tensor_map
                              if (key.startswith(layer_name) and '#0' in key) or key == layer_name]
        else:
            assert layers_to_onnx_op_names is not None
            op_names = layers_to_onnx_op_names.get(layer_name, [])
            onnx_op_outputs = set()
            for op_name in op_names:
                for op_output in op_to_io_tensor_map[op_name].outputs:
                    onnx_op_outputs.add(op_output)

            start_op_names = set()
            for op_name in op_names:
                # For each op's inputs, if the input comes from an op not associated with this layer, add it to
                # start_op_names.
                for inp in op_to_io_tensor_map[op_name].inputs:
                    if inp not in onnx_op_outputs and inp not in param_inputs:
                        start_op_names.add(op_name)

        input_tensors = []
        input_tensors_set = set()
        for name in start_op_names:
            for input_tensor in op_to_io_tensor_map[name].inputs:
                if input_tensor not in param_inputs and input_tensor not in input_tensors_set:
                    input_tensors.append(input_tensor)
                    input_tensors_set.add(input_tensor)

        return input_tensors

    @classmethod
    def _get_layer_activation_tensors(cls, layer_name: str, op_to_io_tensor_map: Dict,
                                      tensor_to_consumer_map: Dict[str, str] = None,
                                      layers_to_onnx_op_names: Dict[str, str] = None) -> Tuple[List[str], List[str]]:
        """
        This function returns the list of output tensor and intermediate tensor names mapped from a PyTorch Op.

        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: tuple containing list of output tensor names and list of intermediate tensors
        """
        end_op_names, op_names = cls.find_op_names_for_layer(layer_name, op_to_io_tensor_map, tensor_to_consumer_map,
                                                             layers_to_onnx_op_names)

        if len(end_op_names) > 1:
            output_op_map_str = cls._get_output_map_str(end_op_names, layer_name, op_to_io_tensor_map)
            logger.info("layer_name: %s, has multiple output onnx ops: %s", layer_name, output_op_map_str)

        output_tensors = []
        intermediate_tensors = []
        for name in op_names:
            if name in end_op_names:
                output_tensors.extend(op_to_io_tensor_map[name].outputs)
            else:
                intermediate_tensors.extend(op_to_io_tensor_map[name].outputs)

        return output_tensors, intermediate_tensors

    @staticmethod
    def _get_output_map_str(end_op_names, layer_name, op_to_io_tensor_map) -> str:
        """
        This function returns formatted list of output ops tensor mapping

        :param end_op_names: list of output onnx ops
        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :return: formatted string with output ops and their corresponding output count.
        """
        num_output_ops = len(end_op_names)
        op_map_str = ','.join([f'{name.replace(layer_name, "")}:{len(op_to_io_tensor_map[name].outputs)}'
                               for name in end_op_names[:5]])
        if num_output_ops > 5:
            op_map_str += ', ..'
        return f'{num_output_ops},[{op_map_str}]'

    @staticmethod
    def _update_parameters_by_attr(module: torch.nn.Module):
        """
        Updates the internal parameters of a PyTorch module by its attributes
        and remove those attributes from module.__dict__ to avoid onnx export error.

        :param module: The PyTorch module whose parameters need to be updated.
        """
        # pylint: disable=protected-access
        for param_name, _ in module.named_parameters():
            if param_name in module.__dict__ and param_name in module._parameters:
                module._parameters[param_name] = module.__dict__[param_name]
                module.__dict__.pop(param_name)

    def _get_leaf_module_to_name_map(self):
        """
        Returns a mapping from leaf modules to module name, where any _QuantizedModuleProtocol is considered a leaf module,
        and is therefore not further recursed (since we do not want to retrieve all internal quantizers/modules).
        """
        def recursively_populate_map(starting_module, module_map, start_str):
            for name, module in starting_module.named_children():
                if isinstance(module, _QuantizedModuleProtocol) or utils.is_leaf_module(module):
                    module_map[module] = start_str + name
                else:
                    recursively_populate_map(module, module_map, start_str + name + ".")
        module_to_name_map = {}
        recursively_populate_map(self.model, module_to_name_map, "")
        return module_to_name_map

    def _add_inputs_hook(self, hooks):
        module_to_name_map = self._get_leaf_module_to_name_map()

        def inputs_hook(module_ref, inputs, _):
            # Need to remove hook here, otherwise the jit trace of CustomMarker with module ref will error since the
            # hook will be recursively hit.
            hooks[module_ref].remove()
            del hooks[module_ref]
            module_name = module_to_name_map[module_ref]
            if isinstance(module_ref, _QuantizedModuleProtocol):
                module_ref = module_ref.get_original_module()
            marker_layer = torch.jit.trace(CustomMarker(module_ref, module_name, 'True'),
                                           inputs)
            self._module_marker_map[module_name] = marker_layer

        for name, module in self.model.named_modules():
            if name in module_to_name_map.values():
                hooks[module] = module.register_forward_hook(inputs_hook)

    def _validate_module_marker_map(self):
        """
        Check to make sure all leaf modules have traced Custom Markers associated with them.
        """
        all_leaf_modules = self._get_leaf_module_to_name_map().values()
        missing_inputs_entries = []

        for leaf_module in all_leaf_modules:
            if leaf_module not in self._module_marker_map.keys():
                missing_inputs_entries.append(leaf_module)

        if missing_inputs_entries:
            logger.info('In order to export a conditional model, all leaf modules need to be run with some input so '
                        'torch trace can be done.')
            logger.info('The following modules were not run during compute encodings:')
            logger.info(missing_inputs_entries)
            logger.info('Please use the sim.run_modules_for_traced_custom_marker(<module list>, dummy_input) api to '
                        'pass dummy inputs to these modules.')
            logger.info('Modules which can take the same dummy input can be '
                        'grouped as a list. For groups of modules with different input shapes, please call '
                        'sim.run_modules_for_traced_custom_markers() for each group.')
            logger.info('Exiting quantsim export early.')
            return False
        return True

    def _export_conditional(self, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
                            forward_pass_callback: Callable, forward_pass_callback_args,
                            onnx_export_args: Union[OnnxExportApiArgs, None] = OnnxExportApiArgs(),
                            propagate_encodings: bool = False):
        """
        Export function for conditional models. Performs another round of forward passes to create and store traced
        CustomMarker info for each leaf module to be later used when scripting the model for export.
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph. It is required for the dummy_input to
                be placed on CPU.
        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings. The callback should exercise all paths of the conditional model.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        :param onnx_export_args: onnx specific export arguments
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :return: None
        """
        self._is_conditional = True
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()

        # If model is conditional, we need to create traced CustomMarkers to be used later during export. Create hooks
        # here for creating a traced CustomMarker for each leaf module during the forward pass callback.
        hooks = {}
        if self._is_conditional:
            self._add_inputs_hook(hooks)

        with utils.in_eval_mode(self.model), torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        # Any hooks that were hit during forward pass callback would have removed themselves. Remove the remaining
        # hooks that were not run.
        for h in hooks.values():
            h.remove()

        # Check that all paths were exercised
        if not self._validate_module_marker_map():
            return
        self.export(path, filename_prefix, dummy_input, onnx_export_args, propagate_encodings)

    def load_encodings(self, encodings: Union[Mapping, str, os.PathLike], # pylint: disable=arguments-differ
                       strict: bool = True,
                       partial: bool = True,
                       requires_grad: Optional[bool] = None,
                       allow_overwrite: bool = True):
        """
        :param encodings: Encoding dictionary or path to the encoding dictionary json file.
        :param bool strict: If True, an error will be thrown if the model doesn't
            have a quantizer corresponding to the specified encodings.
        :param bool partial: If True, the encoding will be interpreted as a partial encoding,
            and the dangling quantizers with no corresponding encoding will be kept untouched.
            Otherwise, the dangling quantizers will be removed from the model.
        :param bool requires_grad: Whether or not the quantization parameters loaded from the
            encodings require gradient computation during training.
            If None, ``requires_grad`` flag of the quantization parameters will be kept unchanged.
        :param bool allow_overwrite: Whether or not the quantization parameters loaded from the
            encodings can be overwriiten by :ref:`compute_encodings` or another :ref:`load_encodings`.
            If None, whether the quantizer is overwrieable will be kept unchanged.
        """
        if isinstance(encodings, (str, os.PathLike)):
            with open(encodings, mode='r') as f:
                encodings = json.load(f)

        self._load_encodings_impl(encodings, strict, partial, requires_grad, allow_overwrite)

    def _load_encodings_impl(self, encodings: Mapping,
                             strict: bool,
                             partial: bool,
                             requires_grad: Optional[bool],
                             allow_overwrite: bool):
        if 'param_encodings' not in encodings:
            param_encodings = encodings
            activation_encodings = {}
            logger.warning("An older AdaRound exported encoding file type has been detected! "
                           "Please regenerate it using the AdaRound export function from the latest "
                           "AIMET (version 1.32 or higher) if necessary. "
                           "Support for this encoding file will be deprecated in AIMET version 1.33.0.")
        else:
            param_encodings = encodings.get('param_encodings', {})
            activation_encodings = encodings.get('activation_encodings', {})

        if not param_encodings and not activation_encodings:
            raise RuntimeError

        if strict is True:
            encoding_keys = param_encodings.keys() | activation_encodings.keys()
            model_keys = set(name.replace("._module_to_wrap", "") for name, _
                             in chain(self.model.named_modules(), utils.get_all_named_parameters(self.model)))
            keys_not_found = encoding_keys - model_keys
            if keys_not_found:
                keys_not_found = ', '.join(sorted(keys_not_found))
                msg = f"Encoding dictionary contains modules/parameters that doesn't exist in the model: {keys_not_found}"
                raise RuntimeError(msg)

        if param_encodings is not None:
            self._set_param_encodings(param_encodings,
                                      strict, partial, requires_grad, allow_overwrite)

        if activation_encodings is not None:
            self._set_activation_encodings(activation_encodings,
                                           strict, partial, requires_grad, allow_overwrite)

    @deprecated("Use QuantizationSimModel.load_encodings instead.")
    def load_and_freeze_encodings(self, encoding_path: str, ignore_when_quantizer_disabled: bool = False):
        """
        Functionality to set encodings (both activation and parameter) as per the given encodings JSON file and
        freeze them.
        .. note:
            The encodings JSON file should be the {prefix}_torch.encodings json exported during sim.export()

        :param encoding_path: JSON file path from where to load the encodings.
        :param ignore_when_quantizer_disabled: ignore raising RuntimeError while setting encodings,
            when quantizers are disabled.
        """
        self.load_encodings(encoding_path,
                            strict=not ignore_when_quantizer_disabled,
                            partial=True,
                            requires_grad=False,
                            allow_overwrite=False)

    def _set_param_encodings(self,
                             encoding_dict: Mapping,
                             strict: bool,
                             partial: bool,
                             requires_grad: Optional[bool],
                             allow_overwrite: bool):
        for name, quant_module in self.model.named_modules():
            if isinstance(quant_module, _QuantizedModuleProtocol):
                param_encoding = {
                    param_name: encoding_dict[f'{name}.{param_name}']
                    for param_name, _ in quant_module.param_quantizers.items()
                    if f'{name}.{param_name}' in encoding_dict
                }
                try:
                    quant_module.import_param_encodings(param_encoding,
                                                        strict,
                                                        partial,
                                                        requires_grad,
                                                        allow_overwrite)
                except RuntimeError as e:
                    raise RuntimeError(f"Encoding import failed for module: {name}.\n{str(e)}") from e


    def _set_activation_encodings(self,
                                  activation_encoding_dict: Mapping,
                                  strict: bool,
                                  partial: bool,
                                  requires_grad: Optional[bool],
                                  allow_overwrite: bool):
        for module_name, module in self.model.named_modules():
            if not isinstance(module, _QuantizedModuleProtocol):
                continue

            try:
                input_encoding = activation_encoding_dict[module_name]['input']
            except KeyError:
                input_encoding = {}

            try:
                module.import_input_encodings(input_encoding,
                                              strict,
                                              partial,
                                              requires_grad,
                                              allow_overwrite)
            except RuntimeError as e:
                raise RuntimeError(f"Encoding import failed for module: {module_name}.\n{str(e)}") from e

            try:
                output_encoding = activation_encoding_dict[module_name]['output']
            except KeyError:
                output_encoding = {}

            try:
                module.import_output_encodings(output_encoding,
                                               strict,
                                               partial,
                                               requires_grad,
                                               allow_overwrite)
            except RuntimeError as e:
                raise RuntimeError(f"Encoding import failed for module: {module_name}.\n{str(e)}") from e


    @deprecated("Use QuantizationSimModel.load_encodings instead.")
    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file.

        :param encoding_path: path from where to load parameter encodings file
        """
        with open(encoding_path, mode='r') as f:
            encodings = json.load(f)

        if 'activation_encodings' in encodings:
            del encodings['activation_encodings']

        self.load_encodings(encodings,
                            strict=True,
                            partial=True,
                            requires_grad=False,
                            allow_overwrite=False)

    def run_modules_for_traced_custom_marker(self, module_list: List[torch.nn.Module], dummy_input):
        """
        Given a list of modules to run and dummy input for the module, create a traced CustomMarker for each module
        and store it in the module_marker map. The same dummy input will be used for all modules.

        :param module_list: List of modules to create traced CustomMarkers for
        :param dummy_input: Dummy input for all modules
        """

        module_to_name_map = self._get_leaf_module_to_name_map()

        for module in module_list:
            # Only perform init and trace if the given module is a leaf module, and we have not recorded it before
            if module in module_to_name_map and module_to_name_map[module] not in self._module_marker_map:
                name = module_to_name_map[module]
                module = module.get_original_module() if isinstance(module, _QuantizedModuleProtocol) else module
                with utils.in_eval_mode(module), torch.no_grad():
                    marker_layer = torch.jit.trace(CustomMarker(module, name, True), dummy_input)
                    self._module_marker_map[name] = marker_layer




@deprecated("Use pickle.dump instead")
def save_checkpoint(quant_sim_model: _QuantizationSimModelInterface, file_path: str):
    """
    This API provides a way for the user to save a checkpoint of the quantized model which can
    be loaded at a later point to continue fine-tuning e.g.
    See also load_checkpoint()

    :param quant_sim_model: QuantizationSimModel to save checkpoint for
    :param file_path: Path to the file where you want to save the checkpoint
    :return: None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(quant_sim_model, file)


@deprecated("Use pickle.load instead")
def load_checkpoint(file_path: str) -> _QuantizationSimModelInterface:
    """
    Load the quantized model

    :param file_path: Path to the file where you want to save the checkpoint
    :return: A new instance of the QuantizationSimModel created after loading the checkpoint
    """
    with open(file_path, 'rb') as file:
        sim = pickle.load(file)
        return sim


@deprecated("check_accumulator_overflow API will be removed in the future releases.")
def check_accumulator_overflow(model: torch.nn.Module, quant_bw: int, accum_bw: int):
    """
    Checks for any potential for accumulator overflow across all the layers of the given model
    :param model: Model
    :param quant_bw: Bitwidth the layers are quantized at
    :param accum_bw: Bitwidth of the accumulator
    :return: Name of the layer with the most accumulator range used and range used
    """

    most_accum_range_used = 0
    most_accum_range_used_layer = None

    for layer_name, layer in model.named_modules():

        if isinstance(layer, torch.nn.Conv2d):
            was_accum_range_exceeded, accum_range_used = get_conv_accum_bounds(layer.weight.detach().numpy(),
                                                                               quant_bw, accum_bw)
            if accum_range_used > most_accum_range_used:
                most_accum_range_used = accum_range_used
                most_accum_range_used_layer = layer_name

            if was_accum_range_exceeded:
                logger.info('Possible accumulator overflow for layer: %s', layer_name)

    if most_accum_range_used < 1:
        logger.info('No overflow detected. Layer %s had the most accumulator range used: %f%%',
                    most_accum_range_used_layer, most_accum_range_used * 100)
    else:
        logger.info('Overflow detected. Layer %s had the most accumulator range used: %f%%',
                    most_accum_range_used_layer, most_accum_range_used * 100)

    return most_accum_range_used_layer, most_accum_range_used
