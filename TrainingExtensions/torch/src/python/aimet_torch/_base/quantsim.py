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
""" QuantizationSimModel interface """
from abc import ABC, abstractmethod
import copy
from typing import (
    List,
    Union,
    Dict,
    Optional,
    runtime_checkable,
    Protocol,
    Mapping,
    TYPE_CHECKING,
    Tuple,
    Iterable,
)
import torch

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType, SupportedKernelsAction, QuantDtypeBwInfo
from aimet_common.quantsim import validate_quantsim_inputs, extract_global_quantizer_args

from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph, Op
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper
from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_torch._base.nn.modules.custom import MatMul

if TYPE_CHECKING:
    from aimet_torch.v2.quantization.base.encoding import EncodingBase


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SUPPORTED_KERNELS_ACTION = SupportedKernelsAction.warn_on_error


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

    def export_input_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """

    def export_output_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """

    def export_param_encodings(self) -> Dict[str, List[Dict]]:
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
        # pylint: disable=import-outside-toplevel
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
