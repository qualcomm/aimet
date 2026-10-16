# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Find quantizer groups in a model"""

import itertools
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import torch

from aimet_torch.common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_torch.common.connected_graph.operation import Op
from aimet_torch.common.utils import AimetLogger
from aimet_torch.common.amp.utils import CANDIDATE_WITH_DTYPE
from aimet_torch.common.amp.quantizer_groups import (
    QuantizerGroupBase,
    get_supported_candidates_for_quantizers,
    compute_baseline_candidate_options,
)

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.nn import BaseQuantizationMixin
from aimet_torch import onnx_utils
from aimet_torch.translation_mapping import aimet_op_to_backend_op_name_map

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


@dataclass(frozen=True)
class QuantizerGroup(QuantizerGroupBase):
    """
    Group of modules and quantizers
    """

    input_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    output_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    parameter_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    supported_kernel_ops: Tuple[str, ...] = field(default_factory=tuple)

    def get_candidate(self, name_to_quantizer_dict: Dict) -> CANDIDATE_WITH_DTYPE:
        """
        Gets Activation & parameter bitwidth
        :param name_to_quantizer_dict: Gets module from module name
        :return: Tuple of Activation, parameter bitwidth and data type
        """
        activation_bw, parameter_bw = None, None
        activation_data_type, parameter_data_type = None, None

        for quantizer in self._get_input_quantizers(
            name_to_quantizer_dict
        ) + self._get_output_quantizers(name_to_quantizer_dict):
            activation_bw = quantizer.bitwidth
            activation_data_type = quantizer.data_type
            break

        for quantizer in self._get_param_quantizers(name_to_quantizer_dict):
            if quantizer.enabled:
                parameter_bw = quantizer.bitwidth
                parameter_data_type = quantizer.data_type
                break

        return (activation_bw, activation_data_type), (
            parameter_bw,
            parameter_data_type,
        )

    def set_quantizers_to_candidate(
        self, name_to_quantizer_dict: Dict, candidate: CANDIDATE_WITH_DTYPE
    ) -> None:
        """
        Sets a quantizer group to a given candidate bitwidth
        :param name_to_quantizer_dict: Gets module from module name
        :param candidate: candidate with act and param bw and data types
        """
        if len(candidate) == 1:
            ((activation_bw, activation_data_type),) = candidate
            param_bw, param_data_type = None, None
        else:
            (activation_bw, activation_data_type), (param_bw, param_data_type) = (
                candidate
            )

        for quantizer in self._get_input_quantizers(
            name_to_quantizer_dict
        ) + self._get_output_quantizers(name_to_quantizer_dict):
            quantizer.bitwidth = activation_bw
            quantizer.data_type = activation_data_type

        if param_bw is not None:
            for quantizer in self._get_param_quantizers(name_to_quantizer_dict):
                quantizer.bitwidth = param_bw
                quantizer.data_type = param_data_type

    def to_list(self) -> List[Tuple[str, str]]:
        """
        Converts quantizer group to a list
        :return: List containing input/output quantizers & weight quantizers
        """
        return list(
            itertools.chain(
                (("input", module_name) for module_name in self.input_quantizers),
                (("output", module_name) for module_name in self.output_quantizers),
                (("weight", module_name) for module_name in self.parameter_quantizers),
            )
        )

    def get_active_quantizers(self, name_to_quantizer_dict):
        """Find all active tensor quantizers associated with this quantizer group"""
        quantizers = (
            self._get_input_quantizers(name_to_quantizer_dict)
            + self._get_output_quantizers(name_to_quantizer_dict)
            + self._get_param_quantizers(name_to_quantizer_dict)
        )
        return [quantizer for quantizer in quantizers if quantizer.enabled]

    def _get_input_quantizers(self, name_to_quantizer_dict):
        result = []
        for quantizer_name in self.input_quantizers:
            out = quantizer_name.split("_input_quantizer_idx_")
            assert len(out) == 2
            module_name, quantizer_idx = out[0], int(out[1])
            module = name_to_quantizer_dict[module_name]
            result.append(module.input_quantizers[quantizer_idx])
        return result

    def _get_output_quantizers(self, name_to_quantizer_dict):
        result = []
        for module_name in self.output_quantizers:
            module = name_to_quantizer_dict[module_name]
            result += module.output_quantizers
        return result

    def _get_param_quantizers(self, name_to_quantizer_dict):
        result = []
        for module_name in self.parameter_quantizers:
            module = name_to_quantizer_dict[module_name]
            for _, param_quantizer in module.param_quantizers.items():
                result.append(param_quantizer)
        return result

    def get_input_quantizer_modules(self):
        """helper method to get the module names corresponding to input_quantizers"""
        result = set()
        for quantizer_name in self.input_quantizers:
            out = quantizer_name.split("_input_quantizer_idx_")
            assert len(out) == 2
            result.add(out[0])
        return tuple(sorted(result))


def find_wrapper_module(
    op_name: str, module_name_to_quantizer_dict: Dict
) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    """
    Finds quantization (wrapping) module corresponding to the wrapper module's dotted name
    :param op_name: Dotted name of op as represented in connected graph
    :param module_name_to_quantizer_dict: Dict key: name of wrapped module value: quantization wrapper
    :return: Module name and the corresponding torch module in the sim
    """
    # pylint:disable = protected-access
    module_name = op_name[op_name.find(".") + 1 :]
    if module_name in module_name_to_quantizer_dict:
        return module_name, module_name_to_quantizer_dict[module_name]
    # Else it is a functional op
    return None, None


def get_module_name_to_module_dict(sim: QuantizationSimModel) -> Dict:
    """
    Creates a dictionary of wrapped module's name to quantizer module
    :param sim: quantization sim
    :return: Dict key: name of wrapped module value: quantization wrapper
    """
    module_name_to_quantizer_dict = {}
    for name, ref_module in sim.model.named_modules():
        if isinstance(ref_module, BaseQuantizationMixin):
            module_name_to_quantizer_dict[name] = ref_module
    return module_name_to_quantizer_dict


ops_to_skip = [
    "view",
    "NumToTensor",
    "Split",
    CG_SPLIT,
    "PythonOp",
    "Tile",
    "transpose",
    "reshape",
    "flatten",
    "permute",
    "Permute",  # tensor.transpose() results in Permute. Name obtained after mpp
    "Reshape",  # Name obtained after MPP
    "ChannelShuffle",  # Obtained without going thru mpp. torch.nn.ChannelShuffle fails mpp
    "TopK",  # Name obtained after MPP
    "PixelShuffle",  # Name obtained after MPP
    "Expand",  # Name obtained after MPP. Reproduce using tensor.expand
    "Pad",  # Name obtained after MPP
    "Slice",  # Name obtained after MPP
    "Gather",  # Name obtained after MPP
    "ScatterElements",  # Name obtained after MPP
    "ReduceMin",  # Name obtained after MPP
    "ReduceMax",  # Name obtained after MPP
    "Upsample",  # Name obtained after MPP
    "RoIPool",  # Name obtained after MPP
    "MaxPool",  # Name obtained after MPP
    "Transpose",  # Name obtained after MPP
]
ops_not_to_traverse = ["size"]


def find_output_quantizer_groups(
    op: Op, parent_child_op_groups: Dict, map_for_skipped_ops: Dict
):
    """
    Finds quantizer groups along the parent to child flow
    :param op: pytorch module
    :param parent_child_op_groups: parent child relationships in graph
    :param map_for_skipped_ops: map to find first skipped parents of skipped ops
    """
    if op.outputs:
        consumers = op.output_ops
        for consumer in consumers:
            if consumer.type in ops_not_to_traverse:
                continue
            # Only propagate through math invariant ops if the first input is from the current op
            if consumer.type in ops_to_skip and consumer.inputs[0] not in op.outputs:
                continue
            dotted_name = op.dotted_name
            if op.dotted_name in map_for_skipped_ops:
                dotted_name = map_for_skipped_ops[op.dotted_name]
            if consumer.type in ops_to_skip:
                map_for_skipped_ops[consumer.dotted_name] = dotted_name
                find_output_quantizer_groups(
                    consumer, parent_child_op_groups, map_for_skipped_ops
                )
            # If there is a one to one connection between quantizers
            else:
                parent_child_op_groups[dotted_name].append(consumer.dotted_name)


def find_op_groups(graph: ConnectedGraph) -> Dict:
    """
    Finds parent children relationship based on three rules
    1) If there is a direct connection between two ops, op1 and op2, then op1 is parent of op2 and they form a group
    2) If the input to an op (op1) is shared with another op (op2), the op producing the input (op0) is the parent, and op1 and op2 are the children
    :param graph: connected graph
    :return: Dict of parent (key) and children (value) relationship
    """
    parent_child_op_groups = defaultdict(list)
    map_for_skipped_ops = {}

    for op in graph.get_all_ops().values():
        if op.type in ops_not_to_traverse:
            continue
        # Use empty tuple() for input op parent
        if not op.input_ops and op.type not in ops_to_skip:
            parent_child_op_groups[tuple()].append(op.dotted_name)
        if op.type in ops_to_skip and op.input_ops:
            continue
        parent_child_op_groups[op.dotted_name] = []
        find_output_quantizer_groups(op, parent_child_op_groups, map_for_skipped_ops)

    return parent_child_op_groups


# This code is not currently called anywhere but it can be used to combine two ops who feed into an elementwise op
def find_input_quantizer_groups(graph, map_for_skipped_ops, parent_child_op_groups):
    """
    Combines two groups which share the same output
    :param graph: connected graph
    :param map_for_skipped_ops: map to find first skipped parents of skipped ops
    :param parent_child_op_groups: parent child relationships in graph
    """

    for op in graph.get_all_ops().values():
        inputs = op.input_ops
        if len(inputs) > 1:
            new_parents = set()
            new_children = set()
            for input_op in inputs:
                dotted_name = input_op.dotted_name

                if input_op.type in ops_to_skip:
                    dotted_name = map_for_skipped_ops[dotted_name]

                new_parents.add(dotted_name)

                if dotted_name in parent_child_op_groups:
                    for name in parent_child_op_groups[dotted_name]:
                        new_children.add(name)
                    del parent_child_op_groups[dotted_name]
            if len(new_parents) == 1:
                parent_child_op_groups[tuple(new_parents)[0]] = new_children
            else:
                parent_child_op_groups[tuple(new_parents)] = new_children


def get_input_and_param_quantizers(
    child: str, module_name_to_module_dict: Dict
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Adds child's input quantizer and param quantizer to quantizer group
    :param child: name of child
    :param module_name_to_module_dict: name to module ref dict
    :param quantizer_group: quantizer group
    """
    input_quantizers = []
    parameter_quantizers = []
    module_name, module = find_wrapper_module(child, module_name_to_module_dict)
    if module_name is not None:
        for idx, input_quantizer in enumerate(module.input_quantizers):
            if input_quantizer.enabled:
                input_quantizers.append(
                    module_name + "_input_quantizer_idx_" + str(idx)
                )
        for _, param_quantizer in module.param_quantizers.items():
            if param_quantizer.enabled:
                parameter_quantizers.append(module_name)
    return tuple(input_quantizers), tuple(parameter_quantizers)


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def find_quantizer_group(
    sim: QuantizationSimModel,
) -> Tuple[Dict, List[QuantizerGroup]]:
    """
    Finds quantizer groups in a quantization sim model
    :param sim: Quantization sim
    :return: List of Quantizer groups
    """
    # Get connected graph from quantsim for model without wrappers
    connected_graph = sim.connected_graph
    connected_graph.assert_safe()

    quantizer_groups = []

    def _add_quantizer_group(
        activation_quantizers=tuple(),
        parameter_quantizers=tuple(),
        supported_kernel_ops=tuple(),
    ):
        """
        Helper function to add a QuantizerGroup

        :param activation_quantizers: Tuple of activation quantizer names
        :param parameter_quantizers: Tuple of parameter quantizer names
        :param supported_kernel_ops: Tuple of supported kernel op names
        """
        if not activation_quantizers and not parameter_quantizers:
            return
        input_quantizers = tuple(
            q for q in activation_quantizers if "input_quantizer_idx_" in q
        )
        output_quantizers = tuple(
            q for q in activation_quantizers if q not in input_quantizers
        )
        supported_kernel_ops = tuple(
            name for name in supported_kernel_ops if name is not None
        )
        quantizer_group = QuantizerGroup(
            input_quantizers=input_quantizers,
            output_quantizers=output_quantizers,
            parameter_quantizers=parameter_quantizers,
            supported_kernel_ops=supported_kernel_ops,
        )
        if quantizer_group not in quantizer_groups:
            quantizer_groups.append(quantizer_group)
            logger.debug("\n Quantizer Group added: %s", quantizer_group)

    parent_child_op_groups = find_op_groups(connected_graph)

    module_name_to_module_dict = get_module_name_to_module_dict(sim)

    # Based on which quantizers are enabled, create a list of quantizer_groups
    for parents, children in parent_child_op_groups.items():
        if not isinstance(parents, tuple):
            parents = [parents]

        shared_quantizers = ()
        for parent in parents:
            module_name, module = find_wrapper_module(
                parent, module_name_to_module_dict
            )
            if module is not None:
                parent_quantizers = [
                    module_name for q in module.output_quantizers if q.enabled
                ]
                # If parent op is math invariant, can use its input quantizer as a shared quantizer
                if (
                    not parent_quantizers
                    and connected_graph.get_op_from_module_name(parent).type
                    in ops_to_skip
                ):
                    parent_quantizers, _ = get_input_and_param_quantizers(
                        parent, module_name_to_module_dict
                    )
                shared_quantizers += tuple(parent_quantizers)

        param_quantizer_dict = {}
        input_quantizer_dict = {}
        for child in children:
            child_module_name, _ = find_wrapper_module(
                child, module_name_to_module_dict
            )
            input_q, param_q = get_input_and_param_quantizers(
                child, module_name_to_module_dict
            )
            if param_q:
                param_quantizer_dict[child_module_name] = param_q
            input_quantizer_dict[child_module_name] = input_q

        if shared_quantizers:
            # All consumers ops must be considered regardless of whether they have param quantizers
            supported_kernel_ops = tuple(input_quantizer_dict.keys())
            # Add input quantizers from child layers (usually will be none)
            for name in param_quantizer_dict.keys():
                shared_quantizers += input_quantizer_dict.pop(name)
            # Param quantizers include all child parameter quantizers
            parameter_quantizers = tuple(
                itertools.chain.from_iterable(param_quantizer_dict.values())
            )
            # If there are param quantizers, group them with all input/output quantizers
            if parameter_quantizers:
                _add_quantizer_group(
                    activation_quantizers=shared_quantizers,
                    parameter_quantizers=parameter_quantizers,
                    supported_kernel_ops=supported_kernel_ops,
                )
            # Otherwise, group all input and output quantizers separately
            else:
                # Create a quantizer group for each input and output quantizer
                for quantizer in shared_quantizers:
                    _add_quantizer_group(
                        activation_quantizers=(quantizer,),
                        supported_kernel_ops=supported_kernel_ops,
                    )
        else:
            # If no shared quantizers, group param quantizers only with their own layer input quantizers
            for child_name, param_q in param_quantizer_dict.items():
                input_q = input_quantizer_dict.pop(child_name)
                _add_quantizer_group(
                    activation_quantizers=input_q,
                    parameter_quantizers=param_q,
                    supported_kernel_ops=(child_name,),
                )

        # Add remaining input quantizers to their own groups
        for child_name, input_q in input_quantizer_dict.items():
            for quantizer in input_q:
                _add_quantizer_group(
                    activation_quantizers=(quantizer,),
                    supported_kernel_ops=(child_name,),
                )

    return module_name_to_module_dict, quantizer_groups


def find_supported_candidates(
    quantizer_groups: List[QuantizerGroup],
    amp_candidates: List[CANDIDATE_WITH_DTYPE],
    supported_kernels: Dict,
    module_name_to_module_dict: Dict,
    use_all_amp_candidates: bool,
) -> Tuple[Dict, List]:
    """
    Computes 1. a list of supported candidates per Quantizer and 2. List of candidate options for max_candidate
    :param quantizer_groups: List of quantizer groups computed for the given model
    :param amp_candidates: List of candidates specified by the user to be used for the AMP algorithm
    :param supported_kernels: Dict of supported kernels for a given op/defaults specified in the config file
    :param module_name_to_module_dict: Dict mapping module name to module/quantizer
    :param use_all_amp_candidates: Boolean value representing whether the unsupported candidates in the
    "candidates" list need to be considered for creating the output lists. If set to True, all the AMP candidates are
    directly used for all the Quantizers, else the candidates per Quantizers are computed.
    """

    quantizers_with_supported_candidates = defaultdict(list)

    # pylint: disable=too-many-nested-blocks
    # pylint: disable=protected-access
    for quantizer_group in quantizer_groups:
        quantizers = sorted(
            set(
                itertools.chain(
                    quantizer_group.get_input_quantizer_modules(),
                    quantizer_group.output_quantizers,
                    quantizer_group.parameter_quantizers,
                )
            )
        )

        # quantizers are now unique ops present in the given quantizer_group
        onnx_ops = defaultdict(list)
        supported_kernel_types = set()
        for supported_kernel_op in quantizer_group.supported_kernel_ops:
            try:
                module = module_name_to_module_dict[supported_kernel_op]._module_to_wrap
            except AttributeError:
                module = module_name_to_module_dict[
                    supported_kernel_op
                ].get_original_module()
            backend_type = aimet_op_to_backend_op_name_map.get(type(module))

            if backend_type in supported_kernels:
                supported_kernel_types.add(backend_type)
            else:
                onnx_types = onnx_utils.map_torch_types_to_onnx.get(type(module), [])

                if not onnx_types:
                    logger.warning(
                        "No mapping found for %s in the torch to onnx op type mapping dictionary.",
                        type(module),
                    )

                supported_kernel_types.update(onnx_types)

                for onnx_type in onnx_types:
                    if onnx_type not in supported_kernels.keys():
                        if module in supported_kernels:
                            supported_kernels[onnx_type] = supported_kernels[module]

        for quantizer in quantizers:
            onnx_ops[quantizer] = list(supported_kernel_types)

        supported_kernels_for_quantizers = get_supported_candidates_for_quantizers(
            quantizers,
            onnx_ops,
            supported_kernels,
            amp_candidates,
            use_all_amp_candidates,
        )

        quantizers_with_supported_candidates[quantizer_group] = list(
            supported_kernels_for_quantizers
        )

    max_candidate_options = compute_baseline_candidate_options(
        quantizers_with_supported_candidates, amp_candidates, use_all_amp_candidates
    )

    return quantizers_with_supported_candidates, max_candidate_options
