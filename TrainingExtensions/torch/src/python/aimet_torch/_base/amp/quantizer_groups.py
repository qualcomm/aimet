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

""" Find quantizer groups in a model """
import itertools
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import torch

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.connected_graph.operation import Op
from aimet_common.utils import AimetLogger
from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE
from aimet_common.amp.quantizer_groups import QuantizerGroupBase, get_supported_candidates_for_quantizers, \
    compute_baseline_candidate_options

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch._base.quantsim import _QuantizationSimModelInterface, _QuantizedModuleProtocol
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

        for quantizer in self._get_input_quantizers(name_to_quantizer_dict) +\
                         self._get_output_quantizers(name_to_quantizer_dict):
            activation_bw = quantizer.bitwidth
            activation_data_type = quantizer.data_type
            break

        for quantizer in self._get_param_quantizers(name_to_quantizer_dict):
            if quantizer.enabled:
                parameter_bw = quantizer.bitwidth
                parameter_data_type = quantizer.data_type
                break

        return (activation_bw, activation_data_type), (parameter_bw, parameter_data_type)

    def set_quantizers_to_candidate(self,
                                    name_to_quantizer_dict: Dict,
                                    candidate: CANDIDATE_WITH_DTYPE) -> None:
        """
        Sets a quantizer group to a given candidate bitwidth
        :param name_to_quantizer_dict: Gets module from module name
        :param candidate: candidate with act and param bw and data types
        """
        if len(candidate) == 1:
            (activation_bw, activation_data_type), = candidate
            param_bw, param_data_type = None, None
        else:
            (activation_bw, activation_data_type), (param_bw, param_data_type) = candidate

        for quantizer in self._get_input_quantizers(name_to_quantizer_dict) +\
                         self._get_output_quantizers(name_to_quantizer_dict):
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
        return list(itertools.chain(
            (("input", module_name) for module_name in self.input_quantizers),
            (("output", module_name) for module_name in self.output_quantizers),
            (("weight", module_name) for module_name in self.parameter_quantizers),
        ))

    def get_active_quantizers(self, name_to_quantizer_dict):
        """ Find all active tensor quantizers associated with this quantizer group """
        quantizers = self._get_input_quantizers(name_to_quantizer_dict) +\
                     self._get_output_quantizers(name_to_quantizer_dict) +\
                     self._get_param_quantizers(name_to_quantizer_dict)
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


def find_wrapper_module(op_name: str, module_name_to_quantizer_dict: Dict) -> \
        Tuple[Optional[str], Optional[torch.nn.Module]]:
    """
    Finds quantization (wrapping) module corresponding to the wrapper module's dotted name
    :param op_name: Dotted name of op as represented in connected graph
    :param module_name_to_quantizer_dict: Dict key: name of wrapped module value: quantization wrapper
    :return: Module name and the corresponding torch module in the sim
    """
    # pylint:disable = protected-access
    module_name = op_name[op_name.find('.') + 1:]
    if module_name in module_name_to_quantizer_dict:
        return module_name, module_name_to_quantizer_dict[module_name]
    # Else it is a functional op
    return None, None


def get_module_name_to_module_dict(sim: _QuantizationSimModelInterface) -> Dict:
    """
    Creates a dictionary of wrapped module's name to quantizer module
    :param sim: quantization sim
    :return: Dict key: name of wrapped module value: quantization wrapper
    """
    module_name_to_quantizer_dict = {}
    for name, ref_module in sim.model.named_modules():
        if isinstance(ref_module, _QuantizedModuleProtocol):
            module_name_to_quantizer_dict[name] = ref_module
    return module_name_to_quantizer_dict


ops_to_skip = ['view',
               'NumToTensor',
               'Split',
               CG_SPLIT,
               'PythonOp',
               'Tile',
               'transpose',
               'reshape',
               'flatten',
               'permute',
               'Permute', # tensor.transpose() results in Permute. Name obtained after mpp
               'Reshape', # Name obtained after MPP
               'ChannelShuffle', # Obtained without going thru mpp. torch.nn.ChannelShuffle fails mpp
               'TopK', # Name obtained after MPP
               'PixelShuffle', # Name obtained after MPP
               'Expand', # Name obtained after MPP. Reproduce using tensor.expand
               'Pad', # Name obtained after MPP
               'Slice', # Name obtained after MPP
               'Gather', # Name obtained after MPP
               'ScatterElements', # Name obtained after MPP
               'ReduceMin',  # Name obtained after MPP
               'ReduceMax', # Name obtained after MPP
               'Upsample', # Name obtained after MPP
               'RoIPool', # Name obtained after MPP
               'MaxPool', # Name obtained after MPP
               'Transpose' # Name obtained after MPP
               ]
ops_not_to_traverse = ['size']

def find_output_quantizer_groups(op: Op, parent_child_op_groups: Dict, map_for_skipped_ops: Dict):
    """
    Finds quantizer groups along the parent to child flow
    :param op: pytorch module
    :param parent_child_op_groups: parent child relationships in graph
    :param map_for_skipped_ops: map to find first skipped parents of skipped ops
    """
    output = op.output
    if output:
        consumers = output.consumers
        for consumer in consumers:
            if consumer.type in ops_not_to_traverse:
                continue
            dotted_name = op.dotted_name
            if op.dotted_name in map_for_skipped_ops:
                dotted_name = map_for_skipped_ops[op.dotted_name]
            if consumer.type in ops_to_skip:
                map_for_skipped_ops[consumer.dotted_name] = dotted_name
                find_output_quantizer_groups(consumer, parent_child_op_groups, map_for_skipped_ops)
            # If there is a one to one connection between quantizers
            else:
                parent_child_op_groups[dotted_name].append(consumer.dotted_name)
    else:
        if op.dotted_name in map_for_skipped_ops:
            parent_child_op_groups[map_for_skipped_ops[op.dotted_name]] = []


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

    for op in graph.ordered_ops:
        # Add 1st op as child
        if not op.input_ops:
            parent_child_op_groups['input_ops'].append(op.dotted_name)
        # Add output op as child to put output of model as a quantizer group
        if op.output is None:
            parent_child_op_groups['output_ops'].append(op.dotted_name)
    for op in graph.get_all_ops().values():
        if op.type in ops_to_skip or op.type in ops_not_to_traverse:
            continue
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
                input_quantizers.append(module_name + '_input_quantizer_idx_' + str(idx))
        for _, param_quantizer in module.param_quantizers.items():
            if param_quantizer.enabled:
                parameter_quantizers.append(module_name)
    return tuple(input_quantizers), tuple(parameter_quantizers)


# pylint: disable=too-many-branches, too-many-locals
def find_quantizer_group(sim: _QuantizationSimModelInterface) -> Tuple[Dict, List[QuantizerGroup]]:
    """
    Finds quantizer groups in a quantization sim model
    :param sim: Quantization sim
    :return: List of Quantizer groups
    """
    # Get connected graph from quantsim for model without wrappers
    connected_graph = sim.connected_graph

    if connected_graph is None:
        raise AssertionError('Aborting Auto Mixed Precision, connected graph needs to exist for Auto Mixed precision')

    quantizer_groups = []

    parent_child_op_groups = find_op_groups(connected_graph)

    module_name_to_module_dict = get_module_name_to_module_dict(sim)

    if 'input_ops' in parent_child_op_groups:
        for child in parent_child_op_groups['input_ops']:
            # Add one quantizer group for each input and it's weight param
            input_quantizers, parameter_quantizers = get_input_and_param_quantizers(child, module_name_to_module_dict)
            if input_quantizers or parameter_quantizers:
                child_module_name, _ = find_wrapper_module(child, module_name_to_module_dict)
                supported_kernel_ops = []
                if child_module_name is not None:
                    supported_kernel_ops.append(child_module_name)
                quantizer_group = QuantizerGroup(
                    input_quantizers=input_quantizers,
                    parameter_quantizers=parameter_quantizers,
                    supported_kernel_ops=tuple(supported_kernel_ops)
                )
                quantizer_groups.append(quantizer_group)
                logger.debug('\n Quantizer Group Added: %s', quantizer_group)

    # Based on which quantizers are enabled, create a list of quantizer_groups
    for parents, children in parent_child_op_groups.items():
        input_quantizers = ()
        output_quantizers = ()
        parameter_quantizers = ()
        if parents in ['input_ops', 'output_ops']:
            continue
        if not isinstance(parents, tuple):
            parents = [parents]
        for parent in parents:
            module_name, module = find_wrapper_module(parent, module_name_to_module_dict)
            if module is not None:
                for output_quantizer in module.output_quantizers:
                    if output_quantizer.enabled:
                        output_quantizers += (module_name,)

        supported_kernel_ops = []
        for child in children:
            input_q, param_q = get_input_and_param_quantizers(child, module_name_to_module_dict)
            input_quantizers += input_q
            parameter_quantizers += param_q
            child_module_name, _ = find_wrapper_module(child, module_name_to_module_dict)
            if child_module_name is not None:
                supported_kernel_ops.append(child_module_name)

        # Don't add quantizer group if it is empty
        if input_quantizers or output_quantizers or parameter_quantizers:
            quantizer_group = QuantizerGroup(
                input_quantizers=input_quantizers,
                output_quantizers=output_quantizers,
                parameter_quantizers=parameter_quantizers,
                supported_kernel_ops=tuple(supported_kernel_ops)
            )
            quantizer_groups.append(quantizer_group)
            logger.debug('\n Quantizer Group added: %s', quantizer_group)

    if 'output_ops' in parent_child_op_groups:
        for parent in parent_child_op_groups['output_ops']:
            # Add one quantizer group for each input and it's weight param
            module_name, module = find_wrapper_module(parent, module_name_to_module_dict)
            if module is not None:
                for output_quantizer in module.output_quantizers:
                    if output_quantizer.enabled:
                        # Using empty supported kernel ops so that model output quantizers are able to consider all
                        # default candidates
                        quantizer_group = QuantizerGroup(
                            output_quantizers=(module_name,),
                            supported_kernel_ops=tuple()
                        )
                        quantizer_groups.append(quantizer_group)
                        logger.debug('\n Quantizer Group added: %s', quantizer_group)

    return module_name_to_module_dict, quantizer_groups


def find_supported_candidates(quantizer_groups: List[QuantizerGroup],
                              amp_candidates: List[CANDIDATE_WITH_DTYPE],
                              supported_kernels: Dict,
                              module_name_to_module_dict: Dict,
                              use_all_amp_candidates: bool) -> Tuple[Dict, List]:
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
        quantizers = sorted(set(itertools.chain(quantizer_group.get_input_quantizer_modules(),
                                                quantizer_group.output_quantizers,
                                                quantizer_group.parameter_quantizers)))

        # quantizers are now unique ops present in the given quantizer_group
        onnx_ops = defaultdict(list)
        supported_kernel_types = set()
        for supported_kernel_op in quantizer_group.supported_kernel_ops:
            try:
                module = module_name_to_module_dict[supported_kernel_op]._module_to_wrap
            except AttributeError:
                module = module_name_to_module_dict[supported_kernel_op].get_original_module()
            backend_type = aimet_op_to_backend_op_name_map.get(type(module))

            if backend_type in supported_kernels:
                supported_kernel_types.add(backend_type)
            else:
                onnx_types = onnx_utils.map_torch_types_to_onnx.get(type(module), [])

                if not onnx_types:
                    logger.warning("No mapping found for %s in the torch to onnx op type mapping dictionary.",
                                   type(module))

                supported_kernel_types.update(onnx_types)

                for onnx_type in onnx_types:
                    if onnx_type not in supported_kernels.keys():
                        if module in supported_kernels:
                            supported_kernels[onnx_type] = supported_kernels[module]

        for quantizer in quantizers:
            onnx_ops[quantizer] = list(supported_kernel_types)

        supported_kernels_for_quantizers = get_supported_candidates_for_quantizers(quantizers,
                                                                                   onnx_ops,
                                                                                   supported_kernels,
                                                                                   amp_candidates,
                                                                                   use_all_amp_candidates)

        quantizers_with_supported_candidates[quantizer_group] = supported_kernels_for_quantizers.copy()

    max_candidate_options = compute_baseline_candidate_options(quantizers_with_supported_candidates, amp_candidates,
                                                               use_all_amp_candidates)

    return quantizers_with_supported_candidates, max_candidate_options
