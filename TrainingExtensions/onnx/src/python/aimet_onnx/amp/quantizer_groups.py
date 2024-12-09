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
from typing import Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, field

from aimet_common.connected_graph.operation import Op
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops

from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE

from aimet_common.connected_graph.connectedgraph import get_ordered_ops
from aimet_common.amp.quantizer_groups import QuantizerGroupBase, get_supported_candidates_for_quantizers, \
    compute_baseline_candidate_options, find_valid_ops
from aimet_common.utils import AimetLogger

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import QcQuantizeOp

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


@dataclass(frozen=True)
class QuantizerGroup(QuantizerGroupBase):
    """
    Group of modules and quantizers
    """
    parameter_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    activation_quantizers: Tuple[str, ...] = field(default_factory=tuple)

    def get_candidate(self, name_to_quantizer_dict: Dict) -> CANDIDATE_WITH_DTYPE:
        """
        Gets Activation & parameter bitwidth

        :param name_to_quantizer_dict: Gets module from module name
        :return: Tuple of Activation, parameter bitwidth and data type
        """
        activation_bw, parameter_bw = None, None
        activation_dtype, param_dtype = None, None

        for quantizer in self.get_activation_quantizers(name_to_quantizer_dict):
            activation_bw = quantizer.bitwidth
            activation_dtype = quantizer.data_type
            break

        for quantizer in self.get_param_quantizers(name_to_quantizer_dict):
            if quantizer.enabled:
                parameter_bw = quantizer.bitwidth
                param_dtype = quantizer.data_type
                break

        return (activation_bw, activation_dtype), (parameter_bw, param_dtype)

    def set_quantizers_to_candidate(self,
                                    name_to_quantizer_dict: Dict,
                                    candidate: CANDIDATE_WITH_DTYPE):
        """
        Sets a quantizer group to a given candidate bitwidth

        :param name_to_quantizer_dict: Gets module from module name
        :param candidate: candidate with act and param bw and data types
        """
        (activation_bw, activation_dtype), (param_bw, param_dtype) = candidate

        for quantizer in self.get_activation_quantizers(name_to_quantizer_dict):
            quantizer.bitwidth = activation_bw
            quantizer.data_type = activation_dtype

        for quantizer in self.get_param_quantizers(name_to_quantizer_dict):
            quantizer.bitwidth = param_bw
            quantizer.data_type = param_dtype

    def to_list(self) -> List[Tuple[str, str]]:
        """
        Converts quantizer group to a list

        :return: List containing input/output quantizers & weight quantizers
        """
        return list(itertools.chain(
            (("activation", module_name) for module_name in self.activation_quantizers),
            (("weight", module_name) for module_name in self.parameter_quantizers),
        ))

    def get_active_quantizers(self, name_to_quantizer_dict) -> List[QcQuantizeOp]:
        """
        Find all active tensor quantizers associated with this quantizer group

        :param name_to_quantizer_dict: Gets module from module name
        :return: List of active quantizers
        """
        quantizers = self.get_activation_quantizers(name_to_quantizer_dict) + \
                     self.get_param_quantizers(name_to_quantizer_dict)
        return [quantizer for quantizer in quantizers if quantizer.enabled]

    def get_activation_quantizers(self, name_to_quantizer_dict):
        """
        Gets activation quantizers

        :param name_to_quantizer_dict: Gets module from module name

        :return List of activation quantizers
        """
        result = []
        for module_name in self.activation_quantizers:
            quantizer = name_to_quantizer_dict[module_name]
            result.append(quantizer)
        return result

    def get_param_quantizers(self, name_to_quantizer_dict):
        """
        Gets parameter quantizers

        :param name_to_quantizer_dict: Gets module from module name

        :return List of parameter quantizers
        """
        result = []
        for module_name in self.parameter_quantizers:
            quantizer = name_to_quantizer_dict[module_name]
            result.append(quantizer)
        return result


op_types_to_ignore = ['Reshape', 'branch', 'Gather', 'Unsqueeze', 'Pad']
ops_not_to_traverse = ['Shape']


def find_op_groups(connected_graph: ConnectedGraph) -> Dict:
    """
    Finds parent child groups based on following rules.
    1) If there is a direct connection between two ops, op1 and op2, then op1 is parent of op2 and they form a group
    2) If the input to an op (op1) is shared with another op (op2), the op producing the input (op0) is the parent,
    and op1 and op2 are the children

    :param connected_graph: Connected graph
    :return: Dict of parent (key) and children (value) groups
    """
    # Get ordered ops in Connected graph
    ordered_ops = get_ordered_ops(connected_graph.starting_ops)
    valid_ops = find_valid_ops(connected_graph, ops_not_to_traverse)

    parent_child_op_groups = defaultdict(list)
    map_for_skipped_ops = {}

    for op in ordered_ops:
        if op.dotted_name not in valid_ops or op.type in op_types_to_ignore:
            continue
        _find_parent_child_op_groups(op, parent_child_op_groups, map_for_skipped_ops)

    return parent_child_op_groups


def _find_parent_child_op_groups(op: Op, parent_child_op_groups: Dict, map_for_skipped_ops: Dict):
    """
    Finds op groups along the parent to child flow
    :param op: Op
    :param parent_child_op_groups: parent child op groups dict
    :param map_for_skipped_ops: map to find first skipped parents of skipped ops
    """
    output = op.output

    if output:
        consumers = output.consumers
        for consumer in consumers:
            dotted_name = op.dotted_name
            if consumer.type in ops_not_to_traverse:
                continue
            if op.dotted_name in map_for_skipped_ops:
                dotted_name = map_for_skipped_ops[op.dotted_name]

            if consumer.type in op_types_to_ignore:
                map_for_skipped_ops[consumer.dotted_name] = dotted_name
                _find_parent_child_op_groups(consumer, parent_child_op_groups, map_for_skipped_ops)
            else:
                if consumer.dotted_name not in parent_child_op_groups[dotted_name]:
                    parent_child_op_groups[dotted_name].append(consumer.dotted_name)
        if not consumers and op.dotted_name in map_for_skipped_ops:
            parent_child_op_groups[map_for_skipped_ops[op.dotted_name]] = []
    else:
        dotted_name = op.dotted_name
        parent_child_op_groups[dotted_name].append(None)


def find_quantizer_group(sim: QuantizationSimModel) -> Tuple[Dict, List[QuantizerGroup]]:
    """
    Finds quantizer groups in a quantization sim
    :param sim: Quantization sim
    :return: Dictionary of quantized op name to sim.quantizer_config object, List of quantizer groups
    """
    # Get connected graph from quantsim
    connected_graph = sim.connected_graph

    if connected_graph is None:
        raise AssertionError('Aborting Auto Mixed Precision, connected graph needs to exist for Auto Mixed precision')

    # Find parent to children mapping for connected graph ops
    parent_child_op_groups = find_op_groups(connected_graph)

    # Find mapping of quantized op name to quantizer info
    op_name_to_quantizer_dict = _get_op_name_to_act_quantizer_name_dicts(sim)
    op_to_param_dict = _get_op_to_param_name_dict(sim)

    quantizer_groups = []
    _add_input_quantizer_group(op_to_param_dict, sim, quantizer_groups)
    for parent, children in parent_child_op_groups.items():
        activation_quantizers = []
        if parent in op_name_to_quantizer_dict:
            activation_quantizers.append(op_name_to_quantizer_dict[parent])
        parameter_quantizers = []
        for child in children:
            if child and child in op_to_param_dict:
                parameter_quantizers.append(op_to_param_dict[child])

        if activation_quantizers or parameter_quantizers:
            _add_quantizer_group(quantizer_groups, tuple(activation_quantizers), tuple(parameter_quantizers))

    _add_output_quantizer_group(op_name_to_quantizer_dict, sim, quantizer_groups)

    return sim.qc_quantize_op_dict, quantizer_groups


def _add_quantizer_group(quantizer_groups: List[QuantizerGroup], activation_quantizers: Tuple,
                         parameter_quantizers: Tuple):
    """
    Adds quantizer group to the quantizer groups list
    :param quantizer_groups: List of Quantizer groups
    :param activation_quantizers: Tuple of activation quantizers
    :param parameter_quantizers: Tuple of parameter quantizers
    """
    quantizer_group = QuantizerGroup(parameter_quantizers=parameter_quantizers,
                                     activation_quantizers=activation_quantizers)
    quantizer_groups.append(quantizer_group)
    logger.info('Quantizer Group added: %s', quantizer_group)


def _add_input_quantizer_group(op_to_param_dict: Dict, sim: QuantizationSimModel, quantizer_groups: List):
    """
    Adds input's (of the model) quantizer group
    :param op_to_param_dict: Key: op_name Value: Weight name associated
    :param sim: Quantization Sim
    :param quantizer_groups: Quantizer Groups List
    """
    conn_graph_ops = get_all_input_ops(sim.connected_graph)
    for input_op in conn_graph_ops:
        parameter_quantizers = []
        activation_quantizers = []
        if input_op.dotted_name in op_to_param_dict:
            parameter_quantizers.append(op_to_param_dict[input_op.dotted_name])
        for input_product in input_op.inputs:
            activation_quantizer = input_product.tensor_dict[input_op]
            if isinstance(activation_quantizer, str) and \
                    activation_quantizer in sim.activation_names and \
                    sim.qc_quantize_op_dict[activation_quantizer].enabled:
                activation_quantizers.append(input_product.tensor_dict[input_op])
        if activation_quantizers or parameter_quantizers:
            _add_quantizer_group(quantizer_groups, tuple(activation_quantizers), tuple(parameter_quantizers))


def _add_output_quantizer_group(op_name_to_quantizer_dict: Dict, sim: QuantizationSimModel, quantizer_groups: List):
    """
    Adds output's (of the model) quantizer group
    :param op_name_to_quantizer_dict: Key: op_name Value: quantizer associated with op name
    :param sim: Quantization Sim
    :param quantizer_groups: Quantizer Groups List
    """
    conn_graph_ops = get_all_output_ops(sim.connected_graph)
    for output_op in conn_graph_ops:
        activation_quantizers = []
        if output_op.dotted_name in op_name_to_quantizer_dict:
            activation_quantizers.append(op_name_to_quantizer_dict[output_op.dotted_name])
        if activation_quantizers:
            _add_quantizer_group(quantizer_groups, tuple(activation_quantizers), ())


def _get_op_to_param_name_dict(sim: QuantizationSimModel) -> Dict:
    """
    Creates the dict where param name (weight) is mapped to op's name
    :param sim: Quantization Sim
    """
    op_to_param_dict = {}
    conn_graph_ops = sim.connected_graph.get_all_ops()
    for op in conn_graph_ops.values():
        for param_name in op.parameters:
            _, param_type = op.parameters[param_name]
            if param_type == 'weight' and sim.qc_quantize_op_dict[param_name].enabled:
                op_to_param_dict[op.dotted_name] = param_name

    return op_to_param_dict


def _get_op_name_to_act_quantizer_name_dicts(sim: QuantizationSimModel) -> Dict:
    """
    Creates the dict where param quantizers if enabled are mapped to their param_names and activation
    quantizer if enabled is mapped to it's inputs name
    :param sim: Quantization Sim
    :return op_name_to_activation_quantizer_name_dict
    """
    op_name_to_activation_quantizer_name_dict = {}
    for node in sim.model.model.graph.node:
        if 'QcQuantizeOp' in node.name:
            continue
        for output_product in node.output:
            if output_product in sim.activation_names:
                activation_quantizer_op = sim.qc_quantize_op_dict[output_product]
                if activation_quantizer_op.enabled:
                    op_name_to_activation_quantizer_name_dict[node.name] = output_product
    return op_name_to_activation_quantizer_name_dict


def find_supported_candidates(quantizer_groups: List[QuantizerGroup],
                              amp_candidates: List[CANDIDATE_WITH_DTYPE],
                              supported_kernels: Dict,
                              quantizer_to_op_type: Dict,
                              use_all_amp_candidates: bool) -> Tuple[Dict, List]:
    """
    Computes 1. a list of supported candidates per Quantizer and 2. List of candidate options for max_candidate
    :param quantizer_groups: List of quantizer groups computed for the given model
    :param amp_candidates: List of candidates specified by the user to be used for the AMP algorithm
    :param supported_kernels: Dict of supported kernels for a given op/defaults specified in the config file
    :param quantizer_to_op_type: Dict of quantizers to onnx op type
    :param use_all_amp_candidates: Boolean value representing whether the unsupported candidates in the
    "candidates" list need to be considered for creating the output lists. If set to True, all the AMP candidates are
    directly used for all the Quantizers, else the candidates per Quantizers are computed.
    """

    quantizers_with_supported_candidates = defaultdict(list)

    for quantizer_group in quantizer_groups:
        quantizers = sorted(set(itertools.chain(quantizer_group.activation_quantizers,
                                                quantizer_group.parameter_quantizers)))

        supported_kernels_for_quantizers = get_supported_candidates_for_quantizers(quantizers,
                                                                                   quantizer_to_op_type,
                                                                                   supported_kernels,
                                                                                   amp_candidates,
                                                                                   use_all_amp_candidates)

        quantizers_with_supported_candidates[quantizer_group] = supported_kernels_for_quantizers.copy()

    max_candidate_options = compute_baseline_candidate_options(quantizers_with_supported_candidates, amp_candidates,
                                                               use_all_amp_candidates)

    return quantizers_with_supported_candidates, max_candidate_options
