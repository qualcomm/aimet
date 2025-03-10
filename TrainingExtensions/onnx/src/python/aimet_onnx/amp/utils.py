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
""" Utilities for mixed precision feature """

from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Import AIMET specific modules
from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE, get_effective_bitwidth
from aimet_common.cost_calculator import CostCalculator
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.amp.quantizer_groups import QuantizerGroup
from aimet_onnx import utils
from aimet_onnx.quantsim import QuantizationSimModel


def get_activation_shapes(sim: QuantizationSimModel) -> Dict[str, Any]:
    """
    Runs data through the model to get activation shapes

    :param sim: QuantizationSim model
    :return Dict of activation shapes where key is activation name and value is shape
    """
    activations = utils.get_graph_intermediate_activations(sim.model.graph())
    hooks = []
    for name in activations:
        hooks.append(utils.add_hook_to_get_activation(sim.model.model, name))
    dummy_input = utils.make_dummy_input(sim.model.model)
    # pylint: disable=protected-access
    sess = QuantizationSimModel.build_session(sim.model.model, ["CPUExecutionProvider"], sim._user_onnx_libs)
    outputs = sess.run(None, dummy_input)
    activation_shapes = {}
    for idx, node in enumerate(sim.model.graph().output):
        act_name = node.name
        activation_shapes[act_name] = outputs[idx].shape
    utils.remove_activation_hooks(sim.model.model, hooks)
    return activation_shapes


allowed_data_types = ['Conv', 'Gemm']


class Layer:
    """ Data structure to hold a layer's output shape and weight shape """
    def __init__(self):
        self.weight_shape = []
        self.output_shape = []


def find_layer_database_for_mac_calculation(sim: QuantizationSimModel) -> Dict[str, Layer]:
    """
    Finds layer database for allowed ops of type Conv & Linear

    :param sim: QuantizationSim model
    :return Dict of op database where key is name of the layer and value is Layer object
    """
    def _get_weight_shape(op):
        for param, param_type in op.parameters.values():
            if param_type == 'weight':
                return param.shape
        return None

    activation_shapes = get_activation_shapes(sim)
    ops = sim.connected_graph.get_all_ops()
    # Either conv or Linear ops are allowed
    allowed_ops = {op.dotted_name for op in ops.values() if op.type in allowed_data_types}

    op_database = {}
    for node in sim.model.model.graph.node:
        if node.name in allowed_ops and node.output[0] in activation_shapes:
            layer = Layer()
            layer.output_shape = activation_shapes[node.output[0]]
            if len(layer.output_shape) == 2:
                # Append 1, 1 to Linear layer's shape
                layer.output_shape = list(layer.output_shape) + [1, 1]
            layer.weight_shape = _get_weight_shape(ops[node.name])
            op_database[node.name] = layer

    return op_database


def create_mac_dict(sim: QuantizationSimModel) -> Dict[str, int]:
    """
    Creates a mac dict where key is layer name and value is mac count

    :param sim: QuantizationSim model
    :return mac dictionary
    """
    layer_db = find_layer_database_for_mac_calculation(sim)
    mac_dict = {}

    for layer_name, layer in layer_db.items():
        mac_dict[layer_name] = CostCalculator.compute_layer_cost(layer).mac

    return mac_dict


def find_param_name_to_parent_name_dict(connected_graph: ConnectedGraph) -> Dict[str, str]:
    """
    Find mapping of op (only Conv and Linear ops) names to their corresponding read variable op names

    :param connected_graph: Connected graph
    :return: Dictionary mapping param name -> parent op name
    """
    param_to_op_name = {}

    all_ops = connected_graph.get_all_ops()
    for op in all_ops.values():
        if op.parameters:
            for param, param_type in op.parameters.values():
                if param_type == 'weight':
                    param_to_op_name[param.name] = op.dotted_name

    return param_to_op_name


def _create_quantizer_op_dict(quantizer_group: QuantizerGroup) -> Dict[str, List[str]]:
    """
    Creates quantizer op dictionary from a quantizer group

    :param quantizer_group: Quantizer Group
    :return: Dictionary of quantizers ('activation' -> activation quantizers, 'weight' -> weight quantizers)
    """
    quantizer_op_dict = defaultdict(list)
    for op_name in quantizer_group.activation_quantizers:
        quantizer_op_dict['activation'].append(op_name)
    for op_name in quantizer_group.parameter_quantizers:
        quantizer_op_dict['weight'].append(op_name)

    return quantizer_op_dict


def calculate_running_bit_ops(mac_dict: Dict[str, int], quantizer_group: QuantizerGroup,
                              param_name_to_parent_name_dict: Dict[str, str], op_bw_dict: Dict[str, List[int]],
                              max_candidate: CANDIDATE_WITH_DTYPE, new_candidate: CANDIDATE_WITH_DTYPE,
                              running_bit_ops: int) -> int:
    """
    Returns new running bit ops given previous running bit ops value and the current quantizer to change bitwidth of

    :param mac_dict: Dictionary mapping op names to mac count of an op (only Conv and Linear ops are present in
    the dictionary)
    :param quantizer_group: Quantizer group
    :param param_name_to_parent_name_dict: Dictionary mapping param to their corresponding parent ops
    :param op_bw_dict: Dictionary mapping op names to most recently used bitwidths for each quantizer type
    :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
    :param new_candidate: New bitwidth (activation, weight)
    :param running_bit_ops: previous running bit ops count
    :return: Current running bit ops count
    """
    # pylint: disable=too-many-locals
    def _calculate_bit_ops(quantizer_type: str, running_bit_ops: int, op_name: str) -> int:
        """
        Helper function to compute bit ops for weight or activation feeding into an op
        """
        if op_name in mac_dict:
            if op_name in op_bw_dict.keys():
                # If an activation or weight quantizer of op was quantized previously, there will be an entry in
                # the dictionary. Then, we simply need to update the corresponding index (activation or weight) with
                # the new bitwidth.

                # Subtract the previous bitops count for this op (will add on the new bit ops count for this
                # op later, taking new bitwidth value into account)
                running_bit_ops -= (op_bw_dict[op_name][act_index] *
                                    op_bw_dict[op_name][weight_index] *
                                    mac_dict[op_name])
                if quantizer_type == 'activation':
                    op_bw_dict[op_name][act_index] = act_bw_new
                else:
                    op_bw_dict[op_name][weight_index] = param_bw_new
            else:
                # New entry in op_bitwidth_dict needs to be made. One index will be default_bitwidth, and the
                # other index will be the new bitwidth we are currently using.

                # Subtract the previous bitops count for this op (will add on the new bitops count for this
                # op later, taking new bitwidth value into account)
                running_bit_ops -= (act_bw_max * param_bw_max) * mac_dict[op_name]
                if quantizer_type == 'activation':
                    op_bw_dict[op_name] = [act_bw_new, param_bw_max]
                else:
                    op_bw_dict[op_name] = [act_bw_max, param_bw_new]
            # Add new bitops count to the running bit ops value, taking into account the updated bitwidths for
            # activation and weight quantizers.
            running_bit_ops += (op_bw_dict[op_name][act_index] *
                                op_bw_dict[op_name][weight_index] *
                                mac_dict[op_name])
        return running_bit_ops

    act_index = 0
    weight_index = 1

    quantizer_op_dict = _create_quantizer_op_dict(quantizer_group)

    (act_bw_max, act_dtype_max), (param_bw_max, param_dtype_max) = max_candidate
    (act_bw_new, act_dtype), (param_bw_new, param_dtype) = new_candidate

    act_bw_max = get_effective_bitwidth(act_dtype_max, act_bw_max)
    param_bw_max = get_effective_bitwidth(param_dtype_max, param_bw_max)
    act_bw_new = get_effective_bitwidth(act_dtype, act_bw_new)
    param_bw_new = get_effective_bitwidth(param_dtype, param_bw_new)

    if 'activation' in quantizer_op_dict and 'weight' in quantizer_op_dict:
        for quant_op_name in quantizer_op_dict['weight']:
            # Get parent op name from its read variable op name
            parent_op_name = param_name_to_parent_name_dict[quant_op_name]
            running_bit_ops = _calculate_bit_ops('activation', running_bit_ops, parent_op_name)

    if 'weight' in quantizer_op_dict:
        for quant_op_name in quantizer_op_dict['weight']:
            # Get parent op name from its read variable op name
            parent_op_name = param_name_to_parent_name_dict[quant_op_name]
            running_bit_ops = _calculate_bit_ops('weight', running_bit_ops, parent_op_name)

    return running_bit_ops


def find_bit_ops_reduction(quantizer_group: QuantizerGroup, mac_dict: Dict[str, int],
                           param_name_to_parent_dict: Dict[str, str],
                           max_candidate: Tuple, candidate: CANDIDATE_WITH_DTYPE) -> int:
    """
    Find bit ops reduction when Bitwidth changes from max_candidate to candidate

    :param quantizer_group: Quantizer group for which we want to find bit ops reduction
    :param mac_dict: Dictionary mapping modules to mac counts
    :param param_name_to_parent_dict: Dictionary mapping op to their corresponding read variable ops
    :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
    :param candidate: Current BW candidate
    :return: Bit ops reduction
    """
    # pylint: disable=too-many-locals
    quantizer_group_dict = _create_quantizer_op_dict(quantizer_group)

    bit_ops_reduction = 0

    (act_bw_max, act_dtype_max), (param_bw_max, param_dtype_max) = max_candidate
    (act_bw, act_dtype), (param_bw, param_dtype) = candidate

    act_bw_max = get_effective_bitwidth(act_dtype_max, act_bw_max)
    param_bw_max = get_effective_bitwidth(param_dtype_max, param_bw_max)
    act_bw = get_effective_bitwidth(act_dtype, act_bw)
    param_bw = get_effective_bitwidth(param_dtype, param_bw)


    if 'activation' in quantizer_group_dict and 'weight' in quantizer_group_dict:
        for quant_name in quantizer_group_dict['weight']:
            parent_op_name = param_name_to_parent_dict[quant_name]
            if parent_op_name in mac_dict:
                bit_ops_reduction = bit_ops_reduction - mac_dict[parent_op_name] * act_bw * param_bw + \
                                    mac_dict[parent_op_name] * act_bw_max * param_bw_max

    elif 'weight' in quantizer_group_dict:
        for quant_name in quantizer_group_dict['weight']:
            parent_op_name = param_name_to_parent_dict[quant_name]
            if parent_op_name in mac_dict:
                bit_ops_reduction = bit_ops_reduction - mac_dict[parent_op_name] * act_bw_max * param_bw + \
                                    mac_dict[parent_op_name] * act_bw_max * param_bw_max
    return bit_ops_reduction


def get_quantizer_to_op_type_dict(sim: QuantizationSimModel) -> Dict:
    """
    Get quantizer to op type dict

    :return: Dict with quantizer as key and op_type as value
    """
    quantizer_to_op_type = {}
    for input_tensor in sim.model.model.graph.input:
        for node in sim.model.model.graph.node:
            quantized_name = input_tensor.name + '_updated'
            if quantized_name in node.input:
                quantizer_to_op_type[input_tensor.name] = [node.op_type]

    for node in sim.model.model.graph.node:
        for input_tensor_name in node.input:
            if 'qdq' in input_tensor_name:
                product_name = utils.get_product_name_from_quantized_name(input_tensor_name)
                if product_name in sim.qc_quantize_op_dict:
                    quantizer_to_op_type[product_name] = [node.op_type]
        for output_tensor in node.output:
            if output_tensor in sim.qc_quantize_op_dict:
                quantizer_to_op_type[output_tensor] = [node.op_type]
    return quantizer_to_op_type
