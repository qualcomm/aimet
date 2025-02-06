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
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import tensorflow as tf


from aimet_common.connected_graph.operation import Op
from aimet_common.utils import AimetLogger
from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE
from aimet_common.amp.quantizer_groups import QuantizerGroupBase

from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.quantsim import QuantizationSimModel, substitutable_modules
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


ops_to_skip = ['view', 'NumToTensor', 'Split', 'PythonOp']
ops_not_to_traverse = ['size']
INPUT_OPS_STR = 'input_ops'
OUTPUT_OPS_STR = 'output_ops'



@dataclass(frozen=True)
class QuantizerGroup(QuantizerGroupBase):
    """
    Group of modules and quantizers
    """
    input_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    output_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    parameter_quantizers: Tuple[str, ...] = field(default_factory=tuple)

    def get_candidate(self, name_to_quantizer_dict: Dict) -> CANDIDATE_WITH_DTYPE:
        """
        Gets Activation & parameter bitwidth
        :param name_to_quantizer_dict: Gets module from module name
        :return: Tuple of Activation, parameter bitwidth and data type
        """
        activation_bw, parameter_bw = None, None
        activation_data_type, parameter_data_type = None, None
        for module_name in self.input_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            for layer in module.input_quantizers.layers:
                activation_bw = layer.bitwidth
                activation_data_type = layer.data_type
                break
            break

        for module_name in self.output_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            for layer in module.output_quantizers.layers:
                activation_bw = layer.bitwidth
                activation_data_type = layer.data_type
                break
            break

        if self.parameter_quantizers:
            for module_name in self.parameter_quantizers:
                module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
                for layer in module.param_quantizers.layers:
                    parameter_bw = layer.bitwidth
                    parameter_data_type = layer.data_type
                    break
                break


        return (activation_bw, activation_data_type), (parameter_bw, parameter_data_type)

    @staticmethod
    def lookup_quantizer(quantizer_name: str, name_to_quantizer_dict: Dict) -> tf.keras.layers.Layer:
        """
        Returns the quantizer layer corresponding to the name
        :quantizer_name: Name of the quantizer
        :name_to_quantizer_dict: Dictionary of mappings from quantizer name to quantizer layer
        """
        if isinstance(quantizer_name, tuple):
            quantizer_name = quantizer_name[0]
        module = name_to_quantizer_dict[quantizer_name]
        return module

    def set_quantizers_to_candidate(self,
                                    name_to_quantizer_dict: Dict,
                                    candidate: CANDIDATE_WITH_DTYPE) -> None:
        """
        Sets a quantizer group to a given candidate bitwidth
        :param name_to_quantizer_dict: Gets module from module name
        :param candidate: candidate with act and param bw and data types
        """
        (activation_bw, activation_data_type), (param_bw, param_data_type) = candidate
        for module_name in self.input_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            for layer in module.input_quantizers.layers:
                layer.bitwidth = activation_bw
                layer.data_type = activation_data_type

        for module_name in self.output_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            for layer in module.output_quantizers.layers:
                layer.bitwidth = activation_bw
                layer.data_type = activation_data_type

        if not self.parameter_quantizers:
            return

        for module_name in self.parameter_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            for layer in module.param_quantizers:
                layer.bitwidth = param_bw
                layer.data_type = param_data_type

    def to_list(self) -> List[Tuple[str, str]]:
        """
        Converts quantizer group to a list
        :return: List containing input/output quantizers & weight quantizers
        """
        if self.parameter_quantizers:
            ret_list = list(itertools.chain(
                (("input", module_name) for module_name in self.input_quantizers),
                (("output", module_name) for module_name in self.output_quantizers),
                (("weight", module_name) for module_name in self.parameter_quantizers),
            ))
        else:
            ret_list = list(itertools.chain(
                (("input", module_name) for module_name in self.input_quantizers),
                (("output", module_name) for module_name in self.output_quantizers),
            ))
        return ret_list

    def get_active_quantizers(self, name_to_quantizer_dict: Dict) -> List[TensorQuantizer]:
        """ Find all active tensor quantizers associated with this quantizer group """
        quantizers = []
        for module_name in self.input_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            quantizers += list(module.input_quantizers.layers)

        for module_name in self.output_quantizers:
            module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
            quantizers += list(module.output_quantizers.layers)

        if self.parameter_quantizers:
            for module_name in self.parameter_quantizers:
                module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
                quantizers += list(module.param_quantizers.layers)

        return list(set(quantizer for quantizer in quantizers if quantizer.is_enabled()))

    def get_active_param_quantizers(self, name_to_quantizer_dict: Dict) -> List[TensorQuantizer]:
        """ Find all active param tensor quantizers associated with this quantizer group
        :param name_to_quantizer_dict: Contains mapping of module name to sim.quantizer_config object
        """
        quantizers = []
        if self.parameter_quantizers:
            for module_name in self.parameter_quantizers:
                module = self.lookup_quantizer(module_name, name_to_quantizer_dict)
                quantizers += list(module.param_quantizers.layers)
        return list(set(quantizer for quantizer in quantizers if quantizer.is_enabled()))

def find_output_quantizer_groups(op: Op, parent_child_op_groups: Dict, map_for_skipped_ops: Dict):
    """
    Finds quantizer groups along the parent to child flow
    :param op: pytorch module
    :param parent_child_op_groups: parent child relationships in graph
    :param map_for_skipped_ops: map to find first skipped parents of skipped ops
    """
    if op.outputs:
        for consumer in op.output_ops:
            name = op.name
            if consumer.type in ops_not_to_traverse:
                continue
            if op.dotted_name in map_for_skipped_ops:
                name = map_for_skipped_ops[op.name]

            if consumer.type in ops_to_skip:
                map_for_skipped_ops[consumer.name] = name
                find_output_quantizer_groups(consumer, parent_child_op_groups, map_for_skipped_ops)
            # If there is a one to one connection between quantizers
            else:
                if name in map_for_skipped_ops:
                    name = map_for_skipped_ops[name]
                parent_child_op_groups[name].append(consumer.name)
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
            parent_child_op_groups[INPUT_OPS_STR].append(op.name)
        # Add output op as child to put output of model as a quantizer group
        if not op.outputs:
            parent_child_op_groups[OUTPUT_OPS_STR].append(op.name)

    for op in graph.get_all_ops().values():
        if op.type in ops_to_skip or op.type in ops_not_to_traverse:
            continue
        find_output_quantizer_groups(op, parent_child_op_groups, map_for_skipped_ops)

    return parent_child_op_groups

def get_module_name_to_module_dict(sim: QuantizationSimModel) -> Dict:
    """
    Creates a dictionary of wrapped module's name to quantizer module
    :param sim: quantization sim
    :return: Dict key: name of wrapped module value: quantization wrapper
    """
    module_name_to_quantizer_dict = {}
    for layer in sim.quant_wrappers():
        for quantizer in layer.input_quantizers:
            module_name_to_quantizer_dict[quantizer.name] = layer
        for quantizer in layer.output_quantizers:
            module_name_to_quantizer_dict[quantizer.name] = layer
        for quantizer in layer.param_quantizers:
            module_name_to_quantizer_dict[quantizer.name] = layer

    return module_name_to_quantizer_dict

# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-branches
def find_quantizer_group(sim: QuantizationSimModel) -> Tuple[Dict, List[QuantizerGroup]]:
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

    quantized_op_name_to_quantizer_dict = get_module_name_to_module_dict(sim)

    if INPUT_OPS_STR in parent_child_op_groups:
        for child in parent_child_op_groups[INPUT_OPS_STR]:
            # Add one quantizer group for each input and it's weight param
            layer = connected_graph.get_layer_from_op_name(child)
            if isinstance(layer, tuple(substitutable_modules.keys())):
                sub_quantizer_groups = get_quantizers_groups_substituted_layer(sim, layer)
                quantizer_groups.extend(sub_quantizer_groups)
                continue

            input_quantizer_names, output_quantizer_names, param_quantizer_names = sim.get_quantizer_name_by_layer(layer)

            if input_quantizer_names or param_quantizer_names:
                quantizer_group = QuantizerGroup(
                    input_quantizers=input_quantizer_names,
                    parameter_quantizers=param_quantizer_names
                )
                quantizer_groups.append(quantizer_group)
                logger.debug('\n Quantizer Group Added: %s', quantizer_group)

    # Based on which quantizers are enabled, create a list of quantizer_groups
    for parents in parent_child_op_groups:
        if parents in [INPUT_OPS_STR, OUTPUT_OPS_STR]:
            continue

        if not isinstance(parents, tuple):
            parents = [parents]

        for parent in parents:
            layer = connected_graph.get_layer_from_op_name(parent)
            if isinstance(layer, tuple(substitutable_modules.keys())):
                sub_quantizer_groups = get_quantizers_groups_substituted_layer(sim, layer)
                quantizer_groups.extend(sub_quantizer_groups)
                continue

            input_quantizer_names, output_quantizer_names, param_quantizer_names = sim.get_quantizer_name_by_layer(layer)

        # Don't add quantizer group if it is empty
        if input_quantizer_names or output_quantizer_names or param_quantizer_names:
            quantizer_group = QuantizerGroup(
                input_quantizers=input_quantizer_names,
                output_quantizers=output_quantizer_names,
                parameter_quantizers=param_quantizer_names
            )
            quantizer_groups.append(quantizer_group)
            logger.debug('\n Quantizer Group added: %s', quantizer_group)

    if OUTPUT_OPS_STR in parent_child_op_groups:
        for parent in parent_child_op_groups[OUTPUT_OPS_STR]:
            # Add one quantizer group for each input and it's weight param

            layer = connected_graph.get_layer_from_op_name(parent)
            if isinstance(layer, tuple(substitutable_modules.keys())):
                sub_quantizer_groups = get_quantizers_groups_substituted_layer(sim, layer)
                quantizer_groups.extend(sub_quantizer_groups)
                continue

            input_quantizer_names, output_quantizer_names, param_quantizer_names = sim.get_quantizer_name_by_layer(layer)

            if output_quantizer_names:
                quantizer_group = QuantizerGroup(
                    input_quantizers=input_quantizer_names,
                    output_quantizers=output_quantizer_names,
                    parameter_quantizers=param_quantizer_names
                )
                quantizer_groups.append(quantizer_group)
                logger.debug('\n Quantizer Group added: %s', quantizer_group)

    return quantized_op_name_to_quantizer_dict, quantizer_groups

# pylint: disable=protected-access
def get_quantizers_groups_substituted_layer(sim: QuantizationSimModel, layer) -> List[QuantizerGroup]:
    """ Helper function to return the quantizer groups for the substituted layers """
    layer = sim._substituted_layer[layer]
    quantizer_groups = []
    for quant_wrapper in layer.quant_wrappers():
        input_quantizers = quant_wrapper.input_quantizers
        output_quantizers = quant_wrapper.output_quantizers
        param_quantizers = quant_wrapper.param_quantizers


        input_quantizer_names = QuantizationSimModel._quantizer_to_name_tuple(input_quantizers)
        output_quantizer_names = QuantizationSimModel._quantizer_to_name_tuple(output_quantizers)
        param_quantizer_names = QuantizationSimModel._quantizer_to_name_tuple(param_quantizers)

        if input_quantizer_names or output_quantizer_names or param_quantizer_names:
            quantizer_group = QuantizerGroup(
                input_quantizers=input_quantizer_names,
                output_quantizers=output_quantizer_names,
                parameter_quantizers=param_quantizer_names
            )
            quantizer_groups.append(quantizer_group)
            logger.debug('\n Quantizer Group added: %s', quantizer_group)

    return quantizer_groups


def find_wrapper_module(op_name: str, module_name_to_quantizer_dict: Dict) -> Tuple[str, tf.keras.layers.Layer]:
    """
    Finds quantization (wrapping) module corresponding to the wrapper module's dotted name
    :param op_name: Dotted name of op as represented in connected graph
    :param module_name_to_quantizer_dict: Dict key: name of wrapped module value: quantization wrapper
    :return: Module name and the corresponding quant-wrapper module in the sim
    """
    # pylint:disable = protected-access
    module_name = op_name[op_name.find('.') + 1:]
    if module_name in module_name_to_quantizer_dict:
        return module_name, module_name_to_quantizer_dict[module_name]
    # Else it is a functional op
    raise KeyError
