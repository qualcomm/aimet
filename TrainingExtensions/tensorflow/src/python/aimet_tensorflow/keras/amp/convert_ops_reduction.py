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

""" This module implements the convert op overhead reduction stage of AMP """
import enum
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantizationDataType
from aimet_common.amp.quantizer_groups import QuantizerGroupBase
from aimet_common.amp.convert_ops_reduction import ReduceConvertOps as BaseReduceConvertOps

from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.amp.quantizer_groups import find_op_groups, find_wrapper_module


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


class ReduceConvertOps(BaseReduceConvertOps):
    """ This class implements Convert Ops reduction algo"""

    class TensorAxis(enum.Enum):
        """ Enum to find the axis index for tensor dimensions"""
        N = 0
        H = 1
        W = 2
        C = 3

    def __init__(self, aimet_sim: QuantizationSimModel, quantizer_groups: List[QuantizerGroupBase],
                 candidates: List, mac_dict: Dict):
        """
        :param aimet_sim: Sim object with loaded quantizer settings
        :param quantizer_groups: List of quantizer groups of aimet_sim
        :param candidates: List of tuples for all possible bitwidth values for activations and parameters
            Note: Currently only supports (8, 8) & (16, 8)
        :param mac_dict: mac dictionary of the ops
        """
        super().__init__(aimet_sim, quantizer_groups, candidates, mac_dict)

    def get_qg_idx(self) -> Dict:
        """
        The function returns quantizer-group name to its list index dictionary
        :return: mapping of quantizer-group name and its position in the quantizer-group list
        """
        quantizer_group2node_index = {}
        counter = 0
        for qgroup in self._quantizer_groups:
            if len(qgroup.input_quantizers) == 1 and not qgroup.output_quantizers:
                # TODO: (temporary fix for now) conv1_Conv_input_quantizer_idx_0_input --> conv1_Conv_input
                # quantizer_group2node_index[qgroup.input_quantizers[0]+"_input"] = counter
                quantizer_group2node_index[qgroup.input_quantizers[0].split('_input_quantizer_')[0] + "_input"] = counter
                counter += 1
            elif len(qgroup.output_quantizers) == 1:
                quantizer_group2node_index[qgroup.output_quantizers[0].split('_output_quantizer_')[0] + "_output"] = counter
                counter += 1
            elif not qgroup.input_quantizers and not qgroup.output_quantizers and len(qgroup.parameter_quantizers) >= 1:
                quantizer_group2node_index[qgroup.parameter_quantizers[0].split('/')[0] + "_weights_only"] = counter
                counter += 1
            else:
                raise RuntimeError("Issue with this quantizer group:{}".format(qgroup))
        return quantizer_group2node_index

    def generate_graphs(self) -> Tuple:
        """
        This function generates the quantizer group graph (nodes are contracted) and the op graph (nodes are not
        contracted). The implementation leverages networkx's contracted_nodes function.
        The nodes in the quantizer group graph are quantizer groups which are derived by contracting the non-quantizer
        group nodes of the graph. The contracted nodes includes the data movement ops (such as reshape, permute) and are
        present in the contracted_nodes field of quantizer group node.
        The op graph is a version of quantizer group graph before contraction of any node.

        :return: Tuple containing quantizer-group graph & op graph
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals

        quantizer_group2node_index = self.get_qg_idx()
        parameter_quantizers = self.get_all_param_quantizers()

        G = self.get_networkx_graph()

        # pylint: disable=protected-access
        ops_dict = self._sim.connected_graph._ops
        dotted_name2output_shape, dotted_name2op_name = self.fetch_op_info(ops_dict)

        # Add tensor dims and size to the nodes of the graph
        self.add_node_info(G, dotted_name2output_shape)
        _logger.info("Networkx Graph Created")

        # Rename nodes from op dotted name to quantizer-group name
        module_name_to_module_dict = self._sim._layer_name_to_quant_wrapper
        G = self.rename_nodes(G, module_name_to_module_dict, dotted_name2op_name, ops_dict, find_wrapper_module)
        _logger.info("Networkx Graph nodes renamed to QuantizerGroup names")

        # Tensor dimension correction (needed because of the transpose ops, which have either "nchw" or "nhwc" in their names)
        for node in G.nodes:
            if "tensor_dims" in G.nodes[node]:
                if "nchw" in node:
                    G.nodes[node]["tensor_dims"] = self.permute_tensor_dims(G.nodes[node]["tensor_dims"], "nchw")

        # Identify the nodes in the graph that represents a quantizer-group
        self.mark_qg_nodes(G, quantizer_group2node_index)
        _logger.info("Identified actual QuantizerGroup nodes")

        # Identify the nodes in the graph who has parameters
        nx.set_node_attributes(G, False, "has_weight")
        for i_para in parameter_quantizers:
            op_name = i_para[:i_para.rfind('/')] + '_output'
            if op_name in G:
                G.nodes[op_name]['has_weight'] = True
            else:
                raise RuntimeError(f"check the weight quantizer {i_para}")

        op_graph = G.copy()
        _logger.info("Generated Op Graph")

        # Contract the non-quantizer-group nodes
        G = self.contract_non_qg_nodes(G)
        quantizer_group_graph = G
        _logger.info("Generated QuantizerGroup Graph by contracting Op Graph nodes")

        return quantizer_group_graph, op_graph

    @staticmethod
    def find_op_groups_all(graph: ConnectedGraph) -> Dict:
        """
        This method returns the grouping of parent and child ops in op-dotted-name form using CG
        """
        op_name_to_dotted_name = {}
        # pylint: disable=protected-access
        for op_obj in graph._ops.values():
            op_name_to_dotted_name[op_obj.name] = op_obj.dotted_name
        parent_child_op_groups = find_op_groups(graph)
        parent_child_op_groups_dotted = defaultdict(list)
        for parent, children in parent_child_op_groups.items():
            dotted_parent = parent
            if parent not in ['input_ops', 'output_ops']:
                dotted_parent = op_name_to_dotted_name[parent]
            dotted_children = [op_name_to_dotted_name[child] for child in children]
            parent_child_op_groups_dotted[dotted_parent] = dotted_children
        return parent_child_op_groups_dotted

    def get_phase_two_solution(self, qg_graph) -> Dict:
        """
        This function returns the phase 2 solution i.e. the act bw of each quantizer-group
        :param qg_graph: quantizer-group graph
        """
        QUANTIZER_TYPE_INPUT = 'input'
        QUANTIZER_TYPE_OUTPUT = 'output'

        solution_dict = {}
        # pylint: disable=protected-access
        for layer_name, layer in self._sim._layer_name_to_quant_wrapper.items():
            for quantizer in layer.input_quantizers.layers:
                name = layer_name + "_" + QUANTIZER_TYPE_INPUT
                if name in qg_graph.nodes:
                    solution_dict[name] = int(quantizer.bitwidth)
            for quantizer in layer.output_quantizers.layers:
                name = layer_name + "_" + QUANTIZER_TYPE_OUTPUT
                if name in qg_graph.nodes:
                    solution_dict[name] = int(quantizer.bitwidth)
        return solution_dict

    def generate_qg_solution(self, phase_three_sol) -> List[Tuple]:
        """
        Given the phase-3 solution i.e. the act bw of each quantizer-group, this function returns the quantizer-group
        object and its appropriate amp-candidate
        """
        qg_candidates = []
        for quantizer_group in self._quantizer_groups:
            if len(quantizer_group.input_quantizers) == 1 and not quantizer_group.output_quantizers:
                # TODO: (temporary fix for now) conv1_Conv_input_quantizer_idx_0_input --> conv1_Conv_input
                # bitwidth = phase_three_sol[quantizer_group.input_quantizers[0] + "_input"]
                bitwidth = phase_three_sol[quantizer_group.input_quantizers[0].split('_input_quantizer_')[0] + "_input"]
            elif len(quantizer_group.output_quantizers) == 1:
                bitwidth = phase_three_sol[quantizer_group.output_quantizers[0].split('_output_quantizer_')[0] + "_output"]  ## Example: conv1_output = 8 bits (followed by conv2)
            elif not quantizer_group.input_quantizers and not quantizer_group.output_quantizers and len(
                    quantizer_group.parameter_quantizers) == 1:
                param_bitwidth_candidates = [candidate[1][0] for candidate in self._candidates]
                all_eight_bits = [i == 8 for i in param_bitwidth_candidates]
                if all(all_eight_bits):
                    bitwidth = 8
                else:
                    raise RuntimeError("Candidate parameter bandwidths are not all 8 bits -- unsupported.")
            else:
                raise RuntimeError("Unexpected quantizer group type.")

            if bitwidth == 8:
                candidate = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
            elif bitwidth == 16:
                candidate = ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
            else:
                raise NotImplementedError

            qg_candidates.append((quantizer_group, candidate))

        return qg_candidates
