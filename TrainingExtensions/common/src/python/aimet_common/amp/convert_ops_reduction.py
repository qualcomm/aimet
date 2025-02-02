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

""" This module contains common utilities for convert op reduction stage of AMP """
import os
import re
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Callable, Tuple
import math
import abc
import json
try:
    import cvxpy as cp
except ImportError:
    print("Unable to import cvxpy")
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

import numpy as np

from aimet_common.utils import AimetLogger
from aimet_common.amp.quantizer_groups import QuantizerGroupBase


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


class SamplingStrategy(Enum):
    """ Enum to represent the sampling strategy used"""
    unweighted_cut_size = 1
    weighted_with_tensor_size = 2
    weighted_with_predicted_convert_cost = 3


# pylint: disable=too-many-public-methods
class ReduceConvertOps(abc.ABC):
    """ Base class for convert op reduction alogrithm """

    DEFAULT_ALPHA_OPTIONS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    def __init__(self, aimet_sim, quantizer_groups: List[QuantizerGroupBase], candidates: List, mac_dict: Dict):
        """
        :param aimet_sim: Sim object with loaded quantizer settings
        :param quantizer_groups: List of quantizer groups of aimet_sim
        :param candidates: List of tuples for all possible bitwidth values for activations and parameters
            Note: Currently only supports (8, 8) & (16, 8)
        :param mac_dict: mac dictionary of the ops
        """
        self._sim = aimet_sim
        self._quantizer_groups = quantizer_groups
        self._candidates = candidates
        self._mac_dict = mac_dict

        self.quantizer_group_graph, self.op_graph = self.generate_graphs()

        # Add convert-op cost to the edges of quantizer_group graph
        self.add_convert_cost_predictions_to_graph()
        # Add bit-operation cost to the nodes of quantizer_group graph
        for i_candidate in self._candidates:
            ReduceConvertOps.add_bit_op_cost_to_graph(self.quantizer_group_graph, self._mac_dict, i_candidate, self.op_graph)

        self._phase_two_sol = self.get_phase_two_solution(self.quantizer_group_graph)

    def get_all_param_quantizers(self):
        """
        Returns all parameter quantizers from the quantizer-groups
        """
        parameter_quantizers = []
        for qgroup in self._quantizer_groups:
            if qgroup.parameter_quantizers:
                for i_para in qgroup.parameter_quantizers:
                    parameter_quantizers.append(i_para)
        return parameter_quantizers

    def get_networkx_graph(self):
        """
        Converts the connected graph to a networkx directional graph
        """
        G = nx.DiGraph()

        # pylint: disable=no-member
        parent_child_op_groups = self.find_op_groups_all(self._sim.connected_graph)
        for parent, children in parent_child_op_groups.items():
            _logger.info("-----------------------------")
            _logger.info("parent: %s", parent)
            _logger.info("children: %s", children)

            if parent not in G.nodes:
                G.add_node(parent)

            for child in children:
                if child not in G.nodes:
                    G.add_node(child)

                G.add_edge(parent, child)
        return G

    @staticmethod
    def rename_nodes(G: nx.DiGraph, module_name_to_module_dict: Dict, dotted_name2op_name: Dict, ops_dict: Dict,
                     find_wrapper_module: Callable):
        """
        Rename nodes of networkx graph from op dotted name to quantizer-group name
        :param G: networkx graph
        :param module_name_to_module_dict: module-name to quant-wrapper dict
        :param dotted_name2op_name: cg-op dotted name to cg-op name dict
        :param ops_dict: cg-op name to cg op dict
        :param find_wrapper_module: function that returns module-name & quant-wrapper given cg-op dotted name
        :return: renamed networkx graph
        """
        # pylint: disable=bare-except
        G_copy = G.copy()

        for node in G_copy.nodes:
            if node not in ["input_ops", "output_ops"]:
                if ("input_ops", node) in G_copy.edges:  # quantizer groups with an input quantizer
                    module_name, _ = find_wrapper_module(node, module_name_to_module_dict)
                    if module_name is not None:
                        mapping = {node: module_name + "_output"}
                        G = nx.relabel_nodes(G, mapping)

                        G.add_node(module_name + "_input")
                        G.add_edge(module_name + "_input", module_name + "_output")

                        # populate the data for the new input node
                        new_node_name = module_name + "_input"
                        op_name = dotted_name2op_name[node]
                        input_shape = ops_dict[op_name].inputs[0].shape # TODO: if there is more than one input tensor, this could be wrong

                        if input_shape is None:
                            input_shape = [1, 1, 1, 1]
                            _logger.info(
                                "WARNING: input_shape is None; setting it to [1,1,1,1] as a temporary fix.")  # TODO: to be revisited for a more permanent solution

                        input_shape = [dim if dim else 1 for dim in input_shape]
                        input_size = 1
                        for dim in input_shape:
                            input_size *= dim

                        G.nodes[new_node_name]["tensor_dims"] = input_shape
                        G.nodes[new_node_name]["tensor_size"] = input_size
                    else:
                        _logger.info("did not change node name: %s", node)
                else:
                    module_name, _ = find_wrapper_module(node, module_name_to_module_dict)
                    if module_name is not None:
                        mapping = {node: module_name + "_output"}
                        G = nx.relabel_nodes(G, mapping)
                    else:
                        _logger.info("did not change node name: %s", node)

        G.remove_node("input_ops")
        G.remove_node("output_ops")
        return G

    @staticmethod
    def mark_qg_nodes(G: nx.DiGraph, quantizer_group2node_index: Dict):
        """
        Identifies the nodes in the graph that represents a quantizer-group
        :param G: networkx graph
        :param quantizer_group2node_index: mapping of quantizer-group name and its position in the quantizer-group list
        :return:
        """
        nx.set_node_attributes(G, False, "is_quantizer_group")
        for qgroup_name in quantizer_group2node_index:
            if "_weights_only" in qgroup_name:
                G.add_node(qgroup_name)
                G.nodes[qgroup_name]["is_quantizer_group"] = True
            else:
                if qgroup_name in G.nodes:
                    G.nodes[qgroup_name]["is_quantizer_group"] = True
                else: # This happens when intermediate ops have input quantizers enabled (not a common scenario, happens for LeViT).
                    if "_input" in qgroup_name:
                        _logger.info("Warning: Unexpected enabling of input quantizer for %s", qgroup_name)
                        G.add_node(qgroup_name)
                        G.nodes[qgroup_name]["is_quantizer_group"] = True

                        ind = qgroup_name.rfind("_input")
                        successor_node = qgroup_name[:ind] + "_output"
                        G.add_edge(qgroup_name, successor_node)
                        G.nodes[qgroup_name]["tensor_dims"] = G.nodes[successor_node]["tensor_dims"]
                        G.nodes[qgroup_name]["tensor_size"] = G.nodes[successor_node]["tensor_size"]

                        # reroute the edges
                        edges_to_remove = []
                        predecessors = [pred for pred in G.predecessors(successor_node) if pred != qgroup_name]
                        for pred in predecessors:
                            G.add_edge(pred, qgroup_name)
                            edges_to_remove.append((pred, successor_node))

                        for edge in edges_to_remove:
                            G.remove_edge(edge[0], edge[1])
                    else:
                        raise RuntimeError("Unexpected quantizer group type.")

    @staticmethod
    def contract_non_qg_nodes(G: nx.DiGraph):
        """
        Contract the non-quantizer-group nodes to their successor or predecessor quantizer-group node
        :param G: Op graph
        :return: QuantizerGroup Graph
        """
        num_quantizer_group_nodes = sum(G.nodes[node]["is_quantizer_group"] for node in G.nodes)
        while num_quantizer_group_nodes < G.number_of_nodes():
            for node in G.nodes:
                if not G.nodes[node]["is_quantizer_group"]:
                    successors = list(G.successors(node))
                    predecessors = list(G.predecessors(node))
                    if len(predecessors) == 1:
                        pred_node = predecessors[0]
                        if G.nodes[pred_node]["is_quantizer_group"]:
                            G = nx.contracted_nodes(G, pred_node, node, self_loops=False)
                            break
                    if len(successors) == 1:  # we need this for the Add op, which has 2 predecessors and one successor
                        succ_node = successors[0]
                        if G.nodes[succ_node]["is_quantizer_group"]:
                            G = nx.contracted_nodes(G, succ_node, node, self_loops=False)
                            break
            num_quantizer_group_nodes = sum(G.nodes[node]["is_quantizer_group"] for node in G.nodes)
        return G

    @staticmethod
    def permute_tensor_dims(input_tensor_dims: List, tensor_format: str):
        """
        Permute tensor dimensions according to the given format.
        :param input_tensor_dims: tensor dims
        :param tensor_format: axis format
        :return:
        """
        if tensor_format == "nchw":  # this is the 'transpose' op, example: [1,2,3,4] --> [1,4,2,3]
            input_tensor_dims[1:] = input_tensor_dims[3], input_tensor_dims[1], input_tensor_dims[2]
        elif tensor_format == "nhwc":  # this is the 'transpose' op, example: [1,2,3,4] --> [1,3,4,2]
            input_tensor_dims[1:] = input_tensor_dims[2], input_tensor_dims[3], input_tensor_dims[1]
        else:
            raise RuntimeError("Unsupported permute format.")

        return input_tensor_dims

    @staticmethod
    def fetch_op_info(ops_dict) -> Tuple[Dict, Dict]:
        """
        Collects op name, dotted name and output shape of CG ops
        :param ops_dict: CG ops
        :return:
        """
        dotted_name2output_shape = {}
        dotted_name2op_name = {}
        for op in ops_dict:
            # pylint: disable=protected-access
            dotted_name2output_shape[ops_dict[op].dotted_name] = [dim if dim else 1 for dim in ops_dict[op]._output_shape]
            dotted_name2op_name[ops_dict[op].dotted_name] = op
        return dotted_name2output_shape, dotted_name2op_name

    @staticmethod
    def add_node_info(G: nx.DiGraph, dotted_name2output_shape: Dict):
        """
        Adds output tensor dims and size to each node of the graph
        :param G: nextworkx graph
        :param dotted_name2output_shape:
        :return:
        """
        for node in G.nodes:
            if node not in ["input_ops", "output_ops"]:
                tensor_dims = dotted_name2output_shape[node]
                tensor_size = 1
                for dim in tensor_dims:
                    tensor_size *= dim
                G.nodes[node]["tensor_dims"] = tensor_dims
                G.nodes[node]["tensor_size"] = tensor_size

    @abc.abstractmethod
    def generate_graphs(self):
        """
        This function generates the quantizer group graph (nodes are contracted) and the op graph (nodes are not
        contracted). The implementation leverages networkx's contracted_nodes function.
        The nodes in the quantizer group graph are quantizer groups which are derived by contracting the non-quantizer
        group nodes of the graph. The contracted nodes includes the data movement ops (such as reshape, permute) and are
        present in the contracted_nodes field of quantizer group node.
        The op graph is a version of quantizer group graph before contraction of any node.

        :return: Tuple containing quantizer-group graph & op graph
        """

    def add_convert_cost_predictions_to_graph(self):
        """
        Predict the conversion cost and add it as edge weights to the graph.
        mode should be set to either "max" or "last_op"; this is used to determine which tensor dims to use for convert cost prediction

        Add tensor dims and size to the edges of the graph (needed for weight cut size calculation later)
        Procedure:
          Step 1: Create an empty list for each node of the QG_graph
          Step 2: Add the current node and all of its contracted nodes to this list
          Step 3 (if mode is "last_op"): Find the node in this list that doesn't have any successors (in the op_graph, i.e. graph before contractions)
          Step 3 (if mode is "max"): Find the node in this list that has the largest convert cost (in the op_graph, i.e. graph before contractions)
          Step 4: Compute the convert cost prediction based on the tensor dims for the node found in step 3
        """

        def convert_cost_prediction(tensor_dims, tensor_size):
            """
            Using convert cost prediction function computes the convert cost for given tensor dimensions
            :param tensor_dims: tensor dims
            :param tensor_size: tensor size
            :return:
            """
            # prediction_coefficients = [1.66187677e+01, 0, 0, 0, 2.76830774e-01, 3.67453119e+00, 0, 2.43569237e+03]
            prediction_coefficients = [2.75170939e+00, 0, 0, 8.06526234e-01, 0, 4.27417544e+00, 3.62439292e-02,
                                       2.25527950e+03]  # new coeffs 1/22/2024
            if len(tensor_dims) == 4:
                tensor_dims_corr = tensor_dims
            elif len(tensor_dims) == 2:
                tensor_dims_corr = list(tensor_dims)
                tensor_dims_corr.extend([1, 1])
            elif len(tensor_dims) == 3:  # TODO: the way to address 3-dim tensors needs to be reviewed further later!
                tensor_dims_corr = list(tensor_dims)
                tensor_dims_corr.extend([1])
            elif len(
                    tensor_dims) == 5:  # TODO: this is only temporary to make the SwinV2 model work, which has 5-dim tensors
                tensor_dims_corr = list(tensor_dims)[1:]
                _logger.info("WARNING: 5-dimensional tensor, taking the last 4 dimensions only!")
            else:
                raise RuntimeError("Unexpected tensor dimensions.")

            # pylint: disable=no-member
            convert_cost = \
                prediction_coefficients[0] * tensor_dims_corr[self.TensorAxis.C.value] + \
                prediction_coefficients[1] * tensor_dims_corr[self.TensorAxis.W.value] + \
                prediction_coefficients[2] * tensor_dims_corr[self.TensorAxis.H.value] + \
                prediction_coefficients[3] * tensor_dims_corr[self.TensorAxis.H.value] * tensor_dims_corr[self.TensorAxis.C.value] + \
                prediction_coefficients[4] * tensor_dims_corr[self.TensorAxis.W.value] * tensor_dims_corr[self.TensorAxis.C.value] + \
                prediction_coefficients[5] * tensor_dims_corr[self.TensorAxis.H.value] * tensor_dims_corr[self.TensorAxis.W.value] + \
                prediction_coefficients[6] * tensor_size + \
                prediction_coefficients[7] * math.ceil(tensor_dims_corr[self.TensorAxis.H.value] / 8)

            return convert_cost

        for edge in self.quantizer_group_graph.edges:
            origin_node = edge[0]
            candidate_nodes = []
            candidate_nodes.append(origin_node)
            if "contraction" in self.quantizer_group_graph.nodes[origin_node]:
                # pylint: disable=unnecessary-comprehension
                candidate_nodes.extend([contracted_node for contracted_node in
                                        self.quantizer_group_graph.nodes[origin_node]["contraction"]])

            max_cost = 0
            for node_i in candidate_nodes:
                convert_cost_node_i = convert_cost_prediction(self.op_graph.nodes[node_i]["tensor_dims"],
                                                              self.op_graph.nodes[node_i]["tensor_size"])
                if convert_cost_node_i >= max_cost:
                    max_cost = convert_cost_node_i
                    max_node = node_i
            self.quantizer_group_graph.edges[edge]["tensor_dims"] = self.op_graph.nodes[max_node]["tensor_dims"]
            self.quantizer_group_graph.edges[edge]["tensor_size"] = self.op_graph.nodes[max_node]["tensor_size"]
            self.quantizer_group_graph.edges[edge]["convert_cycles"] = max_cost

    @staticmethod
    def load_profiling_cost(qnn_profiling_file):
        """
        Loads convert cost from the qnn profiling file
        """
        cost_info = defaultdict(dict)
        with open(qnn_profiling_file, 'r') as file:
            lines = file.readlines()
            start = 0
            for i_line in lines:
                if 'Backend (Accelerator (execute) time (cycles))' in i_line:
                    start = 1
                    continue
                if 'Backend (Accelerator (execute) time)' in i_line:
                    start = 0
                    break
                if start == 1:
                    op_fullname = i_line.split(':')[0].strip().split(' ')[0]
                    op_type = op_fullname.split('_')[-1]
                    cycle = int(i_line.split(':')[-1].split('cycles')[0])
                    cost_info[op_fullname].update({'cycle': cycle, 'op': op_type})
        return cost_info

    @staticmethod
    def add_bit_op_cost_to_graph(networkx_graph, mac_dict, candidate, op_graph, op_cost_from_hw=False,
                                 qnn_profiling_file=None):
        """
        Generates bit-operations cost for every node in the graph

        :param networkx_graph: networkx directional graph
        :param mac_dict: mac dict of the ops
        :param candidate: candidate to use to compute cost
        :param op_graph: networkx graph before contraction
        :param op_cost_from_hw: use hw generated cost results
        :param qnn_profiling_file: if op_cost_from_hw is True, qnn_profiling_file needs to be passed in
        """
        (act_bw, _), (param_bw, _) = candidate
        cost_info = None
        if op_cost_from_hw:
            assert os.path.isfile(qnn_profiling_file)
            # TODO change the file name based on act_bw/param_bw
            cost_info = ReduceConvertOps.load_profiling_cost(qnn_profiling_file)
        for i_node in networkx_graph.nodes:
            bit_op = ReduceConvertOps.generate_cost_for_node(networkx_graph, i_node, mac_dict, act_bw, param_bw,
                                                             op_graph, cost_info)
            networkx_graph.nodes[i_node][(act_bw, param_bw)] = bit_op

    @staticmethod
    def generate_cost_for_node(networkx_graph, node, mac_dict, act_bw, param_bw, op_graph, cost_info=None):
        """
        Helper method to generate bit-op cost for every node in the graph

        :param networkx_graph: networkx directional graph
        :param node: node in the graph
        :param mac_dict: mac dict of the ops
        :param act_bw: activation bw to use
        :param param_bw: param bw to use
        :param op_graph: networkx graph before contraction
        :param cost_info: cost info from the hardware
        """
        # pylint: disable=too-many-nested-blocks

        op_cost = 0
        nodes_collection = [node]
        if 'contraction' in networkx_graph.nodes[node]:
            for i in networkx_graph.nodes[node]['contraction']:
                nodes_collection.append(i)
        pattern = r'_(input|output|weights_only)$'
        for i_node in nodes_collection:
            if 'output' in i_node:
                op_name = re.sub(pattern, '', i_node)

                if cost_info is not None:
                    if op_name in cost_info:
                        op_cost += cost_info[op_name]['cycle']
                else:
                    if op_name in mac_dict:
                        if op_graph.nodes[i_node]['has_weight']:
                            op_cost += mac_dict[op_name] * act_bw * param_bw
                        else:
                            # pylint: disable=unnecessary-comprehension
                            i_node_pred = [i for i in op_graph.predecessors(i_node)]
                            if len(i_node_pred) == 1:
                                op_cost += mac_dict[op_name] * act_bw
                            elif len(i_node_pred) == 2:
                                # extend this logic for other Ops if needed
                                assert 'Mul' in op_name
                                op_cost += mac_dict[op_name] * act_bw * act_bw
                            else:
                                raise RuntimeError('unexpected cases for op_cost calculation')

        return op_cost

    @abc.abstractmethod
    def get_phase_two_solution(self, qg_graph):
        """
        This function returns the phase 2 solution i.e. the act bw of each quantizer-group
        :param qg_graph: quantizer-group graph
        """

    @staticmethod
    def compute_num_precision_changes(qg_graph, solution_dict):
        """
        This function computes the number of precision changes for a given AMP solution.
        """
        nodes_8bit = set()
        nodes_16bit = set()

        for qgroup in solution_dict:
            if solution_dict[qgroup] == 8:
                nodes_8bit.add(qgroup)
            elif solution_dict[qgroup] == 16:
                nodes_16bit.add(qgroup)
            else:
                raise RuntimeError("Bitwidth not supported.")

        _logger.info("Number of 8-bit activations is %d", len(nodes_8bit))
        _logger.info("Number of 16-bit activations is %d", len(nodes_16bit))

        return nx.cut_size(qg_graph, nodes_8bit, nodes_16bit)

    @staticmethod
    def compute_weighted_cut_size(qg_graph, solution_dict, weight_key="convert_cycles"):
        """
        This function computes the weighted cut size for a given AMP solution.
        Assume that the weights are already specified in the QG_graph.
        """
        nodes_8bit = set()
        nodes_16bit = set()

        for qgroup in solution_dict:
            if solution_dict[qgroup] == 8:
                nodes_8bit.add(qgroup)
            elif solution_dict[qgroup] == 16:
                nodes_16bit.add(qgroup)
            else:
                raise RuntimeError("Bitwidth not supported.")

        return nx.cut_size(qg_graph, nodes_8bit, nodes_16bit, weight=weight_key)

    def run_amp_phase_3(self, alpha, sampling_strategy: SamplingStrategy):
        """
        This is the function to use for running phase 3.
        """
        _logger.info("Solving MIP for alpha = %f", alpha)

        if alpha < 1:
            phase_three_sol, solve_data_dict = ReduceConvertOps.run_convert_op_reduction_solver(
                self.quantizer_group_graph,
                alpha, self._phase_two_sol, self._candidates,
                sampling_strategy)

        elif alpha == 1.0:  # output the phase 2 solution
            phase_three_sol = self._phase_two_sol.copy()
            solve_data_dict = {}
            solve_data_dict["status"] = "alpha = 1.0; solution copied from phase 2."
            solve_data_dict["num_precision_changes_phase_2"] = ReduceConvertOps.compute_num_precision_changes(
                self.quantizer_group_graph,
                self._phase_two_sol)
        else:
            raise RuntimeError("Invalid value for alpha.")

        return phase_three_sol, solve_data_dict

    @staticmethod
    def run_convert_op_reduction_solver(qg_graph, alpha, phase_two_sol, candidates, sampling_strategy):
        """
        This function solves the MIP for controlling conversion ops.
        qg_graph is the quantizer group graph in networkx.
        NOTE: This function currently only supports 8bit & 16bit activations (i.e. two precision sets);
        will be extended to the general case of 2+ precision candidates later
        """

        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals

        assert len(candidates) == 2  # support for more later

        num_nodes = qg_graph.number_of_nodes()

        node2index = {}
        for i, node in enumerate(qg_graph.nodes):
            if "_weights_only" not in node:
                node2index[node] = i

        # calculate num_precision_changes
        num_precision_changes = ReduceConvertOps.compute_num_precision_changes(qg_graph, phase_two_sol)
        _logger.info(
            "Number of precision changes (i.e. cut size with all weights set to 1) for the phase 2 solution is: %d",
            num_precision_changes)

        # calculate weighted cut size
        weighted_cut_size = ReduceConvertOps.compute_weighted_cut_size(qg_graph, phase_two_sol)
        _logger.info("Weighted cut size for the phase 2 solution is: %d", weighted_cut_size)

        # variables
        x = cp.Variable(num_nodes, boolean=True)
        y = cp.Variable(num_nodes, boolean=True)

        # constraints
        constraints = []
        constraints += [x + y == 1]

        # don't change precision of a node if phase 2 solution is 16 bits
        for node in qg_graph.nodes:
            if "_weights_only" not in node:
                if phase_two_sol[node] == 16:
                    constraints += [y[node2index[node]] == 1]

        # constraint for number of precision changes
        if sampling_strategy == SamplingStrategy.unweighted_cut_size:
            precision_change_terms = []
            for u, v in qg_graph.edges:
                precision_change_terms.append(cp.abs(x[node2index[u]] - x[node2index[v]]))
            constraints += [cp.sum(precision_change_terms) <= alpha * num_precision_changes]
        elif sampling_strategy == SamplingStrategy.weighted_with_tensor_size:  # using tensor size as conversion cost
            weight_cut_terms = []
            for u, v in qg_graph.edges:
                weight_cut_terms.append(
                    cp.abs(x[node2index[u]] - x[node2index[v]]) * qg_graph.edges[(u, v)]["tensor_size"])
            weighted_cut_size_tensor_size = ReduceConvertOps.compute_weighted_cut_size(qg_graph, phase_two_sol, weight_key="tensor_size")
            constraints += [cp.sum(weight_cut_terms) <= alpha * weighted_cut_size_tensor_size]
        elif sampling_strategy == SamplingStrategy.weighted_with_predicted_convert_cost:
            # similar to SamplingStrategy.weighted_with_tensor_size but the upper bound of the constraint is determined differently
            # # Step 1: get a list of conversion costs in phase 2 solution
            # conversion_costs = []
            # for edge in QG_graph.edges:
            #     if phase_two_sol[edge[0]] != phase_two_sol[edge[1]]: # check for conversion
            #         conversion_costs.append(QG_graph.edges[edge]["convert_cycles"])
            # Step 1 (new -- 02/07/2024): get a list of convert values for all edges
            conversion_costs = []
            for edge in qg_graph.edges:
                conversion_costs.append(qg_graph.edges[edge]["convert_cycles"])
            # Step 2: sort the conversion costs list
            conversion_costs.sort()
            conversion_costs_np = np.array(conversion_costs)
            conversion_costs_np_cumsum = np.cumsum(conversion_costs_np)
            # Step 3: compute cumulative sums for the sorted conversion costs and determine the threshold
            if alpha == 0:
                cost_sum_threshold = 0
            else:
                ind_conv_cost_total_ph2 = np.where(conversion_costs_np_cumsum < weighted_cut_size)[0][-1]
                # cost_sum_threshold = conversion_costs_np_cumsum[int(alpha * len(conversion_costs))]
                cost_sum_threshold = conversion_costs_np_cumsum[int(alpha * ind_conv_cost_total_ph2)]
            # Step 4: Enforce the constraint
            weight_cut_terms = []
            for u, v in qg_graph.edges:
                weight_cut_terms.append(
                    cp.abs(x[node2index[u]] - x[node2index[v]]) * qg_graph.edges[(u, v)]["convert_cycles"])
            constraints += [cp.sum(weight_cut_terms) <= cost_sum_threshold]
        else:
            raise RuntimeError(
                "Available versions are unweighted_cut_size(1), weighted_with_tensor_size(2), and weighted_with_predicted_convert_cost(3)")

        # objective
        objective_terms = []
        for i, node in enumerate(qg_graph.nodes):
            if "_weights_only" not in node:
                for i_candidate in candidates:
                    (act_bw, _), (param_bw, _) = i_candidate
                    key = (act_bw, param_bw)
                    if key == (8, 8):
                        objective_terms.append(qg_graph.nodes[node][key] * x[i])
                    elif key == (16, 8):
                        objective_terms.append(qg_graph.nodes[node][key] * y[i])
                    else:
                        raise RuntimeError("Considering only 8 or 16 bit activation candidates.")
        objective = cp.sum(objective_terms)

        # cvxpy problem instance
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.CBC, maximumSeconds=10, verbose=True)

        # print information about the cvxpy solution and save data to solve_data_dict
        solve_data_dict = {}
        solve_data_dict["graph_number_of_nodes"] = qg_graph.number_of_nodes()
        solve_data_dict["graph_number_of_edges"] = qg_graph.number_of_edges()
        solve_data_dict["alpha"] = alpha
        solve_data_dict["sampling_version"] = str(sampling_strategy)

        solve_data_dict["problem_status"] = prob.status
        solve_data_dict["objective_value"] = objective.value
        _logger.info("Solver outputs:")
        _logger.info("---------------")
        _logger.info("prob.status: %s", prob.status)
        _logger.info("objective.value: %s", objective.value)
        phase_three_sol = {}
        for node, idx in node2index.items():
            if round(x[idx].value) == 1 and round(y[idx].value) == 0:  # "round" is for tolerance
                phase_three_sol[node] = 8
            elif round(x[idx].value) == 0 and round(y[idx].value) == 1:
                phase_three_sol[node] = 16
            else:
                raise RuntimeError("The solution has a bug")

        solve_data_dict["num_nodes_8bits_phase2"] = sum(sol == 8 for sol in phase_two_sol.values())
        solve_data_dict["num_nodes_16bits_phase2"] = sum(sol == 16 for sol in phase_two_sol.values())
        solve_data_dict["num_nodes_8bits_phase3"] = sum(sol == 8 for sol in phase_three_sol.values())
        solve_data_dict["num_nodes_16bits_phase3"] = sum(sol == 16 for sol in phase_three_sol.values())

        num_precision_changes_phase_3 = ReduceConvertOps.compute_num_precision_changes(qg_graph, phase_three_sol)
        _logger.info("num_precision_changes_phase_3: %d", num_precision_changes_phase_3)
        weighted_cut_size_phase_3 = ReduceConvertOps.compute_weighted_cut_size(qg_graph, phase_three_sol)
        _logger.info("weighted_cut_size_phase_3: %d", weighted_cut_size_phase_3)

        solve_data_dict["num_precision_changes_phase_2"] = num_precision_changes
        solve_data_dict["num_precision_changes_phase_3"] = num_precision_changes_phase_3
        solve_data_dict["weighted_cut_size_phase_2"] = weighted_cut_size
        solve_data_dict["weighted_cut_size_phase_3"] = weighted_cut_size_phase_3

        objective_eval_for_phase2_sol = 0
        for i, node in enumerate(qg_graph.nodes):
            if "_weights_only" not in node:
                if phase_two_sol[node] == 8:
                    objective_eval_for_phase2_sol += qg_graph.nodes[node][(8, 8)]
                elif phase_two_sol[node] == 16:
                    objective_eval_for_phase2_sol += qg_graph.nodes[node][(16, 8)]
                else:
                    raise NotImplementedError("Considering only 8 or 16 bit activation candidates.")

        solve_data_dict["objective_eval_for_phase2_sol"] = objective_eval_for_phase2_sol

        return phase_three_sol, solve_data_dict

    @abc.abstractmethod
    def generate_qg_solution(self, phase_three_sol):
        """
        Given the phase-3 solution i.e. the act bw of each quantizer-group, this function returns the quantizer-group
        object and its appropriate amp-candidate
        """

    @staticmethod
    def save_phase_3_data_as_json(solve_data_dict, results_dir, filename_suffix=""):
        """
        This function saves the phase 3 solution as a json file.
        """
        filename = results_dir + "/AMP_phase_3_data_{}.json".format(filename_suffix)
        with open(filename, 'w') as fp:
            json.dump(solve_data_dict, fp)

    def apply(self, results_dir: str,
              sampling_strategy: SamplingStrategy = SamplingStrategy.weighted_with_predicted_convert_cost,
              alpha_options: List = None):
        """
        Applies convert op reduction algorithm

        :param results_dir: results directory
        :param sampling_strategy: Sampling strategy to apply
        :param alpha_options: List of alpha values to try (alpha values assumed to be between 0 and 1 both inclusive)
        """

        if not alpha_options:
            alpha_options = self.DEFAULT_ALPHA_OPTIONS

        for alpha in alpha_options:
            _logger.info("Solving MIP for alpha: %f", alpha)
            phase_three_sol, solve_data_dict = self.run_amp_phase_3(alpha, sampling_strategy)
            _ = self.generate_qg_solution(phase_three_sol)

            ReduceConvertOps.save_graph_dot_w_color("dummy_model", self.quantizer_group_graph, self._phase_two_sol,
                                                    results_dir, filename_suffix="phase2sol")
            ReduceConvertOps.save_graph_dot_w_color("dummy_model", self.quantizer_group_graph, phase_three_sol,
                                                    results_dir, filename_suffix="phase3sol_alpha{}".format(alpha))
            ReduceConvertOps.save_phase_3_data_as_json(solve_data_dict, results_dir, filename_suffix="alpha_{}".format(alpha))

    @staticmethod
    def save_graph_dot_w_color(model_name, qg_graph, solution_dict, results_dir, filename_suffix=""):
        """
        This function saves the quantizer group graph as a dot file with node colors indicating their bitwidth.
        We need a copy without the contraction attribute; otherwise, pydot throws the following error:
        ValueError: Node names and attributes should not contain ":"
        Also we need to remove tuples, otherwise dot2png conversion doesn't work
        """
        G_without_contraction_attribute = qg_graph.copy()
        for node in G_without_contraction_attribute.nodes:
            if "contraction" in G_without_contraction_attribute.nodes[node]:
                del G_without_contraction_attribute.nodes[node]["contraction"]
            for field in qg_graph.nodes[node]:
                if isinstance(field, tuple):
                    del G_without_contraction_attribute.nodes[node][field]

        # fill in node color attributes
        for node in G_without_contraction_attribute.nodes:
            if "_weights_only" not in node:
                if solution_dict[node] == 8:
                    G_without_contraction_attribute.nodes[node]["color"] = "cyan"
                elif solution_dict[node] == 16:
                    G_without_contraction_attribute.nodes[node]["color"] = "red"
                else:
                    raise NotImplementedError("Just considering 8 or 16 bit activation candidates.")
            G_without_contraction_attribute.nodes[node]["style"] = "filled"

        # Also, need to remove edge attributes, otherwise we get a pydot error
        for edge in G_without_contraction_attribute.edges: # pylint: disable=consider-using-dict-items
            for attribute in qg_graph.edges[edge]:
                del G_without_contraction_attribute.edges[edge][attribute]

        p = to_pydot(G_without_contraction_attribute)
        p.write(results_dir + "/AMP_ph3_quantizer_group_graph_{}_color_{}.dot".format(model_name, filename_suffix))

        # Note: We can later convert the dot file to png using: "dot filename.dot -Tpng -o filename.png"
