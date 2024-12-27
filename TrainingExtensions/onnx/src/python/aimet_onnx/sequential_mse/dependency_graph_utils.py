# /usr/bin/env python
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

""" Dependency Graph Utils implementation """

# pylint: disable=ungrouped-imports
from aimet_onnx.sequential_mse.dependency_graph import DependencyGraph
from aimet_onnx.utils import create_input_dict
from aimet_onnx.meta.operations import Op

import numpy as np

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines
from onnx import NodeProto

# The following modules with weights are supported
SUPPORTED_MODULES = ("Conv", "Gemm", "MatMul")
DEPENDENCY_MODULES = ("Conv", "Gemm", "MatMul", "Add")


class DependencyGraphUtils:
    """
    Class containing utils to create the dependency graph
    """

    def __init__(self, connected_graph, node_name_to_input_names, static_tensor_name_to_proto):
        """
        Initializes the DependencyGraphUtils object

        :param connected_graph: Connected Graph for which the dependency graph needs to be created
        :param node_name_to_input_names: Mapping from name to input names
        :param static_tensor_name_to_proto: Mapping from static tensor name to proto buf
        """

        self.connected_graph = connected_graph
        self.node_name_to_input_names = node_name_to_input_names
        self.indegree = dict()
        self.name_to_dependent_on_supported_module = dict()
        self.static_tensor_name_to_proto = static_tensor_name_to_proto
        self.starting_ops = list()
        self.graph_outputs = [output.name for output in self.connected_graph.model.graph.output]
        self.input_ops_name = list()

        self._fill_indegree()
        self._fill_input_ops_name()
        self._init_name_to_dependent_on_supported_module()

    def _fill_input_ops_name(self):
        """
        Fill the input op names dict with ops having at least one graph input
        """

        graph_inputs = list()

        for input_tensor in self.connected_graph.model.graph.input:
            graph_inputs.append(input_tensor.name)

        for node_name, input_names in self.node_name_to_input_names.items():
            for input_name in input_names:
                if input_name in graph_inputs:
                    self.input_ops_name.append(node_name)
                    break

    def _fill_indegree(self):
        """
        Initializes the indegree using the connected graph
        """

        for op in self.connected_graph.ordered_ops:
            self.indegree[op.name_op] = len(op.input_ops)
            if self.indegree[op.name_op] == 0:
                self.starting_ops.append(op)

    def _init_name_to_dependent_on_supported_module(self):
        """
        Initializes name to dependent supported module dict with empty list
        """

        for op in self.connected_graph.ordered_ops:
            self.name_to_dependent_on_supported_module[op.name_op] = list()

    def _populate_data_for_starting_ops(self, dependency_graph, data_loader, num_batches):
        """
        Initializes data for input to the graph using data loader

        :param dependency_graph: Dependency Graph
        """

        model_inputs = [node.name for node in self.connected_graph.model.graph.input]

        data = dict()

        for model_input in model_inputs:
            data[model_input] = list()

        for i, batch in enumerate(data_loader):
            if i == len(data_loader) - 1 and len(batch) < data_loader.batch_size:
                continue
            if i >= num_batches:
                break
            batch_dict = create_input_dict(self.connected_graph.model, batch)
            for model_input in model_inputs:
                if model_input not in batch_dict.keys():
                    raise ValueError("All inputs to the graph must be present in the dataloader. ", model_input,
                                     " is missing in the dataloader")

                data_to_insert = np.array(batch_dict[model_input])
                data[model_input].append(data_to_insert)

        for (input_name, data_value) in data.items():
            dependency_graph.update_float_data([input_name], [data_value])
            dependency_graph.update_sim_data([input_name], [data_value])

    def create_dependency_graph(self, data_loader, num_batches):
        """
        Create the dependency graph using topo sort starting from starting ops i.e. nodes with indegree zero
        :return: Dependency Graph
        """

        dependency_graph = DependencyGraph()

        for start_op in self.starting_ops:
            self._create_dependency_graph_helper(start_op, dependency_graph)

        self._populate_data_for_starting_ops(dependency_graph, data_loader, num_batches)

        return dependency_graph

    def _create_dependency_graph_helper(self, src_op: Op, dependency_graph: DependencyGraph):
        """
        1) Checks, if we can insert the node in dependency graph using the module type
        2) If the module is supported then we insert the node in dependency graph
        3) update the name_to_dependent_on_supported_module dict of the child op

        :param src_op: Current Op
        :param dependency_graph: Dependency Graph
        :return:
        """

        is_module_supported = False

        if src_op.model_module is not None:
            module = src_op.model_module.get_module()
            if self.is_dependency_module(module) or src_op.name in self.input_ops_name:
                is_module_supported = True
                op_name = src_op.name_op
                op_type = src_op.type
                op_output_names = module.output
                fp_module_input_names = self.node_name_to_input_names[src_op.name_op]
                op_input_names = [module_input_name for module_input_name in fp_module_input_names if module_input_name
                                  not in self.static_tensor_name_to_proto.keys()]
                dependent_on_supported_module = self.name_to_dependent_on_supported_module[src_op.name_op]
                dependency_graph.add_node(op_name=op_name, op_output_names=op_output_names,
                                          op_input_names=op_input_names, op_type=op_type,
                                          dependent_on_supported_module=dependent_on_supported_module)

        for child_op in src_op.output_ops:

            if is_module_supported:
                self.name_to_dependent_on_supported_module[child_op.name_op].append(src_op.name_op)
            else:
                self.name_to_dependent_on_supported_module[child_op.name_op] += self.name_to_dependent_on_supported_module[src_op.name_op]

            self.name_to_dependent_on_supported_module[child_op.name_op] = list(set(self.name_to_dependent_on_supported_module[child_op.name_op]))

            self.indegree[child_op.name] = self.indegree[child_op.name] - 1

            if self.indegree[child_op.name] == 0:
                self._create_dependency_graph_helper(child_op, dependency_graph)

    def is_supported_module(self, node: NodeProto):
        """
        Checks if the node is supported depending on the type

        :param node: Corresponding node proto
        :return: True, if the module is supported
        """

        if node.op_type not in SUPPORTED_MODULES:
            return False

        if len(node.input) < 1:
            return False

        weight_input_name = node.input[1]

        return weight_input_name in self.static_tensor_name_to_proto.keys()

    def is_dependency_module(self, node: NodeProto):
        """
        hecks if the node can be inserted in the dependency graph

        :param node: Corresponding node proto
        :return: True, if we can insert the node in dependency graph
        """

        for node_output_name in node.output:
            if node_output_name in self.graph_outputs:
                return False

        if node.op_type in (set(DEPENDENCY_MODULES) - set(SUPPORTED_MODULES)) and len(node.input) >= 1:
            return True

        return self.is_supported_module(node)
