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

""" Dependency Graph implementation """

import numpy as np

class DependencyNode:
    """
    Class for node of dependency graph
    """
    def __init__(self, op_name, op_output_names, op_input_names, op_type):
        """
        Initializes the DependencyNode object

        :param op_name: name of the op
        :param op_output_names: output names of the op
        :param op_input_names: input names of the op
        :param op_type: type (Conv/Gemm/MatMul/Add) of the op
        """

        self.op_name = op_name
        self.op_output_names = op_output_names # Check
        self.op_input_names = op_input_names
        self.inward_nodes = []
        self.outward_nodes = []
        self.outdegree = 0
        self.op_type = op_type
        self.indegree = 0

class DependencyGraph:
    """
    Dependency Graph APIs
    """
    def __init__(self):
        """
        Intializes the object of the Dependency Graph
        """
        self.starting_ops = []
        self.node_by_name = {}
        self.float_data = {}
        self.sim_data = {}
        self.ref_cnt_for_float_data = {}
        self.ref_cnt_for_sim_data = {}

    def add_node(self, op_name, op_output_names, op_input_names, op_type, dependent_on_supported_module):
        """
        Insert the dependency node in the graph

        :param op_name: name of the op
        :param op_output_names: output names of the op
        :param op_input_names: input names of the op
        :param op_type: type (Conv/Gemm/MatMul/Add) of the op
        :param dependent_on_supported_module: nodes on which this node is dependent on (inward nodes)
        """

        dependency_node = DependencyNode(op_name, op_output_names, op_input_names, op_type)

        for input_name in op_input_names:
            self.float_data[input_name] = np.array([])
            self.sim_data[input_name] = np.array([])
            if input_name in self.ref_cnt_for_float_data:
                self.ref_cnt_for_float_data[input_name] += 1
                self.ref_cnt_for_sim_data[input_name] += 1
            else:
                self.ref_cnt_for_float_data[input_name] = 1
                self.ref_cnt_for_sim_data[input_name] = 1

        self.node_by_name[op_name] = dependency_node

        dependency_node.indegree = len(dependent_on_supported_module)

        if dependency_node.indegree == 0:
            self.starting_ops.append(dependency_node)

        for parent_dependency_node_name in dependent_on_supported_module:
            parent_dependency_node =  self.node_by_name[parent_dependency_node_name]
            dependency_node.inward_nodes.append(parent_dependency_node)
            parent_dependency_node.outward_nodes.append(dependency_node)
            parent_dependency_node.outdegree += 1

    def get_float_data(self, dependency_node):
        """
        :param dependency_node: Corresponding dependency node
        :return: returns the float data of the input tensor
        """

        float_data_for_dependency_node = {}

        for input_name in dependency_node.op_input_names:
            float_data_for_dependency_node[input_name] = self.float_data[input_name]
        return float_data_for_dependency_node

    def get_sim_data(self, dependency_node):
        """
        :param dependency_node: Corresponding dependency node
        :return: returns the sim data of the input tensor
        """

        sim_data_for_dependency_node = {}

        for input_name in dependency_node.op_input_names:
            sim_data_for_dependency_node[input_name] = self.sim_data[input_name]

        return sim_data_for_dependency_node

    def update_float_data(self, names, data):
        """
        Updates the float values of the corresponding names

        :param names: name for which the value needs to updated
        :param data:  value
        """

        for i, name in enumerate(names):
            self.float_data[name] = data[i]

    def update_sim_data(self, names, data):
        """
        Updates the sim values of the corresponding names

        :param names: name for which the value needs to updated
        :param data:  value
        """

        for i, name in enumerate(names):
            self.sim_data[name] = data[i]

    def dec_ref_count(self, dependency_node):
        """
        Decreases the reference count for the float and sim data

        :param dependency_node: Corresponding dependency node
        """

        for input_name in dependency_node.op_input_names:

            self.ref_cnt_for_float_data[input_name] -= 1
            self.ref_cnt_for_sim_data[input_name] -= 1

            if self.ref_cnt_for_float_data[input_name] == 0:
                self.float_data[input_name] = {}

            if self.ref_cnt_for_sim_data[input_name] == 0:
                self.sim_data[input_name] = {}
