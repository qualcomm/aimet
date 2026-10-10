# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Dependency Graph implementation"""

from collections import defaultdict, deque
from typing import Union, List, Optional, Iterable, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
import onnx
from onnxruntime.quantization.onnx_model import ONNXModel
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph, WEIGHT_INDEX
from aimet_onnx.utils import create_input_dict, ParamUtils
from aimet_onnx.meta.operations import Op
from aimet_onnx.spinquant import is_online_rotation_op

# The following modules with weights are supported
SUPPORTED_MODULES = ("Conv", "Gemm", "MatMul")

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


@dataclass
class DependencyNode:
    """
    Class for node of dependency graph
    """

    cg_op: Op
    op_output_names: List[str]
    op_input_names: List[str]
    inward_nodes: List["DependencyNode"] = field(default_factory=list)
    outward_nodes: List["DependencyNode"] = field(default_factory=list)
    out_degree: int = 0
    in_degree: int = 0


class DependencyGraph:
    """
    The Dependency Graph is designed to cache intermediate inputs in memory for SUPPORTED_MODULES,
     allowing these intermediate inputs to be passed to child nodes to obtain the intermediate outputs.

    Flow:
        1. Iterate over each Op in ConnectedGraph
        2. If Op is in SUPPORTED_MODULES or Op contains at least one model input(s)
            2a. Create DependencyNode corresponding to Op in ConnectedGraph
            2b. Link the DependencyNode with adjacent nodes
            2c. Update the reference count of DependencyNode's inputs
                - which decides when to remove intermediate inputs from memory when no longer needed.
        3. Populate model inputs for starting ops.

    Methods:
        get_topologically_sorted_nodes(self): Get the topologically sorted SUPPORTED_MODULES.
        get_subgraph_inp_out_names(self, dependency_node: DependencyNode): Get the subgraph's input and output names to collect the inputs.
    """

    def __init__(
        self,
        connected_graph: ConnectedGraph,
        data_loader: Iterable,
        nodes_to_exclude: Optional[List[str]] = None,
    ):
        """
        Initializes the object of the Dependency Graph

        :param connected_graph: ConnectedGraph object
        :param data_loader: DataLoader object
        nodes_to_exclude: List of supported node name(s) to exclude from sequential MSE optimization
        """
        self.conn_graph = connected_graph
        self._nodes_to_exclude = list(nodes_to_exclude or [])

        # Always skip SpinQuant R3 online rotation MatMuls (fixed Hadamards, not
        # learnable weights, and unsafe to parallelize under GQA). The naming
        # convention is owned by the R3 pass; see is_online_rotation_op.
        online_rotation_ops = [
            op.name
            for op in self.conn_graph.ordered_ops
            if is_online_rotation_op(op) and op.name not in self._nodes_to_exclude
        ]
        if online_rotation_ops:
            _logger.info(
                "Excluding %d SpinQuant R3 online rotation op(s) from sequential MSE: %s",
                len(online_rotation_ops),
                ", ".join(online_rotation_ops),
            )
            self._nodes_to_exclude.extend(online_rotation_ops)

        #: Resolved exclusion set (user-supplied plus auto-excluded ops).
        self.nodes_to_exclude = self._nodes_to_exclude

        self.starting_ops = []  # Tracks nodes with zero in-degree (starting ops)
        self._name_to_node = {}  # Tracks a node name to the dependency node itself
        self._sim_data = {}  # Store data from the Quantsim model in memory
        self._ref_cnt_for_sim_data = defaultdict(
            int
        )  # Reference count to decide when to remove intermediate Quantsim data from memory

        # Tracks a cg op name to the names of cg ops it depends on
        self._op_name_to_dependency_names = defaultdict(list)
        for op in self.conn_graph.ordered_ops:
            self._op_name_to_dependency_names[op.name] = []

        # Tracks onnx op names with at least one model graph input(s)
        self._op_names_with_model_inputs = self._fill_op_names_with_model_inputs()

        # Tracks the cg op name to the number of incoming edges to the op.
        self._in_degree_map = self._fill_in_degree_map(self.conn_graph)

        for start_op in self.conn_graph.starting_ops:
            self._fill_dependency_graph(start_op)

        self._populate_data_for_starting_ops(data_loader)

    def get_topologically_sorted_nodes(self):
        """
        Get topologically sorted nodes.

        - The in_degree dictionary keeps track of the number of incoming edges for each node.
        - Nodes with zero in-degree are added to the queue.
        - Nodes are processed in BFS order, and their children's in-degrees are updated.

        :return: List of nodes.
        """

        def _populate_in_degree(dep_node: DependencyNode, in_degree: dict):
            """
            Helper function to populate in_degree dictionary

            :param dep_node: DependencyNode
            :param in_degree: Dictionary of in_degree for each node
            """
            stack = [dep_node]
            while stack:
                current_node = stack.pop()
                for child_node in current_node.outward_nodes:
                    if child_node.cg_op.name not in in_degree:
                        in_degree[child_node.cg_op.name] = child_node.in_degree
                        stack.append(child_node)

        def _top_sort_helper(queue, sorted_order):
            """
            Helper function for topologically sorted nodes
            """
            while queue:
                level, dep_node = queue.popleft()
                sorted_order[level].append(dep_node)
                _logger.debug(
                    f"Adding {dep_node.cg_op.name} to sorted_order at level {level}"
                )
                next_level = level + 1

                # Decrease the in-degree for children nodes
                for child_node in dep_node.outward_nodes:
                    in_degree[child_node.cg_op.name] -= 1
                    if in_degree[child_node.cg_op.name] == 0:
                        queue.append((next_level, child_node))

        # Populate in-degrees of all nodes
        in_degree = {}
        for dep_node in self.starting_ops:
            assert dep_node.in_degree == 0
            in_degree[dep_node.cg_op.name] = dep_node.in_degree
        for dep_node in self.starting_ops:
            _populate_in_degree(dep_node, in_degree)

        sorted_order = defaultdict(list)
        level = 0
        queue = deque()

        # Initialize the queue with nodes having zero in-degree
        for dep_node in self.starting_ops:
            assert dep_node.in_degree == 0
            queue.append((level, dep_node))

        _top_sort_helper(queue, sorted_order)

        return sorted_order

    def get_subgraph_inp_out_names(
        self, dep_nodes: List[DependencyNode]
    ) -> Tuple[List[str], List[str]]:
        """
        To gather input for the dependency nodes, retrieve the subgraph's input and output names.

        The subgraph input names should correspond to the parent dependency node input(s),
        while the subgraph output names should correspond to the dependency node's input(s).

        :param dep_nodes: List of dependency nodes at same level
        :return: Subgraph's input and output names.
        """
        input_names = set()
        output_names = set()

        for dep_node in dep_nodes:
            for inward_node in dep_node.inward_nodes:
                input_names.update(inward_node.op_input_names)

        model_inputs = {tensor.name for tensor in self.conn_graph.model.graph.input}

        for dep_node in dep_nodes:
            for name in dep_node.op_input_names:
                if name not in model_inputs:
                    output_names.update([name])

        return list(input_names), list(output_names)

    def dependency_node_inputs(
        self, dep_nodes: List[DependencyNode]
    ) -> Dict[str, np.ndarray]:
        """
        For given dependency nodes at given level, return inputs by iterating the parent dependency nodes.

        :param dep_nodes: List of dependency nodes at same level
        :return: float inputs and sim inputs
        """
        sim_inputs = {}
        for dep_node in dep_nodes:
            for inward_node in dep_node.inward_nodes:
                sim_inputs.update(self.get_sim_data([inward_node]))

        return sim_inputs

    def get_sim_data(
        self, dep_nodes: List[DependencyNode]
    ) -> Dict[str, List[np.ndarray]]:
        """
        :param dep_nodes: Corresponding dependency node
        :return: returns the sim data of the input tensor
        """
        sim_data = {}

        for dep_node in dep_nodes:
            for input_name in dep_node.op_input_names:
                sim_data[input_name] = self._sim_data[input_name]

        return sim_data

    def update_sim_data(self, names: List[str], data: List[List[np.ndarray]]):
        """
        Updates the sim values of the corresponding names

        :param names: name for which the value needs to updated
        :param data:  value
        """
        assert len(names) == len(data)
        for i, name in enumerate(names):
            self._sim_data[name] = data[i]

    def dec_ref_count(self, dependency_node: DependencyNode):
        """
        Decreases the reference count for the float and sim data

        :param dependency_node: Corresponding dependency node
        """
        for input_name in dependency_node.op_input_names:
            self._ref_cnt_for_sim_data[input_name] -= 1
            if self._ref_cnt_for_sim_data[input_name] == 0:
                del self._sim_data[input_name]
                _logger.debug(f"Deleted: {input_name}")

    @staticmethod
    def get_param_name(dep_node: DependencyNode) -> str:
        """
        Get the name of a parameter in the dependency node

        :param dep_node: dependency node
        :return: parameter name
        """
        if dep_node.cg_op.type not in SUPPORTED_MODULES:
            raise ValueError(
                f"get_param_name expects an op of type {SUPPORTED_MODULES}, but got "
                f"{dep_node.cg_op.type} ({dep_node.cg_op.name})."
            )
        name = None
        for param_name, (_, param_type) in dep_node.cg_op.parameters.items():
            if param_type == "weight":
                name = param_name
        if name is None:
            raise ValueError(
                f"No static weight parameter found for {dep_node.cg_op.name} "
                f"(type {dep_node.cg_op.type})."
            )
        return name

    def _populate_data_for_starting_ops(self, inputs: Iterable[Dict[str, np.ndarray]]):
        """
        Initializes float_data and sim_data dictionaries for model input(s) using data loader and number of batches.

        :param data_loader: DataLoader object
        """
        model_inputs = [node.name for node in self.conn_graph.model.graph.input]
        data = {model_input: [] for model_input in model_inputs}

        for batch_dict in inputs:
            if not isinstance(batch_dict, dict):
                raise TypeError(
                    f"Expected each input sample to be type `Dict[str, np.ndarray]` but got type {type(batch_dict)}. "
                    "To resolve this, ensure that `inputs` argument is type `Iterable[Dict[str, np.ndarray]]`"
                )
            for model_input in model_inputs:
                if model_input not in batch_dict.keys():
                    raise ValueError(
                        f"All inputs to the graph must be present in the dataloader. {model_input} is missing in the dataloader"
                    )
                data[model_input].append(batch_dict[model_input])

        for input_name, data_value in data.items():
            self.update_sim_data([input_name], [data_value])

    def _add_dependency_node(
        self, cg_op: Op, dependent_node_names: Optional[List[str]]
    ):
        """
        - Insert the dependency node in the graph
        - Link the node with adjacent nodes
        - Update the reference count of dependency_node's inputs

        :param cg_op: Connected graph op.
        :param dependent_node_names: nodes that this node depends on. (inward nodes)
        """
        op_output_names = [out.name for out in cg_op.outputs]
        # Strip transpose op if exist.
        op_inputs = [
            inp.producer.inputs[0]
            if cg_op.transposed_params
            and inp.producer
            and inp.producer.type == "Transpose"
            else inp
            for inp in cg_op.inputs
        ]
        op_input_names = [
            inp.name for inp in op_inputs if not (inp.is_const or inp.is_parm)
        ]
        dep_node = DependencyNode(cg_op, op_output_names, op_input_names)
        dep_node.in_degree = len(dependent_node_names)
        if dep_node.in_degree == 0:
            self.starting_ops.append(dep_node)

        self._name_to_node[cg_op.name] = dep_node

        # Link the node with adjacent nodes
        for name in dependent_node_names:
            parent_dep_node = self._name_to_node[name]
            dep_node.inward_nodes.append(parent_dep_node)
            parent_dep_node.outward_nodes.append(dep_node)
            parent_dep_node.out_degree += 1

        # Update the reference count of dependency_node's inputs
        self._update_input_ref_count(dep_node)

    def _fill_op_names_with_model_inputs(self) -> List[str]:
        """
        Fill the input op names dict with ops having at least one graph input
        """
        graph_inputs = [
            graph_inp.name for graph_inp in self.conn_graph.model.graph.input
        ]

        op_names_with_model_input = [
            op.name
            for op in self.conn_graph.ordered_ops
            if any(inp.name in graph_inputs for inp in op.inputs)
        ]

        return op_names_with_model_input

    @staticmethod
    def _fill_in_degree_map(conn_graph: ConnectedGraph) -> Dict[str, int]:
        """
        Initializes the in-degree dictionary which keeps track of the number of incoming edges for each node.
        """
        in_degree_map = {}
        for op in conn_graph.ordered_ops:
            in_degree_map[op.name] = len(op.input_ops)
        return in_degree_map

    def _fill_dependency_graph(self, src_op: Op):
        """
        Fill dependency graph by traversing connected graph ops using BFS.

        NOTE:
         - _op_name_to_dependency_names tracks a node name to the names of nodes it depends on
         - Add dependency node only if op is of type SUPPORTED_MODULES or has at least one input as model graph input(s)

        1) Checks if we can insert the dependency node in dependency graph using the module type
        2) Update the _op_name_to_dependency_names of the child op based on the parent op type.

        :param src_op: Current Op
        """
        is_op_supported = False

        if src_op.name not in self._nodes_to_exclude:
            op = src_op.get_module()
            if op and (
                self.is_supported_op(src_op)
                or src_op.name in self._op_names_with_model_inputs
            ):
                is_op_supported = True
                dependent_op_names = self._op_name_to_dependency_names[src_op.name]
                self._add_dependency_node(src_op, dependent_op_names)
                _logger.debug(
                    f"Added {src_op.name} to dependency graph with dependent_op_names: {len(dependent_op_names)}"
                )
        else:
            _logger.debug(f"Skipping {src_op.name} as it's in nodes_to_exclude")

        # If src_op is part of SUPPORTED_MODULES or has at least one input as a model input, include its name in _op_name_to_dependency_names.
        # Otherwise, include the names of the dependencies of parent_op in _op_name_to_dependency_names.
        for child_op in src_op.output_ops:
            parent_dependencies = (
                [src_op.name]
                if is_op_supported
                else self._op_name_to_dependency_names[src_op.name]
            )
            self._op_name_to_dependency_names[child_op.name].extend(parent_dependencies)
            self._op_name_to_dependency_names[child_op.name] = list(
                set(self._op_name_to_dependency_names[child_op.name])
            )

            self._in_degree_map[child_op.name] -= 1
            if self._in_degree_map[child_op.name] == 0:
                self._fill_dependency_graph(child_op)

    def is_supported_op(self, cg_op: Op) -> bool:
        """
        Checks if the node is supported depending on the type

        :param cg_op: Corresponding node proto
        :return: True, if the module is supported
        """
        if cg_op.type not in SUPPORTED_MODULES:
            return False
        if len(cg_op.inputs) < 1:
            return False
        tensor = cg_op.inputs[WEIGHT_INDEX].tensor
        if isinstance(tensor, onnx.TensorProto):
            return True

        # Check for transposed MatMul weight pattern
        if cg_op.type == "MatMul":
            weight_producer = cg_op.inputs[WEIGHT_INDEX].producer
            if (
                cg_op.transposed_params
                and weight_producer
                and weight_producer.type == "Transpose"
            ):
                transpose_input = weight_producer.inputs[0].tensor
                return isinstance(transpose_input, onnx.TensorProto)

        return False

    def _update_input_ref_count(self, dependency_node: DependencyNode):
        """
        Updates the input reference count for the given dependency node.

        :param dependency_node: Dependency node
        """
        for input_name in dependency_node.op_input_names:
            self._sim_data[input_name] = np.array([])
            self._ref_cnt_for_sim_data[input_name] += 1
