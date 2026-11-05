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
        2. If Op is in SUPPORTED_MODULES, Op contains at least one model input(s), or Op is a
           residual Add (see _is_residual_add)
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

        # Tracks the cg op name to its visit position in _fill_dependency_graph. Ops are visited
        # only after all their producers, so this is a topological index.
        self._visit_index = {}

        # NOTE: The traversal starts at top-level ops and visits only top-level ops, so the ops
        # inside control-flow bodies (e.g. Scan) are not added, and their weights are not
        # optimized. A body runs as a whole inside its op, so its per-iteration inputs can't be
        # cached, and the body can't be split into subgraphs like the top-level graph.
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
         - Add dependency node only if op is of type SUPPORTED_MODULES, has at least one input as
           model graph input(s), or is a residual Add (see _is_residual_add)

        1) Checks if we can insert the dependency node in dependency graph using the module type
        2) Update the _op_name_to_dependency_names of the child op based on the parent op type.

        :param src_op: Current Op
        """
        self._visit_index[src_op.name] = len(self._visit_index)

        if src_op.name not in self._nodes_to_exclude:
            op = src_op.get_module()
            supported_op = self.is_supported_op(src_op)
            residual_add = self._is_residual_add(src_op)
            if op and (
                supported_op
                or src_op.name in self._op_names_with_model_inputs
                or residual_add
            ):
                dependent_op_names = self._op_name_to_dependency_names[src_op.name]
                self._add_dependency_node(src_op, dependent_op_names)
                _logger.debug(
                    f"Added {src_op.name} to dependency graph with dependent_op_names: {len(dependent_op_names)}"
                )
                if residual_add:
                    _logger.debug(
                        f"{src_op.name} is a residual Add: added as a dependency boundary, not to be optimized"
                    )
        else:
            _logger.debug(f"Skipping {src_op.name} as it's in nodes_to_exclude")

        # Children of a dependency node depend on that node; children of any other op inherit
        # the op's own dependencies.
        for child_op in src_op.output_ops:
            parent_dependencies = self._contributed_dependency_names(src_op.name)
            self._op_name_to_dependency_names[child_op.name].extend(parent_dependencies)
            self._op_name_to_dependency_names[child_op.name] = list(
                set(self._op_name_to_dependency_names[child_op.name])
            )

            self._in_degree_map[child_op.name] -= 1
            if self._in_degree_map[child_op.name] == 0:
                self._fill_dependency_graph(child_op)

    def _contributed_dependency_names(self, op_name: str) -> List[str]:
        """
        Returns the dependency names that the given (already visited) op passes to its children.

        :param op_name: Name of a visited connected graph op
        :return: [op_name] if the op is a dependency node, else the op's own dependency names
        """
        if op_name in self._name_to_node:
            return [op_name]
        return self._op_name_to_dependency_names[op_name]

    def _is_residual_add(self, cg_op: Op) -> bool:
        """
        Checks whether the op is a residual Add that should be a dependency boundary.

        Without a boundary, the children of an unsupported join inherit the union of both input
        branches' dependencies; on a residual stream this union grows with depth and delays the
        release of cached data. Only residual Adds are boundaries: other joins (attention
        MatMuls, mask Adds, RoPE, gating Muls) can have much larger inputs than the tensors they
        would replace, and their dependency sets do not grow across blocks.

        An Add is residual if it has exactly two activation inputs that contribute different
        dependency sets, the earlier producer feeds this Add and no other Add, and that producer
        is an ancestor of the later one through a path that contains a supported (weight) op,
        i.e. a skip connection around a weight branch.

        :param cg_op: Connected graph op whose producers have all been visited
        :return: True if the op is a residual Add
        """
        if cg_op.type != "Add":
            return False

        producers = [
            inp.producer
            for inp in cg_op.inputs
            if not (inp.is_const or inp.is_parm) and inp.producer is not None
        ]
        if len(producers) != 2:
            return False

        first, second = producers
        if frozenset(self._contributed_dependency_names(first.name)) == frozenset(
            self._contributed_dependency_names(second.name)
        ):
            return False

        early, late = sorted(producers, key=lambda op: self._visit_index[op.name])
        # A tensor broadcast to every block (attention mask, RoPE cos/sin) is an operand of
        # one Add per block, so from block 1 onwards it is an ancestor of that
        # block's scores through the preceding blocks' weight ops, and the path test below
        # accepts it. A residual skip tensor instead feeds exactly one Add, because each
        # block consumes the stream once and produces a new one.
        if [op for op in early.output_ops if op.type == "Add"] != [cg_op]:
            return False

        return self._has_path_through_weight_op(late, early)

    def _has_path_through_weight_op(self, late: Op, early: Op) -> bool:
        """
        Checks whether early is an ancestor of late through a path containing a supported op.

        The walk is bounded by early's visit index: an op visited no later than early cannot lie
        on a path out of early, so it is pruned. On a residual stack early is the previous block's
        residual Add, which confines the walk to a single block and keeps the total cost linear in
        the number of blocks rather than quadratic.

        Entries are keyed on (op, through_weight) rather than op alone, because an op that
        is reachable both with and without a weight op on the path must be expanded once
        per case; keying on the op alone would prune the weight-bearing path.

        :param late: Connected graph op visited after early
        :param early: Connected graph op
        :return: True if such a path exists
        """
        early_index = self._visit_index[early.name]
        stack = [(late, self.is_supported_op(late))]
        seen = set()
        while stack:
            op, through_weight = stack.pop()
            if (op.name, through_weight) in seen:
                continue
            # through_weight belongs to the path, not the op: an op reachable both with
            # and without a weight op on the way must be expanded once for each, so it
            # is part of the key.
            seen.add((op.name, through_weight))
            for parent in op.input_ops:
                if parent.name == early.name:
                    if through_weight:
                        return True
                    continue
                # Descendants of early are visited after it, so earlier ops cannot be on a path.
                if self._visit_index.get(parent.name, -1) <= early_index:
                    continue
                stack.append((parent, through_weight or self.is_supported_op(parent)))
        return False

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
