# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""For constructing a uniform representation of the computational graph for an ONNX model,
that is easy to navigate and stores information for the purpose of AIMET features.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a node that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional.

Operations and products are views of the ``onnx_ir`` nodes and values of the model, and read
their attributes and links from them on every call."""

import itertools
from collections import deque
from typing import Dict, List, Optional, Tuple
import onnx_ir

from aimet_onnx.common.connected_graph.connectedgraph import (
    ConnectedGraph as AimetCommonConnectedGraph,
    get_ordered_ops,
)
from aimet_onnx.common.onnx._utils import _iterate_graphs_recursive_ir
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import (  # pylint: disable=unused-import
    Op,
    INPUT_INDEX,
    WEIGHT_INDEX,
    BIAS_INDEX,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
)
from aimet_onnx.meta.product import Product
from aimet_onnx.ir_utils import static_tensor

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

OPS_WITH_PARAMS = [
    "Conv",
    "Gemm",
    "ConvTranspose",
    "BatchNormalization",
    "MatMul",
    "RNN",
    "LSTM",
    "GRU",
]
CONSTANT_TYPE = ["Constant", "ConstantOfShape"]


def _subgraphs_of(node: onnx_ir.Node) -> List[Tuple[str, onnx_ir.Graph]]:
    """
    Return every graph ``node`` carries as an attribute, with the name of that attribute.

    :param node: Node to inspect
    :return: Its subgraphs, outermost attribute order preserved
    """
    subgraphs = []
    for attr in node.attributes.values():
        if attr.type == onnx_ir.AttributeType.GRAPH:
            subgraphs.append((attr.name, attr.as_graph()))
        elif attr.type == onnx_ir.AttributeType.GRAPHS:
            subgraphs.extend(
                (f"{attr.name}[{i}]", graph) for i, graph in enumerate(attr.as_graphs())
            )
    return subgraphs


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together as producers and consumers of tensors.
    Note that the graph has two kinds of nodes: operations and products.

    Operations represent the nodes in an onnx model, while products represent the tensors.
    The connected graph holds one op per node and one product per value, so the links of ops and
    products, read from the model, always return the same objects.
    """

    def __init__(self, model: onnx_ir.Model, include_if_branches: bool = False):
        """
        :param: model: ONNX model to create connected graph from. Ops and products refer to the
            nodes and values of ``model``, so a caller that keeps ``model`` shares them
        :param: include_if_branches: Whether to also represent the ops inside If branches

            x -> Relu -> h
            cond -> If -> y       then_branch: h -> MatMul -> t_o
                                  else_branch: h -> Identity -> e_o

            False: ops {Relu, If},                   If.subgraphs == {}
            True:  ops {Relu, If, MatMul, Identity}, If.subgraphs ==
                   {"then_branch": [MatMul], "else_branch": [Identity]}
        """
        super().__init__()
        self.model = model
        self._include_if_branches = include_if_branches
        self._op_index: Dict[onnx_ir.Node, int] = {}
        self.fill_op_product_graph()
        self.starting_ops = list(self._get_starting_ops())
        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = self._get_ordered_ops()
        self._assert_no_conflicting_shared_parameters()
        self._warn_on_conflicting_shared_bias_initializers()

    def _reset_model(self, model: onnx_ir.Model):
        """
        Point the static tensors of the connected graph's model at the same-named tensors of
        ``model``.

        Ops and products keep representing the graph they were built from, e.g. the sim's graph
        before quantizers were inserted, but read the tensors of ``model``, e.g. once the sim
        has folded its parameter quantizers. Once no tensor of the previous ``model`` is left,
        the ModelProto it was read from can be released.

        :param model: Model defining the tensors of the connected graph's model
        """
        _rebind_static_tensors(self.model.graph, model.graph)

    def op_of(self, node: Optional[onnx_ir.Node]) -> Optional[Op]:
        """
        Op of ``node``, or None if the connected graph does not represent it, e.g. a Constant
        """
        return self._ops.get(node.name) if node is not None else None

    def product_of(self, value: onnx_ir.Value) -> Product:
        """
        Product of ``value``. A value added to the model after construction gets one on first use.
        """
        product = self._products.get(value.name)
        if product is None:
            product = self._products[value.name] = Product(value, self)
        return product

    def consumers_of(self, value: onnx_ir.Value) -> List[Op]:
        """
        Ops reading ``value``, once per input it is passed to, in the order they were created
        """
        ops = [
            self._ops[use.node.name]
            for use in value.uses()
            if use.node.name in self._ops
        ]
        return sorted(ops, key=lambda op: self._op_index[op.get_module()])

    def subgraphs_of(self, node: onnx_ir.Node) -> Dict[str, List[Op]]:
        """
        Ops of each body of ``node`` the connected graph represents, keyed by the attribute
        carrying it
        """
        if node.op_type != "Scan" and not (
            self._include_if_branches and node.op_type == "If"
        ):
            return {}
        return {
            attr_name: [self._ops[n.name] for n in body if n.name in self._ops]
            for attr_name, body in _subgraphs_of(node)
        }

    def inputs(self) -> List[Product]:
        return [self.product_of(value) for value in self.model.graph.inputs]

    def outputs(self) -> List[Product]:
        return [self.product_of(value) for value in self.model.graph.outputs]

    def get_op_from_module_name(self, name: str) -> Op:
        """
        Gets CG op given the module name
        :param name: Name of the module
        """
        return self._ops[name]

    def _assert_no_conflicting_shared_parameters(self):
        """
        Checks for shared parameters with conflicting consumer types.

        Example: Shared LM head

            W -+-> Gather (required per-tensor)
               └-> Gemm   (requires per-channel)
        """
        all_parameters = [
            param
            for op in self._ops.values()
            for param, param_type in op.parameters.values()
            if param_type == "weight"
        ]
        consumers: dict[Product, list[Op]] = {}
        conflicting_shared_parameters = []

        for param in all_parameters:
            queue = deque(param.consumers)

            while queue:
                consumer = queue.popleft()
                if consumer.type == "Shape":
                    continue
                if consumer.type in ("Transpose", "Identity"):
                    queue.extend(consumer.outputs[0].consumers)
                else:
                    consumers.setdefault(param, []).append(consumer)

            consumer_types = set(consumer.type for consumer in consumers[param])

            if len(consumer_types) > 1 and consumer_types.intersection(
                {"Conv", "ConvTranspose", "MatMul", "Gemm"}
            ):
                conflicting_shared_parameters.append(param)

        if not conflicting_shared_parameters:
            return

        msg = [
            "Found shared parameter(s) with conflicting consumer types:\n",
        ]

        for param in conflicting_shared_parameters:
            msg.append(f"  - input name: {param.name}")

            for i, consumer in enumerate(consumers[param]):
                msg.append(f"    - consumer {i}: {consumer.name} ({consumer.type})")

        msg.append(
            "\nPlease call ``aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)``"
            " before creating QuantizationSimModel"
            " to ensure each consumer takes a unique copy of the initializer as input."
        )
        raise RuntimeError("\n".join(msg))

    def _warn_on_conflicting_shared_bias_initializers(self):
        """
        Warns when a bias initializer is consumed by more than one node, which prevents
        computation of correct bias scales.
        """
        bias_to_consumers: dict[Product, list[Op]] = {}
        for op in self._ops.values():
            if op.type not in ("Gemm", "Conv", "ConvTranspose"):
                continue
            for product, param_type in op.parameters.values():
                if param_type == "bias":
                    bias_to_consumers.setdefault(product, []).append(op)

        multi_consumer_bias = [
            bias for bias, consumers in bias_to_consumers.items() if len(consumers) > 1
        ]

        if multi_consumer_bias:
            logger.warning(
                "Bias tensors %s are consumed by multiple nodes. This will prevent the computation"
                " of correct bias scales. Please call ``aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)``"
                " prior to QuantizationSimModel creation to allow exporting bias scales.",
                multi_consumer_bias,
            )

    def _get_starting_ops(self):
        """
        Yield the ops the model starts at.
        """
        for op in self._top_level_ops():
            if not op.input_ops:
                yield op

    def fill_op_product_graph(self):
        """
        - Creates a product for all tensors (model inputs, constants/initializers, node outputs) in the onnx graph
        - Creates an op for all nodes in the onnx graph
        - Identifies which products should be considered parameters

        Ops and products read their links from the nodes and values they represent.
        """
        graphs = list(
            _iterate_graphs_recursive_ir(
                self.model.graph, include_if=self._include_if_branches
            )
        )

        for graph in graphs:
            values = [
                *graph.initializers.values(),
                *graph.inputs,
                *(output for node in graph for output in node.outputs),
            ]
            for value in values:
                self.product_of(value)

        for graph in graphs:
            for node in graph:
                if node.op_type == "Constant":
                    continue

                for inp in node.inputs:
                    if inp is not None and inp.name not in self._products:
                        raise RuntimeError(
                            f"Input tensor {inp.name} to node {node.name} was not found as a graph input, "
                            "initializer, or as the output of another node. Please verify that the input "
                            "model is properly defined."
                        )

                op = Op(node, self)
                self._op_index[node] = len(self._ops)
                self._ops[node.name] = op

    def _top_level_ops(self) -> List[Op]:
        """
        Yield the ops of the model's own graph, in the order it declares them.

        An op belongs to a body if some op holds it there, so the rest are top level.

        :return: The ops the model's own graph declares
        """
        nested = {body_op for op in self._ops.values() for body_op in op.subgraph_ops}
        return [op for op in self._ops.values() if op not in nested]

    def _ordered_top_level_ops(self) -> List[Op]:
        """
        Traverse the model's own graph, ignoring the ops that make up any body.

        :return: Its ops, each after the ops in it that produce what it reads
        """
        owner_of = {
            body_op: op for op in self._ops.values() for body_op in op.subgraph_ops
        }

        def top_level_owner(op: Op) -> Op:
            while op in owner_of:
                op = owner_of[op]
            return op

        def consumers_of(op: Op) -> List[Op]:
            # A body reads outer tensors by name, so the op running it consumes them as well
            return [
                owner
                for owner in map(top_level_owner, op.output_ops)
                if owner is not op
            ]

        return get_ordered_ops(self.starting_ops, consumers_of=consumers_of)

    def _get_ordered_ops(self) -> List[Op]:
        """
        Order every op in the model, bodies inlined ahead of the ops that run them.

        NOTE: subgraph_ops are already ordered.

        :return: Every op, each after the ops it depends on
        """
        ordered = []
        emitted = set()

        def emit(ops: List[Op]):
            for op in ops:
                if op in emitted:
                    continue
                emitted.add(op)
                emit(op.subgraph_ops)
                ordered.append(op)

        emit(self._ordered_top_level_ops())
        return ordered


def _get_matmul_add_bias_idx(cg_op: Op) -> Optional[int]:
    """
    Identifies the bias input index in an Add node that directly follows a MatMul.

    :param cg_op: The MatMul op to analyse
    :return: The index of the bias input in the Add node, or None if not found
    """
    if cg_op.type != "MatMul":
        return None

    # Dynamic MatMul does not get fused with Add
    if not cg_op.parameters:
        return None

    # Ensure MatMul has exactly one consumer
    consumers = cg_op.outputs[0].consumers
    if len(consumers) != 1:
        return None

    add_op = consumers[0]
    if add_op.type != "Add":
        return None

    for inp1, inp2 in itertools.permutations(add_op.inputs):
        # Ensure inp1 is the output of this MatMul (cg_op)
        if inp1 not in cg_op.outputs:
            continue

        if len(inp1.consumers) > 1:
            return None

        # The bias must be static itself, not read through an Identity: an Identity output is
        # not quantized, so the pass would find no bias quantizer to configure
        param = static_tensor(inp2.value)
        # TODO: Refine this check. Checks that param is static tensor with rank 1
        if param is not None and len(param.shape) == 1:
            return add_op.inputs.index(inp2)

    return None


def _rebind_static_tensors(graph: onnx_ir.Graph, source: onnx_ir.Graph):
    """
    Point the initializers and tensor attributes of ``graph``, and of the bodies it holds, at the
    same-named ones of ``source``. Nodes are matched by their first output, which quantizer
    insertion leaves in place.

    Every tensor is rebound, not only parameters, since any tensor read from a ModelProto keeps
    all of it alive.
    """
    for name, value in graph.initializers.items():
        if name in source.initializers:
            value.const_value = source.initializers[name].const_value

    source_nodes = {node.outputs[0].name: node for node in source if node.outputs}
    for node in graph:
        source_node = source_nodes.get(node.outputs[0].name) if node.outputs else None
        if source_node is None:
            continue
        for name, attr in list(node.attributes.items()):
            source_attr = source_node.attributes.get(name)
            if source_attr is None or source_attr.type != attr.type:
                continue
            if attr.type == onnx_ir.AttributeType.GRAPH:
                _rebind_static_tensors(attr.as_graph(), source_attr.as_graph())
            elif attr.type == onnx_ir.AttributeType.GRAPHS:
                for body, source_body in zip(attr.as_graphs(), source_attr.as_graphs()):
                    _rebind_static_tensors(body, source_body)
            elif attr.type in _TENSOR_ATTRIBUTE_TYPES:
                node.attributes[name] = source_attr


_TENSOR_ATTRIBUTE_TYPES = (
    onnx_ir.AttributeType.TENSOR,
    onnx_ir.AttributeType.TENSORS,
    onnx_ir.AttributeType.SPARSE_TENSOR,
    onnx_ir.AttributeType.SPARSE_TENSORS,
)
