# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""For constructing a uniform representation of the computational graph for an ONNX model,
that is easy to navigate and stores information for the purpose of AIMET features.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a node that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""

import itertools
from collections import deque
from typing import Dict, List, Optional, Tuple, Union
from onnxruntime.quantization.onnx_quantizer import ONNXModel
import onnx
import onnx_ir
from packaging import version

from aimet_onnx.common.connected_graph.connectedgraph import (
    ConnectedGraph as AimetCommonConnectedGraph,
    get_ordered_ops,
)
from aimet_onnx.common.onnx._utils import _iterate_graphs_recursive
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.model_module import ONNXModelModule
from aimet_onnx.meta.operations import Op
from aimet_onnx.meta.product import Product
from aimet_onnx.utils import ParamUtils
from aimet_onnx.ir_utils import get_constant_value, get_weight_value, _has_transposed_b

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto, NodeProto, TensorProto
else:
    from onnx.onnx_pb import ModelProto, NodeProto, TensorProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

INPUT_INDEX = 0
WEIGHT_INDEX = 1
BIAS_INDEX = 2
RUNNING_MEAN_INDEX = 3
RUNNING_VAR_INDEX = 4
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


def _subgraphs_of(node: NodeProto) -> List[onnx.GraphProto]:
    """
    Return every graph ``node`` carries as an attribute.

    :param node: Node to inspect
    :return: Its subgraphs, outermost attribute order preserved
    """
    subgraphs = []
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            subgraphs.append(attr.g)
        elif attr.type == onnx.AttributeProto.GRAPHS:
            subgraphs.extend(attr.graphs)
    return subgraphs


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together as producers and consumers of tensors.
    Note that the graph has two kinds of nodes: operations and products.

    Operations represent the nodes in an onnx model, while products represent the tensors.
    """

    def __init__(self, model: ModelProto):
        """
        :param: model: ONNX model to create connected graph from
        """
        super().__init__()
        self.model = model
        if isinstance(self.model, ONNXModel):
            self.model = self.model.model
        self.fill_op_product_graph()
        self.starting_ops = list(self._get_starting_ops())
        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = self._get_ordered_ops()
        self._assert_no_conflicting_shared_parameters()
        self._warn_on_conflicting_shared_bias_initializers()

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

    @staticmethod
    def _create_ir_op(node: NodeProto) -> Op:
        """
        Creates connected graphs internal representation Op
        :param node: ONNX proto node for which Op needs to be created
        """
        op = Op(
            name=node.name,
            dotted_name=node.name,
            output_shape=None,
            is_anonymous=False,
            op_type=node.op_type,
            domain=node.domain,
        )
        # Add corresponding node to op
        op.model_module = ONNXModelModule(node)

        if op.type in ["Conv", "ConvTranspose"]:
            op.groups = get_op_attributes(node, "group")

        if op.type == "MatMul":
            op.transposed_params = False

        if op.type == "Gemm":
            op.transposed_params = bool(get_op_attributes(node, "transB"))

        return op

    def _get_starting_ops(self):
        """
        Yield the ops the model starts at.
        """
        for op in self._top_level_ops():
            if not op.input_ops:
                yield op

    @staticmethod
    def _create_product_for_inputs(input_value_info: onnx.ValueInfoProto):
        """
        Create products between input and op consuming the input
        """
        shape = [dim.dim_value for dim in input_value_info.type.tensor_type.shape.dim]
        product = Product(input_value_info.name, shape)
        product.is_const = False
        product.is_model_input = True
        return product

    def _create_product_for_activations(
        self, producer: onnx.NodeProto, tensor_name: str
    ):
        if producer.op_type == "Constant":
            return self._create_constant_product(
                tensor_name, producer.attribute[0].t.dims
            )
        return Product(tensor_name, None)

    @staticmethod
    def _create_constant_product(name, dims):
        """
        Create constant product

        :param consumer: Consumer of the product
        :param connecting_tensor_name: tensor that connects consumer and constant op
        """
        product = Product(name, dims)
        product.is_const = True
        return product

    def fill_op_product_graph(self):
        """
        - Creates a product for all tensors (model inputs, constants/initializers, node outputs) in the onnx graph
        - Creates an op for all nodes in the onnx graph
        - Links products with their producer and consumer ops
        - Identifies which products should be considered parameters
        """
        graphs = list(_iterate_graphs_recursive(self.model.graph))

        for graph in graphs:
            # Add products for all tensors in initializer
            for tensor in graph.initializer:
                self._products[tensor.name] = self._create_constant_product(
                    tensor.name, tensor.dims
                )

            # Add products for all model inputs
            for input_info in graph.input:
                product = self._create_product_for_inputs(input_info)
                # A body's declared inputs are fed by the control-flow op, not by the caller
                # of the model.
                product.is_model_input = graph is self.model.graph
                self._products[input_info.name] = product

            # Create products for all intermediate tensors
            for node in graph.node:
                for output in node.output:
                    self._products[output] = self._create_product_for_activations(
                        node, output
                    )

        # A graph's position - it is used to map ops within it.
        graph_index_of = {id(graph): index for index, graph in enumerate(graphs)}
        # The ops of each graph, in the order the graph declares them
        ops_by_graph_index: Dict[int, List[Op]] = {}
        # Map control-flow op -> corresponding subgraph index
        subgraphs_to_order: List[Tuple[Op, int]] = []

        # Create ops and link with products
        for graph_index, graph in enumerate(graphs):
            for node in graph.node:
                if node.op_type == "Constant":
                    continue

                op = self._create_ir_op(node)
                self._ops[node.name] = op
                ops_by_graph_index.setdefault(graph_index, []).append(op)
                for inp in node.input:
                    if not inp:
                        continue  # Empty string indicates omitted optional input
                    if inp not in self._products:
                        raise RuntimeError(
                            f"Input tensor {inp} to node {node.name} was not found as a graph input, "
                            "initializer, or as the output of another node. Please verify that the input "
                            "model is properly defined."
                        )
                    product = self._products[inp]
                    op.add_input(product)
                    product.add_consumer(op)

                for output in node.output:
                    product = self._products[output]
                    op.outputs.append(product)
                    product.producer = op

                subgraphs_to_order.extend(
                    (op, graph_index_of[id(subgraph)])
                    for subgraph in _subgraphs_of(node)
                    if id(subgraph) in graph_index_of
                )

        # Add subgraph ops for the op containing the body.
        for op, graph_index in subgraphs_to_order:
            for body_op in ops_by_graph_index.get(graph_index, []):
                op.add_subgraph_op(body_op)

        # TODO: Move this process outside of ConnectedGraph altogether
        self._set_param_products()

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
        top_level = set(self._top_level_ops())
        return get_ordered_ops(self.starting_ops, restrict_to=top_level)

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

    def _set_param_products(self):
        """Identify products which are parameters of select modules"""

        def set_as_param(
            param_name: str,
            param_tensor: TensorProto,
            my_op: Op,
            product_type: Union[str, None],
        ):
            """Create product with given name, shape, and corresponding tensor.  Connect product to my_op."""
            product_shape = param_tensor.dims
            product = self._products[param_name]
            product.shape = product_shape
            product.is_parm = True
            my_op.add_param(param_name, product, product_type)
            product.tensor = param_tensor
            product.is_const = False  # Backward compatibility

        ir_model = onnx_ir.from_proto(self.model)
        info_dict = find_all_node_info(ir_model)

        tensor_dict = {}
        for graph in _iterate_graphs_recursive(self.model.graph):
            tensor_dict.update({init.name: init for init in graph.initializer})
            for node in graph.node:
                if node.op_type != "Constant":
                    continue

                for attribute in node.attribute:
                    if attribute.name == "value":
                        param = attribute.t
                        tensor_dict[node.output[0]] = param

        for node_name, op in self._ops.items():
            info = info_dict[node_name]
            for param_type, param_name in info.parameters.items():
                param_tensor = tensor_dict.get(param_name)
                if param_tensor is not None:
                    set_as_param(param_name, param_tensor, op, param_type)

            op.transposed_params = info.transposed_params


def _get_matmul_add_bias_idx(cg_op: Op, model: ModelProto) -> Optional[int]:
    """
    Identifies the bias input index in an Add node that directly follows a MatMul.

    :param cg_op: The MatMul op to analyse
    :param model: The model containing the param metadata
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

        param = ParamUtils.get_param_by_name(model, inp2.name)
        # TODO: Refine this check. Checks that param is static tensor with rank 1
        if param and len(param.dims) == 1:
            return add_op.inputs.index(inp2)

    return None


def get_op_attributes(node: NodeProto, attribute_name: str):
    """
    Gets attribute information for layer

    :param node: ONNX node
    :param attribute_name: The attribute we are searching for
    """
    for attribute in node.attribute:
        if attribute.name == attribute_name:
            return attribute.i
    return None


class NodeInfo:
    """Captures extra metadata stored on CG Op"""

    parameters = {}
    transposed_params = False


def find_all_node_info(model: onnx_ir.Model) -> dict[str, NodeInfo]:
    """
    Extracts all extra metadata for ops held by connectedgraph
    """

    def _find_parameters(node, param_type_to_index: dict[str, int]):
        parameters = {}
        for param_type, index in param_type_to_index.items():
            if index >= len(node.inputs):
                continue
            tensor = get_constant_value(node.inputs[index])
            if tensor:
                parameters[param_type] = tensor.name

        return parameters

    def find_weight_bias_params(node: onnx_ir.Node) -> dict[str, str]:
        param_type_to_index = {
            "weight": WEIGHT_INDEX,
            "bias": BIAS_INDEX,
        }
        return _find_parameters(node, param_type_to_index)

    def find_weight_params(node: onnx_ir.Node) -> dict[str, str]:
        param_type_to_index = {"weight": WEIGHT_INDEX}
        return _find_parameters(node, param_type_to_index)

    def find_matmul_params(node: onnx_ir.Node) -> dict[str, str]:
        parameters = {}
        weight_tensor, _ = get_weight_value(node)
        if weight_tensor:
            parameters["weight"] = weight_tensor.name
        return parameters

    def find_recurrent_type_params(node: onnx_ir.Node) -> dict[str, str]:
        param_type_to_index = {
            "weight": 1,
            "weight_r": 2,
            "bias": 3,
        }
        return _find_parameters(node, param_type_to_index)

    def find_batchnorm_params(node: onnx_ir.Node) -> dict[str, str]:
        param_type_to_index = {
            "weight": WEIGHT_INDEX,
            "bias": BIAS_INDEX,
            "running_mean": RUNNING_MEAN_INDEX,
            "running_var": RUNNING_VAR_INDEX,
        }
        return _find_parameters(node, param_type_to_index)

    switcher = {
        "Conv": find_weight_bias_params,
        "Gemm": find_weight_bias_params,
        "ConvTranspose": find_weight_bias_params,
        "RNN": find_recurrent_type_params,
        "LSTM": find_recurrent_type_params,
        "GRU": find_recurrent_type_params,
        "BatchNormalization": find_batchnorm_params,
        "InstanceNormalization": find_weight_bias_params,
        "LayerNormalization": find_weight_bias_params,
        "GroupNormalization": find_weight_bias_params,
        "RMSNormalization": find_weight_params,
        "MatMul": find_matmul_params,
    }

    info_dict = {}
    for node in model.graph.all_nodes():
        node_info = NodeInfo()
        handler = switcher.get(node.op_type)
        if handler:
            node_info.parameters = handler(node)

        if node.op_type == "Gemm":
            node_info.transposed_params = _has_transposed_b(node)
        elif node.op_type == "MatMul":
            _, transposed = get_weight_value(node)
            node_info.transposed_params = transposed

        info_dict[node.name] = node_info

    return info_dict
