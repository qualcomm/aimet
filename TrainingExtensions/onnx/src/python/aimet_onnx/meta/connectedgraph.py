
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
#  =============================================================================

"""For constructing a uniform representation of the computational graph for a PyTorch model,
that is easy to navigate and stores information for the purpose of AIMET features.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a module or a function that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""

from typing import List, Union, Dict
from onnxruntime.quantization.onnx_quantizer import ONNXModel
import onnx
from packaging import version  # pylint: disable=wrong-import-order

from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph, get_ordered_ops
from aimet_common.utils import AimetLogger
from aimet_common.model_module import ONNXModelModule
from aimet_onnx.meta.operations import Op
from aimet_onnx.meta.product import Product
from aimet_onnx.utils import ParamUtils, retrieve_constant_input

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto, NodeProto, TensorProto
else:
    from onnx.onnx_pb import ModelProto, NodeProto, TensorProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

INPUT_INDEX = 0
WEIGHT_INDEX = 1
BIAS_INDEX = 2
RECURRENT_WEIGHT_INDEX = 2
RUNNING_MEAN_INDEX = 3
RUNNING_VAR_INDEX = 4
OPS_WITH_PARAMS = ["Conv", "Gemm", "ConvTranspose", "BatchNormalization", "MatMul", "RNN", "LSTM", "GRU"]
CONSTANT_TYPE = ['Constant', 'ConstantOfShape']


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together (
    either module or functional) as producers and consumers of tensors.
    Note that the graph has two kinds of nodes: operations and products."""

    def __init__(self, model: ModelProto):
        """
        :param: model: ONNX model to create connected graph from
        """
        super().__init__()
        self.model = model
        if isinstance(self.model, ONNXModel):
            self.model = self.model.model

        # Maps output to consumer node
        self._input_to_node = {}
        self._get_input_to_node()
        self._unnamed_op = 0

        self.starting_ops = []
        self._branch_count = 0

        # Counts number of constant inputs there are in the graph
        self._constant_count = 0
        self._constant_nodes_to_output = self._create_set_of_constant_nodes()

        self.fill_op_product_graph()
        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = get_ordered_ops(self.starting_ops)

    def _create_set_of_constant_nodes(self) -> Dict:
        constant_nodes = {}
        for node in self.model.graph.node:
            if node.op_type in CONSTANT_TYPE:
                for output in node.output:
                    if node.name in constant_nodes:
                        constant_nodes[node.name].append(output)
                    else:
                        constant_nodes[node.name] = [output]
        return constant_nodes

    def get_op_from_module_name(self, name: str) -> Op:
        """
        Gets CG op given the module name
        :param name: Name of the module
        """
        return self._ops[name]

    def _get_input_to_node(self):
        """
        Maps input names to nodes
        """
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name in self._input_to_node:
                    self._input_to_node[input_name].append(node)
                else:
                    self._input_to_node[input_name] = [node]

    # pylint: disable=too-many-branches
    def _get_input_ops(self) -> List:
        """ Gets list of names of starting nodes"""

        def check_if_node_has_predecessor(node):
            for input_name in node.input:
                if input_name in output_names:
                    return True
            return False

        output_names = {}
        input_tensors_names = []
        for node in self.model.graph.node:
            for node_output in node.output:
                output_names[node_output] = node

        # Capture constant tensors associated to a node that has only contant tensor inputs and are not in the form of a constant node.
        for node in self.model.graph.node:
            if node.op_type != 'Identity' and node.op_type not in OPS_WITH_PARAMS and not check_if_node_has_predecessor(node):
                for input_name in node.input:
                    if input_name not in output_names:
                        input_tensors_names.append(input_name)

        # Capture model input tensors.
        for tensor in self.model.graph.input:
            if tensor.name not in input_tensors_names and tensor.name in self._input_to_node:
                input_tensors_names.append(tensor.name)

        # Capture nodes having all the inputs as constant tensors and these constants are coming from a constant node.
        input_ops = []
        for node in self.model.graph.node:
            flag = True
            if node.op_type == 'Constant':
                continue
            for input_name in node.input:
                if input_name in output_names and output_names[input_name].op_type == 'Constant':
                    continue
                flag = False
                break
            if flag and node not in input_ops:
                input_ops.append(node)

        for input_tensor_name in input_tensors_names:
            if input_tensor_name in self._input_to_node:
                for node in self._input_to_node[input_tensor_name]:
                    if node not in input_ops:
                        input_ops.append(node)

        return input_ops

    @staticmethod
    def _create_ir_op(node: NodeProto) -> Op:
        """
        Creates connected graphs internal representation Op
        :param node: ONNX proto node for which Op needs to be created
        """
        op = Op(name=node.name, dotted_name=node.name, output_shape=None, is_anonymous=False, op_type=node.op_type)
        # Add corresponding node to op
        op.model_module = ONNXModelModule(node)

        if op.type in ['Conv', 'ConvTranspose']:
            op.groups = get_op_attributes(node, 'group')

        if op.type == 'MatMul':
            op.transposed_params = False

        if op.type == 'Gemm':
            op.transposed_params = bool(get_op_attributes(node, 'transB'))

        return op

    def _add_children_ops_to_op_queue(self, node: NodeProto, op_queue: List):
        """
        Utility function for adding all children of op to self._op_queue
        :param node: node whose children will be added to op_queue
        :param op_queue: Queue for performing dfs
        """
        for output_tensor in node.output:
            if output_tensor in self._input_to_node:
                for child_node in self._input_to_node[output_tensor]:
                    op_queue.append((child_node, node, output_tensor))

    def _process_starting_ops(self, op_queue: List):
        """
        Processes input ops
        :param op_queue: Queue for performing dfs
        """
        input_ops = self._get_input_ops()
        for node in input_ops:
            if not node.name:
                node.name = str(node.op_type) + '_unnamed_' + str(self._unnamed_op)
                self._unnamed_op += 1
            node_name = node.name
            if node_name not in self._ops:
                op = self._create_ir_op(node)
                self._ops[node_name] = op
                self._add_children_ops_to_op_queue(node, op_queue)
                self.starting_ops.append(op)
            for index, input_tensor_name in enumerate(node.input):
                if self.check_if_param(node, index):
                    continue
                if self.check_if_const(input_tensor_name):
                    op = self._ops[node.name]
                    self._create_constant_product(op, input_tensor_name)
                else:
                    self._create_and_link_product_for_inputs(node_name, input_tensor_name)

    @staticmethod
    def check_if_param(node: NodeProto, index: int) -> bool:
        """
        Checks if given tensor is a param

        :param node: ONNX node
        :param index: input index we are looking at
        """
        if node.op_type in ['Gemm', 'Conv', 'ConvTranspose'] and index in [WEIGHT_INDEX, BIAS_INDEX]:
            return True
        if node.op_type == 'MatMul' and index == WEIGHT_INDEX:
            return True
        if node.op_type == 'BatchNormalization' and index in [WEIGHT_INDEX, BIAS_INDEX, RUNNING_VAR_INDEX, RUNNING_MEAN_INDEX]:
            return True
        if node.op_type in ['RNN', 'LSTM', 'GRU'] and index != INPUT_INDEX:
            return True

        return False

    def check_if_const(self, input_tensor_name: str) -> bool:
        """
        Checks if given tensor is a constant

        :param input_tensor_name: input tensor name we are looking at
        """
        for output_names in self._constant_nodes_to_output.values():
            if input_tensor_name in output_names:
                return True
        return False

    def _create_and_link_product_for_inputs(self, consumer_node_name: str, input_tensor_name: str):
        """
        Create products between input and op consuming the input
        """

        assert input_tensor_name, "No inputs present in the model"

        if input_tensor_name + '_to_' + consumer_node_name in self._products:
            logger.debug("%s already exists", input_tensor_name + '_to_' + consumer_node_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(input_tensor_name + '_to_' + consumer_node_name, None)
            # add product to self._products dictionary
            self._products[input_tensor_name + '_to_' + consumer_node_name] = product
            logger.debug("Created new product " + input_tensor_name + '_to_' + consumer_node_name)
            product.is_model_input = True

            consumer_op = self._ops[consumer_node_name]
            product.tensor_dict[consumer_op] = input_tensor_name

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            consumer_op.add_input(product)
            product.add_consumer(consumer_op)

    def _create_op_if_not_exists(self, node: NodeProto):
        """ Creates a CG op for a node"""
        if node.name not in self._ops:
            op = self._create_ir_op(node)
            self._ops[node.name] = op
            logger.debug("Created new op: %s ", node.name)
        else:
            logger.debug("Op %s already exists", node.name)

    def _create_and_link_product_if_not_exists(self, child_node: NodeProto, parent_node: NodeProto,
                                               connecting_tensor_name: str):
        """ Create and link new product if it does not yet exist """
        parent_module_name = parent_node.name
        child_node_name = child_node.name
        if parent_module_name + '_to_' + child_node_name in self._products:
            logger.debug("%s already exists", parent_module_name + '_to_' + child_node_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(parent_module_name + '_to_' + child_node_name, None)
            # add product to self._products dictionary
            self._products[parent_module_name + '_to_' + child_node_name] = product
            logger.debug("Created new product " + parent_module_name + '_to_' + child_node_name)

            current_op = self._ops[child_node_name]
            parent_op = self._ops[parent_module_name]

            # find Tensor that corresponds to this product
            if connecting_tensor_name is None:
                logger.error("Could not find corresponding tensor between %s and %s",
                             parent_module_name,
                             child_node_name)
                assert False
            product.tensor_dict[current_op] = connecting_tensor_name

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            current_op.add_input(product)
            product.producer = parent_op
            product.add_consumer(current_op)
            parent_op.output = product

    def _create_link_for_output_product(self, output_tensor_name: str, producer_node_name: str):
        """ Creates link between nodes and outputs of the model """
        product_name = producer_node_name + '_to_' + output_tensor_name
        if product_name in self._products:
            logger.debug("%s already exists", product_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(product_name, None)
            # add product to self._products dictionary
            self._products[product_name] = product
            logger.debug("Created new product %s", product_name)

            producer_op = self._ops[producer_node_name]
            product.tensor_dict[producer_node_name] = output_tensor_name

            # Link producer op, product, and current tensor
            producer_op.output = product
            product.producer = producer_op

    def _create_output_products(self):
        """ Create products between last node and output """
        for output in self.model.graph.output:
            for node in self.model.graph.node:
                if output.name in node.output:
                    self._create_link_for_output_product(output.name, node.name)

    def fill_op_product_graph(self):
        """
        - DFS over the graph beginning with input op given as start_op_name
        - Creates op/product graph
        """
        visited_ops = set()
        op_queue = []
        self._process_starting_ops(op_queue)
        # op_queue is treated as a stack, containing operations to traverse. Elements are tuple
        # - Index 0 contains the node to visit.
        # - Index 1 contains the parent node.
        while op_queue:
            child_node, parent_node, connecting_tensor_name = op_queue.pop()
            if not child_node.name:
                child_node.name = str(child_node.op_type) + '_unnamed_' + str(self._unnamed_op)
                self._unnamed_op += 1
            # new module, create op/product and link to parent
            if child_node.name != parent_node.name:
                self._create_op_if_not_exists(child_node)
                self._create_and_link_product_if_not_exists(child_node, parent_node, connecting_tensor_name)

                # add children to op_queue if not visited
                if child_node.name not in visited_ops:
                    self._add_children_ops_to_op_queue(child_node, op_queue)
                    visited_ops.add(child_node.name)
                    logger.debug("visited op: %s", child_node.name)

        # Add output products
        self._create_output_products()

        # Add parameter products during postprocess
        logger.debug("finished initial pass, num_products is %s", len(self._products))
        self._create_param_products()

        # Identify places where branch Ops need to be inserted
        self._branch_ops_processing()

        # Create constant products
        self._create_constant_products_for_model()

    def _create_constant_products_for_model(self):
        """ Create constant products for all ops in the model """
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                for output in node.output:
                    if output in self._input_to_node:
                        for op in self._input_to_node[output]:
                            op_name = op.name
                            cg_op = self._ops[op_name]
                            self._create_constant_product(cg_op, output)

    def _create_constant_product(self, consumer, connecting_tensor_name):
        """
        Create constant product

        :param consumer: Consumer of the product
        :param connecting_tensor_name: tensor that connects consumer and constant op
        """
        product_name = f'constant_{connecting_tensor_name}' + '_to_' + consumer.name
        if product_name not in self._products:
            product = Product(product_name, None)
            # add product to self._products dictionary
            self._products[product_name] = product
            logger.debug("Created new product %s", product_name)

            product.tensor_dict[consumer] = product_name

            product.is_const = True

            self._constant_count += 1

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            consumer.add_input(product)
            product.add_consumer(consumer)

    def _branch_ops_processing(self):
        """ Identify places in the op/product graph where branch ops need to be inserted, and create them """

        # Dictionary that will map producers (ops) to products
        # Ops that map to multiple products will be ones to create branch ops for
        product_producer_dict = dict()
        for product in self._products.values():
            if product.producer is not None:
                if product.producer not in product_producer_dict:
                    product_producer_dict[product.producer] = [product]
                else:
                    product_producer_dict[product.producer].append(product)

        for op in product_producer_dict:
            if len(product_producer_dict[op]) > 1:
                # check if this has an input node
                is_model_input = False
                for product in product_producer_dict[op]:
                    if product.is_model_input:
                        is_model_input = True
                # Create branch op directly under op
                branch_op = self._create_branch_op()
                # Create product to link op and branch_op
                self._link_previous_op_to_branch_op(op, branch_op)
                # Create product to link branch op with multiple children modules
                self._link_branch_op_to_multiple_ops(branch_op, product_producer_dict[op], is_model_input)

    def _create_branch_op(self):
        """ Create a new branch op in self._ops """

        op = Op(name='branch_' + str(self._branch_count), output_shape=None,
                dotted_name='branch_' + str(self._branch_count),
                is_anonymous=True,
                op_type='branch')
        self._ops[op.name] = op
        self._branch_count += 1
        return op

    def _link_previous_op_to_branch_op(self, original_op: Op, branch_op: Op):
        """ Link the original op to the new branch op by creating a product """

        product = Product(original_op.name + '_to_' + branch_op.name, None)
        product.producer = original_op
        product.add_consumer(branch_op)
        original_op.output = product
        branch_op.add_input(product)
        self._products[product.name] = product

    def _link_branch_op_to_multiple_ops(self, branch_op: Op, product_list: list,
                                        is_model_input: bool = False):
        """ Create new product with multiple consumers, linking branch op with children ops"""

        branch_op_product = Product(branch_op.name + '_to_' + 'multiple_ops', branch_op.output_shape)
        branch_op_product.is_model_input = is_model_input
        branch_op_product.producer = branch_op
        branch_op.output = branch_op_product

        # For each product from original op to multiple children, we must:
        # 1: Add each child op as a consumer of the new branch_op_product
        # 2: Append the old product's corresponding Tensor to the new branch op product's tensor_list
        # 3: Add new branch_op_product as input for each child op
        # 4: Remove product from original op to child in child's inputs
        # 5: Remove product from self._products
        for product in product_list:
            assert len(product.consumers) <= 1

            if len(product.consumers) == 1:
                # item 1
                branch_op_product.add_consumer(product.consumers[0])
                # item 2
                assert len(product.tensor_dict.keys()) == 1
                branch_op_product.tensor_dict[product.consumers[0]] = product.tensor_dict[product.consumers[0]]
                # items 3 and 4
                # replace the old product with the new branch product, in the same index as the old product
                index = product.consumers[0].inputs.index(product)
                product.consumers[0].inputs[index] = branch_op_product
                # item 5
                del self._products[product.name]
            else:
                for output in self.model.graph.output:
                    if product.name.startswith(output.name + '_to_') or \
                        product.name.endswith('_to_' + output.name) or \
                        product.name == output.name:
                        del self._products[product.name]

        self._products[branch_op_product.name] = branch_op_product

    def _create_param_products(self):
        """ Create products for parameters of select modules """

        def create_and_connect_product(param_name: str, product_shape: List, my_op: Op,
                                       param_tensor: TensorProto, product_type: Union[str, None]):
            """ Create product with given name, shape, and corresponding tensor.  Connect product to my_op. """

            product = Product(param_name, product_shape)
            product.is_parm = True
            product.add_consumer(my_op)
            product.tensor_dict[my_op] = param_tensor
            product.tensor = param_tensor
            my_op.add_input(product)
            self._products[product.name] = product
            my_op.add_param(param_name, product, product_type)

        def create_weight_bias_params(my_op: Op):
            """ Create products for conv2d, dense, depthwise conv2d, and similar """
            op = my_op.get_module()

            weight_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if weight_tensor:
                create_and_connect_product(weight_tensor.name, weight_tensor.dims, my_op, weight_tensor, 'weight')

            bias_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            if bias_tensor:
                create_and_connect_product(bias_tensor.name, bias_tensor.dims, my_op, bias_tensor, 'bias')

        def create_matmul_params(my_op: Op):
            """
            Create products for MatMul layer

            :param my_op: Connected Graph Op
            """
            op = my_op.get_module()
            weight_tensor, _ = retrieve_constant_input(op, self.model, WEIGHT_INDEX)
            if weight_tensor:
                create_and_connect_product(weight_tensor.name, weight_tensor.dims, my_op, weight_tensor, 'weight')

        def create_recurrent_type_params(my_op: Op):
            """
            Create products for RNN, LSTM and GRU layer

            :param my_op: Connected Graph Op
            """
            op = my_op.get_module()
            weight_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if weight_tensor:
                create_and_connect_product(weight_tensor.name, weight_tensor.dims, my_op, weight_tensor, 'weight_x')

            recurrent_weight_tensor = ParamUtils.get_param(self.model, op, RECURRENT_WEIGHT_INDEX)
            if recurrent_weight_tensor:
                create_and_connect_product(recurrent_weight_tensor.name, recurrent_weight_tensor.dims, my_op, recurrent_weight_tensor, 'weight_r')

        def create_batchnorm_params(my_op: Op):
            """ Create products for fusedbatchnorm """
            op = my_op.get_module()

            gamma_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if gamma_tensor:
                create_and_connect_product(gamma_tensor.name, gamma_tensor.dims, my_op, gamma_tensor, 'weight')

            beta_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            if beta_tensor:
                create_and_connect_product(beta_tensor.name, beta_tensor.dims, my_op, beta_tensor, 'bias')

            moving_mean_tensor = ParamUtils.get_param(self.model, op, RUNNING_MEAN_INDEX)
            if moving_mean_tensor:
                create_and_connect_product(moving_mean_tensor.name, moving_mean_tensor.dims, my_op, moving_mean_tensor, "running_mean")

            moving_variance_tensor = ParamUtils.get_param(self.model, op, RUNNING_VAR_INDEX)
            if moving_variance_tensor:
                create_and_connect_product(moving_variance_tensor.name, moving_variance_tensor.dims, my_op, moving_variance_tensor, "running_var")

        def handle_default(my_op: Op):
            """ Handler for other modules """
            logger.debug("Nothing to handle for op %s", my_op.name)

        switcher = {
            "Conv": create_weight_bias_params,
            "Gemm": create_weight_bias_params,
            "ConvTranspose": create_weight_bias_params,
            "RNN": create_recurrent_type_params,
            "LSTM": create_recurrent_type_params,
            "GRU": create_recurrent_type_params,
            "BatchNormalization": create_batchnorm_params,
            "InstanceNormalization": create_weight_bias_params,
            "LayerNormalization": create_weight_bias_params,
            "GroupNormalization": create_weight_bias_params,
            "MatMul": create_matmul_params,
        }

        for op in self._ops.values():
            handler = switcher.get(op.type, handle_default)
            handler(op)


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
