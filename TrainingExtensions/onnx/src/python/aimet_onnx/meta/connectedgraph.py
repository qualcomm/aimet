
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

"""For constructing a uniform representation of the computational graph for an ONNX model,
that is easy to navigate and stores information for the purpose of AIMET features.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a node that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""

from typing import Union
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
        self.ordered_ops = get_ordered_ops(self.starting_ops)

    def get_op_from_module_name(self, name: str) -> Op:
        """
        Gets CG op given the module name
        :param name: Name of the module
        """
        return self._ops[name]

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

    def _get_starting_ops(self):
        for op in self._ops.values():
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

    def _create_product_for_activations(self, producer: onnx.NodeProto, tensor_name: str):
        if producer.op_type == "Constant":
            return self._create_constant_product(tensor_name, producer.attribute[0].t.dims)
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

        # Add products for all tensors in initializer
        for tensor in self.model.graph.initializer:
            self._products[tensor.name] = self._create_constant_product(tensor.name, tensor.dims)

        # Add products for all model inputs
        for input_info in self.model.graph.input:
            self._products[input_info.name] = self._create_product_for_inputs(input_info)

        # Create products for all intermediate tensors
        for node in self.model.graph.node:
            for output in node.output:
                self._products[output] = self._create_product_for_activations(node, output)

        # Create ops and link with products
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                continue

            op = self._create_ir_op(node)
            self._ops[node.name] = op
            for inp in node.input:
                if not inp:
                    continue # Empty string indicates omitted optional input
                if inp not in self._products:
                    raise RuntimeError(f"Input tensor {inp} to node {node.name} was not found as a graph input, "
                                       "initializer, or as the output of another node. Please verify that the input "
                                       "model is properly defined.")
                product = self._products[inp]
                op.add_input(product)
                product.add_consumer(op)
                product.tensor_dict[op] = inp # TODO: Delete Product.tensor_dict attribute

            for idx, output in enumerate(node.output):
                product = self._products[output]
                # TODO: Make Op.output list of products to support multi-output ops
                if idx == 0: # For backwards compatibility, op.output is always first product
                    op.output = product
                product.producer = op

        # TODO: Move this process outside of ConnectedGraph altogether
        self._identify_param_products()

    def _identify_param_products(self):
        """ Identify products which are parameters of select modules """

        def set_as_param(param_tensor: TensorProto, my_op: Op, product_type: Union[str, None]):
            """ Create product with given name, shape, and corresponding tensor.  Connect product to my_op. """
            param_name = param_tensor.name
            product_shape = param_tensor.dims
            product = self._products[param_name]
            product.shape = product_shape
            product.is_parm = True
            my_op.add_param(param_name, product, product_type)
            # TODO: Delete Product.tensor_dict, Product.tensor attributes
            product.tensor_dict[my_op] = param_tensor
            product.tensor = param_tensor
            product.is_const = False # Backward compatibility

        def create_weight_bias_params(my_op: Op):
            """ Create products for conv2d, dense, depthwise conv2d, and similar """
            op = my_op.get_module()

            weight_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if weight_tensor:
                set_as_param(weight_tensor, my_op, 'weight')

            bias_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            if bias_tensor:
                set_as_param(bias_tensor, my_op, 'bias')

        def create_matmul_params(my_op: Op):
            """
            Create products for MatMul layer

            :param my_op: Connected Graph Op
            """
            op = my_op.get_module()
            weight_tensor, _ = retrieve_constant_input(op, self.model, WEIGHT_INDEX)
            if weight_tensor:
                set_as_param(weight_tensor, my_op, 'weight')

        def create_recurrent_type_params(my_op: Op):
            """
            Create products for RNN, LSTM and GRU layer

            :param my_op: Connected Graph Op
            """
            op = my_op.get_module()
            weight_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if weight_tensor:
                set_as_param(weight_tensor, my_op, 'weight_x')

            recurrent_weight_tensor = ParamUtils.get_param(self.model, op, RECURRENT_WEIGHT_INDEX)
            if recurrent_weight_tensor:
                set_as_param(recurrent_weight_tensor, my_op, 'weight_r')

        def create_batchnorm_params(my_op: Op):
            """ Create products for fusedbatchnorm """
            op = my_op.get_module()

            gamma_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            if gamma_tensor:
                set_as_param(gamma_tensor, my_op, 'weight')

            beta_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            if beta_tensor:
                set_as_param(beta_tensor, my_op, 'bias')

            moving_mean_tensor = ParamUtils.get_param(self.model, op, RUNNING_MEAN_INDEX)
            if moving_mean_tensor:
                set_as_param(moving_mean_tensor, my_op, "running_mean")

            moving_variance_tensor = ParamUtils.get_param(self.model, op, RUNNING_VAR_INDEX)
            if moving_variance_tensor:
                set_as_param(moving_variance_tensor, my_op, "running_var")

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
