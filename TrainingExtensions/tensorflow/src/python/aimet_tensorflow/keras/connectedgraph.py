# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""
Connected graph class and utilities
"""
import typing
import tensorflow as tf
from keras.layers.core.tf_op_layer import TFOpLambda

from aimet_common.connected_graph.connectedgraph import (
        ConnectedGraph as AimetCommonConnectedGraph,
        )
from aimet_common.connected_graph.operation import (
        Op,
        determine_preceding_op_input_product_index_in_multi_input_op,
        )
from aimet_common.connected_graph.product import Product
from aimet_common.model_module import KerasModelModule
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.utils import common

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

map_keras_types_to_onnx = {
    tf.keras.layers.Add: ["Add"],
    tf.keras.layers.AveragePooling2D: ["AveragePool"],
    tf.keras.layers.BatchNormalization: ["BatchNormalization"],
    tf.keras.layers.Concatenate: ["Concat"],
    tf.keras.layers.Conv1D: ["Conv"],
    tf.keras.layers.Conv2D: ["Conv"],
    tf.keras.layers.Conv2DTranspose: ["ConvTranspose"],
    tf.keras.layers.Dense: ["Gemm", "MatMul"],
    tf.keras.layers.DepthwiseConv2D: ["Conv"],
    tf.keras.layers.Dropout: ["Dropout"],
    tf.keras.layers.ELU: ["Elu"],
    tf.keras.layers.Flatten: ["Flatten"],
    tf.keras.layers.GRU: ["GRU"],
    tf.keras.layers.GlobalAveragePooling1D: ["GlobalAveragePool"],
    tf.keras.layers.GlobalAveragePooling2D: ["GlobalAveragePool"],
    tf.keras.layers.LSTM: ["LSTM"],
    tf.keras.layers.LayerNormalization: ["LayerNormalization"],
    tf.keras.layers.LeakyReLU: ["LeakyRelu"],
    tf.keras.layers.MaxPool2D: ["MaxPool"],
    tf.keras.layers.Multiply: ["Mul"],
    tf.keras.layers.PReLU: ["PRelu"],
    tf.keras.layers.RNN: ["RNN"],
    tf.keras.layers.ReLU: ["Relu"],
    tf.keras.layers.Reshape: ["Reshape"],
    tf.keras.layers.Subtract: ["Sub"],
    tf.keras.layers.UpSampling1D: ["Upsample"],
    tf.keras.layers.UpSampling2D: ["Upsample"],
    tf.keras.layers.ZeroPadding1D: ["Pad"],
    tf.keras.layers.ZeroPadding2D: ["Pad"],
}

# TFOpLambda layers are abstracted. Meaning we have to look at the "symbol" property
# to see what the underlying call is.

map_tf_op_lambda_symbol_to_onnx = {
    "__operators__.add": ["Add"],
    "__operators__.getitem": ["OptionalGetItem"],
    "cast": ["Cast"],
    "compat.v1.gather_nd": ["GatherND"],
    "compat.v1.shape": ["Shape"],
    "compat.v1.transpose": ["Transpose"],
    "concat": ["Concat"],
    "expand_dims": ["Expand"],
    "math.add": ["Add"],
    "math.cumsum": ["CumSum"],
    "math.multiply": ["Mul"],
    "math.subtract": ["Sub"],
    "math.truediv": ["Div"],
    "pad": ["Pad"],
    "range": ["Range"],
    "reshape": ["Reshape"],
    "scatter_nd": ["ScatterND"],
    "tile": ["Tile"],
    "where": ["Where"],
}

class ConnectedGraph(AimetCommonConnectedGraph):
    """
    Connected Graph class
    """

    def __init__(
            self,
            model: tf.keras.Model,
            ):
        """
        If the model object is implemented in a subclassing manner, resulting object is different from
        the original object because this method is converting to Functional manner

        :param model: Keras Model that is built (Sequential, Functional)
        """
        super().__init__()

        self._name_to_layer = {}
        self._op_name_to_layer = {}
        self._layer_to_op = {}
        self.ordered_ops = []
        self._ops_index = 0
        self._split_count = 0

        if not model.built:
            raise RuntimeError("Keras Model should be built before passing it")
        self._model = model

        # Generate Ops by parsing layer information
        self._parse_layers(self._model.layers)

        # Generate Products by parsing layer connection information
        self._parse_layer_connections()
        self._fill_op_params()

        # For each split in the model, insert a corresponding split Op in the connected graph.
        self._find_and_parse_split_ops()

    def _parse_layers(self, layers: typing.List[tf.keras.layers.Layer]):
        """
        Parse layers iteratively to obtain Ops

        :param layers: list of Keras layers
        """
        for layer in layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            self._parse_layer(layer)

    def _parse_layer(self, layer: tf.keras.layers.Layer):
        """
        Parse a layer and convert it to Op with recursive manner if it is nested

        :param layer: Keras layer
        """
        if hasattr(layer, "layers"):
            self._parse_layers(layer.layers)
        else:
            if isinstance(layer, tf.keras.layers.Activation):
                op_type = common.parse_activation_layer(layer)
            elif isinstance(layer, TFOpLambda):
                # We can't use 'type' because the type is just TFOpLambda. To be
                # more granular, we look at the symbol.
                op_type = map_tf_op_lambda_symbol_to_onnx.get(layer.symbol)
                if not op_type:
                    logger.warning(
                        "No matching mapping for TFOpLambda symbol '%s'", layer.symbol
                    )
            else:
                op_type = map_keras_types_to_onnx.get(type(layer))

            if op_type is None:
                op_type = "Unknown"
            else:
                op_type = op_type[0]

            op = self._generate_op(op_type, layer)

            if isinstance(layer.output, typing.List):
                for output in layer.output:
                    self._name_to_layer[output.name] = layer
            else:
                self._name_to_layer[layer.output.name] = layer

            self._op_name_to_layer[op.name] = layer
            self._layer_to_op[layer] = op
            self._ops[op.name] = op
            self.ordered_ops.append(op)
            self._ops_index += 1

    def _generate_op(self, op_type: str, layer: tf.keras.layers.Layer) -> Op:
        """
        Generate operation object using operation type and keras layer

        :param op_type: Operation type compatible with ONNX
        :param layer: Keras layer
        :return: Operation object
        """
        op_name = f"{op_type}_{self._ops_index}"
        dotted_name = f"{self._model.name}.{layer.name}"
        output_shape = layer.output_shape

        op = Op(
            name=op_name,
            dotted_name=dotted_name,
            output_shape=output_shape,
            is_anonymous=False,
            op_type=op_type,
        )
        op.model_module = KerasModelModule(layer)

        if op.type == "Conv" and hasattr(layer, "groups"):
            op.groups = layer.groups

        return op

    def _parse_layer_connections(self):
        """
        Obtain layer connection by parsing layer information and generate product
        """
        for op in self.ordered_ops:
            self._generate_product(op.get_module())

    def _generate_product(self, target_layer: tf.keras.layers.Layer):
        """
        Generate products while traversing the inbound layer of the target layer

        :param target_layer: tf.keras.layer related to consumer Op
        """
        for _, node in enumerate(target_layer.inbound_nodes):
            for inbound_layer, _, _, _ in node.iterate_inbound():
                if isinstance(inbound_layer, tf.keras.layers.InputLayer):
                    self._generate_input_product(target_layer)
                else:
                    self._generate_usual_product(target_layer, inbound_layer)

    def _generate_input_product(self, target_layer: tf.keras.layers.Layer):
        """
        Generate product which is producer is model input

        :param target_layer: tf.keras.layer related to consumer Op
        """
        consumer_op = self._layer_to_op.get(target_layer)
        product_name = f"input_to_{consumer_op.name}"

        if product_name not in self._products:
            product = Product(product_name, None)
            product.is_model_input = True
            product.add_consumer(consumer_op)
            consumer_op.add_input(product)
            self._products[product_name] = product

    def _generate_usual_product(
            self,
            target_layer: tf.keras.layers.Layer,
            inbound_layer: tf.keras.layers.Layer
    ):
        """
        Generate usual product which is producer is not model input

        :param target_layer: tf.keras.layer related to consumer Op
        :param inbound_layer: tf.keras.layer related to producer Op
        """
        consumer_op = self._layer_to_op.get(target_layer)
        inbound_layer_output_name = \
            inbound_layer.output[0].name if isinstance(inbound_layer.output, typing.List) else inbound_layer.output.name
        producer_op = self.get_op_from_module_name(inbound_layer_output_name)

        if producer_op is None:
            raise RuntimeError("Producer Op must exist")

        product_name = f"{producer_op.name}_to_{consumer_op.name}"
        if product_name not in self._products:
            product = Product(product_name, producer_op.output_shape)
            product.producer = producer_op
            self._products[product_name] = product
        else:
            product = self._products[product_name]

        product.add_consumer(consumer_op)
        consumer_op.add_input(product)
        producer_op.outputs = [product]

    def _fill_op_params(self):
        """
        For certain ops like convolution, batch norm, and linear,
        create products for their parameters if they don't exist yet.
        """

        for op in self._ops.values():
            layer = op.get_module()
            layer_name = layer.name

            if not layer.built:
                raise RuntimeError("Layer should be built before executing this method")

            if op.type in ["Conv", "ConvTranspose", "Gemm"]:
                weight_tensors = layer.get_weights()

                self._create_and_add_param_product_if_not_exists(
                    op, f"{layer_name}.weight", list(weight_tensors[0].shape)
                )
                if layer.use_bias:
                    self._create_and_add_param_product_if_not_exists(
                        op, f"{layer_name}.bias", list(weight_tensors[1].shape)
                    )

            if op.type == "BatchNormalization":
                tensor_descriptions = ["weight", "bias", "running_mean", "running_var"]
                weight_tensors = layer.get_weights()

                for weight_tensor, tensor_description in zip(weight_tensors, tensor_descriptions):
                    self._create_and_add_param_product_if_not_exists(
                        op,
                        f"{layer_name}.{tensor_description}",
                        list(weight_tensor.shape),
                    )

    def _create_and_add_param_product_if_not_exists(
            self, op: Op, product_name: str, shape: typing.List[int]
    ):
        """
        Given a name of a product, create it if it doesn't exist, and attach it to the specified op as a parameter.
        :param op: Op to connect the parameter product to.
        :param product_name: Name of the product to create.
        :param shape: Shape of the product to create.
        """
        if product_name not in self._products:
            product = Product(product_name, shape)
            product.is_parm = True
            product.add_consumer(op)
            op.add_input(product)
            self._products[product_name] = product

    def _find_and_parse_split_ops(self):
        """
        Find split ops whose output is used as input in many ops.
        After finding split ops, create and link product as intended
        """
        for op in self.ordered_ops:
            self._determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(op)

    def _determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(self, op: Op):
        """
        Determine if an Op's output is used as an input to more than one Op. If it is, create a Split Op and
        insert it in the connected graph, below this Op.
        Note that the split is done in the forward() function of a model and is NOT a PyTorch OP.

        :param op: Op to check if output is used as an input to more than one op.
        """
        output_product_names = self.get_product_names_from_dotted_name(op.dotted_name)

        if len(output_product_names) > 1:
            logger.debug("%s is a split Op", op.dotted_name)
            # Create a Split Op
            split_op = self._create_split_op(op)

            # Insert the Split Op in the connected graph
            self._insert_split_op_in_connected_graph(op, split_op)

    def _create_split_op(self, op: Op) -> Op:
        """
        The op's output is split in the forward function. To model it correctly, create a Split Op.
        :param op: Op to create split op after
        :return: Split op that was created
        """
        split_name = f"Split_{self._split_count}"
        split_dotted_name = f"{self._model.name}.{split_name}"
        self._split_count += 1

        split_op = Op(
            name=split_name,
            dotted_name=split_dotted_name,
            output_shape=op.output_shape,
            is_anonymous=True,
            op_type="Split",
        )
        self._ops[split_name] = split_op

        return split_op

    def _insert_split_op_in_connected_graph(self, preceding_op: Op, split_op: Op):
        """
        Insert a Split Op below the preceding Op in the connected graph.
        :param preceding_op: Op prior to split op
        :param split_op: Split op to insert
        """

        # Important Notes
        # Op:
        # An Op class represents a module in a model.
        #
        # Product:
        # In this version of the Winnower, the Product class represents the following entities in a model.
        # 1) a Tensor between two modules (in Winnower, 2 Ops).
        # 2) an input
        # 3) a constant
        # 4) a parameter
        #
        # Considering only the definition #1) above, i.e., Product is a Tensor between 2 Ops,
        # an Op's inputs and output are Products.
        # That means, an Op could have multiple input Products and one output Product.
        # Examples of Op with multiple input products: add, cat (Concat)
        # A Product's Producer and Consumer are Ops.
        # A Product could have only one Producer but could have multiple consumers.
        # For example, a Split Op has one output.  The Split Op's single output isa Product.
        # That product's single Producer is the Split Op and multiple consumers are the 2 Ops in the 2 branches of
        # the Split, that receive the Split output.

        # Steps:
        # 1. Create a new Product for Split Op's output.
        # 2. This product has multiple consumers. Add the consumers to the Product.
        #   Get the consumers from the op's multiple products.
        # 3. Set the the current Op's output Product's consumer to Split Op. The output product's name must be changed.
        # 4. Set the Split Op's input to point to current Op's output. Its name must be changed.

        # 1. Create a new Product for Split Op's output.
        split_op_product = self._create_split_op_output_product(preceding_op, split_op)
        split_op.outputs = [split_op_product]

        # 2.This product has multiple consumers. Add the consumers to the Product.
        # Get the consumers from the op's multiple products.
        self._add_consumers_to_split_op_product(preceding_op, split_op_product)

        # 3. Create a new product to connect the preceding Op to the Split Op.
        # Set the the preceding Op's output Product's consumer to Split Op.
        self._create_product_linking_preceding_op_to_split_op(preceding_op, split_op)

        # 4. Set the Split Op's input to point to current Op's output.
        split_op.inputs.append(preceding_op.outputs[0])

    def _create_split_op_output_product(self, preceding_op: Op, split_op: Op) -> Product:
        """
        Create output product of the split op and connected it to the split op
        :param preceding_op: Op prior to split op
        :param split_op: Split op to create output product for
        :return: Output product of the split op
        """
        split_op_product_name = f"{split_op.name}__to__multiple_ops"
        split_op_product_shape = preceding_op.outputs[0].shape
        split_op_product = self._add_product(
            split_op_product_name, split_op_product_shape
        )
        split_op_product.producer = split_op
        return split_op_product

    def _add_product(self, name: str, shape: typing.List[int]) -> Product:
        """
        Add product to self._products dictionary
        :param name: Name of product
        :param shape: Shape of product
        :return: Product that was created
        """
        assert name not in self._products
        product = Product(name, shape)
        self._products[name] = product
        return product

    def _add_consumers_to_split_op_product(self, preceding_op: Op, split_op_product: Product):
        """
        A Split Op's output product has multiple consumers. Add them to the product.
        :param preceding_op: Op prior to split op
        :param split_op_product: Output product of split op
        """
        dotted_name = preceding_op.dotted_name
        output_product_names = self.get_product_names_from_dotted_name(dotted_name)

        # Important Notes
        # ResNet model uses the same Relu twice in the forward function of ResNet's BasicBlock.
        # The first Relu feeds in to the BasicBlock's Conv2.
        # The second Relu's output is split with one branch feeding the next BasicBlock's conv1 and the other
        # branch feeding in to the next BasicBlock's Add.
        # The following line filters out the Relu whose output is NOT split :(
        #
        # Note2 (Geunho)
        # There was no such above phenomenon when testing tf.keras.applications.resnet.ResNet50
        #   but I left it as defense logic, can remove this logic if it's clear it doesn't happen in tf.keras
        out_product_names = [
            name for name in output_product_names if preceding_op.name in name
        ]

        num_products = len(out_product_names)
        consumer_index = 0
        for a_product_index in range(num_products):
            a_product = self.get_product(out_product_names[a_product_index])
            a_consumer = a_product.consumers[0]
            split_op_product.consumers.append(a_consumer)
            logger.debug("Insert Split Op: Step 2a. Consumer Op: %s, a_product_index: %s",
                         a_consumer.dotted_name, a_product_index)
            # Need to insert the newly created split_op product in the correct input index of the op
            logger.debug("Insert Split Op: Step 2b. Op has multiple input products: %s", a_consumer.inputs)
            input_product_index = determine_preceding_op_input_product_index_in_multi_input_op(preceding_op,
                                                                                               a_consumer)

            a_consumer.inputs[input_product_index] = split_op_product
            logger.debug("Insert Split Op: Step 2c. For product: %s, split_op input_product_index: %s",
                         split_op_product.name, input_product_index)
            consumer_index += 1

    def _create_product_linking_preceding_op_to_split_op(self, preceding_op: Op, split_op: Op):
        """
        Create a new product to connect the preceding Op to the Split Op

        :param preceding_op: Op prior to split op
        :param split_op: Split op to create output product for
        """
        # The preceding Op's output products (products, since it was behaving like a Split) are going to be deleted,
        # since a Split is being inserted in the connected graph.
        # Save the preceding Op's output Product shape.
        # This is needed to create the new product from the preceding Op to the newly being inserted Split Op.

        # Since the preceding Op was behaving like a Split Op, it  would have 2 products with the preceding Op as the
        # producer. Delete these products from the product dictionary.
        preceding_op_product_names = self.get_product_names_from_dotted_name(
            preceding_op.dotted_name
        )
        for name in preceding_op_product_names:
            # Important Notes
            # The following check is needed since ResNet uses the same Relu twice in BasicBlock's forward()
            # Please read the details comments in _add_consumers_to_split_op_product()
            if preceding_op.name in name:
                deleted_product = self._products.pop(name)
                logger.debug("Insert Split Op: Step 3. Deleted product: %s", deleted_product)

        new_product = self._add_product(
            f"{preceding_op.name}__to__{split_op.name}", preceding_op.outputs[0].shape
        )
        new_product.producer = preceding_op
        preceding_op.outputs = [new_product]
        preceding_op.outputs[0].consumers.append(split_op)

    def get_op_from_module_name(self, name: str) -> typing.Union[Op, None]:
        """
        Given the name of a operation/module, return the corresponding op in ops dict
        :param name: tf.keras.layer name
        :return: Connected graph operation corresponding to tf.keras.layer name. Returns None if not found
        """

        layer = self._name_to_layer.get(name, None)
        if layer:
            return self._layer_to_op.get(layer, None)
        return None

    def get_layer_from_op_name(self, name: str) -> tf.keras.layers.Layer:
        """
        Given the name of the op return the corresponding layer
        :param name: Name of the op
        :return: Layer corrsponding to the name provided
        """
        return self._op_name_to_layer.get(name)

    def get_all_ops(self) -> typing.Dict[str, Op]:
        """
        Returns the ops dictionary
        :return: Ops dictionary
        """
        return self._ops

    def get_all_products(self) -> typing.Dict[str, Product]:
        """
        Returns the products dictionary
        :return: Product dictionary
        """
        return self._products

    def get_product(self, name: str) -> Product:
        """
        Returns the product with the name passed in the argument
        :param name: Product name
        """
        return self._products.get(name, None)

    def get_product_names_from_dotted_name(self, dotted_name: str) -> typing.List[str]:
        """
        Returns all names of products whose producer op dotted name matches the argument dotted name.
        For Residual models, same producer will have multiple products.
        During connected graph construction, only one output product can be associated with an op, so previous output
        products are overwritten when a new op is created.  Thus we must search through products dictionary for all
        output products corresponding to an op.
        :param dotted_name: Dotted name for connected graph op to check for output products.
        :return: List of products
        """

        matched_products = []

        for product in self._products.values():
            if product.producer:
                if product.producer.dotted_name == dotted_name:
                    matched_products.append(product.name)

        return matched_products
