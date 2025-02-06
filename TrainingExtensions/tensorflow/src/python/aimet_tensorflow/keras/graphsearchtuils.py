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
Connected graph search utilities
"""

import collections
import typing
import numpy as np
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CrosslayerEqualization)

ClsSet = typing.Union[typing.Tuple[tf.keras.layers.Conv2D,
                                   tf.keras.layers.Conv2D],
                      typing.Tuple[tf.keras.layers.Conv2D,
                                   tf.keras.layers.DepthwiseConv2D,
                                   tf.keras.layers.Conv2D]]

cls_supported_layers = (tf.keras.layers.Conv2D, tf.keras.layers.Conv1D)
zero_padding_layers = (tf.keras.layers.ZeroPadding2D, tf.keras.layers.ZeroPadding1D)
cls_supported_activations = (tf.keras.layers.ReLU, tf.keras.layers.PReLU)




class GraphSearchUtils:
    """Implements graph search utils required by CLE feature"""

    def __init__(self, model: tf.keras.Model):
        """
        :param model: Keras Model that is built (Sequential, Functional)
        """
        self._connected_graph = ConnectedGraph(model)
        self._ordered_module_list = self._get_ordered_list_of_conv_modules()

    def _get_ordered_list_of_conv_modules(self):
        """
        Finds order of nodes in graph
        :return: List of name, layer tuples in graph in order
        """
        result = []
        for op in self._connected_graph.ordered_ops:
            layer = op.get_module()
            if isinstance(layer, cls_supported_layers):
                result.append([layer.name, layer])

        return result

    def find_layer_groups_to_scale(self) -> typing.List[typing.List[tf.keras.layers.Conv2D]]:
        """
        Find layer groups to scale
        :return: List of groups of layers. Each group can be independently equalized
        """
        # Find the input node(s) in the graph
        input_nodes = []
        for op in self._connected_graph.get_all_ops().values():
            if op.inputs and op.inputs[0].is_model_input:
                input_nodes.append(op)

        layer_groups = []
        for op in input_nodes:
            self.find_downstream_layer_groups_to_scale(op, layer_groups)

        # Sort the layer groups in order of occurrence in the model
        ordered_layer_groups = []
        for _, module in self._ordered_module_list:
            for layer_group in layer_groups:
                if layer_group[0] is module:
                    ordered_layer_groups.append(layer_group)

        return ordered_layer_groups

    @staticmethod
    def find_downstream_layer_groups_to_scale(op, layer_groups, current_group=None, visited_nodes=None):
        """
        Recursive function to find cls layer groups downstream from a given op
        :param op: Starting op to search from
        :param layer_groups: Running list of layer groups
        :param current_group: Running current layer group
        :param visited_nodes: Running list of visited nodes (to short-circuit recursion)
        :return: None
        """

        if not visited_nodes:
            visited_nodes = []
        if not current_group:
            current_group = []

        if op in visited_nodes:
            return
        visited_nodes.append(op)

        current_layer = op.get_module()
        # Conv2D, Conv1D or its subclass is added to the current group
        if current_layer and isinstance(current_layer, cls_supported_layers):
            current_group.append(current_layer)

        # Terminating condition for current group
        if not current_layer or not GraphSearchUtils._is_supported_layer_case(current_layer):
            if (len(current_group) > 1) and (current_group not in layer_groups):
                layer_groups.append(current_group)
            current_group = []

        for consumer in op.output_ops:
            GraphSearchUtils.find_downstream_layer_groups_to_scale(consumer, layer_groups,
                                                                   current_group, visited_nodes)

        # Reached a leaf.. See if the current group has something to grab
        if (len(current_group) > 1) and (current_group not in layer_groups):
            layer_groups.append(current_group)

    @staticmethod
    def _is_supported_layer_case(layer: tf.keras.layers.Layer) -> bool:
        """
        Check if the current layer is CLS supported layers or a supported activation layer
        :param layer: tf.keras.layers.Layer
        :return: True if it's CLS supported layers or a supported layer
        """
        return isinstance(layer, (cls_supported_layers + zero_padding_layers)) or \
            GraphSearchUtils._is_supported_activations(layer) or \
            GraphSearchUtils.is_folded_batch_normalization(layer)

    @staticmethod
    def is_folded_batch_normalization(layer: tf.keras.layers.Layer) -> bool:
        """
        Method to check if layer is folded batchnorm or not
        :param layer: layer to check if it is folded batch norm
        :return: True if it is folded batch norm, False if not
        """
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            return False

        return np.all(layer.beta == 0.0) and np.all(layer.gamma == 1.0)

    @staticmethod
    def _is_supported_activations(layer: tf.keras.layers.Layer) -> bool:
        """
        Check if the current layer is a supported activation layer
        :param layer: tf.keras.layers.Layer
        :return: True if layer is ReLU, PReLU or Activation with supported type
        """
        # Case of explicit layer such as tf.keras.layers.ReLU
        if isinstance(layer, cls_supported_activations):
            return True

        # Case of implicit layer such as tf.keras.layers.Activation(tf.nn.relu)
        # Note: PReLU is not supported by implicit approach until TF 2.4
        layer_config = layer.get_config()
        activation = layer_config.get("activation")

        if activation is None:
            return False

        return activation in ["relu", "relu6"]

    @staticmethod
    def convert_layer_group_to_cls_sets(layer_group: typing.List[tf.keras.layers.Conv2D]) \
            -> typing.List[ClsSet]:
        """
        Helper function to convert a layer group to a list of cls sets
        :param layer_group: Given layer group to convert
        :return: List of cls sets
        """
        cls_sets = []

        layer_group = collections.deque(layer_group)
        prev_layer_to_scale = layer_group.popleft()
        while layer_group:
            next_layer_to_scale = layer_group.popleft()

            if isinstance(next_layer_to_scale, tf.keras.layers.DepthwiseConv2D):
                next_non_depthwise_conv_layer = layer_group.popleft()
                # DepthwiseConv layer right after DepthwiseConv layer is not currently supported
                if isinstance(next_non_depthwise_conv_layer, tf.keras.layers.DepthwiseConv2D):
                    _logger.error("Consecutive DepthwiseConv layer not currently supported")
                    raise NotImplementedError("Consecutive DepthwiseConv layer not currently supported")

                cls_sets.append(
                    (prev_layer_to_scale, next_layer_to_scale, next_non_depthwise_conv_layer))
                prev_layer_to_scale = next_non_depthwise_conv_layer
            else:
                cls_sets.append((prev_layer_to_scale, next_layer_to_scale))
                prev_layer_to_scale = next_layer_to_scale

        return cls_sets

    @staticmethod
    def is_relu_activation_present_in_cls_sets(cls_sets: typing.List[ClsSet]) \
            -> typing.List[typing.Union[bool, typing.Tuple[bool, bool]]]:
        """
        Check if there is ReLU or PReLU activation between cls sets
        :param cls_sets: List of ClsSet to find ReLU activation in
        :return: List of ReLU activation preset flags (bool or tuple of bool) corresponding to input cls_sets param
        """

        is_relu_activation_in_cls_sets = []
        for cls_set in cls_sets:
            cls_set = cls_set[:-1]

            is_relu_activation_in_cls_set = []
            for layer in cls_set:
                has_relu_activation = GraphSearchUtils._does_layer_have_relu_activation(layer)
                is_relu_activation_in_cls_set.append(has_relu_activation)

            if len(is_relu_activation_in_cls_set) == 1:
                is_relu_activation_in_cls_sets.append(is_relu_activation_in_cls_set[0])
            else:
                is_relu_activation_in_cls_sets.append(tuple(is_relu_activation_in_cls_set))

        return is_relu_activation_in_cls_sets

    @staticmethod
    def _does_layer_have_relu_activation(layer: tf.keras.layers.Conv2D) -> bool:
        """
        Check if layer has ReLU or PReLU activation function
        :param layer: Conv2D or it's subclass to check activation function
        :return: True If layer has ReLU or PReLU activation, otherwise False
        """
        def _get_activation_type(_layer: tf.keras.layers.Layer) -> str:
            """
            Get activation name string from _layer
            :param _layer: tf.keras.layers.Layer
            :return: activation name string
            """
            activation_info = tf.keras.activations.serialize(_layer.activation)

            if isinstance(activation_info, str):
                # Instantiating like tf.keras.layers.Conv2D(8, kernel_size=3, activation=tf.keras.activations.relu)
                #   has the result of serialization as str type
                _activation_type = activation_info
            elif isinstance(activation_info, dict):
                # Instantiating like tf.keras.layers.Conv2D(8, kernel_size=3, activation=tf.keras.layers.ReLU())
                #   has the result of serialization as dict type
                _activation_type = activation_info["class_name"].lower()
            else:
                raise NotImplementedError("Not supported format")

            return _activation_type

        def _get_outbound_layer(_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
            """
            Get outbound layer from current layer
            If right after layer is folded batchnorm, find next applicable layer
            :param _layer: tf.keras.layers.Layer
            :return: outbound layer of _layer
            """
            assert len(_layer.outbound_nodes) == 1

            _outbound_layer = _layer.outbound_nodes[0].outbound_layer
            return _get_outbound_layer(_outbound_layer) \
                if GraphSearchUtils.is_folded_batch_normalization(_outbound_layer) \
                else _outbound_layer

        activation_type = _get_activation_type(layer)
        supported_activation_types = {"relu", "prelu"}

        # If activation parameter is not set or None, default activation_type is linear
        if activation_type == "linear" and layer.outbound_nodes:
            outbound_layer = _get_outbound_layer(layer)

            # Case 1. Non-fused use case
            # Case 1-1. Conv(..., activation=None) -> ReLU() or
            #           Conv(..., activation=None) -> Folded BN -> ReLU()
            is_using_relu_layer = isinstance(outbound_layer, cls_supported_activations)

            # Case 1-2. Conv(..., activation=None) -> Activation(activation="relu") or
            #           Conv(..., activation=None) -> Folded BN -> Activation(activation="relu")
            is_using_activation_layer = isinstance(outbound_layer, tf.keras.layers.Activation) and \
                                        _get_activation_type(outbound_layer) in supported_activation_types
            return is_using_relu_layer or is_using_activation_layer

        # Case 2. Fused use case
        # e.g., Conv(..., activation="relu") or Conv(..., activation=tf.keras.layers.PReLU())
        return activation_type in supported_activation_types
