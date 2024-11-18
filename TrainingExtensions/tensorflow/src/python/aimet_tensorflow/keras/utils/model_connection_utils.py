# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Model connections utility """
import typing
from collections import OrderedDict
from enum import Enum
import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

class ModelLayerConnectionsProperties(Enum):
    """
    Enum class for model layer connections dict keys and it's type
    """
    INBOUND_NODES = 'inbound_nodes'
    OUTPUT_TENSORS = 'output_tensors'
    CALL_ARGS = 'call_args'
    CALL_KWARGS = 'call_kwargs'
    TYPE = typing.Dict[typing.Dict[str, typing.List[str]],
                       typing.Dict[str, typing.Union[KerasTensor, typing.List[KerasTensor]]]]

class ModelLayerConnections:
    """
    Utility class to handle model connections
    """

    @staticmethod
    def get_model_layers_connection_properties(model: tf.keras.Model) -> typing.Dict:
        """
        A Utility function to create an OrderedDict and then go through each layer of the model to get its inbound nodes, call_args, and call_kwargs.
        This function will also have an empty dict for output tensors if a user wants to use the returned OrderedDict to restructure the model.

        :param model: TensorFlow model
        :return: An OrderedDict with properties for inbound_nodes, call_args, call_kwargs, and an empty dict for output_tensors.
        """
        # Dictionary for mapping layers inbound_nodes, call_args, call_kwargs, and output_tensors
        model_layer_connections = OrderedDict()
        model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES] = OrderedDict()
        model_layer_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS] = OrderedDict()
        model_layer_connections[ModelLayerConnectionsProperties.CALL_ARGS] = OrderedDict()
        model_layer_connections[ModelLayerConnectionsProperties.CALL_KWARGS] = OrderedDict()

        for current_layer in model.layers:
            for outbound_node in current_layer.outbound_nodes:
                outbound_layers_name = outbound_node.outbound_layer.name

                # Get the inbound nodes for a given outbound layer
                model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                    {
                        outbound_layers_name: [
                            *model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES].get(outbound_layers_name, []), current_layer.name
                        ]
                    }
                )

                # Get call args and kwargs for a given layer
                model_layer_connections[ModelLayerConnectionsProperties.CALL_ARGS].update(
                    {outbound_node.layer.name: outbound_node.call_args}
                )
                model_layer_connections[ModelLayerConnectionsProperties.CALL_KWARGS].update(
                    {outbound_node.layer.name: outbound_node.call_kwargs}
                )

        # After having all the inbound nodes for a given layer, we go back through and for any layers that
        # have multi input, we sort the inputs based on the original call args.
        KERAS_SYMBOLIC_TENSOR_INDEX = 0
        for layer_name, inbound_nodes in model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES].items():
            # If the original keras symbolic tensors for a given layer are a List, then we set the `original_keras_symbolic_tensors_order`
            # param and sort the layers inbound nodes.
            if isinstance(
                    original_keras_symbolic_tensors_order :=
                    model_layer_connections[ModelLayerConnectionsProperties.CALL_ARGS][layer_name][KERAS_SYMBOLIC_TENSOR_INDEX],
                    typing.List):
                ordered_inputs = {
                    k._keras_history.layer.name: v #pylint: disable=protected-access
                    for v, k in enumerate(original_keras_symbolic_tensors_order)
                }

                correctly_ordered_inbound_nodes = sorted(inbound_nodes, key=lambda current_input, oi=ordered_inputs: oi[current_input])

                model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layer_name] = correctly_ordered_inbound_nodes

        return model_layer_connections

    @staticmethod
    def merge_model_layers_connections(model_layers_connections1: typing.Dict, model_layers_connections2: typing.Dict) -> typing.Dict:
        """
        Merge two network dictionaries

        :param model_layers_connections1: The model layer connection dictionary 1
        :param model_layers_connections2: The model layer connection dictionary 2
        :return: Merged network dictionary
        """
        for key in model_layers_connections1:
            model_layers_connections1[key].update(model_layers_connections2[key])

        return model_layers_connections1
