# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#pylint: disable=too-many-lines
""" Utility for batch norm fold in tf 2.x """
from typing import Iterable, Optional, Tuple, Union, List, Dict, Set
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from packaging import version  # pylint: disable=wrong-import-order

if version.parse(tf.version.VERSION) >= version.parse("2.10"):
    # Ignore pylint errors as keras module is not available in TF 2.4
    from keras.layers.core.tf_op_layer import TFOpLambda # pylint: disable=import-error
    from keras.engine.functional import Functional # pylint: disable=import-error
else:
    # Ignore pylint errors due to conditional imports
    from tensorflow.python.keras.engine.functional import Functional # pylint: disable=ungrouped-imports
    from tensorflow.python.keras.layers.core import TFOpLambda # pylint: disable=ungrouped-imports

# pylint: disable=wrong-import-position
from aimet_common.defs import QuantScheme, MAP_ROUND_MODE_TO_PYMO

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.model_preparer import _KerasModelPreparer
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ParamPerTensorQuantizer
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.utils import common
from aimet_tensorflow.keras.utils.model_connection_utils import ModelLayerConnections, ModelLayerConnectionsProperties
from aimet_tensorflow.keras.utils.quantizer_utils import get_wrappers_bias_quantizer, get_wrappers_weight_quantizer
from aimet_tensorflow.keras.utils.weight_tensor_utils import WeightTensorUtils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

LayerType = Union[
    tf.keras.layers.Conv2D,
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2DTranspose,
    tf.keras.layers.DepthwiseConv2D
]
_supported_layers = LayerType.__args__

PairType = Union[Tuple[LayerType, tf.keras.layers.BatchNormalization, bool],
                 Tuple[tf.keras.layers.BatchNormalization, LayerType, bool]]

BatchNormType = tf.keras.layers.BatchNormalization
_supported_batchnorms = BatchNormType

# Todo: search for more types of convolution
LinearType = tf.keras.layers.Dense
ConvType = Union[tf.keras.layers.Conv1D,
                 tf.keras.layers.Conv2D,
                 tf.keras.layers.DepthwiseConv2D,
                 tf.keras.layers.Conv2DTranspose]
_supported_convs = ConvType.__args__

FlattenType = Union[tf.keras.layers.Flatten, tf.keras.layers.Reshape]

MAP_PYMO_TO_ROUND_MODE = {v: k for k, v in MAP_ROUND_MODE_TO_PYMO.items()}
def _check_layer_to_find_pattern(cur_layer: tf.keras.layers.Layer,
                                 conv_linear_with_bn_dict: Dict[Union[ConvType, LinearType],
                                                                List[Union[None, BatchNormType]]],
                                 layer_out_node_ref: Dict,
                                 has_seen: List[Union[None, ConvType, BatchNormType, FlattenType]]):
    """
    find all paths in the model considering all inputs.

    :param cur_layer: layer to investigate for finding a pattern
    :param conv_linear_with_bn_dict: dictionary to store possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    :param layer_out_node_ref: dictionary includes layer_ref as a key, outbound nodes as value
    :param has_seen: for storing the layer which is useful for finding pattern in the next layers;
        index 0 is for conv op, index 2 is for bn op and index 3 is for storing flatten/reshape op
    """

    # pylint: disable=too-many-branches
    if isinstance(cur_layer, _supported_convs):
        if has_seen[1] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
            has_seen[1] = None
        if (cur_layer.activation is tf.keras.activations.linear) and \
                (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            has_seen[0] = cur_layer
    elif isinstance(cur_layer, BatchNormType):
        if has_seen[0] is not None:
            if has_seen[0] in conv_linear_with_bn_dict:
                conv_linear_with_bn_dict[has_seen[0]][1] = cur_layer
            else:
                conv_linear_with_bn_dict[has_seen[0]] = [None, cur_layer]
            has_seen[0] = None
        if (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            has_seen[1] = cur_layer
    elif isinstance(cur_layer, (tf.keras.layers.Flatten, tf.keras.layers.Reshape)):
        if (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            if has_seen[1]:
                has_seen[2] = cur_layer
            else:
                has_seen[1] = None
        if has_seen[0]:
            has_seen[0] = None
    elif isinstance(cur_layer, LinearType):
        if has_seen[1] is not None and has_seen[2] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
        has_seen[2] = None
        has_seen[1] = None
    else:
        has_seen[0] = None
        has_seen[1] = None
        has_seen[2] = None


def _add_children_layer_before_parent_layer(cur_layer: tf.keras.layers.Layer, node_layer_map: Dict,
                                            layer_out_node_map: Dict,
                                            visited_layers: Set[tf.keras.layers.Layer],
                                            reversed_ordered_layers: List[tf.keras.layers.Layer]):
    """
    Function to use topological sorting for finding all the layers which are accessible
    from the specific input_layer in the opposite order of occurrence.

    :param cur_layer:layer that we want to find path from
    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :param visited_layers: Set of all layers that have been visited
    :param reversed_ordered_layers: List of layers in the opposite order of occurrence
        for the layers that we have visited so far
    """

    # Mark the current layer as visited.
    visited_layers.add(cur_layer)

    if cur_layer in layer_out_node_map:
        # Recur for all the layers adjacent to this layer
        for next_node in layer_out_node_map[cur_layer]:
            next_layer = node_layer_map[next_node][1]
            if next_layer not in visited_layers:
                _add_children_layer_before_parent_layer(next_layer, node_layer_map,
                                                        layer_out_node_map, visited_layers,
                                                        reversed_ordered_layers)
            reversed_ordered_layers.append(cur_layer)
    else:
        reversed_ordered_layers.append(cur_layer)


def _get_ordered_layers(node_layer_map: Dict,
                        layer_out_node_map: Dict) -> List[tf.keras.layers.Layer]:
    """
    Function to return the list with all the layers in which layers come before parent layer.

    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: ordered_layers: List of all layers in the order of occurrence
    """
    # to find the input layers of the model
    input_layers = common.find_input_layers(node_layer_map)

    #  Set of all layers that have been visited (to cut short duplicate traversals)
    visited_layers = set()

    # List of all layers in the opposite of order of occurrence
    reversed_ordered_layers = []

    for input_layer in input_layers:
        _add_children_layer_before_parent_layer(input_layer, node_layer_map, layer_out_node_map,
                                                visited_layers, reversed_ordered_layers)

    # reverse the list because layers are in reverse order
    ordered_layers = reversed_ordered_layers[::-1]

    # # filter ordered ops for only valid ops
    # ordered_ops = [op for op in ordered_ops if op in valid_ops]

    return ordered_layers


def _get_ordered_conv_linears(node_layer_map: Dict,
                              layer_out_node_map: Dict) -> List[Union[ConvType, LinearType]]:
    """
    helper to select a list of conv_linears in the order of occurence

    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: return List of conv/linear layer refs
    """
    # get ordered layers list in node_layer map dictionary
    list_of_ordered_layers = _get_ordered_layers(node_layer_map, layer_out_node_map)

    # look for conv layers
    ordered_conv_linears = []
    for layer in list_of_ordered_layers:
        if isinstance(layer, _supported_layers):
            ordered_conv_linears.append(layer)
    return ordered_conv_linears


def _fill_conv_linear_bn_dict(cur_layer: tf.keras.layers.Layer, node_layer_ref: Dict,
                              layer_out_node_ref: Dict,
                              has_seen: List[Union[None, ConvType, BatchNormType, FlattenType]],
                              visited_layer: Set[tf.keras.layers.Layer],
                              conv_linear_with_bn_dict: Dict[Union[ConvType, LinearType],
                                                             List[Union[None, BatchNormType]]]):
    """
    fill conv_linear_bn_dict for the model

    :param cur_layer: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param node_layer_ref: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_ref: dictionary includes layer_ref as a key, outbound nodes as value
    :paramm has_seen: for storing the layer which is useful for finding pattern in the next layers;
        index 0 is for conv op, index 2 is for bn op and index 3 is for storing flatten/reshape op
    :param visited_layer: to store all the layers that have been visited so far in the dictionary
    :param conv_linear_with_bn_dict: dictionary of all possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    """

    # Mark the current layer as visited to prevent passing from one layer more than once
    visited_layer.add(cur_layer)

    _check_layer_to_find_pattern(cur_layer, conv_linear_with_bn_dict, layer_out_node_ref, has_seen)

    if cur_layer in layer_out_node_ref:
        for next_node in layer_out_node_ref[cur_layer]:
            next_layer = node_layer_ref[next_node][1]
            if next_layer not in visited_layer:
                _fill_conv_linear_bn_dict(next_layer, node_layer_ref, layer_out_node_ref, has_seen,
                                          visited_layer, conv_linear_with_bn_dict)
            else:
                has_seen[0] = None
                has_seen[1] = None
                has_seen[2] = None


def _find_possible_convs_linears_bn(node_layer_map: Dict, layer_out_node_map: Dict)\
        -> Dict[Union[ConvType, LinearType], List[Union[None, BatchNormType]]]:
    """
    find all possible convs_linears_bn by traversing all paths in the model considering all inputs

    :param node_layer_map:  dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: return dictionary of all possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    """

    input_layers = common.find_input_layers(node_layer_map)
    visited_layer = set()
    conv_linear_with_bn_dict = {}

    for input_layer in input_layers:
        _fill_conv_linear_bn_dict(input_layer, node_layer_map, layer_out_node_map,
                                  [None, None, None], visited_layer, conv_linear_with_bn_dict)

    return conv_linear_with_bn_dict


def _get_bn_params(bn: tf.keras.layers.BatchNormalization) -> libpymo.BNParams():
    """
    helper to populate BN params from given BN Layer, required for fold

    :param bn: BatchNorm Layer
    :return: return bn params in libpymo.TensorParams() format.
    """
    if bn.gamma is None:
        _logger.warning("Gamma for BatchNormalization '%s' is None. Setting to ones.", bn.name)
        # Batch Normalization layers can having missing gammas with two different cases. One is that the 'gamma' attribute
        # is set to None. The second is if `scale` is set to False upon creation of the layer which turns off gamma.
        with tf.name_scope(bn.name):
            weights_with_gamma_and_before_rebuild = [np.ones_like(bn.beta)] + bn.get_weights()
            bn.scale = True
            bn.build(bn.input.shape)
            bn.set_weights(weights_with_gamma_and_before_rebuild)
            bn.gamma = next(filter(lambda w: 'gamma' in w.name, bn.weights))

    bn_params = libpymo.BNParams()

    bn_params.gamma = bn.gamma.numpy().reshape(-1)
    bn_params.beta = bn.beta.numpy().reshape(-1)
    bn_params.runningMean = bn.moving_mean.numpy().reshape(-1)
    bn_params.runningVar = bn.moving_variance.numpy().reshape(-1)
    epsilon = bn.epsilon
    var = bn.moving_variance.numpy()
    var_with_epsilon = var + epsilon
    sigma = np.sqrt(var_with_epsilon)
    bn_params.runningVar = sigma

    return bn_params


def _get_bias_tensor(conv_linear: LayerType) -> libpymo.TensorParams():
    """
    Get bias tensor in given conv layer.

    Packs bias in the format required for BN fold
    (libpymo.TensorParams()).
    :param conv_linear: conv Layer
    :return: return bias param in libpymo.TensorParams() format.
    """

    bias_tensor = libpymo.TensorParams()
    if conv_linear.bias is not None:
        bias_tensor.data = conv_linear.bias.numpy().reshape(-1)
        bias_tensor.shape = np.array(conv_linear.bias.shape)

    return bias_tensor


def _get_weight_tensor_transpose_reshape(conv_linear: LayerType) -> libpymo.TensorParams():
    """
    Get weight tensor from conv layer.

    Converts to right format - performs transpose and reshape.
    Packs it to the format required for BN fold (libpymo.TensorParams()).
    :param conv_linear: conv layer
    :return: return weight tensor in libpymo.TensorParams() format.
    """

    # Weight tensor libpymo format
    weight_tensor = libpymo.TensorParams()

    # linear array to be sent for bn fold
    weight = conv_linear.get_weights()[0]
    shape = weight.shape

    if isinstance(conv_linear, tf.keras.layers.DepthwiseConv2D):
        # Depthwise conv layers in TF have outputs(Noc) set to 1.
        # we will use format [Nic, Noc, kh, kw] -
        # to be compatible with cpp backend.
        weight = np.transpose(weight, (2, 3, 0, 1))
        # [Nic, Noc, kh, kw]
        shape = np.array([shape[2], shape[3], shape[0], shape[1]])
    elif isinstance(conv_linear, tf.keras.layers.Dense):
        shape = np.concatenate((np.array([1, 1]), shape))
        weight = np.transpose(weight, (1, 0))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[3], shape[2], shape[0], shape[1]])
    elif isinstance(conv_linear, tf.keras.layers.Conv2DTranspose):
        weight = np.transpose(weight, (2, 3, 0, 1))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[2], shape[3], shape[0], shape[1]])
    elif isinstance(conv_linear, tf.keras.layers.Conv2D):
        weight = np.transpose(weight, (3, 2, 0, 1))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[3], shape[2], shape[0], shape[1]])
    else:
        _logger.error("_get_weight_tensor_transpose_reshape(): Operation type unsupported")

    weight_tensor.data = weight.reshape(-1)
    weight_tensor.shape = shape

    return weight_tensor


class PassThroughOp(tf.keras.layers.Layer):
    """
    This is a pass-through op, used for purpose of making an op a no-op
    """

    # pylint: disable=arguments-differ
    @staticmethod
    def call(inputs):
        """
        This is a function to return input as an output
        :param inputs: input to pass through
        """
        return inputs

# pylint: disable=too-many-branches, protected-access, too-many-locals, too-many-nested-blocks
@common.to_functional
def _delete_bn_from_functional(model: tf.keras.Model,
                               bn_layers_to_remove: List[tf.keras.layers.BatchNormalization]) -> tf.keras.Model:
    """
    This function is used to remove ALL batch normalization layers from a functional model passed via the
    bn_layers_to_remove parameter. Removing in place is not possible for functional models as the layers inbound and
    outbound connections are immutable. This function returns a new model with the batch normalization layers removed.

    :param model: Model to remove bn_layers from
    :param bn_layers_to_remove: List of batch normalization layers to remove from the model
    :return: A new model with the batch normalization layers removed
    """

    # In order to do this, we first need to know the original models inbound and outbound connections to each layer.
    # We then need to create a new model with the same inbound and outbound connections, but with the batch normalization
    # layers removed. This is done by rerouting the inbound nodes of the batch normalization layers to the inbound nodes
    # of the next layer. This can be seen in the following diagram:
    #
    # Original model flow ------------------------->
    #   ______________        ______________        ______________
    #  |             |       |             |       |             |
    #  |    Conv     |  -X-> |  Batch Norm |  -X-> |    ReLU     |
    #  |_____________|       |_____________|     ^ |_____________|
    #  New model flow   \                       /
    #                    \                     /
    #                     \___________________/


    def wrapped_bn_layer_in_bns_to_remove(layer: tf.keras.layers.Layer) -> bool:
        return isinstance(layer, QcQuantizeWrapper) and layer._layer_to_wrap in bn_layers_to_remove

    tf.keras.backend.clear_session() # clear session to not have tensor name conflicts

    # Step 1: Get the inbound and outbound connections for each layer in the model
    model_layer_connections = ModelLayerConnections.get_model_layers_connection_properties(model)

    for inp in model.inputs:
        model_layer_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update({inp.name: inp})

    # Step 2: Create a new model with the batch normalization layers removed by iterating through the layers in the model
    # and using the inbound and outbound connections to rerouting around the batch normalization layers.
    batch_norms_replaced_with_names = {}
    model_outputs = []
    for current_layer in model.layers:
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            continue

        # Determine input tensors of the given layer
        layer_input = [model_layer_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS][layer_aux]
                       for layer_aux in model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES][current_layer.name]]

        layer_input = layer_input[0] if len(layer_input) == 1 else layer_input

        # Reroute around batch normalization layers if the layer is in the list of layers to remove
        if current_layer in bn_layers_to_remove or wrapped_bn_layer_in_bns_to_remove(current_layer):
            _logger.debug("Removing Batch Normalization layer %s", current_layer.name)

            for outbound_node in current_layer._outbound_nodes:  # pylint: disable=protected-access
                # Find and replace the Batch Normalization output layers input that holds the Batch Normalization layer
                # node and replace it with the input layers of the Batch Normalization layer.
                # For example, if ReLU's inputs are [conv1_bn] and conv1_bn's inputs are [conv1], then we replace
                # ReLU's inputs with [conv1]

                all_batch_norms_inbound_layers_names = \
                    [inbound_node.inbound_layers.name for inbound_node in current_layer._inbound_nodes]

                # Go through all the outbound layers of the batch normalization layer and replace the batch normalization
                # layer name with the input layer names of the batch normalization layer.
                batch_norms_outbound_layers_new_inbound_layers_names = \
                    [outlayer.replace(current_layer.name, *all_batch_norms_inbound_layers_names)
                     for outlayer in model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES][outbound_node.outbound_layer.name]]

                # Keras Batch Norm only supports one input tensors. Meaning there is one singular layer coming into it.
                # Hence, 'inbound_nodes[0]'.
                batch_norms_replaced_with_names[current_layer.name] = current_layer._inbound_nodes[0].inbound_layers.name

                model_layer_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
                    {outbound_node.outbound_layer.name: batch_norms_outbound_layers_new_inbound_layers_names})

                # The above updates our dict for the mapping of the inputs, but we need to also update what Keras thinks
                # the inputs are. This is done by updating the inbound nodes of the output layer of the Batch Normalization.
                # THIS IS ONLY FOR MAPPING THE INPUTS TO BUILD A NEW MODEL. The original models underlying structure is
                # not changed.
                outbound_node.outbound_layer._inbound_nodes = current_layer.inbound_nodes  # pylint: disable=protected-access

        # Otherwise, treat like a normal layer
        else:
            # For layers that have multiple inputs, order matters for what is fed into the layer. For example, if we have
            # an Add layer with inputs from a ReLU and a Batch Norm, the order they go into the Add matters. Furthermore,
            # if the Batch Norm is deleted, then it needs to be replaced with it's folded layer in the same order.

            KERAS_SYMBOLIC_TENSORS_INDEX = 0
            # Check if we need to change layer_input order. If there is just one input, there is no order.
            # Special case when there is a Lambda layer with multiple inputs is handled seperately
            if isinstance(layer_input, List) and not isinstance(current_layer, TFOpLambda):
                # Original models keras symbolic tensor order
                original_keras_symbolic_tensors_order = model_layer_connections[ModelLayerConnectionsProperties.CALL_ARGS][
                    current_layer.name][KERAS_SYMBOLIC_TENSORS_INDEX]

                # Special case for Lambda layers. Lambda layers can be thought of as z = x + y. Unfortunately, their call
                # args for the keras symbolic tensors will ONLY have the x portion. In our layer_input we have both x and y.
                # This statement is added to wrap the x portion of the original call args and check if it's a batch norm
                # folded out.
                if not isinstance(original_keras_symbolic_tensors_order, List):
                    original_keras_symbolic_tensors_order = [original_keras_symbolic_tensors_order]

                # Check if a Batch Norm that was deleted is in the original keras symbolic order.
                name_of_bn_replaced = [
                    tensor._keras_history.layer.name
                    for tensor in original_keras_symbolic_tensors_order
                    if tensor._keras_history.layer.name in batch_norms_replaced_with_names
                ]

                # If a Batch Norm is found, then the keras symbolic tensor order is slightly updated to replace the
                # Batch Norm with the folded layer. Otherwise, we can just use the original keras symbolic tensor order.
                if name_of_bn_replaced:

                    updated_keras_symbolic_tensors_order = []
                    for keras_symbolic_tensor in original_keras_symbolic_tensors_order:
                        if (name_of_bn := keras_symbolic_tensor._keras_history.layer.name) in name_of_bn_replaced: #pylint: disable=superfluous-parens
                            updated_keras_symbolic_tensors_order.append(
                                model_layer_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS][
                                    batch_norms_replaced_with_names[name_of_bn]
                                ]
                            )
                        else:
                            updated_keras_symbolic_tensors_order.append(keras_symbolic_tensor)

                    # Dictionary of the keras symbolic tensor name to the order.
                    ordered_inputs = {k.name: v for v, k in enumerate(updated_keras_symbolic_tensors_order)}

                    # Sort layer_input based on the above dictionary.
                    layer_input = sorted(layer_input, key=lambda current_input, oi=ordered_inputs: oi[current_input.name])

            # Since we are rerouting around the batch normalization layers, we need to temporarily remove the inbound
            # and outbound nodes of the batch normalization layers so that the model can be built correctly and not
            # duplicate the non batch normalization layers inbound/outbound nodes.
            current_layer._inbound_nodes = []  # pylint: disable=protected-access
            # Special case for when there is a Lambda operation with multiple inputs. For example, z = x + y.
            if isinstance(current_layer, TFOpLambda):
                kmp = _KerasModelPreparer.get_instance_for_common_layer_passthrough_functions(model_layer_connections)
                x = kmp._handle_normal_keras_layer(current_layer)  # pylint: disable=protected-access
                # Updating the Model layer connections
                kmp._update_output_tensors_in_model_layers_connections(  # pylint: disable=protected-access
                    current_layer, x, model
                )
            else:
                x = current_layer(layer_input)
            current_layer._outbound_nodes = []  # pylint: disable=protected-access

            # Set new output tensor (in this case, it will be the same as the original model)
            model_layer_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update({current_layer.name: x})

        # Save tensor in output list if it is output in the initial model
        if current_layer.name in model.output_names:
            model_outputs.append(x)

    return tf.keras.Model(inputs=model.inputs, outputs=model_outputs)


def _delete_bn_from_sequential(layer: tf.keras.layers.Layer,
                               bn: tf.keras.layers.BatchNormalization):

    """
    This is the function for removing batch normalization layers that are layers of sequential model

    :param layer: model to obtain bn_layer that we want to remove
    :param bn: batch normalization layer that needs to be removed
    """

    layers_after_bn = []
    visited = False
    idx = None
    # pylint: disable=protected-access
    for index, inner_layer in enumerate(layer.layers):
        if visited:
            layers_after_bn.append(inner_layer)

        elif inner_layer == bn:
            visited = True
            idx = index

        elif inner_layer.submodules:
            _delete_bn_for_non_subclassed_model(inner_layer, bn)

    if visited and idx is not None:
        # pylint: disable=protected-access
        for _ in range(len(layer.layers) - idx):
            layer.pop()
        for layer_to_add in layers_after_bn:
            layer.add(layer_to_add)


def _delete_bn_for_non_subclassed_model(model: Union[tf.keras.Model, tf.keras.layers.Layer],
                                        bn_layer: tf.keras.layers.BatchNormalization):
    """
    Remove bn layer for those model which are not part of model subclassing

    :param model: model to delete bn layers from
    :param bn_layer: bn layer that should be removed
    """

    if isinstance(model, tf.keras.Sequential):
        _delete_bn_from_sequential(model, bn_layer)

    # We are expecting to find sequential model in functional model
    # or model subclassing in the elif statement
    elif isinstance(model, (tf.keras.layers.Layer, tf.keras.Model)):
        for layer in model.layers:
            if layer.submodules:
                _delete_bn_for_non_subclassed_model(layer, bn_layer)


def _delete_bn_from_model_subclassing(module_to_name_map: Dict[tf.keras.layers.Layer,
                                                               Tuple[tf.keras.Model, str]],
                                      bn_layer: tf.keras.layers.BatchNormalization):
    """
    Remove bn layer which is part of model subclassing api
    or model inheriting from tf.keras.layers.Layer

    :param module_to_name_map: model to remove bn from
    :param bn_layer: bn layer that should be removed
    """

    parent_ref, module_name = module_to_name_map[bn_layer]
    op = PassThroughOp()
    setattr(parent_ref, module_name, op)

# pylint: disable=inconsistent-return-statements
def _delete_all_bns_from_model(model: Union[tf.keras.Model, tf.keras.layers.Layer],
                               bn_layers: List[tf.keras.layers.BatchNormalization]) -> Optional[tf.keras.Model]:
    """
    Remove all bn layers for a given model.

    :param model: Model to have the bn layers removed from
    :param bn_layers: bn layers that should be removed
    :return: new model with bn layers removed, if model is functional else None
    """
    if bn_layers:
        # QuantizationSimModel's model will fall into this case.
        if isinstance(model, Functional) and not isinstance(model, tf.keras.Sequential) or any(isinstance(l, QcQuantizeWrapper) for l in model.layers):
            return _delete_bn_from_functional(model, bn_layers)

        module_to_name_map = common.module_to_name_map(model)

        for bn_layer in bn_layers:
            if bn_layer in module_to_name_map:
                _delete_bn_from_model_subclassing(module_to_name_map, bn_layer)
            else:
                _delete_bn_for_non_subclassed_model(model, bn_layer)


def _find_all_batch_norms_to_fold(model: tf.keras.Model) -> Tuple[List[PairType], List[PairType], Set[tf.keras.layers.BatchNormalization]]:
    """
    uses searcher to choose layers for bias correction

    :param model: model to obtain conv_linear pairs for
    :return: List of conv/linear layers with associated bn op / activation info and
            a Set of all the batch norms which are marked for folding.
    """

    node_layer_map = common.create_node_to_layer_map(model)
    layer_out_node_map = common.create_layer_to_out_node_map(model)

    possible_convs_linears_bn = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

    # get all ordered convs/ linears layers
    ordered_conv_linears = _get_ordered_conv_linears(node_layer_map, layer_out_node_map)

    bn_picked_for_folding = set()
    def get_pairs(conv_is_first=False) -> List:
        index = 1 if conv_is_first else 0

        pairs_list = []
        for conv_linear in ordered_conv_linears:
            if conv_linear in possible_convs_linears_bn and (bn_info := possible_convs_linears_bn[conv_linear]):
                if bn_info[index] and bn_info[index] not in bn_picked_for_folding:
                    pairs_list.append((conv_linear, bn_info[index]) if conv_is_first else (bn_info[index], conv_linear))
                    bn_picked_for_folding.add(bn_info[index])

        return pairs_list

    conv_bn_pairs = get_pairs(conv_is_first=True)
    bn_conv_pairs = get_pairs(conv_is_first=False)

    return conv_bn_pairs, bn_conv_pairs, bn_picked_for_folding


def fold_all_batch_norms(model: tf.keras.Model) \
        -> Tuple[List[Tuple[LayerType, BatchNormType]], tf.keras.Model]:
    """
    Fold all batch_norm layers in a model into corresponding conv/linear layers

    :param model: model to find all batch norms for

    :return: A tuple of List of conv/linear layers with associated bn op / activation info
        and a new model with the Batch Normalization layers folded
    """

    conv_bn_pairs, bn_conv_pairs, folded_bns = _find_all_batch_norms_to_fold(model)

    # Potential new model is returned in case the model is a functional model
    potential_new_model = _fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)
    model = potential_new_model if potential_new_model else model

    # Convert the standalone BNs which are not folded
    bn_converted = convert_standalone_batchnorms(model, folded_bns)
    if bn_converted:
        _logger.info("%d BatchNorms' weights got converted", len(bn_converted))
        model.compile()
    _logger.warning("A new model is returned with the Batch Normalization layers removed for Keras models. "
                    "Please use this new model for the rest of the AIMET flow.")

    return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs], model


def convert_standalone_batchnorms(model: tf.keras.Model, folded_bns: set) -> List[tf.keras.layers.BatchNormalization]:
    """
    Converts the weights of standalone batch norms remaining in the model after BN folding
    :param model: keras model on which batch norm folding is being performed
    :param folded_bns: list of batch norms which got folded
    :return: list of BatchNorms whose weights is converted
    """

    bn_converted = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization) and layer not in folded_bns:
            convert_batchnorm_parameters(layer)
            _logger.debug("%s weights got converted", layer.name)
            bn_converted.append(layer)
    return bn_converted


def convert_batchnorm_parameters(bn: tf.keras.layers.BatchNormalization):
    """
    Convert the weights of BN such that it works as y = weights * x + bias
    :param bn: Batch Norm layer whose weights need to be converted
    """
    bn_params = _get_bn_params(bn)

    # inv ::  1/ Sqrt(var + eps)
    inv = tf.math.rsqrt(bn.moving_variance.numpy() + bn.epsilon)
    weight = np.array(bn_params.gamma) * np.array(inv)
    bias = np.array(bn_params.beta) - np.array(bn_params.runningMean) * weight

    new_bn_weights = [weight.data, bias.data,
                      np.zeros(shape=bn.moving_mean.shape, dtype=np.float32),
                      np.ones(shape=bn.moving_variance.shape, dtype=np.float32)]
    bn.trainable = False
    bn.set_weights(new_bn_weights)
    bn.epsilon = 0


# pylint: disable=protected-access
def fold_all_batch_norms_to_scale(sim: QuantizationSimModel) -> List[Tuple[QcQuantizeWrapper, QcQuantizeWrapper]]:
    """
    Fold all batch_norm layers in a model into the quantization scale parameter
    of the corresponding conv layers

    :param sim: QuantizationSimModel to be folded
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """

    assert sim.model is not None, "QuantizationSimModel attribute 'model' is None."

    model = sim._model_without_wrappers

    quant_wrappers = {
        quant_wrapper._layer_to_wrap: quant_wrapper
        for quant_wrapper in sim.quant_wrappers()
    }

    conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
    conv_bn_pairs = [
        (quant_wrappers[conv], quant_wrappers[bn]) for conv, bn in conv_bn_pairs
    ]
    bn_conv_pairs = [
        (quant_wrappers[bn], quant_wrappers[conv]) for bn, conv in bn_conv_pairs
    ]

    old_model_without_wrappers = tf.keras.models.clone_model(model)
    conv_bn_pairs_without_wrappers, _, _ = _find_all_batch_norms_to_fold(old_model_without_wrappers)
    old_model_without_wrappers.set_weights(WeightTensorUtils.get_all_sim_models_layer_to_wrap_weights(sim.model))

    # We fold both the sim.model and sim._model_without_wrappers because we rebuild the QuantizationSimModel during
    # export and this utilizes the sim._model_without_wrappers to achieve this.
    bn_fold_sim_model = _fold_given_batch_norms(sim.model, conv_bn_pairs, bn_conv_pairs)
    sim.model = bn_fold_sim_model if bn_fold_sim_model else sim.model

    bn_fold_model = _fold_given_batch_norms(old_model_without_wrappers, conv_bn_pairs_without_wrappers, [])
    sim._model_without_wrappers = bn_fold_model if bn_fold_model else old_model_without_wrappers

    return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]

def fold_given_batch_norms(model: tf.keras.Model, layer_pairs: List[PairType]) -> Optional[tf.keras.Model]:
    """
    Fold a given set of batch_norm layers into conv_linear layers

    :param model: Either a Keras Model or a QuantizationSimModel's model
    :param layer_pairs: Tuple of conv, bn layers and is_batch_norm_second flag
    :return: new model with batch norm layers folded if model is a functional model, else None
    """
    # pylint: disable=protected-access
    conv_bn_paris = []
    bn_conv_pairs = []

    def is_batchnorm(layer: tf.keras.layers.Layer) -> bool:
        if isinstance(layer, QcQuantizeWrapper):
            layer = layer._layer_to_wrap
        return isinstance(layer, _supported_batchnorms)

    def is_conv_linear(layer: tf.keras.layers.Layer) -> bool:
        if isinstance(layer, QcQuantizeWrapper):
            layer = layer._layer_to_wrap
        return isinstance(layer, _supported_layers)

    for x, y in layer_pairs:
        if is_batchnorm(x):
            assert is_conv_linear(y)
            bn = x
            conv = y
            bn_conv_pairs.append((bn, conv))
        else:
            assert is_conv_linear(x)
            assert is_batchnorm(y)
            conv = x
            bn = y
            conv_bn_paris.append((conv, bn))

    return _fold_given_batch_norms(model, conv_bn_paris, bn_conv_pairs)

def _fold_given_batch_norms(model: tf.keras.Model,
                            conv_bn_pairs: Iterable[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]],
                            bn_conv_pairs: Iterable[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]]) -> \
                                Optional[tf.keras.Model]:
    """
    Fold a given set of batch_norm layers into conv layers

    :param model: Model
    :param conv_bn_pairs: List of (conv, bn) pairs to fold
    :param bn_conv_pairs: List of (bn, conv) pairs to fold
    """
    for bn, conv in bn_conv_pairs:
        if isinstance(conv, QcQuantizeWrapper):
            raise RuntimeError(f"Forward folding to scale is not possible. Got {conv.name}")

    bn_layers = []

    def _fold(conv, bn, fold_backward):
        is_wrapped = isinstance(conv, QcQuantizeWrapper) or isinstance(bn, QcQuantizeWrapper)
        try:
            if is_wrapped:
                assert isinstance(conv, QcQuantizeWrapper) and isinstance(bn, QcQuantizeWrapper)
                bn._layer_to_wrap.trainable = False
                _fold_to_scale(conv, bn)
                bn_layers.append(bn._layer_to_wrap)
            else:
                bn.trainable = False
                _fold_to_weight(conv, bn, fold_backward=fold_backward)
        except _BatchNormFoldingNotSupported as e:
            bn_name = bn._layer_to_wrap.name if is_wrapped else bn.name
            conv_name = conv._layer_to_wrap.name if is_wrapped else conv.name
            _logger.warning(
                "Failed to fold %s to %s. [Reason] %s", bn_name, conv_name, str(e)
            )
        else:
            bn_layers.append(bn._layer_to_wrap if is_wrapped else bn)

    for conv, bn in conv_bn_pairs:
        _fold(conv, bn, fold_backward=True)

    for bn, conv in bn_conv_pairs:
        _fold(conv, bn, fold_backward=False)

    return _delete_all_bns_from_model(model, bn_layers)

class _BatchNormFoldingNotSupported(RuntimeError):
    pass

def _fold_to_scale(conv_wrapper: QcQuantizeWrapper, bn_wrapper: QcQuantizeWrapper):
    """
    Fold BatchNorm into the scale and bias of the given layer.

    :param conv_wrapper: QcQuantizeWrapper that wraps conv or linear layer
    :param bn_wrapper: QcQuantizeWrapper that wraps the Batch Norm layer
    """
    # pylint: disable=protected-access, too-many-statements, too-many-locals
    conv = conv_wrapper._layer_to_wrap
    bn = bn_wrapper._layer_to_wrap

    weight_quantizer = get_wrappers_weight_quantizer(conv_wrapper.param_quantizers)
    bias_quantizer   = get_wrappers_bias_quantizer(conv_wrapper.param_quantizers)

    # Checking QuantScheme as aimet_tensorflow.keras does not have LearnedGridTensorQuantizer
    if weight_quantizer.quant_scheme not in [QuantScheme.training_range_learning_with_tf_init,
                                             QuantScheme.training_range_learning_with_tf_enhanced_init]:
        raise _BatchNormFoldingNotSupported(
            "BatchNorm folding to scale supports training_range_learning_with_tf_init or "
            "training_range_learning_with_tf_enhanced_init only. "
            f"got {weight_quantizer.quant_scheme}"
        )

    output_quantizer = conv_wrapper.output_quantizers[0]

    if output_quantizer.is_enabled():
        raise _BatchNormFoldingNotSupported(
            "BatchNorm should belong to the same supergroup with the layer to be folded to."
        )

    if bias_quantizer:
        if bias_quantizer.is_enabled():
            raise _BatchNormFoldingNotSupported(
                "Can't fold BatchNorm to scale if bias quantizer is enabled."
            )

    enc_min = weight_quantizer._encoding_min
    enc_max = weight_quantizer._encoding_max

    if not weight_quantizer.is_encoding_valid():
        raise RuntimeError

    with bn_wrapper._quantize_params():
        _fold_to_weight(conv, bn, fold_backward=True)

        gamma = bn.gamma
        sigma = K.sqrt(bn.moving_variance + bn.epsilon)

        for i, c in enumerate(gamma/sigma):
            c = float(c)
            if c >= 0:
                enc_max[i].assign(enc_max[i] * c)
                enc_min[i].assign(enc_min[i] * c)
            else:
                enc_max_before_reassign = enc_max[i]
                enc_max[i].assign(enc_min[i] * c)
                enc_min[i].assign(enc_max_before_reassign * c)


    # Copy batchnorm's output quantizers to conv output quantizers
    for conv_output_quantizer, bn_output_quantizer in \
            zip(conv_wrapper.output_quantizers, bn_wrapper.output_quantizers):
        if bn_output_quantizer.encoding is not None:
            conv_output_quantizer._encoding_min.assign(bn_output_quantizer._encoding_min)
            conv_output_quantizer._encoding_max.assign(bn_output_quantizer._encoding_max)
            conv_output_quantizer._is_encoding_valid = True

            tensor_quantizers = conv_output_quantizer._tensor_quantizer if isinstance(conv_output_quantizer._tensor_quantizer, List) else [conv_output_quantizer._tensor_quantizer]
            for tensor_quantizer in tensor_quantizers:
                tensor_quantizer.isEncodingValid = True

        if bn_output_quantizer.is_enabled():
            conv_output_quantizer.enable()
        else:
            conv_output_quantizer.disable()



        bn_output_quantizer.disable()

    if bias_quantizer is None:
        bias_quantizer = ParamPerTensorQuantizer(conv,
                                                 conv.bias.name.split(':')[0],
                                                 weight_quantizer.quant_scheme,
                                                 MAP_PYMO_TO_ROUND_MODE[weight_quantizer.round_mode],
                                                 weight_quantizer.bitwidth,
                                                 weight_quantizer.data_type,
                                                 weight_quantizer.is_symmetric,
                                                 weight_quantizer.use_strict_symmetric,
                                                 weight_quantizer.use_unsigned_symmetric,
                                                 enabled=False)
        tensor_quantizers = bias_quantizer._tensor_quantizer if isinstance(bias_quantizer._tensor_quantizer, List) else [bias_quantizer._tensor_quantizer]
        for tensor_quantizer in tensor_quantizers:
            tensor_quantizer.isEncodingValid = True
        conv_wrapper.param_quantizers.append(bias_quantizer)


def _fold_to_weight(conv_linear: LayerType, bn: BatchNormType, fold_backward: bool):
    """
    Fold BatchNorm into the weight and bias of the given layer.

    :param conv_linear: Conv or linear layer to fold BN into.
    :param bn: BatchNorm to fold.
    :param fold_backward: To fold backwards or not
    """

    is_bias_valid = conv_linear.bias is not None

    bn_params = _get_bn_params(bn)
    weight_tensor = _get_weight_tensor_transpose_reshape(conv_linear)
    bias_tensor = _get_bias_tensor(conv_linear)

    # Updated weight and bias
    bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, fold_backward)

    if isinstance(conv_linear, tf.keras.layers.DepthwiseConv2D):
        # Depthwise conv layers in TF have outputs(Noc) set to 1.
        # we send in format [Nic, Noc, kh, kw]
        numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))

    elif isinstance(conv_linear, tf.keras.layers.Dense):
        # o, i - convert to i , o
        numpy_weight_reshaped = np.reshape(
            weight_tensor.data,
            [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose(1, 0)

    elif isinstance(conv_linear, tf.keras.layers.Conv2DTranspose):
        # we sent in format [Noc, Nic, kh, kw]
        numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))

    else:
        # conv2D case
        # we sent in format [Noc, Nic, kh, kw]
        numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 1, 0))

    # update bias tensor, even in case there was no existing bias add op in given conv2D op.
    bias_tensor_shape = [weight_tensor.shape[0]]
    numpy_bias_reshaped = np.reshape(bias, bias_tensor_shape)

    if not is_bias_valid:
        conv_linear.use_bias = True
        conv_linear.bias = conv_linear.add_weight(name=f"{conv_linear.name}/bias",
                                                  shape=(weight_tensor.shape[0],),
                                                  dtype=conv_linear.dtype,
                                                  trainable=True)
    conv_linear.set_weights([numpy_weight_reshaped.data, numpy_bias_reshaped])
