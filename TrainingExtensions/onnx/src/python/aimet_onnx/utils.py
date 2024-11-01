# -*- mode: python -*-
# =============================================================================
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
# =============================================================================
""" Utility functions for ONNX """
import copy
import itertools
from typing import Dict, List, Union, Tuple
import os
import pickle
import numpy as np
import onnx
from onnx import helper, numpy_helper, mapping

from aimet_common.utils import AimetLogger
from packaging import version

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, TensorProto, ModelProto, GraphProto, ValueInfoProto
else:
    from onnx.onnx_pb import NodeProto, TensorProto, ModelProto, GraphProto, ValueInfoProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


OP_TYPES_WITH_PARAMS = ['Conv', 'Gemm', 'ConvTranspose', 'BatchNormalization', 'MatMul', 'Transpose',
                        'InstanceNormalization', 'LayerNormalization', 'GroupNormalization', 'RNN', 'LSTM', 'GRU']


def remove_nodes_with_type(node_type: str, onnx_graph: onnx.GraphProto):
    """
    Remove specific type of nodes from graph

    :param node_type: string, type of node to be removed
    :param onnx_graph: onnx graph to modify

    """
    input_output_pairs = {}

    onnx_graph_node_list = copy.deepcopy(onnx_graph.node)

    for node in onnx_graph_node_list:
        if node.op_type == node_type:
            input_output_pairs[node.output[0]] = node.input[0]
            onnx_graph.node.remove(node)
    for node in onnx_graph.node:
        for i in range(len(node.input)):
            if node.input[i] in input_output_pairs.keys():
                node.input[i] = input_output_pairs[node.input[i]]
        for outputs in onnx_graph.output:
            if outputs.name in input_output_pairs.keys() and \
                    node.output[0] == input_output_pairs[outputs.name]:
                node.output[0] = outputs.name


def _prune_unused_initializer(graph: onnx.GraphProto, init_name):
    """
    Remove initializer from graph if it is unused
    """
    for node in graph.node:
        if init_name in node.input:
            return # Don't prune if initializer is still used

    for initializer in graph.initializer:
        if initializer.name == init_name:
            graph.initializer.remove(initializer)
            break


def remove_node(node: NodeProto, onnx_graph: onnx.GraphProto):
    """
    Remove a specific node from graph along with associated initializers

    :param node: the node to be removed
    :param onnx_graph: onnx graph to modify

    """
    onnx_graph.node.remove(node)
    for other_node in onnx_graph.node:
        if other_node.input and other_node.output:
            # Check if other node takes input from removed node
            for idx in range(len(other_node.input)):
                if other_node.input[idx] == node.output[0]:
                    other_node.input[idx] = node.input[0]
            # Check if removed node output is an output of the graph
            for outputs in onnx_graph.output:
                if outputs.name == node.output[0] and other_node.output[0] == node.input[0]:
                    other_node.output[0] = outputs.name

    for input_name in node.input:
        _prune_unused_initializer(onnx_graph, input_name)


def transpose_tensor(t: TensorProto, axes: Union[List, Tuple]) -> TensorProto:
    """
    Permutes the axes of a given array using numpy.transpose

    :param t: tensor to transpose
    :param axes: tuple or list containing the permuted axis ordering
    :return: t permuted according to the axis ordering in axes
    """
    t_np = numpy_helper.to_array(t)
    # Expand tensor with singleton dimensions to match axes
    while len(t_np.shape) < len(axes):
        t_np = np.expand_dims(t_np, len(t_np.shape))
    return numpy_helper.from_array(np.transpose(t_np, axes), name=t.name)


def replace_node_with_op(node_type: str, new_type: str, onnx_graph: onnx.GraphProto):
    """
    Replace the given op type of nodes to new op type

    :param node_type: string, type of node to be replaced
    :param new_type: string, type of node to substitute for
    :param onnx_graph: onnx graph to modify

    """
    for node in onnx_graph.node:
        if node.op_type == node_type:
            node.op_type = new_type


def get_node_attribute(node: NodeProto, name: str):
    """
    Return the value of a node's attribute specified by its name

    :param node: NodeProto object to retrieve the attribute from
    :param name: string containing the name of the attribute to retrieve
    :return: value of the attribute
    """
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return None


def get_weights(name: str, onnx_graph: onnx.GraphProto) -> bytes:
    """
    Return the weights by given name
    :param name, name of the weights to find
    :param onnx_graph, onnx graph to find the corresponding weight data
    :return onnx tensor
    """
    for param in onnx_graph.initializer:
        if param.name == name:
            return param.raw_data
    assert Exception("Couldn't find weights by the given name")
    return None


def get_ordered_dict_of_nodes(onnx_graph: onnx.GraphProto) -> Dict:
    """
    Return the ordered list of nodes

    :param onnx_graph: onnx graph to provide node info
    :return dict of ordered nodes with name as key

    """
    ordered_dict = {}
    for node in onnx_graph.node:
        ordered_dict[node.name] = node
    return ordered_dict


def make_dummy_input(model: ModelProto, dynamic_size: int = 1) -> Dict[str, np.ndarray]:
    """
    Create a dummy input based on the model input types and shapes

    :param model: Model to create an input for
    :param dynamic_size: Dimension size to use for dynamic axes
    :return: Dictionary of input_name : input array
    """
    input_dict = {}
    for item in model.graph.input:
        name = item.name
        dtype = item.type.tensor_type.elem_type
        shape = []
        for dim in item.type.tensor_type.shape.dim:
            if dim.dim_param:
                # Evaluates true if axis is dynamic. We set the size of dynamic axes to dynamic_size
                shape.append(dynamic_size)
            else:
                # Else, axis has a fixed dimension size stored in dim.dim_value
                shape.append(dim.dim_value)
        if shape:
            input_dict[name] = np.random.randn(*shape).astype(mapping.TENSOR_TYPE_TO_NP_TYPE[dtype])
        else:
            input_dict[name] = np.array(np.random.randn(*shape)).astype(mapping.TENSOR_TYPE_TO_NP_TYPE[dtype])
    return input_dict


def replace_relu6_with_relu(model: ModelProto):
    """
    Replace relu6 op with relu op

    :param model: ONNX model
    """
    for node in model.model.graph.node:
        if node.op_type == 'Clip' and check_if_clip_node_minimum_is_zero(node, model):
            parent_node = None
            child_node = None
            for temp_node in model.model.graph.node:
                if node.input[0] in temp_node.output:
                    parent_node = temp_node
                if node.output[0] in temp_node.input:
                    child_node = temp_node
            assert parent_node, "Parent Node for Clip operation does not exist"
            if parent_node.op_type in ['Conv', 'ConvTranspose'] and child_node and \
                    child_node.op_type in ['Conv', 'ConvTranspose']:
                name = node.name
                remove_node(node, model.model.graph)
                inputs = [parent_node.output[0]]
                model.replace_input_of_all_nodes(parent_node.output[0], parent_node.output[0] + '_replaced')
                relu_node = onnx.helper.make_node(
                    op_type="Relu",
                    inputs=inputs,
                    outputs=[parent_node.output[0] + '_replaced'],
                    name='Relu_' + name,
                )

                model.add_node(relu_node)
    model.topological_sort()


def check_if_clip_node_minimum_is_zero(node: NodeProto, model: ModelProto):
    """
    Check if the clip node's minimum is 0

    :param node: ONNX node
    :param model: ONNX model
    """
    if len(node.input) == 3:
        input_node = node.input[1]
        for node_graph in model.model.graph.node:
            if node_graph.output[0] == input_node:
                if hasattr(node_graph, "attribute") and hasattr(node_graph.attribute[0], "t") and \
                        numpy_helper.to_array(node_graph.attribute[0].t) == 0:
                    return True
    elif hasattr(node, "attribute") and node.attribute[1].name == "min" and node.attribute[1].f == 0.0:
        return True
    return False


def add_hook_to_get_activation(model: ModelProto, name: str) -> ValueInfoProto:
    """
    Adds a given activation to the model output
    :param model: The model to add the hook to
    :param name: The name of the activation
    :return: ValueInfoProto for the given activation that has been appended to model.graph.output
    """
    val_info = onnx.helper.ValueInfoProto()
    val_info.name = name
    model.graph.output.append(val_info)
    return val_info


def remove_activation_hooks(model: ModelProto,
                            hooks: Union[List[ValueInfoProto], ValueInfoProto]):
    """
    Removes activation hooks from the model output
    :param model: The model from which to remove the hooks
    :param hooks: Value info or list of value infos to remove from the model output
    """
    if not isinstance(hooks, List):
        hooks = [hooks]
    for hook in hooks:
        model.graph.output.remove(hook)


def get_graph_intermediate_activations(graph: GraphProto) -> List[str]:
    """
    Returns the names of all activations within a graph that are used as the input to another node
    :param graph: The graph for which to retrieve the activations
    :return: A list containing the names of all found activations
    """
    param_names = []
    for param in graph.initializer:
        if param.name not in param_names and param.name:
            param_names.append(param.name)
    activation_names = []
    for node in graph.node:
        for name in node.input:
            if name not in activation_names and name not in param_names and name:
                activation_names.append(name)
    return activation_names


class ParamUtils:
    """ Param utilities """
    @staticmethod
    def get_shape(model: ModelProto, node: NodeProto, param_index: int) -> List:
        """
        Returns a list of shape for the param specifies
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        param = ParamUtils.get_param(model, node, param_index)
        if param:
            return param.dims
        return None

    @staticmethod
    def get_param(model: ModelProto, node: NodeProto, param_index: int) -> TensorProto:
        """
        Returns the param tensor
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        def find_param_in_model_initializers(param_name: str, model: ModelProto):
            for param in model.graph.initializer:
                if param.name == param_name:
                    return param
            return None

        def find_param_in_model_constants(param_name: str, model: ModelProto):
            for node in model.graph.node:
                if node.op_type == 'Constant' and param_name in node.output:
                    for attribute in node.attribute:
                        if attribute.name == 'value':
                            param = attribute.t
                            param.name = param_name
                            return param
                if node.op_type == 'Identity' and param_name == node.output[0]:
                    return ParamUtils.get_param(model, node, 0)
            return None

        if len(node.input) >= param_index + 1:
            param_name = node.input[param_index]
            param = find_param_in_model_initializers(param_name, model)
            if param is None:
                param = find_param_in_model_constants(param_name, model)
            return param
        return None


def get_product_name_from_quantized_name(quantized_name: str):
    """
    Gets product's name from quantized name
    :param quantized_name: Quantized name
    """
    if '_updated' in quantized_name:
        return quantized_name[:quantized_name.index('_updated')]
    if '_qdq' in quantized_name:
        return quantized_name[:quantized_name.index('_qdq')]
    # If there is no quantizer added then return None
    return None


def retrieve_constant_input(node: NodeProto, model: ModelProto, index: int
                            ) -> Tuple[TensorProto, bool]:
    """
    Retrieves node input at the specified index if the input has a corresponding initializer in model.graph.initializer
    and is separated from node by no more than one Transpose operation.
    :param node: The node to find the input for
    :param model: The model to which the node belongs
    :param index: The index of the desired input within node.input
    :return: Tuple containing the input parameter and a bool specifying whether the param is transposed before entering
             the node
    """
    weight_input = node.input[index]
    transposed = False
    weight = ParamUtils.get_param(model, node, index)
    if not weight:
        # Check if the weight is transposed before entering the node
        for other_node in model.graph.node:
            if weight_input in other_node.output and other_node.op_type == "Transpose":
                weight = ParamUtils.get_param(model, other_node, 0)
                transposed = True
    return weight, transposed


class CachedDataset:
    """
    Cache number of batches from the data loader at given path location and
    provide interface to fetch single batch of model inputs.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, data_loader, num_batches: int, path: str):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to fetch from data loader
        :param path: Path to save model inputs
        """
        if len(data_loader) < num_batches:
            raise ValueError(f'Can not fetch {num_batches} batches from '
                             f'a data loader of length {len(data_loader)}.')

        self._num_batches = num_batches
        self._path = path

        self._cache_model_inputs(itertools.islice(data_loader, num_batches))

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index: int):
        path = os.path.join(self._path, 'model_inputs_' + str(index))

        with open(path, 'rb') as file:
            batch = pickle.load(file)

        return batch

    def _cache_model_inputs(self, data_loader):
        """
        Function to cache number of batches individually in separate file at provided path location
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        for i, batch in enumerate(data_loader):
            path = os.path.join(self._path, f'model_inputs_{i}')
            with open(path, 'wb') as file:
                pickle.dump(batch, file)

        logger.info('Caching %d batches from data loader at path location: %s', self._num_batches, self._path)


def create_input_dict(model: ModelProto, input_batch: Union[Dict, np.ndarray, List[np.ndarray], Tuple[np.ndarray]]) -> Dict:
    """
    Creates input dictionary (input name to input value map) for session.run

    :param model: ONNX model
    :param input_batch: either a dict, single numpy array, list or tuple of numpy array
    :return: input dictionary
    """
    if isinstance(input_batch, dict):
        return input_batch

    input_names = [input.name for input in model.graph.input]

    # single input
    if isinstance(input_batch, np.ndarray):
        input_batch_list = [input_batch]

    # list of multiple inputs
    elif isinstance(input_batch, list):
        input_batch_list = input_batch

    # tuple of multiple inputs
    elif isinstance(input_batch, tuple):
        input_batch_list = list(input_batch)

    else:
        raise ValueError('Input batch should be either dict, numpy array, list or tuple')

    if not len(input_names) == len(input_batch_list):
        raise ValueError('There is mismatch between number of input names and input tensors')

    return dict(zip(input_names, input_batch_list))
