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
""" ONNX Code to fold batch-norm layers """

from typing import Dict, List, Tuple
import contextlib
import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

# pylint: disable=wrong-import-order
from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.connected_graph.connectedgraph_utils import get_ordered_ops
import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.connectedgraph import WEIGHT_INDEX, BIAS_INDEX, RUNNING_MEAN_INDEX, RUNNING_VAR_INDEX
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import get_node_attribute, remove_node, transpose_tensor, ParamUtils, retrieve_constant_input

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, TensorProto, ModelProto
else:
    from onnx.onnx_pb import NodeProto, TensorProto, ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)

ConvType = ['Conv', 'ConvTranspose']
LinearType = ['Gemm', 'MatMul']
BatchNormType = ['BatchNormalization']

class BNLayer:
    """ Captures beta and gamma parameter for BatchNorm layers to be used during High Bias absorption """
    def __init__(self):
        self.bn_layer = None
        self.gamma = None
        self.beta = None


def _find_conv_bn_pairs(connected_graph: ConnectedGraph) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param connected_graph: ConnectedGraph object.
    :return: dictionary of conv/linear Op with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    preceding_linear_op_types = ['Flatten', 'Reshape']

    # Linear layer combinations
    for linear_op in LinearType:
        for preceding_linear_op_type in preceding_linear_op_types:
            # BN -> Linear
            patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', preceding_linear_op_type, linear_op],
                                                       action=layer_select_handler))

    for op_type in ConvType + LinearType:
        patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', op_type],
                                                   action=layer_select_handler))
        patterns_with_callbacks.append(PatternType(pattern=[op_type, 'BatchNormalization'],
                                                   action=layer_select_handler))

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict


def find_all_batch_norms_to_fold(connected_graph: ConnectedGraph,
                                 ) -> Tuple[List[Tuple[NodeProto, NodeProto]],
                                            List[Tuple[NodeProto, NodeProto]]]:
    """
    Find all possible batch norm layers that can be folded. Returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param connected_graph: connected graph model to search
    :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
             where `bn` can be folded into to `layer`.
    """
    conv_linear_bn_activation_info_dict = _find_conv_bn_pairs(connected_graph)
    model = connected_graph.model
    # To mark BN's already picked for backward folding
    bn_picked_for_folding = set()

    ordered_conv_fc_nodes = get_ordered_conv_linears(connected_graph)

    conv_bn_pairs = []
    # Backward fold is given priority over Forward fold
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                if is_valid_bn_fold(node.get_module(), model, True):
                    conv_bn_pairs.append((node.get_module(), bn_info.output_bn.get_module()))
                    bn_picked_for_folding.add(bn_info.output_bn)
                else:
                    logger.info('...... invalid combination to fold %s', [node.name, bn_info.output_bn.name])

    bn_conv_pairs = []
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                if is_valid_bn_fold(node.get_module(), model, False):
                    bn_conv_pairs.append((bn_info.input_bn.get_module(), node.get_module()))
                    bn_picked_for_folding.add(bn_info.input_bn)
                else:
                    logger.info('...... invalid combination to fold %s', [bn_info.input_bn.name, node.name])

    return conv_bn_pairs, bn_conv_pairs


def get_ordered_conv_linears(conn_graph: ConnectedGraph) -> List[Op]:
    """
    helper to select a list of candidate layers for BatchNorm folding
    :param conn_graph: connected graph to search
    :return: List of conv/linear layers
    """
    # get ordered operations list from the connected graph
    list_of_ordered_ops = get_ordered_ops(conn_graph.starting_ops)

    # look for conv/linear layers
    ordered_convs = []
    for op in list_of_ordered_ops:
        if op.type in ConvType + LinearType:
            ordered_convs.append(op)
    return ordered_convs


def is_valid_bn_fold(conv_linear: NodeProto, model: ModelProto, fold_backward: bool) -> bool:
    """
    Determine if a given layer can successfully absorb a BatchNorm given the layer type and parameters
    :param conv_linear: The Conv/Linear layer to fold a BatchNorm into.
    :param model: The model to which the Conv/Linear layer belongs.
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: True if a BatchNorm layer can be folded without causing output error.
    """
    valid = True
    if conv_linear.op_type in LinearType:
        # Check if this is actually a fully connected layer or a dynamic matmul
        w = retrieve_constant_input(conv_linear, model, WEIGHT_INDEX)[0]
        if w is None:
            valid = False
    if not fold_backward:
        # Cannot fold BN -> Conv with padding. AIMET does not support forward folding to grouped or DW Conv
        if conv_linear.op_type == 'Conv':
            valid &= all(item == 0 for item in get_node_attribute(conv_linear, "pads"))
            valid &= get_node_attribute(conv_linear, "group") == 1
        # AIMET does not support forward folding to ConvTranspose
        elif conv_linear.op_type == 'ConvTranspose':
            valid = False
    else:
        # AIMET does not support backwards folding to grouped ConvTranspose
        if conv_linear.op_type == 'ConvTranspose':
            valid &= get_node_attribute(conv_linear, "group") in (1, get_input_output_channels(conv_linear, model)[0])
    return valid


def fold_all_batch_norms_to_weight(model: ModelProto) -> [List]:
    """
    Fold all possible batch_norm layers in a model into the weight of the corresponding conv layers

    :param model: onnx Model to perform BN fold on
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    if isinstance(model, ONNXModel):
        model = model.model
    connected_graph = ConnectedGraph(model)
    model = connected_graph.model
    conv_bn_pairs, bn_conv_pairs = find_all_batch_norms_to_fold(connected_graph)
    conv_bns = []
    bn_convs = []
    for conv, bn in conv_bn_pairs:
        bn_layer = _fold_to_weight(model, conv, bn, True)
        conv_bns.append((conv, bn_layer))
        remove_node(bn, model.graph)

    for bn, conv in bn_conv_pairs:
        bn_layer = _fold_to_weight(model, conv, bn, False)
        bn_convs.append((conv, bn_layer))
        remove_node(bn, model.graph)

    _update_standalone_batchnorm_ops(model)

    return conv_bns, bn_convs


def _fold_to_weight(model: ModelProto,
                    conv_linear: NodeProto,
                    bn: NodeProto,
                    fold_backward: bool):
    """
    Fold BatchNorm into the weight and bias of the given layer.

    :param model: onnx model to which the conv/bn pair belong
    :param conv_linear: Conv or linear layer to fold BN into.
    :param bn: BatchNorm to fold.
    :param fold_backward: True if the BatchNorm comes after the Conv
    """
    # Must convert MatMul layers to Gemm to allow bias
    if conv_linear.op_type == "MatMul":
        _matmul_to_gemm(conv_linear, model)

    weight = ParamUtils.get_param(model, conv_linear, WEIGHT_INDEX)
    bias = ParamUtils.get_param(model, conv_linear, BIAS_INDEX)
    groups = get_node_attribute(conv_linear, "group")

    # If layer doesn't have bias, create a bias initializer and add it to the model, then retrieve it
    if not bias:
        bias_data = np.zeros(get_input_output_channels(conv_linear, model)[1])
        bias_name = conv_linear.name + ".bias"
        bias = numpy_helper.from_array(bias_data.astype(np.float32), name=bias_name)
        model.graph.initializer.append(bias)
        conv_linear.input.append(bias_name)
        bias = ParamUtils.get_param(model, conv_linear, BIAS_INDEX)

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    # if conv_linear.type == "ConvTranspose" and conv_linear groups == 1:
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight = transpose_tensor(weight, (1, 0, 2, 3))
    # Gemm layers may or may not need to have weights transposed depending on value of transB attribute
    elif conv_linear.op_type in LinearType and not get_node_attribute(conv_linear, "transB"):
        weight = transpose_tensor(weight, (1, 0))

    channels = weight.dims[0] if fold_backward else weight.dims[1]
    bn_param = get_bn_params(model, bn, channels)
    bn_layer = copy_bn_params_to_bn_layer(bn, bn_param)

    _call_mo_batch_norm_fold(weight, bias, bn_param, fold_backward=fold_backward)

    # Transpose weight back to original configuration
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight = transpose_tensor(weight, (1, 0, 2, 3))
    elif conv_linear.op_type in LinearType and not get_node_attribute(conv_linear, "transB"):
        weight = transpose_tensor(weight, (1, 0))

    weight_param = ParamUtils.get_param(model, conv_linear, WEIGHT_INDEX)
    weight_param.raw_data = weight.raw_data
    return bn_layer


def _matmul_to_gemm(node: NodeProto, model: ModelProto):
    """
    Convert MatMul node to Gemm and initialize bias to zeros

    :param node: MatMul node to convert to Gemm
    :param model: model to which the node belongs
    """
    assert node.op_type == "MatMul"

    weight, transposed = retrieve_constant_input(node, model, WEIGHT_INDEX)
    if transposed:
        node.input[WEIGHT_INDEX] = weight.name
        model.graph.initializer.remove(weight)
        weight = transpose_tensor(weight, (1, 0))
        model.graph.initializer.append(weight)
    node.op_type = "Gemm"
    node.name = node.name.replace("MatMul", "Gemm")
    # Create bias vector for Gemm operation
    bias_name = node.name + ".bias"
    bias_data = np.zeros(weight.dims[1])
    bias = numpy_helper.from_array(bias_data.astype(np.float32), name=bias_name)
    model.graph.initializer.append(bias)
    node.input.append(bias_name)


def _call_mo_batch_norm_fold(weight: TensorProto,
                             bias: TensorProto,
                             bn_params: libpymo.BNParams,
                             fold_backward: bool):
    """
    Calls C++ batch norm folding API.

    :param weight: Weight or scale tensor to fold BN into.
    :param bias: Bias tensor to fold BN into.
    :param bn_params: Batch Norm layer
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    """
    weight_tensor = libpymo.TensorParams()

    weight_tensor.data = numpy_helper.to_array(weight).reshape(-1)
    weight_tensor.shape = np.array(weight.dims)

    bias_tensor = libpymo.TensorParams()

    bias_tensor.data = numpy_helper.to_array(bias).reshape(-1)
    bias_tensor.shape = np.array(bias.dims)
    is_bias_valid = True

    with _expand_shape_to_4d(weight_tensor):
        _bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, fold_backward)

    bias.raw_data = np.asarray(_bias, dtype=np.float32).tobytes()
    weight.raw_data = np.asarray(weight_tensor.data, dtype=np.float32).tobytes()


def get_bn_params(model: ModelProto, bn: NodeProto, channels: int) -> libpymo.BNParams:
    """
    Returns the populated libpymo.BNParams object for the given BatchNormalization layer with
    parameters repeated if necessary.

    :param model: model to which the bn layer belongs
    :param bn: BatchNormalization layer to retrieve the parameters from
    :param channels: The effective number of channels the BatchNorm layer operates on (needed for Gemm layers)
    :return: libpymo.BNParams object for the input BatchNorm layer
    """
    bn_params = libpymo.BNParams()
    gamma = numpy_helper.to_array(ParamUtils.get_param(model, bn, WEIGHT_INDEX)).reshape(-1)
    # In the case of BatchNorm2d -> Flatten -> Gemm, must resize the BN parameters to the Gemm input feature length
    resize = channels / len(gamma)
    bn_params.gamma = np.repeat(gamma, resize)
    bn_params.beta = np.repeat(numpy_helper.to_array(ParamUtils.get_param(model, bn, BIAS_INDEX)).reshape(-1), resize)
    bn_params.runningMean = np.repeat(
        numpy_helper.to_array(ParamUtils.get_param(model, bn, RUNNING_MEAN_INDEX)).reshape(-1), resize)
    runningVar = numpy_helper.to_array(ParamUtils.get_param(model, bn, RUNNING_VAR_INDEX))

    epsilon = get_node_attribute(bn, "epsilon")
    if epsilon is None:
        epsilon = 1e-5 # Default onnx epsilon value
    sigma = np.sqrt(runningVar + epsilon)
    bn_params.runningVar = np.repeat(sigma.reshape(-1), resize)

    return bn_params


def copy_bn_params_to_bn_layer(bn: NodeProto, bn_params: libpymo.BNParams) -> BNLayer:
    """
    Copies bn params to a BN layer which can be used later by High bias absorption
    :param bn: BN layer
    :param bn_params: libpymo.BNParams object for the BatchNorm layer
    :return BNLayer object
    """
    bn_layer = BNLayer()
    bn_layer.bn_layer = bn
    bn_layer.gamma = bn_params.gamma
    bn_layer.beta = bn_params.beta
    return bn_layer


@contextlib.contextmanager
def _expand_shape_to_4d(weight_tensor: libpymo.TensorParams):
    """ Expand the shape of the weight into 4d.  """
    dims = len(weight_tensor.shape)

    if dims > 4:
        raise RuntimeError

    if dims == 4:
        yield weight_tensor

    else:
        orig_shape = weight_tensor.shape
        _4d_shape = np.append(orig_shape, [1 for _ in range(4-dims)]).astype(int)
        try:
            weight_tensor.shape = _4d_shape
            yield weight_tensor
        finally:
            weight_tensor.shape = orig_shape


def get_input_output_channels(node: NodeProto, model: ModelProto) -> Tuple[int, int]:
    """
    Find the input and output channels of a given layer.
    :param node: The node to find the input/output channels of
    :param model: The onnx model to which the layers belong
    :return: Tuple of (num channels in, num channels out)
    """
    weight = ParamUtils.get_param(model, node, WEIGHT_INDEX)
    groups = get_node_attribute(node, "group")
    # If group atttribute does not exist in the node,then default is 1
    if not groups:
        groups = 1
    if node.op_type == "Conv":
        num_in_channels = weight.dims[1] * groups
        num_out_channels = weight.dims[0]
    elif node.op_type == "ConvTranspose":
        num_in_channels = weight.dims[0]
        num_out_channels = weight.dims[1] * groups
    elif node.op_type == "Gemm":
        transB = get_node_attribute(node, "transB")
        if transB == 1:
            num_out_channels = weight.dims[0]
            num_in_channels = weight.dims[1]
        else:
            num_out_channels = weight.dims[1]
            num_in_channels = weight.dims[0]
    else:
        num_out_channels = None
        num_in_channels = None
    return num_in_channels, num_out_channels


# pylint: disable=too-many-locals
def _update_standalone_batchnorm_ops(model: ModelProto):
    """
    Update weight and bias of standalone batchnorm ops in the model.
    :param model: onnx Model for which batchnorm parameters are to be updated.
    """

    for node in model.graph.node:
        if node.op_type in BatchNormType:

            # get parameter names and indices
            weight_name, bias_name, running_mean_name, running_var_name = node.input[1:]
            init_w, init_b, init_rm, init_rv = [ParamUtils.get_param(model, node, idx) for idx in range(1, 5)]

            attr = [item for item in node.attribute if item.name == "epsilon"]
            if not attr:
                attr = onnx.helper.make_attribute("epsilon", 1e-5) # Default epsilon value
                node.attribute.append(attr)
            else:
                attr = attr[0]

            epsilon = attr.f
            tensor_w = numpy_helper.to_array(init_w)
            tensor_b = numpy_helper.to_array(init_b)
            tensor_rm = numpy_helper.to_array(init_rm)
            tensor_rv = numpy_helper.to_array(init_rv)

            # update values
            inv_sigma = np.reciprocal(np.sqrt(tensor_rv + epsilon))
            tensor_w = tensor_w * inv_sigma
            tensor_b = tensor_b - tensor_rm * tensor_w
            tensor_rm = np.zeros(tensor_w.shape, tensor_w.dtype)
            tensor_rv = np.ones(tensor_w.shape, tensor_w.dtype)
            attr.f = 0.

            init_w_ = numpy_helper.from_array(tensor_w, weight_name)
            init_b_ = numpy_helper.from_array(tensor_b, bias_name)
            init_rm_ = numpy_helper.from_array(tensor_rm, running_mean_name)
            init_rv_ = numpy_helper.from_array(tensor_rv, running_var_name)

            # update initializers
            init_w.CopyFrom(init_w_)
            init_b.CopyFrom(init_b_)
            init_rm.CopyFrom(init_rm_)
            init_rv.CopyFrom(init_rv_)
