# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""ONNX Code to fold batch-norm layers"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import onnx
import onnx_ir
import onnx_ir.passes.common
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_onnx.common.batch_norm_fold import batch_norm_fold, ConvBnPatternHandler
from aimet_onnx.common.graph_pattern_matcher import PatternType
from aimet_onnx.common.graph_searcher import GraphSearcher
from aimet_onnx.common.connected_graph.connectedgraph_utils import get_ordered_ops
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.onnx._utils import to_array

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.connectedgraph import (
    WEIGHT_INDEX,
    BIAS_INDEX,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
)
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import (
    get_node_attribute,
    remove_node,
    transpose_tensor,
    ParamUtils,
    retrieve_constant_input,
    find_shared_param_names,
    _tensor_from_array,
    _tensor_np_dtype,
)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, ModelProto
else:
    from onnx.onnx_pb import NodeProto, ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)

ConvType = ["Conv", "ConvTranspose"]
LinearType = ["Gemm", "MatMul"]
BatchNormType = ["BatchNormalization"]


def _has_shared_weight(node: Op, shared_weight_names: set) -> bool:
    """
    Check if a Conv/Linear node's weight tensor is shared with other nodes.

    :param node: Conv/Linear node to check
    :param shared_weight_names: Set of weight names that are shared
    :return: True if the node's weight is shared
    """
    for param, _ in node.parameters.values():
        if param.name in shared_weight_names:
            return True

    return False


class BNLayer:
    """Captures beta and gamma parameter for BatchNorm layers to be used during High Bias absorption"""

    def __init__(self, bn_layer=None, gamma=None, beta=None):
        self.bn_layer = bn_layer
        self.gamma = gamma
        self.beta = beta


def _find_conv_bn_pairs(connected_graph: ConnectedGraph) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param connected_graph: ConnectedGraph object.
    :return: dictionary of conv/linear Op with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    preceding_linear_op_types = ["Flatten", "Reshape"]

    # Linear layer combinations
    for linear_op in LinearType:
        for preceding_linear_op_type in preceding_linear_op_types:
            # BN -> Linear
            patterns_with_callbacks.append(
                PatternType(
                    pattern=["BatchNormalization", preceding_linear_op_type, linear_op],
                    action=layer_select_handler,
                )
            )

    for op_type in ConvType + LinearType:
        patterns_with_callbacks.append(
            PatternType(
                pattern=["BatchNormalization", op_type], action=layer_select_handler
            )
        )
        patterns_with_callbacks.append(
            PatternType(
                pattern=[op_type, "BatchNormalization"], action=layer_select_handler
            )
        )

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict


def find_all_batch_norms_to_fold(
    connected_graph: ConnectedGraph,
) -> Tuple[List[Tuple[NodeProto, NodeProto]], List[Tuple[NodeProto, NodeProto]]]:
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

    # Find weights that are shared between multiple Conv/Linear nodes
    shared_weight_names = find_shared_param_names(connected_graph)

    ordered_conv_fc_nodes = get_ordered_conv_linears(connected_graph)

    conv_bn_pairs = []
    # Backward fold is given priority over Forward fold
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict:
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                if _has_shared_weight(node, shared_weight_names):
                    logger.info(
                        "...... skipping fold due to shared weights %s",
                        [node.name, bn_info.output_bn.name],
                    )
                elif is_valid_bn_fold(node.get_module(), model, True):
                    conv_bn_pairs.append(
                        (node.get_module(), bn_info.output_bn.get_module())
                    )
                    bn_picked_for_folding.add(bn_info.output_bn)
                else:
                    logger.info(
                        "...... invalid combination to fold %s",
                        [node.name, bn_info.output_bn.name],
                    )

    bn_conv_pairs = []
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict:
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                if _has_shared_weight(node, shared_weight_names):
                    logger.info(
                        "...... skipping fold due to shared weights %s",
                        [bn_info.input_bn.name, node.name],
                    )
                elif is_valid_bn_fold(node.get_module(), model, False):
                    bn_conv_pairs.append(
                        (bn_info.input_bn.get_module(), node.get_module())
                    )
                    bn_picked_for_folding.add(bn_info.input_bn)
                else:
                    logger.info(
                        "...... invalid combination to fold %s",
                        [bn_info.input_bn.name, node.name],
                    )

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


def is_valid_bn_fold(
    conv_linear: NodeProto, model: ModelProto, fold_backward: bool
) -> bool:
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
        if conv_linear.op_type == "Conv":
            pads = get_node_attribute(conv_linear, "pads")
            if pads:
                valid &= all(item == 0 for item in pads)
            valid &= get_node_attribute(conv_linear, "group") in (None, 1)
        # AIMET does not support forward folding to ConvTranspose
        elif conv_linear.op_type == "ConvTranspose":
            valid = False
    else:
        # AIMET does not support backwards folding to grouped ConvTranspose
        if conv_linear.op_type == "ConvTranspose":
            if get_node_attribute(conv_linear, "group") is not None:
                valid &= get_node_attribute(conv_linear, "group") in (
                    1,
                    get_input_output_channels(conv_linear, model)[0],
                )
    return valid


def fold_all_batch_norms_to_weight(model: ModelProto) -> Tuple[List, List]:
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


def _fold_to_weight(
    model: ModelProto, conv_linear: NodeProto, bn: NodeProto, fold_backward: bool
):
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
    if not groups:
        groups = 1
    _, num_out_channels = get_input_output_channels(conv_linear, model)

    # If layer doesn't have bias, create a bias initializer and add it to the model, then retrieve it
    if not bias:
        bias_data = np.zeros(num_out_channels, _tensor_np_dtype(weight))
        bias_name = conv_linear.name + ".bias"
        bias = _tensor_from_array(bias_data, name=bias_name)
        model.graph.initializer.append(bias)
        conv_linear.input.append(bias_name)
        bias = ParamUtils.get_param(model, conv_linear, BIAS_INDEX)

    weight_np = to_array(weight)
    weight_np = np.expand_dims(weight_np, axis=tuple(range(weight_np.ndim, 4)))
    bias_np = to_array(bias)

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    # if conv_linear.type == "ConvTranspose" and conv_linear groups == 1:
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight_np = weight_np.transpose(1, 0, 2, 3)
    # Gemm layers may or may not need to have weights transposed depending on value of transB attribute
    elif conv_linear.op_type in LinearType and not get_node_attribute(
        conv_linear, "transB"
    ):
        weight_np = weight_np.transpose(1, 0, 2, 3)

    gamma = ParamUtils.get_param(model, bn, WEIGHT_INDEX)
    beta = ParamUtils.get_param(model, bn, BIAS_INDEX)
    mu = ParamUtils.get_param(model, bn, RUNNING_MEAN_INDEX)
    running_var = ParamUtils.get_param(model, bn, RUNNING_VAR_INDEX)

    gamma_np = to_array(gamma)
    beta_np = to_array(beta)
    mu_np = to_array(mu)
    epsilon = get_node_attribute(bn, "epsilon") or 1e-5
    sigma_np = np.sqrt(to_array(running_var) + epsilon)

    # In the case of BatchNorm2d -> Flatten -> Gemm, must resize the BN parameters to the Gemm input feature length
    channels = weight_np.shape[0] if fold_backward else weight_np.shape[1]
    gamma_np = gamma_np.repeat(channels / gamma_np.size)
    beta_np = beta_np.repeat(channels / beta_np.size)
    mu_np = mu_np.repeat(channels / mu_np.size)
    sigma_np = sigma_np.repeat(channels / sigma_np.size)

    weight_np, bias_np = batch_norm_fold(
        weight_np, bias_np, gamma_np, beta_np, mu_np, sigma_np, fold_backward
    )

    # Transpose weight back to original configuration
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight_np = weight_np.transpose(1, 0, 2, 3)
    elif conv_linear.op_type in LinearType and not get_node_attribute(
        conv_linear, "transB"
    ):
        weight_np = weight_np.transpose(1, 0, 2, 3)

    weight_np = weight_np.astype(_tensor_np_dtype(weight))
    bias_np = bias_np.astype(_tensor_np_dtype(bias))
    weight.raw_data = weight_np.tobytes()
    bias.raw_data = bias_np.tobytes()

    return BNLayer(bn, gamma_np, beta_np)


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
    bias_data = np.zeros(weight.dims[1], _tensor_np_dtype(weight))
    bias = _tensor_from_array(bias_data, name=bias_name)
    model.graph.initializer.append(bias)
    node.input.append(bias_name)


def _get_input_output_channel_axes(node: NodeProto) -> Tuple[int, int]:
    if node.op_type == "Conv":
        return 1, 0
    elif node.op_type == "ConvTranspose":
        return 0, 1
    elif node.op_type == "Gemm":
        transB = get_node_attribute(node, "transB")
        if transB == 1:
            return 1, 0
        else:
            return 0, 1
    else:
        raise RuntimeError


def get_input_output_channels(node: NodeProto, model: ModelProto) -> Tuple[int, int]:
    """
    Find the input and output channels of a given layer.
    :param node: The node to find the input/output channels of
    :param model: The onnx model to which the layers belong
    :return: Tuple of (num channels in, num channels out)
    """
    weight = ParamUtils.get_param(model, node, WEIGHT_INDEX)
    in_axis, out_axis = _get_input_output_channel_axes(node)
    groups = get_node_attribute(node, "group")
    # If group atttribute does not exist in the node,then default is 1
    if not groups:
        groups = 1
    if node.op_type == "Conv":
        num_in_channels = weight.dims[in_axis] * groups
        num_out_channels = weight.dims[out_axis]
    elif node.op_type == "ConvTranspose":
        num_in_channels = weight.dims[in_axis]
        num_out_channels = weight.dims[out_axis] * groups
    elif node.op_type == "Gemm":
        transB = get_node_attribute(node, "transB")
        if transB == 1:
            num_out_channels = weight.dims[out_axis]
            num_in_channels = weight.dims[in_axis]
        else:
            num_out_channels = weight.dims[out_axis]
            num_in_channels = weight.dims[in_axis]
    else:
        num_out_channels = None
        num_in_channels = None
    return num_in_channels, num_out_channels


def _resolve_const_input(value: onnx_ir.Value) -> onnx_ir.Value | None:
    """Resolve a BN parameter input to the root constant Value, propagating through Identity nodes."""
    if onnx_ir.convenience.get_const_tensor(value) is not None:
        return value
    producer = value.producer()
    if producer is None:
        return None
    if producer.op_type == "Identity":
        return _resolve_const_input(producer.inputs[0])
    return None


def _get_batchnorm_param_consumers(
    ir_model: onnx_ir.Model,
) -> Dict[str, set[onnx_ir.Node]]:
    """
    Get a mapping of root constant Value names to the set of BN nodes that consume them.

    :param ir_model: The ONNX IR model to analyze
    :return: A dictionary mapping parameter names to sets of consuming BN nodes
    """
    consumer_dict = defaultdict(set)
    for node in ir_model.graph.all_nodes():
        if node.op_type not in BatchNormType:
            continue
        for value in node.inputs[1:]:
            root = _resolve_const_input(value)
            if root is not None:
                consumer_dict[root.name].add(node)
    return consumer_dict


def _unique_name(base: str, existing: set[str]) -> str:
    """Generate a unique name based on the provided base that does not exist in the existing set."""
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# pylint: disable=too-many-locals
def _update_standalone_batchnorm_ops(model: ModelProto):
    """
    Update weight and bias of standalone batchnorm ops in the model.
    :param model: onnx Model for which batchnorm parameters are to be updated.
    """
    if not any(node.op_type in BatchNormType for node in model.graph.node):
        return

    ir_model = onnx_ir.from_proto(model)
    tensor_names = set(onnx_ir.convenience.create_value_mapping(ir_model.graph).keys())
    param_to_consumers = _get_batchnorm_param_consumers(ir_model)

    for node in ir_model.graph.all_nodes():
        if node.op_type in BatchNormType:
            params = [_resolve_const_input(v) for v in node.inputs[1:]]
            if None in params:
                continue  # Cannot update if any initializer is missing

            eps_attr = node.attributes.get("epsilon")
            epsilon = eps_attr.value if eps_attr else 1e-5

            tensor_w = onnx_ir.convenience.get_const_tensor(params[0]).numpy()
            tensor_b = onnx_ir.convenience.get_const_tensor(params[1]).numpy()
            tensor_rm = onnx_ir.convenience.get_const_tensor(params[2]).numpy()
            tensor_rv = onnx_ir.convenience.get_const_tensor(params[3]).numpy()
            # numpy promotes bfloat16 math to float32; the updated parameters must keep
            # their original dtypes.
            dtypes = [t.dtype for t in (tensor_w, tensor_b, tensor_rm, tensor_rv)]

            # update values
            inv_sigma = np.reciprocal(np.sqrt(tensor_rv + epsilon))
            tensor_w = tensor_w * inv_sigma
            tensor_b = tensor_b - tensor_rm * tensor_w
            tensor_rm = np.zeros(tensor_w.shape, tensor_w.dtype)
            tensor_rv = np.ones(tensor_w.shape, tensor_w.dtype)
            tensor_w, tensor_b, tensor_rm, tensor_rv = (
                t.astype(dtype)
                for t, dtype in zip((tensor_w, tensor_b, tensor_rm, tensor_rv), dtypes)
            )
            node.attributes["epsilon"] = onnx_ir.convenience.convert_attribute(
                "epsilon", 0.0
            )

            new_data = [tensor_w, tensor_b, tensor_rm, tensor_rv]
            for idx, (param_val, new_array) in enumerate(zip(params, new_data)):
                param_to_consumers[param_val.name].discard(node)
                is_shared = len(param_to_consumers[param_val.name]) > 0
                is_indirect = node.inputs[idx + 1] is not param_val

                if is_shared or is_indirect or param_val.const_value is None:
                    # Insert new initializer for the updated parameter
                    name = _unique_name(param_val.name, tensor_names)
                    tensor_names.add(name)
                    new_val = onnx_ir.val(
                        name=name,
                        const_value=onnx_ir.Tensor(new_array, name=name),
                    )
                    ir_model.graph.register_initializer(new_val)
                    node.replace_input_with(idx + 1, new_val)
                else:
                    # Update the existing intializer in place
                    param_val.const_value = onnx_ir.Tensor(
                        new_array, name=param_val.name
                    )

    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    model.CopyFrom(onnx_ir.to_proto(ir_model))


def _has_unfolded_batchnorms(
    model: ModelProto, connected_graph: ConnectedGraph | None = None
) -> bool:
    """
    Check if the model has any BatchNormalization layers that can be folded.

    Args:
        model: onnx Model to check for foldable BatchNormalization layers.
        connected_graph: ConnectedGraph object. If None, it will be created from the model.

    Returns:
        True if there are foldable BatchNormalization layers, False otherwise.
    """
    if connected_graph is None:
        connected_graph = ConnectedGraph(model)
    conv_bn_pairs, bn_conv_pairs = find_all_batch_norms_to_fold(connected_graph)
    if len(conv_bn_pairs) + len(bn_conv_pairs) > 0:
        return True

    # Note: Remaining batchnorms should have running stats folded into beta/gamma parameters
    return _has_batchnorms_with_fusable_running_stats(model)


def _has_batchnorms_with_fusable_running_stats(model: ModelProto) -> bool:
    for node in model.graph.node:
        if node.op_type not in BatchNormType:
            continue

        inits = [ParamUtils.get_param_by_name(model, name) for name in node.input[1:]]

        # If any of the initializers is missing, stats are not fusable
        if None in inits:
            continue

        init_rm, init_rv = inits[2], inits[3]

        tensor_rm = to_array(init_rm)
        tensor_rv = to_array(init_rv)
        if not np.allclose(tensor_rm, 0) or not np.allclose(tensor_rv, 1):
            return True

    return False
