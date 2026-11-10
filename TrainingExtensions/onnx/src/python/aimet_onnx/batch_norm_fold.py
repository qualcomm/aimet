# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""ONNX Code to fold batch-norm layers"""

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple
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

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.connectedgraph import (
    WEIGHT_INDEX,
    BIAS_INDEX,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
)
from aimet_onnx.meta.operations import Op
from aimet_onnx.ir_utils import (
    get_constant_value,
    set_static_tensor,
    static_tensor,
    unique_name,
)
from aimet_onnx.utils import find_shared_param_names

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


def _zip_nodes(
    proto_graph: onnx.GraphProto, ir_graph: onnx_ir.Graph
) -> Iterator[Tuple[NodeProto, onnx_ir.Node]]:
    """
    Pair every node of ``proto_graph`` with the node at the same position in ``ir_graph``,
    including nodes in subgraphs. The two graphs must have the same structure, e.g. one was
    converted from the other with ``onnx_ir.from_proto`` or ``onnx_ir.to_proto``.

    Node names are not used, because ONNX allows them to be empty or repeated.
    """
    for proto_node, ir_node in zip(proto_graph.node, ir_graph, strict=True):
        yield proto_node, ir_node
        for attr in proto_node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _zip_nodes(attr.g, ir_node.attributes[attr.name].as_graph())
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for proto_subgraph, ir_subgraph in zip(
                    attr.graphs, ir_node.attributes[attr.name].as_graphs(), strict=True
                ):
                    yield from _zip_nodes(proto_subgraph, ir_subgraph)


def _ir_nodes_from_proto_id(
    proto: ModelProto, model: onnx_ir.Model
) -> Dict[int, onnx_ir.Node]:
    """Map ``id()`` of each NodeProto in ``proto`` to the matching node in ``model``."""
    return {
        id(proto_node): ir_node
        for proto_node, ir_node in _zip_nodes(proto.graph, model.graph)
    }


def _proto_nodes_from_ir_id(
    proto: ModelProto, model: onnx_ir.Model
) -> Dict[int, NodeProto]:
    """Map ``id()`` of each node in ``model`` to the matching NodeProto in ``proto``."""
    return {
        id(ir_node): proto_node
        for proto_node, ir_node in _zip_nodes(proto.graph, model.graph)
    }


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
    model = onnx_ir.from_proto(connected_graph.model)
    nodes = _ir_nodes_from_proto_id(connected_graph.model, model)
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
                elif _has_static_params(
                    nodes[id(bn_info.output_bn.get_module())]
                ) and is_valid_bn_fold(nodes[id(node.get_module())], True):
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
                elif _has_static_params(
                    nodes[id(bn_info.input_bn.get_module())]
                ) and is_valid_bn_fold(nodes[id(node.get_module())], False):
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


def _get_constant_input(
    node: onnx_ir.Node, index: int
) -> Tuple[Optional[onnx_ir.Value], bool]:
    """
    Retrieves the static value feeding ``node.inputs[index]``, either directly or through
    a single Transpose.
    :param node: The node to find the input for
    :param index: The index of the desired input within node.inputs
    :return: Tuple containing the static value (or None) and a bool specifying whether the value is
             transposed before entering the node
    """
    value = node.inputs[index] if index < len(node.inputs) else None
    constant = get_constant_value(value)
    if constant is not None:
        return constant, False
    producer = value.producer() if value is not None else None
    if producer is not None and producer.op_type == "Transpose":
        return get_constant_value(producer.inputs[0]), True
    return None, False


def _has_static_params(bn: onnx_ir.Node) -> bool:
    """Return True if gamma, beta, mean and variance of ``bn`` are all static."""
    return all(get_constant_value(value) is not None for value in bn.inputs[1:])


def is_valid_bn_fold(conv_linear: onnx_ir.Node, fold_backward: bool) -> bool:
    """
    Determine if a given layer can successfully absorb a BatchNorm given the layer type and parameters
    :param conv_linear: The Conv/Linear layer to fold a BatchNorm into.
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: True if a BatchNorm layer can be folded without causing output error.
    """
    # The weight and bias must be static to absorb the BN. This also rules out dynamic MatMuls
    weight, _ = _get_constant_input(conv_linear, WEIGHT_INDEX)
    bias = (
        conv_linear.inputs[BIAS_INDEX] if len(conv_linear.inputs) > BIAS_INDEX else None
    )
    if weight is None or (bias is not None and get_constant_value(bias) is None):
        return False
    valid = True
    if not fold_backward:
        # Cannot fold BN -> Conv with padding. AIMET does not support forward folding to grouped or DW Conv
        if conv_linear.op_type == "Conv":
            pads = conv_linear.attributes.get_ints("pads")
            if pads:
                valid &= all(item == 0 for item in pads)
            valid &= conv_linear.attributes.get_int("group") in (None, 1)
        # AIMET does not support forward folding to ConvTranspose
        elif conv_linear.op_type == "ConvTranspose":
            valid = False
    else:
        # AIMET does not support backwards folding to grouped ConvTranspose
        if conv_linear.op_type == "ConvTranspose":
            group = conv_linear.attributes.get_int("group")
            if group is not None:
                valid &= group in (1, get_input_output_channels(conv_linear)[0])
    return valid


def _get_input_output_channel_axes(node: onnx_ir.Node) -> Tuple[int, int]:
    if node.op_type == "Conv":
        return 1, 0
    elif node.op_type == "ConvTranspose":
        return 0, 1
    elif node.op_type == "Gemm":
        if node.attributes.get_int("transB") == 1:
            return 1, 0
        else:
            return 0, 1
    else:
        raise RuntimeError


def get_input_output_channels(node: onnx_ir.Node) -> Tuple[int, int]:
    """
    Find the input and output channels of a given layer.
    :param node: The node to find the input/output channels of
    :return: Tuple of (num channels in, num channels out)
    """
    weight_shape = static_tensor(get_constant_value(node.inputs[WEIGHT_INDEX])).shape
    in_axis, out_axis = _get_input_output_channel_axes(node)
    # If group atttribute does not exist in the node,then default is 1
    groups = node.attributes.get_int("group") or 1
    if node.op_type == "Conv":
        num_in_channels = weight_shape[in_axis] * groups
        num_out_channels = weight_shape[out_axis]
    elif node.op_type == "ConvTranspose":
        num_in_channels = weight_shape[in_axis]
        num_out_channels = weight_shape[out_axis] * groups
    elif node.op_type == "Gemm":
        num_out_channels = weight_shape[out_axis]
        num_in_channels = weight_shape[in_axis]
    else:
        num_out_channels = None
        num_in_channels = None
    return num_in_channels, num_out_channels


def fold_all_batch_norms_to_weight(model: ModelProto) -> Tuple[List, List]:
    """
    Fold all possible batch_norm layers in a model into the weight of the corresponding conv layers.
    The model is modified in place.

    :param model: onnx Model to perform BN fold on
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    if isinstance(model, ONNXModel):
        model = model.model
    connected_graph = ConnectedGraph(model)
    if not _has_batchnorm_ops(connected_graph):
        return [], []
    return _fold_all_batch_norms_to_weight(connected_graph)


def _has_batchnorm_ops(connected_graph: ConnectedGraph) -> bool:
    return any(
        op.type in BatchNormType for op in connected_graph.get_all_ops().values()
    )


def _fold_all_batch_norms_to_weight(
    connected_graph: ConnectedGraph,
) -> Tuple[List, List]:
    """
    Fold all possible batch_norm layers in ``connected_graph.model`` into the weight of the
    corresponding conv layers, in place

    :param connected_graph: ConnectedGraph of the model to perform BN fold on
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    conv_bn_pairs, bn_conv_pairs = find_all_batch_norms_to_fold(connected_graph)
    proto = connected_graph.model
    model = onnx_ir.from_proto(proto)
    nodes = _ir_nodes_from_proto_id(proto, model)
    conv_bns = []
    bn_convs = []
    for conv, bn in conv_bn_pairs:
        conv, bn = nodes[id(conv)], nodes[id(bn)]
        conv_bns.append((conv, _fold_to_weight(conv, bn, True)))
        _remove_batchnorm(bn)

    for bn, conv in bn_conv_pairs:
        conv, bn = nodes[id(conv)], nodes[id(bn)]
        bn_convs.append((conv, _fold_to_weight(conv, bn, False)))
        _remove_batchnorm(bn)

    _update_standalone_batchnorm_ops(model)

    proto.CopyFrom(onnx_ir.to_proto(model))
    # CopyFrom replaces every NodeProto, so return the matching nodes of the new proto
    proto_nodes = _proto_nodes_from_ir_id(proto, model)
    return (
        [(proto_nodes[id(conv)], bn_layer) for conv, bn_layer in conv_bns],
        [(proto_nodes[id(conv)], bn_layer) for conv, bn_layer in bn_convs],
    )


def _fold_to_weight(
    conv_linear: onnx_ir.Node, bn: onnx_ir.Node, fold_backward: bool
) -> BNLayer:
    """
    Fold BatchNorm into the weight and bias of the given layer.

    :param conv_linear: Conv or linear layer to fold BN into.
    :param bn: BatchNorm to fold.
    :param fold_backward: True if the BatchNorm comes after the Conv
    """
    # Must convert MatMul layers to Gemm to allow bias
    if conv_linear.op_type == "MatMul":
        _matmul_to_gemm(conv_linear)

    weight = get_constant_value(conv_linear.inputs[WEIGHT_INDEX])
    bias = (
        get_constant_value(conv_linear.inputs[BIAS_INDEX])
        if len(conv_linear.inputs) > BIAS_INDEX
        else None
    )
    groups = conv_linear.attributes.get_int("group") or 1
    trans_b = conv_linear.attributes.get_int("transB")
    _, num_out_channels = get_input_output_channels(conv_linear)

    weight_np = static_tensor(weight).numpy()
    weight_shape, weight_dtype = weight_np.shape, weight_np.dtype

    # If layer doesn't have bias, create a bias initializer and add it to the model
    if bias is None:
        bias = _add_zero_bias(conv_linear, num_out_channels, weight_dtype)

    weight_np = np.expand_dims(weight_np, axis=tuple(range(weight_np.ndim, 4)))
    bias_np = static_tensor(bias).numpy()
    bias_dtype = bias_np.dtype

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight_np = weight_np.transpose(1, 0, 2, 3)
    # Gemm layers may or may not need to have weights transposed depending on value of transB attribute
    elif conv_linear.op_type in LinearType and not trans_b:
        weight_np = weight_np.transpose(1, 0, 2, 3)

    gamma_np = static_tensor(get_constant_value(bn.inputs[WEIGHT_INDEX])).numpy()
    beta_np = static_tensor(get_constant_value(bn.inputs[BIAS_INDEX])).numpy()
    mu_np = static_tensor(get_constant_value(bn.inputs[RUNNING_MEAN_INDEX])).numpy()
    running_var_np = static_tensor(
        get_constant_value(bn.inputs[RUNNING_VAR_INDEX])
    ).numpy()
    epsilon = bn.attributes.get_float("epsilon") or 1e-5
    sigma_np = np.sqrt(running_var_np + epsilon)

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
    elif conv_linear.op_type in LinearType and not trans_b:
        weight_np = weight_np.transpose(1, 0, 2, 3)

    set_static_tensor(
        weight,
        np.ascontiguousarray(weight_np.astype(weight_dtype).reshape(weight_shape)),
    )
    set_static_tensor(bias, bias_np.astype(bias_dtype))

    return BNLayer(onnx_ir.serde.serialize_node(bn), gamma_np, beta_np)


def _add_zero_bias(
    node: onnx_ir.Node, num_channels: int, dtype: np.dtype
) -> onnx_ir.Value:
    """Feed ``node`` a zero bias initializer named ``<node name>.bias`` and return it."""
    existing = set(onnx_ir.convenience.create_value_mapping(node.graph))
    name = unique_name(node.name + ".bias", existing)
    bias = onnx_ir.val(
        name, const_value=onnx_ir.tensor(np.zeros(num_channels, dtype), name=name)
    )
    node.graph.register_initializer(bias)
    if len(node.inputs) <= BIAS_INDEX:
        node.resize_inputs(BIAS_INDEX + 1)
    node.replace_input_with(BIAS_INDEX, bias)
    return bias


def _matmul_to_gemm(node: onnx_ir.Node):
    """
    Convert MatMul node to Gemm and initialize bias to zeros

    :param node: MatMul node to convert to Gemm
    """
    assert node.op_type == "MatMul"

    weight, transposed = _get_constant_input(node, WEIGHT_INDEX)
    if transposed:
        array = np.ascontiguousarray(static_tensor(weight).numpy().T)
        weight.const_value = onnx_ir.tensor(array, name=weight.name)
        weight.shape = onnx_ir.Shape(array.shape)
        node.replace_input_with(WEIGHT_INDEX, weight)
    node.op_type = "Gemm"
    node.name = node.name.replace("MatMul", "Gemm")
    # Create bias vector for Gemm operation
    tensor = static_tensor(weight)
    _add_zero_bias(node, tensor.shape[1], tensor.dtype.numpy())


def _remove_batchnorm(bn: onnx_ir.Node):
    """
    Remove ``bn`` from its graph, feeding its consumers the BN's input instead, along with
    the initializers no other node uses.

    :param bn: BatchNorm node to remove
    """
    graph = bn.graph
    source, output = bn.inputs[0], bn.outputs[0]
    params = list(bn.inputs[1:])

    name = None
    if output.is_graph_output():
        if source.producer() is None:
            raise RuntimeError(
                f"Cannot remove BatchNormalization '{bn.name}': its output '{output.name}' is a "
                f"graph output, but its input '{source.name}' is not produced by any node"
            )
        # The graph output keeps its name, type and shape
        if source.type is None:
            source.type = output.type
        if source.shape is None:
            source.shape = output.shape
        name = output.name

    output.replace_all_uses_with(source, replace_graph_outputs=True)
    graph.remove(bn, safe=True)
    if name is not None:
        output.name = None
        source.name = name

    for param in params:
        if (
            param is not None
            and param.is_initializer()
            and not param.uses()
            and not param.is_graph_input()
        ):
            param.graph.initializers.pop(param.name)


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


# pylint: disable=too-many-locals
def _update_standalone_batchnorm_ops(model: onnx_ir.Model):
    """
    Update weight and bias of standalone batchnorm ops in the model.
    :param model: onnx_ir Model for which batchnorm parameters are to be updated, in place.
    """
    if not any(node.op_type in BatchNormType for node in model.graph):
        return

    tensor_names = set(onnx_ir.convenience.create_value_mapping(model.graph).keys())
    param_to_consumers = _get_batchnorm_param_consumers(model)

    for node in model.graph.all_nodes():
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
                    name = unique_name(param_val.name, tensor_names)
                    tensor_names.add(name)
                    new_val = onnx_ir.val(
                        name=name,
                        const_value=onnx_ir.Tensor(new_array, name=name),
                    )
                    model.graph.register_initializer(new_val)
                    node.replace_input_with(idx + 1, new_val)
                else:
                    # Update the existing intializer in place
                    param_val.const_value = onnx_ir.Tensor(
                        new_array, name=param_val.name
                    )

    onnx_ir.passes.common.RemoveUnusedNodesPass().call(model)


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
    if not _has_batchnorm_ops(connected_graph):
        return False
    conv_bn_pairs, bn_conv_pairs = find_all_batch_norms_to_fold(connected_graph)
    if len(conv_bn_pairs) + len(bn_conv_pairs) > 0:
        return True

    # Note: Remaining batchnorms should have running stats folded into beta/gamma parameters
    return _has_batchnorms_with_fusable_running_stats(onnx_ir.from_proto(model))


def _has_batchnorms_with_fusable_running_stats(model: onnx_ir.Model) -> bool:
    for node in model.graph:
        if node.op_type not in BatchNormType:
            continue

        params = [get_constant_value(value) for value in node.inputs[1:]]

        # If any of the parameters is missing, stats are not fusable
        if None in params:
            continue

        tensor_rm = static_tensor(params[2]).numpy()
        tensor_rv = static_tensor(params[3]).numpy()
        if not np.allclose(tensor_rm, 0) or not np.allclose(tensor_rv, 1):
            return True

    return False
