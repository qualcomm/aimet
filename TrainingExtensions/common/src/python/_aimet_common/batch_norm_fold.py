# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Batch normalization fold"""

import math
from typing import List, Tuple, Union
import numpy as np
from .defs import ActivationType
from .connected_graph.operation import Op


def expand_shape_to_4d(shape: Tuple) -> Union[List, np.ndarray]:
    """
    Expand the shape of the weight into 4d.

    :param shape:
    :return: 4d shape.
    """
    shape = list(shape)
    dims = len(shape)

    if dims > 5:
        raise RuntimeError

    if dims == 4:
        _4d_shape = shape

    else:
        if dims < 4:
            # If we have less dimensions, we add 1s to make 4 dimensions
            _4d_shape = np.append(shape, [1 for _ in range(4 - dims)]).astype(int)
        else:
            # If we have more dimensions, we concatenate all the dimensions beyond 3 into one dimension
            _4d_shape = np.array(shape[:3] + [math.prod(shape[3:])])
    return _4d_shape


def batch_norm_fold(
    weight: np.ndarray,
    bias: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    fold_backward: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param weight: conv/linear weight
    :param bias: conv/linear bias
    :param gamma: Batch Norm layer weight
    :param beta: Batch Norm layer bias
    :param mu: Batch Norm layer running mean
    :param sigma: Batch Norm layer running variance (calculated as square root of running variance)
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: Updated weight, bias
    """
    assert len(weight.shape) == 4

    assert not np.any(sigma == 0)
    scale = gamma / sigma

    if fold_backward:
        _weight = weight * scale[:, None, None, None]
        _bias = beta - (mu - bias) * scale
    else:
        _w_2d = weight.sum(3).sum(2)
        mu_hat = np.matmul(_w_2d, mu * scale)
        beta_hat = np.matmul(_w_2d, beta)
        _weight = weight * scale[None, :, None, None]
        _bias = beta_hat - mu_hat + bias
    return _weight, _bias


CONV_OP_TYPES = [
    "Conv1d",
    "Conv2D",
    "DepthwiseConv2dNative",
    "Conv",
    "ConvTranspose",
    "Conv3d",
]
LINEAR_OP_TYPES = ["Dense", "Gemm", "MatMul"]
BN_OP_TYPES = [
    "FusedBatchNormV3",
    "FusedBatchNorm",
    "BatchNormalization",
    "BatchNorm3d",
]


class ConvBnInfoType:
    """
    Type for hoding convs with bn info and activation types
    Activation types supported are Relu and Relu6
    """

    def __init__(
        self,
        input_bn=None,
        output_bn=None,
        in_activation_type: ActivationType = ActivationType.no_activation,
        out_activation_type: ActivationType = ActivationType.no_activation,
    ):
        """
        :param input_bn: Reference to Input BatchNorm to layer
        :param output_bn: Reference to Output BatchNorm to layer
        :param in_activation_type: Type of Activation
        :param out_activation_type: Type of Activation
        """

        self.input_bn = input_bn
        self.output_bn = output_bn
        self.in_activation_type = in_activation_type
        self.out_activation_type = out_activation_type


class ConvBnPatternHandler:
    """
    common handler for matched patterns for bias correction and batchnorm fold.
    """

    def __init__(self):
        self.conv_linears_with_bn_dict = {}

    def get_conv_linear_bn_info_dict(self):
        """
        returns the dictionary created
        :return: dictionary of convs/linears with bn and activation info
        """
        return self.conv_linears_with_bn_dict

    def __call__(self, *args, **kwargs):
        """
        custom pattern match handler that keeps a dictionary of convs/linears with bn and activation info.
        """

        _, op_subset = args

        bn_activation_info = ConvBnInfoType()

        activation_type = ActivationType.no_activation
        conv_op = None
        bn_op = None

        for op in op_subset:
            if op.type in CONV_OP_TYPES + LINEAR_OP_TYPES:
                conv_op = op
                op_key = get_op_dict_key(conv_op)
                if op_key in self.conv_linears_with_bn_dict:
                    bn_activation_info = self.conv_linears_with_bn_dict[op_key]
            elif op.type in BN_OP_TYPES:
                bn_op = op
            elif op.type in ["Relu6", "Clip"]:
                activation_type = ActivationType.relu6
            elif op.type in ["Relu"]:
                activation_type = ActivationType.relu

        if len(op_subset) >= 2:
            if op_subset[0].type in BN_OP_TYPES:
                bn_activation_info.input_bn = bn_op
                bn_activation_info.in_activation_type = activation_type
            # we do not match linear layers with preceding bn for bias correction
            elif op_subset[0].type in CONV_OP_TYPES + LINEAR_OP_TYPES:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
            # in tf linear layer has two ops together [flatten/reshape -- dense] , check for len 3
            elif len(op_subset) >= 3 and op_subset[1].type in ["Dense"]:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
        op_key = get_op_dict_key(conv_op)
        self.conv_linears_with_bn_dict[op_key] = bn_activation_info


def get_op_dict_key(op: Op):
    """
    Returns the object to be used as a key in the conv/linear BN dict.
    For torch and tensorflow models, returns op.get_module(). For onnx models, returns the original op,
    since ONNX callers look this dict up by op rather than by the underlying node.

    :param op: connected graph layer to be used as a dictionary key
    :return: object (op or op.get_module()) to be used as a key in the conv/linear BN dict
    """
    module = op.get_module()
    # onnx_ir.Node carries an op_type; no other framework's module/op object does
    if hasattr(module, "op_type"):
        return op
    try:
        hash(module)
    except TypeError:
        return op

    return module
