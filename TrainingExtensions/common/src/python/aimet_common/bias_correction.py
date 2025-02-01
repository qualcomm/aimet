# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""  holds common code for bias correction """

import numpy as np
from scipy.stats import norm

from aimet_common.defs import ActivationType
from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.operation import Op

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

CONV_OP_TYPES = ['Conv1d', 'Conv2D', 'DepthwiseConv2dNative', 'Conv', 'ConvTranspose', 'Conv3d']
LINEAR_OP_TYPES = ['Dense', 'Gemm', 'MatMul']
BN_OP_TYPES = ['FusedBatchNormV3', 'FusedBatchNorm', 'BatchNormalization', 'BatchNorm3d']


class ConvBnInfoType:
    """
    Type for hoding convs with bn info and activation types
    Activation types supported are Relu and Relu6
    """
    def __init__(self,
                 input_bn=None,
                 output_bn=None,
                 in_activation_type: ActivationType = ActivationType.no_activation,
                 out_activation_type: ActivationType = ActivationType.no_activation):
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
            elif op.type in ['Relu6', 'Clip']:
                activation_type = ActivationType.relu6
            elif op.type in ['Relu']:
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
            elif len(op_subset) >= 3 and op_subset[1].type in ['Dense']:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
        op_key = get_op_dict_key(conv_op)
        self.conv_linears_with_bn_dict[op_key] = bn_activation_info


def get_op_dict_key(op: Op):
    """
    Returns the object to be used as a key in the conv/linear BN dict.
    For torch and tensorflow models, returns op.get_module(). For onnx models, returns the original op.

    :param op: connected graph layer to be used as a dictionary key
    :return: object (op or op.get_module()) to be used as a key in the conv/linear BN dict
    """
    module = op.get_module()
    # ONNX NodeProto objects are not hashable, return the original Op object instead
    if module.__hash__ is None:
        return op
    return module


def empirical_bias_correction(reference_outputs: np.ndarray, quantized_outputs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Empirical bias correction.

    :param quantized_outputs:
    :param reference_outputs:
    :param bias:
    :return: Updated bias
    """
    error = quantized_outputs - reference_outputs
    error = error.mean(3).mean(2).mean(0)
    _bias = bias - error
    return _bias


def analytical_bias_correction(fp_weight: np.ndarray,
                               q_dq_weight: np.ndarray,
                               bias: np.ndarray,
                               beta: np.ndarray,
                               gamma: np.ndarray,
                               activation_type: ActivationType) -> np.ndarray:
    """
    Analytical bias correction.

    :param fp_weight:
    :param q_dq_weight:
    :param bias:
    :param beta:
    :param gamma:
    :param activation_type:
    :return: Updated bias
    """
    diff = q_dq_weight - fp_weight
    epsilon = diff.sum(3).sum(2)

    if activation_type == ActivationType.no_activation:
        e_x = beta
    elif activation_type == ActivationType.relu:
        e_x = beta * (1 - norm.cdf(-beta / gamma)) + gamma * norm.pdf(-beta / gamma)
    elif activation_type == ActivationType.relu6:
        b = 6
        z = norm.pdf(-beta / gamma) - norm.pdf((b - beta) / gamma)
        Z = norm.cdf((b - beta) / gamma) - norm.cdf(-beta / gamma)
        e_x = gamma * z + beta * Z + b * (1 - norm.cdf((b - beta) / gamma))
    else:
        raise ValueError('Unsupported activation type: ', activation_type)

    if epsilon.shape[1] == 1:
        ep = epsilon.reshape(epsilon.shape[0])
        error = np.multiply(ep, e_x)
    else:
        error = np.matmul(epsilon, e_x)

    _bias = bias - error
    return _bias
