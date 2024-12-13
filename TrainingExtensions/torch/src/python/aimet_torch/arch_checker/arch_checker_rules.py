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

"""
Utility for rules to check architecture.
Node checks should follow :param node: :return bool:.
Pattern checks should follow :param connected_graph: :return list[ops]:
"""

from typing import List, Callable
import torch

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.utils import AimetLogger

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.batch_norm_fold import find_standalone_batchnorm_ops

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def _check_conv_channel_32_base(node: torch.nn.Module)-> bool:
    """
    Channels should be mutilple of 32 for better performance.
    :param node: torch node to be checked.
    :return True if both model's input and output channel depth is 32's multiple.
    """
    if node.in_channels % 32 == 0 and node.out_channels % 32 == 0:
        return True
    return False

def _check_conv_channel_larger_than_32(node: torch.nn.Module)-> bool:
    """
    Channels should be at least 32 for better performance.
    :param node: torch node to be checked.
    :return model's input/output channel depth is larger than 32 or not.
    """
    if node.in_channels >= 32 and node.out_channels >= 32:
        return True
    return False

def _activation_checks(node: torch.nn.Module)-> bool:
    """
    Common checkes for all torch activations.
    Prelu and swish (SiLU) degenerates the quantization performance.
    :param node: torch node to be checked.
    :return True if not a activation with bad bad quantization performance activations function.
    """
    _degenerating_activation_tuple = (torch.nn.modules.activation.SiLU,
                                      torch.nn.modules.activation.PReLU)

    if isinstance(node, _degenerating_activation_tuple):
        return False
    return True

def _check_batch_norm_fold(connected_graph: ConnectedGraph) -> List:
    """
    Pattern checker: return all standalone batchnorms.
    :param connected_graph: Connected_graph object.
    :return: List of stand alone (not foldable) batch norms in connected_graph.
    """
    stand_alone_bn_ops = find_standalone_batchnorm_ops(connected_graph)

    return list(stand_alone_bn_ops)

def _check_intermediate_padding(connected_graph: ConnectedGraph) -> List:
    """
    Checks is there intermediate padding in conv2 sequence: [Conv -> Activation -> (Optionally) BN -> Conv].
    Activation can be [relu, tanh, hardswish].
    :param connected_graph: Connected_graph object.
    :return: List of conv Ops contains intermediate padding in connected_graph.
    """
    def _examine_intermediate_padding_in_op_subset(op_subset):
        """
        Check is there intermediate padding in the second conv in op_subset.
        If both 1st and 2nd conv have paddings, add conv2 to inter_pad_node_list.
        """
        if len(op_subset) == 4:
            conv1, _, _, conv2 = op_subset
        else:
            conv1, _, conv2 = op_subset

        conv1_padding = sum(conv1.get_module().padding)
        conv2_padding = sum(conv2.get_module().padding)
        if conv1_padding and conv2_padding:
            inter_pad_op_list.append(conv2)

    _support_activation_op_type = ("Relu", "Tanh", "HardSwish")
    _support_conv_op_type = ("Conv", "Conv2D")

    inter_pad_op_list = []
    handler = PatternHandler(_examine_intermediate_padding_in_op_subset)

    patterns_with_callbacks = []
    for _act_op_type in _support_activation_op_type:
        for _conv_op_type in _support_conv_op_type:
            patterns_with_callbacks.append(PatternType(pattern=[_conv_op_type, _act_op_type, "BatchNormalization", _conv_op_type], action=handler))
            patterns_with_callbacks.append(PatternType(pattern=[_conv_op_type, "BatchNormalization", _act_op_type, _conv_op_type], action=handler))
            patterns_with_callbacks.append(PatternType(pattern=[_conv_op_type, _act_op_type, _conv_op_type], action=handler))

    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)
    graph_searcher.find_all_patterns_in_graph_apply_actions()

    return inter_pad_op_list


def _find_all_split_bn_in_graph(connected_graph: ConnectedGraph):
    """
    Find all conv/linear -> split -> bn patterns.
    :param connected_graph: ConnectedGraph object.
    :return conv_bn_pair_list: Captured patterns.
    """
    def _examine_split_bn(op_subset):
        """
        :param op_subset: found subset pattern [split, bn]
        """
        split_bn_pair_list.append(op_subset)

    split_bn_pair_list = []

    handler = PatternHandler(_examine_split_bn)
    _support_split_op = ["Concat"]

    patterns_with_callbacks = []
    for _split_op in _support_split_op:
        patterns_with_callbacks.append(PatternType(pattern=[_split_op, "BatchNormalization"], action=handler))
    patterns_with_callbacks.append(PatternType(pattern=[_split_op, 'BatchNorm3d'], action=handler))

    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)
    graph_searcher.find_all_patterns_in_graph_apply_actions()

    return split_bn_pair_list

def _check_foldable_bn_with_split(connected_graph: ConnectedGraph) -> List[List]:
    """
    Check if bn in [split, bn] pattern is foldable to all input for a split node.
    :param connected_graph: ConnectedGraph object.
    :return not_foldable_pattern: List of List:[unfoldable node, split_node, bn_node ]
    """
    foldable_type = ['Conv1d', 'Conv', 'ConvTranspose'] + ['Gemm']
    split_bn_pair_list = _find_all_split_bn_in_graph(connected_graph)
    not_foldable_pattern = []

    # exam each branch through the split node.
    for (split, bn) in split_bn_pair_list:
        if bn.type == 'BatchNorm3d':
            _foldable_type = ["Conv3d"]
        else:
            _foldable_type = foldable_type

        input_ops_tuple = _get_split_point_input(split.input_ops)
        for _node in input_ops_tuple:
            if _node.type in _foldable_type:
                not_foldable_pattern.append([input_ops_tuple, split, bn])

    return not_foldable_pattern

def _get_split_point_input(input_ops: List):
    """
    Ignore "Split" node and get the node splited.
    :param input_ops: list of ops.
    :return input_ops_tuple: tuple of ops
    """
    input_ops_list = []
    for op in input_ops:
        if op.type == CG_SPLIT:
            op = op.input_ops[0]
        input_ops_list.append(op)
    return tuple(input_ops_list)

class CheckType(type):
    """ Metaclass to overwrite __instancecheck__. """
    def __instancecheck__(cls, obj):
        return cls._test(obj)

# A type class to put all torch activations together.
# pylint: disable=too-few-public-methods
class TorchActivations(metaclass=CheckType):
    """ Type class for all torch activations. """
    @classmethod
    def _test(cls, module):
        if module.__module__ == 'torch.nn.modules.activation':
            return True
        return False

class PatternHandler():
    """ Object to handle pattern checkes. """
    def __init__(self, check: Callable):
        self.check = check

    def __call__(self, *args, **kwargs):
        """
        Run pattern check on PatternType_object, op_subset.
        """
        _, op_subset = args
        self.check(op_subset)

NODE_CHECK_DICT = {torch.nn.modules.conv.Conv2d: [_check_conv_channel_32_base,
                                                  _check_conv_channel_larger_than_32],
                   TorchActivations: [_activation_checks],}
PATTERN_CHECK_LIST = [_check_batch_norm_fold, _check_intermediate_padding, _check_foldable_bn_with_split]
