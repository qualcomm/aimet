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
""" Utilities for ONNX Connected Graph """
from typing import Dict, List
import onnx
from packaging import version

from aimet_onnx.meta.connectedgraph import ConnectedGraph

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto


ActivationTypes = ['Relu', 'Clip', 'Sigmoid', 'Tanh', 'PRelu', 'Softmax']


def get_op_given_param_name(connected_graph: ConnectedGraph, param_name: str):
    """
    Gets op for a given param name

    :param connected_graph: Connected graph
    :param param_name: Name of the parameter
    """
    ops = connected_graph.get_all_ops()
    for op in ops.values():
        if op.parameters:
            for param, _ in op.parameters.values():
                if param.name == param_name:
                    return op
    return None


def get_param_shape_using_connected_graph(connected_graph: ConnectedGraph, param_name: str):
    """
    Gets param shape given param name

    :param connected_graph: Connected graph
    :param param_name: Name of the parameter
    """
    ops = connected_graph.get_all_ops()
    for op in ops.values():
        if op.parameters:
            for param, _ in op.parameters.values():
                if param.name == param_name:
                    return param.shape
    return None

def get_module_act_func_pair(model: ModelProto) -> Dict[str, str]:
    """
    For given model, returns dictionary of module to immediate following activation function else maps
    module to None.

    Activation functions should be defined as nn.Modules in model and not as functional in the forward pass.

    :param model: ONNX model
    :return: Dictionary of module name to activation function name
    """
    # Create ConnectedGraph
    graph = ConnectedGraph(model)

    # Maps module to next following activation function else None
    module_act_func_pair = {}

    # Get all the ops
    all_ops = graph.get_all_ops()

    for op in all_ops.values():

        module_act_func_pair[op.name] = None

        if op.output and op.output.consumers:
            # Get the next op
            next_op = op.output.consumers[0]
            # Get the appropriate activation function
            if next_op.type in ActivationTypes:
                module_act_func_pair[op.name] = next_op.type

    return module_act_func_pair


def get_ordered_ops(model: ModelProto) -> List:
    """
    Gets list of ordered ops

    :param model: ONNX model
    :return: A list of ordered ops
    """
    cg = ConnectedGraph(model)
    return cg.ordered_ops
