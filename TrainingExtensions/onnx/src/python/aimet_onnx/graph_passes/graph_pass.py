# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=missing-module-docstring

from abc import abstractmethod
from aimet_common.connected_graph.operation import Op
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.graph_passes.utils import get_const_input_names, get_output_names
from aimet_onnx.utils import ModelProto
from typing import Dict, List

class GraphPass:
    """
    Abstract GraphPass to iterate over Ops from ConnectedGraph
    """
    @abstractmethod
    def match_pattern(self, op: Op, model: ModelProto):
        """
        Pattern match and collect ops starting from given Op.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_on_op(self, op: Op, model: ModelProto, op_quantizers: Dict[str, QcQuantizeOp]):
        """
        Operate on given op.
        """
        raise NotImplementedError

    def apply_on_graph(self, model: ModelProto, graph: ConnectedGraph, op_quantizers: Dict[str, QcQuantizeOp]):
        """
        Iterate over all the ops in ConnectedGraph and call `apply_on_op`

        Args:
            graph (ConnectedGraph): Source Graph
            op_quantizers (Dict[str, QcQuantizeOp]): Activation or param name to QuantizeOp mapping
        """
        for op in graph.ordered_ops:
            self.apply_on_op(op, model, op_quantizers)

    def __call__(self, model: ModelProto, graph: ConnectedGraph, op_quantizers: Dict[str, QcQuantizeOp]):
        """
        Entry function to iterate over Ops from ConnectedGraph and apply pattern match
        Args:
            graph (ConnectedGraph): Input ConnectedGraph
            op_quantizers (Dict[str, QcQuantizeOp]): Global map of QcQuantizeOp
        """
        self.apply_on_graph(model, graph, op_quantizers)


class SupergroupGraphPass(GraphPass):
    """
    GraphPass Utility to modify QcQuantizeOp from ConnectedGraph

    Flow:
        1. Iterate over each Op in ConnectedGraph
        2. Check for pattern match
            - Specified by `match_pattern`
            - User to inherit `SupergroupGraphPass` and specify `match_pattern` for specific sub-graph
        3. If pattern matches, calls `disable_quantizers`
            - This by default disables quantization for all collected quantizer names
            - If intent is to set quantization options differently, override `disable_quantizers` 

    How to use SupergroupGraphPass?
        1. Inherit SupergroupGraphPass and implement `match_pattern` to capture
            - Pattern of interest
            - Ops to disable output quantizers
        2. If disabling output quantization for intermediate ops, you are done and can skip 3.
        3. If need special handling for quantizers, override `disable_quantizers`
            - In this case, you can capture more variables in `match_pattern` as required
    """
    def __init__(self):
        """
        Collect quantizer names to disable quantization
        """
        self.disable_quantizers : List[str] = []

    @abstractmethod
    def match_pattern(self, op: Op, model: ModelProto):
        """
        Pattern match and collect ops starting from given Op.
        """
        raise NotImplementedError

    def apply_on_op(self, op: Op, model: ModelProto, op_quantizers: Dict[str, QcQuantizeOp]):
        """
        Check for pattern match for given Op.
        If pattern matches, then invoke disable_quantizers with collected candidate nodes.

        Args:
            op (Op): Op to check for pattern match
            op_quantizers (Dict[str, QcQuantizeOp]): Global map of QcQuantizeOp
        """
        if self.match_pattern(op, model):
            self.update_quantizers(op_quantizers)

    def disable_output_quantizers(self, op_list: List[Op]):
        """
        Adds output names of given ops to list for disabling quantization later, when pattern is successfully matched.

        Args:
            op_list (List[Op]): List of ops to disable output quantizers for
        """
        self.disable_quantizers.extend(get_output_names(op_list))

    def disable_const_quantizers(self, op_list: List[Op]):
        """
        Adds constant input names of given ops to list for disabling quantization later, when pattern is successfully matched.

        Args:
            op_list (List[Op]): List of ops to disable constant input quantizers for
        """
        self.disable_quantizers.extend(get_const_input_names(op_list))

    def update_quantizers(self, op_quantizers:Dict[str, QcQuantizeOp]):
        """
        Disable quantization for collected quantizers during match_pattern

        Args:
            op_quantizers (Dict[str, QcQuantizeOp]): Global map of QcQuantizeOp
        """
        for name in self.disable_quantizers:
            if name in op_quantizers:
                op_quantizers[name].enabled = False
