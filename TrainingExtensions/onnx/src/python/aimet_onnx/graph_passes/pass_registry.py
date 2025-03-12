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

from aimet_onnx.graph_passes.graph_pass import GraphPass
from typing import Dict, List
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.utils import ModelProto


class PassRegistry:
    """
    Registry for Graph passes.
    """
    def __init__(self):
        self.passes: Dict[str, GraphPass] = {}

    def __getitem__(self, name:str) -> GraphPass:
        """
        return Graph Pass class associated with given pass name
        """
        if name in self.passes:
            return self.passes[name]()
        raise KeyError(f"Pass {name} not found.")

    def __contains__(self, name: str) -> bool:
        """
        Check if given pass is registered.
        """
        return name in self.passes

    def register(self, pass_cls: GraphPass, name: str, override: bool=False):
        """
        Register Graph Pass

        Args:
            pass_cls (GraphPass): GraphPass class being registered.
            name (str): Pass name to register graph pass with. 
            override (bool, optional): Override existing pass if set. Defaults to False.

        Raises:
            RuntimeError: If override is not set and GraphPass already exists.
        """
        if name in self.passes and not override:
            raise RuntimeError(f"Pass {name} is already registered. Consider using override if intent to replace default pass.")
        self.passes[name] = pass_cls

# Global Pass Registry to hold all graph passes
PASS_REGISTRY = PassRegistry()

def register_pass(name: str, override: bool = False):
    """
    Decorate to register graph pass

    Args:
        name (str): Pass name to register graph pass with.
        override (bool, optional): Override pass if already registered. Defaults to False.
    """
    def wrapper(pass_cls: GraphPass):
        PASS_REGISTRY.register(pass_cls, name, override)
        return pass_cls

    return wrapper


def apply_graph_passes(model: ModelProto, connected_graph:ConnectedGraph, op_to_quantizers:Dict[str, QcQuantizeOp], passes_to_run:List[str]):
    """
    Runs list of graph passes on input ConnectedGraph

    Args:
        connected_graph (ConnectedGraph): Input graph to run graph passes on
        op_to_quantizers (Dict[str, QcQuantizeOp]): Global map of Quantization ops.
        passes_to_run (List[str]): List of graph passes to run.

    Raises:
        ValueError: If requested GraphPass does not exists.
    """
    for p in passes_to_run:
        if p in PASS_REGISTRY:
            PASS_REGISTRY[p](model, connected_graph, op_to_quantizers)
        else:
            raise ValueError(f"Graph pass requested but not found: {p}")
