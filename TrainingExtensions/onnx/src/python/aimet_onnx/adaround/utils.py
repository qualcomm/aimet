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
"""Utilities for Adaround ONNX"""

from typing import Dict
from collections import defaultdict
import onnx
import torch
from packaging import version

from aimet_onnx import QuantizationSimModel

# pylint: disable=no-name-in-module, ungrouped-imports

# The following modules with weights are supported by Adaround
AdaroundSupportedModules = ["Conv", "ConvTranspose", "MatMul", "Gemm"]
ACTIVATION_MAP = {
    "Relu": torch.nn.ReLU(),
    "PRelu": torch.nn.PReLU(),
    "Tanh": torch.nn.Tanh(),
    "Clip": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Softmax": torch.nn.Softmax(),
}


class ModuleInfo:
    """Class object containing information about a module"""

    def __init__(self):
        self.params = {}
        self.inputs = []
        self.outputs = []
        self.type = None
        self.attributes = None
        self.transposed_params = False


class ModelData:
    """
    Class to collect data for each module of a class
    """

    def __init__(self, quant_sim: QuantizationSimModel):
        """
        :param quant_sim: quantization sim model object
        """
        self.quant_sim = quant_sim
        self.module_to_info = {}
        self._populate_model_data()

    def _populate_model_data(self):
        for op in self.quant_sim.connected_graph.ordered_ops:
            self.module_to_info[op.name] = ModuleInfo()
            if op.type in AdaroundSupportedModules:
                self.module_to_info[op.name].type = op.type
                self.module_to_info[op.name].transposed_params = op.transposed_params
                if hasattr(op.get_module(), "attribute"):
                    self.module_to_info[op.name].attributes = op.get_module().attribute
            for param, param_type in op.parameters.values():
                self.module_to_info[op.name].params[param_type] = param
        for node in self.quant_sim.model.nodes():
            if node.name in self.module_to_info:
                module_info = self.module_to_info[node.name]
                param = {param.name for param in module_info.params.values()}
                for input_name in node.input:
                    if input_name not in param:
                        module_info.inputs.append(input_name)
                for output_name in node.output:
                    module_info.outputs.append(output_name)


def read_attributes_for_op(module_info: ModuleInfo) -> Dict:
    """
    For every op populate it's attributes

    :param module_info: Information about each module
    :return attributes
    """
    attributes = defaultdict(None)
    module_info_attribute = module_info.attributes
    if module_info.type in ["Conv", "ConvTranspose"]:
        for attribute in module_info_attribute:
            if attribute.name == "dilations":
                attributes["dilations"] = list(attribute.ints)
            elif attribute.name == "pads":
                attributes["pads"] = list(attribute.ints)
            elif attribute.name == "strides":
                attributes["strides"] = list(attribute.ints)
            elif attribute.name == "group":
                attributes["group"] = attribute.i
    return attributes


def apply_activation_fn(
    activation_fn: str, activation_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Given the input tensor, apply the activation function

    :param activation_fn: activation function to be applied. If None, no activation function is applied
    :param activation_tensor: input tensor to be passed through the activation function
    :return: output tensor
    """
    module = ACTIVATION_MAP.get(activation_fn)
    if module:
        module = module.to(activation_tensor.device)
        return module(activation_tensor)
    else:
        return activation_tensor
