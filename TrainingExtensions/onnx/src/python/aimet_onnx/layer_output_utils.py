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

""" This module contains utilities to capture and save intermediate layer-outputs of a model """

import copy
from typing import List, Dict, Tuple, Union
import re
import numpy as np
import onnxruntime as ort
import onnx
from packaging import version

# pylint: disable=wrong-import-order
from aimet_common.utils import AimetLogger
from aimet_common.layer_output_utils import SaveInputOutput, save_layer_output_names

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import create_input_dict, add_hook_to_get_activation

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)


class LayerOutputUtil:
    """ Implementation to capture and save outputs of intermediate layers of a model (fp32/quantsim) """

    def __init__(self, model: ModelProto, dir_path: str, device: int = 0):
        """
        Constructor - It initializes the utility classes that captures and saves layer-outputs

        :param model: ONNX model
        :param dir_path: Directory wherein layer-outputs will be saved
        :param device: CUDA device-id to be used
        """
        self.model = model

        # Fetch appropriate execution providers depending on availability
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'device_id': device}), 'CPUExecutionProvider']

        # Utility to capture layer-outputs
        self.layer_output = LayerOutput(model=model, providers=providers, dir_path=dir_path)

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_input_output = SaveInputOutput(dir_path, 'NCHW')

    def generate_layer_outputs(self, input_instance: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]):
        """
        This method captures output of every layer of a model & saves the inputs and corresponding layer-outputs to disk.

        :param input_instance: Single input instance for which we want to obtain layer-outputs.
        :return: None
        """
        logger.info("Generating layer-outputs for input instance %d", self.save_input_output.input_cntr+1)

        input_dict = create_input_dict(self.model, input_instance)

        layer_output_dict = self.layer_output.get_outputs(input_dict)
        self.save_input_output.save(input_instance, layer_output_dict)

        logger.info('Layer-outputs generated for input instance %d', self.save_input_output.input_cntr)


class LayerOutput:
    """
    This class creates a layer-output name to layer-output dictionary.
    """
    def __init__(self, model: ModelProto, providers: List, dir_path: str):
        """
        Constructor - It initializes few lists that are required for capturing and naming layer-outputs.

        :param model: ONNX model
        :param providers: execution providers to execute onnxruntime
        :param dir_path: directory to store topological order of layer-output names
        """
        self.model = copy.deepcopy(model)
        self.activation_names = LayerOutput.get_activation_names(self.model)

        quantized_activation_names = [name for name in self.activation_names if name.endswith('_updated')]
        if quantized_activation_names:
            self.activation_names = quantized_activation_names

        LayerOutput.register_activations(self.model, self.activation_names)

        self.session = QuantizationSimModel.build_session(self.model, providers)

        # Replace special characters with underscore. This gives valid file names to store activation tensors.
        self.sanitized_activation_names = [re.sub(r'\W+', "_", name.replace('_updated', '')) for name in self.activation_names]

        # Save activation names which are in topological order of model graph. This order can be used while comparing layer-outputs.
        save_layer_output_names(self.sanitized_activation_names, dir_path)

    def get_outputs(self, input_dict: Dict) -> Dict[str, np.ndarray]:
        """
        This function creates layer-output name to layer-output dictionary.

        :param input_dict: input name to input tensor map
        :return: layer-output name to layer-output dictionary
        """
        activation_values = self.session.run(self.activation_names, input_dict)
        return dict(zip(self.sanitized_activation_names, activation_values))

    @staticmethod
    def get_activation_names(model: ModelProto) -> List[str]:
        """
        This function fetches the activation names (layer-output names) of the given onnx model.

        :param model: ONNX model
        :return: list of activation names
        """
        activation_names = []
        constant_activations = []  # No need to capture them
        for node in model.graph.input:
            activation_names.append(node.name)
        for node in model.graph.node:
            if node.op_type == 'Constant':
                # Ignore and keep track to further ignore its quantized version
                constant_activations.append(node.output[0]+'_updated')
            else:
                for output in node.output:
                    if output not in constant_activations:
                        activation_names.append(output)
        return activation_names

    @staticmethod
    def register_activations(model: ModelProto, activation_names: List):
        """
        This function adds the intermediate activations into the model's ValueInfoProto so that they can be fetched via
        running the session.

        :param model: ONNX model
        :param activation_names: list of activation names to be registered
        :return:
        """
        for act_name in activation_names:
            _ = add_hook_to_get_activation(model, act_name)
