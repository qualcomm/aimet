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

""" This module contains utilities to capture and save intermediate layer-outputs of a model. """

import os
from typing import Union, Dict, List, Tuple
from enum import Enum
import shutil
import re

import numpy as np
import onnx
import torch

from aimet_common.utils import AimetLogger
from aimet_common.layer_output_utils import SaveInputOutput, save_layer_output_names

from aimet_torch._base.quantsim import _QuantizedModuleProtocol
from aimet_torch import utils
from aimet_torch import torchscript_utils
from aimet_torch.onnx_utils import OnnxSaver, OnnxExportApiArgs
from aimet_torch.v2.nn.base import BaseQuantizationMixin

try:
    from aimet_torch.v1.qc_quantize_recurrent import QcQuantizeRecurrent
    _quantized_module_types = (_QuantizedModuleProtocol, QcQuantizeRecurrent)
except ImportError:
    _quantized_module_types = (_QuantizedModuleProtocol,)


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)


class NamingScheme(Enum):
    """ Enumeration of layer-output naming schemes. """

    PYTORCH = 1
    """ Names outputs according to exported pytorch model. Layer names are used. """
    ONNX = 2
    """ Names outputs according to exported onnx model. Layer output names are generally numeric. """
    TORCHSCRIPT = 3
    """ Names outputs according to exported torchscript model. Layer output names are generally numeric. """


class LayerOutputUtil:
    """ Implementation to capture and save outputs of intermediate layers of a model (fp32/quantsim). """

    def __init__(self, model: torch.nn.Module, dir_path: str, naming_scheme: NamingScheme = NamingScheme.PYTORCH,
                 dummy_input: Union[torch.Tensor, Tuple, List] = None, onnx_export_args: Union[OnnxExportApiArgs, Dict] = None):
        """
        Constructor for LayerOutputUtil.

        :param model: Model whose layer-outputs are needed.
        :param dir_path: Directory wherein layer-outputs will be saved.
        :param naming_scheme: Naming scheme to be followed to name layer-outputs. There are multiple schemes as per
            the exported model (pytorch, onnx or torchscript). Refer the NamingScheme enum definition.
        :param dummy_input: Dummy input to model. Required if naming_scheme is 'NamingScheme.ONNX' or 'NamingScheme.TORCHSCRIPT'.
        :param onnx_export_args: Should be same as that passed to quantsim export API to have consistency between
            layer-output names present in exported onnx model and generated layer-outputs. Required if naming_scheme is
            'NamingScheme.ONNX'.
        """

        # Utility to capture layer-outputs
        self.layer_output = LayerOutput(model=model, naming_scheme=naming_scheme, dir_path=dir_path, dummy_input=dummy_input,
                                        onnx_export_args=onnx_export_args)

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_input_output = SaveInputOutput(dir_path=dir_path, axis_layout='NCHW')

    def generate_layer_outputs(self, input_instance: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]):
        """
        This method captures output of every layer of a model & saves the single input instance and corresponding layer-outputs to disk.

        :param input_instance: Single input instance for which we want to obtain layer-outputs.
        :return: None
        """

        logger.info("Generating layer-outputs for input instance %d", self.save_input_output.input_cntr+1)

        # Obtain layer-output name to output dictionary
        layer_output_dict = self.layer_output.get_outputs(input_instance)

        # Place inputs and layer-outputs on CPU
        input_instance = LayerOutputUtil._get_input_in_numpy(input_instance)
        layer_output_dict = LayerOutputUtil._get_layer_output_in_numpy(layer_output_dict)

        # Save inputs and layer-outputs
        self.save_input_output.save(input_instance, layer_output_dict)

        logger.info('Successfully generated layer-outputs for input instance %d', self.save_input_output.input_cntr)

    @staticmethod
    def _get_input_in_numpy(input_instance: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> \
            Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]:
        """
        Coverts the torch tensors into numpy arrays
        :param input_instance: Single input instance with torch tensors
        :return: Input with numpy arrays
        """
        if isinstance(input_instance, (List, Tuple)):
            numpy_input = []
            for ith_input in input_instance:
                numpy_input.append(ith_input.cpu().numpy())
            return numpy_input
        return input_instance.cpu().numpy()

    @staticmethod
    def _get_layer_output_in_numpy(layer_output_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Converts the torch tensors into numpy arrays
        :param layer_output_dict: layer output dictionary with torch tensors
        :return: layer output dictionary with numpy arrays
        """
        layer_output_numpy_dict = {}
        for output_name, output_tensor in layer_output_dict.items():
            layer_output_numpy_dict[output_name] = output_tensor.cpu().numpy()
        return layer_output_numpy_dict


class LayerOutput:
    """
    This class creates a layer-output name to layer-output dictionary. The layer-output names are as per the AIMET exported
    pytorch/onnx/torchscript model.
    """
    def __init__(self, model: torch.nn.Module, dir_path: str, naming_scheme: NamingScheme = NamingScheme.PYTORCH,
                 dummy_input: Union[torch.Tensor, Tuple, List] = None, onnx_export_args: Union[OnnxExportApiArgs, Dict] = None):
        """
        Constructor - It initializes few dictionaries that are required for capturing and naming layer-outputs.

        :param model: Model whose layer-outputs are needed.
        :param dir_path: Directory wherein layer-output names arranged in topological order will be saved. It will also
            be used to temporarily save onnx/torchscript equivalent of the given model.
        :param naming_scheme: Naming scheme to be followed to name layer-outputs. There are multiple schemes as per
            the exported model (pytorch, onnx or torchscript). Refer the NamingScheme enum definition.
        :param dummy_input: Dummy input to model (required if naming_scheme is 'onnx').
        :param onnx_export_args: Should be same as that passed to quantsim export API to have consistency between
            layer-output names present in exported onnx model and generated layer-outputs (required if naming_scheme is
            'onnx').
        """
        self.model = model
        self.module_to_name_dict = utils.get_module_to_name_dict(model=model, prefix='')

        # Check whether the given model is quantsim model
        self.is_quantsim_model = any(isinstance(module, _quantized_module_types) for module in model.modules())

        # Obtain layer-name to layer-output name mapping
        self.layer_name_to_layer_output_dict = {}
        self.layer_name_to_layer_output_name_dict = {}
        if naming_scheme == NamingScheme.PYTORCH:
            for name, module in model.named_modules():
                if utils.is_leaf_module(module) or isinstance(module, BaseQuantizationMixin):
                    name = name.replace('._module_to_wrap', '')
                    self.layer_name_to_layer_output_name_dict[name] = name
        else:
            self.layer_name_to_layer_output_name_dict = LayerOutput.get_layer_name_to_layer_output_name_map(
                self.model, naming_scheme, dummy_input, onnx_export_args, dir_path)

        # Replace any delimiter in layer-output name string with underscore
        for layer_name, output_name in self.layer_name_to_layer_output_name_dict.items():
            self.layer_name_to_layer_output_name_dict[layer_name] = re.sub(r'\W+', "_", output_name)

        # Save layer-output names which are in topological order of model graph. This order can be used while comparing layer-outputs.
        layer_output_names = list(self.layer_name_to_layer_output_name_dict.values())
        save_layer_output_names(layer_output_names, dir_path)

    def get_outputs(self, input_instance: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        This function captures layer-outputs and renames them as per the AIMET exported pytorch/onnx/torchscript model.

        :param input_instance: Single input instance for which we want to obtain layer-outputs.
        :return: layer-name to layer-output dict
        """

        # Fetch outputs of all the layers
        self.layer_name_to_layer_output_dict = {}
        if self.is_quantsim_model:
            # Apply record-output hook to QuantizeWrapper modules (one node above leaf node in model graph)
            utils.run_hook_for_layers_with_given_input(self.model, input_instance, self.record_outputs,
                                                       module_type_for_attaching_hook=_quantized_module_types,
                                                       leaf_node_only=False)
        else:
            # Apply record-output hook to Original modules (leaf node in model graph)
            utils.run_hook_for_layers_with_given_input(self.model, input_instance, self.record_outputs, leaf_node_only=True)

        # Rename outputs according to pytorch/onnx/torchscript model
        layer_output_name_to_layer_output_dict = LayerOutput.rename_layer_outputs(self.layer_name_to_layer_output_dict,
                                                                                  self.layer_name_to_layer_output_name_dict)

        return layer_output_name_to_layer_output_dict

    def record_outputs(self, module: torch.nn.Module, _, output: torch.Tensor):
        """
        Hook function to capture output of a layer.

        :param module: Layer-module in consideration.
        :param _: Placeholder for the input of the layer-module.
        :param output: Output of the layer-module.
        :return: None
        """
        layer_name = self.module_to_name_dict[module]
        if isinstance(output, torch.Tensor):
            self.layer_name_to_layer_output_dict[layer_name] = output.clone()
        else:
            logger.info("Skipping constant scalar output of layer %s", layer_name)

    @staticmethod
    def rename_layer_outputs(layer_name_to_layer_output_dict: Dict[str, torch.Tensor],
                             layer_name_to_layer_output_name_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Rename layer-outputs based on the layer-name to layer-output name map

        :param layer_name_to_layer_output_dict: Dict containing layer-outputs
        :param layer_name_to_layer_output_name_dict: Dict containing layer-output names
        :return: layer_output_name_to_layer_output_dict
        """
        layer_names = list(layer_name_to_layer_output_dict.keys())

        for layer_name in layer_names:
            if layer_name in layer_name_to_layer_output_name_dict:
                # Rename the layer-output by using layer-output name, instead of layer-name
                layer_output_name = layer_name_to_layer_output_name_dict[layer_name]
                layer_name_to_layer_output_dict[layer_output_name] = layer_name_to_layer_output_dict.pop(layer_name)
            else:
                # Delete the layer-output as it doesn't have a name
                layer_name_to_layer_output_dict.pop(layer_name)

        return layer_name_to_layer_output_dict

    @staticmethod
    def get_layer_name_to_layer_output_name_map(model, naming_scheme: NamingScheme, dummy_input: Union[torch.Tensor, Tuple, List],
                                                onnx_export_args: Union[OnnxExportApiArgs, Dict], dir_path: str) -> Dict[str, str]:
        """
        This function produces layer-name to layer-output name map w.r.t the AIMET exported onnx/torchscript model. If a
        layer gets expanded into multiple layers in the exported model then the intermediate layers are ignored and
        output-name of last layer is used.

        :param model: model
        :param naming_scheme: onnx/torchscript
        :param dummy_input: dummy input that is used to construct onnx/torchscript model
        :param onnx_export_args: OnnxExportApiArgs instance same as that passed to quantsim export API
        :param dir_path: directory to temporarily save the constructed onnx/torchscrip model
        :return: dictionary of layer-name to layer-output name
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        if any(isinstance(module, BaseQuantizationMixin) for module in model.modules()):
            from aimet_torch.v2.quantsim import QuantizationSimModel
        else:
            from aimet_torch.v1.quantsim import QuantizationSimModel

        # Restore original model by removing quantization wrappers if present.
        original_model = QuantizationSimModel.get_original_model(model)

        # Set path to store exported onnx/torchscript model.
        LayerOutput._validate_dir_path(dir_path)
        exported_model_dir = os.path.join(dir_path, 'exported_models')
        os.makedirs(exported_model_dir, exist_ok=True)

        # Get node to i/o tensor name map from the onnx/torchscript model
        if naming_scheme == NamingScheme.ONNX:
            exported_model_node_to_io_tensor_map = LayerOutput.get_onnx_node_to_io_tensor_map(
                original_model, exported_model_dir, dummy_input, onnx_export_args)
        else:
            exported_model_node_to_io_tensor_map = LayerOutput.get_torchscript_node_to_io_tensor_map(
                original_model, exported_model_dir, dummy_input)

        layer_names_list = [name for name, module in original_model.named_modules() if utils.is_leaf_module(module)]
        layers_missing_in_exported_model = []
        layer_name_to_layer_output_name_map = {}

        # Get mapping between layer names and layer-output names.
        logger.info("Layer Name to Layer Output-name Mapping")
        # pylint: disable=protected-access
        for layer_name in layer_names_list:
            if layer_name in exported_model_node_to_io_tensor_map:
                # pylint: disable=protected-access, unused-variable
                layer_output_names, intermediate_layer_output_names = QuantizationSimModel._get_layer_activation_tensors(
                    layer_name, exported_model_node_to_io_tensor_map)
                layer_name_to_layer_output_name_map[layer_name] = layer_output_names[0]
                logger.info("%s -> %s", layer_name, layer_output_names[0])
            else:
                layers_missing_in_exported_model.append(layer_name)

        if layers_missing_in_exported_model:
            logger.warning("The following layers were not found in the exported model:\n"
                           "%s\n"
                           "This can be due to below reason:\n"
                           "\t- The layer was not seen while exporting using the dummy input provided in sim.export(). "
                           "Ensure that the dummy input covers all layers.",
                           layers_missing_in_exported_model)

        # Delete onnx/torchscript models
        shutil.rmtree(exported_model_dir, ignore_errors=False, onerror=None)

        return layer_name_to_layer_output_name_map

    @staticmethod
    def get_onnx_node_to_io_tensor_map(model: torch.nn.Module, exported_model_dir: str, dummy_input: Union[torch.Tensor, Tuple, List],
                                       onnx_export_args: Union[OnnxExportApiArgs, Dict]) -> Dict[str, Dict]:
        """
        This function constructs an onnx model equivalent to the give pytorch model and then generates node-name to i/o
        tensor-name map.
        :param model: pytorch model without quantization wrappers
        :param exported_model_dir: directory to save onnx model
        :param dummy_input: dummy input to be used for constructing onnx model
        :param onnx_export_args: configurations to generate onnx model
        :return: onnx_node_to_io_tensor_map
        """
        LayerOutput._validate_dummy_input(dummy_input)
        LayerOutput._validate_onnx_export_args(onnx_export_args)

        onnx_path = os.path.join(exported_model_dir, 'model.onnx')

        OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_model_path=onnx_path, pytorch_model=model,
                                                             dummy_input=dummy_input, onnx_export_args=onnx_export_args)
        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, _ = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        return onnx_node_to_io_tensor_map

    @staticmethod
    def get_torchscript_node_to_io_tensor_map(model: torch.nn.Module, exported_model_dir: str,
                                              dummy_input: Union[torch.Tensor, Tuple, List]) -> Dict[str, Dict]:
        """
        This function constructs a torchscript model equivalent to the give pytorch model and then generates node-name to i/o
        tensor-name map.
        :param model: pytorch model without quantization wrappers
        :param exported_model_dir: directory to save onnx model
        :param dummy_input: dummy input to be used for constructing onnx model
        :return: torchscript_node_to_io_tensor_map
        """
        LayerOutput._validate_dummy_input(dummy_input)

        ts_path = os.path.join(exported_model_dir, 'model.torchscript.pth')

        with utils.in_eval_mode(model), torch.no_grad():
            torchscript_utils.create_torch_script_model(ts_path, model, dummy_input)
            trace = torch.jit.load(ts_path)
            torch_script_node_to_io_tensor_map, _ = \
                torchscript_utils.get_node_to_io_tensor_names_map(model, trace, dummy_input)

        return torch_script_node_to_io_tensor_map

    @staticmethod
    def _validate_dir_path(dir_path: str):
        """
        Validate directory path in which onnx/torchscript models will be temporarily saved
        :param dir_path: directory path
        :return:
        """
        if dir_path is None:
            raise ValueError("Missing directory path to save onnx/torchscript models")

    @staticmethod
    def _validate_dummy_input(dummy_input: Union[torch.Tensor, Tuple, List]):
        """
        Validates dummy input which is used to generate onnx/torchscript model
        :param dummy_input: single input instance
        :return:
        """
        if not isinstance(dummy_input, (torch.Tensor, tuple, list)):
            raise ValueError("Invalid dummy_input data-type")

    @staticmethod
    def _validate_onnx_export_args(onnx_export_args: Union[OnnxExportApiArgs, Dict]):
        """
        Validates export arguments which are used to generate an onnx model
        :param onnx_export_args: export arguments
        :return:
        """
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()
        if not isinstance(onnx_export_args, (OnnxExportApiArgs, dict)):
            raise ValueError("Invalid onnx_export_args data-type")
