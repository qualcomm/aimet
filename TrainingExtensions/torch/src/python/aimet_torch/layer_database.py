# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Stores and updates Layer Attributes"""
import copy
from typing import Tuple, Union, List
import torch

from aimet_common.utils import AimetLogger
import aimet_common.layer_database
from aimet_torch import utils
import aimet_torch._base.nn.modules.custom as aimet_modules

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class Layer(aimet_common.layer_database.Layer):
    """ Holds attributes for a given layer """

    def _set_type_specific_params(self, module):

        if isinstance(module, torch.nn.Conv2d):
            params = aimet_common.layer_database.Conv2dTypeSpecificParams(module.stride, module.padding, module.groups)
            self.type_specific_params = params

    def __init__(self, module: torch.nn.Module, name, output_shape: Union[List, Tuple],
                 input_shape: Union[List, Tuple] = None):
        """
        Constructor
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param output_shape: Shape of the output activations
        :param input_shape: Shape of the input activations
        """
        if isinstance(module, torch.nn.Conv2d):
            if module.groups > 1:
                if module.in_channels == module.groups: # Depthwise convolution
                    assert module.in_channels == module.out_channels
                    weight_shape = (module.out_channels, 1, module.kernel_size[0], module.kernel_size[1])
                elif module.in_channels % module.groups == 0: # Grouped convolution
                    weight_shape = (module.out_channels, module.in_channels // module.groups, module.kernel_size[0], module.kernel_size[1])
                else:
                    raise AssertionError("Conv2d with invalid in_channels and groups values")
            else:
                weight_shape = (module.out_channels, module.in_channels, module.kernel_size[0], module.kernel_size[1])

        elif isinstance(module, torch.nn.Linear):
            weight_shape = (module.out_features, module.in_features, 1, 1)
        else:
            logger.info("Module:%s does not have any weight associated. Using default weight dim", name)
            weight_shape = (1,)

        aimet_common.layer_database.Layer.__init__(self, module, name, weight_shape, output_shape, input_shape)

        self.var_name_of_module_in_parent = None
        self.parent_module = None


class LayerDatabase(aimet_common.layer_database.LayerDatabase):
    """
    Stores, creates and updates the Layer database
    Also stores compressible layers to model optimization
    """
    _SUPPORTED_MODULE_TYPES = (
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.Sigmoid,
        torch.nn.AvgPool2d,
        aimet_modules.Multiply,
    )


    def __init__(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]):
        """
        LayerDatabase constructor
        :param model: Model
        :param dummy_input: Dummy input to the model. If the model has more than one input,
                            pass a tuple.
        """
        aimet_common.layer_database.LayerDatabase.__init__(self, model)
        self._create_database(model, dummy_input)

    def __deepcopy__(self, memodict):

        # pylint: disable=protected-access

        # Allocate a new LayerDatabase
        layer_db = copy.copy(self)
        memodict[id(self)] = layer_db

        # Create a deep copy of the model
        layer_db._model = copy.deepcopy(self._model, memodict)

        # Re-create the compressible layers dict
        layer_db._compressible_layers = {}

        modules_in_copy = list(layer_db._model.modules())

        # For all modules in the current model
        for index, module in enumerate(self._model.modules()):

            # If this module is in the current layer database
            if id(module) in self._compressible_layers:
                existing_layer = self._compressible_layers[id(module)]
                new_layer = Layer(modules_in_copy[index], existing_layer.name,
                                  existing_layer.output_shape)
                new_layer.picked_for_compression = existing_layer.picked_for_compression
                layer_db._compressible_layers[id(modules_in_copy[index])] = new_layer

        # Now we need to set parent references
        layer_db.set_reference_to_parent_module(layer_db._model, layer_db._compressible_layers)
        return layer_db

    def replace_layer(self, old_layer: Layer, new_layer: Layer):
        """
        Replace given layer with a new layer in the LayerDatabase
        :param old_layer: Existing Layer
        :param new_layer: New Layer
        """

        del self._compressible_layers[id(old_layer.module)]

        # set parent ref
        new_layer.parent_module = old_layer.parent_module
        new_layer.var_name_of_module_in_parent = old_layer.var_name_of_module_in_parent

        self._compressible_layers[id(new_layer.module)] = new_layer

    def replace_layer_with_sequential_of_two_layers(self, layer_to_replace: Layer,
                                                    layer_a: Layer, layer_b: Layer):
        """
        Replaces a layer with a sequential of layer in the database

        :param layer_to_replace: Layer to be replaced
        :param layer_a: 1st new layer
        :param layer_b: 2nd new layer
        """

        # Create a sequential of these modules
        seq = torch.nn.Sequential(layer_a.module, layer_b.module)

        # Replace the original layer_to_replace in the model with this sequential
        setattr(layer_to_replace.parent_module, layer_to_replace.var_name_of_module_in_parent, seq)

        # Set parent correctly
        layer_a.parent_module = seq
        layer_a.var_name_of_module_in_parent = '0'

        layer_b.parent_module = seq
        layer_b.var_name_of_module_in_parent = '1'

        # Add the new layer to the database
        self._compressible_layers[id(layer_a.module)] = layer_a
        self._compressible_layers[id(layer_b.module)] = layer_b

        # Remove the the layer being replaced from the database
        del self._compressible_layers[id(layer_to_replace.module)]

    def update_layer_with_module_in_sequential(self, layer_to_update: Layer, seq: torch.nn.Sequential):
        """
        Update layer attributes with sequential in database

        :param layer_to_update: Layer to be updated
        :param seq: Sequential of modules in DownSample and layer
        """
        # Remove the layer being updated from the database
        del self._compressible_layers[id(layer_to_update.module)]

        # Find the first conv2d within the sequential
        index, new_module = next((index, module) for (index, module) in enumerate(seq)
                                 if isinstance(module, torch.nn.Conv2d))

        # Determine new output shape
        new_output_shape = [new_module.in_channels, new_module.out_channels,
                            layer_to_update.output_shape[2], layer_to_update.output_shape[3]]
        new_module_name = layer_to_update.name + '.' + str(index)

        # Create a new layer
        new_layer = Layer(new_module, new_module_name, new_output_shape)

        # Set parent correctly
        new_layer.parent_module = seq
        new_layer.var_name_of_module_in_parent = str(index)

        # Add the updated layer to the database
        self._compressible_layers[id(new_layer.module)] = new_layer

    def _custom_hook_to_collect_layer_attributes(self, module, input_tensor, output_tensor):
        """
        Custom hook function which will be applied to all the layers in the model and store following
        information :
        model name (which will be removed), model reference, input shape and output shape
        """
        input_activation_shape = [list(i_input.size()) for i_input in input_tensor]
        output_activation_shape = list(output_tensor.size())
        # activation dimension for FC layer is (1,1)
        if isinstance(module, torch.nn.Linear):
            # In cases where batch dimension is 1
            if len(output_activation_shape) == 1:
                output_activation_shape = [1, *output_activation_shape, 1, 1]
            else:
                output_activation_shape.extend([1, 1])

        module_name = None
        for name, module_ref in self._model.named_modules():
            if module is module_ref:
                module_name = name

        self._compressible_layers[id(module)] = Layer(module, module_name, output_activation_shape,
                                                      input_activation_shape)

    @classmethod
    def set_reference_to_parent_module(cls, module, layers):
        """
        Recursive function to set the parent references for each layer in the database
        :param module: Reference to the parent module
        :param layers: Layers to set reference for
        """
        for module_name, module_ref in module.named_children():
            # first check if the module is leaf module or not
            if isinstance(module_ref, cls._SUPPORTED_MODULE_TYPES):
                # iterate over all the layer attributes and if the match is found
                # then set the parent class and module name for that module
                if id(module_ref) in layers:
                    layer = layers[id(module_ref)]
                    layer.parent_module = module
                    layer.var_name_of_module_in_parent = module_name

            # if module is not leaf, call recursively
            else:
                cls.set_reference_to_parent_module(module_ref, layers)

    def _create_database(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]):
        """
        Register custom hook for the model with run_graph provided by user
        if the user wants to experiment with custom hook, we can support that option by
        exposing the hook parameter to compress_net method
        :param model: Model
        :param dummy_input: Dummy input to the model. If the model has more than one input,
                            pass a tuple.
        """

        utils.run_hook_for_layers_with_given_input(model, dummy_input,
                                                   hook=self._custom_hook_to_collect_layer_attributes,
                                                   module_type_for_attaching_hook=self._SUPPORTED_MODULE_TYPES,
                                                   leaf_node_only=False)

        # set the parent_class reference
        self.set_reference_to_parent_module(self._model, self._compressible_layers)

    def get_compressible_layers(self):
        """
        :return: Returns compressible layers
        """
        return self._compressible_layers

    def destroy(self):
        """
        Destroys the layer database
        """
        # clear the dictionary
        self._compressible_layers.clear()
        self._model = None
