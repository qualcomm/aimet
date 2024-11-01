# -*- mode: python -*-
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
#  =============================================================================

""" Winnow the API provided input channels from the modules in a model. """

import copy
from typing import List, Tuple, Dict
import torch
from aimet_common.utils import AimetLogger, ModelApi
from aimet_common.winnow.mask_propagation_winnower import MaskPropagationWinnower as AimetCommonMaskPropagationWinnower
from aimet_common.winnow.mask_propagator import MaskPropagator
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.utils import get_layer_name, has_hooks
from aimet_torch.winnow.module_reducer import ModuleReducer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class MaskPropagationWinnower(AimetCommonMaskPropagationWinnower):
    """ The MaskPropagationWinnower class implements winnowing based on propagating masks corresponding to each
    module's input channels identified to be winnowed.  """

    def __init__(self, model: torch.nn.Module, input_shape: Tuple,
                 list_of_modules_to_winnow: List[Tuple[torch.nn.Module, List]] = None, reshape=True,
                 in_place=False, verbose=False):
        """
        MaskPropagationWinnower object initialization.
        :param model: The model to be winnowed.
        :param input_shape: The input shape of the model.
        :param list_of_modules_to_winnow: A list of Tuples with each Tuple containing a module and a list of
        channels to be winnowed for that module.
        :param reshape: If set to True a Down Sample Layer is added between modules to match the number of channels.
                    If set to False, the modules that need a Down Sample Layer will not be winnowed.
        :param in_place: If set to True, the model will be winnowed in place.
                     If set to False, a copy of the model will be winnowed.
        :param verbose: If set to True, logs detailed winnowing log messages.
        """

        super().__init__(list_of_modules_to_winnow, reshape, in_place, verbose)

        if any(has_hooks(module) for module in model.modules()):
            logger.warning("The specified model has registered hooks which might break winnowing")

        debug_level = logger.getEffectiveLevel()
        logger.debug("Current log level: %s", debug_level)

        self._using_cuda = next(model.parameters()).is_cuda

        if self._in_place is False:
            # Do not winnow the model in place
            self._model = copy.deepcopy(model)
            logger.info("A copy of the model will be winnowed")
        else:
            # Winnow the model in place
            logger.info("Model will be winnowed in place")
            self._model = model

        # Construct connected graph representation of the computational graph
        dummy_input = torch.rand(input_shape)
        if self._using_cuda:
            dummy_input = torch.tensor(dummy_input).cuda()  # pylint: disable=not-callable

        self._graph = ConnectedGraph(self._model, (dummy_input,))
        self.list_of_modules_to_winnow_with_names = \
            generate_and_add_module_winnow_list_with_names(model, self._list_of_modules_to_winnow)
        self._mask_propagator = MaskPropagator(self._graph, ModelApi.pytorch)
        self._module_reducer = ModuleReducer(self._model, self._using_cuda, self._reshape,
                                             self._mask_propagator.op_to_mask_dict)

    def propagate_masks_and_winnow(self):
        """  For the modules to be winnowed, create and propagate the masks.
        Once mask propagation is completed, winnow the model. """

        # Propagate the masks
        self._propagate_masks()

        modified_op_list = self._mask_propagator.get_ops_with_non_default_ip_op_masks()
        for name in modified_op_list:
            logger.info("Modified Op: %s", name)

        modified_modules_dict = self._module_reducer.reduce_modules(modified_op_list)

        if modified_modules_dict:
            ordered_module_list = self._create_modified_modules_list(modified_modules_dict)
        else:
            ordered_module_list = None
            logger.info("No modules were winnowed. Original model is returned.")

        return self._model, ordered_module_list

    def _propagate_masks(self):
        """  For the modules to be winnowed, set the channels to winnow and propagate the masks."""
        for module, list_of_channels_to_winnow, name in self.list_of_modules_to_winnow_with_names:
            self.validate_winnow_api_parameters(module, name, list_of_channels_to_winnow)

            input_channels_to_winnow = list_of_channels_to_winnow
            output_channels_to_winnow = None
            if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
                self._mask_propagator.update_channels_to_winnow(name, self._reshape, input_channels_to_winnow,
                                                                output_channels_to_winnow)

        # The channels to winnow have been updated
        # Propagate the masks.
        self._mask_propagator.propagate_masks()

    @staticmethod
    def _create_modified_modules_list(modified_modules: Dict[str, torch.nn.Module]):
        """ Creates and returns a list of tuples with each tuple containing
        the original module and its replacement module
        :param modified_modules: dictionary of modules modified during module reduction
        :return list of tuples of name of the original module in the model and corresponding new module
        """

        modified_module_list = []
        for orig_module_name, new_module in modified_modules.items():
            # Remove prefix of the model name
            # E.g. the module_name maybe Net.layer1.conv1, we only want layer1.conv1
            first_dot_position = orig_module_name.find('.')
            if first_dot_position != -1:
                orig_module_name = orig_module_name[first_dot_position + 1:]
            modified_module_list.append((orig_module_name, new_module))

        return modified_module_list

    def validate_winnow_api_parameters(self, module, name, list_of_channels_to_winnow):
        """
        For a given module, validate Winnow API parameters.
        :param module: module whose channel numbers are being validated.
        :param name: module's name
        :param list_of_channels_to_winnow: list of channels that must be winnowed.
        """

        if not isinstance(module, torch.nn.Conv2d):
            error_msg = (f'Winnowing is currently only supported for torch.nn.Conv2d modules. Attempting to winnow '
                         f'module of type {type(module)}')
            logger.error(error_msg)
            raise NotImplementedError(error_msg)

        # Validate the list of channels.
        num_channels_to_winnow = len(list_of_channels_to_winnow)
        if num_channels_to_winnow == 0:
            raise ValueError("The list of channels to winnow is empty for the module: %s" % name)

        max_channel_num = max(list_of_channels_to_winnow)
        max_in_channel_index = (module.in_channels - 1)
        if max_channel_num > max_in_channel_index:
            raise ValueError("Channel number: %s exceeds module's max channel number index: %s for module: %s" %
                             (max_channel_num, max_in_channel_index, name))

        if num_channels_to_winnow == module.in_channels:
            raise ValueError("Winnowing all the input channels is not allowed, module: %s" % name)

        module_op = self._graph.get_op_from_module_name(name)
        input_index = 0     # Using op index 0 to examine input to op
        if module_op.inputs[input_index].is_model_input:
            error_msg = (f'Winnowing the first module of a model is NOT supported. Please ignore the first '
                         f'module and try again. First module: {module_op.dotted_name}, shape '
                         f'{module_op.inputs[input_index].shape}, channels to winnow: {list_of_channels_to_winnow}')
            logger.error(error_msg)
            raise NotImplementedError(error_msg)


def generate_and_add_module_winnow_list_with_names(model: torch.nn.Module,
                                                   list_of_modules_to_winnow: List[Tuple[torch.nn.Module, List[int]]]):
    """
    Generates the module names for the modules to winnow.
    Creates a Tuple with Module, list of channels to winnow and Module Name
    and adds the Tuple to a list for later use.

    :param model: model for which to compare modules from
    :param list_of_modules_to_winnow: List of tuples of modules to winnow and corresponding channels to winnow
    :return: list of module information.
    """

    list_of_module_info = []
    if list_of_modules_to_winnow is not None:

        model_name = type(model).__name__
        logger.debug("Model name: %s", model_name)

        for module, list_of_channels_to_winnow, in list_of_modules_to_winnow:
            name = get_layer_name(model, module)

            # This name doesn't contain the model's name.
            # Prepend the model's name to the module's name.
            name = '.'.join([model_name, name])

            mod_tuple = (module, list_of_channels_to_winnow, name)
            list_of_module_info.append(mod_tuple)

    return list_of_module_info
