# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Sample input to a passed module (for our case, it is the quantized wrapper module)"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.utils import (
    StopForwardException,
    change_tensor_device_placement,
    get_device,
    in_eval_mode,
    get_module_to_name_dict,
)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class BlockModuleData:
    """
    Collect input tensor from block-level modules
    """
    def __init__(
        self,
        model: torch.nn.Module,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
        module_names: Optional[List[str]] = None,
    ):
        self._model = model
        self._forward_fn = forward_fn
        self._module_names = list(module_names)
        self._module_to_name = get_module_to_name_dict(model)

    def collect_module_to_input_tensor(
        self,
        model_input: Union[torch.tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collect input tensors corresponding to module names in block level

        :param model_input: Input to model, Can be a single tensor or a list/tuple of tensors
        :return: Dictionary of module to input tensors
        """
        def _hook_to_collect_inp_data(module, inp, _):
            """
            hook to collect input and output data
            """
            inp_data_dict[self._module_to_name[module]] = inp[0].detach()

            if self._module_to_name[module] == self._module_names[-1]:
                raise StopForwardException

        inp_data_dict = {}
        handles = []
        for module_name in self._module_names:
            handles.append(
                self._model.get_submodule(module_name).register_forward_hook(
                    _hook_to_collect_inp_data
                )
            )

        # get the model's device placement information
        device = get_device(self._model)

        # place the input to appropriate device
        model_input = change_tensor_device_placement(model_input, device)

        # Custom injected exception is raised when the activations data from desired module is collected.
        try:
            with in_eval_mode(self._model), torch.no_grad():
                _ = self._forward_fn(self._model, model_input)
        except StopForwardException:
            pass
        finally:
            # remove hook handle
            for handle in handles:
                handle.remove()

        return inp_data_dict


class ActivationSampler:
    """
    For a module in the original model and the corresponding module in the weight quantized QuantSim model,
    collect the module's output and input activation data respectively
    """
    def __init__(self,
                 model: torch.nn.Module,
                 forward_fn: Callable[[torch.nn.Module, Any], Any],
                 module_names: List[str]):
        """
        :param model:  model to run forward pass
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        :param module_names: Topologically ordered module names
        """
        self._model = model
        self._module_collector = BlockModuleData(model, forward_fn, module_names)

    def sample_activation_tensors(self, model_inputs: Union[torch.tensor, List, Tuple]) -> Dict[str, torch.Tensor]:
        """
        For given model_inputs, collect input activations data to quant module

        :param model_inputs: Model inputs.
        :return: Input activations data to the current module

        """
        # Collect input activation data to quantized wrapper module
        # (with all preceding weight modules quantized)
        inp_dict = self._module_collector.collect_module_to_input_tensor(model_inputs)

        return inp_dict
