# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Sample input to quantized wrapper module and output from original module for Adaround feature """
from typing import Tuple, Union, List, Callable, Any, Dict, Type, Optional
import torch
from torch.utils.data import Dataset

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.utils import CachedDataset, ModuleData, get_named_module, cache_intermediate_datasets,\
    change_tensor_device_placement, in_eval_mode, save_to_cache, get_ordered_list_of_modules
from aimet_torch._base.quantsim import _QuantizationSimModelInterface, _QuantizedModuleProtocol

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def create_modulelist_for_group_modules(model: torch.nn.Module, sim: _QuantizationSimModelInterface, grouped_modules: Dict)\
        -> Tuple[List[torch.nn.ModuleList], List[torch.nn.ModuleList]]:
    """
    Use torch.nn.ModuleList to group modules from a single block.

    :param model: FP32 model
    :param sim: QuantizationSimModel object
    :param grouped_modules: Group modules
    :return: List of modulelist for FP32 and quant models
    """
    sub_fp_models = []
    sub_sim_models = []
    for _, modules in grouped_modules.items():
        fp_modulelist = torch.nn.ModuleList()
        quant_modulelist = torch.nn.ModuleList()
        for name in modules:
            fp_modulelist.append(get_named_module(model, name))
            quant_modulelist.append(get_named_module(sim.model, name))
        sub_fp_models.append(fp_modulelist)
        sub_sim_models.append(quant_modulelist)

    return sub_fp_models, sub_sim_models


def get_block_inputs(model: torch.nn.Module, sim: _QuantizationSimModelInterface,
                     breakpoint_module_name: str, cached_dataset: CachedDataset,
                     cache_on_cpu: bool, forward_fn: Callable, num_batches: int, working_dir: str,
                     incl_kwargs: bool = False) \
        -> Union[Tuple[List, List], Tuple[CachedDataset, CachedDataset]]:
    # pylint: disable=too-many-arguments
    """
    Get inputs to block/module from FP32 and QuantizationSimModel models

    :param model: FP32 model
    :param sim: QuantizationSimModel object
    :param breakpoint_module_name: Breakpoint block/module name
    :param cached_dataset: Cached dataset
    :param cache_on_cpu: Whether to cache intermediate data on CPU or store to disk
    :param forward_fn: adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model
      as second argument.
    :param num_batches: Number of batches
    :param working_dir: Working to directory to save block inputs data to disk
    :param incl_kwargs: if True, capture kwargs, normalize and attach to inputs.
    :return: Inputs to block from FP32 and QuantizationSimModel models
    """
    # Cache input data to first block from both FP32 and quant models
    if cache_on_cpu:
        cached_fp_dataset = cache_intermediate_datasets(cached_dataset, cache_on_cpu, model,
                                                        breakpoint_module_name, forward_fn, incl_kwargs=incl_kwargs)
        cached_quant_dataset = cache_intermediate_datasets(cached_dataset, cache_on_cpu,
                                                           sim.model, breakpoint_module_name, forward_fn,
                                                           incl_kwargs=incl_kwargs)
    else:
        fp32_cache_path = working_dir + 'fp32/'
        quant_cache_path = working_dir + 'quant/'
        cache_intermediate_datasets(cached_dataset, cache_on_cpu, model, breakpoint_module_name,
                                    forward_fn, fp32_cache_path, incl_kwargs=incl_kwargs)
        cache_intermediate_datasets(cached_dataset, cache_on_cpu, sim.model, breakpoint_module_name,
                                    forward_fn, quant_cache_path, incl_kwargs=incl_kwargs)
        cached_fp_dataset = CachedDataset(None, num_batches, fp32_cache_path)
        cached_quant_dataset = CachedDataset(None, num_batches, quant_cache_path)
    return cached_fp_dataset, cached_quant_dataset


def get_block_outputs(fp_block: torch.nn.ModuleList, quant_block: torch.nn.ModuleList, include_static_inputs: str,
                      cached_fp_dataset: List, cached_quant_dataset: List,
                      cache_on_cpu: bool, forward_fn: Callable, device: torch.device, working_dir: str):
    """
    Get outputs from block/module from FP32 and QuantizationSimModel models and assign for next block/module.

    NOTE: "static_inputs" (like attention_mask, position_ids) remains the same across different blocks.
     So, if "include_static_inputs" is set to True, then such inputs are reused.

    :param fp_block: ModuleList for fp32 modules
    :param quant_block: ModuleList for quant modules
    :param include_static_inputs: Flag to include "static_inputs" or not
    :param cached_fp_dataset: Cached dataset for fp32 model
    :param cached_quant_dataset: Cached dataset for quant model
    :param cache_on_cpu: Whether to cache intermediate data on CPU or store to disk
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    :param device: torch device
    :param working_dir: Working to directory to save block inputs data to disk
    """
    # pylint: disable=too-many-locals, too-many-arguments
    fp_block.to(device)
    quant_block.to(device)

    fp_iterator = iter(cached_fp_dataset)
    quant_iterator = iter(cached_quant_dataset)
    for idx in range(len(cached_fp_dataset)): # pylint: disable=consider-using-enumerate
        fp_inputs = change_tensor_device_placement(next(fp_iterator), device)
        quant_inputs = change_tensor_device_placement(next(quant_iterator), device)

        with in_eval_mode(fp_block), in_eval_mode(quant_block), torch.no_grad():
            fp_outputs = forward_fn(fp_block, fp_inputs)
            fp_outputs = fp_outputs[0].cpu() if isinstance(fp_outputs, (tuple, list)) else fp_outputs.cpu()
            quant_outputs = forward_fn(quant_block, quant_inputs)
            quant_outputs = quant_outputs[0].cpu() if isinstance(quant_outputs, (tuple, list)) else quant_outputs.cpu()

            # Check if the next ModuleList needs static inputs or not and assign
            # the outputs (fp32/quant) from current block to be the input (fp32/quant) of next block
            if include_static_inputs == "True":
                fp_inputs[0], quant_inputs[0] = fp_outputs, quant_outputs
            else:
                fp_inputs, quant_inputs = [fp_outputs], [quant_outputs]

            # Cache the outputs on CPU or disk
            if cache_on_cpu:
                cached_fp_dataset[idx] = fp_inputs
                cached_quant_dataset[idx] = quant_inputs
            else:
                fp32_cache_path = working_dir + 'fp32/'
                quant_cache_path = working_dir + 'quant/'
                save_to_cache(fp_inputs, fp32_cache_path, idx)
                save_to_cache(quant_inputs, quant_cache_path, idx)

    fp_block.cpu()
    quant_block.cpu()


class ActivationSampler:
    """
    For a module in the original model and the corresponding module in the weight quantized QuantSim model,
    collect the module's output and input activation data respectively
    """
    def __init__(self, orig_module: torch.nn.Module, quant_module: _QuantizedModuleProtocol,
                 orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                 forward_fn: Callable[[torch.nn.Module, Any], Any]):
        """
        :param orig_module: Module from original model.
        :param quant_module: Quant wrapper from sim model.
        :param orig_model: Original model.
        :param quant_model: Sim model.
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        """
        self._orig_module = orig_module
        self._quant_module = quant_module
        self._orig_model = orig_model
        self._quant_model = quant_model
        self._orig_module_collector = ModuleData(orig_model, orig_module, forward_fn)
        self._quant_module_collector = ModuleData(quant_model, quant_module, forward_fn)

    def sample_and_place_all_acts_on_cpu(self, cached_dataset: Dataset,
                                         cached_quant_dataset: Dataset = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From the original module, collect output activations and input activations
        to corresponding quantized module.

        NOTE: Keeps collected activation data on CPU memory so this function should only be invoked
        if collected activation data can be fit entirely in CPU memory.

        :param cached_dataset: Cached dataset for fp32 model
        :param cached_quant_dataset: Cached dataset for quant model
        :return: Input data, output data
        """
        all_inp_data = []
        all_out_data = []

        iterator = iter(cached_dataset)
        if cached_quant_dataset:
            assert len(cached_dataset) == len(cached_quant_dataset)
            quant_iterator = iter(cached_quant_dataset)
        for batch_index in range(len(cached_dataset)):
            if cached_quant_dataset:
                inp_data, _ = self.sample_acts(next(quant_iterator), collect_input=True, collect_output=False)
                _, out_data = self.sample_acts(next(iterator), collect_input=False, collect_output=True)
            else:
                inp_data, out_data = self.sample_acts(next(iterator))

            # Keep activation data on CPU memory and then append.
            all_inp_data.append(inp_data.cpu())
            all_out_data.append(out_data.cpu())

            if batch_index == len(cached_dataset) - 1:
                break
        all_inp_data = torch.cat(all_inp_data, dim=0)
        all_out_data = torch.cat(all_out_data, dim=0)

        return all_inp_data, all_out_data

    def sample_acts(self, model_inputs: Union[torch.tensor, List, Tuple], collect_input=True, collect_output=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For given model_inputs, collect input activations data to quant module and
        output activations data from original module.

        :param model_inputs: Model inputs.
        :param collect_input: True if collect input data of the quant module, False otherwise
        :param collect_output: True if collect output data of the fp32 model, False otherwise
        :return: Input and output activations data.
        """
        # Collect input activation data to quantized wrapper module
        # (with all preceding weight modules quantized)
        inp_data, out_data = None, None
        if collect_input:
            inp_data, _ = self._quant_module_collector.collect_inp_out_data(model_inputs,
                                                                            collect_input=True,
                                                                            collect_output=False)
        # Collect output activation data from original module
        if collect_output:
            _, out_data = self._orig_module_collector.collect_inp_out_data(model_inputs,
                                                                           collect_input=False,
                                                                           collect_output=True)
        return inp_data, out_data

def create_cached_block_schedule_list(model: torch.nn.Module, dummy_input, block_names: List[str], supported_modules: Tuple[Type]) \
        -> List[Tuple[Optional[Tuple[torch.nn.Module, str]], List[Tuple[str, torch.nn.Module]]]]:
    """
    Creates a schedule for modules with corresponding block if applicable.

    :param model: Original FP model.
    :param dummy_input: Model inputs.
    :param block_names: List of block names to generate caching point.
    :param supported_modules: module types that can be adaround'ed.
    :return: Schedule list containing tuple of group (if True) and order modules in sequence of forward pass.
    """
    # pylint: disable=too-many-locals
    modules = get_ordered_list_of_modules(model, dummy_input)
    modules = [m for m in modules if isinstance(m[1], supported_modules)]
    caching_modules = {module: {'block': None, 'name': name} for name, module in modules}

    for name in block_names:
        parent_module: torch.nn.Module = get_named_module(model, name)
        for _, module in parent_module.named_modules():
            if module in caching_modules:
                module_name = caching_modules[module]['name']
                caching_modules[module].update(
                    {
                        'block':(name, parent_module),
                        'name': module_name[len(name)+1:]
                    })

    block_list = []
    block = None
    for module, value in caching_modules.items():
        block_module = value['block']
        name_module_pair = [value['name'], module]
        if block is None: # init ?
            block = (block_module, [name_module_pair])
        elif block[0] != block_module: # end of block ?
            block_list.append(block)
            block = (block_module, [name_module_pair])
        else:
            block[1].append(name_module_pair)
    block_list.append(block) # last block

    return block_list
