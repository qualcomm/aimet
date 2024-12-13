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
""" Defines Adaround base classes shared across aimet_torch.v1 and v2 """

from abc import ABC, abstractmethod
import os
import contextlib
import json
import tempfile
from typing import Tuple, Union, Dict, List, Callable, Optional, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import AIMET specific modules
from aimet_common.utils import AimetLogger, convert_configs_values_to_bool
from aimet_common.defs import QuantScheme

from aimet_torch import utils
from aimet_torch.meta import connectedgraph_utils
from aimet_torch._base.quantsim import _QuantizationSimModelInterface, _QuantizedModuleProtocol
from aimet_torch._base.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_torch._base.adaround.adaround_loss import AdaroundHyperParameters
from aimet_torch._base.adaround.activation_sampler import create_modulelist_for_group_modules, get_block_inputs, \
    get_block_outputs, create_cached_block_schedule_list
from aimet_torch.utils import get_named_module

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# The following modules with weights are supported by Adaround
AdaroundSupportedModules = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)


class AdaroundParameters:
    """
    Configuration parameters for Adaround
    """
    def __init__(self, data_loader: DataLoader, num_batches: int,
                 default_num_iterations: int = None, default_reg_param: float = 0.01,
                 default_beta_range: Tuple = (20, 2), default_warm_start: float = 0.2,
                 forward_fn: Callable[[torch.nn.Module, Any], Any] = None):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to be used for Adaround.
         A commonly recommended value for this parameter is the smaller value among (1) len(data_loader) and (2) ceil(2000/batch_size)
        :param default_num_iterations: Number of iterations to adaround each layer.
         The default value is 10K for models with 8- or higher bit weights, and 15K for models with lower than 8 bit weights.
        :param default_reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss.
         Default 0.01
        :param default_beta_range: Start and stop beta parameter for annealing of rounding loss (start_beta, end_beta).
         Default (20, 2)
        :param default_warm_start: warm up period, during which rounding loss has zero effect. Default 20% (0.2)
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
         yielded from the data loader. The function expects model as first argument and inputs to model
         as second argument.
        """
        if len(data_loader) < num_batches:
            raise ValueError(f'Can not fetch {num_batches} batches from '
                             f'a data loader of length {len(data_loader)}.')

        self.data_loader = data_loader
        self.num_batches = num_batches
        self.num_iterations = default_num_iterations
        self.reg_param = default_reg_param
        self.beta_range = default_beta_range
        self.warm_start = default_warm_start
        self.forward_fn = forward_fn


class AdaroundBase(ABC):
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_adaround(cls, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple], params: AdaroundParameters,
                       path: str, filename_prefix: str, default_param_bw: int = 4,
                       param_bw_override_list: List[Tuple[torch.nn.Module, int]] = None,
                       ignore_quant_ops_list: List[torch.nn.Module] = None,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       default_config_file: str = None) -> torch.nn.Module:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param model: Model to Adaround
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param param_bw_override_list: List of Tuples. Each Tuple is a module and the corresponding parameter bitwidth
                                       to be used for that module.
        :param ignore_quant_ops_list: Ops listed here are skipped during quantization needed for AdaRounding. Do not
                                      specify Conv and Linear modules in this list. Doing so, will affect accuracy.
        :param default_quant_scheme: Quantization scheme. Supported options are using Quant Scheme Enum
                                    QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param default_config_file: Default configuration file for model quantizers
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """
        # pylint: disable=too-many-arguments
        # Create Quant sim with given parameters
        quant_sim = cls._get_quantsim(model, dummy_input=dummy_input, quant_scheme=default_quant_scheme,
                                      default_param_bw=default_param_bw,
                                      config_file=default_config_file)

        # For the modules in the param_bw_override_list, override the default parameter bitwidths in the QuantSim
        if param_bw_override_list:
            cls._override_param_bitwidth(model, quant_sim, param_bw_override_list)

        if ignore_quant_ops_list:
            cls._exclude_modules(model, quant_sim, ignore_quant_ops_list)

        # Compute only param encodings
        cls._compute_param_encodings(quant_sim)

        return cls._apply_adaround(quant_sim, model, dummy_input, params, path, filename_prefix)

    @classmethod
    def _apply_adaround(cls, quant_sim: _QuantizationSimModelInterface, model: torch.nn.Module,
                        dummy_input: Union[torch.Tensor, Tuple], params: AdaroundParameters,
                        path: str, filename_prefix: str, checkpoints_config: str = None) -> torch.nn.Module:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param model: Original fp32 model from which quant_sim was created.
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param checkpoints_config: Config files to split fp32/quant model by checkpoints
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """

        # Sanity check: All the input/output quantizers should be disabled
        cls._check_input_output_quantizers_for_adaround(quant_sim.model)

        # Get the module - activation function pair using ConnectedGraph
        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, dummy_input)

        cls._adaround_model(model, quant_sim, module_act_func_pair, params, dummy_input, checkpoints_config)

        # Export quantization encodings to JSON-formatted file
        cls._export_encodings_to_json(path, filename_prefix, quant_sim)

        cls._remove_quantization_wrappers(quant_sim.model)
        logger.info('Completed Adarounding Model')
        return quant_sim.model

    @classmethod
    def _adaround_model(cls, model: torch.nn.Module, quant_sim: _QuantizationSimModelInterface, module_act_func_pair: Dict,
                        params: AdaroundParameters, dummy_input: Union[torch.Tensor, Tuple],
                        checkpoints_config: str = None):
        """
        Optimize weight rounding of every module (AdaroundSupportedModules) of model in sequential manner
        based on occurrence

        NOTE: When checkpoints_config file is provided, assumption is that the outputs from previous group modules (block)
         should feed directly into next group modules (block)

        :param model: Original fp32 model from which quant_sim was created.
        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param module_act_func_pair: Dictionary of module to immediate following activation function
        :param params: Adaround parameters
        :param dummy_input: Dummy input to the model
        :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
        """
        # pylint: disable=too-many-locals, protected-access, too-many-branches, too-many-statements

        num_iterations = params.num_iterations

        if num_iterations is None:
            lowest_weight_bw = cls._get_lowest_weight_bw(quant_sim.model)
            # If the lowest wegith bitwidth is < 8, then set num_iterations to 15K by default
            if lowest_weight_bw < 8:
                num_iterations = 15000
            else:
                num_iterations = 10000

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cache model input data to temporary directory
            cached_dataset = utils.CachedDataset(params.data_loader, params.num_batches, tmp_dir)

            # Optimization Hyper parameters
            opt_params = AdaroundHyperParameters(num_iterations, params.reg_param, params.beta_range,
                                                 params.warm_start)

            # AdaRound must be applied to modules in the order of occurrence
            if checkpoints_config:
                # Load the predefined json file for checkpoints info
                checkpoint_config = json.load(open(checkpoints_config))
                convert_configs_values_to_bool(checkpoint_config)

                assert 'cache_on_cpu' in checkpoint_config.keys(), \
                    "Please define cache_on_cpu to determine whether to cache intermediate tensors on CPU"
                cache_on_cpu = checkpoint_config['cache_on_cpu']

                checkpoint_type = checkpoint_config.get('checkpoint_type', 'sequential')
                if checkpoint_type == 'sequential':
                    assert 'grouped_modules' in checkpoint_config.keys(), \
                        "Please provide a dictionary of grouped_modules in the file to define checkpoints"
                    assert 'include_static_inputs' in checkpoint_config.keys(), \
                        "Please provide a dictionary of include_static_inputs in the file to define checkpoints"

                    grouped_modules = checkpoint_config['grouped_modules']
                    breakpoint_module_name = checkpoint_config['grouped_modules'][list(grouped_modules.keys())[0]][0]
                    include_static_inputs = checkpoint_config['include_static_inputs']
                    cached_fp_dataset, cached_quant_dataset = get_block_inputs(model, quant_sim,
                                                                               breakpoint_module_name,
                                                                               cached_dataset, cache_on_cpu,
                                                                               params.forward_fn, params.num_batches,
                                                                               tmp_dir)
                    # Get the device of model to latter be used to place input tensor on the same device
                    device = utils.get_device(model)
                    model.cpu()
                    quant_sim.model.cpu()

                    # Forward function for the ModuleList object
                    def fwd_mod_ls(mod_ls, x):
                        for mod in mod_ls:
                            x = params.forward_fn(mod, x)
                        return x

                    sub_fp_models, sub_sim_models = create_modulelist_for_group_modules(model, quant_sim, grouped_modules)
                    for i, (fp_block, quant_sim_block, static_input) in enumerate(zip(sub_fp_models,
                                                                                      sub_sim_models,
                                                                                      include_static_inputs)):
                        modules = utils.get_ordered_list_of_modules(fp_block, cached_fp_dataset[0], fwd_mod_ls)
                        cls._run_adaround_model(modules, fp_block, quant_sim_block,
                                                module_act_func_pair, opt_params,
                                                fwd_mod_ls,
                                                cached_fp_dataset, cached_quant_dataset)

                        # Get the outputs from the current block and assign to be the inputs for next block
                        # except for the last block
                        if i < len(sub_fp_models) - 1:
                            get_block_outputs(fp_block, quant_sim_block, static_input,
                                              cached_fp_dataset, cached_quant_dataset, cache_on_cpu,
                                              fwd_mod_ls, device, tmp_dir)

                    # After finishing Adaround, placing the quant model back to its original device
                    quant_sim.model.to(device)
                else:
                    assert 'cached_blocks' in checkpoint_config.keys(), \
                        "Please provide a list of modules that  can be cached"

                    block_list = create_cached_block_schedule_list(
                        model, dummy_input, checkpoint_config['cached_blocks'], AdaroundSupportedModules)

                    for block_cfg, modules in tqdm(block_list, desc='block'):
                        if block_cfg is None: # doesn't belong to a cached block
                            cls._run_adaround_model(modules, model, quant_sim.model, module_act_func_pair, opt_params,
                                                    params.forward_fn, cached_dataset)
                        else:
                            block_name, fp_block = block_cfg
                            quant_sim_block: torch.nn.Module = get_named_module(quant_sim.model, block_name)

                            cached_fp_dataset, cached_quant_dataset = get_block_inputs(model, quant_sim,
                                                                                       block_name,
                                                                                       cached_dataset, cache_on_cpu,
                                                                                       params.forward_fn,
                                                                                       params.num_batches,
                                                                                       tmp_dir,
                                                                                       incl_kwargs=True)

                            def block_fwd(_model, x):
                                return _model(*x)

                            cls._run_adaround_model(modules, fp_block, quant_sim_block, module_act_func_pair,
                                                    opt_params,
                                                    block_fwd, cached_fp_dataset, cached_quant_dataset)
                            del cached_fp_dataset
                            del cached_quant_dataset
            else:
                modules = utils.get_ordered_list_of_modules(model, dummy_input)
                cls._run_adaround_model(modules, model, quant_sim.model, module_act_func_pair, opt_params,
                                        params.forward_fn, cached_dataset)

    @classmethod
    def _run_adaround_model(cls, modules: List, model: torch.nn.Module, quant_sim_model: torch.nn.Module,
                            module_act_func_pair: Dict, opt_params: AdaroundHyperParameters, forward_fn: Callable,
                            cached_dataset: utils.CachedDataset,
                            cached_quant_dataset: Optional[utils.CachedDataset] = None):
        """
        Iterate through all modules to find out Adaround supported modules and
         apply Adaround optimization to those modules

        :param modules: Candidate modules
        :param model: Original fp32 model
        :param quant_sim_model: QuantSim model
        :param module_act_func_pair: Activation function pairs
        :param opt_params: Optimization parameters
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader
        :param cached_dataset: Cached dataset for the fp32 model
        :param cached_quant_dataset: Cached dataset for the quant model
        """
        # pylint: disable=too-many-arguments, too-many-locals, protected-access
        for name, module in tqdm(modules):
            if isinstance(module, AdaroundSupportedModules):
                # Using name, get corresponding quantized wrapper module from Quant sim model
                quant_wrapper = cls._get_quant_wrapper(quant_sim_model, name)
                if not quant_wrapper:
                    continue

                # Wraps the quant module with adaround wrapper
                # and temporarily replace quant module with wrapped module
                with cls._replace_quantization_layer(quant_sim_model, name) as adaround_wrapper:

                    # Get module's next following activation function
                    act_func = module_act_func_pair[module]

                    logger.info("Started Optimizing weight rounding of module: %s", name)
                    AdaroundOptimizer.adaround_module(module, adaround_wrapper, model, quant_sim_model, act_func,
                                                      cached_dataset, forward_fn, opt_params, cached_quant_dataset)
                    weight = adaround_wrapper.weight

                    # Fold trained alpha to weight
                    with torch.no_grad():
                        # Use soft rounding to compute Adarounded weight
                        adaround_wrapper.use_soft_rounding = True
                        adarounded_weight = adaround_wrapper.apply_adaround(weight)
                        weight.copy_(adarounded_weight)
                        del adarounded_weight

    @staticmethod
    @abstractmethod
    def _compute_param_encodings(quant_sim: _QuantizationSimModelInterface):
        """
        Compute encodings for parameters, needed for initializing Adaround quantizers
        :param quant_sim: Quant sim
        """

    @staticmethod
    @abstractmethod
    def _get_quantsim(model: torch.nn.Module, dummy_input: torch.Tensor,
                      quant_scheme: QuantScheme, default_param_bw: int, config_file: str):
        ...

    @staticmethod
    @abstractmethod
    def _get_adaround_wrapper(quant_module: _QuantizedModuleProtocol):
        ...

    @staticmethod
    @abstractmethod
    def _remove_quantization_wrappers(module: torch.nn.Module):
        ...

    @staticmethod
    @contextlib.contextmanager
    def _patch_module_layer(model, layer_name, new_layer):
        """
        Temporarily replace model layer
        """
        original_layer = getattr(model, layer_name)
        setattr(model, layer_name, new_layer)
        yield
        setattr(model, layer_name, original_layer)

    @staticmethod
    @abstractmethod
    def _validate_quant_module_for_adaround(quant_module: _QuantizedModuleProtocol):
        ...

    @staticmethod
    @abstractmethod
    def _check_input_output_quantizers_for_adaround(quant_model: torch.nn.Module):
        ...

    @staticmethod
    @abstractmethod
    def _get_lowest_weight_bw(quant_model: torch.nn.Module):
        ...

    @classmethod
    @contextlib.contextmanager
    def _replace_quantization_layer(cls, quant_sim_model: torch.nn.Module, module_name: str):
        """
        Replace the quantized module's weight tensor quantizer with the Adaround tensor quantizer
        :param quant_module: quant module
        """
        quant_module = utils.get_named_module(quant_sim_model, module_name)
        cls._validate_quant_module_for_adaround(quant_module)
        adaround_layer = cls._get_adaround_wrapper(quant_module)

        # We need to look for the container to patch for modules inside submodule
        upper_module = quant_sim_model
        upper_module_name, _, target_module_name = module_name.rpartition('.')
        if upper_module_name:
            upper_module = utils.get_named_module(quant_sim_model, upper_module_name)

        # Temporarily replace quant module with wrapped module
        with cls._patch_module_layer(upper_module, target_module_name, adaround_layer):
            yield adaround_layer

    @staticmethod
    @abstractmethod
    def _get_quant_wrapper(quant_sim_model: torch.nn.Module, module_name: str) -> Union[_QuantizedModuleProtocol, None]:
        """
        For given module name, get the quantized wrapper module from the QuantSim model
        :param quant_sim_model: Model with simulation ops
        :param module_name: Module name
        :return: Quantized wrapper module or None
        """
        ...

    @classmethod
    def _export_encodings_to_json(cls, path: str, filename_prefix: str, quant_sim: _QuantizationSimModelInterface):
        """
        Save Adadrounded module's parameter encodings to JSON file
        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param quant_sim: QunatSim that contains the model and Adaround tensor quantizers
        """
        # pylint: disable=protected-access
        # Create a dictionary to export to JSON file
        param_encodings = {}

        for name, quant_module in quant_sim.model.named_modules():
            if isinstance(quant_module, _QuantizedModuleProtocol) and \
                    isinstance(quant_module.get_original_module(), AdaroundSupportedModules):

                if 'weight' in quant_module.param_quantizers:
                    cls._update_param_encodings_dict(quant_module, name, param_encodings)

        # Unify the encoding format to be same as that of full encoding export file
        encoding = {'param_encodings': param_encodings}
        # export encodings to JSON file
        os.makedirs(os.path.abspath(path), exist_ok=True)
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(encoding, encoding_fp, sort_keys=True, indent=4)

    @classmethod
    def _update_param_encodings_dict(cls, quant_module: _QuantizedModuleProtocol, name: str, param_encodings: Dict):
        """
        Add module's weight parameter encodings to dictionary to be used for exporting encodings
        :param quant_module: quant module
        :param name: name of module
        :param param_encodings: Dictionary of param encodings
        """
        for orig_param_name, encodings in quant_module.export_param_encodings(encoding_version='0.6.1').items():
            if orig_param_name == 'weight' and encodings:
                param_name = name + '.' + orig_param_name
                param_encodings[param_name] = encodings

    @staticmethod
    def _override_param_bitwidth(model: torch.nn.Module, quant_sim: _QuantizationSimModelInterface,
                                 param_bw_override_list: List[Tuple[torch.nn.Module, int]]):
        """
        For the QuantSim, for the list of modules in the param_bw_override_list,
        overrides the default parameter bitwidths with the provided bitwidth.

        :param model: The original model
        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param param_bw_override_list: List of Tuples. Each Tuple is a module and the corresponding parameter bitwidth
                                       to be used for that module.
        """
        # Create a mapping of original model's AdaRoundable module and their name
        module_to_name = {}
        for name, module in model.named_modules():
            if isinstance(module, AdaroundSupportedModules):
                module_to_name[module] = name

        # Create a mapping of QuantSim model's AdaRoundable module name and their module
        name_to_module = {}
        for q_name, q_module in quant_sim.model.named_modules():
            if isinstance(q_module, _QuantizedModuleProtocol):
                if isinstance(q_module.get_original_module(), AdaroundSupportedModules):  # pylint: disable=protected-access
                    name_to_module[q_name] = q_module

        # For the modules specified in the param_bw_override_list, set the weight quantizer bitwidth
        for (module, bw) in param_bw_override_list:
            module_name = module_to_name[module]
            quant_wrapper = name_to_module[module_name]
            quant_wrapper.param_quantizers['weight'].bitwidth = bw

    @classmethod
    def _exclude_modules(cls, model: torch.nn.Module, quant_sim: _QuantizationSimModelInterface,
                         ignore_quant_ops_list: List[torch.nn.Module]):
        """
        For the modules mentioned in the ignore_quant_ops_list, remove the corresponding quant wrappers from the
        quantSim and excludes modules from adaround optimization.

        :param model: The original model
        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param ignore_quant_ops_list: The list of modules for which the Quantization wrappers are removed from the
                                      QuantSim object.
        """
        quant_wrappers_to_exclude = []
        for module in ignore_quant_ops_list:
            for m in module.modules():
                name = utils.get_layer_name(model, m)
                quant_wrapper = cls._get_quant_wrapper(quant_sim.model, name)
                if quant_wrapper:
                    quant_wrappers_to_exclude.append(quant_wrapper)

        quant_sim.exclude_layers_from_quantization(quant_wrappers_to_exclude)

    @classmethod
    def apply_adaround_with_cache(cls, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                                  params: AdaroundParameters,
                                  path: str, filename_prefix: str, default_param_bw: int = 4,
                                  param_bw_override_list: List[Tuple[torch.nn.Module, int]] = None,
                                  ignore_quant_ops_list: List[torch.nn.Module] = None,
                                  default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                                  default_config_file: str = None,
                                  checkpoints_config: str = None) -> torch.nn.Module:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT
        :param model: Model to Adaround
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param param_bw_override_list: List of Tuples. Each Tuple is a module and the corresponding parameter bitwidth
                                       to be used for that module.
        :param ignore_quant_ops_list: Ops listed here are skipped during quantization needed for AdaRounding. Do not
                                      specify Conv and Linear modules in this list. Doing so, will affect accuracy.
        :param default_quant_scheme: Quantization scheme. Supported options are using Quant Scheme Enum
                                    QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param default_config_file: Default configuration file for model quantizers
        :param checkpoints_file: JSON file to define checkpoints for caching intermediate tensors of fp32/quant model
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """
        # pylint: disable=too-many-arguments
        assert checkpoints_config is not None, "To run Adaround with cached tensors, please provide a JSON file with checkpoints defined"
        # Create Quant sim with given parameters
        quant_sim = cls._get_quantsim(model, dummy_input=dummy_input, quant_scheme=default_quant_scheme,
                                      default_param_bw=default_param_bw,
                                      config_file=default_config_file)

        # For the modules in the param_bw_override_list, override the default parameter bitwidths in the QuantSim
        if param_bw_override_list:
            cls._override_param_bitwidth(model, quant_sim, param_bw_override_list)

        if ignore_quant_ops_list:
            cls._exclude_modules(model, quant_sim, ignore_quant_ops_list)

        # Compute only param encodings
        cls._compute_param_encodings(quant_sim)

        return cls._apply_adaround(quant_sim, model, dummy_input, params, path, filename_prefix, checkpoints_config)
