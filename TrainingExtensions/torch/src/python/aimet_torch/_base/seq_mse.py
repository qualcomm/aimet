# /usr/bin/env python
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

""" Sequential MSE base """

from abc import abstractmethod, ABC
import json
import os
import tempfile
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from aimet_common.utils import AimetLogger

from aimet_torch.utils import CachedDataset, get_ordered_list_of_modules, in_eval_mode, StopForwardException,\
    change_tensor_device_placement, get_device
from aimet_torch._base.adaround.activation_sampler import create_modulelist_for_group_modules,\
    get_block_inputs, get_block_outputs

# The following modules with weights are supported
SUPPORTED_MODULES = (torch.nn.Linear, torch.nn.Conv2d, )

# Skip running Sequential MSE if param BW is higher than supported PARAM_BW.
SUPPORTED_PARAM_BW = 4

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


def default_forward_fn(model, inputs):
    """
    Default forward function.
    :param model: pytorch model
    :param inputs: model inputs
    """
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    return model(*inputs)


@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Available options are 'asym', 'symfp' and 'symqt'. Default 'symqt'.
    :param loss_fn: Loss function. Available options are 'mse', 'l1' and 'sqnr'. Default 'mse'.
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    """
    num_batches: int
    num_candidates: int = 20
    inp_symmetry: str = 'symqt'
    loss_fn: str = 'mse'
    forward_fn: Callable = default_forward_fn


class SequentialMseBase(ABC):
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """
    @classmethod
    def apply_seq_mse(cls,
                      model: torch.nn.Module,
                      sim,
                      data_loader: DataLoader,
                      params: SeqMseParams,
                      modules_to_exclude: Optional[List[torch.nn.Module]] = None,
                      checkpoints_config: Optional[str] = None):
        """
        Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.

            1 Disable all input/output quantizers, param quantizers of non-supported modules
            2 Find and feeze optimal parameter encodings candidate for remaining supported modules
            3 Re-enable disabled quantizers from step 1

        Example userflow:
        model = Model().eval()
        sim = QuantizationSimModel(...)
        apply_seq_mse(...)
        sim.compute_encodings(...) [compute encodings for all activations and parameters of non-supported modules]
        sim.export(...)

        NOTE:
        1) module reference passed to modules_to_exclude should be from FP32 model.
        2) module from modules_to_exclude won't be quantized and skipped when applying sequential MSE.
        3) Except finding param encodings for supported modules, config JSON file will be respected and
        final state of sim will be unchanged.

        :param model: Original fp32 model
        :param sim: Corresponding QuantizationSimModel object
        :param data_loader: Data loader
        :param params: Sequential MSE parameters
        :param modules_to_exclude: List of supported type module(s) to exclude when applying Sequential MSE
        :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
        """
        # disable all input/output activation quantizers and
        # param quantizers of all the non-supported modules and from modules_to_exclude list.
        # then, re-enable disabled quantizers after running sequential mse.
        # this ensures that config JSON file will be respected and final state of sim will be unchanged.
        with cls.temporarily_disable_quantizers(model, sim, modules_to_exclude), \
            tempfile.TemporaryDirectory() as tempdir:

            # Initialize param encodings of modules of supported types.
            cls.compute_all_param_encodings(sim)

            cached_dataset = CachedDataset(data_loader, params.num_batches, os.path.join(tempdir, 'cached_dataset'))
            if checkpoints_config:
                cls.apply_seq_mse_using_opt_sampling(checkpoints_config, model, sim, modules_to_exclude, cached_dataset, params,
                                                     tempdir)
            else:
                dummy_input = change_tensor_device_placement(next(iter(data_loader)), get_device(model))
                fp32_modules = get_ordered_list_of_modules(model, dummy_input, fwd_func=params.forward_fn)
                fp32_modules = [(name, module) for name, module in fp32_modules if isinstance(module, SUPPORTED_MODULES)]
                if modules_to_exclude:
                    fp32_modules = [(name, module) for name, module in fp32_modules if not module in modules_to_exclude]

                # Find and freeze optimal param encodings candidate
                cls.run_seq_mse(fp32_modules, model, sim.model, params, params.forward_fn,
                                cached_dataset, cached_quant_dataset=None)

    @classmethod
    def apply_seq_mse_using_opt_sampling(cls,
                                         checkpoints_config: str,
                                         model: torch.nn.Module,
                                         sim,
                                         modules_to_exclude: Optional[List[torch.nn.Module]],
                                         cached_dataset: CachedDataset,
                                         params: SeqMseParams,
                                         tempdir: str):
        """
        Apply sequential MSE using optimized sampling of intermediate data. When checkpoints_config file is provided,
        intermediate activations from breakpoint are treated as model inputs for next blocks.

        NOTE: Assumption is that the outputs from the current block are fed directly to following block
        and there are no functional operations in-between.

        :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
        :param model: Original fp32 model
        :param sim: Corresponding QuantizationSimModel object
        :param modules_to_exclude: List of supported type module(s) to exclude when applying Sequential MSE
        :param cached_dataset: Cached dataset
        :param params: Sequential MSE parameters
        :param tempdir: temporary working directory
        """
        # pylint: disable=too-many-locals
        ckpts_file = json.load(open(checkpoints_config))
        assert 'grouped_modules' in ckpts_file.keys(), \
            "Please provide a dictionary of grouped_modules in the file to define checkpoints"
        assert 'include_static_inputs' in ckpts_file.keys(), \
            "Please provide a dictionary of include_static_inputs in the file to define checkpoints"
        assert 'cache_on_cpu' in ckpts_file.keys(), \
            "Please define cache_on_cpu to determine whether to cache intermediate tensors on CPU"

        grouped_modules = ckpts_file['grouped_modules']
        breakpoint_module_name = ckpts_file['grouped_modules'][list(grouped_modules.keys())[0]][0]
        include_static_inputs = ckpts_file['include_static_inputs']
        cache_on_cpu = ckpts_file['cache_on_cpu']
        cached_fp_dataset, cached_quant_dataset = get_block_inputs(model, sim,
                                                                   breakpoint_module_name,
                                                                   cached_dataset, cache_on_cpu,
                                                                   params.forward_fn, params.num_batches,
                                                                   tempdir)
        device = get_device(model)
        model.cpu()
        sim.model.cpu()

        # Forward function for the ModuleList object
        def fwd_fn_modulelist(modulelists, x):
            for mod in modulelists:
                x = mod(*x) if isinstance(x, (tuple, list)) else mod(x)
            return x

        sub_fp_models, sub_sim_models = create_modulelist_for_group_modules(model, sim, grouped_modules)
        for i, (fp_block, quant_sim_block, static_input) in enumerate(zip(sub_fp_models,
                                                                          sub_sim_models,
                                                                          include_static_inputs)):
            fp32_modules = get_ordered_list_of_modules(fp_block, cached_fp_dataset[0], fwd_func=fwd_fn_modulelist)
            fp32_modules = [(name, module) for name, module in fp32_modules if isinstance(module, SUPPORTED_MODULES)]
            if modules_to_exclude:
                fp32_modules = [(name, module) for name, module in fp32_modules if not module in modules_to_exclude]

            cls.run_seq_mse(fp32_modules, fp_block, quant_sim_block, params, fwd_fn_modulelist,
                            cached_fp_dataset, cached_quant_dataset=cached_quant_dataset)

            # Get the outputs from the current block and assign to be the inputs for next block
            # except for the last block
            if i < len(sub_fp_models) - 1:
                get_block_outputs(fp_block, quant_sim_block, static_input,
                                  cached_fp_dataset, cached_quant_dataset, cache_on_cpu,
                                  fwd_fn_modulelist, device, tempdir)
        model.to(device)
        sim.model.to(device)

    @classmethod
    def run_seq_mse(cls,
                    fp32_modules: List[Tuple[str, torch.nn.Module]],
                    model: torch.nn.Module,
                    quant_model: torch.nn.Module,
                    params: SeqMseParams,
                    forward_fn: Callable,
                    cached_fp_dataset: CachedDataset,
                    cached_quant_dataset: Optional[CachedDataset] = None,
                    ):
        """
        Run Sequential MSE

        :param fp32_modules: List of FP32 candidate modules in order of occurence
        :param model: FP32 model
        :param quant_model: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
        yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
        :param cached_fp_dataset: Cached dataset object
        :param cached_quant_dataset: Cached dataset object
        """
        name_to_quant_module = {}
        for name, quant_module in quant_model.named_modules():
            name_to_quant_module[name] = quant_module

        if not cached_quant_dataset:
            cached_quant_dataset = cached_fp_dataset

        for module_qualified_name, fp32_module in fp32_modules:
            try:
                quant_module = name_to_quant_module[module_qualified_name]
            except KeyError:
                continue

            if quant_module.param_quantizers['weight'].bitwidth > SUPPORTED_PARAM_BW:
                continue

            _logger.info("Finding and freezing optimal param encodings candidate of module: %s", module_qualified_name)
            if params.inp_symmetry == "asym":
                fp32_inp_acts = cls.get_module_inp_acts(fp32_module, model, params, forward_fn, cached_fp_dataset)
                quant_inp_acts = cls.get_module_inp_acts(quant_module, quant_model, params, forward_fn, cached_quant_dataset)
                cls.optimize_module(quant_module, fp32_inp_acts, quant_inp_acts, params)
            elif params.inp_symmetry == "symfp":
                fp32_inp_acts = cls.get_module_inp_acts(fp32_module, model, params, forward_fn, cached_fp_dataset)
                cls.optimize_module(quant_module, fp32_inp_acts, fp32_inp_acts, params)
            elif params.inp_symmetry == "symqt":
                quant_inp_acts = cls.get_module_inp_acts(quant_module, quant_model, params, forward_fn, cached_quant_dataset)
                cls.optimize_module(quant_module, quant_inp_acts, quant_inp_acts, params)
            else:
                raise ValueError(f"Invalid inp_symmetry: {params.inp_symmetry}")

    @staticmethod
    def get_module_inp_acts(module: torch.nn.Module,
                            model: torch.nn.Module,
                            params: SeqMseParams,
                            forward_fn: Callable,
                            cached_dataset: CachedDataset,
                            ) -> torch.Tensor:
        """
        For given module, get inputs to the module.

        :param module: FP32/quant module
        :param model: FP32/quant model
        :param params: Sequential MSE parameters
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
        yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
        :param cached_dataset: Cached dataset
        :return: Concatenated inputs
        """
        inp_acts = []
        def hook_fn(_, inp, __):
            if isinstance(inp, tuple):
                inp_acts.append(inp[0])
            raise StopForwardException
        handle = module.register_forward_hook(hook_fn)

        iterator = iter(cached_dataset)
        for _ in range(params.num_batches):
            batch = change_tensor_device_placement(next(iterator), get_device(model))
            try:
                with in_eval_mode(model), torch.no_grad():
                    forward_fn(model, batch)
            except StopForwardException:
                pass
        handle.remove()

        inp_acts = torch.stack(inp_acts)
        return inp_acts

    @staticmethod
    def _get_quantizers_to_be_disabled(
            model: torch.nn.Module,
            sim,
            modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, get all quantizers to be disabled before applying sequential MSE.
        """
        # pylint: disable=protected-access
        name_to_fp32_module_dict = {}
        for name, fp32_module in model.named_modules():
            name_to_fp32_module_dict[name] = fp32_module

        quantizers_to_be_disabled = []
        for name, quant_wrapper in sim.quant_wrappers():
            for quantizer in quant_wrapper.input_quantizers:
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)

            for quantizer in quant_wrapper.output_quantizers:
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)

            for quantizer in quant_wrapper.param_quantizers.values():
                if not isinstance(quant_wrapper._module_to_wrap, SUPPORTED_MODULES) and quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)

            # disable param quantizers from exclusion list
            if modules_to_exclude:
                with contextlib.suppress(KeyError):
                    fp32_module = name_to_fp32_module_dict[name]
                    if fp32_module in modules_to_exclude:
                        for quantizer in quant_wrapper.param_quantizers.values():
                            if quantizer.enabled:
                                quantizers_to_be_disabled.append(quantizer)

        return quantizers_to_be_disabled

    @staticmethod
    def get_candidates(num_candidates: int,
                       per_channel_max: torch.Tensor,
                       per_channel_min: Optional[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform grid search.

        :param num_candidates: Number of candidates
        :param per_channel_max: Per channel max values
        :param per_channel_min: Per channel min values
        :return: candidates
        """
        candidates = []
        if per_channel_min is not None:
            for cand in range(num_candidates):
                cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
                cand_min = torch.tensor(per_channel_min / num_candidates * (cand + 1))
                candidates.append((cand_max, cand_min))
        else:
            for cand in range(num_candidates):
                cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
                cand_min = -cand_max
                candidates.append((cand_max, cand_min))
        return candidates

    @staticmethod
    def compute_recon_loss(xqwq: torch.Tensor, xw: torch.Tensor, params: SeqMseParams):
        """
        Compute reconstruction loss and return the sum by reducing over all the dimensions except last channel dimension.

        :param xqwq: X^Q^ quantized-dequantized values
        :param xw: XW FP32 values
        :param params: Sequential MSE parameters
        :return: loss
        """
        if params.loss_fn == "mse":
            loss_fn = functional.mse_loss
        elif params.loss_fn == "l1":
            loss_fn = functional.l1_loss
        elif params.loss_fn == "sqnr":
            loss_fn = neg_sqnr
        else:
            raise ValueError(f"Invalid loss function: {params.loss_fn}")

        channel_dim = xqwq.shape[-1]
        xqwq = xqwq.reshape(-1, channel_dim)
        xw = xw.reshape(-1, channel_dim)
        loss = loss_fn(xqwq, xw, reduction="none").sum(0)
        assert loss.size() == torch.Size([channel_dim])
        return loss

    @classmethod
    def get_per_channel_min_and_max(cls, quant_module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per channel min/max values across output channel.

        :param quant_module: Quant module to be optimized
        :return:
        """
        # pylint: disable=protected-access
        module = cls._get_original_module(quant_module)

        if isinstance(module, torch.nn.Conv2d):
            channel_dim = module.out_channels
            weight = module.weight.reshape(channel_dim, -1)
        elif isinstance(module, torch.nn.Linear):
            weight = module.weight
        else:
            raise ValueError('Unsupported module: ', module)

        if cls._is_symmetric_quantizer(quant_module.param_quantizers["weight"]):
            per_channel_max = torch.max(weight.abs(), dim=1)[0].detach()
            per_channel_min = None
        else:
            per_channel_max = torch.max(weight, dim=1)[0].detach()
            per_channel_min = torch.min(weight, dim=1)[0].detach()

        return per_channel_min, per_channel_max

    @classmethod
    def compute_outputs(cls,
                        quant_module,
                        x: torch.Tensor,
                        xq: torch.Tensor,
                        w: torch.Tensor,
                        wq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute X^W^ and XW output activations.

        :param quant_module: Wrapper module to be optimized
        :param x: Inputs from FP32 model
        :param xq: Inputs from QuantSim model
        :param w: FP32 weights
        :param wq: Quantized-dequantized weights
        :return: xqwq, xw
        """
        # pylint: disable=protected-access
        module = cls._get_original_module(quant_module)

        if isinstance(module, torch.nn.Linear):
            xqwq = functional.linear(xq, wq)
            xw = functional.linear(x, w)
        elif isinstance(module, torch.nn.Conv2d):
            xqwq = functional.conv2d(xq, wq, stride=module.stride, dilation=module.dilation,
                                     padding=module.padding, groups=module.groups)
            xw = functional.conv2d(x, w, stride=module.stride, dilation=module.dilation,
                                   padding=module.padding, groups=module.groups)

            # [N, C, H, W] --> [N, H, W, C], so that loss can be computed across channel dimension.
            xqwq = xqwq.permute(0, 2, 3, 1)
            xw = xw.permute(0, 2, 3, 1)
        else:
            raise ValueError('Unsupported module: ', module)
        return xqwq, xw

    @classmethod
    @abstractmethod
    def temporarily_disable_quantizers(
            cls,
            model: torch.nn.Module,
            sim,
            modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, disable quantizers needed to be disabled before applying sequential MSE.

        :param model: Original fp32 model
        :param sim: QuantizationSimModel object
        :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
        :return: List of quantizers to be disabled.
        """

    @classmethod
    @abstractmethod
    def compute_all_param_encodings(cls, sim):
        """
        Compute encodings for all parameters, needed for initializing Sequential MSE

        :param sim: Quant sim
        """

    @classmethod
    @abstractmethod
    def optimize_module(cls,
                        quant_module,
                        x: torch.Tensor,
                        xq: torch.Tensor,
                        params: SeqMseParams):
        """
        Find and freeze optimal parameter encodings candidate for given module.

        :param quant_module: Quant module to be optimized
        :param x: Inputs to module from FP32 model
        :param xq: Inputs to module from QuantSim model
        :param params: Sequenial MSE parameters
        """

    @classmethod
    @abstractmethod
    def compute_param_encodings(cls,
                                quantizer,
                                x_min: torch.Tensor,
                                x_max: torch.Tensor):
        """
        Compute encodings for parameter quantizer using given x_min and x_max values.

        :param quantizer: Tensor quantizer
        :param x_min: min values
        :param x_max: max values
        """

    @classmethod
    @abstractmethod
    def _is_symmetric_quantizer(cls, quantizer):
        ...

    @classmethod
    @abstractmethod
    def _freeze_quantizer_encoding(cls, quantizer):
        ...

    @classmethod
    @abstractmethod
    def _get_quantized_weight(cls, quant_module):
        ...

    @classmethod
    @abstractmethod
    def _get_original_module(cls, quant_module):
        ...


def neg_sqnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-10, reduction="none"):
    """
    Loss function to minimize negative SQNR which is equivalent to maximizing SQNR.

    :param pred: X^Q^ quantized-dequantized values
    :param target: XW FP32 values
    :param eps: epsilon
    :param reduction: unused arg
    :return: Negative SQNR
    """
    # pylint: disable=unused-argument
    quant_error = target - pred
    exp_noise = torch.mean(quant_error ** 2, 0, keepdim=True) + eps
    exp_signal = torch.mean(target ** 2, 0, keepdim=True)
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * torch.log10(sqnr)
    return -sqnr_db
