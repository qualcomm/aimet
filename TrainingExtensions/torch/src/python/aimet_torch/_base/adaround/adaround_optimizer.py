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

""" Adaround optimizer """

from typing import Union, Tuple, Callable, Any
from functools import reduce
import psutil
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, Subset
import torch.distributed as dist

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch._base.quantsim import _QuantizedModuleProtocol
from aimet_torch._base.adaround.activation_sampler import ActivationSampler
from aimet_torch._base.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters
from aimet_torch._base.adaround.adaround_wrapper import AdaroundWrapperBase

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
BATCH_SIZE = 32
EMPIRICAL_THRESHOLD = 3 / 4


class AdaroundOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """
    is_activation_caching_enabled = True

    @classmethod
    def adaround_module(cls, module: torch.nn.Module, quant_module: _QuantizedModuleProtocol,
                        orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                        act_func: Union[torch.nn.Module, None], cached_dataset: Dataset,
                        forward_fn: Callable[[torch.nn.Module, Any], Any],
                        opt_params: AdaroundHyperParameters, cached_quant_dataset: Dataset = None):
        """
        Adaround module
        :param module: Original module
        :param quant_module: Quantized wrapper module
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader
        :param cached_quant_dataset: Cached dataset for quant model
        :param opt_params: Optimization parameters
        """
        # pylint: disable=too-many-locals, too-many-arguments
        assert isinstance(quant_module, AdaroundWrapperBase), '%s is not adaround wrapper module.' % quant_module

        # Get input and output data of batch size to compute reconstruction error of output activations
        # before and after optimization
        act_sampler = ActivationSampler(module, quant_module, orig_model, quant_model, forward_fn)
        if cached_quant_dataset:
            inp_data, _ = act_sampler.sample_acts(cached_quant_dataset[0], collect_input=True, collect_output=False)
            _, out_data = act_sampler.sample_acts(cached_dataset[0], collect_input=False, collect_output=True)
        else:
            inp_data, out_data = act_sampler.sample_acts(cached_dataset[0])

        recons_err_hard, recons_err_soft = cls._compute_recons_metrics(quant_module, act_func, inp_data, out_data)
        logger.debug("Before opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        # Optimize weight rounding
        cls._optimize_rounding(module, quant_module, orig_model, quant_model, act_func, cached_dataset, forward_fn,
                               opt_params, cached_quant_dataset)

        recons_err_hard, recons_err_soft = cls._compute_recons_metrics(quant_module, act_func, inp_data, out_data)
        logger.debug("After opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        # After optimization, set the optimized layer's rounding mode to "Hard rounding"
        quant_module.use_soft_rounding = False

    @classmethod
    def _optimize_rounding(cls, module: torch.nn.Module, quant_module: AdaroundWrapperBase,
                           orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                           act_func: Union[torch.nn.Module, None], cached_dataset: Dataset,
                           forward_fn: Callable[[torch.nn.Module, Any], Any],
                           opt_params: AdaroundHyperParameters, cached_quant_dataset: Dataset = None):
        """
        Optimizes the weight rounding of quantized wrapper module.

        NOTE:
        1) Tries to cache intermediate activation data on CPU RAM. If succeeds, tries to place cached intermediate
         activation data on GPU else keep it on CPU RAM only and incur CPU-GPU memory transfer.
        2) If 1) fails, reads model inputs dirctly from the disk.

        :param module: Original module
        :param quant_module: Adaround wrapper module
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader
        :param opt_params: Optimization parameters
        """
        # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, too-many-statements
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Shard dataset
        indices = tuple(range(rank, len(cached_dataset), world_size))
        cached_dataset = Subset(cached_dataset, indices=indices)
        if cached_quant_dataset is not None:
            cached_quant_dataset = Subset(cached_quant_dataset, indices=indices)

        assert isinstance(quant_module, AdaroundWrapperBase), '%s is not adaround wrapper module.' % quant_module
        assert quant_module.use_soft_rounding, 'optimization should use soft rounding only.'
        assert quant_module.alpha is not None, 'alpha parameter should be initialized.'

        # Create and set up Adam optimizer with parameter 'alpha' to be optimized
        optimizer = torch.optim.Adam([quant_module.alpha])

        for group in optimizer.param_groups:
            group['lr'] *= world_size # Scale up learning rate by world_size

        # Check if we can cache intermediate activation data.
        model_inputs = cached_dataset[0]
        act_sampler = ActivationSampler(module, quant_module, orig_model, quant_model, forward_fn)
        inp_data, out_data = act_sampler.sample_acts(model_inputs)
        use_cache_acts_data = cls._can_cache_acts_data(len(cached_dataset), inp_data.shape, out_data.shape,
                                                       inp_data.dtype)
        del inp_data, out_data

        device = utils.get_device(module)
        if use_cache_acts_data and AdaroundOptimizer.enable_caching_acts_data():
            all_inp_data, all_orig_out_data = act_sampler.sample_and_place_all_acts_on_cpu(cached_dataset,
                                                                                           cached_quant_dataset)
            # Place both the models temporarily to CPU
            # Try to put all cached activations data on GPU for faster optimization if possible.
            if 'cuda' in str(device):
                orig_model.cpu()
                quant_model.cpu()
                all_inp_data, all_orig_out_data = cls._place_cached_acts_data(all_inp_data, all_orig_out_data, device)

        for iteration in range(opt_params.num_iterations // world_size):
            if use_cache_acts_data and AdaroundOptimizer.enable_caching_acts_data():
                indices = torch.randperm(all_inp_data.size(0))[:BATCH_SIZE]
                inp_data = all_inp_data[indices].to(device)
                orig_out_data = all_orig_out_data[indices].to(device)
            else:
                model_inputs = cached_dataset[np.random.randint(len(cached_dataset))]
                inp_data, orig_out_data = act_sampler.sample_acts(model_inputs)

            # Clear alpha's gradients before optimization step
            optimizer.zero_grad()

            try:
                quant_out_data = cls._compute_output_with_adarounded_weights(quant_module, inp_data)
                if act_func is not None:
                    orig_out_data = act_func(orig_out_data)
                    quant_out_data = act_func(quant_out_data)

                # Calculate total loss
                recon_loss = AdaroundLoss.compute_recon_loss(quant_out_data, orig_out_data)
                round_loss = AdaroundLoss.compute_round_loss(quant_module.alpha, opt_params, iteration)
                total_loss = recon_loss + round_loss
                total_loss.backward()

            except RuntimeError as error:
                if use_cache_acts_data and 'cuda' in str(device) and AdaroundOptimizer.enable_caching_acts_data():
                    logger.debug("Not enough CUDA memory for adaround optimization."
                                 " Placed cached activations data on CPU. RuntimeError: %s", str(error))
                    all_inp_data = all_inp_data.cpu()
                    all_orig_out_data = all_orig_out_data.cpu()
                    continue
                raise error

            if dist.is_initialized():
                dist.all_reduce(quant_module.alpha.grad)
            quant_module.alpha.grad /= world_size

            optimizer.step()

        # Place both the models back to original device
        orig_model.to(device)
        quant_model.to(device)

    @classmethod
    def _compute_recons_metrics(cls, quant_module: AdaroundWrapperBase, act_func, inp_data: torch.Tensor,
                                out_data: torch.Tensor) -> Tuple[float, float]:
        """
        Compute Mean square error of output activations using soft rounding which maps alpha parameter
        between zero and one and hard rounding which maps to exact zero and one
        :param quant_module: Adaround wrapper module
        :param act_func: Activation function
        :param inp_data: Input data to quantized wrapper module
        :param out_data: Output data from module
        :return: Reconstruction error using hard rounding and soft rounding
        """

        # Enable hard rounding and get adaround wrapper module's output
        quant_module.use_soft_rounding = False
        out_data_hard = cls._compute_output_with_adarounded_weights(quant_module, inp_data)

        # Enable soft rounding and get adaround wrapper module's output
        quant_module.use_soft_rounding = True
        out_data_soft = cls._compute_output_with_adarounded_weights(quant_module, inp_data)

        # If followed by an activation function
        if act_func is not None:
            out_data = act_func(out_data)
            out_data_soft = act_func(out_data_soft)
            out_data_hard = act_func(out_data_hard)

        recons_err_soft = functional.mse_loss(out_data_soft, out_data)
        recons_err_hard = functional.mse_loss(out_data_hard, out_data)

        return float(recons_err_hard), float(recons_err_soft)

    @staticmethod
    def _compute_output_with_adarounded_weights(quant_module: AdaroundWrapperBase, inp_data: torch.Tensor):
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param quant_module: Adaround wrapper module
        :param inp_data: The input data to be used for computing the output
        :return: output of the module computed with AdaRounded weights
        """
        module = quant_module.get_original_module()

        # Compute adarounded weights
        device = inp_data.device
        quant_module.to(device)
        adarounded_weights = quant_module.apply_adaround(quant_module.weight)

        if isinstance(module, torch.nn.Conv2d):
            out_data = functional.conv2d(inp_data, adarounded_weights, bias=module.bias, stride=module.stride,
                                         dilation=module.dilation, padding=module.padding, groups=module.groups)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            out_data = functional.conv_transpose2d(inp_data, adarounded_weights, bias=module.bias,
                                                   stride=module.stride, padding=module.padding,
                                                   output_padding=module.output_padding, groups=module.groups,
                                                   dilation=module.dilation)
        elif isinstance(module, torch.nn.Linear):
            out_data = functional.linear(inp_data, adarounded_weights, bias=module.bias)

        else:
            raise ValueError('AdaRound is not supported for the module: ', module)

        return out_data

    @staticmethod
    def _can_cache_acts_data(num_batches: int, input_shape: torch.Size, output_shape: torch.Size, dtype: torch.dtype)\
            -> bool:
        """
        Function to check whether activations data can be cached and fit in CPU memory for given
        input and output shape in advance. The threshold CPU memory is determined by multiplying threshold and
        available CPU memory so that remaining CPU memory is available for other processes.

        NOTE: The threshold value is empirically chosen. Threshold ensures the safety from OOM for remaining run.

        :param num_batches: Number of batches.
        :param input_shape: Shape of input activations data.
        :param output_shape: Shape of output activations data.
        :param dtype: Data type of input/output activations data
        :return: True if we can cache, false otherwise.
        """
        can_cache_data = False

        # Available CPU memory in GB.
        threshold_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        threshold_mem = threshold_mem * EMPIRICAL_THRESHOLD

        # required CPU memory in GB.
        data_size_in_bits = 16 if dtype == torch.half else 32
        req_mem = 0
        req_mem += reduce(lambda x, y: x * y, input_shape) * num_batches * data_size_in_bits / (1024 * 1024 * 1024 * 8)
        req_mem += reduce(lambda x, y: x * y, output_shape) * num_batches * data_size_in_bits / (1024 * 1024 * 1024 * 8)

        if req_mem < threshold_mem:
            can_cache_data = True
        logger.debug("Placing cached activations data on CPU: %s, required_memory: %f GB, available_memory: %f GB",
                     str(can_cache_data), req_mem, threshold_mem)

        return can_cache_data

    @staticmethod
    def _place_cached_acts_data(inp_data: torch.Tensor, out_data: torch.Tensor, device: torch.device) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function decides whether cached activation data can be placed on device or not. If yes, it puts
        cached activation data to given device. If there is not enough device memory, it keeps the
        cached activation data to CPU memory.

        NOTE: The threshold value is empirically chosen. Threshold ensures the safety from OOM for remaining run.

        :param inp_data: Input activations data.
        :param out_data: Output activations data.
        :param device: Device.
        :return: Input and output activations data.
        """
        torch.cuda.empty_cache()

        # Available GPU memory in GB
        threshold_mem = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        threshold_mem = threshold_mem / (1024 * 1024 * 1024)
        threshold_mem = threshold_mem * EMPIRICAL_THRESHOLD

        # required GPU memory in GB
        data_size_in_bits = 16 if inp_data.dtype == torch.half else 32
        req_mem = 0
        req_mem += reduce(lambda x, y: x * y, inp_data.size())  * data_size_in_bits / (1024 * 1024 * 1024 * 8)
        req_mem += reduce(lambda x, y: x * y, out_data.size()) * data_size_in_bits / (1024 * 1024 * 1024 * 8)

        if req_mem < threshold_mem:
            try:
                inp_data = inp_data.to(device)
                out_data = out_data.to(device)
                logger.debug("Placed cached activations data on GPU.")
            except RuntimeError as error:
                inp_data = inp_data.cpu()
                out_data = out_data.cpu()
                logger.debug("Could not place cached activations data on GPU."
                             " Placed cached activations data on CPU. RuntimeError: %s", str(error))

        return inp_data, out_data

    @staticmethod
    def enable_caching_acts_data() -> bool:
        """
        Function to enable/disable caching intermediate activation data. By default, it returns True.
        """
        return AdaroundOptimizer.is_activation_caching_enabled
