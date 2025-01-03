# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Union, Tuple, Dict, List
from functools import reduce
import numpy as np
import psutil
import onnx
from onnx import numpy_helper
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset
from packaging import version  # pylint: disable=wrong-import-order

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_onnx.adaround.activation_sampler import ActivationSampler
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.adaround.utils import ModuleInfo, read_attributes_for_op
from aimet_onnx.utils import create_input_dict
from aimet_onnx.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters
from aimet_onnx.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
BATCH_SIZE = 32
EMPIRICAL_THRESHOLD = 3 / 4
DATA_SIZE_IN_BITS = 32
ACTIVATION_MAP = {'Relu': torch.nn.ReLU(), 'PRelu': torch.nn.PReLU(), 'Tanh': torch.nn.Tanh(),
                  'Clip': torch.nn.ReLU6(), 'Sigmoid': torch.nn.Sigmoid(), 'Softmax': torch.nn.Softmax()}


class AdaroundOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """
    @classmethod
    def adaround_module(cls, module: ModuleInfo, quantized_input_name: str,
                        orig_model: ModelProto, quant_model: QuantizationSimModel,
                        act_func: Union[torch.nn.Module, None], cached_dataset: Dataset,
                        opt_params: AdaroundHyperParameters, param_to_adaround_tensor_quantizer: Dict,
                        use_cuda: bool, device: int = 0, user_onnx_libs: List[str] = None):
        """
        Adaround module

        :param module: Original module's information
        :param quantized_input_name: Name of input to the quantized layer/ layer to be adarounded
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
         yielded from the data loader
        :param opt_params: Optimization parameters
        :param param_to_adaround_tensor_quantizer: Param name to adaround tensor quantizer dictionary
        :param use_cuda: If we should use cuda
        :param device: CUDA device ID
        :param user_onnx_libs: List of paths to all compiled ONNX custom ops libraries
        """
        # pylint: disable=too-many-arguments

        # Optimize weight rounding
        cls._optimize_rounding(module, quantized_input_name, orig_model, quant_model, act_func, cached_dataset,
                               opt_params, param_to_adaround_tensor_quantizer, use_cuda, device, user_onnx_libs)

        # After optimization, set the optimized layer's rounding mode to "Hard rounding"
        param_to_adaround_tensor_quantizer[module.params['weight'].name].use_soft_rounding = False

    # pylint: disable=too-many-statements
    @classmethod
    def _optimize_rounding(cls, module: ModuleInfo, quantized_input_name,
                           orig_model: ModelProto, quant_model: QuantizationSimModel,
                           act_func: Union[None, str], cached_dataset: Dataset,
                           opt_params: AdaroundHyperParameters, param_to_adaround_tensor_quantizer: Dict,
                           use_cuda: bool, device: int = 0, user_onnx_libs: List[str] = None):
        """
        Optimizes the weight rounding of quantized wrapper module
        :param module: Original module
        :param quantized_input_name: Name of input to the quantized layer/ layer to be adarounded
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
        :param opt_params: Optimization parameters
        :param param_to_adaround_tensor_quantizer: Param name to adaround tensor quantizer dictionary
        :param user_onnx_libs: List of paths to all compiled ONNX custom ops libraries
        """
        # pylint: disable=too-many-locals, too-many-arguments
        adaround_quantizer = param_to_adaround_tensor_quantizer[module.params['weight'].name]
        torch_device = 'cpu'
        if use_cuda:
            torch_device = 'cuda:' + str(device)
        weights = torch.from_numpy(numpy_helper.to_array(module.params['weight'].tensor)).to(torch_device)
        enable_grad(weights)

        adaround_quantizer.broadcast_offset_delta(weights)
        adaround_quantizer.initialize_alpha(weights, adaround_quantizer.broadcasted_delta)

        assert adaround_quantizer.use_soft_rounding, 'optimization should use soft rounding only.'
        assert adaround_quantizer.alpha is not None, 'alpha parameter should be initialized.'

        # Create and set up Adam optimizer with parameter 'alpha' to be optimized
        optimizer = torch.optim.Adam([adaround_quantizer.alpha])

        # Check if we can cache intermediate activation data.
        model_inputs = cached_dataset[0]
        act_sampler = ActivationSampler(module.outputs[0], quantized_input_name, orig_model, quant_model,
                                        use_cuda, device, user_onnx_libs)
        inp_data, out_data = act_sampler.sample_acts(create_input_dict(orig_model.model, model_inputs))
        inp_data_torch, out_data_torch = torch.from_numpy(inp_data[0]), torch.from_numpy(out_data[0])
        use_cache_acts_data = cls._can_cache_acts_data(len(cached_dataset), inp_data_torch.shape,
                                                       out_data_torch.shape, inp_data_torch.dtype)

        attributes = read_attributes_for_op(module)
        if 'pads' in attributes:
            if len(attributes['pads']) > 4:
                logger.info("Skipping the Convolution layer because padding size greater than 4 is not supported for optimization")
                return

        if use_cache_acts_data and AdaroundOptimizer.enable_caching_acts_data():
            logger.debug("Caching intermediate activations data for optimization.")
            all_inp_data, all_orig_out_data = act_sampler.sample_and_place_all_acts_on_cpu(cached_dataset)
            all_inp_data, all_orig_out_data = torch.from_numpy(all_inp_data[0]), \
                                         torch.from_numpy(all_orig_out_data[0])

            # Try to put all cached activations data on GPU for faster optimization if possible.
            if use_cuda:
                all_inp_data, all_orig_out_data = cls._place_cached_acts_data(all_inp_data, all_orig_out_data,
                                                                              torch_device)

        for iteration in range(opt_params.num_iterations):
            if use_cache_acts_data and AdaroundOptimizer.enable_caching_acts_data():
                indices = torch.randperm(all_inp_data.size(0))[:BATCH_SIZE]
                inp_data = all_inp_data[indices].to(torch_device)
                orig_out_data = all_orig_out_data[indices].to(torch_device)
            else:
                model_inputs = cached_dataset[np.random.randint(len(cached_dataset))]
                inp_data, orig_out_data = act_sampler.sample_acts(create_input_dict(orig_model.model, model_inputs))
                inp_data, orig_out_data = torch.from_numpy(inp_data[0]).to(torch_device), torch.from_numpy(out_data[0]).to(torch_device)
                # This assumes there's only 1 input and 1 output in the list output by sample_acts


            # Clear alpha's gradients before optimization step
            optimizer.zero_grad()

            # Get the module's output activations using AdaRounded weights
            quant_out_data = cls._compute_output_with_adarounded_weights(weights, module, inp_data, adaround_quantizer)

            # If followed by an activation function
            if act_func is not None:
                orig_out_data = ACTIVATION_MAP[act_func](orig_out_data)
                quant_out_data = ACTIVATION_MAP[act_func](quant_out_data)

            # Calculate total loss
            recon_loss = AdaroundLoss.compute_recon_loss(quant_out_data, orig_out_data)
            round_loss = AdaroundLoss.compute_round_loss(adaround_quantizer.alpha, opt_params, iteration)
            total_loss = recon_loss + round_loss

            # Back propagate and Update the parameter 'alpha'
            total_loss.backward()
            optimizer.step()

            if iteration == 0 or iteration % 100 == 0:
                logger.debug("After iterations=%d, Total loss=%5f, Recons. loss=%5f, Rounding loss=%5f",
                             iteration, float(total_loss), float(recon_loss), float(round_loss))

        adaround_quantizer.use_soft_rounding = True
        adarounded_weights = adaround_quantizer.adaround_weights(weights)
        weights = adarounded_weights.detach().cpu().numpy().tobytes()
        weight_name = module.params['weight'].name
        update_sim_weight(quant_model, weights, weight_name)

    @classmethod
    def _compute_recons_metrics(cls, quant_module: ModuleInfo, act_func: Union[None, str], inp_data: torch.Tensor,
                                out_data: torch.Tensor, param_to_adaround_tensor_quantizer: Dict,
                                use_cuda: bool, device: int = 0) -> Tuple[float, float]:
        """
        Compute Mean square error of output activations using soft rounding which maps alpha parameter
        between zero and one and hard rounding which maps to exact zero and one

        :param quant_module: Quantized wrapper module
        :param act_func: Activation function
        :param inp_data: Input data to quantized wrapper module
        :param out_data: Output data from module
        :param param_to_adaround_tensor_quantizer: Dict
        :param use_cuda: Bool, true if we use GPU
        :param device: Cuda device
        :return: Reconstruction error using hard rounding and soft rounding
        """
        adaround_quantizer = param_to_adaround_tensor_quantizer[quant_module.params['weight'].name]
        torch_device = 'cpu'
        if use_cuda:
            torch_device = 'cuda:' + str(device)
        weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor)).to(torch_device)
        inp_data = inp_data.to(torch_device)
        # Enable hard rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = False
        out_data_hard = cls._compute_output_with_adarounded_weights(weights, quant_module, inp_data, adaround_quantizer)

        # Enable soft rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = True
        out_data_soft = cls._compute_output_with_adarounded_weights(weights, quant_module, inp_data, adaround_quantizer)

        # If followed by an activation function
        if act_func is not None:
            out_data = ACTIVATION_MAP[act_func](out_data)
            out_data_soft = ACTIVATION_MAP[act_func](out_data_soft)
            out_data_hard = ACTIVATION_MAP[act_func](out_data_hard)

        recons_err_soft = functional.mse_loss(out_data_soft, out_data)
        recons_err_hard = functional.mse_loss(out_data_hard, out_data)

        return float(recons_err_hard), float(recons_err_soft)

    @staticmethod
    def _compute_output_with_adarounded_weights(weights: torch.Tensor, quant_module, inp_data: torch.Tensor,
                                                adaround_quantizer: AdaroundTensorQuantizer):
        """
        Compute output of AdaroundSupportedModules with adarounded weights

        :param weights: Torch tensor weights to be adarounded
        :param quant_module: Quantized wrapper module
        :param inp_data: The input data to be used for computing the output
        :param adaround_quantizer: Adaround tensor quantizer
        :return: output of the module computed with AdaRounded weights
        """
        # Compute adarounded weights
        # pylint: disable=too-many-branches

        device = 'cpu'
        if inp_data.is_cuda:
            device = inp_data.device

        adarounded_weights = adaround_quantizer.adaround_weights(weights)

        if quant_module.type == 'Conv':
            attributes = read_attributes_for_op(quant_module)
            if 'pads' in attributes:
                onnx_padding = attributes['pads']
                torch_padding = [onnx_padding[1], onnx_padding[3], onnx_padding[0], onnx_padding[2]]
                # Takes care of asymmetric padding within a spatial axis
                inp_data = functional.pad(inp_data, pad=torch_padding)
            else:
                auto_pad = attributes.get("auto_pad", "NOTSET")
                if auto_pad not in {"NOTSET", "VALID"}:
                    raise NotImplementedError(f"Layer with auto_pad: {auto_pad} attribute is not supported.")
            bias = None
            if 'bias' in quant_module.params:
                bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor)).to(device)
            out_data = functional.conv2d(inp_data, adarounded_weights, bias=bias, stride=attributes.get('strides', 1),
                                         dilation=attributes.get('dilations', 1), groups=attributes.get('group', 1))
        elif quant_module.type == 'ConvTranspose':
            attributes = read_attributes_for_op(quant_module)
            if attributes.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
                raise NotImplementedError("Layers with auto_pad attribute are currently not supported")
            onnx_padding = attributes.get('pads', [0, 0, 0, 0])
            torch_padding = [-onnx_padding[i] for i in (1, 3, 0, 2)]
            bias = None
            if 'bias' in quant_module.params:
                bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor)).to(device)
            out_data = functional.conv_transpose2d(inp_data, adarounded_weights, bias=bias, stride=attributes.get('strides', 1),
                                                   dilation=attributes.get('dilations', 1), groups=attributes.get('group', 1))
            out_data = functional.pad(out_data, pad=torch_padding)
        elif quant_module.type in ['Gemm']:
            if not quant_module.transposed_params:
                # Pytorch requires tranposed weights in functional.linear
                adarounded_weights = adarounded_weights.t()
            bias = None
            if 'bias' in quant_module.params:
                bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor)).to(device)
            out_data = functional.linear(inp_data, adarounded_weights, bias=bias)
        elif quant_module.type in ['MatMul']:
            out_data = torch.matmul(inp_data, adarounded_weights)

        else:
            raise ValueError('AdaRound is not supported for the module type: ', quant_module.type)

        return out_data

    @staticmethod
    def enable_caching_acts_data() -> bool:
        """
        Function to enable/disable caching intermediate activation data. By default, it returns True.
        """
        return True

    @staticmethod
    def _can_cache_acts_data(num_batches: int, input_shape: torch.Size, output_shape: torch.Size, dtype: torch.dtype) \
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
        req_mem += reduce(lambda x, y: x * y, inp_data.size()) * data_size_in_bits / (1024 * 1024 * 1024 * 8)
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


def enable_grad(tensor: torch.Tensor):
    """
    Enables gradient

    :param tensor: Tensor for which we should enable grad
    """
    if tensor.is_leaf:
        tensor.requires_grad = True

def update_sim_weight(quant_model: onnx.ModelProto, weights: onnx.TensorProto, weight_name: str):
    """
    Updates weights in sim for a given name

    :param quant_model: Quantized model
    :param weights: Weight tensor
    :param weight_name: Name of the weight to be updated
    """
    for tensor in quant_model.model.graph.initializer:
        if tensor.name == weight_name:
            tensor.raw_data = weights
            return
    logger.info("Could not find %s in QuantSim model", weight_name)
