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

"""Adaround optimizer"""

from typing import Union, Tuple, Dict, List
from functools import reduce
import numpy as np
import psutil
import onnx
from onnx import numpy_helper
import torch
from torch.nn import functional
from torch.utils.data import Dataset
from packaging import version

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_onnx.adaround.activation_sampler import ActivationSampler
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.adaround.utils import (
    ModuleInfo,
    read_attributes_for_op,
    apply_activation_fn,
)
from aimet_onnx.utils import create_input_dict, get_torch_device
from aimet_onnx.adaround.adaround_loss import AdaroundLoss
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

# enable/disable caching intermediate activation data
ENABLE_CACHING_ACTIVATIONS = True


class AdaroundOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """

    @classmethod
    def adaround_module(
        cls,
        module: ModuleInfo,
        quantized_input_name: str,
        quant_model: QuantizationSimModel,
        fp32_model: ModelProto,
        act_func: Union[str, None],
        cached_dataset: Dataset,
        num_iterations: int,
        param_to_adaround_tensor_quantizer: Dict,
    ):
        """
        Adaround module

        :param module: Original module's information
        :param quantized_input_name: Name of input to the quantized layer/ layer to be adarounded
        :param quant_model: QuantSim model
        :param fp32_model: FP32 model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
         yielded from the data loader
        :param num_iterations: Num of iterations to adaround a layer
        :param param_to_adaround_tensor_quantizer: Param name to adaround tensor quantizer dictionary
        """
        # pylint: disable=too-many-locals, too-many-arguments, too-many-statements
        adaround_quantizer = param_to_adaround_tensor_quantizer[
            module.params["weight"].name
        ]
        torch_device = get_torch_device(quant_model.session)
        weights = torch.from_numpy(
            numpy_helper.to_array(module.params["weight"].tensor)
        ).to(torch_device)
        enable_grad(weights)

        adaround_quantizer.broadcast_offset_delta(weights)
        adaround_quantizer.initialize_alpha(weights)

        assert adaround_quantizer.alpha is not None, (
            "alpha parameter should be initialized."
        )

        # Create and set up Adam optimizer with parameter 'alpha' to be optimized
        optimizer = torch.optim.Adam([adaround_quantizer.alpha])

        # Check if we can cache intermediate activation data.
        model_inputs = cached_dataset[0]
        act_sampler = ActivationSampler(
            module.outputs[0],
            quantized_input_name,
            quant_model,
            fp32_model,
        )
        inp_data, out_data = act_sampler.sample_acts(
            create_input_dict(quant_model.model.model, model_inputs)
        )

        if len(inp_data) == 0:
            act_sampler.restore_graph()
            logger.warning(
                f"Unable to retrieve data for the tensor {quantized_input_name}. Skipping AdaRound optimization for the layer {module.name}"
            )
            return

        inp_data_torch, out_data_torch = (
            torch.from_numpy(inp_data[0]),
            torch.from_numpy(out_data[0]),
        )
        use_cache_acts_data = cls._can_cache_acts_data(
            len(cached_dataset),
            inp_data_torch.shape,
            out_data_torch.shape,
            inp_data_torch.dtype,
        )

        if use_cache_acts_data and ENABLE_CACHING_ACTIVATIONS:
            logger.debug("Caching intermediate activations data for optimization.")
            all_inp_data_np, all_orig_out_data_np = (
                act_sampler.sample_and_place_all_acts_on_cpu(cached_dataset)
            )
            all_inp_data = [
                torch.from_numpy(inp_data_np) for inp_data_np in all_inp_data_np
            ]
            all_orig_out_data = [
                torch.from_numpy(orig_out_data_np)
                for orig_out_data_np in all_orig_out_data_np
            ]

            # Try to put all cached activations data on GPU for faster optimization if possible.
            if torch_device != torch.device("cpu"):
                all_inp_data, all_orig_out_data = cls._place_cached_acts_data(
                    all_inp_data, all_orig_out_data, torch_device
                )

        for iteration in range(num_iterations):
            if use_cache_acts_data and ENABLE_CACHING_ACTIVATIONS:
                # batch idx is chosen using iteration % len(all_inp_data_np). Of all the samples in a given batch,
                # min(batch size, BATCH_SIZE) is operated on in a single iteration
                indices = torch.randperm(
                    all_inp_data[iteration % len(all_inp_data_np)].size(0)
                )[:BATCH_SIZE]
                inp_data = all_inp_data[iteration % len(all_inp_data_np)][indices].to(
                    torch_device
                )
                orig_out_data = all_orig_out_data[iteration % len(all_inp_data_np)][
                    indices
                ].to(torch_device)
            else:
                model_inputs = cached_dataset[np.random.randint(len(cached_dataset))]
                inp_data, orig_out_data = act_sampler.sample_acts(
                    create_input_dict(quant_model.model.model, model_inputs)
                )
                inp_data, orig_out_data = (
                    torch.from_numpy(inp_data[iteration % len(inp_data)]).to(
                        torch_device
                    ),
                    torch.from_numpy(orig_out_data[iteration % len(inp_data)]).to(
                        torch_device
                    ),
                )
                # This assumes there's only 1 input and 1 output in the list output by sample_acts

            # Clear alpha's gradients before optimization step
            optimizer.zero_grad()

            # Get the module's output activations using AdaRounded weights
            quant_out_data = cls._compute_output_with_adarounded_weights(
                weights, module, inp_data, adaround_quantizer, True
            )

            orig_out_data = apply_activation_fn(act_func, orig_out_data)
            quant_out_data = apply_activation_fn(act_func, quant_out_data)

            # Calculate total loss
            recon_loss = AdaroundLoss.compute_recon_loss(quant_out_data, orig_out_data)
            round_loss = AdaroundLoss.compute_round_loss(
                adaround_quantizer.alpha, num_iterations, iteration
            )
            total_loss = recon_loss + round_loss

            # Back propagate and Update the parameter 'alpha'
            total_loss.backward()
            optimizer.step()

            if iteration == 0 or iteration % 100 == 0:
                logger.debug(
                    "After iterations=%d, Total loss=%5f, Recons. loss=%5f, Rounding loss=%5f",
                    iteration,
                    float(total_loss),
                    float(recon_loss),
                    float(round_loss),
                )

        adarounded_weights = adaround_quantizer.adaround_weights(weights, True)
        weights = adarounded_weights.detach().cpu().numpy().tobytes()
        weight_name = module.params["weight"].name
        update_sim_weight(quant_model.model, weights, weight_name)
        act_sampler.restore_graph()

    @staticmethod
    def _compute_output_with_adarounded_weights(
        weights: torch.Tensor,
        quant_module,
        inp_data: torch.Tensor,
        adaround_quantizer: AdaroundTensorQuantizer,
        use_soft_rounding: bool,
    ):
        """
        Compute output of AdaroundSupportedModules with adarounded weights

        :param weights: Torch tensor weights to be adarounded
        :param quant_module: Quantized wrapper module
        :param inp_data: The input data to be used for computing the output
        :param adaround_quantizer: Adaround tensor quantizer
        :param use_soft_rounding: Soft rounding maps alpha parameter between zero and one using rectified sigmoid function,
         and hard rounding maps it to exactly zero or one
        :return: output of the module computed with AdaRounded weights
        """
        # Compute adarounded weights
        # pylint: disable=too-many-branches

        device = "cpu"
        if inp_data.is_cuda:
            device = inp_data.device

        adarounded_weights = adaround_quantizer.adaround_weights(
            weights, use_soft_rounding
        )

        if quant_module.type == "Conv":
            attributes = read_attributes_for_op(quant_module)
            if "pads" in attributes:
                onnx_padding = attributes["pads"]
                torch_padding = [
                    onnx_padding[1],
                    onnx_padding[3],
                    onnx_padding[0],
                    onnx_padding[2],
                ]
                # Takes care of asymmetric padding within a spatial axis
                inp_data = functional.pad(inp_data, pad=torch_padding)
            else:
                auto_pad = attributes.get("auto_pad", "NOTSET")
                if auto_pad not in {"NOTSET", "VALID"}:
                    raise NotImplementedError(
                        f"Layer with auto_pad: {auto_pad} attribute is not supported."
                    )
            bias = None
            if "bias" in quant_module.params:
                bias = torch.from_numpy(
                    numpy_helper.to_array(quant_module.params["bias"].tensor)
                ).to(device)
            out_data = functional.conv2d(
                inp_data,
                adarounded_weights,
                bias=bias,
                stride=attributes.get("strides", 1),
                dilation=attributes.get("dilations", 1),
                groups=attributes.get("group", 1),
            )
        elif quant_module.type == "ConvTranspose":
            attributes = read_attributes_for_op(quant_module)
            if attributes.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
                raise NotImplementedError(
                    "Layers with auto_pad attribute are currently not supported"
                )
            onnx_padding = attributes.get("pads", [0, 0, 0, 0])
            torch_padding = [-onnx_padding[i] for i in (1, 3, 0, 2)]
            bias = None
            if "bias" in quant_module.params:
                bias = torch.from_numpy(
                    numpy_helper.to_array(quant_module.params["bias"].tensor)
                ).to(device)
            out_data = functional.conv_transpose2d(
                inp_data,
                adarounded_weights,
                bias=bias,
                stride=attributes.get("strides", 1),
                dilation=attributes.get("dilations", 1),
                groups=attributes.get("group", 1),
            )
            out_data = functional.pad(out_data, pad=torch_padding)
        elif quant_module.type in ["Gemm"]:
            if not quant_module.transposed_params:
                # Pytorch requires tranposed weights in functional.linear
                adarounded_weights = adarounded_weights.t()
            bias = None
            if "bias" in quant_module.params:
                bias = torch.from_numpy(
                    numpy_helper.to_array(quant_module.params["bias"].tensor)
                ).to(device)
            out_data = functional.linear(inp_data, adarounded_weights, bias=bias)
        elif quant_module.type in ["MatMul"]:
            out_data = torch.matmul(inp_data, adarounded_weights)

        else:
            raise ValueError(
                "AdaRound is not supported for the module type: ", quant_module.type
            )

        return out_data

    @staticmethod
    def _can_cache_acts_data(
        num_batches: int,
        input_shape: torch.Size,
        output_shape: torch.Size,
        dtype: torch.dtype,
    ) -> bool:
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
        req_mem += (
            reduce(lambda x, y: x * y, input_shape)
            * num_batches
            * data_size_in_bits
            / (1024 * 1024 * 1024 * 8)
        )
        req_mem += (
            reduce(lambda x, y: x * y, output_shape)
            * num_batches
            * data_size_in_bits
            / (1024 * 1024 * 1024 * 8)
        )

        if req_mem < threshold_mem:
            can_cache_data = True
        logger.debug(
            "Placing cached activations data on CPU: %s, required_memory: %f GB, available_memory: %f GB",
            str(can_cache_data),
            req_mem,
            threshold_mem,
        )

        return can_cache_data

    @staticmethod
    def _place_cached_acts_data(
        inp_data: List[torch.Tensor], out_data: List[torch.Tensor], device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Function decides whether cached activation data can be placed on device or not. If yes, it puts
        cached activation data to given device. If there is not enough device memory, it keeps the
        cached activation data to CPU memory.

        NOTE: The threshold value is empirically chosen. Threshold ensures the safety from OOM for remaining run.

        :param inp_data: List of input activations data.
        :param out_data: List of output activations data.
        :param device: Device.
        :return: Input and output activations data.
        """
        torch.cuda.empty_cache()

        # Available GPU memory in GB
        threshold_mem = torch.cuda.get_device_properties(
            device
        ).total_memory - torch.cuda.memory_allocated(device)
        threshold_mem = threshold_mem / (1024 * 1024 * 1024)
        threshold_mem = threshold_mem * EMPIRICAL_THRESHOLD

        # required GPU memory in GB
        data_size_in_bits = 16 if inp_data[0].dtype == torch.half else 32

        tensor_size = 0
        for tensor in inp_data + out_data:
            tensor_size += reduce(lambda x, y: x * y, tensor.size())

        req_mem = tensor_size * data_size_in_bits / (1024 * 1024 * 1024 * 8)

        if req_mem < threshold_mem:
            try:
                inp_data = [t.to(device) for t in inp_data]
                out_data = [t.to(device) for t in out_data]
                logger.debug("Placed cached activations data on GPU.")
            except RuntimeError as error:
                inp_data = [t.to("cpu") for t in inp_data]
                out_data = [t.to("cpu") for t in out_data]

                logger.debug(
                    "Could not place cached activations data on GPU."
                    " Placed cached activations data on CPU. RuntimeError: %s",
                    str(error),
                )

        return inp_data, out_data


def enable_grad(tensor: torch.Tensor):
    """
    Enables gradient

    :param tensor: Tensor for which we should enable grad
    """
    if tensor.is_leaf:
        tensor.requires_grad = True


def update_sim_weight(
    quant_model: onnx.ModelProto, weights: onnx.TensorProto, weight_name: str
):
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
