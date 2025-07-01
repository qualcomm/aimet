# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""AdaScale implementation"""

from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Callable, List, Any, Tuple, Type

import torch
from torch.utils.data import DataLoader

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer

from aimet_common.utils import AimetLogger
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn import QuantizedLinear, compute_param_encodings, QuantizedConv2d
from aimet_torch.v2.utils import (
    default_forward_fn,
    remove_all_quantizers,
    remove_activation_quantizers,
    patch_attr,
)
from aimet_torch.experimental.adascale.adascale_quantizer import (
    AdaScaleQuantizeDequantize,
    AdaScaleLinearQuantizeDequantize,
    AdaScaleConv2dQuantizeDequantize,
)
from aimet_torch.blockwise_sampler import (
    BlockwiseSampler,
    change_tensor_and_cache_device_placement,
)
from aimet_torch.utils import get_device


@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    LlamaModel: AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    Qwen2Model: AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 0.5
_LOSS_FN = torch.nn.MSELoss()

supported_modules: List = [QuantizedLinear, QuantizedConv2d]


class AdaScale:
    """
    AdaScale is PTQ technique which performs Knowledge Distillation on blocks of modules by using the FP32 output as its
    reference output. Adascale is based on FlexRound: https://arxiv.org/abs/2306.00317 but integrates LWC from Omniquant.

    The optimization is performed on a block-by-block basis by comparing the quantized output of the block with its FP32
    equivalent and by training the parameters (gamma, beta, s2, s3) which are temporarily introduced in every supported
    module.

    A block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor
    Currently only Linear layers are supported, and all the linears in a block are optimized at the same time.

    While performing the optimization, the activation quantizers are disabled, linear modules' weight quantizers are
    changed to specialized QDQ (with learnable parameters introduced) and rest of the param's are left quantized with
    default QuantizeDequantize.


    """

    @classmethod
    def apply_adascale(
        cls,
        qsim: QuantizationSimModel,
        data_loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
        num_iterations: int = 1500,
    ):
        """
        :param qsim: Quantization Sim model
        :param data_loader: DataLoader object to load the input data
        :param forward_fn: forward function to run the forward pass of the model
        :param num_iterations: Number of iterations to optimize for during AdaScale BKD

        Note that the forward_fn should take exactly two arguments -
        1) the model
        2) The object returned from the dataloader irrespective of whether it's a tensor/tuple of tensors/dict/etc

        The forward_fn should prepare the "input sample" as needed and call the forward pass in the very end. The forward_fn
        should not be running any sort of eval, creating full dataloader inside the method, etc.

        Example usage:
            >>> model = DummyModel()
            >>> dummy_input = ...
            >>> data_set = DataSet(dummy_input)
            >>> data_loader = DataLoader(data_set, ...)
            >>> sim = QuantizationSimModel(model, dummy_input)
            >>> apply_adascale(sim, data_loader, forward_fn=forward_fn, num_iterations=1500)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=too-many-locals
        if not forward_fn:
            forward_fn = default_forward_fn

        compute_param_encodings(qsim.model)

        adascale_blocks = cls._get_blocks(qsim)

        # replace with adascale weight quantizer which introduces trainable params - beta, gamma, s2, s3
        device = get_device(qsim.model)
        cls._replace_with_adascale_weight_quantizers(adascale_blocks)
        qsim.model.to("cpu")

        sampler = BlockwiseSampler(
            qsim,
            adascale_blocks,
            data_loader,
            forward_fn,
            keep_unused_blocks_on_cpu=True,
            cache_activations_on_disk=True,
        )

        qsim.model.requires_grad_(False)
        beta_gamma_lr, scales_lr = AdaScale._model_specific_lr(qsim)

        with remove_activation_quantizers(adascale_blocks):
            for block, fp_block_inputs, qt_block_inputs in sampler.sample(
                device=device, desc="AdaScale blocks processed"
            ):
                # only set adascale params to train mode
                all_beta_gamma_parameters, all_scale_parameters = (
                    cls._get_adascale_trainable_params(block)
                )
                trainable_params = [
                    {"params": all_beta_gamma_parameters, "lr": beta_gamma_lr},
                    {"params": all_scale_parameters, "lr": scales_lr},
                ]
                optimizer = torch.optim.Adam(trainable_params)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_iterations, eta_min=0.0
                )
                cls._set_requires_grad(
                    all_beta_gamma_parameters + all_scale_parameters, True
                )

                fp_out = []  # save fp batchwise block outputs to use across epochs
                for batch_idx in range(len(data_loader)):
                    with remove_all_quantizers(block):
                        with patch_attr(
                            torch.Tensor,
                            "__deepcopy__",
                            lambda self, memo: self.detach().clone(),
                        ):
                            with fp_block_inputs[batch_idx].load():
                                fp_args = change_tensor_and_cache_device_placement(
                                    deepcopy(fp_block_inputs[batch_idx].args), device
                                )
                                fp_kwargs = change_tensor_and_cache_device_placement(
                                    deepcopy(fp_block_inputs[batch_idx].kwargs), device
                                )
                        fp_block_results = change_tensor_and_cache_device_placement(
                            block(*fp_args, **fp_kwargs), "cpu"
                        )
                        fp_out.append(fp_block_results)
                        del fp_args, fp_kwargs

                curr_iteration = 0
                while curr_iteration < num_iterations:
                    for batch_idx in range(len(data_loader)):
                        curr_iteration += 1
                        if curr_iteration > num_iterations:
                            break
                        with torch.set_grad_enabled(True):
                            with patch_attr(
                                torch.Tensor,
                                "__deepcopy__",
                                lambda self, memo: self.detach().clone(),
                            ):
                                with qt_block_inputs[batch_idx].load():
                                    qt_args = change_tensor_and_cache_device_placement(
                                        deepcopy(qt_block_inputs[batch_idx].args),
                                        device,
                                    )
                                    qt_kwargs = (
                                        change_tensor_and_cache_device_placement(
                                            deepcopy(qt_block_inputs[batch_idx].kwargs),
                                            device,
                                        )
                                    )

                                with fp_block_inputs[batch_idx].load():
                                    fp_args = change_tensor_and_cache_device_placement(
                                        deepcopy(fp_block_inputs[batch_idx].args),
                                        device,
                                    )

                                if _QT_SAMPLING_PROB == 1.0:
                                    combined_args = qt_args
                                elif _QT_SAMPLING_PROB == 0.0:
                                    combined_args = fp_args
                                else:
                                    combined_args = tuple(
                                        torch.where(
                                            torch.rand_like(qt_arg, dtype=qt_arg.dtype)
                                            < _QT_SAMPLING_PROB,
                                            qt_arg,
                                            fp_arg,
                                        )
                                        for qt_arg, fp_arg in zip(qt_args, fp_args)
                                    )

                            quant_out = block(*combined_args, **qt_kwargs)
                            if isinstance(quant_out, tuple):
                                quant_out = torch.cat(quant_out)
                            del qt_args, fp_args, combined_args, qt_kwargs

                            batch_fp_out = change_tensor_and_cache_device_placement(
                                deepcopy(fp_out[batch_idx]), device
                            )
                            if isinstance(batch_fp_out, tuple):
                                batch_fp_out = torch.cat(batch_fp_out)

                            loss = _LOSS_FN(quant_out, batch_fp_out)

                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                            del quant_out, batch_fp_out, loss

        del sampler

        cls._fold_weights_and_replace_with_qdq(adascale_blocks)
        qsim.model.to(device)

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        """
        Helper to get the model type to optimize
        This is needed because the target module might not be at the top level in which case we go deeper and fetch it
        """
        for module in model.modules():
            for target in adascale_model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(qsim: QuantizationSimModel):
        """helper to get all the blocks in the model represented by adascale_model_config_dict"""

        target_type = AdaScale._screen_for_target_type(qsim.model)
        block_type = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        ).block_type
        target_modules = []
        if block_type is not None:
            target_modules = [
                m for m in qsim.model.modules() if isinstance(m, block_type)
            ]
        return target_modules

    @staticmethod
    def _model_specific_lr(qsim: QuantizationSimModel) -> tuple[float, float]:
        """
        Given the sim object, query the model type and return the custom lr to be used
        """
        target_type = AdaScale._screen_for_target_type(qsim.model)
        model_config = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        )
        return model_config.beta_gamma_lr, model_config.scales_lr

    @classmethod
    def _replace_with_adascale_weight_quantizers(cls, adascale_blocks: List):
        """Replace all the weight quantizers in supported modules with adascale quantizers"""
        for block in adascale_blocks:
            for layer in block.modules():
                if isinstance(layer, tuple(supported_modules)):
                    layer.param_quantizers["weight"] = cls._get_adascale_qdq_mapping()[
                        type(layer)
                    ](layer.param_quantizers["weight"], layer.weight.shape)

    @classmethod
    @torch.no_grad()
    def _fold_weights_and_replace_with_qdq(
        cls, adascale_blocks: List, allow_overwrite: bool = False
    ):
        """Replace adascale weight quantizers back with QDQ, and fold the params s2, s3 into weight"""
        for block in adascale_blocks:
            for layer in block.modules():
                if isinstance(layer, tuple(supported_modules)):
                    layer.weight.copy_(
                        layer.param_quantizers["weight"].get_folded_weight(layer.weight)
                    )
                    layer.param_quantizers["weight"] = layer.param_quantizers[
                        "weight"
                    ].get_qdq()
                    layer.param_quantizers["weight"].allow_overwrite(allow_overwrite)
                    layer.requires_grad_(False)

    @staticmethod
    def _get_adascale_trainable_params(
        non_leaf_module: torch.nn.Module,
    ) -> Tuple[List, List]:
        """Get all the adascale scale params present in the non-leaf module"""
        all_scale_parameters = []
        all_beta_gamma_parameters = []
        for module in non_leaf_module.modules():
            if isinstance(module, tuple(supported_modules)) and isinstance(
                module.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                beta_gamma_params, scale_parameters = module.param_quantizers[
                    "weight"
                ].get_adascale_trainable_parameters()
                all_beta_gamma_parameters.extend(beta_gamma_params)
                all_scale_parameters.extend(scale_parameters)
        return all_beta_gamma_parameters, all_scale_parameters

    @staticmethod
    def _set_requires_grad(adascale_params: list, val: bool):
        """Helper to update requires_grad to the input `val` for all the params in `adascale_params`"""
        for p in adascale_params:
            p.requires_grad = val

    @staticmethod
    def _get_adascale_qdq_mapping() -> dict:
        return {
            QuantizedLinear: AdaScaleLinearQuantizeDequantize,
            QuantizedConv2d: AdaScaleConv2dQuantizeDequantize,
        }


apply_adascale = AdaScale.apply_adascale
