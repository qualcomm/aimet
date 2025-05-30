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
"""Optimizer for Omniquant"""

import contextlib
from copy import deepcopy
import os
from pathlib import Path
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.peft_model import PeftModel
from safetensors.numpy import save_file, load_file
import torch
from typing import Union, Callable
import time

from aimet_torch._base.quantsim import logger
from aimet_torch.blockwise_sampler import (
    BlockwiseSampler,
    change_tensor_and_cache_device_placement,
)
from aimet_torch.utils import disable_all_quantizers
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.nn import compute_param_encodings
from aimet_common.utils import AimetLogger
from .decoder_processor import get_transformer_processor

from ._utils import (
    get_sqnr,
    disable_quantizers_for_omq,
    freeze_let_optimized_param_quantizers,
    replace_with_omniquant_weight_quantizers,
    SUPPORTED_QUANTIZED_MODULES,
    OMQ_QUANTIZERS,
)

OMNIQUANT_ARTIFACT_DIR = "./aimet_omniquant_artifact/"
OMNIQUANT_METADATA_SAFETENSOR_NAME = "aimet_omniquant_metadata.safetensor"
OMNIQUANT_COMPUTE_SQNR = True
OMNIQUANT_LR = 5e-4  # 1st fp/qt to choose input source on fp block to get ground truth. 2nd fp/qt to choose input source on qt block to get prediction.
CACHE_ON_CPU = True  # Will be removed after using blockwise sampler.

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class Omniquant:
    """
    Omniquant for Post Training Quantization (PTQ)
    """

    # pylint: disable=too-many-arguments
    @classmethod
    def apply_omniquant(
        cls,
        quant_sim: QuantizationSimModel,
        dataloader,
        forward_fn: Callable,
        num_iterations: int = 800,
        output_path: str = OMNIQUANT_ARTIFACT_DIR,
    ):
        """
        Returns model with with omniquant weight, and save metadata in safetensor format to output path. Metadata safetensor
        can be used in update_lora_weights to update lora adaptor weights for peft lora model.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param dataloader: Dataloader used to train model.
        :param forward_fn: Model forward function used to cache intermediate data.
                           Expect to have model and inputs as function argument. e.g. lambda model, inputs: model(*inputs)
        :param num_iterations: Number of iterations to train each block with omniquant.
        :param output_path: Path to save {layer_name: scale} metadata safetensor.
        :return: Model with Omniquant weights.
        """
        num_batch = len(dataloader)

        @contextlib.contextmanager
        def disable_dynamic_cache():
            # Disable dynamic_cache for LET blockwise training, and restore after optimization.
            quant_sim_use_cache_bool = quant_sim.model.config.use_cache
            quant_sim.model.config.use_cache = False
            try:
                yield
            finally:
                quant_sim.model.config.use_cache = quant_sim_use_cache_bool

        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)

        start_omq_optmztn_time = time.perf_counter()
        with disable_dynamic_cache():
            cls._apply_omniquant(
                quant_sim,
                dataloader,
                forward_fn,
                num_iterations,
                num_batch,
                output_path,
            )
        total_omq_optmztn_time = time.perf_counter() - start_omq_optmztn_time
        _logger.info("Took %.4f seconds for omq optimization ", total_omq_optmztn_time)

    # pylint: disable=too-many-locals
    # pylint: disable=unused-variable
    # pylint: disable=unused-argument
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-statements
    @classmethod
    def _apply_omniquant(
        cls,
        quant_sim: QuantizationSimModel,
        dataloader,
        forward_fn,
        num_iterations: int,
        num_batch: int,
        output_path: str,
    ) -> torch.nn.Module:
        """
        Implemenatation to run omniquant optimization block by block. Return model with optimized weights.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param dataloader: Dataloader used to train model.
        :param forward_fn: Model forward function used to cache intermediate data.
                           Expect to have model and inputs as function argument. e.g. lambda model, inputs: model(*inputs)
        :param num_iterations: Number of iterations to train each block with omniquant.
        :param num_batch: Number of batches in dataloader.
        :param output_path: Path where to store artifacts.
        """
        device = quant_sim.model.device
        if quant_sim.model.device.type != "cuda":
            _logger.info(
                "Omniquant optimization is recommended to be run on a GPU system"
            )

        # Disable activation quantizers for all quantized module during LET blockwise training, and restore after optimization.
        # Disable param quantizers except for linear and conv during LET blockwise training, and restore after optimization.
        with disable_quantizers_for_omq(quant_sim.model):
            compute_param_encodings(quant_sim.model)
            transformer_processor = get_transformer_processor(quant_sim.model)
            qt_transformer_block_list = transformer_processor.get_decoder_list(
                quant_sim
            )
            replace_with_omniquant_weight_quantizers(qt_transformer_block_list)

            # num_repeats is used for setting the foll_scale shape for GQA pair (self_attn.v_proj, self.attn_o_proj)
            num_repeats = (
                quant_sim.model.config.num_attention_heads
                // quant_sim.model.config.num_key_value_heads
            )
            sampler = BlockwiseSampler(
                sim=quant_sim,
                blocks=qt_transformer_block_list,
                dataloader=dataloader,
                forward_fn=forward_fn,
                keep_unused_blocks_on_cpu=True,
                cache_activations_on_disk=True,
            )

            _logger.info("Starting blockwise training for params")
            for block_num, (qt_block, fp_block_inputs, qt_block_inputs) in enumerate(
                sampler.sample()
            ):
                qt_let_pair_list = transformer_processor.get_let_module_pair(qt_block)
                transformer_processor.init_let_params(qt_let_pair_list, num_repeats)
                qt_block = qt_block.to(device)

                def _set_qt_params_trainable(qt_block):
                    for name, module in qt_block.named_modules():
                        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                            if module.param_quantizers.weight:
                                for (
                                    param
                                ) in module.param_quantizers.weight.parameters():
                                    param.requires_grad = True

                _set_qt_params_trainable(qt_block)

                encoding_params, param_names = cls._get_trainable_params(qt_block)

                grouped_params = [
                    {
                        "params": encoding_params,
                        "lr": OMNIQUANT_LR,
                        "weight_decay": 0.0,
                    },
                ]

                optimizer = torch.optim.AdamW(grouped_params)
                loss_fn = torch.nn.MSELoss(reduction="sum")
                curr_iteration = 0

                while curr_iteration < num_iterations:
                    sqnr_list, loss_list = [], []
                    for batch_num in range(num_batch):
                        curr_iteration += 1
                        if curr_iteration > num_iterations:
                            break
                        # Do block-wise training.
                        with torch.set_grad_enabled(True):
                            with fp_block_inputs[batch_num].load():
                                with qt_block_inputs[batch_num].load():
                                    loss, sqnr = cls._block_wise_training_step(
                                        fp_block_inputs[batch_num],
                                        qt_block_inputs[batch_num],
                                        qt_block,
                                        qt_let_pair_list,
                                        optimizer,
                                        loss_fn,
                                        device,
                                    )

                        sqnr_list.append(sqnr)
                        loss_list.append(loss)

                    # Get message for sqnr and loss mean
                    loss_mean = torch.stack(loss_list).mean()
                    log_msg = f"layer {block_num} | loss: {loss_mean:.3f}"
                    if OMNIQUANT_COMPUTE_SQNR:
                        sqnr_mean = torch.stack(sqnr_list).mean()
                        log_msg = f"{log_msg} | sqnr: {sqnr_mean:.3f}"

                _logger.info(log_msg)

                # Freeze the param quantizers for LET optimized modules
                freeze_let_optimized_param_quantizers(qt_block)

            # fold_let_params
            for module in quant_sim.model.modules():
                if isinstance(module, SUPPORTED_QUANTIZED_MODULES):
                    for key, quantizer in module.param_quantizers.items():
                        if isinstance(quantizer, OMQ_QUANTIZERS):
                            quantizer.fold_let_params(module, key)

            # pylint: disable=protected-access
            # pylint: disable=unnecessary-comprehension
            cls._dump_meta_data(quant_sim.model, output_path)

            # QDQ on models to fold quantizations into weight params.
            quant_sim.model.to(device)
            # pylint:disable = protected-access
            quant_sim._apply_qdq_to_model_parameters(quant_sim.model)

    # pylint: disable=too-many-arguments
    @classmethod
    def _block_wise_training_step(
        cls,
        fp_input,
        qt_input,
        qt_block,
        qt_let_pair_list,
        optimizer,
        loss_fn,
        device: str,
    ):
        """
        Run block-wise traing on LET parameters. Use fp_block output as ground truth and qt_block output as
        model output.

        :param fp_input: block output from previous block in fp model.
        :param qt_input: block output from previous block in qt model.
        :param qt_block: decoder block in qt model.
        :param qt_let_pair_list: let_pair_list in qt model. Use to get LET training parameters.
        :param optimizer: optimizer used for LET blockwise training
        :param loss_fn: loss_fn used for LET bloackwise training
        """
        optimizer.zero_grad()

        def _process_block_input(_batch_block_input):
            args = change_tensor_and_cache_device_placement(
                deepcopy(_batch_block_input.args), device
            )
            kwargs = change_tensor_and_cache_device_placement(
                deepcopy(_batch_block_input.kwargs), device
            )
            return args, kwargs

        # Get target output (ground truth)
        _args, _kwargs = _process_block_input(fp_input)

        with disable_all_quantizers(qt_block):
            target_outputs = qt_block(*_args, **_kwargs)[0]

        # Get model output (prediction)
        _qt_args, _qt_kwargs = _process_block_input(fp_input)
        qt_output = qt_block(*_qt_args, **_qt_kwargs)[0]

        with torch.no_grad():
            sqnr = (
                torch.tensor(get_sqnr(target_outputs, qt_output))
                if OMNIQUANT_COMPUTE_SQNR
                else None
            )

        loss = loss_fn(qt_output, target_outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.detach().cpu(), sqnr

    @classmethod
    # pylint: disable=protected-access
    def _dump_meta_data(cls, model, output_path):
        """Traverse quantized model, get LET scales, and dump {module_name: scale} dict to output path."""
        meta_data = {}
        for name, module in model.named_modules():
            if isinstance(module, SUPPORTED_QUANTIZED_MODULES):
                quantizer = module.param_quantizers["weight"]
                if isinstance(quantizer, OMQ_QUANTIZERS):
                    if quantizer._cached_prev_scale is not None:
                        meta_data[f"{name}.prev"] = quantizer._cached_prev_scale
                    if quantizer._cached_foll_scale is not None:
                        meta_data[f"{name}.foll"] = quantizer._cached_foll_scale

        save_file(meta_data, output_path / OMNIQUANT_METADATA_SAFETENSOR_NAME)
        logger.info(
            "Aimet omniquant metadata saved at %s",
            (output_path / OMNIQUANT_METADATA_SAFETENSOR_NAME).absolute(),
        )

    @classmethod
    def _get_trainable_params(cls, model):
        enc_params = []
        names = []

        def collect_enc_params(quantizer, tag):
            enc_params = []
            names = []
            if quantizer:
                for param in quantizer.parameters():
                    if param.requires_grad:
                        enc_params.append(param)
                        names.append(name + tag)
            return enc_params, names

        for name, module in model.named_modules():
            if isinstance(module, QuantizationMixin):
                for quantizer in module.input_quantizers:
                    ep, n = collect_enc_params(quantizer, "_input_quantizers")
                    enc_params += ep
                    names += n

                for quantizer in module.output_quantizers:
                    ep, n = collect_enc_params(quantizer, "_output_quantizers")
                    enc_params += ep
                    names += n

                for key, quantizer in module.param_quantizers.items():
                    ep, n = collect_enc_params(quantizer, "_param_quantizers")
                    enc_params += ep
                    names += n

        return enc_params, names


def _get_meta_dict(meta_data):
    """Process meta_data to {module_name: {"foll"/"prev": scale}} dict."""
    meta_data_dict = {}

    def _get_name_suf(key):
        """split module name and foll/prev suffix"""
        split_key = key.split(".")
        return ".".join(split_key[:-1]), split_key[-1]

    for key, scale in meta_data.items():
        let_module_name, prev_or_foll = _get_name_suf(key)
        assert prev_or_foll in ("foll", "prev"), (
            f"Expect metadata suffix = foll or prev, but got {prev_or_foll}"
        )
        if meta_data_dict.get(let_module_name) is not None:
            assert prev_or_foll not in meta_data_dict[let_module_name], (
                f"{prev_or_foll} already exists for let_module_name"
            )
            assert len(meta_data_dict[let_module_name]) == 1
            meta_data_dict[let_module_name][prev_or_foll] = torch.from_numpy(scale)
        else:
            meta_data_dict[let_module_name] = {prev_or_foll: torch.from_numpy(scale)}

    return meta_data_dict


def update_lora_weights(
    peft_model: PeftModel, omniquant_metadata_path: Union[str, Path]
):
    """
    Read omniquant metadata safetensor, and apply LET scale to LoraLinear's lora_A and lora_B.
    Lora_A = Lora_A*foll and Lora_B = Lora_B/prev.

    :param peft_model: Peft Lora model.
    :param omniquant_metadata_path: Path to omniquant metadata generated at the end of apply omniquant.
    """
    meta_data = load_file(omniquant_metadata_path)
    meta_data_dict = _get_meta_dict(meta_data)
    assert isinstance(peft_model, PeftModel), (
        f"Expect peft_model class PeftModel, but got {peft_model.__class__}"
    )
    hf_model = (
        peft_model.base_model.model
    )  # PeftModel -> LoraModel (base_model) -> Transformer model (model)
    with torch.no_grad():
        for _module_name, _let_scale_dict in meta_data_dict.items():
            let_module = hf_model.get_submodule(_module_name)
            num_repeats = 1
            if isinstance(let_module, LoraLinear):
                prev = _let_scale_dict["prev"] if "prev" in _let_scale_dict else None
                foll = _let_scale_dict["foll"] if "foll" in _let_scale_dict else None
                if foll is not None:
                    # Lora_A weight [lora_dim, lin_in_dim]
                    # For some pairs prev_layer out channel != foll_layer in channel.
                    # We will repeat the "scale" num_repeats times to match the dimension in foll_layer
                    # Same approach was followed when doing omniquant optimization on the base model
                    # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
                    # This needs to be taken care of when applying the learnt scales to lora adapter
                    # For such pairs the foll scale shape will not match the lora_A in-channel dimension
                    # We will repeat the foll scale as we did for base model for lora_A as well
                    for module in getattr(let_module, "lora_A").values():
                        if foll.shape != module.in_features:
                            num_repeats = module.in_features // foll.shape[0]
                        foll = torch.repeat_interleave(foll, dim=0, repeats=num_repeats)
                        new_weight = module.weight * (foll.unsqueeze(0))
                        module.weight.copy_(new_weight)

                if prev is not None:
                    # Lora_B weight [lin_out_dim, lora_dim]
                    for module in getattr(let_module, "lora_B").values():
                        new_weight = module.weight / (prev.unsqueeze(-1))
                        module.weight.copy_(new_weight)


def update_base_model_with_omniquant_metadata(
    model: torch.nn.Module, omniquant_metadata_path: Union[str, Path]
):
    """
    Read omniquant metadata safetensor, and apply LET scale to base model with original weights
    :param model: Base model.
    :param omniquant_metadata_path: Path to omniquant metadata generated at the end of apply omniquant.
    """
    meta_data = load_file(omniquant_metadata_path)
    meta_data_dict = _get_meta_dict(meta_data)
    with torch.no_grad():
        for _module_name, _let_scale_dict in meta_data_dict.items():
            let_module = model.get_submodule(_module_name)
            num_repeats = 1
            prev = _let_scale_dict["prev"] if "prev" in _let_scale_dict else None
            foll = _let_scale_dict["foll"] if "foll" in _let_scale_dict else None
            if foll is not None:
                # We will repeat the "scale" num_repeats times to match the dimension in foll_layer
                # Same approach was followed when doing omniquant optimization on the base model
                # Ex pair:  self_attn.v_proj and self_attn.o_prj  for llama in gqa
                if foll.shape != let_module.weight.shape[1]:
                    num_repeats = let_module.weight.shape[1] // foll.shape[0]
                foll = torch.repeat_interleave(foll, dim=0, repeats=num_repeats)
                new_weight = let_module.weight * foll
                let_module.weight.copy_(new_weight)
            if prev is not None:
                if isinstance(let_module, torch.nn.Linear):
                    new_weight = let_module.weight / prev.reshape(-1, 1)
                else:
                    new_weight = let_module.weight / prev
                let_module.weight.copy_(new_weight)


apply_omniquant = Omniquant.apply_omniquant
