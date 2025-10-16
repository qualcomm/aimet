# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-Torch"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
from copy import deepcopy
import functools
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
from aimet_torch.experimental.spinquant.spinquant_optimizer import apply_spinquant
from aimet_torch.quantization.affine import QuantizeDequantize
from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch.v2.utils import remove_all_quantizers, patch_attr
from aimet_torch import QuantizationSimModel
from aimet_torch.utils import place_model
from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams
from aimet_torch.experimental.omniquant import apply_omniquant

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator


def _compute_encodings(
    quantsim: QuantizationSimModel,
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
):
    """Internal helper function to compute encodings on quantsim model"""
    assert quantsim.model == generator.model

    if num_iterations is None:
        num_iterations = len(dataloader)

    def callback(_):
        sliced_dataloader = itertools.islice(dataloader, num_iterations)
        for batch in tqdm(sliced_dataloader, total=num_iterations, desc="Calibrating"):
            generator(input_ids=batch["input_ids"].to(device=generator.device))

    quantsim.compute_encodings(callback)


class QuantizationTechnique(ABC):
    """Generic GenAI quantization technique"""

    @staticmethod
    @abstractmethod
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        """Apply quantization technique"""


@YAMLConfigParser.register_recipe
class RemoveQuantization(QuantizationTechnique):
    """Remove all quantization nodes from quantsim model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        remove_all_quantizers(quantsim.model)


@YAMLConfigParser.register_recipe
class LoadEncodings(QuantizationTechnique):
    """Load encodings from file"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **recipe_kwargs,
    ):
        if "path" not in recipe_kwargs:
            raise ValueError(
                "Encodings path must be provided for LoadEncodings recipe as 'path'."
            )

        quantsim.load_encodings(recipe_kwargs["path"], partial=False)


@YAMLConfigParser.register_recipe
class PCQ(QuantizationTechnique):
    """Apply vanilla PCQ to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class LPBQ(QuantizationTechnique):
    """Apply LPBQ to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        arg = lambda module: (
            isinstance(module, (QuantizedConv2d, QuantizedLinear))
            and module.param_quantizers["weight"]
            and module.param_quantizers["weight"].bitwidth == 4
        )

        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=64,
            block_grouping=-1,
        )

        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        def callback(model, inputs):
            with patch_attr(generator, "model", model):
                generator(**inputs)

        sliced_dataloader = itertools.islice(dataloader, 20)
        apply_seq_mse(
            quantsim, sliced_dataloader, num_candidates=20, forward_fn=callback
        )
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class LPBQ_SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        arg = lambda module: (
            isinstance(module, (QuantizedConv2d, QuantizedLinear))
            and module.param_quantizers["weight"]
            and module.param_quantizers["weight"].bitwidth == 4
        )

        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=64,
            block_grouping=-1,
        )

        SeqMSE.apply(quantsim, generator, dataloader)


@YAMLConfigParser.register_recipe
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
    ):
        def collate_fn(sample):
            return {
                "input_ids": sample[0]["input_ids"],
                "attention_mask": sample[0]["attention_mask"],
            }

        apply_adascale(
            quantsim,
            DataLoader(
                Subset(dataloader, range(num_batches)),
                batch_size=1,
                collate_fn=collate_fn,
                shuffle=False,
            ),
            lambda model, inputs: generator(
                inputs["input_ids"].to(device=generator.device), use_cache=False
            ),
            num_iterations,
        )

        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class OmniQuant(QuantizationTechnique):
    """Apply OmniQuant to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_batches: int = 40,
        num_iterations: int = 800,
    ):
        class LimitedBatchDataLoader:
            """Internal helper class to reduce number of accessible batches in Dataloader"""

            def __init__(self, dataloader, num_batches):
                self.dataloader = dataloader
                self.num_batches = num_batches
                self.current_batch = 0

            def __iter__(self):
                # pylint: disable=attribute-defined-outside-init
                self.iterator = iter(self.dataloader)
                self.current_batch = 0
                return self

            def __next__(self):
                if self.current_batch < self.num_batches:
                    self.current_batch += 1
                    return next(self.iterator)
                raise StopIteration

            def __len__(self):
                return min(len(self.dataloader), self.num_batches)

        apply_omniquant(
            quant_sim=quantsim,
            dataloader=LimitedBatchDataLoader(dataloader, num_batches=num_batches),
            forward_fn=lambda model, input: generator(**input),
            num_iterations=num_iterations,
        )

        _compute_encodings(quantsim, generator, dataloader, num_iterations=40)


@YAMLConfigParser.register_recipe
class SpinQuant(QuantizationTechnique):
    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        # Set linear layers to 8 bit to more easily observe effects of SpinQuant
        for quant_module in quantsim.qmodules():
            if isinstance(quant_module, torch.nn.Linear):
                quant_module.param_quantizers["weight"] = QuantizeDequantize(
                    quant_module.param_quantizers["weight"].shape,
                    bitwidth=8,
                    symmetric=True,
                )
                quant_module.param_quantizers["weight"].to(quant_module.weight.device)

        # Untie embed_tokens and lm_head if needed
        if (
            quantsim.model.model.model.embed_tokens.weight
            is quantsim.model.model.lm_head.weight
        ):
            old_weight = quantsim.model.model.lm_head.weight
            new_weight = torch.nn.Parameter(
                old_weight.data.clone().detach().to(old_weight.device),
                requires_grad=True,
            )
            quantsim.model.model.lm_head.weight = new_weight

        apply_spinquant(quantsim.model)
        _compute_encodings(quantsim, generator, dataloader, num_iterations=40)
