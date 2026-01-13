# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-Torch"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
from aimet_torch.experimental.spinquant.spinquant_optimizer import apply_spinquant
from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch import QuantizationSimModel
from aimet_torch.utils import change_tensor_device_placement
from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.experimental.omniquant import apply_omniquant

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator


def _prefill_inputs(
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
    device: torch.device = None,
):
    inputs = []
    if num_iterations is not None:
        dataloader = itertools.islice(dataloader, num_iterations)

    with remove_all_quantizers(generator.model), torch.no_grad():
        for sample in tqdm(
            dataloader,
            total=num_iterations if num_iterations else len(dataloader),
            desc="Pre-filling calibration data",
        ):
            inputs.extend(
                change_tensor_device_placement(
                    list(
                        generator.prefill(
                            sample["input_ids"].to(device=generator.device),
                            sample["attention_mask"].to(device=generator.device),
                        )
                    ),
                    device=device if device else generator.device,
                )
            )

    return inputs


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
        inputs = _prefill_inputs(generator, dataloader, 20, torch.device("cpu"))
        apply_seq_mse(quantsim, inputs, num_candidates=20)
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
        inputs = _prefill_inputs(
            generator, dataloader, num_batches, torch.device("cpu")
        )
        apply_adascale(
            quantsim,
            inputs,
            num_iterations=num_iterations,
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

        apply_spinquant(model=quantsim.model.model)
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class SpinQuant_AdaScale(QuantizationTechnique):
    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
    ):
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

        apply_spinquant(model=quantsim.model.model)
        AdaScale.apply(quantsim, generator, dataloader, num_batches, num_iterations)
