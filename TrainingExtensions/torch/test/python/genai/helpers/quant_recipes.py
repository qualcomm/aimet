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
"""Quantization recipes for GenAI models"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
from copy import deepcopy
import functools
import torch
from torch.utils.data import DataLoader

from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch import QuantizationSimModel
from aimet_torch.utils import place_model
from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams
from aimet_torch.experimental.omniquant import apply_omniquant


def _compute_encodings(
    quantsim: QuantizationSimModel, dataloader: DataLoader, num_iterations: int = None
):
    """Internal helper function to compute encodings on quantsim model"""
    if num_iterations is None:
        num_iterations = len(dataloader)

    def callback(model: torch.nn.Module):
        sliced_dataloader = itertools.islice(dataloader, num_iterations)
        for batch in tqdm(sliced_dataloader, total=num_iterations, desc="Calibrating"):
            model(input_ids=batch["input_ids"].to(device=model.device))

    quantsim.compute_encodings(callback)


class QuantizationTechnique(ABC):
    """Generic GenAI quantization technique"""

    @staticmethod
    @abstractmethod
    def apply(quantsim: QuantizationSimModel, dataloader: DataLoader):
        """Apply quantization technique"""


class RemoveQuantization(QuantizationTechnique):
    """Remove all quantization nodes from quantsim model"""

    @staticmethod
    def apply(quantsim: QuantizationSimModel, dataloader: DataLoader):
        remove_all_quantizers(quantsim.model)


class PCQ(QuantizationTechnique):
    """Apply vanilla PCQ to model"""

    @staticmethod
    @torch.no_grad()
    def apply(quantsim: QuantizationSimModel, dataloader: DataLoader):
        _compute_encodings(quantsim, dataloader, num_iterations=20)


class LPBQ(QuantizationTechnique):
    """Apply LPBQ to model"""

    @staticmethod
    @torch.no_grad()
    def apply(quantsim: QuantizationSimModel, dataloader: DataLoader):
        arg = lambda module: (
            isinstance(module, (QuantizedConv2d, QuantizedLinear))
            and module.param_quantizers["weight"]
            and module.param_quantizers["weight"].bitwidth == 4
        )
        BLOCK_QUANT_SIZE = 64
        BITWIDTH = 4
        DECOMPRESSED_BITWIDTH = 8

        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=BITWIDTH,
            symmetric=True,
            decompressed_bw=DECOMPRESSED_BITWIDTH,
            block_size=BLOCK_QUANT_SIZE,
            block_grouping=-1,
        )

        _compute_encodings(quantsim, dataloader, num_iterations=20)


class SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @staticmethod
    @torch.no_grad()
    def apply(quantsim: QuantizationSimModel, dataloader: DataLoader):
        def copy_model_with_shared_weights(source_model):
            target_model = deepcopy(source_model)
            for name, source_parameter in source_model.named_parameters():
                pre, _, post = name.rpartition(".")
                pre_obj = (
                    functools.reduce(getattr, [target_model] + pre.split("."))
                    if pre
                    else target_model
                )
                setattr(pre_obj, post, source_parameter)
            QuantizationSimModel._remove_quantization_wrappers(
                target_model, list(target_model.modules())
            )
            return target_model

        with (
            place_model(quantsim.model, torch.device("cpu")),
            remove_all_quantizers(quantsim.model),
        ):
            weight_shared_fp_model = copy_model_with_shared_weights(quantsim.model)

        params = SeqMseParams(
            num_batches=20,
            inp_symmetry="symqt",
            num_candidates=20,
            loss_fn="mse",
            forward_fn=lambda model, inputs: model(**inputs),
        )
        with place_model(weight_shared_fp_model, quantsim.model.device):
            apply_seq_mse(weight_shared_fp_model, quantsim, dataloader, params)

        _compute_encodings(quantsim, dataloader, num_iterations=20)


class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        dataloader: DataLoader,
        num_batches: int = 20,
        num_iterations: int = 1500,
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

        apply_adascale(
            quantsim,
            LimitedBatchDataLoader(dataloader, num_batches=num_batches),
            lambda model, inputs: model(
                inputs["input_ids"].to(device=model.device), use_cache=False
            ),
            num_iterations,
        )

        _compute_encodings(quantsim, dataloader, num_iterations=20)


class OmniQuant(QuantizationTechnique):
    """Apply OmniQuant to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
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
            forward_fn=lambda model, input: model.forward(**input),
            num_iterations=num_iterations,
        )

        _compute_encodings(quantsim, dataloader, num_iterations=40)
