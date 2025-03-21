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

"""Blockwise sampling utilty"""
import contextlib
from dataclasses import dataclass
from typing import List, Union, Tuple, Generator
import torch

from aimet_torch import QuantizationSimModel, utils

class BlockwiseSampler:
    """Class providing blockwise sampling utilities"""
    def __init__(self,
                 sim: QuantizationSimModel,
                 blocks: List[Union[torch.nn.Module, torch.nn.ModuleList]],
                 dataloader,
                 num_samples: int):
        self.sim = sim
        self.blocks = blocks
        self.samples = [next(dataloader) for _ in range(num_samples)]

    def run_inference(self, sample) -> Generator[torch.Tensor, None, None]:
        @dataclass
        class InputHolder:
            args: tuple
            kwargs: dict

        class StopForwardExceptionWithInput(utils.StopForwardException):
            def __init__(self, captured_input):
                self.captured_input = captured_input

        def hook_fn(module, args, kwargs):
            raise StopForwardExceptionWithInput(InputHolder(args, kwargs))

        with torch.no_grad():
            try:
                hook = self.blocks[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
                self.sim.model(sample)
            except StopForwardExceptionWithInput as e:
                hook.remove()
                next_block_input = e.captured_input
                yield next_block_input

            for block in self.blocks:
                next_block_input.args = block(*next_block_input.args, **next_block_input.kwargs)
                yield next_block_input


    def sample(self) -> Generator[Tuple[Union[torch.nn.Module, torch.nn.ModuleList], torch.Tensor, torch.Tensor], None, None]:
        fp_inferences = [self.run_inference(sample) for sample in self.samples]
        qt_inferences = [self.run_inference(sample) for sample in self.samples]

        blocks = iter(self.blocks)

        while True:
            try:
                block = next(blocks)

                # Quantizers must be ENABLED when calculating quantized block inputs
                qt_block_inputs = [next(block_input) for block_input in qt_inferences]

                # Quantizers must be DISABLED when calculating FP block inputs
                with utils.disable_all_quantizers(self.sim.model):
                    fp_block_inputs = [next(block_input) for block_input in fp_inferences]

                yield block, fp_block_inputs, qt_block_inputs
            except StopIteration:
                break
