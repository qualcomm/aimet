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
import itertools
from dataclasses import dataclass
from typing import List, Union, Tuple, Generator
import torch
from torch.utils.data import DataLoader

from aimet_torch import QuantizationSimModel, utils

class BlockwiseSampler:
    """
    Class providing blockwise sampling utilities. Specifically, BlockWise sampler allows users to specify a list of
    sequential blocks in the module, and the sampler will yield each block, along with the FP and QT inputs to that
    block as a part of the sample() loop.
    BlockwiseSampler caches each block input and resumes computation from that point when the user returns control to
    the sampling loop. This way, excess computational costs for large models is avoided, and the loop is able to make
    use of any user adjustments to quantization parameters for a particular block made in the sampling loop.
    NOTE: users CAN NOT modify model weights during the sample() loop
    """
    def __init__(self,
                 sim: QuantizationSimModel,
                 blocks: List[torch.nn.Module],
                 dataloader: DataLoader,
                 num_samples: int):
        self.sim = sim
        self.blocks = blocks
        self.dataloader = dataloader
        self.num_samples = num_samples

    def run_inference(self, sample) -> Generator[torch.Tensor, None, None]:
        """
        Helper function to run inference on the model using the given sample, pausing and yielding the results after
        each block.
        """

        @dataclass
        class InputHolder:
            """Dataclass to hold input args and kwargs to a pytorch module."""
            args: tuple
            kwargs: dict

        class StopForwardExceptionWithInput(utils.StopForwardException):
            """Exception raised in order to stop forward execution through the model. Holds module input data."""
            def __init__(self, captured_input):
                self.captured_input = captured_input

        def hook_fn(module, args, kwargs):
            raise StopForwardExceptionWithInput(InputHolder(args, kwargs))

        with torch.no_grad():
            try:
                hook = self.blocks[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
                self.sim.model(sample)
            except StopForwardExceptionWithInput as e:
                # pylint: disable=used-before-assignment
                hook.remove()
                next_block_input = e.captured_input
                yield next_block_input.args

            for block in self.blocks:
                next_block_input.args = block(*next_block_input.args, **next_block_input.kwargs)
                if not isinstance(next_block_input.args, tuple):
                    next_block_input.args = (next_block_input.args,)
                yield next_block_input.args


    def sample(self) -> Generator[Tuple[Union[torch.nn.Module, torch.nn.ModuleList], torch.Tensor, torch.Tensor], None, None]:
        """
        Main generator function for blockwise sampler. Each loop of this generator yields a tuple of
        (block, [list of FP inputs to block], [list of QT inputs to block]) based on the list of blocks provided during
        initialization.
        """
        fp_inferences = []
        qt_inferences = []

        for sample in itertools.islice(self.dataloader, self.num_samples):
            fp_inferences.append(self.run_inference(sample))
            qt_inferences.append(self.run_inference(sample))

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
