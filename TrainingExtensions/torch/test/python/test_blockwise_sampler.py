# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import dataclass
from contextlib import ExitStack

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch.blockwise_sampler import BlockwiseSampler
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.utils import StopForwardException, disable_all_quantizers
from .v2.models_ import test_models


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=1
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class ModelWithBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(10))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RandomDataset(Dataset):
    def __init__(self, shape, length):
        self.shape = shape
        self.length = length
        self.generator = torch.Generator().manual_seed(0)
        self.samples = tuple(
            torch.randn(self.shape, generator=self.generator)
            for _ in range(self.length)
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class InputHolder:
    args: tuple
    kwargs: dict


class StopForwardExceptionWithInput(StopForwardException):
    def __init__(self, captured_input):
        self.captured_input = captured_input


def hook_fn(module, args, kwargs):
    raise StopForwardExceptionWithInput(InputHolder(args, kwargs))


def _get_device_type(module: torch.nn.Module):
    for param in module.parameters():
        return param.device.type
    for buffer in module.buffers():
        return buffer.device.type
    return None


@pytest.mark.parametrize("cache_activations_on_disk", [True, False])
@pytest.mark.parametrize("keep_unused_blocks_on_cpu", [True, False])
def test_blockwise_sampler(cache_activations_on_disk, keep_unused_blocks_on_cpu):
    model = ModelWithBlocks()
    sim = QuantizationSimModel(model, dummy_input=torch.randn(3, 3, 3))
    dataset = RandomDataset((3, 3, 3), 10)

    def forward_pass_callback(quantized_model):
        for sample in DataLoader(dataset, batch_size=1, shuffle=False):
            _ = sim.model(sample)

    sim.compute_encodings(forward_pass_callback)

    sampler = BlockwiseSampler(
        sim=sim,
        blocks=sim.model.blocks,
        dataloader=DataLoader(dataset, batch_size=1, shuffle=False),
        cache_activations_on_disk=cache_activations_on_disk,
        keep_unused_blocks_on_cpu=keep_unused_blocks_on_cpu,
    )

    for block, fp_block_inputs, qt_block_inputs in sampler.sample():
        # put a hook on the block, grab fp_inputs, grab qt_inputs without using sampler. Check if they are the same
        hook = block.register_forward_pre_hook(hook_fn, with_kwargs=True)

        qt_block_inputs_without_caching = []
        fp_block_inputs_without_caching = []
        for sample in DataLoader(dataset, batch_size=1, shuffle=False):
            try:
                sim.model(sample)
            except StopForwardExceptionWithInput as e:
                qt_block_inputs_without_caching.append(e.captured_input.args)

            with disable_all_quantizers(sim.model):
                try:
                    sim.model(sample)
                except StopForwardExceptionWithInput as e:
                    fp_block_inputs_without_caching.append(e.captured_input.args)

        hook.remove()

        def _verify_equal_tuples_of_tensors(tuple1, tuple2):
            for tensor1, tensor2 in zip(tuple1, tuple2):
                assert torch.equal(tensor1, tensor2)

        with ExitStack() as stack:
            for fp_block_input in fp_block_inputs:
                stack.enter_context(fp_block_input.load())
            for qt_block_input in qt_block_inputs:
                stack.enter_context(qt_block_input.load())

            fp_block_inputs_args = tuple(
                fp_block_input.args for fp_block_input in fp_block_inputs
            )
            qt_block_inputs_args = tuple(
                qt_block_input.args for qt_block_input in qt_block_inputs
            )

            assert block in sim.model.blocks
            for cached_tensors, uncached_tensors in zip(
                fp_block_inputs_args, fp_block_inputs_without_caching
            ):
                _verify_equal_tuples_of_tensors(uncached_tensors, cached_tensors)
            for cached_tensors, uncached_tensors in zip(
                qt_block_inputs_args, qt_block_inputs_without_caching
            ):
                _verify_equal_tuples_of_tensors(uncached_tensors, cached_tensors)


@pytest.mark.cuda
@pytest.mark.parametrize("num_blocks, disable_cache_until", [(6, 3), (5, 0)])
def test_blockwise_sampler_with_disabled_caching(num_blocks, disable_cache_until):
    device = torch.device("cuda")
    model = test_models.ModelWithLinearBlocks(num_blocks, disable_cache_until)
    sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 32, 64))
    sim.model.to(device)
    dataset = RandomDataset((1, 32, 64), 10)

    def forward_pass_callback(quantized_model):
        for sample in DataLoader(dataset, batch_size=1, shuffle=False):
            _ = quantized_model(sample.cuda())

    sim.compute_encodings(forward_pass_callback)

    sampler = BlockwiseSampler(
        sim=sim,
        blocks=sim.model.blocks,
        dataloader=DataLoader(dataset, batch_size=1, shuffle=False),
        cache_activations_on_disk=True,
        keep_unused_blocks_on_cpu=True,
        disable_caching_until_block=disable_cache_until,
    )

    for block_idx, (block, *_) in enumerate(sampler.sample()):
        # Current block should always be on device
        assert _get_device_type(block) == "cuda"

        if block_idx >= disable_cache_until:
            for parameter in sim.model.parameters():
                if parameter in set(block.parameters()):
                    assert parameter.device.type == "cuda"
                else:
                    assert parameter.device.type == "cpu"
        else:
            # All later blocks should be on CPU
            for block in sim.model.blocks[block_idx + 1 :]:
                assert _get_device_type(block) == "cpu"
            # All earlier blocks should be on cuda
            for block in sim.model.blocks[: block_idx + 1]:
                assert _get_device_type(block) == "cuda"
            # All non-block modules should be on cuda
            for layer in sim.model.postprocessing:
                assert _get_device_type(layer) == "cuda"
