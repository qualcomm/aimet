# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures for the text-to-image diffusion tests."""

import pytest

pytest.importorskip("diffusers")

#: A ~1.4M-parameter SD1.x pipeline (64x64 images) -- real diffusers modules,
#: random weights. Fast enough to run whole generations on CPU. Not the older
#: ``tiny-stable-diffusion-pipe``: its model_index.json names Flax classes,
#: which diffusers >= 0.38 no longer ships.
TINY_SD = "hf-internal-testing/tiny-sd-pipe"


@pytest.fixture(scope="session")
def tiny_pipeline():
    from GenAILab.qai_hub_lm.diffusion import StableDiffusion

    return StableDiffusion.instantiate_float_model(TINY_SD)
