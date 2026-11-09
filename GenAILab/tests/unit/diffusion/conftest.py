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


#: A 10-row stand-in for PartiPrompts, with uneven categories.
FAKE_PARTI = [
    {"prompt": f"{category} prompt {i}", "category": category}
    for category, count in [("Animals", 6), ("Food", 3), ("People", 1)]
    for i in range(count)
]


@pytest.fixture
def fake_parti(monkeypatch):
    """Serve PartiPrompts from :data:`FAKE_PARTI` instead of the Hub."""
    from GenAILab.bench.diffusion import metrics as diffusion_metrics

    monkeypatch.setattr(
        diffusion_metrics,
        "load_parti_prompts",
        lambda num_prompts=64, sample_seed=0: diffusion_metrics.stratified_sample(
            FAKE_PARTI, num_prompts, sample_seed
        ),
    )
