# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""DiffusionGenerator and component wrappers against the diffusers pipeline."""

import numpy as np
import pytest
import torch

from GenAILab.qai_hub_lm.diffusion import (
    DiffusionGenerator,
    GenerationConfig,
    StableDiffusion,
    get_diffusion_model_class,
)
from GenAILab.qai_hub_lm.diffusion.base import (
    COMPONENTS,
    TEXT_ENCODER,
    UNET,
    VAE_DECODER,
)
from GenAILab.qai_hub_lm.diffusion.generator import to_uint8

PROMPT = "a photo of an astronaut riding a horse"


def _generator(pipeline, scheduler=None, **overrides):
    components = StableDiffusion.build_components(pipeline, scheduler=scheduler)
    side = StableDiffusion.default_resolution(pipeline)
    config = dict(height=side, width=side, num_inference_steps=4, guidance_scale=7.5)
    config.update(overrides)
    return DiffusionGenerator(components, GenerationConfig(**config))


def _reference(pipeline, generator, prompt, seed, **kwargs):
    """The diffusers pipeline's image, with the same scheduler and settings."""
    config = generator.config
    pipeline.scheduler = generator.components.scheduler
    image = pipeline(
        prompt,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
        **kwargs,
    ).images[0]
    return (image * 255).round().astype(np.uint8)


class TestPipelineParity:
    """The generator must reproduce StableDiffusionPipeline.__call__."""

    @pytest.mark.parametrize(
        "scheduler",
        [
            None,  # the model default (DPM++)
            "DDIMScheduler",  # the checkpoint's own
            "PNDMScheduler",
            "EulerDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",  # stochastic: draws from the generator
            "DPMSolverMultistepScheduler",
        ],
    )
    def test_matches_diffusers(self, tiny_pipeline, scheduler):
        generator = _generator(tiny_pipeline, scheduler)
        ours = generator.generate(PROMPT, seed=3).astype(np.int32)
        ref = _reference(tiny_pipeline, generator, PROMPT, seed=3).astype(np.int32)
        # Batch-1 cond/uncond UNet calls vs. the pipeline's batch-2 call differ
        # only in reduction order: on CPU, at most an LSB on a handful of pixels.
        # (On GPU the drift is larger; see DiffusionGenerator's docstring.)
        assert np.abs(ours - ref).max() <= 1
        assert (ours != ref).mean() < 1e-3

    def test_matches_diffusers_without_guidance(self, tiny_pipeline):
        generator = _generator(tiny_pipeline, guidance_scale=1.0)
        assert not generator.config.do_classifier_free_guidance
        ours = generator.generate(PROMPT, seed=5).astype(np.int32)
        ref = _reference(tiny_pipeline, generator, PROMPT, seed=5).astype(np.int32)
        assert np.abs(ours - ref).max() <= 1

    def test_matches_diffusers_with_negative_prompt(self, tiny_pipeline):
        generator = _generator(tiny_pipeline, negative_prompt="blurry")
        ours = generator.generate(PROMPT, seed=1).astype(np.int32)
        ref = _reference(
            tiny_pipeline, generator, PROMPT, seed=1, negative_prompt="blurry"
        ).astype(np.int32)
        assert np.abs(ours - ref).max() <= 1

    def test_non_square_resolution(self, tiny_pipeline):
        generator = _generator(tiny_pipeline, height=48, width=80)
        ours = generator.generate(PROMPT, seed=0)
        assert ours.shape == (48, 80, 3)
        ref = _reference(tiny_pipeline, generator, PROMPT, seed=0)
        assert np.abs(ours.astype(np.int32) - ref).max() <= 1


class TestGeneratorBehaviour:
    def test_deterministic_per_seed(self, tiny_pipeline):
        generator = _generator(tiny_pipeline)
        first = generator.generate(PROMPT, seed=7)
        np.testing.assert_array_equal(first, generator.generate(PROMPT, seed=7))
        assert not np.array_equal(first, generator.generate(PROMPT, seed=8))

    def test_image_independent_of_other_prompts(self, tiny_pipeline):
        generator = _generator(tiny_pipeline)
        alone = generator.generate(PROMPT, seed=2)
        batch = generator.generate_all(["a red cube", PROMPT], seeds=[1, 2])
        np.testing.assert_array_equal(batch[1], alone)

    def test_unet_is_called_at_batch_one(self, tiny_pipeline):
        generator = _generator(tiny_pipeline, num_inference_steps=3)
        shapes = []
        handle = generator.components.unet.register_forward_pre_hook(
            lambda _, args: shapes.append(tuple(a.shape for a in args))
        )
        try:
            generator.generate(PROMPT, seed=0)
        finally:
            handle.remove()
        # Conditional + unconditional per step.
        assert len(shapes) == 2 * 3
        assert {s[0][0] for s in shapes} == {1}
        assert {s[1] for s in shapes} == {(1,)}

    def test_rejects_resolution_not_divisible_by_vae_scale(self, tiny_pipeline):
        with pytest.raises(ValueError, match="multiples of the VAE scale factor"):
            _generator(tiny_pipeline, height=63)

    def test_generate_all_length_mismatch(self, tiny_pipeline):
        with pytest.raises(ValueError, match="2 prompts but 1 seeds"):
            _generator(tiny_pipeline).generate_all(["a", "b"], seeds=[0])

    def test_unknown_scheduler(self, tiny_pipeline):
        with pytest.raises(ValueError, match="Unknown diffusers scheduler"):
            StableDiffusion.build_components(tiny_pipeline, scheduler="NoSuchScheduler")


class TestWrappers:
    """Each wrapper is a faithful tensor-in/tensor-out view of its module."""

    def test_text_encoder(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        (ids,) = StableDiffusion.get_sample_inputs(TEXT_ENCODER, components, 64, 64)
        ids = torch.randint(0, 99, ids.shape, dtype=ids.dtype)
        with torch.no_grad():
            ours = components.text_encoder(ids)
            ref = tiny_pipeline.text_encoder(ids.long()).last_hidden_state
        torch.testing.assert_close(ours, ref)

    def test_unet(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        latents, timestep, hidden = StableDiffusion.get_sample_inputs(
            UNET, components, 64, 64
        )
        latents, hidden = torch.randn_like(latents), torch.randn_like(hidden)
        timestep = torch.full_like(timestep, 500.0)
        with torch.no_grad():
            ours = components.unet(latents, timestep, hidden)
            ref = tiny_pipeline.unet(
                latents, timestep, encoder_hidden_states=hidden
            ).sample
        torch.testing.assert_close(ours, ref)
        assert ours.shape == latents.shape

    def test_vae_decoder_applies_scaling_factor(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        (latents,) = StableDiffusion.get_sample_inputs(VAE_DECODER, components, 64, 64)
        latents = torch.randn_like(latents)
        scale = tiny_pipeline.vae.config.scaling_factor
        with torch.no_grad():
            ours = components.vae_decoder(latents)
            ref = tiny_pipeline.vae.decode(latents / scale).sample
        torch.testing.assert_close(ours, ref)
        assert ours.shape == (1, 3, 64, 64)

    def test_sample_inputs_match_input_names(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        for name in COMPONENTS:
            inputs = StableDiffusion.get_sample_inputs(name, components, 64, 64)
            assert len(inputs) == len(StableDiffusion.get_input_names(name))
            with torch.no_grad():
                out = components.component(name)(*inputs)
            assert isinstance(out, torch.Tensor)
            assert len(StableDiffusion.get_output_names(name)) == 1

    def test_unknown_component(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        with pytest.raises(KeyError, match="Unknown diffusion component"):
            components.component("backbone")


class TestRegistryAndHelpers:
    def test_registered_by_pipeline_class(self):
        assert get_diffusion_model_class("StableDiffusionPipeline") is StableDiffusion

    def test_unregistered_pipeline(self):
        with pytest.raises(LookupError, match="No diffusion model registered"):
            get_diffusion_model_class("FluxPipeline")

    def test_to_uint8_matches_diffusers_postprocess(self):
        from diffusers.image_processor import VaeImageProcessor

        image = torch.rand(2, 3, 8, 8) * 2.4 - 1.2  # includes out-of-range values
        ref = VaeImageProcessor().postprocess(image, output_type="np")
        np.testing.assert_array_equal(
            to_uint8(image), (ref * 255).round().astype(np.uint8)
        )

    def test_on_device_restores_placement_after_error(self, tiny_pipeline):
        components = StableDiffusion.build_components(tiny_pipeline)
        moved = []
        original_to = torch.nn.Module.to

        def recording_to(module, *args, **kwargs):
            moved.append(args[0] if args else kwargs.get("device"))
            return original_to(module, *args, **kwargs)

        with pytest.raises(RuntimeError, match="boom"):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(torch.nn.Module, "to", recording_to)
                with components.on_device("cpu"):
                    raise RuntimeError("boom")
        # Three moves in, three moves back to where each component was.
        assert len(moved) == 2 * len(COMPONENTS)
        assert all(torch.device(d) == torch.device("cpu") for d in moved)
