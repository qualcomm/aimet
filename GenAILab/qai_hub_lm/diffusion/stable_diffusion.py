# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable Diffusion 1.x"""

from __future__ import annotations

import torch

from .base import (
    TEXT_ENCODER,
    UNET,
    VAE_DECODER,
    DiffusionComponents,
    DiffusionModel,
    register_diffusion_model,
)
from .wrappers import TextEncoderWrapper, UNetWrapper, VAEDecoderWrapper


@register_diffusion_model("StableDiffusionPipeline")
class StableDiffusion(DiffusionModel):
    """Stable Diffusion 1.x: CLIP text encoder, UNet2DConditionModel, AutoencoderKL."""

    _INPUT_NAMES = {
        TEXT_ENCODER: ("input_ids",),
        UNET: ("latents", "timestep", "encoder_hidden_states"),
        VAE_DECODER: ("latents",),
    }
    _OUTPUT_NAMES = {
        TEXT_ENCODER: ("encoder_hidden_states",),
        UNET: ("noise_pred",),
        VAE_DECODER: ("image",),
    }

    #: Used when a config names no scheduler. SD1.5's checkpoint default, PNDM,
    #: gives partly garbled images at the 10 steps GenAILab defaults to (as the
    #: diffusers pipeline does too); DPM++ is clean at 10.
    DEFAULT_SCHEDULER = "DPMSolverMultistepScheduler"

    @classmethod
    def instantiate_float_model(cls, model_id: str, dtype: torch.dtype = torch.float32):
        # Imported lazily so the LLM path never needs diffusers installed.
        from diffusers import StableDiffusionPipeline

        # The safety checker is a post-hoc image classifier, not part of
        # generation; it is dropped so it neither costs memory nor blacks out
        # images the metrics should score.
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    @classmethod
    def build_components(
        cls, pipeline, scheduler: str | None = None
    ) -> DiffusionComponents:
        for module in (pipeline.text_encoder, pipeline.unet, pipeline.vae):
            module.eval()
        return DiffusionComponents(
            text_encoder=TextEncoderWrapper(pipeline.text_encoder),
            unet=UNetWrapper(pipeline.unet),
            vae_decoder=VAEDecoderWrapper(pipeline.vae),
            tokenizer=pipeline.tokenizer,
            scheduler=_resolve_scheduler(
                pipeline.scheduler, scheduler or cls.DEFAULT_SCHEDULER
            ),
            latent_channels=pipeline.unet.config.in_channels,
            vae_scale_factor=pipeline.vae_scale_factor,
        )

    @classmethod
    def get_sample_inputs(
        cls, component: str, components: DiffusionComponents, height: int, width: int
    ) -> tuple[torch.Tensor, ...]:
        device, dtype = components.device, components.dtype
        tokenizer = components.tokenizer
        latent_shape = (
            1,
            components.latent_channels,
            height // components.vae_scale_factor,
            width // components.vae_scale_factor,
        )
        if component == TEXT_ENCODER:
            return (
                torch.zeros(
                    (1, tokenizer.model_max_length), dtype=torch.int32, device=device
                ),
            )
        if component == UNET:
            hidden_size = components.text_encoder.text_encoder.config.hidden_size
            return (
                torch.zeros(latent_shape, dtype=dtype, device=device),
                # float32 whatever the model dtype; see DiffusionGenerator.denoise.
                torch.zeros((1,), dtype=torch.float32, device=device),
                torch.zeros(
                    (1, tokenizer.model_max_length, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            )
        if component == VAE_DECODER:
            return (torch.zeros(latent_shape, dtype=dtype, device=device),)
        raise KeyError(f"Unknown diffusion component {component!r}.")

    @classmethod
    def get_input_names(cls, component: str) -> tuple[str, ...]:
        return cls._INPUT_NAMES[component]

    @classmethod
    def get_output_names(cls, component: str) -> tuple[str, ...]:
        return cls._OUTPUT_NAMES[component]


def _resolve_scheduler(default, name: str):
    """The diffusers scheduler ``name``, built from the checkpoint's scheduler config."""
    if name == type(default).__name__:
        return default
    import diffusers

    scheduler_cls = getattr(diffusers, name, None)
    if scheduler_cls is None or not hasattr(scheduler_cls, "from_config"):
        raise ValueError(f"Unknown diffusers scheduler {name!r}.")
    return scheduler_cls.from_config(default.config)
