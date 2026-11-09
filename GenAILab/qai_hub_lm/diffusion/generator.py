# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Text-to-image generation over swappable diffusion components."""

from __future__ import annotations

import contextlib
import gc
import inspect
from dataclasses import dataclass

import numpy as np
import torch

from .base import DiffusionComponents


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters shared by every image an experiment generates."""

    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str = ""
    #: Base seed. Callers that generate a prompt set seed prompt ``i`` with
    #: ``seed + i``, so adding prompts never changes earlier images.
    seed: int = 0

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1.0


class DiffusionGenerator:
    """Runs classifier-free-guided denoising over a :class:`DiffusionComponents`.

    Reproduces ``StableDiffusionPipeline.__call__`` numerically, but owns the
    loop instead of delegating to diffusers, so that each component is called
    only through its tensor-in/tensor-out wrapper. That is what lets a QuantSim
    or an ONNX session stand in for any component, and what a calibration hook
    observes.

    Unlike the diffusers pipeline, the conditional and unconditional UNet passes
    run as two batch-1 calls rather than one batch-2 call. That keeps a single
    static UNet shape, which is how the UNet is deployed on device. The two
    differ only in GPU kernel reduction order, but guidance amplifies that over
    the steps: on SD1.5 at 512px, 10 steps, the images are visually identical
    while pixels differ by a mean of ~0.4 (fp32) / ~1.8 (fp16) uint8 levels.
    (On CPU, as in the unit tests, they agree to within one level.)

    Images are seeded individually, so an image depends only on its prompt and
    seed, never on what else is in the batch.
    """

    def __init__(self, components: DiffusionComponents, config: GenerationConfig):
        scale = components.vae_scale_factor
        if config.height % scale or config.width % scale:
            raise ValueError(
                f"height and width must be multiples of the VAE scale factor "
                f"({scale}), got {config.height}x{config.width}."
            )
        self.components = components
        self.config = config

    @contextlib.contextmanager
    def on_device(self, device: torch.device | str):
        with self.components.on_device(device):
            yield

    @contextlib.contextmanager
    def evicted(self):
        """Free the accelerator memory the components hold, for the block.

        For callers that need the device for something else (CLIPScore's CLIP
        model) and will not generate until the block exits. Mirrors
        ``Generator.evicted`` on the LLM side.
        """
        with self.on_device("cpu"):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield

    @property
    def device(self) -> torch.device:
        return self.components.device

    @property
    def dtype(self) -> torch.dtype:
        return self.components.dtype

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> torch.Tensor:
        """``(1, model_max_length)`` int32 token ids, padded and truncated."""
        tokenizer = self.components.tokenizer
        ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return ids.to(device=self.device, dtype=torch.int32)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.components.text_encoder(self.tokenize(prompt))

    def initial_latents(self, generator: torch.Generator) -> torch.Tensor:
        """Unscaled Gaussian latents drawn from a CPU ``generator``.

        Drawn on the CPU and moved, as diffusers does for a CPU generator, so
        the same seed gives the same image on every device.
        """
        shape = (
            1,
            self.components.latent_channels,
            self.config.height // self.components.vae_scale_factor,
            self.config.width // self.components.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, dtype=self.dtype)
        return latents.to(self.device)

    def denoise(
        self,
        latents: torch.Tensor,
        cond_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Run the scheduler loop from initial noise to denoised latents.

        ``generator`` is the one the initial latents were drawn from; stochastic
        schedulers continue drawing from it, as in diffusers.
        """
        config = self.config
        scheduler = _fresh_scheduler(self.components.scheduler)
        scheduler.set_timesteps(config.num_inference_steps, device=self.device)
        step_kwargs = _step_kwargs(scheduler, generator)

        latents = latents * scheduler.init_noise_sigma
        unet = self.components.unet
        for t in scheduler.timesteps:
            model_input = scheduler.scale_model_input(latents, t)
            # Timesteps go in as float32 whatever the model dtype: some
            # schedulers use fractional timesteps that fp16 would round.
            timestep = t.reshape(1).to(device=self.device, dtype=torch.float32)
            noise_pred = unet(model_input, timestep, cond_embeds)
            if uncond_embeds is not None:
                noise_uncond = unet(model_input, timestep, uncond_embeds)
                noise_pred = noise_uncond + config.guidance_scale * (
                    noise_pred - noise_uncond
                )
            latents = scheduler.step(
                noise_pred, t, latents, **step_kwargs, return_dict=False
            )[0]
        return latents

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latents to a ``(H, W, 3)`` uint8 image."""
        image = self.components.vae_decoder(latents)
        return to_uint8(image)[0]

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, prompt: str, seed: int) -> np.ndarray:
        """Generate one ``(H, W, 3)`` uint8 image for ``prompt``."""
        cond_embeds = self.encode_prompt(prompt)
        uncond_embeds = (
            self.encode_prompt(self.config.negative_prompt)
            if self.config.do_classifier_free_guidance
            else None
        )
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = self.initial_latents(generator)
        latents = self.denoise(latents, cond_embeds, uncond_embeds, generator)
        return self.decode(latents)

    def generate_all(self, prompts: list[str], seeds: list[int]) -> list[np.ndarray]:
        if len(prompts) != len(seeds):
            raise ValueError(f"Got {len(prompts)} prompts but {len(seeds)} seeds.")
        return [self.generate(p, s) for p, s in zip(prompts, seeds)]


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """``(B, 3, H, W)`` in ``[-1, 1]`` -> ``(B, H, W, 3)`` uint8.

    Same rounding as diffusers' ``VaeImageProcessor.postprocess`` + ``numpy_to_pil``.
    """
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).float().cpu().numpy()
    return (image * 255).round().astype(np.uint8)


def _fresh_scheduler(template):
    """A stateless copy of ``template``: diffusers schedulers carry per-run state."""
    return type(template).from_config(template.config)


def _step_kwargs(scheduler, generator: torch.Generator) -> dict:
    """Extra ``scheduler.step`` kwargs, as ``prepare_extra_step_kwargs`` builds them.

    ``eta`` is left at its default of 0 (deterministic DDIM).
    """
    params = inspect.signature(scheduler.step).parameters
    kwargs = {}
    if "eta" in params:
        kwargs["eta"] = 0.0
    if "generator" in params:
        kwargs["generator"] = generator
    return kwargs
