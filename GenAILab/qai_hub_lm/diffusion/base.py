# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Base class and registry for text-to-image diffusion models."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

#: Component names in pipeline order. The order fixes reporting, and later
#: per-component quantization and export, so it is part of the contract.
TEXT_ENCODER = "text_encoder"
UNET = "unet"
VAE_DECODER = "vae_decoder"
COMPONENTS: tuple[str, ...] = (TEXT_ENCODER, UNET, VAE_DECODER)


@dataclass
class DiffusionComponents:
    """The traceable modules and host-side helpers of one diffusion pipeline.

    The diffusion analogue of :class:`~GenAILab.qai_hub_lm.models.base.SimCollection`:
    ``text_encoder``, ``unet`` and ``vae_decoder`` are the units a QuantSim
    wraps or an ONNX session replaces, and are swappable without touching the
    generator. ``tokenizer`` and ``scheduler`` run on the host and are never
    quantized.
    """

    text_encoder: torch.nn.Module
    unet: torch.nn.Module
    vae_decoder: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    #: Template scheduler. The generator clones it per image, since diffusers
    #: schedulers keep per-run state.
    scheduler: object
    latent_channels: int
    #: Ratio of image size to latent size.
    vae_scale_factor: int

    def component(self, name: str) -> torch.nn.Module:
        if name not in COMPONENTS:
            raise KeyError(
                f"Unknown diffusion component {name!r}. Known: {COMPONENTS}."
            )
        return getattr(self, name)

    def items(self):
        """Yield ``(name, module)`` for every component, in pipeline order."""
        for name in COMPONENTS:
            yield name, getattr(self, name)

    @contextlib.contextmanager
    def on_device(self, device: torch.device | str):
        """Temporarily place every component on ``device``, restoring afterwards."""
        original = {name: _module_device(module) for name, module in self.items()}
        try:
            for _, module in self.items():
                module.to(device)
            yield
        finally:
            for name, module in self.items():
                module.to(original[name])

    @property
    def device(self) -> torch.device:
        return _module_device(self.unet)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


_DIFFUSION_MODELS: dict[str, type[DiffusionModel]] = {}


def register_diffusion_model(pipeline_class_name: str):
    """Register a model class for a diffusers pipeline ``_class_name``.

    The key is the ``_class_name`` in the checkpoint's ``model_index.json``
    (e.g. ``"StableDiffusionPipeline"``) -- the diffusers counterpart of an HF
    ``model_type``.
    """

    def decorator(model_cls):
        existing = _DIFFUSION_MODELS.get(pipeline_class_name)
        if existing is not None and existing is not model_cls:
            raise RuntimeError(
                f"Pipeline class '{pipeline_class_name}' is already registered by "
                f"{existing.__name__}. Cannot register {model_cls.__name__}."
            )
        _DIFFUSION_MODELS[pipeline_class_name] = model_cls
        return model_cls

    return decorator


def get_diffusion_model_class(pipeline_class_name: str) -> type[DiffusionModel]:
    try:
        return _DIFFUSION_MODELS[pipeline_class_name]
    except KeyError:
        raise LookupError(
            f"No diffusion model registered for pipeline class "
            f"'{pipeline_class_name}'. Registered: {sorted(_DIFFUSION_MODELS)}."
        ) from None


class DiffusionModel(ABC):
    """Interface for a text-to-image diffusion model family.

    Mirrors :class:`~GenAILab.qai_hub_lm.models.base.LLM`: classmethods that
    load the float model, split it into traceable components, and describe each
    component's sample inputs and I/O names. The sample inputs and names are the
    shape contract later phases rely on for QuantSim construction and ONNX
    export, so they must agree with the wrappers' ``forward`` signatures.
    """

    @classmethod
    @abstractmethod
    def instantiate_float_model(cls, model_id: str, dtype: torch.dtype = torch.float32):
        """Load the float diffusers pipeline."""

    @classmethod
    @abstractmethod
    def build_components(
        cls, pipeline, scheduler: str | None = None
    ) -> DiffusionComponents:
        """Split a loaded pipeline into :class:`DiffusionComponents`.

        ``scheduler`` names a diffusers scheduler class to use instead of the
        checkpoint's default; it is built from the default's config.
        """

    @classmethod
    def default_resolution(cls, pipeline) -> int:
        """Image side length the checkpoint was trained at."""
        return pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    @classmethod
    @abstractmethod
    def get_sample_inputs(
        cls, component: str, components: DiffusionComponents, height: int, width: int
    ) -> tuple[torch.Tensor, ...]:
        """Sample inputs for one component, in its ``forward`` argument order."""

    @classmethod
    @abstractmethod
    def get_input_names(cls, component: str) -> tuple[str, ...]:
        """Input names for one component, in its ``forward`` argument order."""

    @classmethod
    @abstractmethod
    def get_output_names(cls, component: str) -> tuple[str, ...]:
        """Output names for one component."""
