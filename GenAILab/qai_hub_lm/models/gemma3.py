# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma3 shared VLM base class"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin

try:
    from transformers.models.gemma3 import modeling_gemma3
except ImportError:
    modeling_gemma3 = None

from .base import VLM
from .generator import Generator, VLM_Generator
from .utils.layer_cache import (
    LayerCacheDescriptor,
    attention_mask_input_names,
    kv_length_dynamic_axes,
)


class Gemma3VisionWrapper(torch.nn.Module):
    """Wraps Gemma3's vision_tower + multi_modal_projector into a single traceable module.

    Inputs:  pixel_values [B, C, H, W]
    Output:  image_embeddings [B, mm_tokens_per_image, text_hidden_size]
    """

    def __init__(self, vision_tower, multi_modal_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
        return self.multi_modal_projector(vision_outputs.last_hidden_state)


class Gemma3_VLM_Generator(VLM_Generator):
    """VLM_Generator subclass for Gemma3.

    Gemma3 uses standard ``pixel_values`` [B, C, H, W] (SigLIP encoder),
    without image_position_ids or image_grid_thw.
    """

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        hidden_size = self.config.text_config.hidden_size

        inputs_embeds = self.embedding(input_ids)
        if not isinstance(
            self.embedding, modeling_gemma3.Gemma3TextScaledWordEmbedding
        ):
            inputs_embeds = inputs_embeds * (hidden_size**0.5)

        image_mask_3d = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

        if pixel_values is not None:
            num_images = pixel_values.shape[0]
            all_embeddings = []
            for i in range(num_images):
                pv_i = pixel_values[i].unsqueeze(0)
                emb_i = self.vision_model(pv_i)
                all_embeddings.append(emb_i)
            image_embeddings = torch.cat(all_embeddings, dim=0).reshape(
                -1, inputs_embeds.shape[-1]
            )
            image_embeddings = image_embeddings.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_embeddings
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        if pixel_values is not None:
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1

        return inputs_embeds, mm_token_type_ids, {}

    def _prefill_visual(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            return
        num_images = pixel_values.shape[0]
        for i in range(num_images):
            yield {"pixel_values": pixel_values[i].unsqueeze(0)}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("mm_token_type_ids", None)
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        return Generator.forward(
            self,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            **{**kwargs, **extra_kwargs},
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if self._quantization_mode:
            yield from self._prefill_component(
                self._quantization_mode,
                input_ids=input_ids,
                pixel_values=pixel_values,
                **kwargs,
            )
            return

        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        yield from Generator.prefill(
            self,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            **{**kwargs, **extra_kwargs},
        )


class Gemma3_VLM(VLM):
    """Shared Gemma3 VLM base (framework-agnostic)."""

    DEFAULT_MODEL_ID = "google/gemma-3-4b-it"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        llm_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )
        if small_model:
            llm_config.text_config.num_hidden_layers = 2
            if (
                hasattr(llm_config.text_config, "layer_types")
                and llm_config.text_config.layer_types is not None
            ):
                llm_config.text_config.layer_types = llm_config.text_config.layer_types[
                    :2
                ]
        return modeling_gemma3.Gemma3ForConditionalGeneration.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> ProcessorMixin:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        return AutoProcessor.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        *args,
        **kwargs,
    ):
        hidden_size = model.config.hidden_size

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=torch.float32
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        prepared = Gemma3_VLM.get_generator_cls().prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            inputs_embeds=dummy_inputs_embeds,
            layer_cache_descriptors=layer_cache_descriptors,
        )
        return tuple(prepared.values())

    @classmethod
    def get_sample_vision_inputs(
        cls, config, image_size=None, dtype: torch.dtype = torch.float32
    ):
        """Dummy inputs for Gemma3 vision QuantSim.

        SigLIP takes standard [B, C, H, W] pixel values.
        """
        vcfg = config.vision_config
        img_size = vcfg.image_size
        dummy_pixel_values = torch.zeros((1, 3, img_size, img_size), dtype=dtype)
        return (dummy_pixel_values,)

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> tuple[str, ...]:
        from .utils.layer_cache import cache_state_names

        return tuple(
            ["inputs_embeds"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
        )

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {
            "inputs_embeds": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
        }
        for name in attention_mask_input_names(layer_cache_descriptors):
            axes[name] = {2: "sequence_length"}
        axes |= kv_length_dynamic_axes(layer_cache_descriptors)
        return axes

    @classmethod
    def instantiate_position_processor(cls):
        return None

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values",)

    @staticmethod
    def get_visual_output_names(**kwargs) -> tuple[str, ...]:
        return ("image_embeddings",)

    @classmethod
    def build_vision_wrapper(cls, model):
        return Gemma3VisionWrapper(
            model.model.vision_tower, model.model.multi_modal_projector
        )

    @staticmethod
    def get_generator_cls():
        return Gemma3_VLM_Generator
