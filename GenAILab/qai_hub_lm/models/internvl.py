# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""InternVL3.5 shared VLM base class"""

from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.models.internvl.processing_internvl import InternVLProcessor
from transformers.models.internvl.video_processing_internvl import (
    InternVLVideoProcessor,
)

from .base import VLM
from .generator import Generator, VLM_Generator
from .utils.layer_cache import (
    LayerCacheDescriptor,
    _resolve_text_config,
)


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class InternVLVisionWrapper(torch.nn.Module):
    """Wraps InternViT + pixel_shuffle + MLP projector into a single traceable module.

    Input:  pixel_values [B, 3, H, W]  (one or more tiles, typically 448x448)
    Output: image_embeddings [B, num_tokens, llm_hidden_size]
            where num_tokens = (image_size // patch_size)^2 * downsample_ratio^2
    """

    def __init__(self, vision_model, mlp1, downsample_ratio=0.5, select_layer=-1):
        super().__init__()
        self.vision_model = vision_model
        self.mlp1 = mlp1
        self.downsample_ratio = downsample_ratio
        self.select_layer = select_layer

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


class InternVL_VLM_Generator(VLM_Generator):
    """VLM_Generator subclass for InternVL.

    InternVL uses per-tile vision processing and fuses via indexed assignment
    at <IMG_CONTEXT> token positions. No custom position_ids needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_context_token_id = self.config.image_token_id

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        inputs_embeds = self.embedding(input_ids)
        extra_kwargs = {}

        image_mask_3d = (
            (input_ids == self.img_context_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

        if pixel_values is not None:
            num_tiles = pixel_values.shape[0]
            all_embeddings = []
            for i in range(num_tiles):
                pv_i = pixel_values[i].unsqueeze(0)
                emb_i = self.vision_model(pv_i)
                all_embeddings.append(emb_i)
            image_embeddings = torch.cat(all_embeddings, dim=0)
            image_embeddings = image_embeddings.reshape(-1, image_embeddings.shape[-1])
            image_embeddings = image_embeddings.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_embeddings
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        if pixel_values is not None:
            mm_token_type_ids[input_ids == self.img_context_token_id] = 1

        return inputs_embeds, mm_token_type_ids, extra_kwargs

    def _prefill_visual(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            return
        num_tiles = pixel_values.shape[0]
        for i in range(num_tiles):
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


class InternVL_VLM(VLM):
    """Shared InternVL VLM base (framework-agnostic)."""

    DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-8B"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.llm_config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = False
        if small_model:
            config.llm_config.num_hidden_layers = 2

        # InternVL's custom code references all_tied_weights_keys which may not
        # exist in all transformers versions.
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}

        model = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True
        )

        processor = cls.instantiate_tokenizer(model_id)
        tokenizer = getattr(processor, "tokenizer", processor)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
        model.config.image_token_id = img_context_token_id

        # Expose LLM attributes on the top-level config so that transformers'
        # generation internals (DynamicCache creation) can find them.
        model.config.num_hidden_layers = config.llm_config.num_hidden_layers

        return model

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> InternVLProcessor:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        tokenizer.start_image_token = "<img>"
        tokenizer.end_image_token = "</img>"
        tokenizer.context_image_token = IMG_CONTEXT_TOKEN
        tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids("<img>")
        tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids("</img>")
        tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )
        tokenizer.video_token = "<video>"

        chat_template = tokenizer.chat_template.replace(
            "'<image>", f"'{IMG_CONTEXT_TOKEN}"
        )

        image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        video_processor = InternVLVideoProcessor()
        return InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )

    @classmethod
    def instantiate_position_processor(cls):
        return None

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
        config = model.config if hasattr(model, "config") else None
        text_config = _resolve_text_config(config) if config else None
        hidden_size = text_config.hidden_size if text_config else 4096

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=torch.float32
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        prepared = InternVL_VLM.get_generator_cls().prepare_inputs(
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
        """Dummy inputs for InternVL vision QuantSim (single tile)."""
        model_image_size = (
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        if image_size is not None:
            h, w = image_size
            if (h, w) != (model_image_size, model_image_size):
                raise ValueError(
                    f"Requested image_size={image_size} conflicts with model's "
                    f"expected image size ({model_image_size}x{model_image_size}). "
                    f"Ensure the values match."
                )
        else:
            h = w = model_image_size
        dummy_pixel_values = torch.zeros((1, 3, h, w), dtype=dtype)
        return (dummy_pixel_values,)

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> dict[str, dict[int, str]]:
        from .utils.layer_cache import (
            attention_mask_input_names,
            kv_length_dynamic_axes,
        )

        axes: dict[str, dict[int, str]] = {
            "inputs_embeds": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
        }
        for name in attention_mask_input_names(layer_cache_descriptors):
            axes[name] = {2: "sequence_length"}
        axes |= kv_length_dynamic_axes(layer_cache_descriptors)
        return axes

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values",)

    @staticmethod
    def get_visual_output_names(**kwargs) -> tuple[str, ...]:
        return ("image_embeddings",)

    @classmethod
    def get_language_model(cls, model):
        return model.language_model

    @classmethod
    def get_lm_head(cls, model):
        return None

    @classmethod
    def build_vision_wrapper(cls, model):
        return InternVLVisionWrapper(
            model.vision_model,
            model.mlp1,
            downsample_ratio=model.downsample_ratio,
            select_layer=model.select_layer,
        )

    @staticmethod
    def get_generator_cls() -> type[VLM_Generator]:
        return InternVL_VLM_Generator
