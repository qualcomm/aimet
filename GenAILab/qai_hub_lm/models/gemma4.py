# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma4 shared VLM base class"""

from __future__ import annotations

import os
import warnings

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    ProcessorMixin,
)
from transformers import masking_utils
from huggingface_hub import hf_hub_download

try:
    from transformers.models.gemma4 import modeling_gemma4
except ImportError:
    modeling_gemma4 = None

from GenAILab.qai_hub_lm.models.base import VLM
from GenAILab.qai_hub_lm.models.components import AUDIO, VISUAL
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.models.utils.layer_cache import LayerCacheDescriptor


class SoftcappedLMHead(torch.nn.Module):
    """LM head wrapper that applies Gemma4's final_logit_softcapping after the linear.

    When softcap is None, acts as a transparent pass-through.
    """

    def __init__(self, lm_head: torch.nn.Module, softcap: float | None):
        super().__init__()
        self.lm_head = lm_head
        self.softcap = softcap

    @property
    def linear(self) -> torch.nn.Module:
        return self.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.softcap is not None:
            logits = logits / self.softcap
            logits = torch.tanh(logits)
            logits = logits * self.softcap
        return logits


class Gemma4VisionWrapper(torch.nn.Module):
    """Wraps Gemma4's vision_tower + embed_vision projector into a single traceable module.

    Inputs:  pixel_values       [B, num_patches, 3*patch_size^2]
             image_position_ids [B, num_patches, 2]
    Output:  image_embeddings   [total_image_tokens, text_hidden_size]
    """

    def __init__(self, vision_tower, embed_vision):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return self.embed_vision(inputs_embeds=vision_out.last_hidden_state)


def _index_based_bidirectional_mask(
    config, inputs_embeds, attention_mask, and_mask_function=None, **kwargs
):
    """``create_bidirectional_mask`` for the audio tower, built without ``vmap``.

    Passing ``and_mask_function`` makes transformers build the mask under
    ``vmap``, which strict ``torch.export`` cannot trace. Every mask function
    here is a plain index comparison, so evaluate them on broadcast indices
    instead; the result is identical.
    """
    batch_size, length = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    indices = masking_utils._non_vmap_expansion_sdpa(
        torch.arange(batch_size, device=device),
        torch.arange(1, device=device),
        torch.arange(length, device=device),
        torch.arange(length, device=device),
    )
    mask = masking_utils.bidirectional_mask_function(*indices)
    if and_mask_function is not None:
        mask = mask & and_mask_function(*indices)
    if attention_mask is not None:
        mask = mask & attention_mask.bool()[:, None, None, :]
    mask = mask.expand(batch_size, 1, length, length)
    if config._attn_implementation == "eager":
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        mask = torch.where(mask, 0.0, min_dtype).to(inputs_embeds.dtype)
    return mask


class Gemma4AudioWrapper(torch.nn.Module):
    """audio_tower + embed_audio projector as one traceable module.

    In:  input_features [B, frames, mel] (time dim 1, unlike Qwen3-ASR), mask
         [B, frames].
    Out: audio_embeddings [B, soft_tokens, text_hidden], audio_mask.

    Shapes stay static because the tower returns a validity mask instead of
    compacting frames with ``nonzero()``; the padded soft tokens are stripped
    outside, during fusion.
    """

    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The vision tower also builds its mask with create_bidirectional_mask,
        # so swap it only for the audio tower's call
        stock = modeling_gemma4.create_bidirectional_mask
        modeling_gemma4.create_bidirectional_mask = _index_based_bidirectional_mask
        try:
            audio_out = self.audio_tower(input_features, input_features_mask)
        finally:
            modeling_gemma4.create_bidirectional_mask = stock
        embeddings = self.embed_audio(inputs_embeds=audio_out.last_hidden_state)
        return embeddings, audio_out.attention_mask


class Gemma4_VLM_Generator(VLM_Generator):
    """VLM_Generator subclass for Gemma4.

    Gemma4 uses ``pixel_values`` + ``image_position_ids`` (one entry per image),
    rather than the Qwen-style ``pixel_values`` + ``image_grid_thw``.
    """

    def __init__(self, *args, embed_tokens_per_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed_tokens_per_layer = embed_tokens_per_layer

    @staticmethod
    def slice_inputs_for_inference(
        inputs, attention_mask, sequence_length, position_ids=None, **kwargs
    ):
        per_layer_inputs = kwargs.pop("per_layer_inputs", None)
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            start = max(0, idx - sequence_length)
            input_slice = inputs[:, start:idx]
            mask_slice = attention_mask[:, start:idx]
            pos_slice = (
                position_ids[..., start:idx] if position_ids is not None else None
            )
            kw_slice = dict(kwargs)
            slice_len = input_slice.shape[1]
            pad_len = sequence_length - slice_len
            if per_layer_inputs is not None:
                ple_slice = per_layer_inputs[:, start:idx]
                if pad_len > 0:
                    pad_shape = (ple_slice.shape[0], pad_len) + ple_slice.shape[2:]
                    ple_slice = torch.cat(
                        [
                            torch.zeros(
                                pad_shape,
                                dtype=ple_slice.dtype,
                                device=ple_slice.device,
                            ),
                            ple_slice,
                        ],
                        dim=1,
                    )
                kw_slice["per_layer_inputs"] = ple_slice
            yield input_slice, mask_slice, pos_slice, kw_slice

    def _compute_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute PLE token-identity embeddings from input_ids.

        Uses the stored ``embed_tokens_per_layer`` embedding table directly.
        Applies sqrt(ple_dim) scaling unconditionally since the embedding may be
        a plain nn.Embedding (e.g. after cache deserialization) that lacks the
        built-in Gemma4TextScaledWordEmbedding scale factor.
        """
        text_config = self.config.text_config
        ple_dim = text_config.hidden_size_per_layer_input
        num_layers = text_config.num_hidden_layers

        pad_token_id = text_config.pad_token_id
        ple_input_ids = input_ids.clone()
        ple_input_ids[input_ids == self.config.image_token_id] = pad_token_id

        with torch.no_grad():
            per_layer_inputs = self._embed_tokens_per_layer(ple_input_ids)
        per_layer_inputs = per_layer_inputs.reshape(
            *ple_input_ids.shape, num_layers, ple_dim
        )
        if not isinstance(
            self._embed_tokens_per_layer, modeling_gemma4.Gemma4TextScaledWordEmbedding
        ):
            per_layer_inputs = per_layer_inputs * (ple_dim**0.5)
        return per_layer_inputs

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size

        inputs_embeds = self.embedding(input_ids)
        if not isinstance(
            self.embedding, modeling_gemma4.Gemma4TextScaledWordEmbedding
        ):
            inputs_embeds = inputs_embeds * (hidden_size**0.5)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids)

        image_mask_3d = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

        if pixel_values is not None:
            all_embeddings = []
            num_images = pixel_values.shape[0]
            for i in range(num_images):
                pv_i = pixel_values[i].unsqueeze(0)
                pid_i = image_position_ids[i].unsqueeze(0)
                emb_i = self.vision_model(pv_i, pid_i)
                all_embeddings.append(emb_i)
            image_embeddings = torch.cat(all_embeddings, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_embeddings
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        if pixel_values is not None:
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1

        return (
            inputs_embeds,
            mm_token_type_ids,
            {"per_layer_inputs": per_layer_inputs},
        )

    def pad_audio_item(self, item: dict, audio_frames: int) -> dict:
        """Pad ``input_features`` to ``audio_frames`` mel frames.

        Time is dim 1 here, so ``F.pad``'s ``(0, 0, 0, pad)`` leaves mel alone.
        The mask stays a real validity mask: :meth:`fuse_audio` uses it to drop
        the padded soft tokens, so no placeholder resizing is needed.
        """
        features = item["input_features"]
        mask = item["input_features_mask"]
        num_frames = features.shape[1]
        if num_frames > audio_frames:
            warnings.warn(
                f"Utterance produced {num_frames} mel frames after waveform "
                f"truncation, still exceeding audio_frames={audio_frames}; "
                f"truncating the mel features directly. This should be rare."
            )
            features = features[:, :audio_frames]
            mask = mask[..., :audio_frames]
        pad = audio_frames - features.shape[1]
        if pad:
            features = torch.nn.functional.pad(features, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        item["input_features"] = features
        item["input_features_mask"] = mask
        return item

    def fuse_audio(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter Gemma4 audio soft tokens into the sequence.

        The encoder returns a padded block plus a validity mask rather than a
        compacted sequence, so the padded soft tokens are dropped here -- outside
        the quantized graph -- to match the placeholder count.
        """
        if input_features is None:
            return inputs_embeds

        audio_embeddings, audio_valid_mask = self.audio_model(
            input_features, input_features_mask
        )
        # Keep only real soft tokens; the encoder emits a fixed-width block.
        audio_embeddings = audio_embeddings[audio_valid_mask.bool()]
        audio_embeddings = audio_embeddings.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        audio_mask_3d = (
            (input_ids == self.config.audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        return inputs_embeds.masked_scatter(audio_mask_3d, audio_embeddings)

    def fuse_multimodal(
        self,
        input_ids: torch.Tensor | None = None,
        **modality_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compose Gemma4's image fusion with audio fusion.

        Overridden rather than inherited because Gemma4's
        :meth:`fuse_text_image_video` takes ``image_position_ids`` (it has no
        ``image_grid_thw``), which the base implementation does not forward, and
        because its ``extra_kwargs`` carry ``per_layer_inputs``.
        """
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=modality_kwargs.get("pixel_values"),
            image_position_ids=modality_kwargs.get("image_position_ids"),
        )
        if self.has_component(AUDIO.name):
            inputs_embeds = self.fuse_audio(
                inputs_embeds,
                input_ids,
                input_features=modality_kwargs.get("input_features"),
                input_features_mask=modality_kwargs.get("input_features_mask"),
            )
        return inputs_embeds, mm_token_type_ids, extra_kwargs

    def _prefill_visual(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            return
        num_images = pixel_values.shape[0]
        for i in range(num_images):
            yield {
                "pixel_values": pixel_values[i].unsqueeze(0),
                "image_position_ids": image_position_ids[i].unsqueeze(0),
            }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("mm_token_type_ids", None)
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
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
        image_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if self._quantization_mode:
            yield from self._prefill_component(
                self._quantization_mode,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                input_features=input_features,
                input_features_mask=input_features_mask,
                **kwargs,
            )
            return

        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
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


class Gemma4_VLM(VLM):
    """Shared Gemma4 VLM base (framework-agnostic).

    Gemma4 is the first model here with *two* input encoders. Audio is optional
    per checkpoint (``audio_config`` may be ``None``), so ``supports_component``
    gates it on the config rather than on this class declaration.
    """

    DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"

    COMPONENTS = (VISUAL.name, AUDIO.name)

    #: Default padded mel-frame count for audio sample inputs (see the "audio
    #: component" note in base.py for the waveform-samples /
    #: audio_frames(=mel frames) / audio-tokens distinction). The feature
    #: extractor runs at a 10 ms hop (100 frames/s) and the encoder's two
    #: stride-2 subsampling convs reduce time by 4, so 800 frames = 8 s of audio
    #: = 200 audio soft tokens.
    DEFAULT_AUDIO_FRAMES = 800

    @classmethod
    def use_dynamo_export_for(cls, component: str | None = None) -> bool:
        # Measured: the audio tower fails torch.jit.trace with
        # "RuntimeError: unordered_map::at" (masking_utils builds its
        # bidirectional/sliding mask through vmap) but exports cleanly via
        # torch.export. The backbone and vision tower still trace fine, so only
        # audio switches paths.
        if component == AUDIO.name:
            return True
        return super().use_dynamo_export_for(component)

    @classmethod
    def supports_component(cls, component: str, config: PretrainedConfig) -> bool:
        if component == AUDIO.name:
            return getattr(config, "audio_config", None) is not None
        if component == VISUAL.name:
            return getattr(config, "vision_config", None) is not None
        return super().supports_component(component, config)

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
        return modeling_gemma4.Gemma4ForConditionalGeneration.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> ProcessorMixin:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        # model_id may be a local export dir (ONNX phase: model_id points at the
        # torch-exported artifacts) rather than an HF repo id. hf_hub_download
        # rejects a local path as an invalid repo id, so read the chat template
        # from disk in that case (tokenizer.save_pretrained wrote it there).
        local_template = os.path.join(model_id, "chat_template.jinja")
        if os.path.isdir(model_id) and os.path.exists(local_template):
            chat_template_path = local_template
        else:
            chat_template_path = hf_hub_download(model_id, "chat_template.jinja")
        with open(chat_template_path) as f:
            return AutoProcessor.from_pretrained(
                model_id, use_fast=True, trust_remote_code=True, chat_template=f.read()
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
        num_layers = model.config.num_hidden_layers
        ple_dim = model.config.hidden_size_per_layer_input

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=model.dtype
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)
        dummy_per_layer_inputs = torch.zeros(
            (1, sequence_length, num_layers, ple_dim), dtype=model.dtype
        )

        prepared = Gemma4_VLM.get_generator_cls().prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            inputs_embeds=dummy_inputs_embeds,
            layer_cache_descriptors=layer_cache_descriptors,
            per_layer_inputs=dummy_per_layer_inputs,
        )
        return tuple(prepared.values())

    @classmethod
    def get_sample_vision_inputs(
        cls, config, image_size=None, dtype: torch.dtype = torch.float32
    ):
        """Dummy inputs for Gemma4 vision QuantSim.

        Gemma4's image processor always pads to 2520 patches
        (image_seq_length=280 * pooling_kernel_size^2=9).
        """
        vcfg = config.vision_config
        patch_dim = 3 * vcfg.patch_size**2
        num_patches = 2520
        dummy_pixel_values = torch.zeros((1, num_patches, patch_dim), dtype=dtype)
        dummy_position_ids = torch.zeros((1, num_patches, 2), dtype=torch.int64)
        return (dummy_pixel_values, dummy_position_ids)

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> tuple[str, ...]:
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            attention_mask_input_names,
            cache_state_names,
        )

        return tuple(
            ["inputs_embeds"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
            + ["per_layer_inputs"]
        )

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> dict[str, dict[int, str]]:
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            AttentionType,
            attention_mask_input_names,
        )

        axes: dict[str, dict[int, str]] = {
            "inputs_embeds": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
            "per_layer_inputs": {1: "sequence_length"},
        }
        for name in attention_mask_input_names(layer_cache_descriptors):
            axes[name] = {2: "sequence_length"}
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                continue
            axes[f"past_key_{i}_in"] = {2: "kv_cache_length"}
            axes[f"past_value_{i}_in"] = {2: "kv_cache_length"}
        return axes

    @classmethod
    def instantiate_position_processor(cls):
        return None

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values", "image_position_ids")

    @staticmethod
    def get_visual_output_names(**kwargs) -> tuple[str, ...]:
        return ("image_embeddings",)

    @classmethod
    def get_lm_head(cls, model):
        softcap = model.config.text_config.final_logit_softcapping
        return SoftcappedLMHead(model.lm_head, softcap)

    @classmethod
    def build_vision_wrapper(cls, model):
        return Gemma4VisionWrapper(model.model.vision_tower, model.model.embed_vision)

    # ---- audio component ----------------------------------------------------
    @classmethod
    def build_audio_wrapper(cls, model):
        return Gemma4AudioWrapper(model.model.audio_tower, model.model.embed_audio)

    @classmethod
    def get_sample_audio_inputs(
        cls,
        config: PretrainedConfig,
        audio_frames: int | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Sample ``(input_features, input_features_mask)`` for the audio tower.

        Time is dim 1, mel dim 2 -- the opposite of Qwen3-ASR. The all-ones mask
        keeps the emitted soft-token count at its maximum, which is the shape the
        export should carry.
        """
        audio_frames = cls.validate_audio_frames(
            config, audio_frames or cls.DEFAULT_AUDIO_FRAMES
        )
        num_mel_bins = cls.get_num_mel_bins(config)
        return (
            torch.zeros((1, audio_frames, num_mel_bins), dtype=torch.float32),
            torch.ones((1, audio_frames), dtype=torch.bool),
        )

    @staticmethod
    def get_num_mel_bins(config: PretrainedConfig) -> int:
        """Mel-bin count for the audio front end.

        No explicit config field, but ``input_proj_linear``'s size only matches
        the tensor reaching it when ``subsampling_conv_channels[0] ==
        num_mel_bins``. Raise rather than default: a wrong mel width mismatches
        the projection silently.
        """
        channels = getattr(config.audio_config, "subsampling_conv_channels", None)
        if not channels:
            raise ValueError(
                f"{type(config.audio_config).__name__} has no "
                "subsampling_conv_channels, so the audio tower's mel-bin count "
                "cannot be derived. Add an explicit source before exporting."
            )
        return channels[0]

    @staticmethod
    def audio_soft_tokens(config: PretrainedConfig, audio_frames: int) -> int:
        """Soft-token count for a fully-valid mel span.

        The subsampling stack is one ``Conv2d(kernel 3, stride 2, padding 1)`` per
        entry in ``subsampling_conv_channels``, and the encoder tracks validity by
        slicing the mask ``[:, ::2]`` once per layer -- so the token count follows
        that same halving, not the conv output-length formula.
        """
        channels = getattr(config.audio_config, "subsampling_conv_channels", None)
        if not channels:
            raise ValueError(
                f"{type(config.audio_config).__name__} has no "
                "subsampling_conv_channels, so the audio token count cannot be "
                "derived."
            )
        tokens = audio_frames
        for _ in channels:
            tokens = (tokens + 1) // 2
        return tokens

    @staticmethod
    def get_audio_input_names() -> tuple[str, ...]:
        return ("input_features", "input_features_mask")

    @staticmethod
    def get_audio_output_names(**kwargs) -> tuple[str, ...]:
        # Two outputs: the soft tokens and the validity mask used to strip
        # padding during fusion.
        return ("audio_embeddings", "audio_mask")

    @classmethod
    def get_extras(cls, model):
        return {
            "embed_tokens_per_layer": model.model.language_model.embed_tokens_per_layer,
        }

    @staticmethod
    def get_generator_cls():
        return Gemma4_VLM_Generator
