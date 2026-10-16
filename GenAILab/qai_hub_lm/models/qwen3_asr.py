# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3-ASR: audio -> text.

An audio encoder whose embeddings are scattered at ``audio_token_id`` positions,
exactly as image embeddings are; the text output contract is unchanged.

Use the ``-hf`` checkpoints: the others publish an Omni-style ``thinker_config``
that ``AutoConfig`` silently reads as all-defaults.

``input_features`` is ``(B, num_mel_bins=128, padded_frames)`` -- mel dim 1, time
dim 2. ``padded_frames`` must be a multiple of ``n_window * 2`` (=100); the
encoder raises otherwise. The conv stack yields 13 audio tokens per 100 frames.

The sample inputs use an all-ones mask because the encoder packs valid frames via
``nonzero()`` and builds ``cu_seqlens`` in a Python loop -- data-dependent, so
untraceable. A fully-valid mask makes every derived length constant, which is why
the exported encoder is fixed-length.
"""

from __future__ import annotations

import warnings

import torch
from transformers import AutoProcessor, PretrainedConfig, PreTrainedModel

from .base import VLM
from .components import AUDIO
from .generator import VLM_Generator

#: Default padded mel-frame count for sample inputs (see the "audio component"
#: note in base.py for the waveform-samples / audio_frames(=mel frames) /
#: audio-tokens distinction). Equal to the encoder's own inference window
#: (``audio_config.n_window_infer`` = 800), which is already a multiple of
#: ``n_window * 2``; 800 frames = 8 s of audio = 104 audio tokens.
DEFAULT_AUDIO_FRAMES = 800


def _audio_config(config: PretrainedConfig):
    """The audio-encoder sub-config, tolerating dict or object form."""
    audio_config = config.audio_config
    if isinstance(audio_config, dict):  # pragma: no cover - defensive
        raise TypeError(
            "config.audio_config is a dict; load the config via AutoConfig so it "
            "is parsed into a Qwen3ASREncoderConfig."
        )
    return audio_config


def validate_audio_frames(config: PretrainedConfig, audio_frames: int) -> int:
    """Check that a padded mel-frame count is admissible, and return it.

    The encoder chunks the time axis into ``n_window * 2`` blocks and raises if
    the padded length is not a whole number of them, so failing here gives a
    caller a message that names the knob instead of one from deep inside
    transformers.
    """
    audio_config = _audio_config(config)
    chunk_len = audio_config.n_window * 2
    if audio_frames <= 0 or audio_frames % chunk_len != 0:
        raise ValueError(
            f"audio_frames must be a positive multiple of n_window*2 "
            f"({chunk_len}) for {type(config).__name__}, got {audio_frames}. "
            f"Nearest valid values: "
            f"{max(chunk_len, (audio_frames // chunk_len) * chunk_len)} or "
            f"{(audio_frames // chunk_len + 1) * chunk_len}."
        )
    return audio_frames


def audio_tokens_per_frames(config: PretrainedConfig, audio_frames: int) -> int:
    """Number of audio placeholder tokens a fully-valid mel span produces.

    Mirrors ``Qwen3ASREncoder._post_cnn_length`` for a *full* chunk: three
    (k=3, s=2, p=1) convolutions take ``n_window*2`` frames down to
    ``((( n-1)//2+1 -1)//2+1 -1)//2+1``. With an all-ones mask every chunk is
    full, so the total is that per-chunk count times the number of chunks.
    """
    audio_config = _audio_config(config)
    chunk_len = audio_config.n_window * 2
    validate_audio_frames(config, audio_frames)
    per_chunk = chunk_len
    for _ in range(3):
        per_chunk = (per_chunk - 1) // 2 + 1
    return per_chunk * (audio_frames // chunk_len)


class Qwen3ASRAudioWrapper(torch.nn.Module):
    """Traceable audio encoder: audio tower + multi-modal projector.

    Returns embeddings already at the audio-token count and text hidden width, so
    the generator scatters them without reshaping. Mel extraction stays in the HF
    processor, outside this module.
    """

    def __init__(
        self, audio_tower: torch.nn.Module, multi_modal_projector: torch.nn.Module
    ):
        super().__init__()
        self.audio_tower = audio_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_output = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
        )
        # Unwrap BaseModelOutputWithPooling; indexing keeps this traceable.
        last_hidden_state = (
            audio_output[0]
            if isinstance(audio_output, tuple)
            else audio_output.last_hidden_state
        )
        return self.multi_modal_projector(last_hidden_state)


def _resize_audio_placeholder(
    input_ids, attention_mask, audio_token_id, target_len: int
):
    """Resize the contiguous run of audio placeholder tokens in ``input_ids``
    to ``target_len``. See ``Qwen3ASRGenerator.pad_audio_item`` for why.
    """
    if audio_token_id is None:
        return input_ids, attention_mask

    ids = input_ids[0].tolist()
    if audio_token_id not in ids:
        return input_ids, attention_mask
    start = ids.index(audio_token_id)
    end = len(ids) - ids[::-1].index(audio_token_id)
    if end - start == target_len:
        return input_ids, attention_mask

    new_ids = ids[:start] + [audio_token_id] * target_len + ids[end:]
    new_input_ids = torch.tensor([new_ids], dtype=input_ids.dtype)
    new_attention_mask = torch.ones_like(new_input_ids)
    return new_input_ids, new_attention_mask


class Qwen3ASRGenerator(VLM_Generator):
    """Generator carrying Qwen3-ASR's fixed-length audio-encoder contract.

    The encoder was traced with an all-ones mask, so it treats the padded region
    as valid audio and always emits a token count fixed by ``audio_frames`` alone.
    The processor's placeholder run, sized for the real duration, must be resized
    to match or generation misaligns.
    """

    def pad_audio_item(self, item: dict, audio_frames: int) -> dict:
        # The generator already pads/truncates the raw waveform with real
        # silence to fit before the processor call (see
        # VLM_Generator.fit_audio_waveform), so num_frames here should be
        # within a few frames of audio_frames -- any leftover pad below is
        # genuine framing-boundary slop, not thousands of frames of
        # synthetic zero, so zero-filling it is harmless. Since the encoder
        # treats the mask as always-fully-valid regardless of real content,
        # the fixed audio_frames alone determines the (fixed) token count.
        audio_frames = validate_audio_frames(self.config, audio_frames)
        features = item["input_features"]
        mask = item["input_features_mask"]
        num_frames = features.shape[-1]
        if num_frames > audio_frames:
            warnings.warn(
                f"Utterance produced {num_frames} mel frames after waveform "
                f"truncation, still exceeding audio_frames={audio_frames}; "
                f"truncating the mel features directly. This should be rare."
            )
            features = features[..., :audio_frames]
        pad = audio_frames - features.shape[-1]
        if pad:
            features = torch.nn.functional.pad(features, (0, pad))
        item["input_features"] = features
        item["input_features_mask"] = torch.ones(
            mask.shape[:-1] + (audio_frames,), dtype=mask.dtype
        )

        audio_token_id = getattr(self.tokenizer, "audio_token_id", None)
        target_len = audio_tokens_per_frames(self.config, audio_frames)
        item["input_ids"], item["attention_mask"] = _resize_audio_placeholder(
            item["input_ids"], item["attention_mask"], audio_token_id, target_len
        )
        return item


class Qwen3ASR_LM(VLM):
    """Model-structure class for Qwen3-ASR (audio input encoder + text decoder)."""

    COMPONENTS = (AUDIO.name,)

    @staticmethod
    def get_generator_cls() -> type[Qwen3ASRGenerator]:
        return Qwen3ASRGenerator

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        """Load the float model.

        Overridden because Qwen3-ASR is a ``ForConditionalGeneration`` model, not
        a plain ``AutoModelForCausalLM``. ``small_model`` shrinks BOTH towers so
        smoke tests stay cheap -- the audio encoder is 24 layers and dominates
        the parameter count at the 0.6B/1.7B sizes.
        """
        from transformers import AutoConfig
        from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration

        config = AutoConfig.from_pretrained(model_id, attn_implementation="eager")
        if small_model:
            config.text_config.num_hidden_layers = 2
            if getattr(config.text_config, "layer_types", None) is not None:
                config.text_config.layer_types = config.text_config.layer_types[:2]
            config.audio_config.encoder_layers = 2
        return Qwen3ASRForConditionalGeneration.from_pretrained(model_id, config=config)

    @staticmethod
    def instantiate_tokenizer(model_id: str):
        """Return the processor -- it owns the mel feature extractor as well as
        the tokenizer, and the audio datasets/metrics need both."""
        return AutoProcessor.from_pretrained(model_id)

    @classmethod
    def instantiate_position_processor(cls):
        """Qwen3-ASR's decoder uses ordinary 1-D RoPE.

        Audio tokens occupy plain sequential positions, so there is no
        modality-aware position remapping (unlike Qwen-VL's mrope). Returning
        ``None`` makes the generator use default position ids.
        """
        return None

    @classmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list | None = None,
        config: PretrainedConfig | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Sample backbone inputs built around ``inputs_embeds``.

        ``LLM.get_sample_backbone_inputs`` builds ``input_ids``, but a multi-modal
        backbone consumes ``inputs_embeds``. Position ids stay 1-D: Qwen3-ASR uses
        ordinary RoPE, not the Qwen-VL family's 3-D mrope.
        """
        from .generator import Generator

        if config is None:
            config = model.config
        text_config = config.text_config
        hidden_size = text_config.hidden_size
        dtype = getattr(model, "dtype", torch.float32)

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=dtype
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        prepared = Generator.prepare_inputs(
            model=model,
            input_ids=None,
            inputs_embeds=dummy_inputs_embeds,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            layer_cache_descriptors=layer_cache_descriptors,
        )
        return tuple(prepared.values())

    # ---- audio component ----------------------------------------------------
    @classmethod
    def build_audio_wrapper(cls, model: PreTrainedModel) -> torch.nn.Module:
        return Qwen3ASRAudioWrapper(
            model.model.audio_tower, model.model.multi_modal_projector
        )

    @classmethod
    def validate_audio_frames(cls, config: PretrainedConfig, audio_frames: int) -> int:
        return validate_audio_frames(config, audio_frames)

    @classmethod
    def get_sample_audio_inputs(
        cls,
        config: PretrainedConfig,
        audio_frames: int | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Sample ``(input_features, input_features_mask)``.

        The mask is deliberately all ones: see the module docstring -- it is what
        makes the encoder's packing traceable to a static graph.
        """
        audio_frames = cls.validate_audio_frames(
            config, audio_frames or DEFAULT_AUDIO_FRAMES
        )
        num_mel_bins = _audio_config(config).num_mel_bins
        return (
            torch.zeros((1, num_mel_bins, audio_frames), dtype=torch.float32),
            torch.ones((1, audio_frames), dtype=torch.long),
        )

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list | None = None,
        **kwargs,
    ) -> dict[str, dict[int, str]]:
        """Same as ``VLM.get_backbone_dynamic_axes``, but ``position_ids`` is
        2-D (``(batch, seq_len)``): Qwen3-ASR uses ordinary RoPE, not the
        Qwen-VL family's 3-D mrope-shaped position_ids that ``VLM``'s default
        assumes (axis 2). See ``get_sample_backbone_inputs`` above.
        """
        axes = VLM.get_backbone_dynamic_axes(layer_cache_descriptors, **kwargs)
        axes["position_ids"] = {1: "sequence_length"}
        return axes

    @staticmethod
    def get_audio_input_names() -> tuple[str, ...]:
        return ("input_features", "input_features_mask")

    @staticmethod
    def get_audio_output_names(**kwargs) -> tuple[str, ...]:
        return AUDIO.default_output_names

    @staticmethod
    def get_audio_dynamic_axes(layer_cache_descriptors=None) -> dict:
        """No dynamic axes.

        Fixed-length on purpose: the encoder's ``nonzero()`` packing would make
        the token count dynamic too, which the scatter cannot express. Re-export
        with a different ``audio_frames`` instead.
        """
        return {}
