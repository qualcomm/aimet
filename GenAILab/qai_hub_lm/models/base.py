# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM base class for GenAI test framework"""

import types
from abc import abstractmethod, ABC
import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.cache_utils import DynamicCache

from .components import AUDIO, COMPONENTS, VISUAL, spec
from .generator import Generator, VLM_Generator
from .utils.layer_cache import (
    LayerCacheDescriptor,
    attention_mask_input_names,
    cache_state_names,
    kv_length_dynamic_axes,
    _resolve_text_config,
)


class SimCollection:
    """Holds the QuantSim models for the components of an LLM.

    Each modality component is an optional field, ``None`` when absent -- the
    same shape as :class:`ModelCacheEntry` and :class:`ResolvedRecipe`, plus a
    :meth:`component` accessor for code generic over the registry.

    Keyword-only after ``backbone``, so adding a component cannot shift an
    existing positional argument.
    """

    def __init__(
        self,
        backbone: "QuantizationSimModel",
        *,
        visual: "QuantizationSimModel" = None,
        audio: "QuantizationSimModel" = None,
        embedding: torch.nn.Module = None,
        config: PretrainedConfig = None,
        position_id_processor: types.FunctionType = None,
        extras: dict[str, torch.nn.Module] = None,
    ):
        self.backbone = backbone
        self.visual = visual
        self.audio = audio
        self.embedding = embedding
        self.config = config
        self.position_id_processor = position_id_processor
        self.extras = extras or {}

    def component(self, name: str) -> "QuantizationSimModel":
        """The sim for a modality component, or ``None`` if this model lacks it."""
        spec(name)  # raises KeyError for an unknown component
        return getattr(self, name)

    def has(self, name: str) -> bool:
        """Whether this model has a live sim for the named component."""
        return self.component(name) is not None

    def iter_members(self):
        """Yield the backbone, every modality component, and the embedding.

        The placeable sims/modules the collection owns -- what device placement
        and quantizer sweeps iterate. ``config``, ``position_id_processor`` and
        ``extras`` are excluded: they are not sims, and extras are placed by the
        generator that consumes them.
        """
        yield self.backbone
        for name in COMPONENTS:
            yield getattr(self, name)
        yield self.embedding

    def present_components(self) -> tuple[str, ...]:
        """Names of the live modality components, in canonical registry order."""
        return tuple(n for n in COMPONENTS if self.has(n))

    def is_vlm(self) -> bool:
        return self.has(VISUAL.name)


class LLM(ABC):
    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        """Instantiate model"""
        llm_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )

        if small_model:
            text_cfg = _resolve_text_config(llm_config)
            text_cfg.num_hidden_layers = 2
            if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
                text_cfg.layer_types = text_cfg.layer_types[:2]

        return AutoModelForCausalLM.from_pretrained(model_id, config=llm_config)

    @staticmethod
    def instantiate_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
        """Instantiate model tokenizer"""
        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def instantiate_float_model(
        cls,
        model_id: str,
        small_model: bool = False,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        """Load the raw float model.

        Separated from :meth:`instantiate_quantsim` so the caller can transform
        the float model (e.g. apply SpinQuant) before the sim is built.
        """
        return cls.instantiate_model(model_id, small_model).to(dtype=dtype)

    @classmethod
    @abstractmethod
    def instantiate_quantsim(cls, model, *args, **kwargs) -> SimCollection:
        """Instantiate QuantSim models for components from a raw float model.

        A model whose checkpoint is a packed QAT variant may override this to
        dequantize, build the sim, and load the trained QAT scales (see
        Gemma4_Torch)
        """
        pass

    @classmethod
    @abstractmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for LLM backbone QuantSim instantiation or ONNX export"""
        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        prepared = Generator.prepare_inputs(
            model=model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            layer_cache_descriptors=layer_cache_descriptors,
        )
        return tuple(prepared.values())

    @staticmethod
    def get_cache_type() -> type:
        """
        Returns ``DynamicCache`` by default. Models with hybrid attention
        (e.g. linear + full) can override this to return ``HybridCache``.
        """
        return DynamicCache

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        """Get input names for the backbone model."""
        return tuple(
            ["input_ids"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
        )

    @staticmethod
    def get_backbone_output_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        """Get output names for the backbone model."""
        return tuple(["logits"] + cache_state_names(layer_cache_descriptors, "out"))

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        """Build ``dynamic_axes`` dict for ``torch.onnx.export``.

        Marks the sequence_length and kv_cache_length dimensions as dynamic so
        that a single ONNX graph can be used with varying sequence lengths.

        This is the single declaration of *which* dims are dynamic, for both
        export paths: the TorchScript tracer consumes it as-is, and the dynamo
        path lowers it to ``dynamic_shapes`` (see
        ``dynamic_axes_to_dynamic_shapes``). Dims sharing a symbol name are the
        same dim; the dynamo lowering leaves the relationships between them to
        torch.export, which recovers them from the traced guards.
        """
        axes: dict[str, dict[int, str]] = {
            "input_ids": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
        } | {
            name: {2: "sequence_length"}
            for name in attention_mask_input_names(layer_cache_descriptors)
        }
        axes |= kv_length_dynamic_axes(layer_cache_descriptors)
        return axes

    @staticmethod
    def use_dynamo_export() -> bool:
        """Whether to use dynamo-based ONNX export. Models with ops unsupported
        by the TorchScript tracer (e.g. data-dependent control flow) should
        override this to return True."""
        return False

    @classmethod
    def use_dynamo_export_for(cls, component: str | None = None) -> bool:
        """Export path for one component, defaulting to the model-wide setting.

        The tracer's limits belong to the sub-graph, not the model: Gemma4's audio
        tower needs dynamo while its backbone and vision tower trace fine.
        """
        return cls.use_dynamo_export()

    @staticmethod
    def get_generator_cls() -> type[Generator]:
        return Generator

    @classmethod
    def get_extras(cls, model: PreTrainedModel) -> dict:
        """Return extra host-side modules the generator needs (e.g. per-layer embeddings)."""
        return {}


def _unimplemented(cls, hook: str):
    """The error every unimplemented component hook raises.

    One message shape, so a model that declares a component but misses one of
    its hooks fails the same way whichever hook it is.
    """
    return NotImplementedError(
        f"{cls.__name__} declares COMPONENTS={getattr(cls, 'COMPONENTS', ())} "
        f"but does not implement {hook}."
    )


class VLM(LLM):
    """Base for multi-modal LLMs -- those whose backbone consumes ``inputs_embeds``
    because an upstream encoder's embeddings are scattered into the sequence.

    Subclasses declare their components via ``COMPONENTS`` and implement only
    those hooks; the rest stay concrete-but-unimplemented so an audio-only model
    need not supply vision hooks.
    """

    #: Names of the modality components this model has, from
    #: :data:`.components.COMPONENTS`. Defaults to
    #: vision, which is what every subclass was before audio existed.
    COMPONENTS: tuple[str, ...] = (VISUAL.name,)

    @classmethod
    def supports_component(cls, component: str, config: PretrainedConfig) -> bool:
        """Whether a *specific checkpoint* actually has a declared component.

        ``COMPONENTS`` is per class, but a modality can be optional per checkpoint
        (Gemma4 ships variants with ``audio_config=None``), so those classes
        override this to consult the config.
        """
        return component in cls.COMPONENTS

    @classmethod
    @abstractmethod
    def instantiate_position_processor(cls):
        pass

    @classmethod
    def get_language_model(cls, model: PreTrainedModel) -> torch.nn.Module:
        """Return the decoder module from the VLM."""
        return model.model.language_model

    @classmethod
    def get_lm_head(cls, model: PreTrainedModel) -> torch.nn.Module | None:
        """Return the LM head module, or None if built into the language model."""
        return model.lm_head

    @classmethod
    def get_embedding(cls, model: PreTrainedModel) -> torch.nn.Module:
        """Return the embedding table."""
        return cls.get_language_model(model).get_input_embeddings()

    @classmethod
    def build_vision_wrapper(cls, model: PreTrainedModel) -> torch.nn.Module:
        """Return a traceable vision wrapper module for quantization/export."""
        raise _unimplemented(cls, "build_vision_wrapper")

    @classmethod
    def get_sample_vision_inputs(
        cls,
        config: PretrainedConfig,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for visual model QuantSim instantiation or ONNX export"""
        raise _unimplemented(cls, "get_sample_vision_inputs")

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        **kwargs,
    ) -> tuple[str, ...]:
        """Get input names for the backbone model."""
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
            "attention_mask": {2: "sequence_length"},
            "position_ids": {2: "sequence_length"},
            "logits": {1: "sequence_length"},
        }
        axes |= kv_length_dynamic_axes(layer_cache_descriptors)
        return axes

    @staticmethod
    def get_visual_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        return {}

    @classmethod
    def get_visual_input_names(cls) -> tuple[str, ...]:
        """Get input names for the visual model"""
        raise _unimplemented(cls, "get_visual_input_names")

    @classmethod
    def get_visual_output_names(cls, **kwargs) -> tuple[str, ...]:
        """Get output names for the visual model"""
        raise _unimplemented(cls, "get_visual_output_names")

    # ---- audio component ----------------------------------------------------
    # Mel extraction stays in the HF processor, outside these wrappers, as image
    # resizing sits outside the vision sim. ``audio_frames`` always means padded
    # mel frames (the analogue of ``image_size``), never waveform samples; the
    # encoder subsamples it further to the audio-token count.

    @classmethod
    def validate_audio_frames(cls, config: PretrainedConfig, audio_frames: int) -> int:
        """Check a padded mel-frame count against this encoder, and return it.

        No constraint by default. An encoder that chunks the time axis overrides
        this so a bad ``model.audio_frames`` fails naming the knob.
        """
        return audio_frames

    @classmethod
    def build_audio_wrapper(cls, model: PreTrainedModel) -> torch.nn.Module:
        """Return a traceable audio-encoder wrapper for quantization/export."""
        raise _unimplemented(cls, "build_audio_wrapper")

    @classmethod
    def get_sample_audio_inputs(
        cls,
        config: PretrainedConfig,
        audio_frames: int | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for audio model QuantSim instantiation or ONNX export.

        ``audio_frames`` is the padded mel-frame count and fixes the traced shape.
        """
        raise _unimplemented(cls, "get_sample_audio_inputs")

    @classmethod
    def get_audio_input_names(cls) -> tuple[str, ...]:
        """Get input names for the audio model"""
        raise _unimplemented(cls, "get_audio_input_names")

    @staticmethod
    def get_audio_output_names(**kwargs) -> tuple[str, ...]:
        """Get output names for the audio model"""
        return AUDIO.default_output_names

    @staticmethod
    def get_audio_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        return {}

    # ---- generic component dispatch -----------------------------------------
    @classmethod
    def build_component_wrapper(
        cls, component: str, model: PreTrainedModel
    ) -> torch.nn.Module:
        """Build the traceable wrapper for any declared component."""
        return getattr(cls, COMPONENTS[component].build_wrapper)(model)

    @classmethod
    def get_sample_component_inputs(
        cls, component: str, config: PretrainedConfig, **kwargs
    ) -> tuple[torch.Tensor, ...]:
        """Sample inputs for any declared component."""
        return getattr(cls, COMPONENTS[component].sample_inputs)(config, **kwargs)

    @classmethod
    def get_component_input_names(cls, component: str) -> tuple[str, ...]:
        return getattr(cls, COMPONENTS[component].input_names)()

    @classmethod
    def get_component_output_names(cls, component: str, **kwargs) -> tuple[str, ...]:
        return getattr(cls, COMPONENTS[component].output_names)(**kwargs)

    @classmethod
    def get_component_dynamic_axes(
        cls,
        component: str,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        return getattr(cls, COMPONENTS[component].dynamic_axes)(layer_cache_descriptors)

    @staticmethod
    def get_generator_cls() -> type[VLM_Generator]:
        return VLM_Generator
