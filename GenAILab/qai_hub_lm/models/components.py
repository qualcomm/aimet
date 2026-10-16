# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Component taxonomy for multi-modal LLMs.

A *component* is a sub-graph carrying its own QuantSim, recipe chain, precision
block and export directory. Every model has a ``backbone``; multi-modal models
add more.

``input_encoder`` components (visual, audio) sit upstream and have their
embeddings scattered into the token sequence, so they participate in fusion and
their generator kwargs are prefill-only. ``output_head`` components sit
downstream and change the output instead; none exists yet, but declaring the kind
keeps adding one a registry entry rather than another clone of the visual path.

Hook names are spelled out per component because the vision ones are
inconsistent (``build_vision_wrapper`` vs ``get_visual_input_names``); the
registry absorbs that so the rest of the framework stays generic.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ..schema.components import (
        BACKBONE,
        require_component,
        MODALITY_COMPONENTS,
        ComponentKind,
    )
except ImportError:  # copied into AI Hub Models, where the schema is ``lm_schema``
    from ..lm_schema.components import (
        BACKBONE,
        require_component,
        MODALITY_COMPONENTS,
        ComponentKind,
    )

__all__ = [
    "AUDIO",
    "BACKBONE",
    "COMPONENTS",
    "ComponentKind",
    "ComponentSpec",
    "VISUAL",
    "input_encoders",
    "model_components",
    "prefill_only_keys",
    "spec",
    "token_id",
]


@dataclass(frozen=True)
class ComponentSpec:
    """Everything the framework needs to treat a component generically."""

    name: str
    kind: ComponentKind

    # Names of the hooks a model class implements for this component.
    build_wrapper: str
    sample_inputs: str
    input_names: str
    output_names: str
    dynamic_axes: str

    # --- input-encoder only -------------------------------------------------
    #: ``config`` attribute holding the placeholder token id that this
    #: modality's embeddings are scattered into.
    token_id_attr: str | None = None
    #: Generator kwargs owned by this component. They must survive prefill and
    #: must NOT be re-fed during decode.
    prefill_only_keys: tuple[str, ...] = ()
    #: Fallback export/output names when the model class does not override.
    default_output_names: tuple[str, ...] = ()
    #: Name of the ``model:`` config knob that fixes this encoder's traced input
    #: shape, and of the kwarg its ``sample_inputs`` hook takes for it.
    shape_kwarg: str = ""
    #: Generator attribute holding this component's encoder module.
    model_attr: str = ""
    #: Generator method yielding that encoder's input tuples, used by
    #: ``component_quantization_mode``.
    prefill_hook: str = ""

    @property
    def is_input_encoder(self) -> bool:
        return self.kind is ComponentKind.input_encoder


VISUAL = ComponentSpec(
    name="visual",
    kind=MODALITY_COMPONENTS["visual"],
    build_wrapper="build_vision_wrapper",
    sample_inputs="get_sample_vision_inputs",
    input_names="get_visual_input_names",
    output_names="get_visual_output_names",
    dynamic_axes="get_visual_dynamic_axes",
    token_id_attr="image_token_id",
    prefill_only_keys=(
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_position_ids",
    ),
    default_output_names=("image_embeddings",),
    shape_kwarg="image_size",
    model_attr="vision_model",
    prefill_hook="_prefill_visual",
)

AUDIO = ComponentSpec(
    name="audio",
    kind=MODALITY_COMPONENTS["audio"],
    build_wrapper="build_audio_wrapper",
    sample_inputs="get_sample_audio_inputs",
    input_names="get_audio_input_names",
    output_names="get_audio_output_names",
    dynamic_axes="get_audio_dynamic_axes",
    token_id_attr="audio_token_id",
    prefill_only_keys=(
        "input_features",
        "input_features_mask",
        "feature_attention_mask",
    ),
    default_output_names=("audio_embeddings",),
    shape_kwarg="audio_frames",
    model_attr="audio_model",
    prefill_hook="_prefill_audio",
)


#: Registry of non-backbone components, keyed by name. Ordered by
#: ``MODALITY_COMPONENTS`` so schema and model layers agree on canonical order.
_SPECS = (VISUAL, AUDIO)
COMPONENTS: dict[str, ComponentSpec] = {
    name: next(s for s in _SPECS if s.name == name) for name in MODALITY_COMPONENTS
}


def spec(name: str) -> ComponentSpec:
    """Look up a component spec by name, or raise with the valid names."""
    return COMPONENTS[require_component(name)]


def input_encoders() -> tuple[ComponentSpec, ...]:
    """The input-encoder specs, in canonical order."""
    return tuple(c for c in COMPONENTS.values() if c.is_input_encoder)


def prefill_only_keys() -> frozenset[str]:
    """Union of every component's prefill-only generator kwargs.

    These are dropped after prefill because decode processes a single token and
    the modality embeddings are already fused into the KV cache.
    """
    return frozenset(k for c in COMPONENTS.values() for k in c.prefill_only_keys)


def model_components(model_cls: type) -> tuple[ComponentSpec, ...]:
    """The component specs a model class declares, in canonical order.

    Reads the class's ``COMPONENTS`` attribute (a tuple of names). Declaring is
    explicit rather than inferred from which hooks exist, because the base
    classes define the modality hooks as abstract -- present but unimplemented --
    so presence proves nothing.
    """
    declared = getattr(model_cls, "COMPONENTS", ()) or ()
    unknown = set(declared) - set(COMPONENTS)
    if unknown:
        raise ValueError(
            f"{model_cls.__name__} declares unknown component(s) {sorted(unknown)}. "
            f"Known components: {sorted(COMPONENTS)}."
        )
    return tuple(COMPONENTS[n] for n in COMPONENTS if n in declared)


def token_id(config, component: ComponentSpec) -> int | None:
    """The placeholder token id for an input encoder, or ``None`` if absent."""
    if component.token_id_attr is None:
        return None
    return getattr(config, component.token_id_attr, None)
