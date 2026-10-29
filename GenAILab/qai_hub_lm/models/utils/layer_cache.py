# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-layer cache descriptors for models with heterogeneous attention types."""

from dataclasses import dataclass
from enum import Enum

import torch
from transformers import PretrainedConfig


class AttentionType(Enum):
    """Attention mechanism used by a decoder layer."""

    FULL = "full_attention"
    SLIDING_WINDOW = "sliding_attention"
    LINEAR = "linear_attention"


# Mapping from HuggingFace config ``layer_types`` strings to AttentionType
_HF_LAYER_TYPE_MAP: dict[str, AttentionType] = {
    "full_attention": AttentionType.FULL,
    "sliding_attention": AttentionType.SLIDING_WINDOW,
    "linear_attention": AttentionType.LINEAR,
    "recurrent": AttentionType.LINEAR,
    # Qwen4-Exp's sparse attention keeps a full KV cache plus indexer keys
    "qwen_sparse_attention": AttentionType.FULL,
}


class StateKind(Enum):
    """How a cache state evolves between generation steps."""

    #: Grows along ``seq_dim`` by the valid tokens of each call, clipped to the
    #: retained context.
    APPEND = "append"
    #: Replaced wholesale by each call's output.
    REPLACE = "replace"


@dataclass(frozen=True)
class CacheState:
    """One state tensor a layer carries between calls, as a graph input/output.

    ``shape`` gives the trailing dims after batch for REPLACE states. APPEND
    states have their sequence axis at ``seq_dim`` (counting batch as dim 0),
    and ``shape`` holds the remaining non-batch dims in order.
    """

    name: str
    kind: StateKind
    shape: tuple[int, ...]
    seq_dim: int | None = None
    dtype: torch.dtype | None = None

    def dummy_shape(self, batch_size: int, kv_length: int) -> tuple[int, ...]:
        """Shape of the empty state fed on the first call."""
        if self.kind == StateKind.REPLACE:
            return (batch_size, *self.shape)
        dims = list(self.shape)
        dims.insert(self.seq_dim - 1, kv_length)
        return (batch_size, *dims)


#: Name of the model-level state holding the position ids of every cached key.
PAST_POSITION_IDS = "past_position_ids"

#: Conv-state slot of a Qwen4-Exp linear-attention cache layer that holds the
#: PLE short-conv window (slot 0 is the GatedDeltaNet conv window).
PLE_CONV_STATE_IDX = 1


@dataclass
class LayerCacheDescriptor:
    """Describes the cache/state requirements of a single decoder layer."""

    layer_idx: int
    attention_type: AttentionType
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int | None = None
    # Linear attention specific dimensions
    conv_dim: int | None = None
    conv_kernel_size: int | None = None
    linear_num_v_heads: int | None = None
    linear_head_k_dim: int | None = None
    linear_head_v_dim: int | None = None
    #: Head dim of Qwen4-Exp's QSA indexer keys; ``None`` for layers without one.
    indexer_head_dim: int | None = None
    #: ``(channels, length)`` of a Qwen4-Exp PLE layer's dilated short-conv
    #: window; ``None`` for layers without PLE.
    ple_conv_shape: tuple[int, int] | None = None

    @property
    def states(self) -> tuple[CacheState, ...]:
        """The states this layer carries, in graph input/output order."""
        i = self.layer_idx
        if self.attention_type == AttentionType.LINEAR:
            # Genie finds linear-attention state by these exact prefixes.
            return (
                CacheState(
                    f"conv_state_{i}",
                    StateKind.REPLACE,
                    (self.conv_dim, self.conv_kernel_size),
                ),
                CacheState(
                    f"recurrent_state_{i}",
                    StateKind.REPLACE,
                    (
                        self.linear_num_v_heads,
                        self.linear_head_k_dim,
                        self.linear_head_v_dim,
                    ),
                ),
            ) + (
                (
                    CacheState(
                        f"ple_conv_state_{i}", StateKind.REPLACE, self.ple_conv_shape
                    ),
                )
                if self.ple_conv_shape is not None
                else ()
            )
        if self.attention_type in (AttentionType.FULL, AttentionType.SLIDING_WINDOW):
            kv = (self.num_kv_heads, self.head_dim)
            states = (
                CacheState(f"past_key_{i}", StateKind.APPEND, kv, seq_dim=2),
                CacheState(f"past_value_{i}", StateKind.APPEND, kv, seq_dim=2),
            )
            if self.indexer_head_dim is not None:
                states += (
                    CacheState(
                        f"past_indexer_key_{i}",
                        StateKind.APPEND,
                        (self.indexer_head_dim,),
                        seq_dim=1,
                    ),
                )
            return states
        raise ValueError(f"Unknown attention type: {self.attention_type}")

    def dummy_state_shapes(
        self, batch_size: int, context_length: int, sequence_length: int
    ) -> tuple[tuple[int, ...], ...]:
        """Shapes of this layer's states during prepare_inputs, in ``states`` order.

        Full and sliding_window layers are padded to the full context length so
        that a single 4D attention mask can be applied across all layers.
        """
        return tuple(
            state.dummy_shape(batch_size, context_length - sequence_length)
            for state in self.states
        )

    def clip_length(self, max_length: int) -> int | None:
        """Maximum number of KV entries to retain between generation steps.

        Returns *None* for linear attention (state is replaced, not clipped).
        """
        if self.attention_type == AttentionType.FULL:
            return max_length
        if self.attention_type == AttentionType.SLIDING_WINDOW:
            return max_length
        if self.attention_type == AttentionType.LINEAR:
            return None
        raise ValueError(f"Unknown attention type: {self.attention_type}")


def _resolve_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Resolve a composite VLM config to its text decoder config.

    VLM configs (e.g. Gemma3Config) nest text decoder attributes under
    ``text_config``.  InternVL uses ``llm_config`` instead.
    Pure text LLM configs have them at the top level.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return text_config
    llm_config = getattr(config, "llm_config", None)
    if llm_config is not None and hasattr(llm_config, "num_hidden_layers"):
        return llm_config
    return config


def build_layer_cache_descriptors(
    config: PretrainedConfig,
) -> list[LayerCacheDescriptor]:
    """Build per-layer cache descriptors from a HuggingFace model config.

    Inspects ``config.layer_types``, ``config.sliding_window``, and
    ``config.sliding_window_pattern`` to determine each layer's cache type.

    For composite VLM configs the text decoder sub-config is resolved
    automatically via :func:`_resolve_text_config`.
    """
    config = _resolve_text_config(config)
    num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
    num_layers = config.num_hidden_layers - num_kv_shared
    sliding_window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)

    # Extract linear attention dimensions from config (e.g. Qwen 3.5)
    linear_num_k_heads = getattr(config, "linear_num_key_heads", None)
    linear_num_v_heads = getattr(config, "linear_num_value_heads", None)
    linear_head_k_dim = getattr(config, "linear_key_head_dim", None)
    linear_head_v_dim = getattr(config, "linear_value_head_dim", None)
    linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", None)

    # conv_dim = key_dim * 2 + value_dim
    if (
        linear_num_k_heads
        and linear_head_k_dim
        and linear_num_v_heads
        and linear_head_v_dim
    ):
        conv_dim = (
            linear_head_k_dim * linear_num_k_heads * 2
            + linear_head_v_dim * linear_num_v_heads
        )
    else:
        conv_dim = None

    descriptors: list[LayerCacheDescriptor] = []
    for i in range(num_layers):
        hf_layer_type = layer_types[i] if layer_types else None
        mapped = _HF_LAYER_TYPE_MAP.get(hf_layer_type) if hf_layer_type else None

        if mapped is not None:
            attention_type = mapped
        elif _is_sliding_window_layer(config, i):
            attention_type = AttentionType.SLIDING_WINDOW
        else:
            attention_type = AttentionType.FULL

        sw_size = (
            sliding_window if attention_type == AttentionType.SLIDING_WINDOW else None
        )
        layer_config = _layer_config(config, i)
        head_dim = _head_dim(layer_config)
        if attention_type == AttentionType.FULL:
            # Some models (e.g. Gemma4) use a larger head_dim for full-attention layers
            head_dim = getattr(layer_config, "global_head_dim", None) or head_dim

        # Include linear attention dimensions when applicable
        linear_kwargs = {}
        if attention_type == AttentionType.LINEAR:
            linear_kwargs = dict(
                conv_dim=conv_dim,
                conv_kernel_size=linear_conv_kernel_dim,
                linear_num_v_heads=linear_num_v_heads,
                linear_head_k_dim=linear_head_k_dim,
                linear_head_v_dim=linear_head_v_dim,
            )

        indexer_kwargs = {}
        if hf_layer_type == "qwen_sparse_attention":
            indexer_kwargs = dict(indexer_head_dim=config.indexer_head_dim)
        if i + 1 in ple_layer_ids(config):
            # Qwen4-Exp PLE: depthwise conv over every residual stream, dilated by
            # the n-gram size, so it keeps (kernel - 1) * ngram_size past columns
            indexer_kwargs["ple_conv_shape"] = (
                config.hidden_size * config.hc_count,
                (config.ple_conv_kernel_size - 1) * config.ngram_size,
            )

        descriptors.append(
            LayerCacheDescriptor(
                layer_idx=i,
                attention_type=attention_type,
                num_kv_heads=layer_config.num_key_value_heads,
                head_dim=head_dim,
                sliding_window_size=sw_size,
                **linear_kwargs,
                **indexer_kwargs,
            )
        )

    return descriptors


def ple_layer_ids(config: PretrainedConfig) -> list[int]:
    """One-indexed ids of Qwen4-Exp's per-layer-embedding (PLE) layers, if any."""
    ids = getattr(_resolve_text_config(config), "ple_layer_ids", None)
    return list(ids) if isinstance(ids, (list, tuple)) else []


def _layer_config(config: PretrainedConfig, layer_idx: int) -> PretrainedConfig:
    """Return the config that holds layer *layer_idx*'s own attention settings.

    From transformers 5.17, settings that differ between layers (e.g. Gemma4's
    larger head_dim on full-attention layers) live in ``config.per_layer_config``,
    and reading them from the top-level config raises. Older versions keep every
    setting on the top-level config.
    """
    if getattr(config, "is_heterogeneous", False):
        return config.per_layer_config[layer_idx]
    return config


def _head_dim(config: PretrainedConfig) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    return config.hidden_size // config.num_attention_heads


def has_sliding_window_layers(
    descriptors: list[LayerCacheDescriptor],
) -> bool:
    """Return True if any descriptor uses sliding window attention."""
    return any(d.attention_type == AttentionType.SLIDING_WINDOW for d in descriptors)


def has_full_attention_layers(
    descriptors: list[LayerCacheDescriptor],
) -> bool:
    """Return True if any descriptor uses full attention."""
    return any(d.attention_type == AttentionType.FULL for d in descriptors)


def attention_mask_input_names(
    layer_cache_descriptors: list[LayerCacheDescriptor],
) -> list[str]:
    """Return the ONNX input names for attention mask tensor(s).

    Models with a mix of full and sliding window layers need two separate
    4D masks; models with only one attention type use a single mask.
    """
    has_sw = has_sliding_window_layers(layer_cache_descriptors)
    has_full = has_full_attention_layers(layer_cache_descriptors)
    if has_sw and has_full:
        return ["attention_mask_full", "attention_mask_sliding_window"]
    if has_sw:
        return ["attention_mask_sliding_window"]
    return ["attention_mask"]


def model_states(
    layer_cache_descriptors: list[LayerCacheDescriptor],
) -> tuple[CacheState, ...]:
    """States carried once per model rather than per layer, after the layers'.

    Qwen4-Exp's QSA indexer ropes each compressed key block at its first token's
    position, so it reads past positions back from the cache: the model keeps
    them on ``Cache.position_ids`` as its 3-axis mrope ids, carried here as
    ``(batch, 3, kv_length)``.
    """
    if any(d.indexer_head_dim is not None for d in layer_cache_descriptors):
        return (
            CacheState(
                PAST_POSITION_IDS,
                StateKind.APPEND,
                (3,),
                seq_dim=2,
                dtype=torch.int64,
            ),
        )
    return ()


def all_states(
    layer_cache_descriptors: list[LayerCacheDescriptor],
) -> tuple[CacheState, ...]:
    """Every carried state, layers first, in graph input/output order."""
    layer_states = tuple(
        state for desc in layer_cache_descriptors for state in desc.states
    )
    return layer_states + model_states(layer_cache_descriptors)


def cache_state_names(
    layer_cache_descriptors: list[LayerCacheDescriptor],
    suffix: str = "in",
) -> list[str]:
    """Return the names of every carried state, in graph order.

    *suffix* is ``"in"`` for input names and ``"out"`` for output names.
    """
    return [f"{state.name}_{suffix}" for state in all_states(layer_cache_descriptors)]


def kv_length_dynamic_axes(
    layer_cache_descriptors: list[LayerCacheDescriptor],
) -> dict[str, dict[int, str]]:
    """``dynamic_axes`` entries for every input that grows with the KV cache."""
    return {
        f"{state.name}_in": {state.seq_dim: "kv_cache_length"}
        for state in all_states(layer_cache_descriptors)
        if state.kind == StateKind.APPEND
    }


def _is_sliding_window_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    """Determine whether *layer_idx* uses sliding window attention.

    Detection heuristics (checked in order):
    1. ``config.sliding_window_pattern`` — modular pattern (e.g. every Nth layer).
    2. Known model types with hardcoded patterns (e.g. Gemma 2 alternates).
    3. If ``config.sliding_window`` is set but no pattern is found, all layers
       are assumed to use sliding window.
    """
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return False

    # Explicit pattern from config
    pattern = getattr(config, "sliding_window_pattern", None)
    if pattern is not None:
        return (layer_idx % pattern) == 0

    # Gemma 2: odd-indexed layers use sliding window
    model_type = getattr(config, "model_type", "")
    if model_type == "gemma2":
        return layer_idx % 2 != 0

    # Fallback: sliding_window is set with no pattern — assume no layers
    return False
