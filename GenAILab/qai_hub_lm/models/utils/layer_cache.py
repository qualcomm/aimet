# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-layer cache descriptors for models with heterogeneous attention types."""

from dataclasses import dataclass
from enum import Enum

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
}


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

    def dummy_state_shapes(
        self, batch_size: int, context_length: int, sequence_length: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Shapes for the two state tensors of this layer during prepare_inputs.

        Returns a pair ``(shape_a, shape_b)`` where *shape_a* corresponds to the
        first flattened tensor (key or conv_state) and *shape_b* to the second
        (value or recurrent_state).

        Full and sliding_window layers are padded to the full context length so
        that a single 4D attention mask can be applied across all layers.
        """
        if self.attention_type in (AttentionType.FULL, AttentionType.SLIDING_WINDOW):
            shape = (
                batch_size,
                self.num_kv_heads,
                context_length - sequence_length,
                self.head_dim,
            )
            return shape, shape
        if self.attention_type == AttentionType.LINEAR:
            conv_shape = (batch_size, self.conv_dim, self.conv_kernel_size)
            recurrent_shape = (
                batch_size,
                self.linear_num_v_heads,
                self.linear_head_k_dim,
                self.linear_head_v_dim,
            )
            return conv_shape, recurrent_shape
        raise ValueError(f"Unknown attention type: {self.attention_type}")

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

        descriptors.append(
            LayerCacheDescriptor(
                layer_idx=i,
                attention_type=attention_type,
                num_kv_heads=layer_config.num_key_value_heads,
                head_dim=head_dim,
                sliding_window_size=sw_size,
                **linear_kwargs,
            )
        )

    return descriptors


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


def cache_state_names(
    layer_cache_descriptors: list[LayerCacheDescriptor],
    suffix: str = "in",
) -> list[str]:
    """Return per-layer state names for the KV / recurrent cache.

    *suffix* is ``"in"`` for input names and ``"out"`` for output names.
    """
    names: list[str] = []
    for desc in layer_cache_descriptors:
        i = desc.layer_idx
        if desc.attention_type == AttentionType.LINEAR:
            # Genie finds linear-attention state by these exact prefixes.
            names += [f"conv_state_{i}_{suffix}", f"recurrent_state_{i}_{suffix}"]
        else:
            names += [f"past_key_{i}_{suffix}", f"past_value_{i}_{suffix}"]
    return names


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
