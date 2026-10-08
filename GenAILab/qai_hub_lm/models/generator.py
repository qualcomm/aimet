# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM Generator class to restore HF API on models with static shape constraints"""

from __future__ import annotations

import contextlib
import functools
import itertools
from collections import OrderedDict
import typing
from typing import Any, Union
import types
import warnings

import numpy as np
import torch
import transformers
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .components import AUDIO, COMPONENTS, VISUAL, prefill_only_keys, spec
from ..schema.components import require_component
from .utils.attention_mask import (
    convert_2d_attention_mask_to_4d,
    convert_2d_attention_mask_to_4d_sliding_window,
)

from .utils.layer_cache import (
    AttentionType,
    build_layer_cache_descriptors,
    _resolve_text_config,
)
from .utils.rope_embedding import RopeEmbedding, RopeEmbeddingProtocol


#: Fallback mel-frame hop, in samples, when a processor's feature extractor
#: doesn't expose its own ``hop_length`` -- 10 ms at 16 kHz, the convention
#: both Qwen3-ASR and Gemma4's mel extractors use. Only used to estimate a
#: safe waveform pad/truncate point for a fixed ``audio_frames`` export; see
#: ``VLM_Generator.fit_audio_waveform``.
DEFAULT_AUDIO_HOP_LENGTH = 160


def ordered_dict_replace(
    d: OrderedDict, old_key: str, replacements: list[tuple]
) -> OrderedDict:
    """Replace *old_key* positionally with one or more new key-value pairs."""
    items = list(d.items())
    idx = next(i for i, (k, _) in enumerate(items) if k == old_key)
    items[idx : idx + 1] = replacements
    return OrderedDict(items)


class _FlatListCache:
    """Lightweight cache wrapper for hybrid attention models.

    Stores the flattened state list (2 tensors per layer) and exposes the
    ``get_seq_length()`` / ``to_legacy_cache()`` interface expected by
    HuggingFace's generation loop, routing ``get_seq_length`` queries to the
    first full-attention layer.
    """

    def __init__(
        self,
        flat_list: list[torch.Tensor],
        descriptors: list,
    ):
        self._flat = flat_list
        self._descriptors = descriptors
        # Find the first full-attention layer index for seq_length queries
        self._full_attn_idx: int | None = None
        for desc in descriptors:
            if desc.attention_type == AttentionType.FULL:
                self._full_attn_idx = desc.layer_idx
                break

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._flat:
            return 0
        idx = self._full_attn_idx if self._full_attn_idx is not None else 0
        tensor = self._flat[idx * 2]  # key tensor for this layer
        return tensor.shape[-2] if tensor.ndim >= 3 else 0

    def to_legacy_cache(self):
        return [
            (self._flat[i], self._flat[i + 1]) for i in range(0, len(self._flat), 2)
        ]

    def to_flat_list(self) -> list[torch.Tensor]:
        """Return the flat ``[k0, v0, k1, v1, ...]`` state list."""
        return list(self._flat)


def _flatten_past_key_values(past_key_values) -> list[torch.Tensor]:
    """Flatten a cache into ``[k0, v0, k1, v1, ...]``.

    Handles the ``_FlatListCache`` produced for hybrid (linear + full) models —
    which stores the flat list directly and is not iterable as (key, value)
    pairs — as well as legacy caches that iterate as ``(key, value)`` tuples.
    """
    if past_key_values is None or past_key_values.get_seq_length() == 0:
        return []
    if isinstance(past_key_values, _FlatListCache):
        return past_key_values.to_flat_list()
    return [t for layer_kv in past_key_values for t in (layer_kv[0], layer_kv[1])]


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    layer_cache_descriptors: list | None = None,
) -> list[torch.Tensor]:
    """
    Combine past_key_vals with new_key_vals and clip them so there are no more than `length` tokens worth of context.

    When *layer_cache_descriptors* is provided, per-layer behaviour is applied:
    - ``"full"`` / ``"sliding_window"`` layers: concatenate and clip to *length*.
    - ``"linear"`` layers: the new state replaces the old state entirely
      (no concatenation).
    """
    ret = []

    def _get_desc(layer_idx):
        if layer_cache_descriptors and layer_idx < len(layer_cache_descriptors):
            return layer_cache_descriptors[layer_idx]
        return None

    # If there are no past_key_vals create some empty ones in the correct shape
    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            desc = _get_desc(i // 2)
            if desc and desc.attention_type == AttentionType.LINEAR:
                past_key_vals.append(torch.zeros_like(new_key_vals[i]))
                past_key_vals.append(torch.zeros_like(new_key_vals[i + 1]))
            else:
                key_shape = new_key_vals[i].shape
                key_shape = (key_shape[0], key_shape[1], 0, key_shape[3])
                past_key_vals.append(torch.zeros(key_shape, device=device))

                value_shape = new_key_vals[i + 1].shape
                value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
                past_key_vals.append(torch.zeros(value_shape, device=device))

    # If there are no new_key_vals create some empty ones in the correct shape
    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            desc = _get_desc(i // 2)
            if desc and desc.attention_type == AttentionType.LINEAR:
                new_key_vals.append(torch.zeros_like(past_key_vals[i]))
                new_key_vals.append(torch.zeros_like(past_key_vals[i + 1]))
            else:
                key_shape = past_key_vals[i].shape
                key_shape = (key_shape[0], key_shape[1], 0, key_shape[3])
                new_key_vals.append(torch.zeros(key_shape, device=device))

                value_shape = past_key_vals[i + 1].shape
                value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
                new_key_vals.append(torch.zeros(value_shape, device=device))

    # Combine past and new values per layer
    for i in range(0, len(past_key_vals), 2):
        desc = _get_desc(i // 2)

        if desc and desc.attention_type == AttentionType.LINEAR:
            # Linear attention: state is replaced, not concatenated
            ret.append(new_key_vals[i].to(device=device, dtype=dtype))
            ret.append(new_key_vals[i + 1].to(device=device, dtype=dtype))
            continue

        # Full or sliding_window: concatenate on sequence dimension and clip
        clip_len = desc.clip_length(length) if desc else length

        key_cache = torch.cat(
            [past_key_vals[i].to(device), new_key_vals[i].to(device)],
            dim=2,
        )
        key_cache = key_cache[:, :, -clip_len:, :]
        val_cache = torch.cat(
            [
                past_key_vals[i + 1].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )
        val_cache = val_cache[:, :, -clip_len:, :]

        ret.append(key_cache.to(dtype=dtype))
        ret.append(val_cache.to(dtype=dtype))
    return ret


class Generator(GenerationMixin, torch.nn.Module):
    """Restores HuggingFace LLM API on models with static shape constraints.

    Provides ``forward`` and ``generate`` APIs that handle input padding,
    KV cache management, and multi-slice prefill for models compiled to fixed
    sequence lengths.

    Model Contract
    --------------
    The ``model`` passed to ``__init__`` must satisfy:

    1. **Callable as** ``model(*tensors) -> tuple[Tensor, ...]``
       Positional tensor args in the order produced by ``prepare_inputs``.
       Returns a flat tuple: ``(logits, *flat_kv_states)``.

    2. **``.config``** — a ``PretrainedConfig`` (or compatible) exposing at
       minimum: ``num_hidden_layers``, ``num_key_value_heads``, ``head_dim``.
       VLM composite configs may nest these under ``text_config``.
       Optional: ``layer_types`` (for hybrid attention), ``sliding_window``.

    3. **``.device``** — ``torch.device`` where the model lives.

    4. **``.dtype``** — ``torch.dtype`` used for KV cache tensor allocation.

    5. **Static sequence dimension** — expects inputs padded to exactly
       ``sequence_length`` tokens. KV cache has ``context_length -
       sequence_length`` slots.

    6. **KV cache layout** — ``(batch, num_kv_heads, seq_len, head_dim)``
       for standard and sliding-window attention. Linear attention layers
       use replacement semantics (state overwritten each step).

    7. **4D attention mask** — shape ``(batch, 1, seq_len, context_len)``,
       float-valued: 0 = attend, negative = block.

    8. **Input order** — ``input_ids`` | ``inputs_embeds``, attention mask(s),
       ``position_ids``, then per-layer ``past_key_{i}_in`` /
       ``past_value_{i}_in``.

    9. **Output order** — ``logits (B, seq_len, vocab)``, then per-layer
       KV states matching the input layer order.

    If the model is a raw HuggingFace ``PreTrainedModel`` (returns Cache
    objects, not flat tensors), wrap it in ``ONNXExportableModuleWithCache``
    first.

    Tokenizer Contract
    ------------------
    Must expose ``.eos_token_id`` (used as pad value for input_ids).
    """

    _is_stateful = False

    def __init__(
        self,
        model,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int | list[int],
        context_length: int,
        config: Union[PretrainedConfig | None] = None,
        attention_mask_min: int = -100,
        sim_collection=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        if isinstance(sequence_length, list):
            self.sequence_lengths = sorted(sequence_length)
        else:
            self.sequence_lengths = [sequence_length]
        self.context_length = context_length
        self.generation_config = None
        self._config = config
        self.attention_mask_min = attention_mask_min
        self.sim_collection = sim_collection

    @property
    def sequence_length(self) -> int:
        return self.sequence_lengths[-1]

    @staticmethod
    def select_sequence_length_from_options(num_tokens: int, options: list[int]):
        """Pick the smallest option that can fit *num_tokens*."""
        # NOTE: expects that options is pre-sorted
        best = options[-1]
        for sl in reversed(options):
            if num_tokens <= sl:
                best = sl
        return best

    def _select_sequence_length(self, num_tokens: int) -> int:
        return self.select_sequence_length_from_options(
            num_tokens, self.sequence_lengths
        )

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        if self._config is not None:
            return self._config
        return self.model.config

    @functools.cached_property
    def layer_cache_descriptors(self):
        try:
            return build_layer_cache_descriptors(_resolve_text_config(self.config))
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to build layer_cache_descriptors from config "
                f"({type(self.config).__name__}): {e}"
            ) from e

    @property
    def main_input_name(self) -> str:
        return "input_ids"

    @property
    def _supports_cache_class(self) -> bool:
        return True

    @contextlib.contextmanager
    def _optimize_model_for_decode(self):
        yield

    @property
    def device(self) -> torch.device:
        return self.model.device

    @contextlib.contextmanager
    def fp_mode(self):
        """Context manager that temporarily disables all quantizers.

        Override in framework-specific mixins (TorchFPModeMixin, ONNXFPModeMixin)
        to provide real implementations. The base implementation is a no-op.
        """
        yield

    @contextlib.contextmanager
    def on_device(self, device):
        """Context manager that temporarily places all models on the given device.

        Override in framework-specific mixins (TorchDevicePlacementMixin) to
        provide a real implementation. The base implementation is a no-op.
        """
        yield

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | DynamicCache | None]:
        """
        Overridden prepare_inputs_for_generation function to enable Huggingface generate() on models with static
        graph constraints
        """

        # We need a way to ensure that all the previous tokens that have already been consumed are stripped out of the
        # input ids

        # If past_key_values is None, this indicates that this `prepare_inputs_for_generation()` is being called for
        # the first time, and nothing should be stripped out of `input_ids`. In other cases though, the number of tokens
        # already inside `past_key_values` indicates how many tokens should be stripped out of `input_ids`

        # Notes: `input_ids`, `attention_mask`, `past_key_values` should NOT have static shape requirements imposed on
        # them by the time they reach this function. That is, in order for this to work, the static shape padding and
        # truncation must happen directly in the model `forward` function

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        num_processed_tokens = (
            past_key_values.get_seq_length()
            if not isinstance(past_key_values, tuple)
            else past_key_values[0][1].shape[-2]
        )

        inputs = (
            {"input_ids": input_ids[:, num_processed_tokens:]}
            if input_ids is not None
            else {"inputs_embeds": inputs_embeds[:, num_processed_tokens:, :]}
        )
        # Forward VLM-specific kwargs so they reach VLM_Generator.forward().
        # Do NOT forward all kwargs — HF's generate adds internal keys like
        # cache_position that would be misinterpreted as extra model inputs.
        # Only forward modality data (image/video/audio) during prefill (multiple
        # tokens remaining). During autoregressive decode only a single new token
        # is passed and it is never a modality placeholder token, so re-running an
        # input encoder would be wasteful.
        # This handles multi-turn chat: a new generate() call with prior KV cache
        # still prefills multiple new tokens (including new images/audio).
        # The key set is owned by the component registry so a new modality does
        # not need an edit here.
        _VLM_KEYS = prefill_only_keys()
        remaining_input = inputs.get("input_ids", inputs.get("inputs_embeds"))
        is_prefill = remaining_input.shape[1] > 1
        vlm_kwargs = (
            {k: v for k, v in kwargs.items() if k in _VLM_KEYS} if is_prefill else {}
        )

        return (
            inputs
            | {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
            | vlm_kwargs
        )

    @staticmethod
    def slice_inputs_for_inference(
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None], None, None
    ]:
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            position_ids_slice = (
                position_ids[..., max(0, idx - sequence_length) : idx]
                if position_ids is not None
                else None
            )
            yield (
                inputs[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
                position_ids_slice,
                kwargs,
            )

    @classmethod
    def prepare_inputs(
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
        attention_mask_min: int = -100,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        layer_cache_descriptors: list | None = None,
        **kwargs,
    ) -> OrderedDict[str, torch.Tensor]:
        """Prepare provided inputs for model forward pass with static graph constraints.

        Returns an OrderedDict with string keys aligned to ONNX input names.
        Callers that need a flat tuple for model invocation should use
        ``model(*prepared.values())``.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if layer_cache_descriptors is None:
            if not hasattr(model, "config"):
                raise ValueError(
                    "Model config is required to build layer cache descriptors if they are not provided directly."
                )
            layer_cache_descriptors = build_layer_cache_descriptors(model.config)

        input_tokens = input_ids if input_ids is not None else inputs_embeds
        input_tokens = input_tokens.to(
            dtype=torch.int32 if input_ids is not None else model.dtype
        )

        device = input_tokens.device
        batch_size = input_tokens.shape[0]
        input_length = input_tokens.shape[1]

        # Create attention mask if one is not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=device,
            )

        # Pad input tokens and attention mask to [batch_size, sequence_length] (necessary for static shape requirements)
        if input_ids is not None:
            input_tokens_extension = torch.full(
                (batch_size, sequence_length - input_length),
                fill_value=pad_token,
                dtype=input_tokens.dtype,
                device=device,
            )
        else:
            embedding_dim = input_tokens.shape[2]
            input_tokens_extension = torch.zeros(
                (batch_size, sequence_length - input_length, embedding_dim),
                dtype=input_tokens.dtype,
                device=device,
            )

        padded_input_tokens = torch.cat((input_tokens_extension, input_tokens), dim=1)
        padded_attention_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, sequence_length - input_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

        # Create dummy KV cache / recurrent state per layer
        dummy_past_key_values = []
        for desc in layer_cache_descriptors:
            shape_a, shape_b = desc.dummy_state_shapes(
                batch_size, context_length, sequence_length
            )
            dummy_past_key_values.append(torch.zeros(shape_a, device=device))
            dummy_past_key_values.append(torch.zeros(shape_b, device=device))

        # Determine current KV cache length (skip linear attention, sliding window attention layers)
        current_key_value_length = 0
        if past_key_values and len(past_key_values) > 0:
            for desc in layer_cache_descriptors:
                if desc.attention_type == AttentionType.FULL:
                    state_idx = desc.layer_idx * 2
                    if state_idx < len(past_key_values):
                        current_key_value_length = past_key_values[state_idx].shape[-2]
                    break
            else:
                current_key_value_length = past_key_values[0].shape[-2]
        key_value_padding_length = (
            context_length - sequence_length
        ) - current_key_value_length

        # Join input past_key_values with dummy_past_key_values, and clip all padding values that go over the max context
        padded_past_key_values = get_past_keyval_with_shift(
            past_key_vals=dummy_past_key_values,
            new_key_vals=past_key_values,
            length=context_length - sequence_length,
            device=device,
            dtype=model.dtype,
            layer_cache_descriptors=layer_cache_descriptors,
        )

        # Mask out dummy entries in KV cache
        kv_cache_attention_mask = torch.cat(
            (
                torch.zeros((batch_size, key_value_padding_length)),
                torch.ones((batch_size, current_key_value_length)),
            ),
            dim=-1,
        ).to(device=device)
        padded_attention_mask = torch.cat(
            (kv_cache_attention_mask, padded_attention_mask), dim=-1
        )

        # Convert attention mask from 2D to 4D and clip values
        cm_attention_mask = convert_2d_attention_mask_to_4d(
            padded_attention_mask, sequence_length, context_length
        )
        cm_attention_mask = cm_attention_mask.clip(attention_mask_min, 0)

        has_sliding_window = False
        for desc in layer_cache_descriptors:
            if desc.attention_type == AttentionType.SLIDING_WINDOW:
                cm_sliding_attention_mask = (
                    convert_2d_attention_mask_to_4d_sliding_window(
                        padded_attention_mask,
                        sequence_length,
                        context_length,
                        desc.sliding_window_size,
                    )
                )
                cm_sliding_attention_mask = cm_sliding_attention_mask.clip(
                    attention_mask_min, 0
                )
                has_sliding_window = True
                break

        # Compute or pad position_ids
        if position_ids is None:
            # Compute position_ids from attention mask
            position_ids = (
                torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
            )
            position_ids = position_ids.clip(0, context_length - 1)
            position_ids = position_ids[:, -sequence_length:]
        else:
            # Pad provided position_ids to sequence_length (pad in last dimension)
            padding_length = sequence_length - position_ids.shape[-1]
            if padding_length > 0:
                # Build padding shape dynamically - works for both 2D and 3D tensors
                pad_shape = list(position_ids.shape)
                pad_shape[-1] = padding_length
                position_ids_padding = torch.zeros(
                    pad_shape,
                    dtype=position_ids.dtype,
                    device=device,
                )
                position_ids = torch.cat((position_ids_padding, position_ids), dim=-1)

        # Build ordered dict with string keys aligned to ONNX input names
        has_full_attention = any(
            d.attention_type == AttentionType.FULL for d in layer_cache_descriptors
        )
        input_key = "inputs_embeds" if input_ids is None else "input_ids"
        prepared = OrderedDict()
        prepared[input_key] = padded_input_tokens
        if has_sliding_window and has_full_attention:
            prepared["attention_mask_full"] = cm_attention_mask.to(dtype=model.dtype)
            prepared["attention_mask_sliding_window"] = cm_sliding_attention_mask.to(
                dtype=model.dtype
            )
        elif has_sliding_window:
            prepared["attention_mask_sliding_window"] = cm_sliding_attention_mask.to(
                dtype=model.dtype
            )
        else:
            prepared["attention_mask"] = cm_attention_mask.to(dtype=model.dtype)
        prepared["position_ids"] = position_ids
        for i, desc in enumerate(layer_cache_descriptors):
            li = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                prepared[f"conv_state_{li}_in"] = padded_past_key_values[i * 2]
                prepared[f"recurrent_state_{li}_in"] = padded_past_key_values[i * 2 + 1]
            else:
                prepared[f"past_key_{li}_in"] = padded_past_key_values[i * 2]
                prepared[f"past_value_{li}_in"] = padded_past_key_values[i * 2 + 1]

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v

        return prepared

    @staticmethod
    def _output_names_from_descriptors(
        layer_cache_descriptors: list,
    ) -> list[str]:
        names = ["logits"]
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                names += [f"conv_state_{i}_out", f"recurrent_state_{i}_out"]
            else:
                names += [f"past_key_{i}_out", f"past_value_{i}_out"]
        return names

    def parse_model_outputs(
        self,
        raw_outputs: tuple[torch.Tensor, ...],
    ) -> OrderedDict[str, torch.Tensor]:
        names = self._output_names_from_descriptors(self.layer_cache_descriptors)
        return OrderedDict(zip(names, raw_outputs))

    def combine_local_and_global_outputs(
        self,
        num_valid_input_tokens: int,
        local_outputs: OrderedDict[str, torch.Tensor],
        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]],
    ):
        # strip logits corresponding to padding tokens
        local_logits = local_outputs["logits"]
        local_logits = torch.narrow(
            local_logits,
            1,
            local_logits.shape[1] - num_valid_input_tokens,
            num_valid_input_tokens,
        )

        # concatenate logits from local inference to global output
        global_outputs["logits"] = (
            torch.cat((global_outputs["logits"], local_logits), dim=1)
            if "logits" in global_outputs
            else local_logits
        )

        # Extract KV tensors from the dict in order
        local_kv_list = [v for k, v in local_outputs.items() if k != "logits"]

        # strip KV cache / recurrent state corresponding to padding tokens
        local_past_key_values = get_past_keyval_with_shift(
            past_key_vals=[],
            new_key_vals=local_kv_list,
            length=num_valid_input_tokens,
            device=self.device,
            layer_cache_descriptors=self.layer_cache_descriptors,
        )

        # shift global KV cache, concatenate local KV cache
        # For linear attention layers, the state is simply replaced.
        current_key_value_length = 0
        for desc in self.layer_cache_descriptors:
            if (
                desc.attention_type == AttentionType.FULL
                and global_outputs["past_key_values"]
            ):
                current_key_value_length = global_outputs["past_key_values"][
                    desc.layer_idx * 2
                ].shape[-2]
                break

        global_outputs["past_key_values"] = get_past_keyval_with_shift(
            past_key_vals=global_outputs["past_key_values"],
            new_key_vals=local_past_key_values,
            length=min(
                current_key_value_length + num_valid_input_tokens,
                self.context_length,
            ),
            device=self.device,
            layer_cache_descriptors=self.layer_cache_descriptors,
        )

    def _init_global_outputs(
        self, past_key_values: DynamicCache | None
    ) -> dict[str, Any]:
        return {"past_key_values": _flatten_past_key_values(past_key_values)}

    def _extra_prepare_kwargs(self, global_outputs: dict) -> dict[str, Any]:
        return {}

    def _wrap_kv_cache(
        self,
        past_key_values_list: list[torch.Tensor],
        global_outputs: dict,
    ) -> Cache | _FlatListCache:
        # Convert KV Cache outputs into a cache object compatible with HF's
        # generation loop.  For hybrid models (linear + full attention) we use
        # a lightweight wrapper so that ``get_seq_length()`` queries only the
        # full-attention layers.
        has_linear = any(
            d.attention_type == AttentionType.LINEAR
            for d in self.layer_cache_descriptors
        )
        if has_linear:
            return _FlatListCache(past_key_values_list, self.layer_cache_descriptors)
        past_key_values = DynamicCache()
        keys = past_key_values_list[::2]
        values = past_key_values_list[1::2]
        for layer_idx, (k, v) in enumerate(zip(keys, values)):
            past_key_values.update(k, v, layer_idx=layer_idx)
        return past_key_values

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        global_outputs = self._init_global_outputs(past_key_values)

        selected_seq_len = self._select_sequence_length(input_tokens.shape[1])
        for (
            input_slice,
            attention_mask_slice,
            position_ids_slice,
            kwargs_slice,
        ) in self.slice_inputs_for_inference(
            input_tokens, attention_mask, selected_seq_len, position_ids, **kwargs
        ):
            prepared_inputs = self.prepare_inputs(
                model=self.model,
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=global_outputs["past_key_values"],
                sequence_length=selected_seq_len,
                context_length=self.context_length,
                pad_token=getattr(self.tokenizer, "eos_token_id", 0),
                attention_mask_min=self.attention_mask_min,
                inputs_embeds=input_slice if inputs_embeds is not None else None,
                position_ids=position_ids_slice,
                layer_cache_descriptors=self.layer_cache_descriptors,
                **self._extra_prepare_kwargs(global_outputs),
                **kwargs_slice,
            )

            raw_outputs = self.model(*prepared_inputs.values())
            local_outputs = self.parse_model_outputs(raw_outputs)

            self.combine_local_and_global_outputs(
                input_slice.shape[1],
                local_outputs,
                global_outputs,
            )

        # make sure all outputs are on the correct device
        # the underlying mock_torch_onnx_inference function does not necessarily move outputs back to CUDA
        assert isinstance(global_outputs["logits"], torch.Tensor)
        logits = global_outputs["logits"].to(device=self.device)
        past_key_values_list = list(
            map(
                lambda tensor: tensor.to(device=self.device),
                global_outputs["past_key_values"],
            )
        )

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=self._wrap_kv_cache(past_key_values_list, global_outputs),
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[OrderedDict[str, torch.Tensor], None, None]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        preconsumed_outputs = self._init_global_outputs(past_key_values)

        slices_iter = self.slice_inputs_for_inference(
            input_tokens,
            attention_mask,
            self.sequence_length,  # always use max sequence length for slicing in prefill
            position_ids,
            **kwargs,
        )

        # Get first slice
        current_slice = next(slices_iter, None)
        if current_slice is None:
            return

        # Iterate with lookahead - process current while peeking at next
        for next_slice in slices_iter:
            input_slice, attention_mask_slice, position_ids_slice, kwargs_slice = (
                current_slice
            )
            prepared_inputs = self.prepare_inputs(
                model=self.model,
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=preconsumed_outputs["past_key_values"],
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                pad_token=getattr(self.tokenizer, "eos_token_id", 0),
                attention_mask_min=self.attention_mask_min,
                inputs_embeds=input_slice if inputs_embeds is not None else None,
                position_ids=position_ids_slice,
                layer_cache_descriptors=self.layer_cache_descriptors,
                **self._extra_prepare_kwargs(preconsumed_outputs),
                **kwargs_slice,
            )

            yield prepared_inputs

            raw_outputs = self.model(*prepared_inputs.values())
            local_outputs = self.parse_model_outputs(raw_outputs)
            self.combine_local_and_global_outputs(
                input_slice.shape[1],
                local_outputs,
                preconsumed_outputs,
            )

            current_slice = next_slice

        # current_slice is now the last - no model inference
        input_slice, attention_mask_slice, position_ids_slice, kwargs_slice = (
            current_slice
        )
        prefilled_inputs = self.prepare_inputs(
            model=self.model,
            input_ids=input_slice if input_ids is not None else None,
            attention_mask=attention_mask_slice,
            past_key_values=preconsumed_outputs["past_key_values"],
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            pad_token=getattr(self.tokenizer, "eos_token_id", 0),
            attention_mask_min=self.attention_mask_min,
            inputs_embeds=input_slice if inputs_embeds is not None else None,
            position_ids=position_ids_slice,
            layer_cache_descriptors=self.layer_cache_descriptors,
            **self._extra_prepare_kwargs(preconsumed_outputs),
            **kwargs_slice,
        )
        yield prefilled_inputs


class PrecomputedCosSinGeneratorMixin:
    """Generator mixin that replaces position_ids with precomputed RoPE (cos, sin).

    Operates on the dict returned by ``prepare_inputs``: removes
    ``position_ids`` and inserts ``position_ids_cos`` / ``position_ids_sin``.
    """

    @classmethod
    def prepare_inputs(cls, **kwargs) -> OrderedDict[str, torch.Tensor]:
        prepared = super().prepare_inputs(**kwargs)

        model = kwargs["model"]
        context_length = kwargs["context_length"]
        position_ids = prepared["position_ids"]

        if hasattr(model, "rope_embedding") and isinstance(
            model.rope_embedding, RopeEmbeddingProtocol
        ):
            embedding = model.rope_embedding
        else:
            embedding = RopeEmbedding(model=model, context_length=context_length)

        cos, sin = embedding.get_embedding(position_ids)

        return ordered_dict_replace(
            prepared,
            "position_ids",
            [
                ("position_ids_cos", cos),
                ("position_ids_sin", sin),
            ],
        )


class TransposedKVGeneratorMixin:
    """Generator mixin that permutes KV cache between HF and Hub tensor layouts.

    HF format:  keys (B, H, S, D), values (B, H, S, D)
    Hub format: keys (H, B, D, S), values (H, B, S, D)

    Operates on the dict returned by ``prepare_inputs`` by permuting all
    entries whose keys match ``past_key_*`` or ``past_value_*``.
    """

    _HF_TO_HUB_K = (1, 0, 3, 2)
    _HF_TO_HUB_V = (1, 0, 2, 3)
    _HUB_TO_HF_K = (1, 0, 3, 2)
    _HUB_TO_HF_V = (1, 0, 2, 3)

    @classmethod
    def prepare_inputs(cls, **kwargs) -> OrderedDict[str, torch.Tensor]:
        prepared = super().prepare_inputs(**kwargs)

        for key in list(prepared.keys()):
            if key.startswith("past_key_"):
                prepared[key] = prepared[key].permute(*cls._HF_TO_HUB_K)
            elif key.startswith("past_value_"):
                prepared[key] = prepared[key].permute(*cls._HF_TO_HUB_V)
        return prepared

    def parse_model_outputs(
        self,
        raw_outputs: tuple[torch.Tensor, ...],
    ) -> OrderedDict[str, torch.Tensor]:
        parsed = super().parse_model_outputs(raw_outputs)

        for key in list(parsed.keys()):
            if key.startswith("past_key_"):
                parsed[key] = parsed[key].permute(*self._HUB_TO_HF_K)
            elif key.startswith("past_value_"):
                parsed[key] = parsed[key].permute(*self._HUB_TO_HF_V)
        return parsed


class HubCompatibleGenerator(
    PrecomputedCosSinGeneratorMixin, TransposedKVGeneratorMixin, Generator
):
    """Generator for AI Hub Models checkpoints.

    Composes PrecomputedCosSinGeneratorMixin (RoPE cos/sin injection) and
    TransposedKVGeneratorMixin (KV cache permutation) with the base Generator.
    """

    pass


class VLM_Generator(Generator):
    def __init__(
        self,
        backbone_model,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int | list[int],
        context_length: int,
        vision_model=None,
        embedding=None,
        position_id_processor=None,
        config: Union[PretrainedConfig | None] = None,
        attention_mask_min: int = -100,
        visual_output_names: tuple[str, ...] = ("image_embeddings",),
        image_size: tuple[int, int] | None = None,
        audio_model=None,
        audio_output_names: tuple[str, ...] = AUDIO.default_output_names,
        audio_frames: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=backbone_model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            config=config,
            attention_mask_min=attention_mask_min,
            *args,
            **kwargs,
        )

        self.vision_model = vision_model
        self.audio_model = audio_model
        self.embedding = embedding
        self.image_size = image_size
        self.audio_frames = audio_frames
        self.audio_output_names = audio_output_names
        # Name of the component whose inputs prefill should yield instead of the
        # backbone's, or None for normal operation. See
        # :meth:`component_quantization_mode`.
        self._quantization_mode: str | None = None
        self.position_id_processor = (
            types.MethodType(position_id_processor, self)
            if position_id_processor
            else None
        )
        self.visual_output_names = visual_output_names
        if self.sim_collection is not None and position_id_processor is None:
            proc = getattr(self.sim_collection, "position_id_processor", None)
            if proc is not None:
                self.position_id_processor = types.MethodType(proc, self)

    # ---- component plumbing -------------------------------------------------
    def component_model(self, component: str):
        """The encoder module for a component, or ``None`` if this model lacks it."""
        return getattr(self, spec(require_component(component)).model_attr, None)

    def has_component(self, component: str) -> bool:
        return self.component_model(component) is not None

    def _prefill_component(
        self, component: str, **modality_kwargs
    ) -> typing.Generator[OrderedDict[str, torch.Tensor], None, None]:
        """Yield one encoder's input tuples instead of the backbone's.

        Every modality kwarg is forwarded; each hook absorbs what it does not
        take, so a new component needs only a ``prefill_hook`` on its spec.
        """
        yield from getattr(self, spec(require_component(component)).prefill_hook)(
            **modality_kwargs
        )

    def fit_audio_waveform(self, waveform, audio_frames: int):
        """Pad/truncate a raw waveform to land within ``audio_frames`` mel frames.

        Pads with real silence before the processor runs, so the extractor
        computes genuine silence mels. Zero-padding the extracted mels instead
        gives the encoder values outside its input space, and since the mask marks
        every frame valid it transcribes them -- that was the torch/ONNX WER gap.

        Only an estimate: the waveform -> frame ratio is the processor's own.
        """
        feature_extractor = getattr(self.tokenizer, "feature_extractor", None)
        hop_length = getattr(feature_extractor, "hop_length", DEFAULT_AUDIO_HOP_LENGTH)
        margin_frames = 4
        target_samples = max(0, audio_frames - margin_frames) * hop_length
        num_samples = len(waveform)
        if num_samples > target_samples:
            return waveform[:target_samples]
        if num_samples < target_samples:
            pad = target_samples - num_samples
            if isinstance(waveform, torch.Tensor):
                return torch.nn.functional.pad(waveform, (0, pad))
            return np.pad(waveform, (0, pad))
        return waveform

    def pad_audio_item(self, item: dict, audio_frames: int) -> dict:
        """Pad ``input_features``/``input_features_mask`` to a fixed mel-frame
        count, for a statically-shaped (ONNX) export.

        Deliberately unimplemented: ``input_features`` has no agreed layout
        (Qwen3-ASR ``[B, mel, time]``, Gemma4 ``[B, time, mel]``) and padding the
        wrong axis fails only inside onnxruntime, so each model states its own.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has an audio component but does not "
            "implement pad_audio_item, so the mel-frame axis of its "
            "input_features is unknown. Implement it on this Generator subclass."
        )

    def present_components(self) -> tuple[str, ...]:
        """Live input-encoder components, in canonical registry order."""
        return tuple(n for n in COMPONENTS if self.has_component(n))

    @contextlib.contextmanager
    def component_quantization_mode(self, component: str):
        """Rewire prefill to yield one component's encoder inputs.

        When active, :meth:`prefill` yields the input tuples that would normally
        be passed to that component's encoder instead of the backbone's inputs.
        This lets recipes like Calibration and SeqMSE operate on an encoder
        without any changes to their own code.
        """
        require_component(component)
        previous = self._quantization_mode
        self._quantization_mode = component
        try:
            yield
        finally:
            self._quantization_mode = previous

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # 1) Convert input_ids to input embeddings using self.embedding
        inputs_embeds = self.embedding(input_ids)
        extra_kwargs = {}

        # 2) Process each image individually through the vision model.
        #    This aligns with ONNX export (fixed single-image input shape)
        #    and on-target deployment (one vision encoder call per image).
        image_mask_3d = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_mask_2d = self._build_image_mask_2d(input_ids, pixel_values)

        if pixel_values is not None and image_grid_thw is not None:
            # Split pixel_values into per-image chunks and process individually
            per_image_sizes = image_grid_thw.prod(-1).tolist()
            per_image_pixels = torch.split(pixel_values, per_image_sizes, dim=0)

            all_embeddings = []
            all_extra = {name: [] for name in self.visual_output_names[1:]}

            for pixels_i, grid_i in zip(per_image_pixels, image_grid_thw):
                vision_output = self.vision_model(
                    pixel_values=pixels_i,
                    image_grid_thw=grid_i.unsqueeze(0),
                    mask=image_mask_2d,
                )

                if isinstance(vision_output, tuple):
                    all_embeddings.append(vision_output[0])
                    for name, value in zip(
                        self.visual_output_names[1:], vision_output[1:]
                    ):
                        all_extra[name].append(value)
                else:
                    all_embeddings.append(vision_output)

            image_embeddings = torch.cat(all_embeddings, dim=0)

            # Merge per-image extra outputs:
            # - Tensors: take the first (e.g. visual_pos_masks is identical across images)
            # - Lists of tensors: concatenate per-layer across images
            for name, vals in all_extra.items():
                if not vals:
                    continue
                if isinstance(vals[0], torch.Tensor):
                    extra_kwargs[name] = vals[0]
                elif isinstance(vals[0], list):
                    extra_kwargs[name] = [
                        torch.cat(per_layer, dim=0) for per_layer in zip(*vals)
                    ]
                else:
                    extra_kwargs[name] = vals
        elif pixel_values is not None:
            vision_output = self.vision_model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mask=image_mask_2d,
            )
            if isinstance(vision_output, tuple):
                image_embeddings = vision_output[0]
                for name, value in zip(self.visual_output_names[1:], vision_output[1:]):
                    extra_kwargs[name] = value
            else:
                image_embeddings = vision_output
        else:
            image_embeddings = None

        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_embeddings
            )

        if pixel_values_videos is not None or video_grid_thw is not None:
            raise RuntimeError("No support for video yet.")

        if "mm_token_type_ids" in extra_kwargs:
            mm_token_type_ids = extra_kwargs.pop("mm_token_type_ids")
        elif pixel_values is None and pixel_values_videos is None:
            # No actual vision data — treat all tokens as text so that
            # get_rope_index does not try to consume image_grid_thw entries.
            mm_token_type_ids = torch.zeros_like(input_ids)
        else:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            mm_token_type_ids[input_ids == self.config.video_token_id] = 2

        return inputs_embeds, mm_token_type_ids, extra_kwargs

    def fuse_audio(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter audio-encoder embeddings into the token sequence.

        Structurally identical to the image scatter: the encoder returns
        embeddings already at the post-merge audio-token count, and they are
        written into ``inputs_embeds`` at every ``audio_token_id`` position.
        The mel/feature extraction stays in the HF processor, outside this graph.
        """
        if input_features is None:
            return inputs_embeds

        audio_output = self.audio_model(
            input_features=input_features,
            input_features_mask=input_features_mask,
        )
        audio_embeddings = (
            audio_output[0] if isinstance(audio_output, tuple) else audio_output
        )
        audio_embeddings = audio_embeddings.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        audio_mask = (
            (input_ids == self.config.audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        return inputs_embeds.masked_scatter(audio_mask, audio_embeddings)

    def fuse_multimodal(
        self,
        input_ids: torch.Tensor | None = None,
        **modality_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Fuse every present input encoder's embeddings into the sequence.

        Vision keeps its own :meth:`fuse_text_image_video` (model-specific, and
        widely overridden); audio scatters on top. An audio-only model never
        touches the vision path, so it never reads the ``image_token_id`` it lacks.

        ``mm_token_type_ids`` is not annotated for audio: 1=image/2=video are read
        by position-id processors, and a third value would change their behaviour.
        """
        if self.has_component(VISUAL.name):
            inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
                input_ids=input_ids,
                pixel_values=modality_kwargs.get("pixel_values"),
                pixel_values_videos=modality_kwargs.get("pixel_values_videos"),
                image_grid_thw=modality_kwargs.get("image_grid_thw"),
                video_grid_thw=modality_kwargs.get("video_grid_thw"),
            )
        else:
            inputs_embeds = self.embedding(input_ids)
            mm_token_type_ids = torch.zeros_like(input_ids)
            extra_kwargs = {}

        if self.has_component(AUDIO.name):
            inputs_embeds = self.fuse_audio(
                inputs_embeds,
                input_ids,
                input_features=modality_kwargs.get("input_features"),
                input_features_mask=modality_kwargs.get("input_features_mask"),
            )

        return inputs_embeds, mm_token_type_ids, extra_kwargs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Remove mm_token_type_ids from kwargs — it is consumed by
        # fuse_text_image_video / position_id_processor and must not
        # propagate to the backbone model.
        kwargs.pop("mm_token_type_ids", None)

        # 1) Obtain fused input embeddings and extra encoder outputs
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            input_features_mask=input_features_mask,
        )

        # 2) Process position_ids through position processor
        position_ids = (
            self.position_id_processor(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            if self.position_id_processor is not None
            else None
        )

        # 3) call super().forward() with concatenated embeddings and extra encoder outputs
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **{**kwargs, **extra_kwargs},
        )

    def _build_image_mask_2d(self, input_ids, pixel_values):
        """Build the 2D image mask padded to sequence_length.

        Shared by :meth:`fuse_text_image_video` and the visual-quantization
        path in :meth:`prefill`.
        """
        if input_ids is not None:
            image_mask_2d = (input_ids == self.config.image_token_id).to(
                input_ids.device
            )
        else:
            device = pixel_values.device if pixel_values is not None else "cpu"
            image_mask_2d = torch.zeros(
                1, self.sequence_length, dtype=torch.bool, device=device
            )
        pad_len = self.sequence_length - image_mask_2d.shape[1]
        if pad_len > 0:
            image_mask_2d = torch.nn.functional.pad(
                image_mask_2d, (0, pad_len), value=False
            )
        return image_mask_2d

    def _prefill_visual(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[OrderedDict[str, torch.Tensor], None, None]:
        """Yield per-image vision model input tuples.

        Each yielded tuple contains the arguments that would be passed to
        ``self.vision_model()`` for a single image: ``(pixel_values_i,
        image_grid_thw_i, mask)``.  This is the visual-quantization
        counterpart of the normal :meth:`prefill` path.
        """
        image_mask_2d = self._build_image_mask_2d(input_ids, pixel_values)

        if pixel_values is not None and image_grid_thw is not None:
            per_image_sizes = image_grid_thw.prod(-1).tolist()
            per_image_pixels = torch.split(pixel_values, per_image_sizes, dim=0)
            for pixels_i, grid_i in zip(per_image_pixels, image_grid_thw):
                yield OrderedDict(
                    [
                        ("pixel_values", pixels_i),
                        ("image_grid_thw", grid_i.unsqueeze(0)),
                        ("mask", image_mask_2d),
                    ]
                )
        elif pixel_values is not None:
            yield OrderedDict(
                [
                    ("pixel_values", pixel_values),
                    ("image_grid_thw", image_grid_thw),
                    ("mask", image_mask_2d),
                ]
            )

    def _prefill_audio(
        self,
        input_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[OrderedDict[str, torch.Tensor], None, None]:
        """Yield audio-encoder input tuples.

        The audio-quantization counterpart of :meth:`_prefill_visual`. Unlike
        images there is no per-item split: one utterance is one encoder call, so
        this yields at most one tuple per sample.
        """
        if input_features is None:
            return
        yield OrderedDict(
            [
                ("input_features", input_features),
                ("input_features_mask", input_features_mask),
            ]
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[OrderedDict[str, torch.Tensor], None, None]:
        if self._quantization_mode:
            yield from self._prefill_component(
                self._quantization_mode,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_features=input_features,
                input_features_mask=input_features_mask,
                **kwargs,
            )
            return

        # 1) Obtain fused input embeddings and extra encoder outputs
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            input_features_mask=input_features_mask,
        )

        # 2) Process position_ids through position processor
        position_ids = (
            self.position_id_processor(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            if self.position_id_processor is not None
            else None
        )

        # 3) call super().prefill() with concatenated embeddings and extra vision outputs
        yield from super().prefill(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **{**kwargs, **extra_kwargs},
        )
