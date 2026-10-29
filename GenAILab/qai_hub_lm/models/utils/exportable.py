# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNXExportableModuleWithCache — wrapper enabling ONNX export of HF models."""

import torch
from transformers import PreTrainedModel, DynamicCache

from .compat import _patch_sdpa_mask  # noqa: F401 — triggers the patch on import
from .compat import linear_attention_states
from .layer_cache import (
    PAST_POSITION_IDS,
    PLE_CONV_STATE_IDX,
    AttentionType,
    all_states,
    build_layer_cache_descriptors,
    _resolve_text_config,
)


class ONNXExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models
    that produce and consume Cache objects. Supports both LLM and VLM backbones.
    """

    _KNOWN_PREFIXES = (
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "past_key_",
        "past_value_",
        "conv_state_",
        "past_indexer_key_",
        "recurrent_state_",
        "ple_conv_state_",
        PAST_POSITION_IDS,
    )

    def __init__(
        self,
        model: PreTrainedModel,
        lm_head: torch.nn.Module | None = None,
        cache_type: type = DynamicCache,
        input_names: tuple[str, ...] = (),
    ):
        """
        :param model: The HuggingFace model to wrap
        :param lm_head: Optional LM head (for VLM backbones where head is separate)
        :param cache_type: Cache class to construct from flattened KV pairs.
            Defaults to ``DynamicCache``. Models with hybrid attention (e.g.
            mixing full attention with linear/recurrent layers) can pass
            ``HybridCache`` or another cache class.
        :param input_names: ONNX input names matching the positional arg layout.
            ``forward`` reconstructs a name→tensor dict from ``*args`` and
            parses by name pattern.
        """
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.cache_type = cache_type
        if not input_names:
            input_names = self._default_input_names()
        self.input_names = tuple(input_names)
        self._restore_rotary_buffers_to_fp32()

    def _restore_rotary_buffers_to_fp32(self) -> None:
        """Restore ``inv_freq`` (and any ``original_inv_freq``) buffers to fp32.

        ONNX ``Range`` (opset 11+) accepts only float/double/int16/int32/int64.
        HF rotary layers do ``torch.arange(seqlen, dtype=self.inv_freq.dtype)``,
        so if ``inv_freq`` was cast to fp16 by ``model.to(dtype=torch.float16)``
        the exported ``Range`` node has fp16 inputs and the graph fails schema
        validation at load time.
        """
        for module in self.model.modules():
            for name in ("inv_freq", "original_inv_freq"):
                buf = getattr(module, name, None)
                if isinstance(buf, torch.Tensor) and buf.dtype != torch.float32:
                    module.register_buffer(
                        name, buf.to(torch.float32), persistent=False
                    )

    def _default_input_names(self) -> tuple[str, ...]:
        """Derive input names from model config when none are provided."""
        from ..base import LLM

        return LLM.get_backbone_input_names(
            build_layer_cache_descriptors(self.model.config)
        )

    @property
    def use_inputs_embeds(self) -> bool:
        return "inputs_embeds" in self.input_names

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        return tuple(
            n
            for n in self.input_names
            if not any(n.startswith(p) for p in self._KNOWN_PREFIXES)
        )

    @property
    def device(self):
        """Return model device"""
        return self.model.device

    @property
    def dtype(self):
        """Return model dtype"""
        return self.model.dtype

    @property
    def config(self):
        """Return model config"""
        return self.model.config

    def _build_cache(self, states: dict[str, torch.Tensor]):
        """Build a cache object from the carried state inputs, keyed by state name."""
        resolved = _resolve_text_config(self.config)
        descriptors = build_layer_cache_descriptors(resolved)
        has_linear = any(d.attention_type == AttentionType.LINEAR for d in descriptors)

        # Hybrid (linear + full) models need a config-constructed cache so the
        # per-layer cache objects are created with the right type — the linear
        # layers expose conv/recurrent state slots that ``update_conv_state`` /
        # ``update_recurrent_state`` write into, and QSA layers an indexer key
        # slot. Pure attention models keep the config-less ``DynamicCache``:
        # passing config there would create DynamicSlidingWindowLayers that clip
        # KV internally, whereas our 4D attention mask already handles windowing
        # and we need uniform caches.
        needs_config = has_linear or any(d.indexer_head_dim for d in descriptors)
        if self.cache_type is DynamicCache and not needs_config:
            kv_cache = DynamicCache()
        else:
            kv_cache = self.cache_type(config=resolved)

        if not states:
            return kv_cache

        for desc in descriptors:
            layer_states = [states[state.name] for state in desc.states]
            if desc.attention_type == AttentionType.LINEAR:
                conv_state, recurrent_state = layer_states[:2]
                kv_cache.update_conv_state(conv_state, desc.layer_idx)
                kv_cache.update_recurrent_state(recurrent_state, desc.layer_idx)
                if desc.ple_conv_shape is not None:
                    # Qwen4-Exp keeps the PLE short-conv window as conv state 1
                    kv_cache.update_conv_state(
                        layer_states[2], desc.layer_idx, state_idx=PLE_CONV_STATE_IDX
                    )
                continue
            kv_cache.update(layer_states[0], layer_states[1], desc.layer_idx, {})
            if desc.indexer_head_dim is not None:
                kv_cache.update_indexer(layer_states[2], desc.layer_idx)

        if PAST_POSITION_IDS in states:
            # The model keeps these as (3, batch, kv_length); the graph state is
            # batch-first like every other state.
            kv_cache.position_ids = states[PAST_POSITION_IDS].transpose(0, 1)
        return kv_cache

    @staticmethod
    def _parse_inputs_by_name(
        input_names: tuple[str, ...],
        args: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Reconstruct a name→tensor dict from positional args."""
        if len(args) != len(input_names):
            raise RuntimeError(
                f"Expected {len(input_names)} inputs but got {len(args)}."
            )
        return dict(zip(input_names, args))

    @staticmethod
    def _collect_indexed_extras(
        extras: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Coalesce ``deepstack_visual_embeds_0, _1, ...`` back into a list.

        TODO: replace with a generator-provided unflatten callback so the
        wrapper doesn't need model-specific knowledge.
        """
        ds_items: list[tuple[int, torch.Tensor]] = []
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        prefix = "deepstack_visual_embeds_"
        for k, v in extras.items():
            if k.startswith(prefix):
                idx = int(k[len(prefix) :])
                ds_items.append((idx, v))
            else:
                result[k] = v
        if ds_items:
            ds_items.sort(key=lambda x: x[0])
            result["deepstack_visual_embeds"] = [v for _, v in ds_items]
        # Qwen4-Exp PLE: decoder layer index -> its host-computed n-gram embeddings.
        # Keyed by int only: strict torch.export cannot sort tuples holding tensors.
        ple_keys = [k for k in result if k.startswith("ple_embeds_")]
        if ple_keys:
            result["ple_embeds"] = {
                int(k[len("ple_embeds_") :]): result.pop(k) for k in ple_keys
            }
        return result

    @staticmethod
    def _unpack_attention_mask(
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        masks = {k: v for k, v in inputs.items() if k.startswith("attention_mask")}
        if len(masks) == 1:
            return next(iter(masks.values()))

        unpacked_attention_masks = {}
        if "attention_mask_full" in masks:
            unpacked_attention_masks[AttentionType.FULL.value] = masks[
                "attention_mask_full"
            ]
        if "attention_mask_sliding_window" in masks:
            unpacked_attention_masks[AttentionType.SLIDING_WINDOW.value] = masks[
                "attention_mask_sliding_window"
            ]
        return unpacked_attention_masks if len(unpacked_attention_masks) > 0 else None

    def forward(self, *args: torch.Tensor):
        """
        Redefine model forward to convert to/from Huggingface DynamicCache objects.

        Positional ``*args`` are mapped to names via ``self.input_names`` and
        then dispatched by name pattern (mask, position, KV cache, extras).
        """
        inputs = self._parse_inputs_by_name(self.input_names, args)

        input_or_embeds = inputs.get("inputs_embeds", inputs.get("input_ids"))
        attention_mask = self._unpack_attention_mask(inputs)
        position_ids = inputs.get("position_ids")
        position_ids_cos = inputs.get("position_ids_cos")
        position_ids_sin = inputs.get("position_ids_sin")

        # No state inputs means an empty cache (e.g. a prefill-only trace);
        # otherwise every state must be present
        expected = all_states(build_layer_cache_descriptors(self.config))
        states = {
            state.name: inputs[f"{state.name}_in"]
            for state in expected
            if f"{state.name}_in" in inputs
        }
        if states and len(states) != len(expected):
            missing = [s.name for s in expected if s.name not in states]
            raise ValueError(f"Missing cache state inputs: {missing}")
        extra_kwargs = {k: v for k, v in inputs.items() if k in self.extra_input_names}
        extra_kwargs = self._collect_indexed_extras(extra_kwargs)

        kv_cache = self._build_cache(states)

        model_kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": kv_cache,
            "use_cache": True,
            "num_logits_to_return": 0,
            "return_dict": False,
            **extra_kwargs,
        }

        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids
        if position_ids_cos is not None:
            model_kwargs["position_ids_cos"] = position_ids_cos
            model_kwargs["position_ids_sin"] = position_ids_sin

        if self.use_inputs_embeds:
            model_kwargs["input_ids"] = None
            model_kwargs["inputs_embeds"] = input_or_embeds
        else:
            model_kwargs["input_ids"] = input_or_embeds

        return self._call_model_and_flatten(model_kwargs)

    def _call_model_and_flatten(self, model_kwargs):
        outputs = self.model(**model_kwargs)
        hidden_states_or_logits, new_past_key_values = outputs[0], outputs[1]

        # Apply lm_head if provided (VLM backbone case)
        if self.lm_head is not None:
            lm_logits = self.lm_head(hidden_states_or_logits)
        else:
            lm_logits = hidden_states_or_logits

        # Flatten output states, in the same order as the inputs
        flat_output_past_key_values = []
        descriptors = build_layer_cache_descriptors(self.config)
        for desc in descriptors:
            cache_layer = new_past_key_values.layers[desc.layer_idx]
            if desc.attention_type == AttentionType.LINEAR:
                # Linear attention: extract conv_state and recurrent_state.
                flat_output_past_key_values.extend(linear_attention_states(cache_layer))
                if desc.ple_conv_shape is not None:
                    flat_output_past_key_values.append(
                        cache_layer.conv_states[PLE_CONV_STATE_IDX]
                    )
                continue
            flat_output_past_key_values.append(cache_layer.keys)
            flat_output_past_key_values.append(cache_layer.values)
            if desc.indexer_head_dim is not None:
                flat_output_past_key_values.append(cache_layer.indexer_keys)
        if any(s.name == PAST_POSITION_IDS for s in all_states(descriptors)):
            flat_output_past_key_values.append(
                new_past_key_values.position_ids.transpose(0, 1)
            )

        return lm_logits, *flat_output_past_key_values
