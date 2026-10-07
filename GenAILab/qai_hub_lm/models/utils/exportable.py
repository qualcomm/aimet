# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNXExportableModuleWithCache — wrapper enabling ONNX export of HF models."""

import torch
from transformers import PreTrainedModel, DynamicCache

from .compat import _patch_sdpa_mask  # noqa: F401 — triggers the patch on import
from .compat import linear_attention_states
from .layer_cache import (
    AttentionType,
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
        "recurrent_state_",
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

    def _build_cache(self, past_key_values: tuple[tuple[torch.Tensor, ...], ...]):
        """Build a cache object from flattened state pairs using ``self.cache_type``."""
        resolved = _resolve_text_config(self.config)
        layer_types = getattr(resolved, "layer_types", None)
        has_linear = bool(layer_types) and "linear_attention" in layer_types

        # Hybrid (linear + full) models need a config-constructed cache so the
        # per-layer cache objects are created with the right type — the linear
        # layers expose conv/recurrent state slots that ``update_conv_state`` /
        # ``update_recurrent_state`` write into. Pure attention models keep the
        # config-less ``DynamicCache``: passing config there would create
        # DynamicSlidingWindowLayers that clip KV internally, whereas our 4D
        # attention mask already handles windowing and we need uniform caches.
        if self.cache_type is DynamicCache and not has_linear:
            kv_cache = DynamicCache()
        else:
            kv_cache = self.cache_type(config=resolved)

        for layer_idx, (state_a, state_b) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            if layer_types and layer_types[layer_idx] == "linear_attention":
                # Linear attention layers carry conv / recurrent states.
                kv_cache.update_conv_state(state_a, layer_idx)
                kv_cache.update_recurrent_state(state_b, layer_idx)
            else:
                kv_cache.update(state_a, state_b, layer_idx, {})
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

        kv_pairs = [
            v
            for k, v in inputs.items()
            if k.startswith(("past_key_", "past_value_", "recurrent_state_"))
        ]
        extra_kwargs = {k: v for k, v in inputs.items() if k in self.extra_input_names}
        extra_kwargs = self._collect_indexed_extras(extra_kwargs)

        kv_cache = self._build_cache(tuple(kv_pairs))

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

        # Flatten output KV cache
        flat_output_past_key_values = []
        resolved = _resolve_text_config(self.config)
        layer_types = getattr(resolved, "layer_types", None)
        for layer_idx in range(len(new_past_key_values)):
            cache_layer = new_past_key_values.layers[layer_idx]
            if layer_types and layer_types[layer_idx] == "linear_attention":
                # Linear attention: extract conv_state and recurrent_state.
                flat_output_past_key_values.extend(linear_attention_states(cache_layer))
            else:
                flat_output_past_key_values.append(cache_layer.keys)
                flat_output_past_key_values.append(cache_layer.values)

        return lm_logits, *flat_output_past_key_values
