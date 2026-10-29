# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Qwen4-Exp model mixin and generator for Torch and ONNX frameworks."""

from __future__ import annotations

import torch

from .generator import Generator
from .utils.layer_cache import LayerCacheDescriptor


class Qwen4ExpGenerator(Generator):
    """Generator that computes Qwen4-Exp's PLE n-gram embeddings on the host.

    Each PLE layer's embeddings depend on the current token and the previous
    ``ngram_size - 1``, so they are computed here from the unpadded token
    stream and fed to the graph as ``ple_embeds_{i}``. The trailing tokens a
    later call needs are kept on the returned cache as ``ple_token_history``.

    :param ngram_embeddings: The model's ``Qwen4ExpTextNGramEmbedding`` modules,
        keyed by decoder layer index (see ``Qwen4ExpMixin.get_extras``)
    """

    def __init__(self, *args, ngram_embeddings=None, **kwargs):
        super().__init__(*args, **kwargs)
        # A ModuleDict keys by str; index by decoder layer
        self._ngram_embeddings = {
            int(idx): module for idx, module in (ngram_embeddings or {}).items()
        }

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None, **kwargs
    ):
        if not self._ngram_embeddings:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if input_ids is None:
            raise NotImplementedError(
                "Qwen4-Exp PLE needs token ids; inputs_embeds-only calls are not supported."
            )
        history = getattr(past_key_values, "ple_token_history", None)
        if attention_mask is not None:
            # Only the current call's real tokens extend the n-gram history
            current_mask = attention_mask[:, -input_ids.shape[1] :].bool()
            if not current_mask.all():
                raise NotImplementedError(
                    "Qwen4-Exp PLE: padded positions inside input_ids are not supported."
                )
        with torch.no_grad():
            embeds = {
                idx: self._ngram_embed(module, input_ids, history)
                for idx, module in self._ngram_embeddings.items()
            }
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            ple_embeds=embeds,
            **kwargs,
        )
        context_len = next(iter(self._ngram_embeddings.values())).context_len
        tokens = input_ids if history is None else torch.cat([history, input_ids], 1)
        out.past_key_values.ple_token_history = tokens[:, -context_len:]
        return out

    @staticmethod
    def _ngram_embed(module, input_ids, history):
        """``module``'s embeddings for ``input_ids``, preceded by ``history`` tokens.

        Same hash and lookup as ``Qwen4ExpTextNGramEmbedding.forward``, but the
        previous tokens come from ``history`` rather than a cache layer.
        """
        context_len = module.context_len
        if history is None:
            history = input_ids.new_full(
                (input_ids.shape[0], context_len), module.eos_token_id
            )
        token_history = torch.cat([history[:, -context_len:], input_ids], dim=-1).long()
        shifted = [
            module._shift_right_ignore_eos(token_history, shift)
            for shift in range(module.ngram_size)
        ]
        blocks = []
        for ngram in range(2, module.ngram_size + 1):
            start = (ngram - 2) * module.heads_per_ngram
            end = start + module.heads_per_ngram
            mixed = shifted[0] * module.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[position] * module.layer_multipliers[position]
                )
            sizes = module.ngram_heads_vocab_sizes[start:end]
            offsets = module.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
            blocks.append(ids + offsets.view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        return module.ngram_embedding(ngram_ids).flatten(-2)

    def slice_inputs_for_inference(
        self, inputs, attention_mask, sequence_length, position_ids=None, **kwargs
    ):
        """Slice ``ple_embeds`` along with the inputs, left-padded the same way.

        The embeddings cover exactly ``inputs``, so a chunk ending ``n`` tokens
        before the end of ``inputs`` takes the same span of each embedding.
        """
        embeds = kwargs.pop("ple_embeds", None)
        consumed = 0
        for (
            input_slice,
            mask_slice,
            pos_slice,
            kw_slice,
        ) in super().slice_inputs_for_inference(
            inputs, attention_mask, sequence_length, position_ids, **kwargs
        ):
            kw_slice = dict(kw_slice)
            if embeds is not None:
                length = input_slice.shape[1]
                pad = sequence_length - length
                for idx, emb in embeds.items():
                    chunk = emb[:, consumed : consumed + length]
                    kw_slice[f"ple_embeds_{idx}"] = torch.nn.functional.pad(
                        chunk, (0, 0, pad, 0)
                    )
                consumed += length
            yield input_slice, mask_slice, pos_slice, kw_slice


def _ple_input_names(descriptors: list[LayerCacheDescriptor] | None) -> list[str]:
    """``ple_embeds_{i}`` for each PLE layer, which carries a PLE conv window."""
    return [
        f"ple_embeds_{d.layer_idx}"
        for d in descriptors or []
        if d.ple_conv_shape is not None
    ]


class Qwen4ExpMixin:
    @staticmethod
    def use_dynamo_export() -> bool:
        # Qwen4-Exp builds its attention masks with ``create_causal_mask`` and
        # ``create_recurrent_attention_mask`` (data-dependent control flow), and
        # its linear-attention layers have the same prefill-only recurrent-state
        # inputs as Qwen 3.5, which TorchScript would prune. See Qwen3_5Mixin.
        return True

    @staticmethod
    def get_generator_cls() -> type[Generator]:
        return Qwen4ExpGenerator

    @classmethod
    def get_extras(cls, model) -> dict:
        """The PLE layers' n-gram embedding modules, which run on the host."""
        layers = model.model.layers
        ngram = {
            idx: layer.ple.ple_embedding
            for idx, layer in enumerate(layers)
            if getattr(layer, "ple", None) is not None
        }
        if not ngram:
            return {}
        return {
            "ngram_embeddings": torch.nn.ModuleDict(
                {str(idx): module for idx, module in ngram.items()}
            )
        }

    @classmethod
    def get_backbone_input_names(cls, layer_cache_descriptors=None, **kwargs):
        return tuple(super().get_backbone_input_names(layer_cache_descriptors)) + tuple(
            _ple_input_names(layer_cache_descriptors)
        )

    @classmethod
    def get_backbone_dynamic_axes(cls, layer_cache_descriptors=None, **kwargs):
        axes = super().get_backbone_dynamic_axes(layer_cache_descriptors)
        for name in _ple_input_names(layer_cache_descriptors):
            axes[name] = {1: "sequence_length"}
        return axes

    @classmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length,
        sequence_length,
        layer_cache_descriptors=None,
        *args,
        **kwargs,
    ):
        names = _ple_input_names(layer_cache_descriptors)
        if not names:
            return super().get_sample_backbone_inputs(
                model,
                context_length,
                sequence_length,
                layer_cache_descriptors,
                *args,
                **kwargs,
            )
        ple_dim = model.config.ple_embed_dim
        prepared = Qwen4ExpGenerator.prepare_inputs(
            model=model,
            input_ids=torch.zeros((1, sequence_length), dtype=torch.int),
            attention_mask=torch.ones((1, sequence_length), dtype=torch.int),
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            layer_cache_descriptors=layer_cache_descriptors,
            **{
                name: torch.zeros((1, sequence_length, ple_dim), dtype=model.dtype)
                for name in names
            },
        )
        return tuple(prepared.values())
