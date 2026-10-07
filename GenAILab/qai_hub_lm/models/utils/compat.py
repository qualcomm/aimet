# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility patches and shims for HuggingFace transformers."""

import torch


def _patch_sdpa_mask():
    # In transformers >=5.3.0, _preprocess_mask_arguments derives q_length from
    # inputs_embeds.shape[1], which under torch.jit.trace yields a 0-dim tensor
    # instead of a Python int.
    # Fix: convert 0-dim tensors to int.
    #
    # ``q_length`` is only a parameter of ``sdpa_mask`` in transformers >=5.3.0;
    # in older versions (and on transformers' internal generation path) it is
    # not passed at all. So accept it as an optional keyword, normalize it when
    # present, and only forward it to the original when the original accepts it.
    try:
        import inspect

        import transformers.masking_utils as _mu

        _orig = _mu.sdpa_mask
        _orig_accepts_q_length = "q_length" in inspect.signature(_orig).parameters

        def _patched(*args, **kwargs):
            if "q_length" in kwargs:
                q_length = kwargs["q_length"]
                if isinstance(q_length, torch.Tensor) and q_length.ndim == 0:
                    q_length = q_length.item()
                if _orig_accepts_q_length:
                    kwargs["q_length"] = q_length
                else:
                    # Older signature has no q_length; drop it so the call
                    # doesn't fail with an unexpected-keyword error.
                    kwargs.pop("q_length")
            return _orig(*args, **kwargs)

        _mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _patched
        _mu.sdpa_mask = _patched
    except (ImportError, AttributeError):
        pass


# TODO: Remove this patch once the fix applied in transformers itself.
_patch_sdpa_mask()


def linear_attention_states(cache_layer) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a linear-attention cache layer's ``(conv_state, recurrent_state)``.

    Since transformers 5.14 a layer can hold several states, so ``conv_states``
    and ``recurrent_states`` are dicts keyed by state index; earlier versions
    store one tensor each. Linear-attention models use a single state, index 0.
    """
    conv_states = cache_layer.conv_states
    recurrent_states = cache_layer.recurrent_states
    if isinstance(conv_states, dict):
        return conv_states[0], recurrent_states[0]
    return conv_states, recurrent_states


class PositionIdContext:
    """Minimal stand-in for ``self`` when calling HF's unbound ``get_rope_index``.

    HF's ``get_rope_index`` is an instance method that accesses ``self.config``
    and may call sibling methods (e.g. ``self.get_vision_position_ids``).  In our
    framework the position-ID computation must work without a real HF model
    instance (e.g. in the ONNX path).  This proxy satisfies the ``self`` contract
    by holding the config and delegating any other attribute lookups to the
    original HF model *class* (bound to this proxy).
    """

    def __init__(self, config, model_cls):
        self.config = config
        self._model_cls = model_cls

    def __getattr__(self, name):
        attr = getattr(self._model_cls, name)
        if callable(attr):
            return attr.__get__(self, type(self))
        return attr
