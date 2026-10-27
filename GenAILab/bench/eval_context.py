# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation context for sharing cached results across metrics."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable

from .fp_cache import DiskBackedFPCache

if TYPE_CHECKING:
    from GenAILab.bench.yaml_config_parser import ModelConfig


class EvaluationContext:
    """Two-tier evaluation cache that wraps a :class:`DiskBackedFPCache`.

    *   **FP results** are keyed by a hash of the full model config and
        persisted to disk via the underlying :class:`DiskBackedFPCache`.  They
        are therefore shared across quantization recipes *and* across runs.
    *   **Quant results** are held in-memory only and scoped to a single
        experiment (one recipe).

    Metrics interact with this object using plain ``collection_name`` strings
    (e.g. ``"mmlu_logits"``).  The context handles key construction internally
    so that metrics never need to know about model identity, adaptations, etc.
    """

    def __init__(self, fp_cache: DiskBackedFPCache, model_config: ModelConfig):
        self._fp_cache = fp_cache
        self._model_config = model_config
        # Convert to dict for hashing and metadata (backward compatible format)
        model_dict = self._model_config_to_dict(model_config)
        self._model_hash = self._hash_dict(model_dict)
        self._model_dict = model_dict  # Keep for metadata
        self._quant_cache: dict[str, Any] = {}

    @staticmethod
    def _model_config_to_dict(config: ModelConfig) -> dict:
        """Convert ModelConfig to the dict format used for hashing and metadata."""
        return {
            "class": config.model_cls,
            "model_id": config.model_id,
            "model_type": config.model_type,
            "context_length": config.context_length,
            "sequence_length": config.sequence_length,
            "adaptations": config.adaptations,
            "image_size": config.image_size,
            "encodings": config.encodings,
            "dtype": config.dtype,
            **config.extra_kwargs,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compute_fp(self, collection_name: str, compute_fn: Callable[[], Any]):
        """Return FP results for *collection_name*, computing and caching if absent.

        Results are persisted to disk so subsequent runs that use the same model config will get a cache hit.
        """
        key = (self._model_hash, collection_name)
        return self._fp_cache.get_or_compute(key, compute_fn, metadata=self._model_dict)

    def get_or_compute_quant(self, collection_name: str, compute_fn: Callable[[], Any]):
        """Return quant results for *collection_name*, computing and caching if absent.

        Results are held in-memory only and scoped to this test invocation.
        """
        if collection_name not in self._quant_cache:
            self._quant_cache[collection_name] = compute_fn()
        return self._quant_cache[collection_name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_dict(d: dict) -> str:
        """Return a short deterministic hex digest of *d*."""
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
