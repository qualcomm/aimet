# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""On-disk cache of exported diffusion ONNX graphs.

Exporting SD1.5's UNet takes about half a minute, so each set of exported
components is kept under a key of everything that changes the graphs::

    {cache_dir}/
    +-- {key}/
        +-- key.json            # what the key hashes, for inspection
        +-- text_encoder/model.onnx (+ model.onnx.data)
        +-- unet/...
        +-- vae_decoder/...

An entry is written to a temporary directory and renamed into place once every
component has exported, so an interrupted export never leaves a partial entry.
If another process shares the cache and finished the same key first, its entry
is kept and this one is dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import torch

from GenAILab.bench.utils.cache_utils import (
    clear_dir,
    discard_entry,
    entry_is_complete,
    staged_entry,
    sweep_stale_staging,
)
from GenAILab.qai_hub_lm.diffusion.base import COMPONENTS
from GenAILab.qai_hub_lm.diffusion.onnx import MODEL_FILE, ONNX_OPSET

_KEY_FILE = "key.json"


def export_key(
    model_id: str,
    revision: str | None,
    dtype: str,
    height: int,
    width: int,
) -> dict:
    """Everything that changes the exported graphs.

    ``revision`` is the checkpoint's commit hash, so a model updated upstream
    re-exports. The torch and diffusers versions are included because both
    shape the traced graph.
    """
    import diffusers

    return {
        "model_id": model_id,
        "revision": revision,
        "dtype": dtype,
        "height": height,
        "width": width,
        "opset": ONNX_OPSET,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
    }


class DiffusionONNXCache:
    """Exported component graphs, keyed by :func:`export_key`."""

    def __init__(self, cache_dir: str | os.PathLike):
        self._dir = Path(cache_dir)
        sweep_stale_staging(self._dir)

    @staticmethod
    def key_hash(key: dict) -> str:
        canonical = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def paths(self, key: dict) -> dict[str, Path] | None:
        """The cached graph of each component, or ``None`` on a miss.

        An entry missing any of its files is discarded.
        """
        entry = self._dir / self.key_hash(key)
        if not entry.exists():
            return None
        if not self._entry_is_complete(entry):
            print(
                f"  [cache] Diffusion ONNX cache entry {entry.name} incomplete — discarding"
            )
            discard_entry(entry)
            return None
        return {name: entry / name / MODEL_FILE for name in COMPONENTS}

    def get_or_export(
        self, key: dict, export: Callable[[Path], object]
    ) -> dict[str, Path]:
        """Cached graphs for ``key``, calling ``export(directory)`` on a miss.

        ``export`` must write ``<directory>/<component>/model.onnx`` for every
        component.
        """
        cached = self.paths(key)
        if cached is not None:
            print(f"  [cache] Diffusion ONNX cache hit ({key['model_id']})")
            return cached
        print(f"  [cache] Diffusion ONNX cache miss ({key['model_id']}), exporting")

        self._dir.mkdir(parents=True, exist_ok=True)
        entry = self._dir / self.key_hash(key)
        with staged_entry(entry, self._entry_is_complete) as staging:
            export(staging)
            if not self._has_every_component(staging):
                raise RuntimeError(
                    f"Export did not write every component to {staging}; expected "
                    f"<component>/{MODEL_FILE} for each of {COMPONENTS}."
                )
            (staging / _KEY_FILE).write_text(json.dumps(key, indent=2, default=str))

        paths = self.paths(key)
        assert paths is not None, f"Cache entry {entry} vanished after export"
        return paths

    @staticmethod
    def _has_every_component(entry: Path) -> bool:
        return all((entry / name / MODEL_FILE).exists() for name in COMPONENTS)

    @classmethod
    def _entry_is_complete(cls, entry: Path) -> bool:
        return entry_is_complete(entry, cls._has_every_component)

    def clear(self):
        clear_dir(self._dir)
