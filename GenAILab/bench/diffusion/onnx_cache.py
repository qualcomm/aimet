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
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import torch

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

    @staticmethod
    def key_hash(key: dict) -> str:
        canonical = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def paths(self, key: dict) -> dict[str, Path] | None:
        """The cached graph of each component, or ``None`` on a miss."""
        entry = self._dir / self.key_hash(key)
        paths = {name: entry / name / MODEL_FILE for name in COMPONENTS}
        if all(p.exists() for p in paths.values()):
            return paths
        return None

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
        staging = Path(tempfile.mkdtemp(prefix=f".{entry.name}-", dir=self._dir))
        try:
            export(staging)
            (staging / _KEY_FILE).write_text(json.dumps(key, indent=2, default=str))
            shutil.rmtree(entry, ignore_errors=True)  # A partial earlier entry.
            staging.rename(entry)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        paths = self.paths(key)
        if paths is None:
            raise RuntimeError(
                f"Export did not write every component to {entry}; expected "
                f"<component>/{MODEL_FILE} for each of {COMPONENTS}."
            )
        return paths

    def clear(self):
        shutil.rmtree(self._dir, ignore_errors=True)
