# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Disk-backed cache for FP model results that persists across runs."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch


class DiskBackedFPCache:
    """Two-layer (disk + in-memory) cache for FP model outputs.

    Cached data is written to disk so it survives across runs.
    On cache hit, data is lazily loaded from disk into memory on first access,
    then served from memory for subsequent accesses within the same session.

    The cache stores an ``index.json`` manifest alongside the tensor files so
    that cached entries can be inspected without loading tensor data.
    """

    _INDEX_FILE = "index.json"
    _INDEX_VERSION = 1

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
        self._memory: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: tuple) -> Any | None:
        """Return cached data if it exists, loading from disk lazily. Else ``None``."""
        path_key = self._key_to_path(key)

        # Layer 1: already in memory
        if path_key in self._memory:
            return self._memory[path_key]

        # Layer 2: on disk but not yet loaded
        if path_key in self._index["entries"]:
            file_path = self._cache_dir / self._index["entries"][path_key]["file"]
            if file_path.exists():
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                self._memory[path_key] = data
                return data

            # Stale index entry -- file was deleted externally
            del self._index["entries"][path_key]
            self._save_index()

        return None

    def put(self, key: tuple, data: Any, metadata: Optional[dict] = None):
        """Write *data* to both disk and memory."""
        path_key = self._key_to_path(key)
        file_path = self._cache_dir / f"{path_key}.pt"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file then rename
        tmp_path = file_path.with_suffix(".pt.tmp")
        torch.save(data, tmp_path)
        tmp_path.rename(file_path)

        self._memory[path_key] = data
        self._index["entries"][path_key] = {
            "metadata": metadata,
            "created": datetime.now().isoformat(),
            "size_bytes": file_path.stat().st_size,
            "file": f"{path_key}.pt",
        }
        self._save_index()

    def get_or_compute(
        self,
        key: tuple,
        compute_fn: Callable[[], Any],
        metadata: Optional[dict] = None,
    ) -> Any:
        """Return cached data for *key*, computing and storing it if absent."""
        result = self.get(key)
        if result is not None:
            _log_fp_cache_hit(key)
            return result
        _log_fp_cache_miss(key)
        result = compute_fn()
        self.put(key, result, metadata=metadata)
        return result

    def clear(self):
        """Remove **all** cached data and reset the index."""
        shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._empty_index()
        self._memory.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key_to_path(key: tuple) -> str:
        """Map ``(model_hash, collection_name)`` to a relative path (no extension)."""
        model_hash, collection_name = key
        return f"{model_hash}/{collection_name}"

    def _load_index(self) -> dict:
        index_path = self._cache_dir / self._INDEX_FILE
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [cache] FP cache index corrupted — resetting")
                index_path.unlink()
        return self._empty_index()

    def _save_index(self):
        index_path = self._cache_dir / self._INDEX_FILE
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    @classmethod
    def _empty_index(cls) -> dict:
        return {"version": cls._INDEX_VERSION, "entries": {}}


_CACHE_BANNER = "\n" + "=" * 70 + "\n"


def _log_fp_cache_hit(key: tuple):
    _, collection_name = key
    print(_CACHE_BANNER, end="")
    print(f"  FP CACHE HIT")
    print(f"  Loading cached FP results for: {collection_name}")
    print("=" * 70 + "\n")


def _log_fp_cache_miss(key: tuple):
    _, collection_name = key
    print(f"  [cache] FP cache miss for {collection_name} — computing from scratch")
