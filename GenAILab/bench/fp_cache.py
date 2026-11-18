# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Disk-backed cache for FP model results that persists across runs."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from GenAILab.bench.utils.cache_utils import atomic_write_json


class DiskBackedFPCache:
    """Two-layer (disk + in-memory) cache for FP model outputs.

    Cached data is written to disk so it survives across runs.
    On cache hit, data is lazily loaded from disk into memory on first access,
    then served from memory for subsequent accesses within the same session.

    The cache stores an ``index.json`` manifest alongside the tensor files so
    that cached entries can be inspected without loading tensor data. The files
    themselves are the source of truth, so a lookup never depends on the index.

    Several processes may share one cache directory: each tensor file is written
    aside and renamed into place, and the index is re-read before every update.
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

        # Layer 2: on disk but not yet loaded. The file is the source of truth:
        # another process may have written it without our index knowing.
        file_path = self._cache_dir / f"{path_key}.pt"
        if file_path.exists():
            data = torch.load(file_path, map_location="cpu", weights_only=True)
            self._memory[path_key] = data
            return data

        # Stale index entry -- file was deleted externally
        if path_key in self._load_index()["entries"]:
            self._update_index(path_key, None)

        return None

    def put(self, key: tuple, data: Any, metadata: Optional[dict] = None):
        """Write *data* to both disk and memory."""
        path_key = self._key_to_path(key)
        file_path = self._cache_dir / f"{path_key}.pt"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: a temp file of our own, then rename. Two processes
        # computing the same entry must not share a temp file.
        tmp_path = file_path.with_name(
            f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            torch.save(data, tmp_path)
            os.replace(tmp_path, file_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        self._memory[path_key] = data
        self._update_index(
            path_key,
            {
                "metadata": metadata,
                "created": datetime.now().isoformat(),
                "size_bytes": file_path.stat().st_size,
                "file": f"{path_key}.pt",
            },
        )

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
                # Not unlinked: another process may have just replaced it with a
                # good one. The next update overwrites it.
                print(f"  [cache] FP cache index corrupted — resetting")
        return self._empty_index()

    def _update_index(self, path_key: str, index_entry: dict | None):
        """Set the index entry for *path_key*, or remove it when ``None``.

        The index is re-read from disk first, so entries other processes sharing
        this cache added since it was loaded are kept. Two processes updating it
        at the same moment can still lose one update, which only costs an
        inspection entry: lookups go by file.
        """
        self._index = self._load_index()
        if index_entry is None:
            self._index["entries"].pop(path_key, None)
        else:
            self._index["entries"][path_key] = index_entry
        self._save_index()

    def _save_index(self):
        atomic_write_json(
            self._cache_dir / self._INDEX_FILE, self._index, indent=2, default=str
        )

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
