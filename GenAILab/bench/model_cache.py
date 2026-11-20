# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Disk-backed cache for ONNX model exports that persists across runs."""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import onnx
import torch
from transformers import AutoConfig, PretrainedConfig

from GenAILab.bench.utils.cache_utils import (
    atomic_write_json,
    checkpoint_revision,
    model_fingerprint,
    staged_entry,
)
from GenAILab.qai_hub_lm.schema.components import (
    MODALITY_COMPONENTS,
    require_component,
)


@dataclass
class ModelCacheEntry:
    """Container for the ONNX model components returned by an export.

    One field per modality component (see
    :data:`GenAILab.qai_hub_lm.schema.components.MODALITY_COMPONENTS`). All of
    them are optional: a plain LLM has only a backbone, a VLM adds ``visual``,
    an ASR model adds ``audio``.
    """

    backbone: onnx.ModelProto
    visual: Optional[onnx.ModelProto] = None
    audio: Optional[onnx.ModelProto] = None
    embedding: Optional[torch.nn.Embedding] = None
    config: Optional[PretrainedConfig] = None
    extras: Optional[dict[str, torch.nn.Module]] = None

    def component(self, name: str) -> Optional[onnx.ModelProto]:
        """The graph for a modality component, or ``None`` if this model lacks it."""
        require_component(name)
        return getattr(self, name)

    def present_components(self) -> tuple[str, ...]:
        """Names of the components with a live graph, in canonical order."""
        return tuple(n for n in MODALITY_COMPONENTS if getattr(self, n) is not None)


class DiskBackedModelCache:
    """Disk-backed cache for exported ONNX models.

    Follows the same pattern as :class:`DiskBackedFPCache` but stores ONNX
    model protos (with external data) instead of tensors.

    The cache stores an ``index.json`` manifest alongside the model directories
    so that cached entries can be inspected without loading model data.

    Storage layout::

        {cache_dir}/
        +-- index.json
        +-- {key_hash}/
            +-- config.json
            +-- backbone/
            |   +-- model.onnx
            |   +-- model.data
            +-- visual/          (vision models only)
            |   +-- model.onnx
            |   +-- model.data
            +-- audio/           (audio models only)
            |   +-- model.onnx
            |   +-- model.data
            +-- embedding.pth    (multi-modal only)

    Each component gets a ``{name}/`` subdir; a missing one reads back as
    ``None``, so entries written before a component existed still load.

    Several processes may share one cache directory. An entry is written to a
    private directory and renamed into place, so it is either absent or whole,
    and lookups go by the entry directory, not by ``index.json`` (which is only
    a manifest for inspection and may briefly trail the directories). If two
    processes export the same key, the first to finish wins.
    """

    _INDEX_FILE = "index.json"
    _ENTRY_FILE = "entry.json"
    _INDEX_VERSION = 1

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> ModelCacheEntry | None:
        """Return a cached :class:`ModelCacheEntry` if it exists on disk, else ``None``."""
        entry_dir = self._cache_dir / key
        backbone_path = entry_dir / "backbone" / "model.onnx"
        if not backbone_path.exists():
            # Never cached, or another process sharing this cache has it listed
            # in the index but the files were deleted externally.
            if key in self._load_index()["entries"]:
                print(
                    f"  [cache] Model cache stale entry for key {key} "
                    f"(backbone not found on disk) — removing"
                )
                self._update_index(key, None)
            return None

        # Load components from disk
        config_path = entry_dir / "config.json"
        backbone = onnx.load(str(backbone_path))

        # An absent subdir yields None, which covers both "no such component"
        # and "entry predates the component".
        component_models = {}
        for name in MODALITY_COMPONENTS:
            component_path = entry_dir / name / "model.onnx"
            component_models[name] = (
                onnx.load(str(component_path)) if component_path.exists() else None
            )

        embedding_path = entry_dir / "embedding.pth"
        if embedding_path.exists():
            weights = torch.load(
                str(embedding_path), map_location="cpu", weights_only=True
            )
            if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
                raise ValueError("Expected a 2D embedding tensor in embedding.pth")
            embedding = torch.nn.Embedding.from_pretrained(weights, freeze=False)
            embedding = embedding.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            embedding = None

        config = (
            AutoConfig.from_pretrained(str(entry_dir), trust_remote_code=True)
            if config_path.exists()
            else None
        )

        # Load extras: any .pth files not already handled (embedding.pth)
        extras_dir = entry_dir / "extras"
        extras = {}
        if extras_dir.exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for pth_file in sorted(extras_dir.glob("*.pth")):
                tensor = torch.load(
                    str(pth_file), map_location="cpu", weights_only=True
                )
                name = pth_file.stem
                module = torch.nn.Embedding(*tensor.shape)
                module.weight = torch.nn.Parameter(tensor, requires_grad=False)
                extras[name] = module.to(device)

        return ModelCacheEntry(
            backbone=backbone,
            embedding=embedding,
            config=config,
            extras=extras or None,
            **component_models,
        )

    def put(self, key: str, entry: ModelCacheEntry, metadata: Optional[dict] = None):
        """Write a :class:`ModelCacheEntry` to disk and update the index.

        If another process published *key* first, its entry is kept.
        """
        index_entry = {
            "metadata": metadata or {},
            "created": datetime.now().isoformat(),
        }

        with staged_entry(self._cache_dir / key, _entry_is_complete) as staging:
            self._write_entry(staging, entry)
            # The entry's own manifest, readable without the shared index.
            atomic_write_json(staging / self._ENTRY_FILE, index_entry, default=str)

        self._update_index(key, index_entry)

    @staticmethod
    def _write_entry(entry_dir: Path, entry: ModelCacheEntry):
        # Save backbone ONNX model with external data
        backbone_dir = entry_dir / "backbone"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        backbone_path = backbone_dir / "model.onnx"
        onnx.save_model(
            entry.backbone,
            str(backbone_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

        # Save each modality component's ONNX model under its own subdir
        for name in MODALITY_COMPONENTS:
            component_model = getattr(entry, name)
            if component_model is None:
                continue
            component_dir = entry_dir / name
            component_dir.mkdir(parents=True, exist_ok=True)
            onnx.save_model(
                component_model,
                str(component_dir / "model.onnx"),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.data",
            )

        # Save embedding if present
        if entry.embedding is not None:
            embedding_path = entry_dir / "embedding.pth"
            torch.save(entry.embedding.state_dict()["weight"], str(embedding_path))

        # Save extras (model-specific auxiliary modules)
        if entry.extras:
            extras_dir = entry_dir / "extras"
            extras_dir.mkdir(parents=True, exist_ok=True)
            for name, module in entry.extras.items():
                if module is not None:
                    torch.save(
                        module.state_dict()["weight"],
                        str(extras_dir / f"{name}.pth"),
                    )

        # Save config if present
        if entry.config is not None:
            entry.config.save_pretrained(str(entry_dir))

    def get_or_export(
        self,
        key: str,
        compute_fn: Callable[[], ModelCacheEntry],
        metadata: Optional[dict] = None,
    ) -> ModelCacheEntry:
        """Return cached entry for *key*, exporting and storing it if absent."""
        result = self.get(key)
        if result is not None:
            _log_model_cache_hit(key, metadata)
            return result
        _log_model_cache_miss(key, metadata)
        start = time.perf_counter()
        result = compute_fn()
        exported = time.perf_counter()
        self.put(key, result, metadata=metadata)
        _log_model_export_time(
            metadata, exported - start, time.perf_counter() - exported
        )
        return self.get(key)

    def clear(self):
        """Remove **all** cached data and reset the index."""
        shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._empty_index()

    @staticmethod
    def build_key(params: dict) -> str:
        """Return a deterministic hex digest of *params* for use as a cache key."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def export_identity(params: dict, model_cls: type) -> dict:
        """Everything an export of *model_cls* with config *params* depends on.

        *params* is the export config (model ID, lengths, dtype, ...). Added to
        it: the class's adaptations with their kwargs, which change the graph
        but not the class name; the checkpoint revision, so a weight-only update
        upstream re-exports; and the package versions and GenAILab code the
        export runs (see :func:`model_fingerprint`).
        """
        return {
            **params,
            "adaptations": getattr(model_cls, "adaptation_identity", []),
            "revision": checkpoint_revision(params["model_id"]),
            "fingerprint": model_fingerprint(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        index_path = self._cache_dir / self._INDEX_FILE
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                # Not unlinked: another process may have just replaced it with a
                # good one. The next update overwrites it.
                print(f"  [cache] Model cache index corrupted — resetting")
        return self._empty_index()

    def _update_index(self, key: str, index_entry: dict | None):
        """Set the index entry for *key*, or remove it when *index_entry* is ``None``.

        The index is re-read from disk first, so entries other processes sharing
        this cache added since it was loaded are kept, not overwritten. Two
        processes updating it at the same moment can still lose one update; that
        only costs an inspection entry, as lookups go by entry directory.
        """
        self._index = self._load_index()
        if index_entry is None:
            self._index["entries"].pop(key, None)
        else:
            self._index["entries"][key] = index_entry
        self._save_index()

    def _save_index(self):
        # Written aside and renamed into place, so a concurrent reader never sees
        # a half-written index (which _load_index would reset as corrupted).
        atomic_write_json(self._cache_dir / self._INDEX_FILE, self._index, indent=2)

    @classmethod
    def _empty_index(cls) -> dict:
        return {"version": cls._INDEX_VERSION, "entries": {}}


def _entry_is_complete(entry_dir: Path) -> bool:
    return (entry_dir / "backbone" / "model.onnx").exists()


_CACHE_BANNER = "\n" + "=" * 70 + "\n"


def _model_label(metadata: dict | None) -> str:
    if metadata:
        model_id = metadata.get("model_id", "")
        if model_id:
            return model_id
    return ""


def _log_model_cache_hit(key: str, metadata: dict | None):
    label = _model_label(metadata)
    print(_CACHE_BANNER, end="")
    print(f"  MODEL CACHE HIT")
    if label:
        print(f"  Model: {label}")
    print(f"  Loading exported ONNX model from disk")
    print("=" * 70 + "\n")


def _log_model_cache_miss(key: str, metadata: dict | None):
    label = _model_label(metadata)
    detail = f" for {label}" if label else ""
    print(f"  [cache] Model cache miss{detail} — exporting from scratch")


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _log_model_export_time(metadata: dict | None, export_s: float, save_s: float):
    label = _model_label(metadata)
    detail = f" for {label}" if label else ""
    print(
        f"  [cache] Exported{detail} in {_format_duration(export_s)} wall clock "
        f"(+ {_format_duration(save_s)} writing to the model cache)"
    )
