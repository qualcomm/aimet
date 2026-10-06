# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Content-addressed recipe cache for per-component recipe chain checkpoints.

Automatically detects shared recipe prefixes across experiments and reuses
checkpointed state. Each recipe step extends a Merkle chain hash, so two
experiments sharing the first N steps will have identical hashes at step N.

Backbone and visual components have independent cache chains, allowing
maximum reuse for VLMs where experiments may share visual recipes but
diverge on backbone recipes (or vice versa).

Storage layout:
    {cache_dir}/
    ├── {chain_hash}/
    │   ├── metadata.json
    │   ├── steps.json
    │   └── state/
    │       ├── checkpoint.pt           (Torch: full state_dict)
    │       ├── quantizer_flags.json    (Torch: per-quantizer overwrite flags)
    │       ├── model.onnx              (ONNX)
    │       ├── model.encodings         (ONNX)
    │       └── quantizer_flags.json   (both: per-quantizer freeze state)
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import torch

from GenAILab.bench.profiler import (
    RecipeStepStats,
    convert_gpu_meter_to_dict,
)

if TYPE_CHECKING:
    from GenAILab.bench.yaml_config_parser import ResolvedStep


@dataclass
class CachedProfiler:
    """Replays profiler summary data from a cached recipe step.

    Duck-types GPUMeter so convert_gpu_meter_to_dict() works unchanged.
    """

    elapsed_ms: float
    cuda_first_mb: float = 0
    cuda_peak_mb: float = 0
    cpu_first_mb: float = 0
    cpu_peak_mb: float = 0
    cuda_running_mb: list = None
    cpu_running_mb: list = None


def _stable_json_hash(obj) -> str:
    """Produce a deterministic SHA-256 hex digest of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


_cached_aimet_source_hash = None


def _hash_aimet_sources() -> str:
    """Hash all .py files in aimet_torch and aimet_onnx package directories.

    Finds the installed location of each package via its __path__ attribute
    and hashes every .py file (sorted for determinism). This catches source
    changes on dev branches where the pip version string doesn't change,
    and works without .git (e.g. code synced to a cluster).

    Results are cached after the first call since source files don't change
    within a single process.
    """
    global _cached_aimet_source_hash
    if _cached_aimet_source_hash is not None:
        return _cached_aimet_source_hash

    hasher = hashlib.sha256()
    for pkg_name in ("aimet_torch", "aimet_onnx"):
        try:
            pkg = __import__(pkg_name)
            pkg_paths = getattr(pkg, "__path__", [])
        except ImportError:
            continue

        for pkg_path in pkg_paths:
            pkg_dir = Path(pkg_path)
            if not pkg_dir.is_dir():
                continue
            # Sort for determinism across filesystems
            for py_file in sorted(pkg_dir.rglob("*.py")):
                try:
                    hasher.update(str(py_file.relative_to(pkg_dir)).encode())
                    hasher.update(py_file.read_bytes())
                except (OSError, ValueError):
                    continue

    _cached_aimet_source_hash = hasher.hexdigest()[:32]
    return _cached_aimet_source_hash


def _adaptation_identity(adaptations: list) -> list:
    """Canonical, order-insensitive identity for an authored adaptations list.

    Handles both forms ``_normalize_adaptations`` accepts; a bare name hashes the
    same as that name with empty kwargs.
    """
    canonical = []
    for entry in adaptations:
        if isinstance(entry, dict):
            for name, kwargs in entry.items():
                items = kwargs if isinstance(kwargs, dict) else {}
                canonical.append([name, sorted((str(k), v) for k, v in items.items())])
        else:
            canonical.append([str(entry), []])
    return sorted(canonical, key=lambda pair: pair[0])


class RecipeCache:
    """Content-addressed cache for per-component recipe chain checkpoints."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear(self):
        """Remove all cached data."""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @functools.cached_property
    def env_hash(self) -> str:
        """Hash the environment for cache invalidation.

        Combines two signals:
        1. pip freeze versions for non-AIMET dependencies (torch, onnxruntime)
           that affect recipe behavior.
        2. Source file hashes for aimet_torch/aimet_onnx packages. This catches
           code changes on dev branches where the pip version string stays the
           same. Works regardless of whether .git is present (e.g. code synced
           to a cluster).
        """
        from GenAILab.bench.profiler import _collect_environment  # circular

        env = _collect_environment()
        pip_freeze = env.get("pip_freeze", [])

        # Non-AIMET dependency versions
        dep_prefixes = (
            "torch==",
            "onnxruntime",
            "onnx==",
            "onnxscript==",
            "transformers==",
            "datasets==",
            "accelerate==",
        )
        dep_versions = sorted(
            p for p in pip_freeze if any(p.startswith(k) for k in dep_prefixes)
        )

        # Hash AIMET source files directly
        aimet_source_hash = _hash_aimet_sources()

        return _stable_json_hash(
            {
                "deps": dep_versions,
                "aimet_sources": aimet_source_hash,
            }
        )

    def compute_base_hash(
        self,
        model_id: str,
        precision_config,
        model_kwargs: dict,
        framework: str,
        component: str = "backbone",
        pre_sim: tuple[ResolvedStep, ...] | None = None,
        adaptations: list | None = None,
    ) -> str:
        """Hash for a freshly-instantiated component (before any recipes).

        Uses precision_config.weight_identity() so experiments differing only
        in activations/kv_cache share cache entries for weight-modifying recipes.

        ``pre_sim`` is the flat pre-sim step tuple (e.g. the SpinQuant step that
        rotates the float model before the sim is built). Its identity is folded
        into the base hash so a chain cached on a rotated graph never collides
        with one on a non-rotated graph (or a graph rotated with different
        R1/R2/R3 flags). When there are no pre-sim steps, NO pre-sim key is added
        -- the base hash is then byte-identical to the pre-pre-sim behavior, so
        existing cache entries for non-rotated runs stay valid.

        ``adaptations`` is the authored ``model.adaptations`` list. It must be
        keyed because adaptations live on the model *class* and never reach
        ``model_kwargs``, so without it two runs differing only in an adaptation
        share encodings. As with ``pre_sim``, an empty list adds no key.
        """
        identity = {
            "model_id": model_id,
            "precision": precision_config.weight_identity(),
            "model_kwargs": model_kwargs,
            "env_hash": self.env_hash,
            "framework": framework,
            "component": component,
        }
        if pre_sim:
            ps_identity = {step.name: step.recipe_kwargs for step in pre_sim}
            # Key kept as "spinquant" for cache-key stability with prior runs.
            identity["spinquant"] = ps_identity
        if adaptations:
            identity["adaptations"] = _adaptation_identity(adaptations)
        return _stable_json_hash(identity)

    @staticmethod
    def step_hash(parent_hash: str, step: ResolvedStep) -> str:
        """Extend a chain hash with one recipe step."""
        dataset_dict = {}
        if step.dataset_cls is not None:
            dataset_dict = {
                "class": step.dataset_cls.__name__,
                **step.dataset_kwargs,
            }
        step_identity = {
            "parent": parent_hash,
            "recipe": step.technique_cls.__name__,
            "recipe_kwargs": step.recipe_kwargs,
            "dataset": _serialize_dataset_config(dataset_dict),
        }
        return _stable_json_hash(step_identity)

    def lookup(
        self,
        recipe_list: list[ResolvedStep] | tuple[ResolvedStep, ...],
        sim_component,
        model_id: str,
        precision_config,
        model_kwargs: dict,
        framework: str,
        component: str = "backbone",
        pre_sim: tuple[ResolvedStep, ...] | None = None,
        adaptations: list | None = None,
    ) -> tuple[int, list, list[str]]:
        """Compute hashes, find the longest cached prefix, load state, and log.

        Returns (skip_to, cached_step_stats, chain_hashes).
        """
        base = self.compute_base_hash(
            model_id,
            precision_config,
            model_kwargs,
            framework,
            component,
            pre_sim,
            adaptations,
        )
        hashes = compute_chain_hashes(self, base, recipe_list)
        skip, chain = find_cache_hit(self, hashes)
        if skip > 0:
            cached_stats = self.load(chain, sim_component, framework)
            remaining = [step.technique_cls.__name__ for step in recipe_list[skip:]]
            log_cache_hit(component, cached_stats, len(recipe_list), remaining)
        else:
            log_cache_miss(component, len(recipe_list))
            cached_stats = []
        return skip, cached_stats, hashes

    def get(self, chain_hash: str) -> dict | None:
        """Return cache entry metadata if it exists, else None."""
        meta_path = self.cache_dir / chain_hash / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def get_state_dir(self, chain_hash: str) -> Path:
        """Return path to the state/ directory for a cache entry."""
        return self.cache_dir / chain_hash / "state"

    def load_step_stats(self, chain_hash: str) -> list[RecipeStepStats]:
        """Load steps.json and reconstruct RecipeStepStats with CachedProfiler."""
        steps_path = self.cache_dir / chain_hash / "steps.json"
        if not steps_path.exists():
            return []
        with open(steps_path) as f:
            steps_data = json.load(f)

        result = []
        for step in steps_data:
            profiler_data = step.get("profiler", {})
            profiler = CachedProfiler(
                elapsed_ms=profiler_data.get("elapsed_ms", 0),
                cuda_first_mb=profiler_data.get("cuda_starting_mb", 0),
                cuda_peak_mb=profiler_data.get("cuda_peak_mb", 0),
                cpu_first_mb=profiler_data.get("cpu_starting_mb", 0),
                cpu_peak_mb=profiler_data.get("cpu_peak_mb", 0),
            )
            result.append(
                RecipeStepStats(
                    recipe_name=step["recipe_name"],
                    recipe_kwargs=step.get("recipe_kwargs", {}),
                    dataset_name=step.get("dataset_name", ""),
                    dataset_kwargs=step.get("dataset_kwargs", {}),
                    profiler=profiler,
                )
            )
        return result

    def _save_steps(self, chain_hash: str, step_stats: list[RecipeStepStats]):
        """Save step stats (profiler data) to steps.json."""
        entry_dir = self.cache_dir / chain_hash
        entry_dir.mkdir(parents=True, exist_ok=True)

        steps_data = []
        for step in step_stats:
            steps_data.append(
                {
                    "recipe_name": step.recipe_name,
                    "recipe_kwargs": step.recipe_kwargs,
                    "dataset_name": step.dataset_name,
                    "dataset_kwargs": step.dataset_kwargs,
                    "profiler": convert_gpu_meter_to_dict(
                        step.profiler, remove_finegrained=True
                    ),
                }
            )
        with open(entry_dir / "steps.json", "w") as f:
            json.dump(steps_data, f, indent=2)

    def _save_metadata(
        self,
        chain_hash: str,
        parent_hash: str,
        component: str,
        env_hash: str,
        model_id: str,
        step_count: int,
    ):
        """Save metadata.json for a cache entry."""
        entry_dir = self.cache_dir / chain_hash
        entry_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "chain_hash": chain_hash,
            "parent_hash": parent_hash,
            "component": component,
            "env_hash": env_hash,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_id": model_id,
            "step_count": step_count,
        }
        with open(entry_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save(
        self,
        chain_hash: str,
        parent_hash: str,
        step_stats: list[RecipeStepStats],
        sim_component,
        framework: str,
        component: str = "backbone",
        model_id: str = "",
    ) -> None:
        """Save a cache entry with framework-specific serialization."""
        entry_dir = self.cache_dir / chain_hash
        state_dir = entry_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        self._save_steps(chain_hash, step_stats)
        self._save_metadata(
            chain_hash, parent_hash, component, self.env_hash, model_id, len(step_stats)
        )

        if framework == "torch":
            self._save_torch(state_dir, sim_component)
        elif framework == "onnx":
            self._save_onnx(state_dir, sim_component)

        log_cache_save(component, step_stats[-1], chain_hash)

    def load(
        self,
        chain_hash: str,
        sim_component,
        framework: str,
    ) -> list[RecipeStepStats]:
        """Load cached weight state into sim_component and return step stats."""
        state_dir = self.cache_dir / chain_hash / "state"
        if not state_dir.exists():
            return []

        if framework == "torch":
            self._load_torch(state_dir, sim_component)
        elif framework == "onnx":
            self._load_onnx(state_dir, sim_component)

        return self.load_step_stats(chain_hash)

    # ---- Torch serialization ------------------------------------------------

    @staticmethod
    def _collect_quantizer_flags(sim_component) -> dict:
        """Snapshot per-quantizer ``_is_overwrite_allowed`` flags.

        Returns a dict keyed by ``(module_path, quantizer_group, quantizer_name)``
        whose values are the ``_is_overwrite_allowed`` dicts.  This captures the
        exact freeze state so that recipes like SeqMSE that selectively freeze
        individual param quantizers are faithfully reproduced on cache load.
        """
        from aimet_torch.v2.nn import BaseQuantizationMixin

        flags = {}
        for path, module in sim_component.model.named_modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue
            for group in ("param_quantizers", "input_quantizers", "output_quantizers"):
                container = getattr(module, group)
                items = (
                    container.items()
                    if hasattr(container, "items")
                    else enumerate(container)
                )
                for name, qtzr in items:
                    if qtzr is not None:
                        flags[f"{path}.{group}.{name}"] = dict(
                            qtzr._is_overwrite_allowed
                        )
        return flags

    @staticmethod
    def _restore_quantizer_flags(sim_component, flags: dict):
        """Restore per-quantizer ``_is_overwrite_allowed`` flags from a snapshot."""
        from aimet_torch.v2.nn import BaseQuantizationMixin

        for path, module in sim_component.model.named_modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue
            for group in ("param_quantizers", "input_quantizers", "output_quantizers"):
                container = getattr(module, group)
                items = (
                    container.items()
                    if hasattr(container, "items")
                    else enumerate(container)
                )
                for name, qtzr in items:
                    key = f"{path}.{group}.{name}"
                    if qtzr is not None and key in flags:
                        qtzr._is_overwrite_allowed.update(flags[key])

    def _save_torch(self, state_dir: Path, sim_component):
        # Save the full quantsim state dict (weights + all quantizer params)
        torch.save(sim_component.model.state_dict(), state_dir / "checkpoint.pt")
        # Save per-quantizer overwrite flags so that the exact freeze state
        # (e.g. from SeqMSE) is reproduced on load.
        flags = self._collect_quantizer_flags(sim_component)
        with open(state_dir / "quantizer_flags.json", "w") as f:
            json.dump(flags, f)

    def _load_torch(self, state_dir: Path, sim_component):
        ckpt_path = state_dir / "checkpoint.pt"
        if ckpt_path.exists():
            sim_component.model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu"), strict=False
            )

        flags_path = state_dir / "quantizer_flags.json"
        if flags_path.exists():
            with open(flags_path) as f:
                flags = json.load(f)
            self._restore_quantizer_flags(sim_component, flags)

    # ---- ONNX serialization -------------------------------------------------

    def _save_onnx(self, state_dir: Path, sim_component):
        # Deep copy to avoid mutating the live model proto — onnx.save_model
        # with save_as_external_data=True strips inline tensor data and replaces
        # it with file references, which would break the running session.
        model_copy = copy.deepcopy(sim_component.model.model)
        onnx.save_model(
            model_copy,
            str(state_dir / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        del model_copy

        # Save full encodings (param + activation)
        sim_component.export(str(state_dir), "model", export_model=False)

        # Save per-op frozen encoding flags so that recipes like SeqMSE/AdaScale
        # that selectively freeze quantizers are faithfully reproduced on load.
        frozen_ops = {
            name: op._is_encoding_frozen
            for name, op in sim_component.qc_quantize_op_dict.items()
        }
        with open(state_dir / "quantizer_flags.json", "w") as f:
            json.dump(frozen_ops, f)

    def _load_onnx(self, state_dir: Path, sim_component):
        from aimet_onnx.quantsim import load_encodings_to_sim

        model_path = state_dir / "model.onnx"
        if model_path.exists():
            # Copy initializers (weight tensors) from the cached proto into
            # into the live proto. Replacing the entire proto would bring
            # stale QcQuantizeOp quant_info C++ pointers, causing a segfault.
            cached_model = onnx.load(str(model_path), load_external_data=True)
            cached_inits = {init.name: init for init in cached_model.graph.initializer}

            for i, init in enumerate(sim_component.model.model.graph.initializer):
                if init.name in cached_inits:
                    sim_component.model.model.graph.initializer[i].CopyFrom(
                        cached_inits[init.name]
                    )

            del cached_model
            sim_component._rebuild_session()

        enc_path = state_dir / "model.encodings"
        if enc_path.exists():
            load_encodings_to_sim(
                sim_component,
                str(enc_path),
                strict=False,
                allow_overwrite=True,
                disable_missing_quantizers=False,
            )

        # Restore per-op frozen encoding flags
        flags_path = state_dir / "quantizer_flags.json"
        if flags_path.exists():
            with open(flags_path) as f:
                frozen_ops = json.load(f)
            for name, was_frozen in frozen_ops.items():
                if was_frozen and name in sim_component.qc_quantize_op_dict:
                    sim_component.qc_quantize_op_dict[name].freeze_encodings()


def _format_elapsed(ms: float) -> str:
    """Format milliseconds as a human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds % 60:.0f}s"
    hours = minutes / 60
    return f"{hours:.0f}h {minutes % 60:.0f}m"


def _step_names(steps: list[RecipeStepStats]) -> str:
    """Format step names as a readable chain like 'SpinQuant -> Calibration'."""
    return " -> ".join(s.recipe_name for s in steps)


def _total_elapsed(steps: list[RecipeStepStats]) -> float:
    """Sum elapsed_ms across all steps."""
    return sum(s.profiler.elapsed_ms for s in steps if s.profiler is not None)


_CACHE_BANNER = "\n" + "=" * 70 + "\n"


def log_cache_hit(
    component: str,
    cached_stats: list[RecipeStepStats],
    total_steps: int,
    remaining_names: list[str],
):
    """Print a prominent cache hit message."""
    skipped = len(cached_stats)
    saved_ms = _total_elapsed(cached_stats)
    remaining = total_steps - skipped

    print(_CACHE_BANNER, end="")
    print(f"  RECIPE CACHE HIT ({component})")
    print(f"  Skipping {skipped}/{total_steps} steps: {_step_names(cached_stats)}")
    print(f"  Time saved: ~{_format_elapsed(saved_ms)}")
    if remaining > 0:
        print(f"  Remaining: {' -> '.join(remaining_names)} ({remaining} step(s))")
    else:
        print(f"  All {component} steps served from cache")
    print("=" * 70 + "\n")


def log_cache_save(component: str, step: RecipeStepStats, chain_hash: str):
    """Print a message when a step is saved to the cache."""
    elapsed = _format_elapsed(step.profiler.elapsed_ms) if step.profiler else "?"
    print(
        f"  [cache] Saved {component}/{step.recipe_name} "
        f"({elapsed}) -> {chain_hash[:12]}..."
    )


def log_cache_miss(component: str, total_steps: int):
    """Print a message when no cache hit is found."""
    print(
        f"  [cache] No cache hit for {component} ({total_steps} steps) — running from scratch"
    )


def _serialize_dataset_config(dataset_config: dict) -> dict:
    """Serialize a dataset config dict for hashing, handling class references."""
    if not dataset_config:
        return {}
    config = dataset_config.copy()
    cls = config.pop("class", None)
    return {
        "class": cls.__name__ if hasattr(cls, "__name__") else str(cls) if cls else "",
        **config,
    }


def compute_chain_hashes(
    recipe_cache: RecipeCache,
    base_hash: str,
    recipe_list: list[ResolvedStep] | tuple[ResolvedStep, ...],
) -> list[str]:
    """Pre-compute Merkle chain hashes for all recipe steps.

    Returns a list of length len(recipe_list) + 1, where index 0 is the base hash
    and index i+1 is the hash after step i.
    """
    hashes = [base_hash]
    for step in recipe_list:
        hashes.append(recipe_cache.step_hash(hashes[-1], step))
    return hashes


def find_cache_hit(recipe_cache: RecipeCache, hashes: list[str]) -> tuple[int, str]:
    """Walk backwards to find the longest cached prefix.

    Returns (skip_to, chain_hash) where skip_to is the number of steps
    to skip (0 means no cache hit).
    """
    for i in range(len(hashes) - 1, 0, -1):
        if recipe_cache.get(hashes[i]):
            return i, hashes[i]
    return 0, hashes[0]
