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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import torch

from GenAILab.bench.profiler import (
    RecipeStepStats,
    convert_gpu_meter_to_dict,
)
from GenAILab.bench.utils.cache_utils import (
    checkpoint_revision,
    clear_dir,
    discard_entry,
    entry_is_complete,
    file_digest,
    recipe_fingerprint,
    staged_entry,
    sweep_stale_staging,
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


def _entry_is_complete(entry_dir: Path) -> bool:
    return entry_is_complete(
        entry_dir,
        lambda entry: (entry / "metadata.json").exists() and (entry / "state").is_dir(),
    )


class RecipeCache:
    """Content-addressed cache for per-component recipe chain checkpoints.

    Several processes may share one cache directory. An entry is written to a
    private directory and renamed into place, so ``metadata.json`` existing
    means the whole entry (steps and state) is there, even if a writer was
    killed mid-save. If two processes save the same chain hash, the first to
    finish wins; they are equivalent, being content-addressed.

    An entry that has lost files since it was written, or cannot be read, is
    discarded and the lookup falls back to the next shorter cached prefix. Its
    files are all read before the sim is touched, so a failed read leaves the
    sim as it was.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sweep_stale_staging(self.cache_dir)

    def clear(self):
        """Remove all cached data."""
        clear_dir(self.cache_dir)

    @functools.cached_property
    def env_hash(self) -> str:
        """Hash of the code and packages a recipe step runs.

        See :func:`recipe_fingerprint`: AIMET is hashed file by file (Python,
        compiled extensions, configs), so a dev build whose version string never
        changes still misses on a code change.
        """
        return _stable_json_hash(recipe_fingerprint())

    def compute_base_hash(
        self,
        model_id: str,
        precision_config,
        model_kwargs: dict,
        framework: str,
        component: str = "backbone",
        pre_sim: tuple[ResolvedStep, ...] | None = None,
        adaptations: list | None = None,
        model_identity: dict | None = None,
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

        ``model_identity`` is the rest of the model section the chain depends on
        (see :func:`chain_model_identity`): lengths, input sizes, precomputed
        encodings and checkpoint revision, none of which are in ``model_kwargs``.
        """
        identity = {
            "model_id": model_id,
            "precision": precision_config.weight_identity(),
            "model_kwargs": model_kwargs,
            "env_hash": self.env_hash,
            "framework": framework,
            "component": component,
            "model": model_identity or {},
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
        model_identity: dict | None = None,
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
            model_identity,
        )
        hashes = compute_chain_hashes(self, base, recipe_list)
        skip, cached_stats = 0, []
        for i in range(len(hashes) - 1, 0, -1):
            if not self.get(hashes[i]):
                continue
            # Only reading is guarded: a read failure is a damaged entry, so the
            # next shorter prefix is tried. Applying is not, as a failure there
            # leaves the sim half-modified and must not be run on.
            try:
                read = self._read(hashes[i], framework)
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._discard(hashes[i], f"failed to load: {type(e).__name__}: {e}")
                continue
            cached_stats = self._apply(read, sim_component, framework)
            skip = i
            break
        if skip > 0:
            remaining = [step.technique_cls.__name__ for step in recipe_list[skip:]]
            log_cache_hit(component, cached_stats, len(recipe_list), remaining)
        else:
            log_cache_miss(component, len(recipe_list))
        return skip, cached_stats, hashes

    def get(self, chain_hash: str) -> dict | None:
        """Return cache entry metadata if the entry is whole, else None.

        An entry missing any of its files is discarded.
        """
        entry_dir = self.cache_dir / chain_hash
        if not entry_dir.exists():
            return None
        if not _entry_is_complete(entry_dir):
            self._discard(chain_hash, "files missing")
            return None
        try:
            with open(entry_dir / "metadata.json") as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            self._discard(chain_hash, f"unreadable metadata: {e}")
            return None

    def _discard(self, chain_hash: str, reason: str):
        print(
            f"  [cache] Recipe cache entry {chain_hash[:12]}... unusable ({reason})"
            " — discarding"
        )
        discard_entry(self.cache_dir / chain_hash)

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

    @staticmethod
    def _save_steps(entry_dir: Path, step_stats: list[RecipeStepStats]):
        """Save step stats (profiler data) to steps.json."""
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

    @staticmethod
    def _save_metadata(
        entry_dir: Path,
        chain_hash: str,
        parent_hash: str,
        component: str,
        env_hash: str,
        model_id: str,
        step_count: int,
    ):
        """Save metadata.json for a cache entry."""
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
        with staged_entry(self.cache_dir / chain_hash, _entry_is_complete) as staging:
            state_dir = staging / "state"
            state_dir.mkdir()

            if framework == "torch":
                self._save_torch(state_dir, sim_component)
            elif framework == "onnx":
                self._save_onnx(state_dir, sim_component)

            self._save_steps(staging, step_stats)
            self._save_metadata(
                staging,
                chain_hash,
                parent_hash,
                component,
                self.env_hash,
                model_id,
                len(step_stats),
            )

        log_cache_save(component, step_stats[-1], chain_hash)

    def load(
        self,
        chain_hash: str,
        sim_component,
        framework: str,
    ) -> list[RecipeStepStats]:
        """Load cached weight state into sim_component and return step stats.

        Every file is read before *sim_component* is modified, so an entry that
        cannot be read raises with the sim untouched.
        """
        return self._apply(self._read(chain_hash, framework), sim_component, framework)

    def _read(self, chain_hash: str, framework: str):
        """Read an entry's step stats and state, without touching any sim."""
        state_dir = self.cache_dir / chain_hash / "state"
        if not state_dir.exists():
            return [], None
        step_stats = self.load_step_stats(chain_hash)
        if framework == "torch":
            return step_stats, self._read_torch(state_dir)
        if framework == "onnx":
            return step_stats, self._read_onnx(state_dir)
        return step_stats, None

    def _apply(self, read, sim_component, framework: str) -> list[RecipeStepStats]:
        """Put state from :meth:`_read` into *sim_component*; return its step stats."""
        step_stats, state = read
        if state is None:
            return step_stats
        if framework == "torch":
            self._apply_torch(sim_component, *state)
        elif framework == "onnx":
            self._apply_onnx(sim_component, *state)
        return step_stats

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

    @staticmethod
    def _read_torch(state_dir: Path) -> tuple[dict | None, dict | None]:
        ckpt_path = state_dir / "checkpoint.pt"
        state_dict = (
            torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
        )
        flags = _read_json(state_dir / "quantizer_flags.json")
        return state_dict, flags

    def _apply_torch(self, sim_component, state_dict, flags):
        if state_dict is not None:
            sim_component.model.load_state_dict(state_dict, strict=False)
        if flags is not None:
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

    @staticmethod
    def _read_onnx(state_dir: Path):
        model_path = state_dir / "model.onnx"
        cached_model = (
            onnx.load(str(model_path), load_external_data=True)
            if model_path.exists()
            else None
        )
        encodings = _read_json(state_dir / "model.encodings")
        frozen_ops = _read_json(state_dir / "quantizer_flags.json")
        return cached_model, encodings, frozen_ops

    @staticmethod
    def _apply_onnx(sim_component, cached_model, encodings, frozen_ops):
        from aimet_onnx.quantsim import load_encodings_to_sim

        if cached_model is not None:
            # Copy initializers (weight tensors) from the cached proto into
            # into the live proto. Replacing the entire proto would bring
            # stale QcQuantizeOp quant_info C++ pointers, causing a segfault.
            cached_inits = {init.name: init for init in cached_model.graph.initializer}

            for i, init in enumerate(sim_component.model.model.graph.initializer):
                if init.name in cached_inits:
                    sim_component.model.model.graph.initializer[i].CopyFrom(
                        cached_inits[init.name]
                    )

            del cached_model, cached_inits
            sim_component._rebuild_session()

        if encodings is not None:
            load_encodings_to_sim(
                sim_component,
                encodings,
                strict=False,
                allow_overwrite=True,
                disable_missing_quantizers=False,
            )

        # Restore per-op frozen encoding flags
        if frozen_ops is not None:
            for name, was_frozen in frozen_ops.items():
                if was_frozen and name in sim_component.qc_quantize_op_dict:
                    sim_component.qc_quantize_op_dict[name].freeze_encodings()


def _read_json(path: Path):
    """The parsed contents of *path*, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


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


def chain_model_identity(
    model_id: str,
    context_length: int,
    sequence_length,
    image_size=None,
    audio_frames=None,
    encodings: str | None = None,
) -> dict:
    """The model-section fields a recipe chain depends on outside ``model_kwargs``.

    The parser takes these out of ``model_kwargs``, so the base hash would
    otherwise not see them: a chain calibrated at one context length would be
    served to a run at another. ``encodings`` is keyed by the content of its
    ``.encodings`` files, since the chain starts from whatever encodings were
    loaded into the sim; the model files an export directory also holds are
    not read.

    A sequence-length list is keyed by its maximum, so lists sharing one (e.g.
    ``[1, 128, 2048]`` and ``[1, 2048]``) share entries: SeqMSE and AdaScale
    sample at the maximum. Calibration runs each sample at the smallest listed
    length that fits it, so activation encodings it computed before a cached
    step can differ between such lists; that is accepted here for the reuse.
    Whether the graph is dynamic (a list of more than one length) is kept, as
    that exports a different graph.
    """
    lengths = (
        sequence_length if isinstance(sequence_length, list) else [sequence_length]
    )
    return {
        "context_length": context_length,
        "sequence_length": max(lengths),
        "dynamic": len(lengths) > 1,
        "image_size": image_size,
        "audio_frames": audio_frames,
        "encodings": file_digest(encodings, suffix=".encodings"),
        "revision": checkpoint_revision(model_id),
    }


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
