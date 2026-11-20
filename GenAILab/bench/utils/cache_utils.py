# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for the on-disk caches: what entries key on, and sharing them safely.

An entry is only reusable while everything that produced it is unchanged. The
config is the obvious part; the fingerprints here cover the rest, so an entry
made by other code, other package versions or other checkpoint weights is never
served. There is one per cache:

* :func:`model_fingerprint` -- an ONNX export. The export runs no AIMET code,
  so an AIMET-only change keeps cached exports.
* :func:`recipe_fingerprint` -- a recipe chain step, which runs everything the
  export does and AIMET as well.

Both include GenAILab's own code, so an edit to it misses, except for the
evaluation code in :data:`EVAL_ONLY` that no cached step runs. Source is hashed
by file content, so the fingerprints work with or without a ``.git`` directory
(code synced to a cluster has none). Each is computed once per process: the
code does not change while it runs.

Several processes may share one cache directory; :func:`staged_entry`,
:func:`discard_entry` and :func:`atomic_write_json` keep that safe.
"""

import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def _unique_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")


@contextmanager
def staged_entry(final: Path, is_complete: Callable[[Path], bool]) -> Iterator[Path]:
    """Yield a private directory to fill, then rename it to *final*.

    Readers only ever see *final* whole: it appears in one atomic rename, so an
    interrupted or concurrent writer never leaves a half-written entry behind.
    If another process published *final* first, its entry is kept and this
    one is discarded (entries are content-addressed, so they are equivalent).
    A *final* that ``is_complete`` rejects is a leftover of an older, non-atomic
    writer and is replaced.

    The staging directory is a sibling of *final*, so the rename stays on one
    filesystem, and it is created with the process umask like any other cache
    directory.
    """
    staging = _unique_sibling(final)
    staging.mkdir(parents=True)
    try:
        yield staging
        _publish(staging, final, is_complete)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _publish(staging: Path, final: Path, is_complete: Callable[[Path], bool]):
    try:
        os.rename(staging, final)
        return
    except OSError:
        if not final.exists():
            raise
    if is_complete(final):
        return
    shutil.rmtree(final, ignore_errors=True)
    try:
        os.rename(staging, final)
    except OSError:
        # Lost the race to a writer that replaced the leftover as well.
        if not is_complete(final):
            raise


def discard_entry(entry: Path):
    """Remove *entry* so a reader never finds it half-deleted.

    It is renamed away in one step, then deleted, so a process looking the entry
    up meanwhile sees it either whole or missing. Another process may have
    discarded it already.
    """
    doomed = _unique_sibling(entry)
    try:
        os.rename(entry, doomed)
    except FileNotFoundError:
        return
    shutil.rmtree(doomed, ignore_errors=True)


def atomic_write_json(path: Path, obj, **dump_kwargs):
    """Write *obj* as JSON so a concurrent reader sees the old file or the new one."""
    tmp_path = _unique_sibling(path)
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, **dump_kwargs)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


_GENAILAB_ROOT = Path(__file__).resolve().parents[2]

# The GenAILab packages whose code can run while a cache entry is computed.
# Run configs, tests and tooling live outside them, so editing those keeps
# cache entries.
_GENAILAB_PACKAGES = ("bench", "qai_hub_lm")

# Evaluation and reporting code, relative to GenAILab/: it runs after a model is
# exported and quantized, never while a cache entry is computed, so editing it
# keeps both caches' entries. Only evaluation entry points may import it;
# tests/unit/helpers/test_cache_fingerprint.py fails if any other module does,
# since then it could shape a cached step after all.
EVAL_ONLY = (
    "qai_hub_lm/scoring",
    "bench/metrics.py",
    "bench/summary.py",
    "bench/analysis.py",
    "bench/grade_responses.py",
    "bench/eval_context.py",
    "bench/fp_cache.py",
    "bench/onnx/analysis_passes.py",
    "bench/onnx/llm_layer_sensitivity.py",
)

# Packages that shape an exported graph.
_EXPORT_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "onnx",
    "onnxscript",
    "accelerate",
)

# Packages that shape a recipe step on top of the export. AIMET's own code is
# hashed file by file (see _aimet_hash), not by version: a dev build keeps its
# version string across changes.
_RECIPE_DISTRIBUTIONS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "datasets",
    "tokenizers",
    "numpy",
    "aimet-onnx",
    "aimet-torch",
)

_AIMET_PACKAGES = ("aimet_onnx", "aimet_torch")


def _digest(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _hash_files(root: Path, keep=lambda rel: True) -> str:
    """Hash the files under *root* that *keep* accepts, by relative path and content.

    *keep* gets each path relative to *root*. Byte-code caches are always
    skipped.
    """
    hasher = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            rel = (rel_dir / name).as_posix()
            if name.endswith(".pyc") or not keep(rel):
                continue
            try:
                data = (root / rel).read_bytes()
            except OSError:
                continue
            hasher.update(rel.encode() + b"\0")
            hasher.update(hashlib.sha256(data).digest())
    return hasher.hexdigest()[:32]


def _versions(names) -> dict:
    found = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def is_eval_only(rel: str) -> bool:
    """Whether *rel* (relative to GenAILab/) is in :data:`EVAL_ONLY`."""
    return any(rel == p or rel.startswith(p + "/") for p in EVAL_ONLY)


@functools.cache
def _genailab_hash() -> str:
    """Every file in GenAILab's packages, except docs and evaluation code."""
    return _digest(
        {
            package: _hash_files(
                _GENAILAB_ROOT / package,
                keep=lambda rel, package=package: not rel.endswith(".md")
                and not is_eval_only(f"{package}/{rel}"),
            )
            for package in _GENAILAB_PACKAGES
        }
    )


@functools.cache
def _aimet_hash() -> str:
    """The Python and data files of the AIMET packages this process imports.

    Hashed where each package is imported from (the directory ``pip show``
    reports for an installed wheel, or the source tree when it is on
    ``PYTHONPATH``), so the quantsim config files count as well as the code.

    Compiled extensions are left out: CI rebuilds them on every run and the
    build is not known to be byte-reproducible, so they could make every run
    miss. A change to AIMET's C++/CUDA alone is therefore not seen while the
    package version stays the same; clear the recipe cache after one.
    """
    installed = {}
    for name in _AIMET_PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            continue
        installed[name] = [
            _hash_files(
                Path(location), keep=lambda rel: ".so" not in Path(rel).suffixes
            )
            for location in spec.submodule_search_locations
        ]
    return _digest(installed)


@functools.cache
def model_fingerprint() -> dict:
    """What an ONNX export depends on besides its config."""
    return {
        "packages": _versions(_EXPORT_DISTRIBUTIONS),
        "genailab": _genailab_hash(),
    }


@functools.cache
def recipe_fingerprint() -> dict:
    """What a recipe step depends on besides its config and the model it runs on.

    Includes the GPU model: recipe numerics differ between GPU generations, so
    an entry is only what a fresh run on this machine would produce if it was
    made on the same kind of GPU.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        gpu = None
    return {
        "model": model_fingerprint(),
        "packages": _versions(_RECIPE_DISTRIBUTIONS),
        "aimet": _aimet_hash(),
        "gpu": gpu,
    }


@functools.cache
def checkpoint_revision(model_id: str) -> str | None:
    """The commit a Hub checkpoint resolves to, so an upstream update misses.

    Covers weight-only updates that leave the config untouched. A local
    directory has no revision; its resolved path is used instead.

    When the Hub cannot be asked (offline, a rate limit, a server error), this
    is the commit of the locally cached snapshot, which is what
    ``from_pretrained`` then loads: a transient Hub error keeps the key rather
    than missing every entry. ``None`` only when neither is available.
    """
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    # pylint: disable=import-outside-toplevel
    from huggingface_hub import HfApi, try_to_load_from_cache

    try:
        return HfApi().model_info(model_id).sha
    except Exception as e:  # pylint: disable=broad-exception-caught
        hub_error = e
    # The cached file sits at <cache>/models--<org>--<name>/snapshots/<commit>/.
    cached = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(cached, str):
        return None
    revision = Path(cached).parent.name
    print(
        f"  [cache] Could not ask the Hub for {model_id}'s revision "
        f"({type(hub_error).__name__}); keying on the cached snapshot {revision[:12]}"
    )
    return revision


def file_digest(path: str | Path | None, suffix: str = "") -> str | None:
    """Content hash of a file, or of the files under a directory, else ``None``.

    Under a directory only files ending in *suffix* are hashed, so a directory
    holding a large model beside small encodings files hashes just the latter.
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return _hash_files(path, keep=lambda rel: rel.endswith(suffix))
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:32]
    return None
