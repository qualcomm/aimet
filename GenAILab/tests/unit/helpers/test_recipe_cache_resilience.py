# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""A recipe cache entry that lost files or will not load is a miss, not a crash.

The lookup falls back to the next shorter cached prefix, and a failed load never
leaves the sim half-overwritten: every file is read before the sim is touched.

recipe_cache imports onnx at module top, so this skips where onnx is
unavailable and runs in CI / on the cluster.
"""

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("onnx", reason="recipe_cache imports onnx at module top")

from GenAILab.bench.profiler import RecipeStepStats  # noqa: E402
from GenAILab.bench.recipe_cache import RecipeCache  # noqa: E402


class _Profiler:
    elapsed_ms = 1.0


@pytest.fixture
def cache(tmp_path, monkeypatch):
    cache = RecipeCache(tmp_path / "recipe")
    monkeypatch.setattr(RecipeCache, "env_hash", "envhash")
    monkeypatch.setattr(
        RecipeCache,
        "_save_torch",
        lambda self, state_dir, sim: (state_dir / "checkpoint.pt").write_text(
            sim["tag"]
        ),
    )
    monkeypatch.setattr(
        RecipeCache,
        "_read_torch",
        staticmethod(
            lambda state_dir: ((state_dir / "checkpoint.pt").read_text(), None)
        ),
    )
    monkeypatch.setattr(
        RecipeCache,
        "_apply_torch",
        lambda self, sim, state, flags: sim.__setitem__("loaded", state),
    )
    monkeypatch.setattr(
        "GenAILab.bench.recipe_cache.convert_gpu_meter_to_dict",
        lambda profiler, **kwargs: {"elapsed_ms": profiler.elapsed_ms},
    )
    return cache


def _step(name):
    step = MagicMock()
    step.name = name
    step.technique_cls.__name__ = name
    step.recipe_kwargs = {}
    step.dataset_cls = None
    step.dataset_kwargs = {}
    return step


class _Precision:
    def weight_identity(self):
        return "w4a16"


STEPS = [_step("SeqMSE"), _step("AdaScale")]


def _lookup(cache, sim):
    return cache.lookup(STEPS, sim, "m", _Precision(), {}, "torch")


def _save_chain(cache):
    """Save entries after step 1 and step 2, as a run of the chain does."""
    _, _, hashes = _lookup(cache, {})
    for i, tag in ((1, "after-seqmse"), (2, "after-adascale")):
        stats = [RecipeStepStats(s.name, {}, "", {}, _Profiler()) for s in STEPS[:i]]
        cache.save(hashes[i], hashes[i - 1], stats, {"tag": tag}, "torch", model_id="m")
    return hashes


class TestRecipeCacheResilience:
    def test_whole_chain_hits(self, cache):
        _save_chain(cache)
        sim = {}
        skip, stats, _ = _lookup(cache, sim)
        assert skip == 2 and len(stats) == 2
        assert sim["loaded"] == "after-adascale"

    def test_entry_that_lost_a_file_falls_back_to_shorter_prefix(self, cache):
        hashes = _save_chain(cache)
        (cache.cache_dir / hashes[2] / "state" / "checkpoint.pt").unlink()
        sim = {}
        skip, _, _ = _lookup(cache, sim)
        assert skip == 1
        assert sim["loaded"] == "after-seqmse"
        assert not (cache.cache_dir / hashes[2]).exists()

    def test_entry_that_fails_to_load_falls_back(self, cache, monkeypatch):
        hashes = _save_chain(cache)
        real_read = RecipeCache._read_torch

        def flaky(state_dir):
            if state_dir.parent.name == hashes[2]:
                raise OSError("stale file handle")
            return real_read(state_dir)

        monkeypatch.setattr(RecipeCache, "_read_torch", staticmethod(flaky))
        sim = {}
        skip, _, _ = _lookup(cache, sim)
        assert skip == 1
        assert sim["loaded"] == "after-seqmse"
        assert not (cache.cache_dir / hashes[2]).exists()

    def test_failed_read_leaves_the_sim_untouched(self, cache, monkeypatch):
        _save_chain(cache)

        def broken(state_dir):
            raise OSError("truncated")

        monkeypatch.setattr(RecipeCache, "_read_torch", staticmethod(broken))
        sim = {}
        skip, stats, _ = _lookup(cache, sim)
        assert skip == 0 and stats == []
        assert sim == {}

    def test_failure_while_applying_is_raised(self, cache, monkeypatch):
        """The sim may be half-modified then, so no run may continue on it."""
        _save_chain(cache)

        def mismatch(self, sim, state, flags):
            sim["loaded"] = "partial"
            raise RuntimeError("shape mismatch")

        monkeypatch.setattr(RecipeCache, "_apply_torch", mismatch)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            _lookup(cache, {})

    def test_corrupt_metadata_is_a_miss(self, cache):
        hashes = _save_chain(cache)
        entry = cache.cache_dir / hashes[2]
        (entry / "metadata.json").write_text("{trunc")
        manifest = json.loads((entry / "files.json").read_text())
        manifest["files"]["metadata.json"] = len("{trunc")
        (entry / "files.json").write_text(json.dumps(manifest))
        assert cache.get(hashes[2]) is None
        assert not entry.exists()
