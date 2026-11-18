# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""RecipeCache entries are safe to share between processes.

An entry is published whole by a rename, so ``metadata.json`` existing means the
state is there too. Before that, ``metadata.json`` was written first and the
state afterwards: a second process (or a re-run after a crash) saw a hit, loaded
nothing, and carried on with an uncalibrated sim.

recipe_cache imports onnx at module top, so this skips where onnx is
unavailable and runs in CI / on the cluster.
"""

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
    # The aimet sim is out of scope: only the entry's layout is under test.
    monkeypatch.setattr(
        RecipeCache,
        "_save_torch",
        lambda self, state_dir, sim: (state_dir / "checkpoint.pt").write_text("w"),
    )
    monkeypatch.setattr(
        "GenAILab.bench.recipe_cache.convert_gpu_meter_to_dict",
        lambda profiler, **kwargs: {"elapsed_ms": profiler.elapsed_ms},
    )
    return cache


def _save(cache, chain_hash="c1"):
    step = RecipeStepStats("Calibration", {}, "", {}, _Profiler())
    cache.save(chain_hash, "parent", [step], None, "torch", model_id="m")


class TestRecipeCacheSharing:
    def test_saved_entry_is_whole(self, cache):
        _save(cache)
        entry = cache.cache_dir / "c1"
        assert cache.get("c1")["step_count"] == 1
        assert (entry / "steps.json").exists()
        assert (entry / "state" / "checkpoint.pt").exists()
        assert [p.name for p in cache.cache_dir.iterdir()] == ["c1"]

    def test_interrupted_save_is_not_a_hit(self, cache, monkeypatch):
        def die_mid_save(self, state_dir, sim):
            (state_dir / "checkpoint.pt").write_text("half")
            raise KeyboardInterrupt

        monkeypatch.setattr(RecipeCache, "_save_torch", die_mid_save)
        with pytest.raises(KeyboardInterrupt):
            _save(cache)
        assert cache.get("c1") is None
        assert list(cache.cache_dir.iterdir()) == []

    def test_second_saver_keeps_the_first_entry(self, cache, monkeypatch):
        original = RecipeCache._save_torch

        def other_process_finishes_first(self, state_dir, sim):
            monkeypatch.setattr(RecipeCache, "_save_torch", original)
            _save(cache)  # the other process publishes "c1" ...
            (state_dir / "checkpoint.pt").write_text("ours")

        monkeypatch.setattr(RecipeCache, "_save_torch", other_process_finishes_first)
        _save(cache)  # ... while this one is still writing it.
        assert (cache.cache_dir / "c1" / "state" / "checkpoint.pt").read_text() == "w"
        assert [p.name for p in cache.cache_dir.iterdir()] == ["c1"]

    def test_pre_atomic_partial_entry_is_replaced(self, cache):
        # What the old writer left behind if killed: metadata but no state.
        leftover = cache.cache_dir / "c1"
        leftover.mkdir()
        (leftover / "metadata.json").write_text("{}")
        _save(cache)
        assert (leftover / "state" / "checkpoint.pt").exists()
