# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DiskBackedFPCache."""

import pytest
import torch

from GenAILab.bench.fp_cache import DiskBackedFPCache


@pytest.fixture
def cache(tmp_path):
    return DiskBackedFPCache(tmp_path / "fp_cache")


class TestDiskBackedFPCache:
    def test_put_and_get(self, cache):
        key = ("abc123", "mmlu_logits")
        data = {"logits": torch.randn(10, 4), "labels": torch.randint(0, 4, (10,))}
        cache.put(key, data)
        result = cache.get(key)
        assert result is not None
        assert torch.equal(result["logits"], data["logits"])
        assert torch.equal(result["labels"], data["labels"])

    def test_get_miss(self, cache):
        result = cache.get(("nonexistent", "collection"))
        assert result is None

    def test_get_or_compute_miss(self, cache):
        key = ("hash1", "ppl_logits")
        expected = {"loss": torch.tensor(2.5)}
        compute_fn = lambda: expected
        result = cache.get_or_compute(key, compute_fn)
        assert torch.equal(result["loss"], expected["loss"])

    def test_get_or_compute_cached(self, cache):
        key = ("hash1", "ppl_logits")
        data = {"loss": torch.tensor(2.5)}
        cache.put(key, data)
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return {"loss": torch.tensor(999.0)}

        result = cache.get_or_compute(key, compute_fn)
        assert call_count == 0
        assert torch.equal(result["loss"], data["loss"])

    def test_persistence_across_instances(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        key = ("hash1", "mmlu")
        data = {"logits": torch.randn(5, 4)}

        cache1 = DiskBackedFPCache(cache_dir)
        cache1.put(key, data)

        cache2 = DiskBackedFPCache(cache_dir)
        result = cache2.get(key)
        assert result is not None
        assert torch.equal(result["logits"], data["logits"])

    def test_clear(self, cache):
        key = ("hash1", "mmlu")
        cache.put(key, {"x": torch.tensor(1.0)})
        cache.clear()
        assert cache.get(key) is None

    def test_stale_index_entry(self, tmp_path):
        cache_dir = tmp_path / "fp_cache_stale"
        cache = DiskBackedFPCache(cache_dir)
        key = ("hash1", "mmlu")
        cache.put(key, {"x": torch.tensor(1.0)})
        # Delete the underlying file
        pt_path = cache._cache_dir / "hash1" / "mmlu.pt"
        pt_path.unlink()
        # Use a fresh cache instance (no memory layer) to test stale detection
        cache2 = DiskBackedFPCache(cache_dir)
        assert cache2.get(key) is None
        assert "hash1/mmlu" not in cache2._index["entries"]

    def test_index_metadata(self, cache):
        key = ("hash1", "mmlu")
        cache.put(key, {"x": torch.tensor(1.0)}, metadata={"model": "llama"})
        entry = cache._index["entries"]["hash1/mmlu"]
        assert entry["metadata"]["model"] == "llama"
        assert "created" in entry
        assert "size_bytes" in entry

    def test_memory_layer_hit(self, cache):
        key = ("hash1", "mmlu")
        data = {"x": torch.tensor(1.0)}
        cache.put(key, data)
        # First get loads from disk into memory
        cache.get(key)
        # Second get should serve from memory (same object)
        result = cache.get(key)
        assert result is cache._memory["hash1/mmlu"]


class TestSharedFPCache:
    """Several processes sharing one cache directory."""

    def test_concurrent_writers_keep_each_others_entries(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        first = DiskBackedFPCache(cache_dir)
        second = DiskBackedFPCache(cache_dir)
        first.put(("h", "a"), {"x": torch.tensor(1.0)})
        second.put(("h", "b"), {"x": torch.tensor(2.0)})

        index = DiskBackedFPCache(cache_dir)._index["entries"]
        assert set(index) == {"h/a", "h/b"}

    def test_get_sees_entry_put_by_another_instance(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        reader = DiskBackedFPCache(cache_dir)
        DiskBackedFPCache(cache_dir).put(("h", "a"), {"x": torch.tensor(1.0)})
        assert reader.get(("h", "a")) is not None

    def test_lookup_does_not_depend_on_the_index(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        DiskBackedFPCache(cache_dir).put(("h", "a"), {"x": torch.tensor(1.0)})
        (cache_dir / "index.json").write_text("{ not json")
        assert DiskBackedFPCache(cache_dir).get(("h", "a")) is not None
        # The damaged file is left for the next update to overwrite, not deleted
        # out from under a process that has just replaced it.
        assert (cache_dir / "index.json").exists()

    def test_stale_entry_is_dropped_from_the_shared_index(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        DiskBackedFPCache(cache_dir).put(("h", "a"), {"x": torch.tensor(1.0)})
        (cache_dir / "h" / "a.pt").unlink()
        fresh = DiskBackedFPCache(cache_dir)
        assert fresh.get(("h", "a")) is None
        assert "h/a" not in DiskBackedFPCache(cache_dir)._index["entries"]

    def test_put_leaves_no_temp_files(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        DiskBackedFPCache(cache_dir).put(("h", "a"), {"x": torch.tensor(1.0)})
        names = sorted(p.name for p in (cache_dir / "h").iterdir())
        assert names == ["a.pt"]


class TestUnreadableEntries:
    def test_truncated_entry_is_recomputed(self, tmp_path):
        cache_dir = tmp_path / "fp_cache"
        key = ("hash1", "mmlu")
        DiskBackedFPCache(cache_dir).put(key, {"x": torch.tensor(1.0)})
        (cache_dir / "hash1" / "mmlu.pt").write_bytes(b"trunc")

        cache = DiskBackedFPCache(cache_dir)
        assert cache.get(key) is None
        result = cache.get_or_compute(key, lambda: {"x": torch.tensor(2.0)})
        assert torch.equal(result["x"], torch.tensor(2.0))
        assert torch.equal(
            DiskBackedFPCache(cache_dir).get(key)["x"], torch.tensor(2.0)
        )
