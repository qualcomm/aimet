# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DiskBackedModelCache."""

import json

import onnx
import pytest
from onnx import TensorProto, helper
from transformers import LlamaConfig

from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry


def _tiny_entry():
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return ModelCacheEntry(backbone=helper.make_model(graph), config=LlamaConfig())


class TestBuildKey:
    def test_deterministic(self):
        params = {"model_id": "org/model", "seq": 32, "ctx": 64}
        k1 = DiskBackedModelCache.build_key(params)
        k2 = DiskBackedModelCache.build_key(params)
        assert k1 == k2

    def test_order_independent(self):
        k1 = DiskBackedModelCache.build_key({"a": 1, "b": 2})
        k2 = DiskBackedModelCache.build_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_different_params_different_keys(self):
        k1 = DiskBackedModelCache.build_key({"model_id": "a"})
        k2 = DiskBackedModelCache.build_key({"model_id": "b"})
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = DiskBackedModelCache.build_key({"x": 1})
        assert len(key) == 16
        int(key, 16)  # Should not raise


class TestDiskBackedModelCache:
    def test_cache_miss(self, tmp_path):
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        assert cache.get("nonexistent_key") is None

    def test_clear(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        cache = DiskBackedModelCache(cache_dir)
        # Just verify clear doesn't crash on empty cache
        cache.clear()
        assert cache._index["entries"] == {}

    def test_index_persists(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        cache1 = DiskBackedModelCache(cache_dir)
        # Manually add an entry to test index persistence
        cache1._index["entries"]["test_key"] = {"metadata": {}, "created": "now"}
        cache1._save_index()

        cache2 = DiskBackedModelCache(cache_dir)
        assert "test_key" in cache2._index["entries"]

    def test_concurrent_writers_keep_each_others_entries(self, tmp_path):
        # Two runs share one cache: both load the (empty) index up front, then
        # each finishes its own export. The later put must not drop the earlier.
        cache_dir = tmp_path / "model_cache"
        first = DiskBackedModelCache(cache_dir)
        second = DiskBackedModelCache(cache_dir)
        first.put("key_a", _tiny_entry())
        second.put("key_b", _tiny_entry())

        reader = DiskBackedModelCache(cache_dir)
        assert reader.get("key_a") is not None
        assert reader.get("key_b") is not None

    def test_get_sees_entry_put_by_another_instance(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        reader = DiskBackedModelCache(cache_dir)
        DiskBackedModelCache(cache_dir).put("key_a", _tiny_entry())
        assert reader.get("key_a") is not None

    def test_lookup_does_not_depend_on_the_index(self, tmp_path):
        # Another process published the entry but its index update was lost.
        cache_dir = tmp_path / "model_cache"
        DiskBackedModelCache(cache_dir).put("key_a", _tiny_entry(), metadata={"k": 1})
        (cache_dir / "index.json").unlink()
        assert DiskBackedModelCache(cache_dir).get("key_a") is not None

    def test_first_published_entry_wins(self, tmp_path):
        cache = DiskBackedModelCache(tmp_path / "model_cache")

        def export_while_another_process_finishes_first():
            DiskBackedModelCache(tmp_path / "model_cache").put(
                "key_a", _tiny_entry(), metadata={"writer": "other"}
            )
            return _tiny_entry()

        cache.get_or_export(
            "key_a", export_while_another_process_finishes_first, {"writer": "us"}
        )
        meta = json.loads(
            (tmp_path / "model_cache" / "key_a" / "entry.json").read_text()
        )
        assert meta["metadata"] == {"writer": "other"}

    def test_failed_export_leaves_no_entry_or_temp_directory(
        self, tmp_path, monkeypatch
    ):
        cache_dir = tmp_path / "model_cache"
        cache = DiskBackedModelCache(cache_dir)

        def fail(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(onnx, "save_model", fail)
        with pytest.raises(RuntimeError, match="disk full"):
            cache.put("key_a", _tiny_entry())
        assert cache.get("key_a") is None
        assert [p.name for p in cache_dir.iterdir()] == []
