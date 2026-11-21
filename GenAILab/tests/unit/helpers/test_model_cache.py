# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DiskBackedModelCache."""

import json
import os
import time

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


class TestUnusableEntries:
    """An entry that lost files or will not load is a miss, not a crash."""

    def _stored(self, tmp_path):
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        cache.put("key_a", _tiny_entry(), metadata={"model_id": "org/model"})
        return cache, tmp_path / "model_cache" / "key_a"

    def test_entry_that_lost_a_file_is_discarded(self, tmp_path):
        cache, entry = self._stored(tmp_path)
        (entry / "config.json").unlink()
        assert cache.get("key_a") is None
        assert not entry.exists()
        assert "key_a" not in cache._load_index()["entries"]

    def test_entry_that_fails_to_load_is_discarded(self, tmp_path, monkeypatch):
        cache, entry = self._stored(tmp_path)

        def corrupt(*args, **kwargs):
            raise OSError("stale file handle")

        monkeypatch.setattr(onnx, "load", corrupt)
        assert cache.get("key_a") is None
        assert not entry.exists()

    def test_discarded_entry_is_re_exported(self, tmp_path):
        cache, entry = self._stored(tmp_path)
        (entry / "config.json").unlink()
        calls = []

        def export():
            calls.append(1)
            return _tiny_entry()

        assert cache.get_or_export("key_a", export) is not None
        assert calls == [1]
        assert cache.get("key_a") is not None

    def test_stale_staging_is_swept_on_open(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        leftover = cache_dir / ".key_a.123.deadbeef"
        leftover.mkdir(parents=True)
        old = time.time() - 3 * 24 * 3600
        os.utime(leftover, (old, old))
        DiskBackedModelCache(cache_dir)
        assert not leftover.exists()

    def test_clear_removes_entries(self, tmp_path):
        cache, entry = self._stored(tmp_path)
        cache.clear()
        assert cache.get("key_a") is None
        assert list((tmp_path / "model_cache").iterdir()) == []


class TestExportIdentity:
    """The model cache key covers everything an export depends on."""

    PARAMS = {"model_id": "org/model", "context_length": 64, "sequence_length": 32}

    @pytest.fixture(autouse=True)
    def _pinned(self, monkeypatch):
        import GenAILab.bench.model_cache as model_cache

        monkeypatch.setattr(model_cache, "checkpoint_revision", lambda _: "rev1")
        monkeypatch.setattr(model_cache, "model_fingerprint", lambda: {"code": "c1"})
        return model_cache

    def _key(self, model_cls=object, params=None):
        return DiskBackedModelCache.build_key(
            DiskBackedModelCache.export_identity(params or self.PARAMS, model_cls)
        )

    def test_stable(self):
        assert self._key() == self._key()

    def test_adaptation_kwargs_change_the_key(self):
        """Adaptation kwargs are class attributes, invisible in the class name."""

        class Half:
            adaptation_identity = [["AttentionMaskScale", [(0, 0.5)]]]

        class Quarter:
            adaptation_identity = [["AttentionMaskScale", [(0, 0.25)]]]

        assert self._key(Half) != self._key(Quarter)

    def test_revision_changes_the_key(self, _pinned, monkeypatch):
        before = self._key()
        monkeypatch.setattr(_pinned, "checkpoint_revision", lambda _: "rev2")
        assert self._key() != before

    def test_fingerprint_changes_the_key(self, _pinned, monkeypatch):
        before = self._key()
        monkeypatch.setattr(_pinned, "model_fingerprint", lambda: {"code": "c2"})
        assert self._key() != before

    def test_params_still_count(self):
        other = {**self.PARAMS, "context_length": 128}
        assert self._key(params=other) != self._key()
