# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the helpers that make the disk caches safe to share."""

import json
import os
import shutil
import time

import pytest

from GenAILab.bench.utils.cache_utils import (
    MANIFEST_FILE,
    STALE_STAGING_SECONDS,
    atomic_write_json,
    clear_dir,
    discard_entry,
    entry_is_complete,
    staged_entry,
    sweep_stale_staging,
)


def _complete(entry):
    return (entry / "done").exists()


def _fill(directory, text="x"):
    (directory / "done").write_text(text)


class TestStagedEntry:
    def test_publishes_on_success(self, tmp_path):
        final = tmp_path / "entry"
        with staged_entry(final, _complete) as staging:
            _fill(staging)
            assert not final.exists()
        assert _complete(final)
        assert [p.name for p in tmp_path.iterdir()] == ["entry"]
        assert (final / MANIFEST_FILE).exists()

    def test_failure_leaves_nothing(self, tmp_path):
        final = tmp_path / "entry"
        with pytest.raises(RuntimeError, match="boom"):
            with staged_entry(final, _complete) as staging:
                _fill(staging)
                raise RuntimeError("boom")
        assert list(tmp_path.iterdir()) == []

    def test_first_writer_wins(self, tmp_path):
        final = tmp_path / "entry"
        with staged_entry(final, _complete) as first:
            with staged_entry(final, _complete) as second:
                _fill(first, "first")
                _fill(second, "second")
            # The second finished first; the first finds it and steps aside.
            assert (final / "done").read_text() == "second"
        assert (final / "done").read_text() == "second"
        assert [p.name for p in tmp_path.iterdir()] == ["entry"]

    def test_replaces_an_incomplete_leftover(self, tmp_path):
        final = tmp_path / "entry"
        final.mkdir()
        (final / "partial").write_text("from an older, non-atomic writer")
        with staged_entry(final, _complete) as staging:
            _fill(staging)
        assert _complete(final)
        assert not (final / "partial").exists()


class TestDiscardEntry:
    def test_removes_the_entry(self, tmp_path):
        entry = tmp_path / "entry"
        entry.mkdir()
        _fill(entry)
        discard_entry(entry)
        assert list(tmp_path.iterdir()) == []

    def test_missing_entry_is_fine(self, tmp_path):
        discard_entry(tmp_path / "gone")


class TestAtomicWriteJson:
    def test_round_trip_and_no_temp_left(self, tmp_path):
        path = tmp_path / "index.json"
        atomic_write_json(path, {"a": 1}, indent=2)
        atomic_write_json(path, {"a": 2}, indent=2)
        assert json.loads(path.read_text()) == {"a": 2}
        assert [p.name for p in tmp_path.iterdir()] == ["index.json"]

    def test_failed_dump_keeps_the_old_file(self, tmp_path):
        path = tmp_path / "index.json"
        atomic_write_json(path, {"a": 1})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"a": object()})
        assert json.loads(path.read_text()) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


class TestManifest:
    def _published(self, tmp_path):
        final = tmp_path / "entry"
        with staged_entry(final, _complete) as staging:
            (staging / "state").mkdir()
            (staging / "state" / "model.data").write_bytes(b"weights")
            _fill(staging)
        return final

    def test_every_file_is_listed(self, tmp_path):
        final = self._published(tmp_path)
        files = json.loads((final / MANIFEST_FILE).read_text())["files"]
        assert files == {"done": 1, "state/model.data": 7}

    def test_complete_while_every_file_is_there(self, tmp_path):
        assert entry_is_complete(self._published(tmp_path), _complete)

    def test_missing_file_is_incomplete(self, tmp_path):
        """What an expiring filer purging file by file leaves behind."""
        final = self._published(tmp_path)
        (final / "state" / "model.data").unlink()
        assert not entry_is_complete(final, _complete)

    def test_truncated_file_is_incomplete(self, tmp_path):
        final = self._published(tmp_path)
        (final / "state" / "model.data").write_bytes(b"wei")
        assert not entry_is_complete(final, _complete)

    def test_corrupt_manifest_is_incomplete(self, tmp_path):
        final = self._published(tmp_path)
        (final / MANIFEST_FILE).write_text("{not json")
        assert not entry_is_complete(final, _complete)

    def test_entry_without_manifest_uses_legacy_check(self, tmp_path):
        """Entries written before manifests existed stay usable."""
        old = tmp_path / "old"
        old.mkdir()
        _fill(old)
        assert entry_is_complete(old, _complete)
        assert not entry_is_complete(tmp_path / "missing", _complete)

    def test_publishing_replaces_an_entry_that_lost_files(self, tmp_path):
        final = self._published(tmp_path)
        (final / "state" / "model.data").unlink()

        def complete(entry):
            return entry_is_complete(entry, _complete)

        with staged_entry(final, complete) as staging:
            (staging / "state").mkdir()
            (staging / "state" / "model.data").write_bytes(b"new weights")
            _fill(staging)
        assert (final / "state" / "model.data").read_bytes() == b"new weights"


class TestClearDir:
    def test_empties_and_keeps_the_directory(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "entry").mkdir(parents=True)
        _fill(cache / "entry")
        clear_dir(cache)
        assert cache.is_dir() and list(cache.iterdir()) == []
        assert [p.name for p in tmp_path.iterdir()] == ["cache"]

    def test_missing_directory_is_created(self, tmp_path):
        clear_dir(tmp_path / "cache")
        assert (tmp_path / "cache").is_dir()

    def test_entries_never_seen_half_deleted(self, tmp_path, monkeypatch):
        """The old directory is moved away before anything inside is deleted."""
        cache = tmp_path / "cache"
        (cache / "entry").mkdir(parents=True)
        _fill(cache / "entry")
        seen = []
        real_rmtree = shutil.rmtree

        def watching_rmtree(path, *args, **kwargs):
            seen.append(sorted(p.name for p in cache.iterdir()))
            real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", watching_rmtree)
        clear_dir(cache)
        # Only the private, moved-aside copy is there while it is deleted.
        assert len(seen) == 1 and len(seen[0]) == 1
        assert seen[0][0].startswith(".")
        assert list(cache.iterdir()) == []

    def test_interrupted_delete_is_left_for_the_sweep(self, tmp_path, monkeypatch):
        """Never leaves anything beside the cache, in a directory it does not own."""
        cache = tmp_path / "cache"
        (cache / "entry").mkdir(parents=True)
        _fill(cache / "entry")
        monkeypatch.setattr(shutil, "rmtree", lambda *args, **kwargs: None)
        clear_dir(cache)
        assert [p.name for p in tmp_path.iterdir()] == ["cache"]
        (leftover,) = cache.iterdir()
        assert leftover.name.startswith(".")
        monkeypatch.undo()
        old = time.time() - 2 * STALE_STAGING_SECONDS
        os.utime(leftover, (old, old))
        sweep_stale_staging(cache)
        assert list(cache.iterdir()) == []


class TestSweepStaleStaging:
    def _age(self, path, seconds):
        old = time.time() - seconds
        os.utime(path, (old, old))

    def test_removes_old_leftovers_only(self, tmp_path):
        entry = tmp_path / "abc123"
        stale = tmp_path / ".abc123.111.deadbeef"
        fresh = tmp_path / ".abc123.222.cafef00d"
        stale_file = tmp_path / ".index.json.333.feedface"
        for d in (entry, stale, fresh):
            d.mkdir()
            _fill(d)
        stale_file.write_text("{}")
        for p in (entry, stale, stale_file):
            self._age(p, 2 * STALE_STAGING_SECONDS)
        sweep_stale_staging(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            [entry.name, fresh.name]
        )

    def test_missing_directory_is_fine(self, tmp_path):
        sweep_stale_staging(tmp_path / "missing")
