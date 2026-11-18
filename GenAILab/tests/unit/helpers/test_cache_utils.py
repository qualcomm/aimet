# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the helpers that make the disk caches safe to share."""

import json

import pytest

from GenAILab.bench.utils.cache_utils import (
    atomic_write_json,
    discard_entry,
    staged_entry,
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
