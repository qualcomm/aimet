# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers that keep the on-disk caches safe to share between processes."""

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
