# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Lay out a docs build the way aimet-pages serves it, so only releases/latest/ is indexed.

Sphinx builds releases/<version>/ once. Every page carries
<meta name="robots" content="noindex,follow" /> (added in conf.py) and a canonical link to
the same page under releases/latest/ (from html_baseurl), and Lumina writes sitemap.xml and
robots.txt. This script then:

- copies the build to releases/latest/ and removes exactly that noindex tag there, leaving
  any other robots meta (e.g. Sphinx's own noindex on search.html) untouched. Its canonical
  links point at the pages themselves, and its sitemap.xml stays.
- removes the canonical links and sitemap.xml from releases/<version>/, which keeps only
  noindex. Google advises against combining noindex with rel=canonical.
- deletes robots.txt from both. Crawlers only read robots.txt at the host root, so a copy
  in a subdirectory does nothing.

    python3 make_publish_layout.py <releases/version dir> <releases/latest dir>
"""

import os
import re
import shutil
import sys

NOINDEX = re.compile(
    rb'\r?\n?[ \t]*<meta\s+name="robots"\s+content="noindex,\s*follow"\s*/?>', re.I
)
NOINDEX_LEFTOVER = re.compile(rb"<meta[^>]*robots[^>]*noindex\s*,\s*follow", re.I)
CANONICAL = re.compile(
    rb'\r?\n?[ \t]*<link\s+rel="canonical"\s+href="[^"]*"\s*/?>', re.I
)
CANONICAL_ANY = re.compile(rb"<link[^>]*rel=[\"']?canonical[^>]*>", re.I)
CANONICAL_HREF = re.compile(rb"href=[\"']([^\"']*)[\"']", re.I)
SITEMAP = "sitemap.xml"
ROBOTS = "robots.txt"


def _html_files(root):
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".html"):
                yield os.path.join(d, f)


def _strip(path, pattern):
    """Remove every match of pattern from the file and return the new contents."""
    with open(path, "rb") as fh:
        data = fh.read()
    new, n = pattern.subn(b"", data)
    if n:
        with open(path, "wb") as fh:
            fh.write(new)
    return new


def _remove(path):
    if os.path.exists(path):
        os.remove(path)


def _make_latest(dst, errors):
    """Make releases/latest/ indexable and check its canonicals point at the pages themselves."""
    pages = 0
    for path in _html_files(dst):
        new = _strip(path, NOINDEX)
        # A tag we failed to match would silently keep latest out of search engines.
        if NOINDEX_LEFTOVER.search(new):
            errors.append(f"noindex,follow still present in {path}")
        rel = os.path.relpath(path, dst).replace(os.sep, "/")
        for tag in CANONICAL_ANY.findall(new):
            href = CANONICAL_HREF.search(tag)
            if not href or not href.group(1).decode().endswith(
                "/releases/latest/" + rel
            ):
                errors.append(
                    f"canonical in {path} does not point at itself: {tag.decode()}"
                )
        if rel == "index.html" and not CANONICAL_ANY.search(new):
            errors.append(f"no canonical link in {path}; is html_baseurl set?")
        pages += 1
    if not os.path.isfile(os.path.join(dst, SITEMAP)):
        errors.append(f"{os.path.join(dst, SITEMAP)} is missing")
    return pages


def _make_versioned(src, errors):
    """Leave releases/<version>/ with noindex only: no canonical links, no sitemap.xml."""
    _remove(os.path.join(src, SITEMAP))
    pages = 0
    for path in _html_files(src):
        if CANONICAL_ANY.search(_strip(path, CANONICAL)):
            errors.append(f"canonical link still present in {path}")
        pages += 1
    return pages


def main(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("source and destination must differ")

    _remove(os.path.join(src, ROBOTS))
    # Recreate from scratch so pages dropped from this release disappear from latest too.
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".doctrees"))

    errors = []
    latest_pages = _make_latest(dst, errors)
    versioned_pages = _make_versioned(src, errors)
    for d in (src, dst):
        if os.path.exists(os.path.join(d, ROBOTS)):
            errors.append(f"{os.path.join(d, ROBOTS)} is present")

    if errors:
        raise SystemExit(
            f"{len(errors)} docs layout errors, e.g.:\n" + "\n".join(errors[:10])
        )
    print(
        f"{dst}: {latest_pages} pages without noindex,follow, self-canonicals, {SITEMAP}; "
        f"{src}: {versioned_pages} noindex pages without canonicals or {SITEMAP}; "
        f"no {ROBOTS} in either"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
