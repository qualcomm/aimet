# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Copy a versioned docs build to releases/latest/ and make the copy indexable.

Every page Sphinx builds carries <meta name="robots" content="noindex,follow" /> (added in
conf.py) so the releases/<version>/ copies published to aimet-pages stay out of search
results. This removes exactly that tag from the latest copy, leaving any other robots meta
(e.g. Sphinx's own noindex on search.html) untouched.

    python3 make_latest_docs.py <releases/version dir> <releases/latest dir>
"""

import os
import re
import shutil
import sys

NOINDEX = re.compile(
    rb'\r?\n?[ \t]*<meta\s+name="robots"\s+content="noindex,\s*follow"\s*/?>', re.I
)
LEFTOVER = re.compile(rb"<meta[^>]*robots[^>]*noindex\s*,\s*follow", re.I)


def main(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("source and destination must differ")
    # Recreate from scratch so pages dropped from this release disappear from latest too.
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".doctrees"))

    stripped = 0
    leftover = []
    for d, _, files in os.walk(dst):
        for f in files:
            if not f.endswith(".html"):
                continue
            path = os.path.join(d, f)
            with open(path, "rb") as fh:
                data = fh.read()
            new, n = NOINDEX.subn(b"", data)
            if n:
                stripped += 1
                with open(path, "wb") as fh:
                    fh.write(new)
            if LEFTOVER.search(new):
                leftover.append(path)

    # A tag we failed to match would silently keep latest out of search engines.
    if leftover:
        raise SystemExit(
            f"noindex,follow still present in {len(leftover)} pages, e.g. {leftover[0]}"
        )
    print(f"Copied {src} -> {dst}; removed noindex from {stripped} pages")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
