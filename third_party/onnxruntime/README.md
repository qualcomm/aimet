# onnxruntime headers

This directory vendors a subset of the onnxruntime C/C++ API headers used to
build AIMET's ONNX custom ops (`TrainingExtensions/onnx/src/`).

## Why these headers are vendored

AIMET's ONNX custom ops (`QcQuantizeOp`, etc.) are compiled against the
onnxruntime C/C++ API, but they are loaded by the onnxruntime runtime as a
plugin at runtime rather than linked against `libonnxruntime` at build time.
Only the headers are needed at build time.

The `onnxruntime` Python package installed via pip does not ship these
headers — only the Python bindings and the shared library. Previously, the
CMake build used `FetchContent` to download the full official release
archive from `github.com/microsoft/onnxruntime/releases` on every configure.
That has been replaced with this smaller, version-pinned, checked-in copy
to avoid a network dependency and a large download at build time (see
`cmake/PrepareONNX.cmake` and `cmake/ThirdPartyDependencies.cmake`).

## What's here

```
include/
├── onnxruntime_c_api.h
├── onnxruntime_cxx_api.h
├── onnxruntime_cxx_inline.h
├── onnxruntime_float16.h
├── onnxruntime_lite_custom_op.h
└── core/providers/
    ├── custom_op_context.h
    ├── resource.h
    └── cuda/
        ├── cuda_context.h   (only needed for -DONNX_CUDA=1 builds)
        └── cuda_resource.h
```

This is **not** the full onnxruntime release header set — only the headers
actually `#include`d (directly or transitively) by AIMET's ONNX extension
sources. Re-derive the required set with:

```bash
grep -rn '#include "' TrainingExtensions/onnx/src
```

and trace each header's own `#include`s within the release archive.

## Version

These headers correspond to onnxruntime **1.19.2** (`ORT_API_VERSION 19`).
This must stay compatible with the `onnxruntime` version AIMET is built and
run against: a mismatch (e.g. headers targeting API version 19 running
against an older `onnxruntime` package built for an earlier API) will cause
a *runtime* crash when the custom ops library is loaded (`The requested API
version [N] is not available`), even though the build succeeds.

The onnxruntime version pin is kept in:
- `packaging/dependencies/fast-release/onnx-*/reqs_pip_*.txt` (`onnxruntime>=...`)
- `scripts/all/build_and_test.py` (`ver_onnxruntime` default, e.g. `'1.19.*'`)

## Updating to a new onnxruntime version

Run the sync script, pointing it at the new version:

```bash
third_party/onnxruntime/sync_headers.sh 1.31.0
```

This downloads the official `onnxruntime-<platform>-<version>.tgz` release
archive for the current OS and re-extracts just the headers listed above
into `include/`. Then:

1. Review `git diff third_party/onnxruntime/include` for any API changes
   that might affect `TrainingExtensions/onnx/src/`.
2. Bump `ONNXRUNTIME_VERSION_DEFAULT` in `sync_headers.sh`.
3. Bump the `onnxruntime>=` pins in
   `packaging/dependencies/fast-release/onnx-*/reqs_pip_*.txt`.
4. Bump the default pin in `scripts/all/build_and_test.py`.
5. Rebuild and run the ONNX test suite to confirm the custom ops still load
   and execute correctly against the new onnxruntime version.
