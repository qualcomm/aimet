#!/usr/bin/env bash

##
## Re-vendors the onnxruntime C/C++ headers under third_party/onnxruntime/include.
##
## AIMET's ONNX custom ops are compiled against the onnxruntime C/C++ API headers,
## but are loaded by the onnxruntime runtime as a plugin rather than linked against
## libonnxruntime. The pip onnxruntime package does not ship these headers, so they
## are vendored into this repo instead of being fetched at build time (see
## cmake/ThirdPartyDependencies.cmake, cmake/PrepareONNX.cmake).
##
## Usage:
##   third_party/onnxruntime/sync_headers.sh [version]
##
## [version] defaults to ONNXRUNTIME_VERSION below. After running, review the diff
## under third_party/onnxruntime/include, bump ONNXRUNTIME_VERSION here, and update
## the onnxruntime>= pins in packaging/dependencies/fast-release/onnx-*/reqs_pip_*.txt
## and the default in scripts/all/build_and_test.py to match.
##

set -euo pipefail

# Keep this in sync with the onnxruntime>= pins in
# packaging/dependencies/fast-release/onnx-*/reqs_pip_*.txt
ONNXRUNTIME_VERSION_DEFAULT="1.19.2"
ONNXRUNTIME_VERSION="${1:-$ONNXRUNTIME_VERSION_DEFAULT}"

REPO_ROOT=$(git rev-parse --show-toplevel)
THIRDPARTY_ONNXRUNTIME="${REPO_ROOT}/third_party/onnxruntime"
INCLUDE_DIR="${THIRDPARTY_ONNXRUNTIME}/include"

# Headers actually #include'd (directly or transitively) by AIMET's ONNX custom
# ops under TrainingExtensions/onnx/src. Re-derive this list with:
#   grep -rn '#include "' TrainingExtensions/onnx/src
# and trace onnxruntime_c_api.h / onnxruntime_cxx_api.h / cuda_context.h includes.
HEADERS=(
    "include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    "include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
    "include/onnxruntime/core/session/onnxruntime_c_api.h"
    "include/onnxruntime/core/session/onnxruntime_float16.h"
    "include/onnxruntime/core/session/onnxruntime_lite_custom_op.h"
    # Required only for CUDA builds (ONNX_CUDA), see QcQuantizeOp.h.
    "include/onnxruntime/core/providers/cuda/cuda_context.h"
    "include/onnxruntime/core/providers/cuda/cuda_resource.h"
    "include/onnxruntime/core/providers/custom_op_context.h"
    "include/onnxruntime/core/providers/resource.h"
)

case "$(uname -s)" in
    Darwin) PLATFORM_TAG="osx-arm64"; EXTENSION="tgz" ;;
    Linux)  PLATFORM_TAG="linux-x64"; EXTENSION="tgz" ;;
    *)      echo "Unsupported platform for fetching onnxruntime release archive: $(uname -s)" >&2; exit 1 ;;
esac

ARCHIVE_NAME="onnxruntime-${PLATFORM_TAG}-${ONNXRUNTIME_VERSION}.${EXTENSION}"
ARCHIVE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ARCHIVE_NAME}"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Downloading ${ARCHIVE_URL}"
curl -fsSL -o "${WORK_DIR}/${ARCHIVE_NAME}" "${ARCHIVE_URL}"

echo "Extracting ${ARCHIVE_NAME}"
tar -xzf "${WORK_DIR}/${ARCHIVE_NAME}" -C "${WORK_DIR}"

EXTRACTED_DIR="${WORK_DIR}/onnxruntime-${PLATFORM_TAG}-${ONNXRUNTIME_VERSION}"

rm -rf "${INCLUDE_DIR}"
mkdir -p "${INCLUDE_DIR}"

for relpath in "${HEADERS[@]}"; do
    src="${EXTRACTED_DIR}/${relpath}"
    # Flatten core/session/*.h to include/; preserve core/providers/* layout,
    # since QcQuantizeOp.h includes "core/providers/cuda/cuda_context.h" by that path.
    if [[ "$relpath" == */core/session/* ]]; then
        dest="${INCLUDE_DIR}/$(basename "$relpath")"
    else
        dest="${INCLUDE_DIR}/${relpath#include/onnxruntime/}"
    fi

    if [[ ! -f "$src" ]]; then
        echo "Expected header not found in release archive: ${relpath}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    echo "Vendored $(basename "$dest")"
done

echo
echo "Done. Review 'git diff ${INCLUDE_DIR}', then update:"
echo "  - ONNXRUNTIME_VERSION_DEFAULT in this script"
echo "  - onnxruntime>= pins in packaging/dependencies/fast-release/onnx-*/reqs_pip_*.txt"
echo "  - the '1.19.*' default in scripts/all/build_and_test.py"
