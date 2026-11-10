#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Builds AIMET from source inside a pod/container.
#
# Usage:
#   bash build_aimet.sh                          # Build with defaults (torch+onnx, CUDA, auto-detect arch)
#   bash build_aimet.sh --torch-only             # Build torch variant only
#   bash build_aimet.sh --onnx-only              # Build onnx variant only
#   bash build_aimet.sh --cuda-arch 80           # Specify CUDA architecture (e.g. A100=80)
#   bash build_aimet.sh --repo-dir /path/to/aimet
#   bash build_aimet.sh --clean                  # Remove build/ before building
#
# Common CUDA architectures:
#   V100  = 70
#   T4    = 75
#   A100  = 80
#   A10G  = 86
#   H100  = 90
#
# Prerequisites:
#   This script will auto-install the following if missing:
#     System:  libeigen3-dev, pkg-config
#     Python:  scikit-build-core, build, pybind11, cython, onnxruntime-gpu
#
# Notes:
#   - Use --no-build-isolation (handled internally) so the build can see
#     packages already installed in your venv (e.g. torch, onnxruntime).
#   - If the build fails with CUDA architecture errors, specify --cuda-arch
#     explicitly to avoid auto-detection picking up unsupported architectures.
#   - Use --clean when switching between build variants or after pulling
#     changes that modify CMakeLists.txt.
#
# Assumes:
#   - Running inside a CUDA-enabled container with a Python venv active
#   - AIMET repo is available at $REPO_DIR (default: /scratch/aimet)

set -euo pipefail

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/scratch/aimet}"
ENABLE_TORCH=ON
ENABLE_ONNX=ON
ENABLE_CUDA=ON
CUDA_ARCH=""
CLEAN_BUILD=false

VARIANT_SPECIFIED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)     REPO_DIR="$2"; shift 2 ;;
    --torch-only)   ENABLE_ONNX=OFF; VARIANT_SPECIFIED=true; shift ;;
    --onnx-only)    ENABLE_TORCH=OFF; VARIANT_SPECIFIED=true; shift ;;
    --no-cuda)      ENABLE_CUDA=OFF; shift ;;
    --cuda-arch)    CUDA_ARCH="$2"; shift 2 ;;
    --clean)        CLEAN_BUILD=true; shift ;;
    -h|--help)
      sed -n '/^# Builds/,/^[^#]/{ /^#/s/^# \?//p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------
# Interactive variant selection (when no --torch-only / --onnx-only given)
# -----------------------------------------------------------------------
if ! $VARIANT_SPECIFIED; then
  echo "Select build variant:"
  echo "  1) torch only"
  echo "  2) onnx only"
  echo "  3) doc only"
  read -rp "Enter choice [1/2/3]: " choice
  case "$choice" in
    1) ENABLE_TORCH=ON; ENABLE_ONNX=OFF ;;
    2) ENABLE_TORCH=OFF; ENABLE_ONNX=ON ;;
    3) ENABLE_TORCH=ON; ENABLE_ONNX=ON ;;
    *) echo "Invalid choice: $choice" >&2; exit 1 ;;
  esac
fi

cd "$REPO_DIR"

echo "=== AIMET Build ==="
echo "Repo:       $REPO_DIR"
echo "CUDA:       $ENABLE_CUDA"
echo "Torch:      $ENABLE_TORCH"
echo "ONNX:       $ENABLE_ONNX"
echo "CUDA arch:  ${CUDA_ARCH:-auto-detect}"
echo ""

# -----------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------
MISSING_APT=()
dpkg -s libeigen3-dev &>/dev/null || MISSING_APT+=(libeigen3-dev)
dpkg -s pkg-config &>/dev/null    || MISSING_APT+=(pkg-config)

if [ ${#MISSING_APT[@]} -gt 0 ]; then
  echo "Installing system dependencies: ${MISSING_APT[*]}..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq "${MISSING_APT[@]}"
fi

# -----------------------------------------------------------------------
# Python build dependencies
# -----------------------------------------------------------------------
echo "Installing Python build dependencies..."
pip install -q \
  "scikit-build-core[wheels]==0.11.1" \
  build \
  pybind11 \
  "cython>=3.0"

if [ "$ENABLE_ONNX" = "ON" ]; then
  pip install -q onnxruntime-gpu
fi

# -----------------------------------------------------------------------
# Clean build directory if requested
# -----------------------------------------------------------------------
if $CLEAN_BUILD; then
  echo "Cleaning build directory..."
  rm -rf build/
fi

# -----------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------
CMAKE_ARGS="-DENABLE_CUDA=$ENABLE_CUDA -DENABLE_TORCH=$ENABLE_TORCH -DENABLE_ONNX=$ENABLE_ONNX"

if [ -n "$CUDA_ARCH" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
fi

echo "Building AIMET..."
echo "  CMAKE_ARGS=$CMAKE_ARGS"
echo ""

CMAKE_ARGS="$CMAKE_ARGS" pip install --no-build-isolation -e .

# -----------------------------------------------------------------------
# Install runtime and test dependencies
# -----------------------------------------------------------------------
DEPS_DIR="$REPO_DIR/packaging/dependencies"

if [ "$ENABLE_TORCH" = "ON" ] && [ "$ENABLE_ONNX" = "ON" ]; then
  REQS_FILE="$DEPS_DIR/fast-release/onnx-torch-cpu/reqs_pip_onnx_torch_gpu.txt"
elif [ "$ENABLE_TORCH" = "ON" ]; then
  REQS_FILE="$DEPS_DIR/fast-release/torch-gpu/reqs_pip_torch_gpu.txt"
else
  REQS_FILE="$DEPS_DIR/fast-release/onnx-gpu/reqs_pip_onnx_gpu.txt"
fi

echo "Installing runtime dependencies from $REQS_FILE..."
pip install -r "$REQS_FILE"

echo "Installing test dependencies..."
pip install -r "$DEPS_DIR/reqs_pip_test.txt"

if [ "$ENABLE_TORCH" = "ON" ]; then
  pip install -r "$DEPS_DIR/reqs_pip_test_torch.txt"
fi

if [ "$ENABLE_ONNX" = "ON" ]; then
  # TODO(temporary): onnxruntime-gpu>=1.27 requires CUDA 13/cuDNN 9, but
  # these GPU pods run an older driver that only supports CUDA 12.x. If
  # left unbounded, the CUDA EP fails to load and onnxruntime silently
  # falls back to CPUExecutionProvider (a stderr warning, not an error),
  # Remove this cap once the pod driver is upgraded to r580+.
  # The reqs files above pull in the CPU `onnxruntime` package, which shares
  # the `onnxruntime` import dir with onnxruntime-gpu. Uninstall it *before*
  # reinstalling onnxruntime-gpu, since uninstalling after would delete files
  # onnxruntime-gpu just wrote.
  pip uninstall -y onnxruntime
  pip install --force-reinstall --no-deps onnxruntime-gpu
fi

# -----------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------
echo ""
echo "=== Verifying build ==="
set +e
python -c "
import sys
ok = True
failed = []
if '$ENABLE_TORCH' == 'ON':
    try:
        import aimet_torch
        print(f'  aimet_torch: {aimet_torch.__version__}')
    except ImportError as e:
        failed.append(f'  aimet_torch: {e}')
        ok = False
if '$ENABLE_ONNX' == 'ON':
    try:
        import aimet_onnx
        print(f'  aimet_onnx: {aimet_onnx.__version__}')
    except ImportError as e:
        failed.append(f'  aimet_onnx: {e}')
        ok = False
if not ok:
    print()
    print('=== Build verification FAILED ===')
    print('Failed to import:')
    for f in failed:
        print(f)
    sys.exit(1)
else:
    print()
    print('=== Build complete ===')
"
verify_status=$?
set -e

exit $verify_status
