#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Sets up a plain docker/pod for GenAI development or CI.
# Mirrors the setup steps from .github/workflows/genai-scorecard.yaml
#
# Usage:
#   bash setup_genai.sh                                  # Interactive dev (defaults)
#   bash setup_genai.sh --with-aws                       # Include AWS CLI
#   bash setup_genai.sh --skip-aimet                     # Skip AIMET install (use with build_aimet.sh)
#   bash setup_genai.sh --wheels-dir /path/to/wheels     # Use pre-built wheels
#   bash setup_genai.sh --repo-dir /path/to/aimet        # Override repo location
#   bash setup_genai.sh --python-version 3.12            # Override venv Python version
#
# Options:
#   --with-aws            Install AWS CLI v2 (for CI uploads to S3)
#   --skip-aimet          Skip AIMET pip/wheel install. Use this when you plan to
#                         build AIMET from source with build_aimet.sh afterwards.
#                         Typical workflow:
#                           1. bash setup_genai.sh --skip-aimet
#                           2. bash build_aimet.sh --cuda-arch 80 --clean
#   --wheels-dir <dir>    Install AIMET from pre-built wheels in <dir>
#   --repo-dir <dir>      Override repo location (default: /scratch/aimet)
#   --python-version <v>  Python version for the venv (default: 3.12). uv
#                         downloads this interpreter itself, independent of
#                         whatever python3 the base image ships.
#
# Assumes:
#   - Running inside a CUDA-enabled container
#   - AIMET repo is available at $REPO_DIR (default: /scratch/aimet)

set -euo pipefail

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
INSTALL_AWS=false
SKIP_AIMET=false
WHEELS_DIR=""
REPO_DIR="${REPO_DIR:-/scratch/aimet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-aws)        INSTALL_AWS=true; shift ;;
    --skip-aimet)       SKIP_AIMET=true; shift ;;
    --wheels-dir)       WHEELS_DIR="$2"; shift 2 ;;
    --repo-dir)         REPO_DIR="$2"; shift 2 ;;
    --python-version)   PYTHON_VERSION="$2"; shift 2 ;;
    -h|--help)
      sed -n '/^# Sets up/,/^[^#]/{ /^#/s/^# \?//p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Default wheels directory if not specified
if [ -z "$WHEELS_DIR" ]; then
  WHEELS_DIR="${REPO_DIR}/downloads"
fi

VENV_DIR="${REPO_DIR}/.venv"

echo "=== GenAI Dev Setup ==="
echo "Repo:    $REPO_DIR"
echo "Venv:    $VENV_DIR"
echo "Wheels:  $WHEELS_DIR"
echo "Python:  $PYTHON_VERSION"

# -----------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  ca-certificates \
  curl \
  git \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  lsb-release \
  unzip \
  zip \
  zstd
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# AWS CLI v2 (optional, for CI uploads to S3)
# -----------------------------------------------------------------------
if $INSTALL_AWS; then
  echo "Installing AWS CLI v2..."
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/awscliv2.zip /tmp/aws
fi

# -----------------------------------------------------------------------
# Git configuration (non-interactive mode)
# -----------------------------------------------------------------------
echo "Configuring git for non-interactive mode..."
git config --global core.askPass ""
git config --global credential.helper ""
git config --global --add safe.directory '*'

# -----------------------------------------------------------------------
# Python virtual environment
# -----------------------------------------------------------------------
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# uv downloads and manages the requested interpreter itself, so the venv's
# Python version doesn't depend on whatever python3 the base image ships.
echo "Setting up Python $PYTHON_VERSION venv at $VENV_DIR..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed
. "$VENV_DIR/bin/activate"
uv pip install --upgrade pip wheel setuptools

# Never install the CPU onnxruntime package alongside onnxruntime-gpu: both
# install into the same `onnxruntime` import package, so whichever lands last
# wins, and the CPU one has no CUDA EP. aimet-onnx on PyPI depends on
# `onnxruntime`, so exclude it from every uv resolution in this script.
UV_EXCLUDES_FILE="$VENV_DIR/uv-excludes.txt"
echo "onnxruntime" > "$UV_EXCLUDES_FILE"
export UV_EXCLUDE="$UV_EXCLUDES_FILE"

# -----------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------
echo "Installing PyTorch with CUDA support..."
uv pip install --upgrade torch torchvision torchaudio

echo "Installing Python dependencies..."
uv pip install -r "$REPO_DIR/GenAILab/requirements.txt"

# -----------------------------------------------------------------------
# AIMET
# -----------------------------------------------------------------------
if $SKIP_AIMET; then
  echo "Skipping AIMET install (--skip-aimet). Use build_aimet.sh to build from source."
else
  # If pre-built wheels exist, install them. Otherwise install from PyPI
  # for dependencies only, and prepend local source to PYTHONPATH so
  # local code always takes priority over the PyPI packages.
  if ls "$WHEELS_DIR"/*.whl 1>/dev/null 2>&1; then
    echo "Installing AIMET from pre-built wheels..."
    ORT_PIN=""
    if ls "$WHEELS_DIR"/aimet_onnx*.whl 1>/dev/null 2>&1; then
      # TODO(temporary): onnxruntime-gpu>=1.27 requires CUDA 13/cuDNN 9, but
      # these GPU pods run an older driver that only supports CUDA 12.x. If
      # left unbounded, the CUDA EP fails to load and onnxruntime silently
      # falls back to CPUExecutionProvider (a stderr warning, not an error),
      # Remove this cap once the pod driver is upgraded to r580+.
      ORT_PIN="onnxruntime-gpu<=1.26"
    fi
    uv pip install "$WHEELS_DIR"/*.whl $ORT_PIN
  else
    echo "No pre-built wheels found. Installing from PyPI (for dependencies)..."
    uv pip install aimet-torch aimet-onnx
  fi
fi

# -----------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------
echo "Configuring environment..."
EXTRA_ENV="
# GenAI environment
export GIT_CLONE_PROTECTION_ACTIVE=\"false\"
export GIT_TERMINAL_PROMPT=\"0\"
export MPLBACKEND=\"Agg\"
export QT_QPA_PLATFORM=\"offscreen\"
"

if [ -n "${SAML2AWS_APP_ID:-}" ]; then
  EXTRA_ENV+="export SAML2AWS_APP_ID=\"$SAML2AWS_APP_ID\"
"
fi

echo "$EXTRA_ENV" >> "$VENV_DIR/bin/activate"

# Re-source to pick up the new vars
. "$VENV_DIR/bin/activate"

# Also add to .bashrc so new shells get it
echo ". $VENV_DIR/bin/activate" >> ~/.bashrc

# -----------------------------------------------------------------------
# HuggingFace authentication
# -----------------------------------------------------------------------
# Source HF_TOKEN from profile.d if not already set
if [ -z "${HF_TOKEN:-}" ] && [ -f /etc/profile.d/hf-token.sh ]; then
  . /etc/profile.d/hf-token.sh
fi

if [ -n "${HF_TOKEN:-}" ]; then
  echo "Logging in to HuggingFace..."
  hf auth login --token "$HF_TOKEN"
else
  echo "WARNING: HF_TOKEN not set. Set it locally or run: hf auth login"
fi

# -----------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------
echo ""
echo "=== Verifying installation ==="
python -c "
try:
    import aimet_torch; print(f'aimet_torch: {aimet_torch.__version__}')
except ImportError:
    print('aimet_torch: not installed')
try:
    import aimet_onnx; print(f'aimet_onnx: {aimet_onnx.__version__}')
except ImportError:
    print('aimet_onnx: not installed')
"
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To use local source over PyPI packages, run:"
echo "  export PYTHONPATH=$REPO_DIR/TrainingExtensions/torch/src/python:$REPO_DIR/TrainingExtensions/onnx/src/python:$REPO_DIR/TrainingExtensions/common/src/python:$REPO_DIR:\$PYTHONPATH"
