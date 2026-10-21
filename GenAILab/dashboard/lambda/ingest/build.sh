#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

###############################################################################
# build.sh — build the ingest Lambda deployment bundle.
#
# Produces build/function.zip containing lambda_ingest.py + its pip
# dependencies, installed for the Lambda runtime's platform so binary wheels
# (pg8000 is pure-Python, but this keeps the build portable) match.
#
# Deploy the resulting zip via the Lambda web console:
#   Lambda → (the ingest function) → Code → Upload from → .zip file
# (No credentials or env config are needed to build.)
#
# Usage:
#   bash build.sh
###############################################################################
set -euo pipefail
cd "$(dirname "$0")"

PY_RUNTIME_TAG="${PY_RUNTIME_TAG:-python3.14}"   # match the deployed runtime
BUILD=build
ZIP="$BUILD/function.zip"

rm -rf "$BUILD" && mkdir -p "$BUILD/pkg"

# --- dependencies -----------------------------------------------------------
if [ -f requirements.txt ]; then
  echo "Installing dependencies for $PY_RUNTIME_TAG (manylinux2014_x86_64)..."
  python3 -m pip install \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --only-binary=:all: \
    --target "$BUILD/pkg" \
    -r requirements.txt
fi

# --- assemble zip -----------------------------------------------------------
cp lambda_ingest.py "$BUILD/pkg/"
( cd "$BUILD/pkg" && zip -qr "../function.zip" . )

echo "Built $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "Upload it via the Lambda console: (the ingest function) → Code → Upload from → .zip file"
