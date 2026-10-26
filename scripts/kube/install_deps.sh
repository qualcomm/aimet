#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set -e

CURL=(curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 30)

if ! command -v argo &>/dev/null; then
  echo "Installing Argo CLI..."
  "${CURL[@]}" "https://github.com/argoproj/argo-workflows/releases/download/v3.5.5/argo-linux-amd64.gz" -o /tmp/argo.gz
  gunzip -f /tmp/argo.gz
  sudo install /tmp/argo /usr/local/bin/argo
  rm /tmp/argo
fi

# The cluster's exec proxy rejects older clients ("kubectl X is too old for exec")
KUBECTL_MIN_VERSION="1.31"

kubectl_too_old() {
  local ver
  ver=$(kubectl version --client 2>/dev/null | sed -n 's/^Client Version: v\([0-9]*\.[0-9]*\).*/\1/p')
  [ -z "$ver" ] || [ "$(printf '%s\n' "$KUBECTL_MIN_VERSION" "$ver" | sort -V | head -1)" != "$KUBECTL_MIN_VERSION" ]
}

if ! command -v kubectl &>/dev/null || kubectl_too_old; then
  echo "Installing kubectl..."
  "${CURL[@]}" "https://dl.k8s.io/release/$("${CURL[@]}" https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /tmp/kubectl
  sudo install /tmp/kubectl /usr/local/bin/kubectl
  rm /tmp/kubectl
  hash -r
  if kubectl_too_old; then
    echo "ERROR: $(command -v kubectl) is older than v$KUBECTL_MIN_VERSION." >&2
    echo "It shadows /usr/local/bin/kubectl on PATH; remove or upgrade it." >&2
    exit 1
  fi
fi

APT_DEPS=()
command -v rsync &>/dev/null       || APT_DEPS+=(rsync)
command -v jq &>/dev/null          || APT_DEPS+=(jq)
command -v inotifywait &>/dev/null || APT_DEPS+=(inotify-tools)

if [ ${#APT_DEPS[@]} -gt 0 ]; then
  echo "Installing ${APT_DEPS[*]}..."
  sudo apt-get update -qq
  sudo apt-get install -yqq "${APT_DEPS[@]}"
fi
