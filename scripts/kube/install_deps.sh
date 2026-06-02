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

if ! command -v kubectl &>/dev/null; then
  echo "Installing kubectl..."
  "${CURL[@]}" "https://dl.k8s.io/release/$("${CURL[@]}" https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /tmp/kubectl
  sudo install /tmp/kubectl /usr/local/bin/kubectl
  rm /tmp/kubectl
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
