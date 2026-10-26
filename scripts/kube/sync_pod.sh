#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set -e

NAMESPACE="aihub"
REMOTE_BASE="/scratch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <pod-name> <local-dir> [signal-file]" >&2
  exit 1
fi

POD="$1"
LOCAL_DIR="$(cd "$2" && pwd)"
SIGNAL_FILE="${3:-}"
REPO_NAME="$(basename "$LOCAL_DIR")"
REMOTE_DIR="$REMOTE_BASE/$REPO_NAME"

export KUBE_NAMESPACE="$NAMESPACE"
export KUBE_POD="$POD"

trap 'kill 0' EXIT

do_sync() {
  if rsync -az --delete \
    --exclude='.git' \
    --exclude='build' \
    --exclude='__pycache__' \
    --exclude='onnx_checkpoints' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='GenAILab/artifacts' \
    -e "$SCRIPT_DIR/kubectl_rsync.sh" \
    "$LOCAL_DIR/" "pod:$REMOTE_DIR/" >/dev/null 2>&1; then
    return 0
  else
    echo "[sync] ERROR: rsync failed (exit code $?)" >&2
    return 1
  fi
}

# Wait for pod to signal readiness (deps installed)
echo "[sync] Waiting for pod to be ready..."
# A missing marker exits non-zero silently; any output means exec itself failed
until EXEC_ERR=$(kubectl exec -n "$NAMESPACE" "$POD" -- test -f /tmp/.pod-ready 2>&1); do
  if [[ "$EXEC_ERR" == *"unable to upgrade connection"* ]]; then
    echo "[sync] ERROR: kubectl exec failed: $EXEC_ERR" >&2
    exit 1
  elif [ -n "$EXEC_ERR" ]; then
    echo "[sync] kubectl exec failed, retrying: $EXEC_ERR" >&2
  fi
  sleep 5
done

# Ensure remote directory exists
kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p "$REMOTE_DIR"

echo "[sync] Initial sync: $REPO_NAME -> $POD:$REMOTE_DIR"
do_sync
echo "[sync] Initial sync complete."

# Signal to parent that initial sync is done
if [ -n "$SIGNAL_FILE" ]; then
  touch "$SIGNAL_FILE"
fi

if ! command -v inotifywait &>/dev/null; then
  echo "inotifywait not found. Install inotify-tools for auto-sync."
  echo "Run: sudo apt install inotify-tools"
  echo "Re-run this script manually to sync again."
  exit 0
fi

echo "[sync] Watching for changes..."
while inotifywait -r -q \
  -e modify -e create -e delete -e move \
  --exclude '(\.git|build|__pycache__|\.pyc|\.venv|GenAILab/artifacts)' \
  "$LOCAL_DIR"; do
  do_sync
done
