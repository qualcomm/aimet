#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NAMESPACE="aihub"

# Always run: besides missing tools, it upgrades a kubectl too old for the cluster
bash "$SCRIPT_DIR/install_deps.sh"

if ! kubectl auth whoami -n "$NAMESPACE" &>/dev/null; then
  echo "ERROR: Cluster credentials are missing or expired." >&2
  echo "" >&2
  echo "To fix: log in to the cluster web UI, copy the kubeconfig to your" >&2
  echo "clipboard, and paste it into ~/.kube/config" >&2
  exit 1
fi

usage() {
  echo "Usage: $0 [options] [local-dir ...]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  -p <pod-name>    Use existing pod (skip launch)" >&2
  echo "  -e <entrypoint>  Command to run on the pod (default: /bin/bash)" >&2
  echo "  -c <cpu>              CPU request (e.g. '4', '500m')" >&2
  echo "  -g <gpu>              GPU request (e.g. '1')" >&2
  echo "  -m <memory>           Memory request (e.g. '16Gi')" >&2
  echo "  --docker-image <img>  Docker image to use for the pod" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  local-dir        Directories to sync (default: current directory)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0                                        # Launch pod, sync cwd, open bash" >&2
  echo "  $0 -p my-pod                              # Reuse pod, open bash" >&2
  echo "  $0 -e /scratch/aimet/setup.sh      # Run a setup script" >&2
  echo "  $0 -p my-pod -e 'python train.py'         # Run a command on existing pod" >&2
  echo "  $0 -c 8 -g 1 -m 32Gi                     # Launch with resource requests" >&2
  echo "  $0 /path/to/aimet /path/to/other-repo" >&2
  exit 1
}

POD=""
ENTRYPOINT="/bin/bash"
DIRS=()
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) POD="$2"; shift 2 ;;
    -e) ENTRYPOINT="$2"; shift 2 ;;
    -c) LAUNCH_ARGS+=(-c "$2"); shift 2 ;;
    -g) LAUNCH_ARGS+=(-g "$2"); shift 2 ;;
    -m) LAUNCH_ARGS+=(-m "$2"); shift 2 ;;
    --docker-image) LAUNCH_ARGS+=(--docker-image "$2"); shift 2 ;;
    -h|--help) usage ;;
    *) DIRS+=("$1"); shift ;;
  esac
done

# Default to current directory if none specified
if [ ${#DIRS[@]} -eq 0 ]; then
  DIRS=("$(pwd)")
fi

resolve_pod() {
  # Given a workflow name, find the running pod for the aihub-interactive step
  local wf="$1"
  kubectl get pods -n "$NAMESPACE" \
    -l "workflows.argoproj.io/workflow=$wf" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
    | grep -v "resolve-user-identity" | head -1
}

if [ -z "$POD" ]; then
  echo "No pod specified, launching new pod..."
  POD=$("$SCRIPT_DIR/launch_pod.sh" "${LAUNCH_ARGS[@]}")
else
  # Check if it's a pod name or a workflow name
  if ! kubectl get pod "$POD" -n "$NAMESPACE" &>/dev/null; then
    echo "$POD looks like a workflow name, resolving to pod..." >&2
    RESOLVED=$(resolve_pod "$POD")
    if [ -z "$RESOLVED" ]; then
      echo "ERROR: Could not find running aihub-interactive pod for workflow $POD" >&2
      exit 1
    fi
    POD="$RESOLVED"
  fi
fi

echo "================================================"
echo "Pod: $POD"
echo "Syncing: ${DIRS[*]}"
echo "================================================"

SYNC_PIDS=()
SIGNAL_FILES=()
SYNC_LOGS=()
for DIR in "${DIRS[@]}"; do
  SIG=$(mktemp /tmp/kube-sync-done.XXXXXX)
  rm -f "$SIG"
  SIGNAL_FILES+=("$SIG")
  LOG=$(mktemp /tmp/kube-sync-log.XXXXXX)
  SYNC_LOGS+=("$LOG")
  # Own session: sync_pod.sh's `kill 0` cleanup must not take dev.sh down with it
  setsid "$SCRIPT_DIR/sync_pod.sh" "$POD" "$DIR" "$SIG" &>"$LOG" &
  SYNC_PIDS+=($!)
done

trap 'for pid in "${SYNC_PIDS[@]}"; do kill -- -"$pid" 2>/dev/null; done; rm -f "${SIGNAL_FILES[@]}" "${SYNC_LOGS[@]}"' EXIT

# cd into repo dir if single dir, otherwise base
if [ ${#DIRS[@]} -eq 1 ]; then
  REMOTE_CWD="/scratch/$(basename "${DIRS[0]}")"
else
  REMOTE_CWD="/scratch"
fi

echo "Waiting for initial sync..."
ALL_DONE=false
while ! $ALL_DONE; do
  ALL_DONE=true
  for i in "${!SIGNAL_FILES[@]}"; do
    if [ ! -f "${SIGNAL_FILES[$i]}" ]; then
      if ! kill -0 "${SYNC_PIDS[$i]}" 2>/dev/null; then
        echo "ERROR: sync of ${DIRS[$i]} exited before the initial sync completed:" >&2
        cat "${SYNC_LOGS[$i]}" >&2
        exit 1
      fi
      ALL_DONE=false
      break
    fi
  done
  $ALL_DONE || sleep 2
done
echo "Sync ready."

# Drop a profile.d script so login shells cd into the repo
kubectl exec -n "$NAMESPACE" "$POD" -- bash -c \
  "echo 'cd $REMOTE_CWD 2>/dev/null' > /etc/profile.d/kube-dev-cwd.sh"

# Forward HF_TOKEN by writing it to profile.d so it survives user switching
HF_TOKEN_VAL=""
if [ -n "$HF_TOKEN" ]; then
  HF_TOKEN_VAL="$HF_TOKEN"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN_VAL="$(cat "$HOME/.cache/huggingface/token")"
fi

if [ -n "$HF_TOKEN_VAL" ]; then
  kubectl exec -n "$NAMESPACE" "$POD" -- bash -c \
    "echo 'export HF_TOKEN=$HF_TOKEN_VAL' > /etc/profile.d/hf-token.sh"
fi

echo "Connecting to pod (entrypoint: $ENTRYPOINT)..."
if [ "$ENTRYPOINT" = "/bin/bash" ]; then
  kubectl exec -it "$POD" -n "$NAMESPACE" -- /bin/bash -l
else
  # No -t flag: prevents the wrapper from detecting an "interactive session"
  # and hijacking the entrypoint with a login shell
  kubectl exec -i "$POD" -n "$NAMESPACE" -- /bin/bash -lc "cd $REMOTE_CWD && $ENTRYPOINT"
fi

echo "Session ended. Stopping sync."
