#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Launches an Argo workflow pod on the MLOps cluster.
#
# Interactive (default):
#   launch_pod.sh                          # aihub-interactive, output pod name
#   launch_pod.sh -c 8 -g 1 -m 32Gi      # with resource requests
#
# CI / ephemeral runner:
#   launch_pod.sh \
#     --template github-actions-runner \
#     --name my-runner-123 \
#     --labels "k=v,k2=v2" \
#     --output workflow \
#     --wait-step ephemeral-runner \
#     -p runner-token=xxx \
#     -p runner-labels=my-label
#
set -e

NAMESPACE="${NAMESPACE:-aihub}"
# Must match the cluster identity: the server rejects workflows whose creator labels
# differ from it, and the local $USER can differ (e.g. truncated cluster usernames)
USERNAME="$(kubectl auth whoami -n "$NAMESPACE" -o jsonpath='{.status.userInfo.username}' 2>/dev/null | cut -d@ -f1 || true)"
USERNAME="${USERNAME:-${USER:-$(whoami)}}"

# Defaults
TEMPLATE="aihub-interactive"
WF_NAME=""
WF_LABELS=""
WAIT_STEP=""
OUTPUT_MODE="pod"
CPU_REQUEST=""
GPU_REQUEST=""
MEMORY_REQUEST=""
DOCKER_IMAGE=""
EXTRA_PARAMS=()

usage() {
  echo "Usage: $0 [options]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --template <name>       Argo WorkflowTemplate (default: aihub-interactive)" >&2
  echo "  --name <wf-name>        Explicit Argo workflow name" >&2
  echo "  --labels <string>       Argo workflow labels (key=value,...)" >&2
  echo "  --wait-step <name>      Poll for this specific step to be Running" >&2
  echo "  --output <mode>         What to print: 'pod' (default) or 'workflow'" >&2
  echo "  --docker-image <image>  Docker image to use for the pod" >&2
  echo "  -p <key=value>          Extra Argo parameter (can be repeated)" >&2
  echo "  -c <cpu>                CPU request (e.g. '4', '500m')" >&2
  echo "  -g <gpu>                GPU request (e.g. '1')" >&2
  echo "  -m <memory>             Memory request (e.g. '16Gi')" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --template)   TEMPLATE="$2"; shift 2 ;;
    --name)       WF_NAME="$2"; shift 2 ;;
    --labels)     WF_LABELS="$2"; shift 2 ;;
    --wait-step)  WAIT_STEP="$2"; shift 2 ;;
    --output)     OUTPUT_MODE="$2"; shift 2 ;;
    -p)           EXTRA_PARAMS+=("$2"); shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    -c)           CPU_REQUEST="$2"; shift 2 ;;
    -g)           GPU_REQUEST="$2"; shift 2 ;;
    -m)           MEMORY_REQUEST="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Build argo submit arguments
ARGO_ARGS=()
ARGO_ARGS+=(--from "workflowtemplate/$TEMPLATE")
ARGO_ARGS+=(-n "$NAMESPACE")

if [ -n "$WF_NAME" ]; then
  ARGO_ARGS+=(--name "$WF_NAME")
fi

# Labels: use provided labels or default to current user
if [ -n "$WF_LABELS" ]; then
  ARGO_ARGS+=(--labels "$WF_LABELS")
else
  ARGO_ARGS+=(--labels "workflows.argoproj.io/creator-email=${USERNAME}.at.qti.qualcomm.com,workflows.argoproj.io/creator-preferred-username=${USERNAME}")
fi

# Resource requests
[ -n "$CPU_REQUEST" ]    && ARGO_ARGS+=(-p "cpu-request=$CPU_REQUEST")
[ -n "$GPU_REQUEST" ]    && ARGO_ARGS+=(-p "gpu-request=$GPU_REQUEST")
[ -n "$MEMORY_REQUEST" ] && ARGO_ARGS+=(-p "memory-request=$MEMORY_REQUEST")
[ -n "$DOCKER_IMAGE" ]   && ARGO_ARGS+=(-p "docker-image=$DOCKER_IMAGE")

# Extra pass-through parameters
for param in "${EXTRA_PARAMS[@]}"; do
  ARGO_ARGS+=(-p "$param")
done

ARGO_ARGS+=(-o name)

echo "Submitting workflow (template=$TEMPLATE)..." >&2
SUBMITTED_WF=$(argo submit "${ARGO_ARGS[@]}")

echo "Workflow: $SUBMITTED_WF" >&2
echo "Waiting for pod to start..." >&2

# Poll for the running pod
while true; do
  # Check if workflow has failed
  WF_STATUS=$(argo get "$SUBMITTED_WF" -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.status.phase // empty')
  if [ "$WF_STATUS" = "Failed" ] || [ "$WF_STATUS" = "Error" ]; then
    echo "ERROR: Workflow $SUBMITTED_WF $WF_STATUS" >&2
    exit 1
  fi

  if [ -n "$WAIT_STEP" ]; then
    # Poll for a specific step by displayName (used by CI)
    STEP_PHASE=$(argo get "$SUBMITTED_WF" -n "$NAMESPACE" -o json 2>/dev/null | jq -r "
      [.status.nodes // {} | to_entries[] | select(.value.displayName == \"$WAIT_STEP\")]
      | first | .value.phase // \"NotStarted\"")

    case "$STEP_PHASE" in
      Running)
        echo "Step '$WAIT_STEP' is running!" >&2
        break
        ;;
      Failed|Error)
        echo "Step '$WAIT_STEP' failed with phase: $STEP_PHASE" >&2
        argo get "$SUBMITTED_WF" -n "$NAMESPACE" >&2
        exit 1
        ;;
      *)
        echo "Step '$WAIT_STEP' phase: ${STEP_PHASE} (workflow: ${WF_STATUS}) -- waiting..." >&2
        sleep 30
        ;;
    esac
  else
    # Poll for any running pod (excluding resolve-user-identity)
    POD=$(kubectl get pods -n "$NAMESPACE" \
      -l "workflows.argoproj.io/workflow=$SUBMITTED_WF" \
      --field-selector=status.phase=Running \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
      | grep -v "resolve-user-identity" | head -1)

    if [ -n "$POD" ]; then
      echo "Pod ready: $POD" >&2
      break
    fi

    sleep 5
  fi
done

# Output the requested identifier
case "$OUTPUT_MODE" in
  workflow)
    echo "$SUBMITTED_WF"
    ;;
  pod)
    if [ -n "${POD:-}" ]; then
      echo "$POD"
    else
      # If we waited by step name, resolve the pod now
      POD=$(kubectl get pods -n "$NAMESPACE" \
        -l "workflows.argoproj.io/workflow=$SUBMITTED_WF" \
        --field-selector=status.phase=Running \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
        | grep -v "resolve-user-identity" | head -1)
      echo "$POD"
    fi
    ;;
  *)
    echo "Unknown output mode: $OUTPUT_MODE" >&2
    exit 1
    ;;
esac
