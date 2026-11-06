#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Stops running Argo workflows.
#
# Interactive (default):
#   stop_pod.sh              # List your workflows, prompt to stop
#   stop_pod.sh -a           # Stop all your running workflows
#   stop_pod.sh -l           # List only (don't stop)
#   stop_pod.sh <wf-name>   # Stop a specific workflow
#
# CI (non-interactive):
#   stop_pod.sh --delete wf-1 wf-2   # Delete (not just terminate) named workflows
#
# Run as another account (e.g. the bot) without editing ~/.kube/config:
#   stop_pod.sh --user qaihm_bot --namespace aihub-ci --specify-token -a
# --specify-token prompts for a bearer token (hidden input) that overrides the
# kubeconfig credential for this invocation only.
#
set -e

NAMESPACE="aihub"
USERNAME=""
ARGO_TOKEN=""

# Wrapper so every argo call honors an optional per-invocation bearer token.
argo() {
  if [ -n "$ARGO_TOKEN" ]; then
    command argo "$@" --token "$ARGO_TOKEN"
  else
    command argo "$@"
  fi
}

usage() {
  echo "Usage: $0 [options] [workflow-name ...]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  -a                Stop all your running workflows" >&2
  echo "  -l                List your running workflows (don't stop)" >&2
  echo "  --delete          Use 'argo delete' instead of 'argo terminate'" >&2
  echo "  --user <name>     Creator username to filter on (default: your cluster username)" >&2
  echo "  --namespace <ns>  Argo namespace (default: aihub)" >&2
  echo "  --specify-token   Prompt for a bearer token (hidden) to use for this run" >&2
  echo "" >&2
  echo "If no workflow name or -a is given, lists workflows and prompts." >&2
  exit 1
}

list_workflows() {
  argo list -n "$NAMESPACE" \
    -l "workflows.argoproj.io/creator-preferred-username=$USERNAME" \
    --status Running 2>/dev/null
}

DELETE_MODE=false

stop_workflow() {
  local wf="$1"
  if $DELETE_MODE; then
    echo "Deleting workflow: $wf"
    argo delete "$wf" -n "$NAMESPACE"
  else
    echo "Stopping workflow: $wf"
    argo terminate "$wf" -n "$NAMESPACE"
  fi
}

MODE=""
SPECIFY_TOKEN=false
WF_NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a)              MODE="all"; shift ;;
    -l)              MODE="list"; shift ;;
    --delete)        DELETE_MODE=true; shift ;;
    --user)          USERNAME="$2"; shift 2 ;;
    --namespace)     NAMESPACE="$2"; shift 2 ;;
    --specify-token) SPECIFY_TOKEN=true; shift ;;
    -h|--help)       usage ;;
    *)               WF_NAMES+=("$1"); shift ;;
  esac
done

# Default to the cluster identity, which is what launch_pod.sh labels workflows with
# (the local login can differ). Skipped with --specify-token: kubectl ignores the token.
if [ -z "$USERNAME" ] && ! $SPECIFY_TOKEN; then
  USERNAME="$(kubectl auth whoami -n "$NAMESPACE" -o jsonpath='{.status.userInfo.username}' 2>/dev/null | cut -d@ -f1 || true)"
fi
USERNAME="${USERNAME:-$(whoami)}"

# Prompt for a bearer token (hidden) to run as another account this invocation.
if $SPECIFY_TOKEN; then
  read -r -s -p "Paste bearer token: " ARGO_TOKEN
  echo >&2
  if [ -z "$ARGO_TOKEN" ]; then
    echo "ERROR: no token entered." >&2
    exit 1
  fi
  if ! argo list -n "$NAMESPACE" --status Running -o name >/dev/null 2>&1; then
    echo "ERROR: argo auth failed with the supplied token (namespace: $NAMESPACE)." >&2
    echo "Check the token value / that it is valid for this cluster." >&2
    exit 1
  fi
fi

# Stop specific workflow(s) by name
if [ ${#WF_NAMES[@]} -gt 0 ]; then
  for wf in "${WF_NAMES[@]}"; do
    wf=$(echo "$wf" | xargs)  # trim whitespace
    [ -n "$wf" ] && stop_workflow "$wf" || true
  done
  exit 0
fi

# List running workflows
WORKFLOWS=$(argo list -n "$NAMESPACE" \
  -l "workflows.argoproj.io/creator-preferred-username=$USERNAME" \
  --status Running \
  -o name 2>/dev/null)

if [ -z "$WORKFLOWS" ]; then
  echo "No running workflows found for $USERNAME."
  exit 0
fi

if [ "$MODE" = "list" ]; then
  list_workflows
  exit 0
fi

if [ "$MODE" = "all" ]; then
  for wf in $WORKFLOWS; do
    stop_workflow "$wf"
  done
  exit 0
fi

# Interactive: show list and prompt
echo "Your running workflows:"
echo ""
list_workflows
echo ""
read -p "Stop all? [y/N] " confirm
if [[ "$confirm" =~ ^[yY]$ ]]; then
  for wf in $WORKFLOWS; do
    stop_workflow "$wf"
  done
else
  echo "Cancelled."
fi
