#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Runs long jobs on the pod detached from the terminal, so a dropped
# `kubectl exec` session doesn't take them down, and keeps each job's output
# in its own directory under KJOB_ROOT (persistent storage, outlives the pod).
#
# Each job directory (<YYYYmmdd-HHMMSS>-<name>) holds:
#   cmd        the command, shell-quoted so it can be pasted to re-run
#   meta       pod, cwd, venv, PYTHONPATH, GPU and start time
#   pid        session id of the job; only meaningful on the pod named in meta
#   run.log    stdout + stderr
#   exit_code  written when the job exits (128+N means killed by signal N)

set -e

KJOB_ROOT="${KJOB_ROOT:-/mnt/vast-ml-infra-inputs/Expire7Days/aimet/${USER:-$(id -un)}/logs}"
STOP_GRACE_SECONDS="${KJOB_STOP_GRACE:-30}"

usage() {
  echo "Usage: $0 <command> [args]" >&2
  echo "" >&2
  echo "Commands:" >&2
  echo "  run <name> [--] <cmd> [args...]  Start <cmd> detached and return immediately" >&2
  echo "  ls [-a]                          List the latest 20 jobs (-a: all)" >&2
  echo "  tail [job]                       Follow a job's log until it exits" >&2
  echo "  wait [job]                       Block until a job exits; exit with its exit code" >&2
  echo "  stop [job]                       SIGTERM the job's process group, SIGKILL after ${STOP_GRACE_SECONDS}s" >&2
  echo "" >&2
  echo "[job] is a job name (its latest run), a job directory name, or omitted for the latest job." >&2
  echo "" >&2
  echo "Environment:" >&2
  echo "  KJOB_ROOT         Where job directories go (default: $KJOB_ROOT)" >&2
  echo "  KJOB_STOP_GRACE   Seconds between SIGTERM and SIGKILL in stop (default: 30)" >&2
  echo "  TQDM_MININTERVAL  Seconds between progress bar redraws in the log (default: 30)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 run qwen27b-adascale python -m GenAILab.bench.runner --config my.yaml" >&2
  echo "  $0 run sweep -- bash -c 'for c in a b; do python run.py \$c; done'" >&2
  echo "  $0 tail qwen27b-adascale" >&2
  exit 1
}

die() {
  echo "kjob: $*" >&2
  exit 1
}

# Runs inside the job's session. It must outlive the job to record the exit
# code, so it traps the signals that stop sends to the whole process group;
# bash defers the trap until the foreground job has exited.
# shellcheck disable=SC2016
WRAPPER='
dir=$1; shift
echo $$ > "$dir/pid"
trap "echo \"[kjob] wrapper received SIGTERM\"" TERM
trap "echo \"[kjob] wrapper received SIGINT\"" INT
echo "[kjob] $(date -Is) started on $HOSTNAME: $(cat "$dir/cmd")"
"$@"
rc=$?
echo "[kjob] $(date -Is) exited with $rc"
echo "$rc" > "$dir/exit_code.tmp" && mv "$dir/exit_code.tmp" "$dir/exit_code"
'

meta_get() {
  sed -n "s/^$2: //p" "$1/meta" 2>/dev/null | head -1
}

job_name() {
  local base
  base=$(basename "$1")
  echo "${base:16}"
}

# Pod names look like aihub-interactive-<id>-aihub-interactive-<n>; keep <id>
short_host() {
  local h="${1#aihub-interactive-}"
  echo "${h%%-*}"
}

describe_rc() {
  if [ "$1" -gt 128 ] 2>/dev/null; then
    echo "$1 SIG$(kill -l $(($1 - 128)))"
  else
    echo "$1"
  fi
}

resolve_job() {
  local arg="$1" dir match=""
  if [ -n "$arg" ] && [ -d "$KJOB_ROOT/$arg" ]; then
    echo "$KJOB_ROOT/$arg"
    return
  fi
  # Directory names start with a timestamp, so the last match is the latest run
  for dir in "$KJOB_ROOT"/*/; do
    [ -d "$dir" ] || continue
    dir="${dir%/}"
    if [ -z "$arg" ] || [ "$(job_name "$dir")" = "$arg" ]; then
      match="$dir"
    fi
  done
  [ -n "$match" ] || die "no job matching '${arg:-<latest>}' in $KJOB_ROOT"
  echo "$match"
}

# Prints running, done, failed(<rc>), dead (gone without an exit code, e.g.
# SIGKILLed with its wrapper), or unknown@<pod> for jobs started on another pod.
job_state() {
  local dir="$1" rc host pid
  if [ -f "$dir/exit_code" ]; then
    rc=$(cat "$dir/exit_code")
    if [ "$rc" = 0 ]; then echo done; else echo "failed($(describe_rc "$rc"))"; fi
    return
  fi
  host=$(meta_get "$dir" host)
  if [ "$host" != "$HOSTNAME" ]; then
    echo "unknown@$(short_host "$host")"
    return
  fi
  pid=$(cat "$dir/pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null; then echo running; else echo dead; fi
}

cmd_run() {
  [ $# -ge 2 ] || usage
  local name dir quoted i
  name=$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-')
  shift
  if [ "$1" = "--" ]; then shift; fi
  [ $# -ge 1 ] || usage

  mkdir -p "$KJOB_ROOT" || die "cannot create $KJOB_ROOT"
  dir="$KJOB_ROOT/$(date +%Y%m%d-%H%M%S)-$name"
  mkdir "$dir" || die "$dir already exists, retry in a second"

  quoted=$(printf '%q ' "$@")
  echo "${quoted% }" > "$dir/cmd"
  {
    echo "name: $name"
    echo "started: $(date -Is)"
    echo "host: $HOSTNAME"
    echo "user: ${USER:-$(id -un)}"
    echo "cwd: $PWD"
    echo "python: $(command -v python || true)"
    echo "venv: ${VIRTUAL_ENV:-}"
    echo "pythonpath: ${PYTHONPATH:-}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, || true)"
  } > "$dir/meta"

  # setsid puts the job in its own session: no controlling terminal, so a
  # dropped exec session can't SIGHUP it, and one process group covers all of
  # its children. -f forks in the foreground; backgrounding with & instead
  # would make the job ignore SIGINT (no KeyboardInterrupt).
  PYTHONUNBUFFERED=1 TQDM_MININTERVAL="${TQDM_MININTERVAL:-30}" \
    setsid -f bash -c "$WRAPPER" kjob "$dir" "$@" </dev/null >>"$dir/run.log" 2>&1

  for i in $(seq 50); do
    if [ -s "$dir/pid" ]; then break; fi
    sleep 0.1
  done
  echo "Started $name (pid $(cat "$dir/pid" 2>/dev/null || echo '?')) on $HOSTNAME"
  echo "  log:    $dir/run.log"
  echo "  follow: $0 tail $name"
}

cmd_ls() {
  local dirs dir state start end
  dirs=()
  for dir in "$KJOB_ROOT"/*/; do
    [ -d "$dir" ] && dirs+=("${dir%/}")
  done
  if [ ${#dirs[@]} -eq 0 ]; then
    echo "No jobs in $KJOB_ROOT"
    return
  fi
  if [ "${1:-}" != "-a" ] && [ ${#dirs[@]} -gt 20 ]; then
    dirs=("${dirs[@]: -20}")
  fi

  printf '%-22s %-44s %8s  %s\n' STATE JOB RUNTIME LAST-LINE
  for dir in "${dirs[@]}"; do
    state=$(job_state "$dir")
    start=$(stat -c %Y "$dir/cmd" 2>/dev/null || echo 0)
    if [ -f "$dir/exit_code" ]; then
      end=$(stat -c %Y "$dir/exit_code")
    elif [ "$state" = running ]; then
      end=$(date +%s)
    else
      end=$(stat -c %Y "$dir/run.log" 2>/dev/null || echo "$start")
    fi
    printf '%-22s %-44s %8s  %s\n' "$state" "$(basename "$dir")" \
      "$(printf '%dh%02dm' $(((end - start) / 3600)) $(((end - start) % 3600 / 60)))" \
      "$(tail -c 4096 "$dir/run.log" 2>/dev/null | tr '\r' '\n' \
        | grep -v -e '^[[:space:]]*$' -e '^\[kjob\]' | tail -1 | cut -c1-100)"
  done
}

cmd_tail() {
  local dir state
  dir=$(resolve_job "${1:-}")
  state=$(job_state "$dir")
  echo "==> $(basename "$dir") ($state)" >&2
  case "$state" in
    # --pid stops following once the job's wrapper exits
    running) tail -n 50 -F --pid="$(cat "$dir/pid")" "$dir/run.log" ;;
    unknown@*) tail -n 50 -F "$dir/run.log" ;;
    *) tail -n 50 "$dir/run.log" ;;
  esac
  if [ "$state" = running ]; then
    echo "==> $(basename "$dir") ($(job_state "$dir"))" >&2
  fi
}

cmd_wait() {
  local dir state
  dir=$(resolve_job "${1:-}")
  while state=$(job_state "$dir"); [ "$state" = running ] || [ "${state%%@*}" = unknown ]; do
    sleep 10
  done
  echo "$(basename "$dir"): $state"
  [ -f "$dir/exit_code" ] || return 255
  return "$(cat "$dir/exit_code")"
}

cmd_stop() {
  local dir state pid i
  dir=$(resolve_job "${1:-}")
  state=$(job_state "$dir")
  [ "$state" = running ] || die "$(basename "$dir") is $state, not running on this pod"
  pid=$(cat "$dir/pid")

  echo "Sending SIGTERM to process group $pid..."
  kill -TERM -- "-$pid"
  for ((i = 0; i < STOP_GRACE_SECONDS; i++)); do
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      echo "$(basename "$dir"): $(job_state "$dir")"
      return
    fi
    sleep 1
  done

  echo "Still running after ${STOP_GRACE_SECONDS}s, sending SIGKILL"
  kill -KILL -- "-$pid" 2>/dev/null || true
  # The wrapper died with the job, so record the exit code on its behalf
  if [ ! -f "$dir/exit_code" ]; then
    echo "[kjob] $(date -Is) killed by kjob stop" >> "$dir/run.log"
    echo 137 > "$dir/exit_code"
  fi
  echo "$(basename "$dir"): $(job_state "$dir")"
}

[ $# -ge 1 ] || usage
COMMAND="$1"
shift
case "$COMMAND" in
  run) cmd_run "$@" ;;
  ls) cmd_ls "$@" ;;
  tail) cmd_tail "$@" ;;
  wait) cmd_wait "$@" ;;
  stop) cmd_stop "$@" ;;
  -h|--help|help) usage ;;
  *) echo "Unknown command: $COMMAND" >&2; usage ;;
esac
