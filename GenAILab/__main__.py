# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI Scorecard launcher.

Usage:
    # Run locally
    python -m GenAILab --framework torch --config GenAILab/example_config.yaml

    # Run on GitHub Actions
    python -m GenAILab --framework torch --config GenAILab/example_config.yaml --online

    # Run online and wait for results to be merged locally
    python -m GenAILab --framework torch --config cfg.yaml --online --wait

    # Download results from a previous online run
    python -m GenAILab --framework torch --config cfg.yaml --download <run_id>

    # All extra arguments are forwarded to pytest (local runs only)
    python -m GenAILab --framework onnx --config cfg.yaml --force-export -v
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description="GenAI Scorecard Launcher")
    parser.add_argument("--framework", choices=["torch", "onnx", "both"], required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Dispatch as a GitHub Actions workflow instead of running locally",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch for online dispatch (default: current branch)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the online run to finish and download results locally (requires --online)",
    )
    parser.add_argument(
        "--download",
        metavar="RUN_ID",
        default=None,
        help="Download results from a previous online run and merge into local results",
    )

    pytest_group = parser.add_argument_group(
        "pytest options", "Forwarded to pytest (local runs only)"
    )
    pytest_group.add_argument(
        "--force-export",
        action="store_true",
        default=False,
        help="Force export of all artifacts (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--export-dir",
        default="GenAILab/artifacts/exports",
        help="Base directory for exported model artifacts (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--results-dir",
        default="GenAILab/artifacts/results",
        help="Directory for profiling results (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--fp-cache-dir",
        default="GenAILab/artifacts/cache/fp",
        help="Directory for FP results cache (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--clear-fp-cache",
        action="store_true",
        default=False,
        help="Clear the FP cache before running (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--model-cache-dir",
        default="GenAILab/artifacts/cache/model",
        help="Directory for ONNX model cache (default: %(default)s)",
    )
    pytest_group.add_argument(
        "--clear-model-cache",
        action="store_true",
        default=False,
        help="Clear the model cache before running (default: %(default)s)",
    )

    args, extra = parser.parse_known_args()

    if args.download:
        download_results(args)
    elif args.online:
        pytest_opts = _build_pytest_args(args)
        if pytest_opts or extra:
            discarded = pytest_opts + extra
            warnings.warn(
                f"Pytest arguments {discarded} are ignored in --online mode. "
                "Only --framework, --config, and --branch are forwarded to "
                "GitHub Actions."
            )
        dispatch_online(args)
    else:
        if args.wait:
            sys.exit("Error: --wait requires --online.")
        run_local(args, extra)


def _build_pytest_args(args):
    """Reconstruct pytest flags from parsed args."""
    pytest_args = []
    if args.force_export:
        pytest_args.append("--force-export")
    if args.export_dir != "GenAILab/artifacts/exports":
        pytest_args.extend(["--export-dir", args.export_dir])
    if args.results_dir != "GenAILab/artifacts/results":
        pytest_args.extend(["--results-dir", args.results_dir])
    if args.fp_cache_dir != "GenAILab/artifacts/cache/fp":
        pytest_args.extend(["--fp-cache-dir", args.fp_cache_dir])
    if args.clear_fp_cache:
        pytest_args.append("--clear-fp-cache")
    if args.model_cache_dir != "GenAILab/artifacts/cache/model":
        pytest_args.extend(["--model-cache-dir", args.model_cache_dir])
    if args.clear_model_cache:
        pytest_args.append("--clear-model-cache")
    return pytest_args


def run_local(args, extra_pytest_args):
    frameworks = ["torch", "onnx"] if args.framework == "both" else [args.framework]
    forwarded = _build_pytest_args(args)
    for framework in frameworks:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-s",
            f"GenAILab/bench/{framework}/test_genai.py",
            "--config",
            args.config,
            *forwarded,
            *extra_pytest_args,
        ]
        ret = subprocess.call(cmd)
        if ret != 0:
            sys.exit(ret)
    _print_summary(args.results_dir, args.config)
    sys.exit(0)


def _print_summary(results_dir, config_path=None):
    results_json = os.path.join(results_dir, "profiling_data.json")
    if os.path.exists(results_json):
        from GenAILab.bench.summary import print_summary

        print_summary(results_json, config_path)


# ---------------------------------------------------------------------------
# Online dispatch
# ---------------------------------------------------------------------------


def _require_gh():
    gh = shutil.which("gh")
    if not gh:
        sys.exit(
            "Error: `gh` CLI not found. Install it (instructions at https://cli.github.com/) to use --online."
        )
    return gh


def dispatch_online(args):
    gh = _require_gh()

    if args.branch:
        branch = args.branch
    else:
        branch = _current_branch()
        if _has_uncommitted_changes():
            warnings.warn(
                f"You have uncommitted changes on '{branch}'. "
                "The online run will use the last pushed commit, "
                "not your local working tree. Consider committing and pushing first, "
                "or use --branch to target a specific ref."
            )

    with open(args.config, "rb") as f:
        config_b64 = base64.b64encode(f.read()).decode()

    cmd = [
        gh,
        "workflow",
        "run",
        "genai-scorecard.yaml",
        "--ref",
        branch,
        "--field",
        "run_mode=ad_hoc",
        "--field",
        f"ad_hoc_variants={args.framework}",
        "--field",
        f"ad_hoc_config_b64={config_b64}",
    ]
    print(f"Dispatching to GitHub Actions on branch '{branch}'...")
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(ret)

    # Poll for the run ID so we can print it for the user
    print("Waiting for workflow run to appear...")
    run_id = _poll_for_run_id(gh, branch)
    if not run_id:
        print("Warning: Could not determine run ID. Check GitHub Actions manually.")
        sys.exit(0)
    print(f"Run ID: {run_id}")
    print(f"To download results later, run:")
    print(
        f"  python -m GenAILab --framework {args.framework} --config {args.config} --download {run_id}"
    )

    if not args.wait:
        sys.exit(0)

    print(f"Waiting for run {run_id} to complete...")

    # Wait for the run to finish
    ret = subprocess.call([gh, "run", "watch", str(run_id), "--exit-status"])
    if ret != 0:
        print(f"Warning: Run {run_id} finished with a non-zero status.")

    # Download, merge, and show summary
    results_dir = args.results_dir
    frameworks = ["torch", "onnx"] if args.framework == "both" else [args.framework]
    _download_and_merge(gh, run_id, results_dir, frameworks)
    _print_summary(results_dir, args.config)


def _poll_for_run_id(gh, branch, max_attempts=10, interval=5):
    """Poll `gh run list` to find the most recent in-progress run on the branch."""
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(interval)
        result = subprocess.run(
            [
                gh,
                "run",
                "list",
                "--workflow",
                "genai-scorecard.yaml",
                "--branch",
                branch,
                "--limit",
                "1",
                "--json",
                "databaseId,status",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        runs = json.loads(result.stdout)
        if runs and runs[0]["status"] in ("queued", "in_progress"):
            return runs[0]["databaseId"]
    return None


# ---------------------------------------------------------------------------
# Download and merge
# ---------------------------------------------------------------------------


def download_results(args):
    """Download results from a previous online run and merge into local results."""
    gh = _require_gh()
    run_id = args.download
    results_dir = args.results_dir
    frameworks = ["torch", "onnx"] if args.framework == "both" else [args.framework]

    status = _get_run_status(gh, run_id)
    if status in ("queued", "in_progress"):
        print(f"Run {run_id} is still {status}.")
        print(
            f"Use --online --wait to wait for completion, or try --download again later."
        )
        sys.exit(1)
    elif status == "completed":
        _download_and_merge(gh, run_id, results_dir, frameworks)
        _print_summary(results_dir, args.config)
    else:
        print(f"Run {run_id} has status '{status}'. Results may be incomplete.")
        _download_and_merge(gh, run_id, results_dir, frameworks)
        _print_summary(results_dir, args.config)


def _get_run_status(gh, run_id):
    """Get the status of a GitHub Actions run."""
    result = subprocess.run(
        [gh, "run", "view", str(run_id), "--json", "status", "-q", ".status"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: Could not find run {run_id}. Check the run ID and try again.")
    return result.stdout.strip()


def _stamp_entries_as_online(json_path, ci_metadata):
    """Update each entry's environment in a profiling JSON to mark it as an online run."""
    with open(json_path, "r") as f:
        data = json.load(f)

    for entries in data.values():
        for entry in entries:
            env = entry.get("environment", {})
            env["run_type"] = "online"
            env.update(ci_metadata)
            entry["environment"] = env

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def _download_and_merge(gh, run_id, results_dir, frameworks):
    """Download artifacts for the given run ID and merge into local results."""
    from GenAILab.bench.profiler import merge_csv_results, merge_json_results

    os.makedirs(results_dir, exist_ok=True)

    for variant in frameworks:
        artifact_name = f"test-data-{variant}-{run_id}"
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Downloading artifact '{artifact_name}'...")
            ret = subprocess.call(
                [
                    gh,
                    "run",
                    "download",
                    str(run_id),
                    "--name",
                    artifact_name,
                    "--dir",
                    tmpdir,
                ]
            )
            if ret != 0:
                print(
                    f"Warning: Failed to download artifact '{artifact_name}'. Skipping."
                )
                continue

            # Read CI metadata if available and stamp entries as online
            run_metadata_path = os.path.join(tmpdir, "run_metadata.json")
            ci_metadata = {}
            if os.path.exists(run_metadata_path):
                with open(run_metadata_path, "r") as f:
                    ci_metadata = json.load(f)

            remote_json = os.path.join(tmpdir, "profiling_data.json")
            if os.path.exists(remote_json):
                _stamp_entries_as_online(remote_json, ci_metadata)

            # Merge JSON
            local_json = os.path.join(results_dir, "profiling_data.json")
            if os.path.exists(remote_json):
                count = merge_json_results(remote_json, local_json)
                print(f"Merged {count} entries from {variant} JSON into {local_json}")

            # Merge CSV
            remote_csv = os.path.join(tmpdir, "profiling_data.csv")
            local_csv = os.path.join(results_dir, "profiling_data.csv")
            if os.path.exists(remote_csv):
                count = merge_csv_results(remote_csv, local_csv)
                print(f"Merged {count} rows from {variant} CSV into {local_csv}")

            # Analysis run dirs hold the reports the merged entries point at.
            # Each dir name ends in a random suffix, so one never collides
            # with a local run; an existing dir is a re-download and is kept.
            remote_analysis = os.path.join(tmpdir, "analysis")
            if os.path.isdir(remote_analysis):
                local_analysis = os.path.join(results_dir, "analysis")
                os.makedirs(local_analysis, exist_ok=True)
                copied = 0
                for name in os.listdir(remote_analysis):
                    dest = os.path.join(local_analysis, name)
                    if not os.path.exists(dest):
                        shutil.copytree(os.path.join(remote_analysis, name), dest)
                        copied += 1
                print(
                    f"Copied {copied} analysis run(s) from {variant} into {local_analysis}"
                )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _current_branch():
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def _has_uncommitted_changes():
    ret = subprocess.call(["git", "diff", "--quiet", "HEAD"])
    return ret != 0


if __name__ == "__main__":
    main()
