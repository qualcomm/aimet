# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate a compact markdown summary table for the workflow results job.

Reads per-framework comparison_summary.json artifacts and produces a
markdown table showing test counts, baseline comparison, quantization
status, and any issues.

Usage:
    python -m AIMETRegression.workflow.summary \
        --onnx-status success \
        --torch-status success \
        --onnx-summary summaries/onnx/comparison_summary.json \
        --torch-summary summaries/torch/comparison_summary.json \
        --suite nightly \
        --trigger schedule
"""

import argparse
import json
import sys


def load_summary(path):
    """Load a comparison summary JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status_emoji(status):
    """Map job status to display string."""
    return {"success": "✅ Passed", "skipped": "⏭️ Skipped"}.get(status, "❌ Failed")


def outcome_emoji(branch, failed):
    """Leading status emoji, by outcome and branch tier.

    Release branches escalate: 🚨 on failure, 🎊 on a pass. Other branches use
    ⚠️ on failure and ✅ on a pass. The emoji always matches the outcome.
    """
    is_release = branch.startswith("release-")
    if failed:
        return "🚨" if is_release else "⚠️"
    return "🎊" if is_release else "✅"


def branch_label(branch):
    """Bracketed branch label, or empty string when no branch is given."""
    return f"[{branch}] " if branch else ""


def baseline_cell(data):
    """Format baseline comparison column."""
    parts = []
    if data["stable"]:
        parts.append(f"✅ Stable: {data['stable']}")
    if data["improvements"]:
        parts.append(f"📈 Improvements: {data['improvements']}")
    if data["regressions"]:
        parts.append(f"⚠️ Regressions: {data['regressions']}")
    return "<br>".join(parts) if parts else "N/A"


def quality_cell(data):
    """Format quantization status column."""
    parts = []
    if data["passed"]:
        parts.append(f"✅ Passed: {data['passed']}")
    if data["warnings"]:
        parts.append(f"⚠️ Warnings: {data['warnings']}")
    if data["failed"]:
        parts.append(f"❌ Failed: {data['failed']}")
    if data.get("crashed"):
        parts.append(f"💥 Crashed: {data['crashed']}")
    return "<br>".join(parts) if parts else "N/A"


def generate_summary(
    onnx_status, torch_status, onnx_summary_path, torch_summary_path, suite, trigger
):
    """Generate and print the markdown summary to stdout."""
    onnx = load_summary(onnx_summary_path) if onnx_summary_path else None
    torch_data = load_summary(torch_summary_path) if torch_summary_path else None

    print(f"## {suite.title()} Regression Results\n")
    print(f"**Suite:** {suite} | **Trigger:** {trigger}\n")

    # Results table
    print("| Framework | Status | Tests | Baseline Comparison | Quantization Status |")
    print("|-----------|--------|-------|---------------------|---------------------|")

    for name, status, data in [
        ("ONNX", onnx_status, onnx),
        ("Torch", torch_status, torch_data),
    ]:
        emoji = status_emoji(status)
        if data:
            tests = data["total"]
            baseline = baseline_cell(data)
            quality = quality_cell(data)
        elif status == "skipped":
            tests = "-"
            baseline = "-"
            quality = "-"
        else:
            tests = "?"
            baseline = "-"
            quality = "-"
        print(f"| {name} | {emoji} | {tests} | {baseline} | {quality} |")

    # Failed tests detail
    failed_sections = []
    for name, data in [("ONNX", onnx), ("Torch", torch_data)]:
        if not data:
            continue
        items = []
        if data["regression_tests"]:
            items.append(("⚠️ Regressions", data["regression_tests"]))
        if data["failed_tests"]:
            items.append(("❌ Quantization failures", data["failed_tests"]))
        if items:
            failed_sections.append((name, items))

    if failed_sections:
        print("\n### Issues\n")
        for name, items in failed_sections:
            for label, tests in items:
                print(f"**{name}** — {label}:")
                for t in tests:
                    print(f"- `{t}`")
                print()


def has_test_failures(onnx, torch_data):
    """True if either framework summary reports per-test failures."""
    return any(data and data.get("has_failures") for data in (onnx, torch_data))


def slack_outcome(onnx, torch_data, failed_stage):
    """Return (failed, outcome_phrase) for the Slack title.

    A failed stage (build, runner startup, etc.) takes precedence and names
    where the pipeline broke. Otherwise per-test failures mark a regression
    failure, and a clean run is reported as passed.
    """
    if failed_stage:
        return True, f"failed at {failed_stage}"
    if has_test_failures(onnx, torch_data):
        return True, "failed"
    return False, "passed"


def generate_slack_summary(
    onnx_summary_path,
    torch_summary_path,
    suite,
    run_url=None,
    branch="",
    failed_stage="",
):
    """Generate a Slack-friendly plain text summary to stdout.

    Failure is derived from the comparison summaries and ``failed_stage``, not
    from job statuses, so this intentionally does not take ``onnx_status`` /
    ``torch_status``.
    """
    onnx = load_summary(onnx_summary_path) if onnx_summary_path else None
    torch_data = load_summary(torch_summary_path) if torch_summary_path else None

    failed, outcome = slack_outcome(onnx, torch_data, failed_stage)
    emoji = outcome_emoji(branch, failed)
    print(
        f"*{emoji} {branch_label(branch)}{suite.title()} Model Accuracy Regression {outcome}*"
    )
    if run_url:
        print(f"Workflow: {run_url}")


def main():
    parser = argparse.ArgumentParser(description="Generate workflow results summary")
    parser.add_argument("--onnx-status", default="skipped")
    parser.add_argument("--torch-status", default="skipped")
    parser.add_argument("--onnx-summary", default=None)
    parser.add_argument("--torch-summary", default=None)
    parser.add_argument("--suite", default="nightly")
    parser.add_argument("--trigger", default="unknown")
    parser.add_argument("--format", default="github", choices=["github", "slack"])
    parser.add_argument("--run-url", default=None)
    parser.add_argument(
        "--branch",
        default="",
        help="git ref name, used to distinguish develop vs release-branch runs",
    )
    parser.add_argument(
        "--failed-stage",
        default="",
        help="name of the pipeline stage that failed (e.g. 'AIMET build (ONNX)'), if any",
    )
    args = parser.parse_args()

    if args.format == "slack":
        generate_slack_summary(
            onnx_summary_path=args.onnx_summary,
            torch_summary_path=args.torch_summary,
            suite=args.suite,
            run_url=args.run_url,
            branch=args.branch,
            failed_stage=args.failed_stage,
        )
    else:
        generate_summary(
            onnx_status=args.onnx_status,
            torch_status=args.torch_status,
            onnx_summary_path=args.onnx_summary,
            torch_summary_path=args.torch_summary,
            suite=args.suite,
            trigger=args.trigger,
        )


if __name__ == "__main__":
    main()
