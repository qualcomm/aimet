# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Artifact management for baseline tracking with configurable source strategy.

Handles downloading baselines via GitHub Actions artifacts. Supports comparing
against different branches via the --baseline-source flag.

Baseline Source Resolution:
    1. Explicit branch name (e.g., "develop", "release-aimet-2.27"):
       Compare against that branch's baseline directly.
    2. Not provided on a release branch (release-aimet-X.Y):
       Auto-detect previous release, fall back to develop.
    3. Not provided on any other branch:
       Compare against own branch's previous run.

Key Features:
    - Configurable baseline source via --baseline-source / BASELINE_SOURCE
    - Develop tracks develop quality over time
    - Feature branches default to self-comparison, can compare against
      any branch (e.g., develop) via --baseline-source
    - Release branches automatically compare against the previous release,
      falling back to develop when no prior release baseline exists
"""

import os
import re
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# Matches branch names like release-aimet-2.27 or release-aimet-2.27.1
# Matches release-aimet-2.x through 9.x only.
# Excludes 1.x (legacy) and bogus branches like release-aimet-132.2.
# TODO: Update regex to include 10+ when AIMET major version exceeds 9.
RELEASE_BRANCH_RE = re.compile(r"^release-aimet-([2-9])\.(\d+)(?:\.(\d+))?$")


@dataclass
class WorkflowConfig:
    """Configuration from GitHub Actions environment."""

    token: str
    repo: str
    workflow_file: str
    current_branch: str
    current_run_id: str
    suite: str
    baseline_source: str

    @classmethod
    def from_env(cls):
        """Create config from environment variables.

        BASELINE_SOURCE defaults to "own" when not set, meaning
        compare against the current branch's own previous run.
        For release branches, "own" triggers auto-detection of
        the previous release branch.
        """
        return cls(
            token=os.environ["GITHUB_TOKEN"],
            repo=os.environ["GITHUB_REPOSITORY"],
            workflow_file=os.environ.get("WORKFLOW_FILE", "onnx-nightly.yaml"),
            current_branch=os.environ["GITHUB_REF_NAME"],
            current_run_id=os.environ["GITHUB_RUN_ID"],
            suite=os.environ.get("INPUT_SUITE", "nightly"),
            baseline_source=os.environ.get("BASELINE_SOURCE", "own"),
        )


def parse_release_version(
    branch: str,
) -> Optional[Tuple[int, int, int]]:
    """Extract the semantic version from a release branch name.

    Examples:
        "release-aimet-2.27"   -> (2, 27, 0)
        "release-aimet-2.27.1" -> (2, 27, 1)
        "develop"              -> None
        "dev/shobitha/fix"     -> None
    """
    match = RELEASE_BRANCH_RE.match(branch)
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def find_previous_release(
    current_branch: str,
    release_branches: List[str],
) -> Optional[str]:
    """Find the release branch immediately before the current one.

    Sorts all release branches by version and returns the highest
    one that is strictly below the current branch's version.

    Examples:
        current="release-aimet-2.28",
        branches=["release-aimet-2.26", "release-aimet-2.27"]
        -> "release-aimet-2.27"

        current="release-aimet-2.1", branches=[]
        -> None
    """
    current_ver = parse_release_version(current_branch)
    if not current_ver:
        return None

    candidates = []
    for branch in release_branches:
        ver = parse_release_version(branch)
        if ver and ver < current_ver:
            candidates.append((branch, ver))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def find_latest_release(release_branches: List[str]) -> Optional[str]:
    """Find the release branch with the highest version number.

    Examples:
        ["release-aimet-2.26", "release-aimet-2.28", "release-aimet-2.27"]
        -> "release-aimet-2.28"

        [] -> None
    """
    candidates = []
    for branch in release_branches:
        ver = parse_release_version(branch)
        if ver:
            candidates.append((branch, ver))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


class BaselineStrategy:
    """Determine which branch's baseline to compare against.

    Supports four modes:
    - Explicit: user passes a branch name via --baseline-source
      (e.g., "develop", "release-aimet-2.27"). Uses exactly that branch.
    - Previous release: when baseline_source is "previous-release",
      finds the latest release branch and falls back to develop.
      Useful for comparing any branch against the most recent release.
    - Release auto-detect: when baseline_source is "own" and the
      current branch is a release branch (release-aimet-X.Y), finds the
      previous release and falls back to develop if none exists.
    - Own branch (default): compare against the current branch's own
      previous run. This is the behavior for develop and all feature
      branches when no explicit source is provided.

    This class contains no I/O. Release branch listing is provided by
    the caller (ArtifactManager fetches it via GitHub API).
    """

    def __init__(
        self,
        current_branch: str,
        baseline_source: str = "own",
    ):
        self.current_branch = current_branch
        self.baseline_source = baseline_source

    def is_main_branch(self) -> bool:
        """Check if current branch is develop."""
        return self.current_branch == "develop"

    def get_baseline_sources(
        self,
        release_branches: Optional[List[str]] = None,
    ) -> List[str]:
        """Return ordered list of branches to try for baseline.

        The downloader tries each branch in order and uses the first
        one that has a baseline artifact available.

        Args:
            release_branches: All known release branch names from
                the remote. Only required when current branch is a
                release branch and baseline_source is "own".

        Returns:
            Branch names in priority order.
        """
        # Latest release branch, with develop as fallback
        if self.baseline_source == "previous-release":
            latest = find_latest_release(release_branches or [])
            if latest:
                return [latest, "develop"]
            return ["develop"]

        # Explicit override — use exactly that branch
        if self.baseline_source != "own":
            return [self.baseline_source]

        # Release branches: previous release -> develop
        if parse_release_version(self.current_branch):
            prev = find_previous_release(
                self.current_branch,
                release_branches or [],
            )
            if prev:
                return [prev, "develop"]
            return ["develop"]

        # Default: compare against own previous run
        return [self.current_branch]

    def get_description(
        self,
        release_branches: Optional[List[str]] = None,
    ) -> str:
        """Human-readable summary of the baseline strategy.

        Used in workflow logs to explain where the baseline is
        coming from and why.
        """
        sources = self.get_baseline_sources(release_branches)

        if self.baseline_source == "previous-release":
            chain = " -> ".join(sources)
            return f"Previous release: {chain}"

        if self.baseline_source != "own":
            return f"Explicit: comparing against {self.baseline_source}"

        if parse_release_version(self.current_branch):
            chain = " -> ".join(sources)
            return f"Release branch: {chain}"

        if self.is_main_branch():
            return "Develop: tracking quality over time"

        return "Feature branch: tracking own progress"


class ArtifactManager:
    """Download and manage GitHub Actions artifacts for baselines.

    Uses the GitHub REST API to find and download baseline artifacts
    from previous workflow runs. The baseline source is determined by
    BaselineStrategy, which supports explicit branch overrides, release
    branch auto-detection, and own-branch comparison.
    """

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.headers = {
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.strategy = BaselineStrategy(
            config.current_branch,
            config.baseline_source,
        )

    def get_artifact_name(self, branch: str, suite: str) -> str:
        """
        Generate artifact name for a given branch and suite.

        Develop uses a short name without branch suffix since it is
        the primary tracking branch. All other branches include a
        normalized branch name to avoid artifact collisions.

        Examples:
            ("develop", "nightly-onnx")
                -> "baseline-nightly-onnx"
            ("release-aimet-2.27", "nightly-onnx")
                -> "baseline-nightly-onnx-release-aimet-2-27"
            ("dev/shobitha/fix", "smoke-onnx")
                -> "baseline-smoke-onnx-dev-shobitha-fix"
        """
        if branch == "develop":
            return f"baseline-{suite}"

        safe_branch = self._normalize_branch_name(branch)
        return f"baseline-{suite}-{safe_branch}"

    def download_baseline(self, output_dir: Path) -> Optional[Path]:
        """
        Download the baseline to compare current results against.

        Iterates through the branches returned by BaselineStrategy
        and downloads the first available baseline artifact. For
        release branches this means trying the previous release first,
        then falling back to develop.

        Args:
            output_dir: Directory to save the downloaded baseline JSON.

        Returns:
            Path to the downloaded baseline file, or None if no
            baseline was found (normal for first runs).
        """
        # For release branches, fetch the remote release branch list
        # so the strategy can find the previous release.
        release_branches = self._list_release_branches_if_needed()
        sources = self.strategy.get_baseline_sources(release_branches)

        print("=" * 60)
        print("BASELINE DOWNLOAD")
        print("=" * 60)
        print(f"Current branch: {self.config.current_branch}")
        print(f"Suite: {self.config.suite}")
        print(f"Strategy: {self.strategy.get_description(release_branches)}")
        print(f"Source chain: {' -> '.join(sources)}")
        print()

        for source_branch in sources:
            result = self._try_download_from_branch(source_branch, output_dir)
            if result:
                return result

        print("No baseline found from any source")
        print()
        print("   This is normal for:")
        print("   - First run on this branch/suite")
        print("   - Previous runs failed or artifacts expired")
        print()
        return None

    def _try_download_from_branch(
        self,
        branch: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """
        Attempt to download a baseline artifact from a single branch.

        Searches recent completed workflow runs on the given branch
        (newest first) for a matching baseline artifact.

        Args:
            branch: Remote branch to search for baseline artifacts.
            output_dir: Directory to save the downloaded baseline.

        Returns:
            Path to the downloaded baseline, or None if not found
            on this branch.
        """
        artifact_name = self.get_artifact_name(branch, self.config.suite)

        print(f"Searching branch: {branch}")
        print(f"  Artifact: {artifact_name}")

        # Only exclude the current run when searching own branch
        exclude_current = branch == self.config.current_branch
        runs = self._get_recent_runs(
            branch=branch,
            exclude_current=exclude_current,
            limit=10,
        )

        if not runs:
            print(f"  No completed runs found on {branch}")
            print()
            return None

        print(f"  Found {len(runs)} completed run(s)")

        for run in runs:
            run_id = run["id"]
            run_number = run.get("run_number", "?")
            run_date = run.get("created_at", "unknown")[:10]

            artifact = self._find_artifact_in_run(run_id, artifact_name)
            if not artifact:
                continue

            print(f"  Found baseline in run #{run_number} ({run_date})")

            baseline_path = self._download_artifact(artifact, output_dir)
            if baseline_path:
                print(f"  Downloaded: {baseline_path}")
                print(f"  Source branch: {branch}")
                self._set_github_output("source_branch", branch)
                print()
                return baseline_path

        print(f"  No baseline artifact in recent {branch} runs")
        print()
        return None

    def _list_release_branches_if_needed(self) -> Optional[List[str]]:
        """
        Fetch release branch names from the remote when needed.

        Required when baseline_source is "previous-release", or when
        baseline_source is "own" and the current branch is a release branch.

        Returns:
            List of release branch names, or None if not needed.
        """
        needs_release_list = self.strategy.baseline_source == "previous-release" or (
            self.strategy.baseline_source == "own"
            and parse_release_version(self.config.current_branch)
        )
        if not needs_release_list:
            return None

        url = (
            f"{self.api_base}/repos/{self.config.repo}"
            f"/git/matching-refs/heads/release-aimet-"
        )

        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            # Refs come back as "refs/heads/release-aimet-2.28" etc.
            branches = [
                ref["ref"].removeprefix("refs/heads/")
                for ref in resp.json()
                if parse_release_version(ref["ref"].removeprefix("refs/heads/"))
            ]
            print(f"Found {len(branches)} release branch(es): {branches}")
            return branches
        except requests.exceptions.RequestException as e:
            print(f"Warning: could not list release branches: {e}")
            return []

    def _get_recent_runs(
        self, branch: str, exclude_current: bool = True, limit: int = 10
    ) -> List[Dict]:
        """Get recent completed workflow runs for a branch, newest first.

        Includes failed runs: a run that failed on a crash or quality gate
        still publishes a usable baseline, and comparing against the most
        recent run keeps the baseline from going stale during a failing streak.
        """
        url = (
            f"{self.api_base}/repos/{self.config.repo}/actions/"
            f"workflows/{self.config.workflow_file}/runs"
        )

        params = {"status": "completed", "branch": branch, "per_page": limit}

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()

            runs = resp.json().get("workflow_runs", [])

            if exclude_current:
                runs = [r for r in runs if str(r["id"]) != self.config.current_run_id]

            return runs

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error querying GitHub API: {e}")
            return []

    def _find_artifact_in_run(self, run_id: int, artifact_name: str) -> Optional[Dict]:
        """Find specific artifact in a run."""
        url = (
            f"{self.api_base}/repos/{self.config.repo}/actions/runs/{run_id}/artifacts"
        )

        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if not resp.ok:
                return None

            artifacts = resp.json().get("artifacts", [])

            for artifact in artifacts:
                if artifact["name"] == artifact_name:
                    return artifact

            return None

        except requests.exceptions.RequestException:
            return None

    def _download_artifact(self, artifact: Dict, output_dir: Path) -> Optional[Path]:
        """Download and extract artifact."""
        artifact_id = artifact["id"]
        url = f"{self.api_base}/repos/{self.config.repo}/actions/artifacts/{artifact_id}/zip"

        try:
            print(f"  Downloading artifact...")
            resp = requests.get(url, headers=self.headers, stream=True, timeout=120)
            resp.raise_for_status()

            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            print(f"  Extracting...")

            # Extract
            output_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            os.unlink(tmp_path)

            # Verify extraction
            baseline_file = output_dir / "latest.json"
            if baseline_file.exists():
                return baseline_file
            else:
                print(f"  ⚠️  Artifact downloaded but latest.json not found inside")
                return None

        except Exception as e:
            print(f"  ⚠️  Error downloading artifact: {e}")
            return None

    @staticmethod
    def _set_github_output(name: str, value: str) -> None:
        """Write a key-value pair to $GITHUB_OUTPUT for step outputs."""
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{name}={value}\n")

    @staticmethod
    def _normalize_branch_name(branch: str) -> str:
        """
        Convert branch name to artifact-safe name.

        Examples:
            develop → develop
            feature/test-adaround → feature-test-adaround
            feature/fix/bug → feature-fix-bug
        """
        return branch.replace("/", "-").replace("_", "-").lower()


def main():
    """CLI entry point for artifact management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage workflow artifacts")
    parser.add_argument("action", choices=["download-baseline"])
    parser.add_argument(
        "--output-dir",
        default="AIMETRegression/baselines/downloaded",
    )
    parser.add_argument(
        "--baseline-source",
        default=None,
        help=(
            "Branch to compare baseline against. "
            "Defaults to own branch, or previous release for "
            "release branches. Examples: develop, release-aimet-2.27"
        ),
    )

    args = parser.parse_args()

    try:
        config = WorkflowConfig.from_env()

        # CLI --baseline-source overrides the environment variable
        if args.baseline_source:
            config.baseline_source = args.baseline_source

        manager = ArtifactManager(config)

        if args.action == "download-baseline":
            output_dir = Path(args.output_dir)
            manager.download_baseline(output_dir)
            exit(0)

    except KeyError as e:
        print(f"❌ Missing required environment variable: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
