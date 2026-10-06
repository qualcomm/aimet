# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GenAILab CLI launcher (__main__.py)."""

import os
from unittest.mock import MagicMock, patch, call

import pytest
import sys


class TestBuildPytestArgs:
    def test_default_args_empty(self):
        from GenAILab.__main__ import _build_pytest_args

        args = MagicMock()
        args.force_export = False
        args.export_dir = "GenAILab/artifacts/exports"
        args.results_dir = "GenAILab/artifacts/results"
        args.fp_cache_dir = "GenAILab/artifacts/cache/fp"
        args.clear_fp_cache = False
        args.model_cache_dir = "GenAILab/artifacts/cache/model"
        args.clear_model_cache = False

        result = _build_pytest_args(args)
        assert result == []

    def test_force_export_added(self):
        from GenAILab.__main__ import _build_pytest_args

        args = MagicMock()
        args.force_export = True
        args.export_dir = "GenAILab/artifacts/exports"
        args.results_dir = "GenAILab/artifacts/results"
        args.fp_cache_dir = "GenAILab/artifacts/cache/fp"
        args.clear_fp_cache = False
        args.model_cache_dir = "GenAILab/artifacts/cache/model"
        args.clear_model_cache = False

        result = _build_pytest_args(args)
        assert "--force-export" in result

    def test_custom_dirs_added(self):
        from GenAILab.__main__ import _build_pytest_args

        args = MagicMock()
        args.force_export = False
        args.export_dir = "/custom/exports"
        args.results_dir = "/custom/results"
        args.fp_cache_dir = "/custom/fp"
        args.clear_fp_cache = True
        args.model_cache_dir = "/custom/model"
        args.clear_model_cache = True

        result = _build_pytest_args(args)
        assert "--export-dir" in result
        assert "/custom/exports" in result
        assert "--results-dir" in result
        assert "--clear-fp-cache" in result
        assert "--clear-model-cache" in result


class TestRunLocal:
    def test_calls_pytest_for_torch(self):
        from GenAILab.__main__ import run_local

        args = MagicMock()
        args.framework = "torch"
        args.config = "test.yaml"
        args.force_export = False
        args.export_dir = "GenAILab/artifacts/exports"
        args.results_dir = "GenAILab/artifacts/results"
        args.fp_cache_dir = "GenAILab/artifacts/cache/fp"
        args.clear_fp_cache = False
        args.model_cache_dir = "GenAILab/artifacts/cache/model"
        args.clear_model_cache = False

        with (
            patch("GenAILab.__main__.subprocess.call", return_value=0) as mock_call,
            patch("GenAILab.__main__._print_summary"),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_local(args, [])
            mock_call.assert_called_once()
            cmd = mock_call.call_args[0][0]
            assert "GenAILab/bench/torch/test_genai.py" in cmd
        assert exc_info.value.code == 0

    def test_both_runs_torch_and_onnx(self):
        from GenAILab.__main__ import run_local

        args = MagicMock()
        args.framework = "both"
        args.config = "test.yaml"
        args.force_export = False
        args.export_dir = "GenAILab/artifacts/exports"
        args.results_dir = "GenAILab/artifacts/results"
        args.fp_cache_dir = "GenAILab/artifacts/cache/fp"
        args.clear_fp_cache = False
        args.model_cache_dir = "GenAILab/artifacts/cache/model"
        args.clear_model_cache = False

        with (
            patch("GenAILab.__main__.subprocess.call", return_value=0) as mock_call,
            patch("GenAILab.__main__._print_summary"),
            pytest.raises(SystemExit),
        ):
            run_local(args, [])
            assert mock_call.call_count == 2

    def test_nonzero_exit_propagated(self):
        from GenAILab.__main__ import run_local

        args = MagicMock()
        args.framework = "torch"
        args.config = "test.yaml"
        args.force_export = False
        args.export_dir = "GenAILab/artifacts/exports"
        args.results_dir = "GenAILab/artifacts/results"
        args.fp_cache_dir = "GenAILab/artifacts/cache/fp"
        args.clear_fp_cache = False
        args.model_cache_dir = "GenAILab/artifacts/cache/model"
        args.clear_model_cache = False

        with (
            patch("GenAILab.__main__.subprocess.call", return_value=1),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_local(args, [])
        assert exc_info.value.code == 1


class TestDispatchOnline:
    def test_requires_gh(self):
        from GenAILab.__main__ import _require_gh

        with patch("GenAILab.__main__.shutil.which", return_value=None):
            with pytest.raises(SystemExit, match="gh.*CLI not found"):
                _require_gh()

    def test_require_gh_returns_path(self):
        from GenAILab.__main__ import _require_gh

        with patch("GenAILab.__main__.shutil.which", return_value="/usr/bin/gh"):
            assert _require_gh() == "/usr/bin/gh"


class TestGitHelpers:
    def test_current_branch(self):
        from GenAILab.__main__ import _current_branch

        with patch(
            "GenAILab.__main__.subprocess.check_output",
            return_value="main\n",
        ):
            assert _current_branch() == "main"

    def test_has_uncommitted_changes_true(self):
        from GenAILab.__main__ import _has_uncommitted_changes

        with patch("GenAILab.__main__.subprocess.call", return_value=1):
            assert _has_uncommitted_changes() is True

    def test_has_uncommitted_changes_false(self):
        from GenAILab.__main__ import _has_uncommitted_changes

        with patch("GenAILab.__main__.subprocess.call", return_value=0):
            assert _has_uncommitted_changes() is False


class TestDownloadAndMerge:
    @staticmethod
    def _fake_download(files):
        """Stand-in for ``gh run download``: writes ``files`` into ``--dir``."""

        def _call(cmd):
            tmpdir = cmd[cmd.index("--dir") + 1]
            for rel, text in files.items():
                path = os.path.join(tmpdir, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
            return 0

        return _call

    def test_analysis_dirs_copied_next_to_merged_results(self, tmp_path):
        from GenAILab.__main__ import _download_and_merge

        results = tmp_path / "results"
        # A dir already present locally (an earlier download) is left as is.
        (results / "analysis" / "run_b").mkdir(parents=True)
        (results / "analysis" / "run_b" / "run_b.md").write_text("local")
        files = {
            "analysis/run_a/run_a.md": "report a",
            "analysis/run_a/trunc_bits8/grader_summary.json": "{}",
            "analysis/run_b/run_b.md": "remote",
        }

        with patch(
            "GenAILab.__main__.subprocess.call", side_effect=self._fake_download(files)
        ):
            _download_and_merge("gh", 123, str(results), ["onnx"])

        assert (results / "analysis/run_a/run_a.md").read_text() == "report a"
        assert (results / "analysis/run_a/trunc_bits8/grader_summary.json").exists()
        assert (results / "analysis/run_b/run_b.md").read_text() == "local"

    def test_no_analysis_dir_in_artifact_creates_nothing(self, tmp_path):
        from GenAILab.__main__ import _download_and_merge

        results = tmp_path / "results"
        with patch(
            "GenAILab.__main__.subprocess.call", side_effect=self._fake_download({})
        ):
            _download_and_merge("gh", 123, str(results), ["onnx"])

        assert not (results / "analysis").exists()
