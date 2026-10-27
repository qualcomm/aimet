# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GenAILab CLI launcher (__main__.py)."""

import os
from unittest.mock import patch

import pytest


def _parse(argv):
    """Parse ``argv`` with the launcher's run flags; returns ``(args, actions)``."""
    import argparse

    from GenAILab.bench.runner import add_run_arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--config", default="test.yaml")
    actions = add_run_arguments(parser)
    return parser.parse_args(argv), actions


class TestBuildRunArgs:
    def test_default_args_empty(self):
        from GenAILab.__main__ import _build_run_args

        args, actions = _parse([])
        assert _build_run_args(args, actions) == []

    def test_flags_forwarded(self):
        from GenAILab.__main__ import _build_run_args

        argv = ["--force-export", "--no-recipe-cache", "--fail-fast", "--pdb"]
        args, actions = _parse(argv)
        assert _build_run_args(args, actions) == argv

    def test_custom_dirs_added(self):
        from GenAILab.__main__ import _build_run_args

        argv = [
            "--export-dir",
            "/custom/exports",
            "--results-dir",
            "/custom/results",
            "--fp-cache-dir",
            "/custom/fp",
            "--clear-fp-cache",
            "--model-cache-dir",
            "/custom/model",
            "--clear-model-cache",
            "--recipe-cache-dir",
            "/custom/recipe",
            "--clear-recipe-cache",
        ]
        args, actions = _parse(argv)
        assert _build_run_args(args, actions) == argv


class TestRunLocal:
    def test_launches_runner_for_torch(self):
        from GenAILab.__main__ import run_local

        args, actions = _parse(["--framework", "torch"])
        with (
            patch("GenAILab.__main__.subprocess.call", return_value=0) as mock_call,
            patch("GenAILab.__main__._print_summary"),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_local(args, actions)
        assert exc_info.value.code == 0
        mock_call.assert_called_once()
        cmd = mock_call.call_args[0][0]
        assert cmd[1:] == [
            "-m",
            "GenAILab.bench.runner",
            "--framework",
            "torch",
            "--config",
            "test.yaml",
        ]

    def test_both_runs_torch_and_onnx(self):
        from GenAILab.__main__ import run_local

        args, actions = _parse(["--framework", "both"])
        with (
            patch("GenAILab.__main__.subprocess.call", return_value=0) as mock_call,
            patch("GenAILab.__main__._print_summary"),
            pytest.raises(SystemExit),
        ):
            run_local(args, actions)
        frameworks = [c[0][0][4] for c in mock_call.call_args_list]
        assert frameworks == ["torch", "onnx"]

    def test_nonzero_exit_propagated(self):
        from GenAILab.__main__ import run_local

        args, actions = _parse([])
        with (
            patch("GenAILab.__main__.subprocess.call", return_value=1),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_local(args, actions)
        assert exc_info.value.code == 1


class TestRunner:
    @staticmethod
    def _write_config(tmp_path, text):
        path = tmp_path / "cfg.yaml"
        path.write_text(text)
        return str(path)

    def _run(self, tmp_path, config_text, run_experiment, extra_argv=()):
        from GenAILab.bench import runner

        args, _ = _parse(
            [
                "--framework",
                "torch",
                "--config",
                self._write_config(tmp_path, config_text),
                "--export-dir",
                str(tmp_path / "exports"),
                "--results-dir",
                str(tmp_path / "results"),
                "--fp-cache-dir",
                str(tmp_path / "fp"),
                "--recipe-cache-dir",
                str(tmp_path / "recipe"),
                *extra_argv,
            ]
        )
        with patch(
            "GenAILab.bench.torch.run_genai.run_experiment", side_effect=run_experiment
        ):
            return runner.run(args)

    def test_load_documents_force_export(self, tmp_path):
        from GenAILab.bench.runner import load_documents

        path = self._write_config(
            tmp_path, "model: {model_id: a}\n---\n---\nexport: true\n"
        )
        docs = load_documents(path, force_export=True)
        assert docs == [
            {"model": {"model_id": "a"}, "export": True},
            None,
            {"export": True},
        ]

    def test_runs_every_document_and_shares_caches(self, tmp_path):
        seen = []

        def fake(doc, **kwargs):
            seen.append((doc["model"]["model_id"], kwargs))

        code = self._run(
            tmp_path, "model: {model_id: a}\n---\n---\nmodel: {model_id: b}\n", fake
        )
        assert code == 0
        assert [model_id for model_id, _ in seen] == ["a", "b"]
        assert seen[0][1]["fp_cache"] is seen[1][1]["fp_cache"]
        assert seen[0][1]["recipe_cache"] is seen[1][1]["recipe_cache"]
        assert "model_cache" not in seen[0][1]

    def test_failure_does_not_stop_later_documents(self, tmp_path):
        seen = []

        def fake(doc, **_):
            seen.append(doc["model"]["model_id"])
            if doc["model"]["model_id"] == "a":
                raise RuntimeError("boom")

        code = self._run(
            tmp_path, "model: {model_id: a}\n---\nmodel: {model_id: b}\n", fake
        )
        assert code == 1
        assert seen == ["a", "b"]

    def test_fail_fast_stops_after_first_failure(self, tmp_path):
        seen = []

        def fake(doc, **_):
            seen.append(doc["model"]["model_id"])
            raise RuntimeError("boom")

        code = self._run(
            tmp_path,
            "model: {model_id: a}\n---\nmodel: {model_id: b}\n",
            fake,
            extra_argv=["--fail-fast"],
        )
        assert code == 1
        assert seen == ["a"]

    def test_pdb_opens_post_mortem_on_failure(self, tmp_path):
        def fake(doc, **_):
            raise RuntimeError("boom")

        with patch("GenAILab.bench.runner.pdb.post_mortem") as post_mortem:
            self._run(tmp_path, "model: {model_id: a}\n", fake, extra_argv=["--pdb"])
        post_mortem.assert_called_once()

    def test_no_recipe_cache(self, tmp_path):
        seen = []
        self._run(
            tmp_path,
            "model: {model_id: a}\n",
            lambda doc, **kwargs: seen.append(kwargs),
            extra_argv=["--no-recipe-cache"],
        )
        assert seen[0]["recipe_cache"] is None


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
