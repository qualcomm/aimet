# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""What the cache fingerprints cover, and what they deliberately leave out.

A fingerprint that misses a file serves stale entries; one that covers too much
costs a needless re-export or a re-run of a long recipe. These pin both sides.
"""

import ast

import pytest

from GenAILab.bench.utils import cache_utils
from GenAILab.bench.utils.cache_utils import (
    _hash_files,
    checkpoint_revision,
    file_digest,
)


_CACHED = (
    cache_utils._aimet_hash,
    cache_utils._genailab_hash,
    cache_utils.model_fingerprint,
    cache_utils.recipe_fingerprint,
    checkpoint_revision,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Computed fingerprints are cached per process; keep fakes out of other tests."""
    for cached in _CACHED:
        cached.cache_clear()
    yield
    for cached in _CACHED:
        cached.cache_clear()


@pytest.fixture
def genailab(tmp_path, monkeypatch):
    """A fake GenAILab tree that the code hash walks instead of the real one."""
    root = tmp_path / "GenAILab"
    for rel in (
        "qai_hub_lm/models/base.py",
        "qai_hub_lm/scoring/grace/grace2.jsonl",
        "bench/recipe_chain.py",
        "bench/metrics.py",
        "bench/prompts/calibration_prompts.yaml",
        "bench/NOTES.md",
        "tests/unit/test_x.py",
        "configs/onnx_nightly.yaml",
        "dashboard/app.py",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel)
    monkeypatch.setattr(cache_utils, "_GENAILAB_ROOT", root)
    return root


def _code_hash():
    cache_utils._genailab_hash.cache_clear()
    return cache_utils._genailab_hash()


def _edit(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.read_text() + " edited" if path.exists() else "new")


class TestGenAILabCodeHash:
    @pytest.mark.parametrize(
        "rel",
        [
            "qai_hub_lm/models/base.py",
            "bench/recipe_chain.py",
            "bench/prompts/calibration_prompts.yaml",
            "bench/new_module.py",
        ],
    )
    def test_package_edit_misses(self, genailab, rel):
        before = _code_hash()
        _edit(genailab, rel)
        assert _code_hash() != before

    @pytest.mark.parametrize(
        "rel",
        [
            "bench/metrics.py",
            "qai_hub_lm/scoring/grace/grace2.jsonl",
            "bench/NOTES.md",
            "tests/unit/test_x.py",
            "configs/onnx_nightly.yaml",
            "dashboard/app.py",
        ],
    )
    def test_eval_docs_tests_and_configs_keep_entries(self, genailab, rel):
        before = _code_hash()
        _edit(genailab, rel)
        assert _code_hash() == before

    def test_bytecode_is_ignored(self, genailab):
        before = _code_hash()
        cache = genailab / "bench" / "__pycache__"
        cache.mkdir()
        (cache / "recipe_chain.cpython-312.pyc").write_bytes(b"\0")
        (genailab / "bench" / "stray.pyc").write_bytes(b"\0")
        assert _code_hash() == before


# The evaluation entry points: they run metrics after the cached steps, so they
# may import EVAL_ONLY code. Nothing else may; see test_eval_only_is_only_
# imported_by_evaluation_entry_points.
_EVAL_ENTRY_POINTS = (
    "bench/runner.py",
    "bench/onnx/run_genai.py",
    "bench/torch/run_genai.py",
    "bench/torch/collect_fp.py",
    "bench/torch/example_custom_script.py",
    "bench/diffusion",
    "bench/onnx/run_diffusion.py",
    "bench/torch/run_diffusion.py",
)


def _genailab_imports(path):
    """The GenAILab paths (relative to GenAILab/) that *path* imports."""
    root = cache_utils._GENAILAB_ROOT
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] != "GenAILab":
                continue
            base = "/".join(parts[1:])
            for rel in (f"{base}.py", f"{base}/__init__.py"):
                if (root / rel).exists():
                    yield rel, node.lineno


class TestEvalOnly:
    def test_listed_paths_exist(self):
        """A renamed module would silently drop out of the list."""
        for rel in cache_utils.EVAL_ONLY:
            assert (cache_utils._GENAILAB_ROOT / rel).exists(), rel

    def test_eval_only_is_only_imported_by_evaluation_entry_points(self):
        """EVAL_ONLY code must not reach a cached step.

        Its edits keep cache entries, which is only sound while no export or
        recipe code imports it. Move a module out of EVAL_ONLY before code
        outside the entry points depends on it.
        """
        root = cache_utils._GENAILAB_ROOT
        offenders = []
        for package in cache_utils._GENAILAB_PACKAGES:
            for path in sorted((root / package).rglob("*.py")):
                rel = path.relative_to(root).as_posix()
                if cache_utils.is_eval_only(rel) or any(
                    rel == p or rel.startswith(p + "/") for p in _EVAL_ENTRY_POINTS
                ):
                    continue
                offenders += [
                    f"{rel}:{line} imports {target}"
                    for target, line in _genailab_imports(path)
                    if cache_utils.is_eval_only(target)
                ]
        assert not offenders, "\n".join(offenders)


class TestHashFiles:
    def test_content_and_path_both_count(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        first = _hash_files(tmp_path)
        (tmp_path / "a.py").write_text("y")
        assert _hash_files(tmp_path) != first
        (tmp_path / "a.py").rename(tmp_path / "b.py")
        (tmp_path / "b.py").write_text("x")
        assert _hash_files(tmp_path) != first

    def test_order_of_creation_does_not_matter(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        for root, names in ((one, "ab"), (two, "ba")):
            root.mkdir()
            for name in names:
                (root / f"{name}.py").write_text(name)
        assert _hash_files(one) == _hash_files(two)


class TestAimetHash:
    @pytest.fixture
    def package(self, tmp_path, monkeypatch):
        package = tmp_path / "site" / "aimet_onnx"
        (package / "common").mkdir(parents=True)
        (package / "quantsim.py").write_text("x = 1")
        (package / "common" / "default_config.json").write_text("{}")
        (package / "common" / "libquant_info.abi3.so").write_bytes(b"build 1")

        class _Spec:
            submodule_search_locations = [str(package)]

        monkeypatch.setattr(
            cache_utils.importlib.util,
            "find_spec",
            lambda name: _Spec() if name == "aimet_onnx" else None,
        )
        return package

    @staticmethod
    def _fresh():
        cache_utils._aimet_hash.cache_clear()
        return cache_utils._aimet_hash()

    @pytest.mark.parametrize("rel", ["quantsim.py", "common/default_config.json"])
    def test_python_and_config_edits_miss(self, package, rel):
        before = self._fresh()
        (package / rel).write_text("changed")
        assert self._fresh() != before

    def test_rebuilt_extension_keeps_entries(self, package):
        """CI rebuilds the wheel every run; a new binary alone must not miss."""
        before = self._fresh()
        (package / "common" / "libquant_info.abi3.so").write_bytes(b"build 2")
        assert self._fresh() == before


class TestCheckpointRevision:
    def test_local_directory_uses_its_path(self, tmp_path):
        assert checkpoint_revision(str(tmp_path)) == str(tmp_path.resolve())

    @pytest.fixture
    def hub_cache(self, tmp_path, monkeypatch):
        """An empty Hugging Face cache, and a way to add a snapshot to it."""
        import huggingface_hub.constants

        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

        def add_snapshot(model_id, commit):
            repo = tmp_path / f"models--{model_id.replace('/', '--')}"
            (repo / "snapshots" / commit).mkdir(parents=True)
            (repo / "snapshots" / commit / "config.json").write_text("{}")
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text(commit)

        return add_snapshot

    @staticmethod
    def _hub(monkeypatch, result):
        import huggingface_hub

        def model_info(self, model_id):
            if isinstance(result, Exception):
                raise result
            return type("Info", (), {"sha": result})()

        monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)

    def test_hub_revision_when_reachable(self, monkeypatch, hub_cache):
        hub_cache("org/model", "c" * 40)
        self._hub(monkeypatch, "a" * 40)
        assert checkpoint_revision("org/model") == "a" * 40

    @pytest.mark.parametrize(
        "error", [OSError("no network"), RuntimeError("429 Too Many Requests")]
    )
    def test_hub_error_keys_on_cached_snapshot(self, monkeypatch, hub_cache, error):
        """A transient Hub error must not switch every entry to another key."""
        hub_cache("org/model", "c" * 40)
        self._hub(monkeypatch, error)
        assert checkpoint_revision("org/model") == "c" * 40

    def test_hub_error_without_a_snapshot_gives_none(self, monkeypatch, hub_cache):
        self._hub(monkeypatch, OSError("no network"))
        assert checkpoint_revision("org/never-downloaded") is None


class TestFileDigest:
    def test_none_and_missing(self, tmp_path):
        assert file_digest(None) is None
        assert file_digest(tmp_path / "missing") is None

    def test_directory_hashes_only_matching_files(self, tmp_path):
        (tmp_path / "model.encodings").write_text("{}")
        (tmp_path / "model.data").write_bytes(b"weights")
        first = file_digest(tmp_path, suffix=".encodings")
        (tmp_path / "model.data").write_bytes(b"other weights")
        assert file_digest(tmp_path, suffix=".encodings") == first

    def test_file_and_directory_follow_content(self, tmp_path):
        enc = tmp_path / "encodings"
        enc.mkdir()
        (enc / "model.encodings").write_text("{}")
        file_first, dir_first = file_digest(enc / "model.encodings"), file_digest(enc)
        (enc / "model.encodings").write_text('{"a": 1}')
        assert file_digest(enc / "model.encodings") != file_first
        assert file_digest(enc) != dir_first
