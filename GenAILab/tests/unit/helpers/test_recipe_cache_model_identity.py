# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Cache base-hash correctness for the model-section fields outside model_kwargs.

The parser takes context/sequence length, image size, audio frames and the
encodings path out of ``model_kwargs``. Before they were keyed, a chain run at
one context length was served to a run at another, and the torch loader's
``strict=False`` hid the mismatch.

recipe_cache imports onnx at module top, so this skips where onnx is
unavailable and runs in CI / on the cluster.
"""

import pytest

pytest.importorskip("onnx", reason="recipe_cache imports onnx at module top")

import GenAILab.bench.recipe_cache as recipe_cache  # noqa: E402
from GenAILab.bench.recipe_cache import (  # noqa: E402
    RecipeCache,
    chain_model_identity,
)


class _FakePrecision:
    def weight_identity(self):
        return "w4a16"


@pytest.fixture(autouse=True)
def _pinned_revision(monkeypatch):
    monkeypatch.setattr(recipe_cache, "checkpoint_revision", lambda _: "rev1")


@pytest.fixture
def cache():
    c = RecipeCache.__new__(RecipeCache)
    c.env_hash = "test-env"
    return c


BASE = dict(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    context_length=4096,
    sequence_length=2048,
)


def _hash(cache, **overrides):
    fields = {**BASE, **overrides}
    return cache.compute_base_hash(
        fields["model_id"],
        _FakePrecision(),
        {},
        "onnx",
        model_identity=chain_model_identity(**fields),
    )


class TestModelIdentityInBaseHash:
    def test_stable(self, cache):
        assert _hash(cache) == _hash(cache)

    @pytest.mark.parametrize(
        "override",
        [
            {"context_length": 2048},
            {"sequence_length": 128},
            {"sequence_length": [1, 128]},
            {"sequence_length": [1, 2048]},
            {"image_size": [448, 448]},
            {"audio_frames": 3000},
        ],
    )
    def test_each_field_changes_the_hash(self, cache, override):
        assert _hash(cache, **override) != _hash(cache)

    @pytest.mark.parametrize(
        "lengths",
        [[1, 128, 2048], [2048, 1], [1, 1, 2048]],
    )
    def test_sequence_lengths_sharing_a_maximum_share_entries(self, cache, lengths):
        assert _hash(cache, sequence_length=lengths) == _hash(
            cache, sequence_length=[1, 2048]
        )

    def test_single_entry_list_is_static(self, cache):
        assert _hash(cache, sequence_length=[2048]) == _hash(cache)

    def test_revision_changes_the_hash(self, cache, monkeypatch):
        before = _hash(cache)
        monkeypatch.setattr(recipe_cache, "checkpoint_revision", lambda _: "rev2")
        assert _hash(cache) != before

    def test_encodings_keyed_by_content(self, cache, tmp_path):
        enc = tmp_path / "encodings"
        enc.mkdir()
        (enc / "model.encodings").write_text("{}")
        first = _hash(cache, encodings=str(enc))
        assert first != _hash(cache)
        (enc / "model.data").write_bytes(b"weights")
        assert _hash(cache, encodings=str(enc)) == first
        (enc / "model.encodings").write_text('{"changed": true}')
        assert _hash(cache, encodings=str(enc)) != first

    def test_env_hash_follows_the_recipe_fingerprint(self, monkeypatch):
        def env_hash(fingerprint):
            monkeypatch.setattr(recipe_cache, "recipe_fingerprint", lambda: fingerprint)
            return RecipeCache.__new__(RecipeCache).env_hash

        assert env_hash({"aimet": "a"}) != env_hash({"aimet": "b"})
