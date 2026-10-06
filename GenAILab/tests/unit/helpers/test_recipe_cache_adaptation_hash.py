# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Cache base-hash correctness for model adaptations.

Adaptations are baked into the model *class* and never passed to the
constructor, so they never appear in ``model_kwargs``. Before this was keyed,
two runs differing only in an adaptation -- ``MoE: {selection: all}`` versus
``selection: routed`` -- shared cached encodings, which silently invalidates
exactly the comparison such a pair of runs exists to make.

As with the pre-sim key, an empty adaptations list must add NO key, so caches
for un-adapted runs stay valid.

recipe_cache imports onnx at module top, so this skips where onnx is
unavailable and runs in CI / on the cluster.
"""

import pytest

pytest.importorskip("onnx", reason="recipe_cache imports onnx at module top")

from GenAILab.bench.recipe_cache import (  # noqa: E402
    RecipeCache,
    _adaptation_identity,
)


class _FakePrecision:
    """Minimal stand-in: compute_base_hash only calls weight_identity()."""

    def __init__(self, ident="w4a16"):
        self._ident = ident

    def weight_identity(self):
        return self._ident


@pytest.fixture
def cache():
    c = RecipeCache.__new__(RecipeCache)
    c.env_hash = "test-env"
    return c


COMMON = dict(
    model_id="Qwen/Qwen3-30B-A3B",
    model_kwargs={"dtype": "bfloat16"},
    framework="torch",
)


def _hash(cache, adaptations):
    return cache.compute_base_hash(
        COMMON["model_id"],
        _FakePrecision(),
        COMMON["model_kwargs"],
        COMMON["framework"],
        adaptations=adaptations,
    )


class TestAdaptationsInBaseHash:
    def test_selection_all_differs_from_routed(self):
        """The bug this fixes: the two arms of the MoE study must not collide."""
        cache = RecipeCache.__new__(RecipeCache)
        cache.env_hash = "test-env"
        routed = _hash(cache, [{"ExportableMoE": {"selection": "routed"}}])
        forced = _hash(cache, [{"ExportableMoE": {"selection": "all"}}])
        assert routed != forced

    def test_no_adaptations_leaves_hash_unchanged(self, cache):
        """Backward compatibility: un-adapted runs keep their cache entries."""
        without = cache.compute_base_hash(
            COMMON["model_id"],
            _FakePrecision(),
            COMMON["model_kwargs"],
            COMMON["framework"],
        )
        for empty in (None, []):
            assert _hash(cache, empty) == without

    def test_presence_of_an_adaptation_changes_the_hash(self, cache):
        bare = _hash(cache, None)
        assert _hash(cache, ["SHA"]) != bare

    def test_different_adaptations_differ(self, cache):
        assert _hash(cache, ["SHA"]) != _hash(cache, ["ExportableMoE"])

    def test_same_config_is_stable(self, cache):
        first = _hash(cache, [{"ExportableMoE": {"selection": "all"}}])
        second = _hash(cache, [{"ExportableMoE": {"selection": "all"}}])
        assert first == second

    def test_order_does_not_matter(self, cache):
        """Stacking order is not part of what the entries request."""
        a = _hash(cache, ["ExportableMoE", "ExportableLinearAttention"])
        b = _hash(cache, ["ExportableLinearAttention", "ExportableMoE"])
        assert a == b

    def test_bare_name_equals_empty_kwargs(self, cache):
        assert _hash(cache, ["ExportableMoE"]) == _hash(cache, [{"ExportableMoE": {}}])

    def test_extra_kwarg_changes_the_hash(self, cache):
        one = _hash(cache, [{"ExportableMoE": {"selection": "all"}}])
        two = _hash(
            cache,
            [{"ExportableMoE": {"selection": "all", "export_execution": "dense"}}],
        )
        assert one != two

    def test_adaptations_compose_with_other_key_fields(self, cache):
        """The adaptation key must not swallow differences elsewhere."""
        adapt = [{"ExportableMoE": {"selection": "all"}}]
        base = _hash(cache, adapt)
        other_model = cache.compute_base_hash(
            "Qwen/Qwen3-235B-A22B",
            _FakePrecision(),
            COMMON["model_kwargs"],
            COMMON["framework"],
            adaptations=adapt,
        )
        other_precision = cache.compute_base_hash(
            COMMON["model_id"],
            _FakePrecision("w8a16"),
            COMMON["model_kwargs"],
            COMMON["framework"],
            adaptations=adapt,
        )
        assert base != other_model
        assert base != other_precision


class TestAdaptationIdentity:
    """The canonicaliser handles both forms YAMLConfigParser accepts."""

    def test_bare_names(self):
        assert _adaptation_identity(["SHA"]) == [["SHA", []]]

    def test_dict_with_kwargs(self):
        assert _adaptation_identity([{"ExportableMoE": {"selection": "all"}}]) == [
            ["ExportableMoE", [("selection", "all")]]
        ]

    def test_kwargs_order_normalised(self):
        one = _adaptation_identity([{"ExportableMoE": {"a": 1, "b": 2}}])
        two = _adaptation_identity([{"ExportableMoE": {"b": 2, "a": 1}}])
        assert one == two

    def test_entry_order_normalised(self):
        assert _adaptation_identity(["B", "A"]) == _adaptation_identity(["A", "B"])

    def test_non_dict_kwargs_tolerated(self):
        """_normalize_adaptations coerces non-dict params to {}; match that."""
        assert _adaptation_identity([{"ExportableMoE": None}]) == [
            ["ExportableMoE", []]
        ]

    def test_mixed_forms(self):
        got = _adaptation_identity(["SHA", {"ExportableMoE": {"selection": "routed"}}])
        assert got == [["ExportableMoE", [("selection", "routed")]], ["SHA", []]]
