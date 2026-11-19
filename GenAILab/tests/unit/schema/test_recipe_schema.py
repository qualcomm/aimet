# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the sync-safe recipe/dataset schema.

These exercise the schema in isolation -- no aimet, no GenAILab internals --
which is the whole point of the schema island.
"""

import pytest
from pydantic import ValidationError

from GenAILab.qai_hub_lm.schema import (
    Recipe,
    contract_mismatch,
    spec_kwargs,
)
from GenAILab.qai_hub_lm.schema.recipe import (
    AdaScaleSpec,
    CalibrationSpec,
    ClipSpec,
    SeqMSESpec,
    SkipSpec,
)


def _names(steps):
    return [s.name for s in steps]


class TestNormalization:
    """The three YAML shapes the current parser accepts."""

    def test_single_step_dict_becomes_backbone_chain(self):
        r = Recipe.model_validate(
            {"name": "Calibration", "dataset": {"name": "Wikitext", "split": "train"}}
        )
        assert _names(r.backbone) == ["Calibration"]
        assert r.visual is None

    def test_top_level_list_becomes_backbone_chain(self):
        r = Recipe.model_validate(
            [
                {"name": "SeqMSE", "dataset": {"name": "Wikitext", "split": "train"}},
                {
                    "name": "Calibration",
                    "dataset": {"name": "Wikitext", "split": "train"},
                },
            ]
        )
        assert _names(r.backbone) == ["SeqMSE", "Calibration"]

    def test_component_form_with_dict_values_coerced_to_lists(self):
        r = Recipe.model_validate(
            {
                "backbone": {"name": "Calibration"},
                "visual": {"name": "Calibration"},
            }
        )
        assert _names(r.backbone) == ["Calibration"]
        assert _names(r.visual) == ["Calibration"]


class TestAutoInsertCalibration:
    def test_chain_without_terminal_gets_calibration_appended(self):
        r = Recipe.model_validate([{"name": "SeqMSE"}])
        assert _names(r.backbone) == ["SeqMSE", "Calibration"]
        # the auto-inserted step defaults to Wikitext/train
        assert r.backbone[-1].dataset.name == "Wikitext"
        assert r.backbone[-1].dataset.split == "train"

    def test_chain_with_calibration_is_untouched(self):
        r = Recipe.model_validate([{"name": "Calibration"}])
        assert _names(r.backbone) == ["Calibration"]

    def test_remove_quantization_counts_as_terminal(self):
        r = Recipe.model_validate([{"name": "RemoveQuantization"}])
        assert _names(r.backbone) == ["RemoveQuantization"]

    def test_visual_chain_also_gets_autoinsert(self):
        r = Recipe.model_validate(
            {"backbone": [{"name": "Calibration"}], "visual": [{"name": "SeqMSE"}]}
        )
        assert _names(r.visual) == ["SeqMSE", "Calibration"]


class TestOrderingIsLoadBearing:
    def test_order_preserved(self):
        r = Recipe.model_validate(
            [{"name": "SeqMSE"}, {"name": "AdaScale"}, {"name": "Calibration"}]
        )
        assert _names(r.backbone) == [
            "SeqMSE",
            "AdaScale",
            "Calibration",
        ]


class TestPerStepKwargValidation:
    """The guarantee that does not exist in the current parser."""

    def test_wrong_type_rejected_with_field_path(self):
        with pytest.raises(ValidationError) as ei:
            Recipe.model_validate(
                [
                    {"name": "AdaScale", "num_iterations": "lots"},
                    {"name": "Calibration"},
                ]
            )
        assert "num_iterations" in str(ei.value)

    def test_typo_kwarg_rejected(self):
        # num_iteration (missing 's') is silently dropped by the current parser
        with pytest.raises(ValidationError):
            Recipe.model_validate([{"name": "Calibration", "num_iteration": 20}])

    def test_unknown_technique_rejected(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate([{"name": "MagicQuant"}])

    def test_unknown_dataset_rejected(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(
                [
                    {
                        "name": "Calibration",
                        "dataset": {"name": "Bogus", "split": "train"},
                    }
                ]
            )

    def test_clip_value_is_float(self):
        r = Recipe.model_validate(
            [{"name": "Clip", "value": 500.0}, {"name": "Calibration"}]
        )
        assert r.backbone[0].value == 500.0


class TestExcludeUnsetLowering:
    """The bridge that lets one schema serve torch and onnx defaults."""

    def test_omitted_kwargs_are_not_emitted(self):
        r = Recipe.model_validate([{"name": "AdaScale"}, {"name": "Calibration"}])
        kwargs = r.backbone[0].model_dump(
            exclude_unset=True, exclude={"name", "dataset"}
        )
        assert kwargs == {}  # nothing forwarded -> backend default wins

    def test_set_kwarg_is_emitted(self):
        r = Recipe.model_validate(
            [{"name": "AdaScale", "num_iterations": 128}, {"name": "Calibration"}]
        )
        kwargs = r.backbone[0].model_dump(
            exclude_unset=True, exclude={"name", "dataset"}
        )
        assert kwargs == {"num_iterations": 128}


class TestSpinQuantAsPhasedStep:
    """SpinQuant is now a normal pre_sim step in the chain (no side-channel)."""

    def test_spinquant_is_a_step_in_the_chain(self):
        r = Recipe.model_validate(
            [
                {
                    "name": "SpinQuant",
                    "enable_r1": True,
                    "enable_r2": False,
                    "enable_r3": False,
                },
                {"name": "Calibration"},
            ]
        )
        assert _names(r.backbone) == ["SpinQuant", "Calibration"]

    def test_at_least_one_rotation_required(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": False,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )

    def test_bare_spinquant_is_rejected(self):
        # The rotation set is the identity of the checkpoint, so it is stated
        # rather than assumed to be R1: a bare step names no rotations at all.
        with pytest.raises(ValidationError):
            Recipe.model_validate([{"name": "SpinQuant"}, {"name": "Calibration"}])

    def test_strict_prefix_visual_must_match_backbone(self):
        # Strict: visual can't omit a pre-sim step backbone declares.
        with pytest.raises(ValidationError):
            Recipe.model_validate(
                {
                    "backbone": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": False,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                    "visual": [{"name": "Calibration"}],
                }
            )

    def test_strict_prefix_divergent_flags_rejected(self):
        # Strict: same technique, different flags across components -> reject
        # (this is the silent "prefer backbone" bug the strict rule kills).
        with pytest.raises(ValidationError):
            Recipe.model_validate(
                {
                    "backbone": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": True,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                    "visual": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": False,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                }
            )

    def test_spinquant_on_both_components_ok(self):
        r = Recipe.model_validate(
            {
                "backbone": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ],
                "visual": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ],
            }
        )
        assert r.backbone[0].name == "SpinQuant"
        assert r.visual[0].name == "SpinQuant"


class TestPhaseAxis:
    def test_phase_property_on_steps(self):
        from GenAILab.qai_hub_lm.schema import Phase

        r = Recipe.model_validate(
            [
                {
                    "name": "SpinQuant",
                    "enable_r1": True,
                    "enable_r2": False,
                    "enable_r3": False,
                },
                {"name": "Calibration"},
            ]
        )
        assert r.backbone[0].phase == Phase.pre_sim
        assert r.backbone[1].phase == Phase.on_sim

    def test_pre_sim_after_on_sim_rejected(self):
        # SpinQuant (pre_sim) after Calibration (on_sim) is illegal: the sim is
        # already built by then.
        with pytest.raises(ValidationError) as ei:
            Recipe.model_validate(
                [
                    {"name": "Calibration"},
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                ]
            )
        assert "pre-sim" in str(ei.value)

    def test_phased_steps_splits_by_phase(self):
        r = Recipe.model_validate(
            [
                {
                    "name": "SpinQuant",
                    "enable_r1": True,
                    "enable_r2": False,
                    "enable_r3": False,
                },
                {"name": "SeqMSE"},
                {"name": "Calibration"},
            ]
        )
        pre, on_sim = r.phased_steps("backbone")
        assert _names(pre) == ["SpinQuant"]
        assert _names(on_sim) == ["SeqMSE", "Calibration"]

    def test_phased_steps_empty_pre_when_no_pre_sim(self):
        r = Recipe.model_validate([{"name": "Calibration"}])
        pre, on_sim = r.phased_steps("backbone")
        assert pre == []
        assert _names(on_sim) == ["Calibration"]


class TestSplitRecipe:
    """split_recipe returns a FLAT pre_sim list + per-component on_sim dict."""

    def test_flat_pre_sim_and_per_component_on_sim(self):
        from GenAILab.qai_hub_lm.schema import split_recipe

        r = Recipe.model_validate(
            {
                "backbone": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "AdaScale"},
                    {"name": "Calibration"},
                ],
                "visual": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ],
            }
        )
        pre_sim, on_sim = split_recipe(r)
        # pre_sim is a flat list (collapsed from the identical prefixes), once
        assert pre_sim == [
            {
                "name": "SpinQuant",
                "enable_r1": True,
                "enable_r2": False,
                "enable_r3": False,
            }
        ]
        assert [s["name"] for s in on_sim["backbone"]] == ["AdaScale", "Calibration"]
        assert [s["name"] for s in on_sim["visual"]] == ["Calibration"]

    def test_no_pre_sim_is_empty_list(self):
        from GenAILab.qai_hub_lm.schema import (
            has_pre_sim,
            pre_sim_identity,
            split_recipe,
        )

        pre_sim, on_sim = split_recipe(Recipe.model_validate([{"name": "Calibration"}]))
        assert pre_sim == []
        assert has_pre_sim(pre_sim) is False
        assert pre_sim_identity(pre_sim) is None


class TestToComponents:
    """The lowering bridge: validated Recipe -> the dict shape parse_document consumes."""

    def test_single_step_lowers_to_backbone_list(self):
        r = Recipe.model_validate(
            {"name": "Calibration", "dataset": {"name": "Wikitext", "split": "train"}}
        )
        assert r.to_components() == {
            "backbone": [
                {
                    "name": "Calibration",
                    "dataset": {"name": "Wikitext", "split": "train"},
                }
            ]
        }

    def test_only_set_kwargs_lowered(self):
        # AdaScale with one kwarg set: only that one appears (backend default wins
        # for the rest downstream).
        r = Recipe.model_validate(
            [{"name": "AdaScale", "num_iterations": 64}, {"name": "Calibration"}]
        )
        bb = r.to_components()["backbone"]
        assert bb[0] == {"name": "AdaScale", "num_iterations": 64}

    def test_spinquant_stays_inline_as_step(self):
        r = Recipe.model_validate(
            [
                {
                    "name": "SpinQuant",
                    "enable_r1": True,
                    "enable_r2": False,
                    "enable_r3": False,
                },
                {"name": "Calibration"},
            ]
        )
        names = [s["name"] for s in r.to_components()["backbone"]]
        assert names == ["SpinQuant", "Calibration"]

    def test_roundtrip_equals_original(self):
        for raw in (
            {"name": "Calibration"},
            [{"name": "SeqMSE"}, {"name": "Calibration"}],
            {
                "backbone": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ],
                "visual": [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ],
            },
        ):
            r = Recipe.model_validate(raw)
            assert Recipe.model_validate(r.to_components()) == r


class TestStepModelsDirectly:
    def test_constructed_step_types(self):
        assert CalibrationSpec(name="Calibration").num_iterations is None
        assert SeqMSESpec(name="SeqMSE").num_iterations is None
        assert AdaScaleSpec(name="AdaScale").num_batches is None


# Fixtures used by the real recipe apply() methods; mirrors
# yaml_config_parser._RECIPE_APPLY_FIXTURES.
_FIXTURES = {"quantsim", "generator", "dataloader", "component"}


class TestStepKwargs:
    def test_maps_every_technique(self):
        # every technique spec yields a kwargs set
        from GenAILab.qai_hub_lm.schema.recipe import _technique_specs

        for spec in _technique_specs():
            assert isinstance(spec_kwargs(spec), set)

    def test_known_kwargs(self):
        assert spec_kwargs(CalibrationSpec) == {"num_iterations"}
        assert spec_kwargs(AdaScaleSpec) == {
            "num_batches",
            "num_iterations",
            "early_stopping",
        }
        assert spec_kwargs(ClipSpec) == {"value"}
        assert spec_kwargs(SkipSpec) == set()
        # name/dataset are never counted as kwargs
        assert "name" not in spec_kwargs(CalibrationSpec)
        assert "dataset" not in spec_kwargs(CalibrationSpec)


class TestContractMismatch:
    """The mechanism behind register_recipe's import-time enforcement.

    Pure: takes any callable, so we test it with synthetic apply() signatures
    (no aimet). The real-recipes-conform check lives in test_vocabulary_sync.py.
    """

    def test_exact_match_passes(self):
        def apply(
            quantsim,
            generator,
            dataloader,
            num_batches=32,
            num_iterations=64,
            early_stopping=None,
            **kwargs,
        ): ...

        missing, extra = contract_mismatch(AdaScaleSpec, apply, ignore_params=_FIXTURES)
        assert missing == set() and extra == set()

    def test_missing_kwarg_flagged(self):
        # num_batches only reachable via **kwargs -> not implemented
        def apply(
            quantsim,
            generator,
            dataloader,
            num_iterations=64,
            early_stopping=None,
            **kwargs,
        ): ...

        missing, extra = contract_mismatch(AdaScaleSpec, apply, ignore_params=_FIXTURES)
        assert missing == {"num_batches"} and extra == set()

    def test_extra_kwarg_flagged(self):
        # a backend-private knob the schema can't express
        def apply(
            quantsim, generator, dataloader, num_iterations=20, secret_knob=1, **kwargs
        ): ...

        missing, extra = contract_mismatch(
            CalibrationSpec, apply, ignore_params=_FIXTURES
        )
        assert missing == set() and extra == {"secret_knob"}

    def test_kwargs_only_does_not_satisfy(self):
        # **kwargs alone must NOT count as implementing a required knob
        def apply(quantsim, generator, dataloader, **kwargs): ...

        missing, extra = contract_mismatch(
            CalibrationSpec, apply, ignore_params=_FIXTURES
        )
        assert missing == {"num_iterations"}

    def test_fixtures_not_counted_as_extra(self):
        # the fixture params must never show up as 'extra'
        def apply(quantsim, generator, dataloader, **kwargs): ...

        _, extra = contract_mismatch(SkipSpec, apply, ignore_params=_FIXTURES)
        assert extra == set()

    def test_terminal_recipe_with_no_kwargs_matches(self):
        def apply(quantsim, generator, dataloader, **kwargs): ...

        missing, extra = contract_mismatch(SkipSpec, apply, ignore_params=_FIXTURES)
        assert missing == set() and extra == set()

    def test_positional_and_keyword_only_both_counted(self):
        # a KEYWORD_ONLY schema knob is still "implemented"
        def apply(quantsim, generator, dataloader, *, num_iterations=64, **kwargs): ...

        missing, extra = contract_mismatch(
            CalibrationSpec, apply, ignore_params=_FIXTURES
        )
        assert missing == set() and extra == set()
